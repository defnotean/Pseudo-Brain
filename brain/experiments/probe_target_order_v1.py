"""Training-only prompt-order intervention, fixed actors and raw baseline."""
import argparse
from collections import Counter
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from collect_behavior_repair_v1 import select_cases,CHECKPOINTS,TRAINING_REPORT
from evaluate_trained_tool_policy import resolve_completed_checkpoint
from evaluate_tool_policy import digest,TOKENIZER,write_json
from run_provenance import apply_deterministic_mode
from irene_brain.agent.parallel_depth_episode import ParallelDepthSoftwareAgent,TransformerSoftwareAgent
from irene_brain.agent.isolated_software_environment import IsolatedSoftwareEnvironment
from irene_brain.evaluation.workspace_sandbox import run_workspace_tests
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.unified.parallel_depth_model import ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer


def moved_prompt(case):
    original=case['initial_prompt']
    line='Target: '+case['target_module']+'\n'
    if original.count(line)!=1:raise ValueError('Expected one exact target line')
    changed=original.replace(line,'')+line
    if Counter(original.splitlines(keepends=True))!=Counter(changed.splitlines(keepends=True)):
        raise ValueError('Prompt content changed beyond line ordering')
    return changed


def binding(episode,files,target):
    writes=[entry for entry in episode['trace'] if entry.get('executed') and entry['feedback']['action_type']=='WRITE_FILE']
    exact=[entry for entry in writes if entry['raw_action'].partition('\n')[0]=='ACTION: WRITE_FILE '+target]
    return {'expected_target_present':target in files,'exact_target_header':bool(exact),
            'successful_target_write':any(entry['feedback']['success'] for entry in exact),
            'write_actions':len(writes),'successful_writes':sum(entry['feedback']['success'] for entry in writes),
            'stop_reason':episode['stop_reason']}


def run_actor(name,cases,baseline,root,output,training,tokenizer):
    random.seed(198);np.random.seed(198);torch.manual_seed(198);torch.cuda.manual_seed_all(198)
    checkpoint,checksum,_=resolve_completed_checkpoint(training,TRAINING_REPORT,name)
    assert checksum==CHECKPOINTS[name]
    cls,agent_cls=(ParallelDepthModel,ParallelDepthSoftwareAgent) if name=='recurrent' else (ComparisonTransformer,TransformerSoftwareAgent)
    model=cls.from_payload(torch.load(checkpoint,map_location='cpu',weights_only=True)).cuda().eval()
    assert sum(p.numel() for p in model.parameters())<36000000
    agent=agent_cls(model,tokenizer)
    rows=[]
    with torch.inference_mode():
        for index,case in enumerate(cases):
            path=baseline/name/f'actor-{index:02d}.json'
            baseline_entry=json.loads((baseline/name/'report.json').read_text())['records'][index]
            assert digest(path)==baseline_entry['raw_attempt_sha256']
            old=json.loads(path.read_text())
            assert old['case_sha256']==case['case_sha256'] and old['checkpoint_sha256']==checksum
            prompt=moved_prompt(case)
            def validator(files):
                result=run_workspace_tests({**files,'__private_check.py':case['private_check']},'__private_check.py')
                return result['passed'],'External checks passed' if result['passed'] else 'External checks failed'
            env=IsolatedSoftwareEnvironment(case['initial_files'],validator=validator)
            episode=agent.execute_episode(prompt,env,max_cycles=2,max_action_tokens=512,max_prompt_tokens=4096,max_observation_tokens=4096)
            if name=='recurrent':assert episode.state_bytes==4096 and episode.prefix_tokens is None
            else:assert episode.state_bytes is None and 0<episode.prefix_tokens<=32768
            row={'source_id':case['source_id'],'case_sha256':case['case_sha256'],'checkpoint_sha256':checksum,
                 'initial_prompt':prompt,'original_prompt_sha256':hashlib.sha256(case['initial_prompt'].encode()).hexdigest(),
                 'baseline_raw_sha256':digest(path),'episode':asdict(episode),'final_files':env.files,
                 'baseline_binding':binding(old['episode'],old['actor_final_files'],case['target_module']),
                 'target_last_binding':binding(asdict(episode),env.files,case['target_module'])}
            write_json(output/f'{name}-{index:02d}.json',row)
            rows.append(row)
            print('TARGET_ORDER_CASE',name,index,row['baseline_binding']['expected_target_present'],row['target_last_binding']['expected_target_present'],episode.stop_reason,flush=True)
    assert digest(checkpoint)==checksum
    result={'planned_cases':8,'recorded_cases':len(rows),'checkpoint_sha256':checksum}
    for condition in ('baseline_binding','target_last_binding'):
        result[condition]={key:sum(row[condition][key] for row in rows) for key in ('expected_target_present','exact_target_header','successful_target_write','write_actions','successful_writes')}
        result[condition]['stop_reasons']=dict(Counter(row[condition]['stop_reason'] for row in rows))
    del agent,model
    gc.collect();torch.cuda.empty_cache()
    return result


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','bank','baseline','training-run','tokenizer','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started=time.monotonic()
    report={'status':'incomplete','model_updates':0,'models':{},'planned_new_attempts':16}
    def deadline(*unused):raise TimeoutError('Registered600-second prompt-order probe limit')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(600)
    try:
        assert digest(args.baseline/'report.json')=='0140ee38b7665ef8a0f9316b3f3f430a8ab44688e04a59775c1b5ce5ae328171'
        assert json.loads((args.baseline/'report.json').read_text())['status']=='complete'
        for relative,expected in json.loads((args.root/'source-manifest.json').read_text()).items():assert digest(args.root/relative)==expected
        assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        torch.set_num_threads(1);apply_deterministic_mode()
        assert digest(args.tokenizer)==TOKENIZER
        tokenizer=BpeSemanticTokenizer(tokenizer_file=args.tokenizer,auto_build_if_missing=False)
        cases=select_cases(args.bank)[:8]
        plan=[{'source_id':case['source_id'],'case_sha256':case['case_sha256'],'changed_prompt':moved_prompt(case)} for case in cases]
        assert plan==json.loads((args.root/'selection.json').read_text())
        report['source_manifest_sha256']=digest(args.root/'source-manifest.json')
        for name in ('recurrent','transformer'):
            write_json(args.output/'status.json',{'model':name,'stage':'running'})
            report['models'][name]=run_actor(name,cases,args.baseline,args.root,args.output,args.training_run,tokenizer)
            write_json(args.output/'report.json',report)
        report['status']='complete'
    except BaseException as exc:
        report.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-started
        write_json(args.output/'report.json',report)
        print('TARGET_ORDER_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
