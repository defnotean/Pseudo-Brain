"""Frozen actor prefixes and actual teacher continuations on training tasks only."""
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch

from evaluate_tool_policy import TOKENIZER, digest, write_json
from evaluate_trained_tool_policy import resolve_completed_checkpoint
from run_provenance import apply_deterministic_mode, provenance
from irene_brain.agent.isolated_software_environment import IsolatedSoftwareEnvironment
from irene_brain.agent.parallel_depth_episode import ParallelDepthSoftwareAgent, TransformerSoftwareAgent
from irene_brain.data.behavior_repair_trajectory import record_behavior_repair, encode_behavior_repair
from irene_brain.data.executed_trajectory import TeacherAction, digest_record
from irene_brain.evaluation.workspace_sandbox import run_workspace_tests
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.unified.parallel_depth_model import ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer

BANK_SHA='e7645a3c98749815098ec66293e583087180b06c6bfd89249d6b1bb96f04d5d0'
TRAINING_REPORT='1b5cf6883fcc3ad536c96ac0c06bcac354c1c5c96622ad8426953106b605a0b4'
CHECKPOINTS={'recurrent':'0b3edca04de150068c81fb8f71f05cd0aad8f42e423b9989bc789b919ffb85b7',
             'transformer':'346c9d4c9d334490cb2e93c54f4a1f0f66020a3833b4e7565f015b6d08c44dfe'}


def select_cases(bank):
    raw=bank.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=BANK_SHA: raise ValueError('MBPP bank changed')
    cases=[json.loads(line) for line in gzip.decompress(raw).splitlines()]
    if len(cases)!=357 or len({case['case_sha256'] for case in cases})!=357: raise ValueError('Case count changed')
    for case in cases:
        if case['partition']!='train' or not 601<=case['task_id']<=974: raise ValueError('Nontraining case selected')
        if case['case_sha256']!=digest_record({k:v for k,v in case.items() if k!='case_sha256'}): raise ValueError('Case digest changed')
    return sorted(cases,key=lambda case:case['case_sha256'])[:32]


def teacher_plan(case):
    return [TeacherAction('ACTION: READ_FILE test_public.py'),TeacherAction('ACTION: RUN_TESTS'),
            TeacherAction('ACTION: READ_FILE '+case['target_module']),
            TeacherAction('ACTION: WRITE_FILE '+case['target_module']+'\n'+case['reference_solution']),
            TeacherAction('ACTION: RUN_TESTS'),TeacherAction('ACTION: FINISH Implementation verified')]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=tuple(CHECKPOINTS),required=True)
    for name in ('root','bank','training-run','tokenizer','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started=time.monotonic()
    report={'status':'incomplete','model':args.model,'protocol':'behavior_repair_collection_v1',
            'model_updates':0,'planned_attempts':32,'recorded_attempts':0,'accepted':0,'dispositions':{},'records':[]}
    def deadline(*unused): raise TimeoutError('Registered900-second actor collection deadline')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(900)
    try:
        source_sha=digest(args.root/'source-manifest.json')
        for relative,expected in json.loads((args.root/'source-manifest.json').read_text()).items():
            if digest(args.root/relative)!=expected: raise ValueError('Collection source changed: '+relative)
        cases=select_cases(args.bank)
        plan=[{'source_id':c['source_id'],'case_sha256':c['case_sha256']} for c in cases]
        if plan!=json.loads((args.root/'selection.json').read_text()): raise ValueError('Saved selection changed')
        checkpoint,checksum,_=resolve_completed_checkpoint(args.training_run,TRAINING_REPORT,args.model)
        if checksum!=CHECKPOINTS[args.model] or digest(args.tokenizer)!=TOKENIZER: raise ValueError('Actor/tokenizer identity changed')
        assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        torch.set_num_threads(1)
        random.seed(198); np.random.seed(198); torch.manual_seed(198); torch.cuda.manual_seed_all(198)
        apply_deterministic_mode()
        tokenizer=BpeSemanticTokenizer(tokenizer_file=args.tokenizer,auto_build_if_missing=False)
        cls,agent_cls=(ParallelDepthModel,ParallelDepthSoftwareAgent) if args.model=='recurrent' else (ComparisonTransformer,TransformerSoftwareAgent)
        model=cls.from_payload(torch.load(checkpoint,map_location='cpu',weights_only=True)).cuda().eval()
        report.update(source_manifest_sha256=source_sha,checkpoint_sha256=checksum,selection_sha256=digest_record(plan),
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      provenance=provenance(model=model,train_seed=1986,eval_seed=198,bank_digest=BANK_SHA,deterministic=True))
        if report['parameter_count']>=36000000: raise ValueError('Parameter ceiling exceeded')
        agent=agent_cls(model,tokenizer)
        records=[]
        dispositions,stops,verbs=Counter(),Counter(),Counter()
        with torch.inference_mode():
            for index,case in enumerate(cases):
                write_json(args.output/'status.json',{'index':index,'source_id':case['source_id'],'stage':'actor'})
                def validator(files):
                    result=run_workspace_tests({**files,'__private_check.py':case['private_check']},'__private_check.py')
                    return result['passed'],'External checks passed' if result['passed'] else 'External checks failed'
                env=IsolatedSoftwareEnvironment(case['initial_files'],validator=validator)
                episode=agent.execute_episode(case['initial_prompt'],env,max_cycles=2,max_action_tokens=512,
                                               max_prompt_tokens=4096,max_observation_tokens=4096)
                raw={'source_id':case['source_id'],'case_sha256':case['case_sha256'],'checkpoint_sha256':checksum,
                     'episode':asdict(episode),'actor_final_files':env.files}
                raw_path=args.output/f'actor-{index:02d}.json'
                write_json(raw_path,raw)
                report['recorded_attempts']+=1
                stops[episode.stop_reason]+=1
                verbs.update(entry['feedback']['action_type'] for entry in episode.trace if entry.get('executed'))
                entry={'source_id':case['source_id'],'case_sha256':case['case_sha256'],'raw_attempt_sha256':digest(raw_path),
                       'actor_stop':episode.stop_reason,'actor_verified_completion':episode.success}
                if args.model=='recurrent' and (episode.state_bytes!=4096 or episode.prefix_tokens is not None): raise ValueError('Recurrent state/replay contract failed')
                if args.model=='transformer' and (episode.state_bytes is not None or not 0<episode.prefix_tokens<=32768): raise ValueError('Transformer prefix contract failed')
                if episode.stop_reason!='cycle_limit' or episode.success:
                    entry['disposition']='actor_stop:'+episode.stop_reason
                else:
                    actor={'architecture':model.architecture,'provenance_kind':'model_checkpoint','collection_protocol':'behavior_repair_collection_v1',
                           'checkpoint_sha256':checksum,'tokenizer_sha256':TOKENIZER,'source_manifest_sha256':source_sha,'initial_case_sha256':case['case_sha256']}
                    write_json(args.output/'status.json',{'index':index,'source_id':case['source_id'],'stage':'teacher'})
                    try:
                        record=record_behavior_repair(case['initial_prompt'],case['initial_files'],episode,env,teacher_plan(case),tokenizer,
                            actor=actor,source_id=case['source_id']+':'+args.model,validator_id=case['validator_sha256'])
                    except ValueError as exc:
                        if str(exc) not in ('Actor changed protected initial files','Ambiguous control token inside action'):
                            raise
                        entry.update(disposition='prefix_exclusion',reason=str(exc))
                    else:
                        del record['record_sha256']
                        record.update(raw_attempt_sha256=entry['raw_attempt_sha256'],partition='train',family_sha256=case['family_sha256'])
                        record['record_sha256']=digest_record(record)
                        write_json(args.output/f'teacher-{index:02d}.json',record)
                        if record['status']!='complete':
                            entry['disposition']='teacher_unverified_completion'
                        else:
                            try:
                                encoded=encode_behavior_repair(record,tokenizer)
                            except ValueError as exc:
                                observation_oversize=(str(exc)=='Observation framing or limit changed' and any(len(step['observation_token_ids'])>4096 for step in record['steps']))
                                if 'token limit' not in str(exc) and not observation_oversize: raise
                                entry.update(disposition='encoded_limit_exclusion',reason=str(exc))
                            else:
                                teacher_tests=[step['feedback']['success'] for step in record['steps'][record['behavior_steps']:] if step['feedback']['action_type']=='RUN_TESTS']
                                if len(teacher_tests)!=2 or not teacher_tests[-1]: raise ValueError('Unexpected teacher tests')
                                entry.update(disposition='accepted',record_sha256=record['record_sha256'],
                                    teacher_repaired_observed_failure=teacher_tests==[False,True],input_tokens=encoded['input_ids'].numel(),
                                    supervised_tokens=int((encoded['labels']!=-100).sum()))
                                records.append(record)
                dispositions[entry['disposition']]+=1
                report['records'].append(entry)
                report.update(accepted=len(records),dispositions=dict(dispositions),actor_stops=dict(stops),actor_action_types=dict(verbs))
                write_json(args.output/'report.json',report)
                print('BEHAVIOR_CASE',args.model,index,episode.stop_reason,entry['disposition'],flush=True)
        if report['recorded_attempts']!=32 or sum(dispositions.values())!=32: raise ValueError('Attempt accounting incomplete')
        if digest(checkpoint)!=checksum: raise ValueError('Actor checkpoint changed during collection')
        raw=('\n'.join(json.dumps(row,sort_keys=True,ensure_ascii=True) for row in records)+ ('\n' if records else '')).encode()
        compressed=gzip.compress(raw,mtime=0)
        (args.output/'training-trajectories.jsonl.gz').write_bytes(compressed)
        report.update(status='complete',trajectory_gzip_sha256=hashlib.sha256(compressed).hexdigest(),
                      teacher_repaired_observed_failures=sum(row.get('teacher_repaired_observed_failure',False) for row in report['records']),
                      input_tokens=sum(row.get('input_tokens',0) for row in report['records']),supervised_tokens=sum(row.get('supervised_tokens',0) for row in report['records']))
    except BaseException as exc:
        report.update(error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-started
        write_json(args.output/'report.json',report)
        print('BEHAVIOR_MODEL_RESULT',json.dumps({k:v for k,v in report.items() if k not in ('records','provenance')}),flush=True)


if __name__=='__main__': main()
