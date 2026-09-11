"""Build the separately registered executed tool-policy development bank."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import signal
import time

from run_provenance import provenance
from irene_brain.agent.procedural_training_generator import ProceduralTrainingGenerator
from irene_brain.agent.isolated_software_environment import IsolatedSoftwareEnvironment
from irene_brain.agent.parallel_depth_episode import observation_frame
from irene_brain.agent.isolated_software_environment import ToolFeedback
from irene_brain.data.procedural_repair_bank import partition_tasks
from irene_brain.data.executed_trajectory import record_executed_trajectory,encode_executed_trajectory,digest_record
from irene_brain.evaluation.workspace_sandbox import run_workspace_tests
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    report={'status':'running','seed':1985,'partitions':{},'completed_records':0}
    started=time.monotonic()
    def deadline(*unused): raise TimeoutError('Registered900-second bank preparation limit')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(900)
    try:
        token_hash=hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
        assert token_hash=='760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
        tokenizer=BpeSemanticTokenizer(tokenizer_file=args.tokenizer,auto_build_if_missing=False)
        tasks=ProceduralTrainingGenerator(seed=1985).generate_training_tasks(count=2048,diversify_indices=False)
        selected,mapping=partition_tasks(tasks)
        (args.output/'family-map.json').write_text(json.dumps(mapping,indent=2))
        report['families']=len(mapping)
        report['tokenizer_sha256']=token_hash
        report['selection_sha256']=digest_record({part:[{k:v for k,v in row.items() if k!='actions'} for row in rows] for part,rows in selected.items()})
        report['provenance']=provenance(model=None,train_seed=1985,eval_seed=None,bank_digest=report['selection_sha256'],deterministic=False)
        for partition,cases in selected.items():
            records=[]
            requests=[]
            graders=[]
            counts=Counter()
            for index,case in enumerate(cases):
                report['current_case']={'partition':partition,'index':index,'source_id':case['source_id']}
                def private(files):
                    snapshot=dict(files)
                    snapshot['__private_check.py']=case['private_check']
                    return run_workspace_tests(snapshot,'__private_check.py')
                good={**case['initial_files'],case['target_module']:case['reference_solution']}
                bad={**case['initial_files'],case['target_module']:'raise RuntimeError("Injected training fault")'}
                controls={'public_good':run_workspace_tests(good,'test_public.py'),
                          'private_good':private(good),'public_bad':run_workspace_tests(bad,'test_public.py'),
                          'private_bad':private(bad)}
                if not (controls['public_good']['passed'] and controls['private_good']['passed'] and
                        all(not controls[name]['passed'] and 'Injected training fault' in controls[name]['output'] for name in ('public_bad','private_bad'))):
                    (args.output/'failed-controls.json').write_text(json.dumps({'case':report['current_case'],'controls':controls},indent=2))
                    raise RuntimeError('Registered reference/fault control failed')
                def validator(files):
                    result=private(files)
                    return result['passed'],'External checks passed' if result['passed'] else 'External checks failed'
                env=IsolatedSoftwareEnvironment(case['initial_files'],validator=validator)
                record=record_executed_trajectory(case['initial_prompt'],env,case['actions'],
                    source_id=case['source_id'],validator_id=hashlib.sha256(case['private_check'].encode()).hexdigest())
                del record['record_sha256']
                record.update(family_sha256=case['family_sha256'],partition=partition,repair=case['repair'],
                              control_results={name:value['passed'] for name,value in controls.items()})
                record['record_sha256']=digest_record(record)
                # Persist even a subsequent encoding/acceptance failure.
                with (args.output/(partition+'-attempts.jsonl')).open('a') as stream:
                    stream.write(json.dumps(record,ensure_ascii=True)+'\n')
                encoded=encode_executed_trajectory(record,tokenizer,max_tokens=8192)
                assert encoded['prompt_tokens'].shape[1]<=4096
                for step in record['steps']:
                    assert len(tokenizer.encode(step['action']))+1<=1024
                    assert len(tokenizer.encode(observation_frame(ToolFeedback(**step['feedback']))))<=4096
                outcomes=[step['feedback']['success'] for step in record['steps'] if step['feedback']['action_type']=='RUN_TESTS']
                assert outcomes==([False,True] if case['repair'] else [True])
                records.append(record)
                counts['input_tokens']+=encoded['input_ids'].numel()
                counts['supervised_tokens']+=int((encoded['labels']!=-100).sum())
                counts['repair_records']+=case['repair']
                if partition=='development':
                    requests.append({'source_id':case['source_id'],'family_sha256':case['family_sha256'],
                                     'initial_prompt':case['initial_prompt'],'initial_files':case['initial_files']})
                    graders.append({'source_id':case['source_id'],'target_module':case['target_module'],
                                    'private_check':case['private_check'],'reference_solution':case['reference_solution']})
                report['completed_records']+=1
                if report['completed_records']%16==0:
                    (args.output/'progress.json').write_text(json.dumps(report,indent=2))
                    print('REPAIR_BANK_PROGRESS',report['completed_records'],partition,index+1,len(cases),flush=True)
            raw=('\n'.join(json.dumps(row,sort_keys=True,ensure_ascii=True) for row in records)+'\n').encode()
            compressed=gzip.compress(raw,mtime=0)
            (args.output/(partition+'.jsonl.gz')).write_bytes(compressed)
            report['partitions'][partition]={'records':len(records),**dict(counts),'raw_sha256':hashlib.sha256(raw).hexdigest(),
                                              'gzip_sha256':hashlib.sha256(compressed).hexdigest()}
            if partition=='development':
                for name,rows in (('requests',requests),('graders',graders)):
                    data=('\n'.join(json.dumps(row,sort_keys=True) for row in rows)+'\n').encode()
                    (args.output/(name+'.jsonl')).write_bytes(data)
                    report[name+'_sha256']=hashlib.sha256(data).hexdigest()
        report['status']='complete'
    except Exception as exc:
        report.update(status='incomplete',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('REPAIR_BANK_RESULT',json.dumps(report),flush=True)


if __name__=='__main__': main()
