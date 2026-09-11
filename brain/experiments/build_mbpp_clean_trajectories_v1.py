"""All audited MBPP training cases as actually executed clean demonstrations."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import signal
import time

from irene_brain.agent.isolated_software_environment import IsolatedSoftwareEnvironment,ToolFeedback
from irene_brain.agent.parallel_depth_episode import observation_frame
from irene_brain.data.executed_trajectory import TeacherAction,record_executed_trajectory,encode_executed_trajectory,digest_record
from irene_brain.evaluation.workspace_sandbox import run_workspace_tests
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer

BANK_SHA='e7645a3c98749815098ec66293e583087180b06c6bfd89249d6b1bb96f04d5d0'


def main():
    parser=argparse.ArgumentParser()
    for name in ('bank','tokenizer','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    started=time.monotonic()
    report={'status':'incomplete','model_updates':0,'model_generated_tokens':0,'attempted':0,'accepted':0,'records':[]}
    def deadline(*unused):raise TimeoutError('Registered600-second clean trajectory deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(600)
    try:
        raw=args.bank.read_bytes()
        assert hashlib.sha256(raw).hexdigest()==BANK_SHA
        cases=[json.loads(line) for line in gzip.decompress(raw).splitlines()]
        assert len(cases)==357 and len({case['case_sha256'] for case in cases})==357
        assert hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()=='760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
        tokenizer=BpeSemanticTokenizer(tokenizer_file=args.tokenizer,auto_build_if_missing=False)
        records=[]
        for index,case in enumerate(cases):
            assert case['partition']=='train' and 601<=case['task_id']<=974
            assert case['case_sha256']==digest_record({k:v for k,v in case.items() if k!='case_sha256'})
            assert case['validator_sha256']==hashlib.sha256(case['private_check'].encode()).hexdigest()
            report['current_source_id']=case['source_id']
            report['attempted']+=1
            def validator(files):
                result=run_workspace_tests({**files,'__private_check.py':case['private_check']},'__private_check.py')
                return result['passed'],'External checks passed' if result['passed'] else 'External checks failed'
            env=IsolatedSoftwareEnvironment(case['initial_files'],validator=validator)
            actions=[TeacherAction('ACTION: READ_FILE test_public.py'),
                     TeacherAction('ACTION: WRITE_FILE '+case['target_module']+'\n'+case['reference_solution']),
                     TeacherAction('ACTION: RUN_TESTS'),TeacherAction('ACTION: FINISH Implementation verified')]
            record=record_executed_trajectory(case['initial_prompt'],env,actions,source_id=case['source_id']+':clean',validator_id=case['validator_sha256'])
            del record['record_sha256']
            record.update(partition='train',family_sha256=case['family_sha256'],source_case_sha256=case['case_sha256'],
                          protocol='mbpp_clean_trajectories_v1',source_revision=case['source_revision'])
            record['record_sha256']=digest_record(record)
            with (args.output/'attempts.jsonl').open('a') as stream:stream.write(json.dumps(record,ensure_ascii=True)+'\n')
            assert all(step['feedback']['success'] for step in record['steps']) and record['status']=='complete'
            encoded=encode_executed_trajectory(record,tokenizer,max_tokens=8192)
            assert encoded['prompt_tokens'].shape[1]<=4096 and int(encoded['reset_mask'].sum())==1
            for step in record['steps']:
                assert len(tokenizer.encode(step['action']))+1<=1024
                assert len(tokenizer.encode(observation_frame(ToolFeedback(**step['feedback']))))<=4096
            assert env.files=={**case['initial_files'],case['target_module']:case['reference_solution']}
            records.append(record)
            report['records'].append({'source_id':record['source_id'],'record_sha256':record['record_sha256'],
                                      'input_tokens':encoded['input_ids'].numel(),'supervised_tokens':int((encoded['labels']!=-100).sum())})
            report['accepted']+=1
            if (index+1)%32==0:
                (args.output/'progress.json').write_text(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
                print('MBPP_CLEAN_PROGRESS',index+1,357,flush=True)
        assert report['attempted']==report['accepted']==357
        raw=('\n'.join(json.dumps(record,sort_keys=True,ensure_ascii=True) for record in records)+'\n').encode()
        compressed=gzip.compress(raw,mtime=0)
        (args.output/'training-trajectories.jsonl.gz').write_bytes(compressed)
        report.update(status='complete',gzip_sha256=hashlib.sha256(compressed).hexdigest(),raw_sha256=hashlib.sha256(raw).hexdigest(),
                      input_tokens=sum(row['input_tokens'] for row in report['records']),supervised_tokens=sum(row['supervised_tokens'] for row in report['records']),
                      max_record_tokens=max(row['input_tokens'] for row in report['records']))
    except BaseException as exc:
        report.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('MBPP_CLEAN_RESULT',json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
