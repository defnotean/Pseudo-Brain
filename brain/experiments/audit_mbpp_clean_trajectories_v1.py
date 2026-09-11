"""Audit saved clean demonstrations without repeating sandbox execution."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from irene_brain.data.executed_trajectory import digest_record,encode_executed_trajectory
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','bank','tokenizer'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    bank_raw=args.bank.read_bytes()
    assert hashlib.sha256(bank_raw).hexdigest()=='e7645a3c98749815098ec66293e583087180b06c6bfd89249d6b1bb96f04d5d0'
    assert hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()=='760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
    cases=[json.loads(line) for line in gzip.decompress(bank_raw).splitlines()]
    corpus=args.root/'corpus'
    report=json.loads((corpus/'report.json').read_text())
    assert report['status']=='complete' and report['model_updates']==report['model_generated_tokens']==0
    compressed=(corpus/'training-trajectories.jsonl.gz').read_bytes()
    raw=gzip.decompress(compressed)
    assert hashlib.sha256(compressed).hexdigest()==report['gzip_sha256']
    assert hashlib.sha256(raw).hexdigest()==report['raw_sha256']
    records=[json.loads(line) for line in raw.splitlines()]
    attempts=[json.loads(line) for line in (corpus/'attempts.jsonl').read_text().splitlines()]
    assert len(cases)==len(records)==len(attempts)==len(report['records'])==report['attempted']==report['accepted']==357
    assert attempts==records
    tokenizer=BpeSemanticTokenizer(tokenizer_file=args.tokenizer,auto_build_if_missing=False)
    input_count=supervised_count=0
    for case,record,row in zip(cases,records,report['records']):
        assert case['case_sha256']==digest_record({k:v for k,v in case.items() if k!='case_sha256'})
        assert record['source_case_sha256']==case['case_sha256']
        assert record['partition']=='train' and case['partition']=='train' and 601<=case['task_id']<=974
        assert record['source_revision']==case['source_revision'] and record['family_sha256']==case['family_sha256']
        assert record['initial_prompt']==case['initial_prompt'] and record['initial_files']==case['initial_files']
        assert record['validator_id']==case['validator_sha256']==hashlib.sha256(case['private_check'].encode()).hexdigest()
        assert record['source_id']==row['source_id']==case['source_id']+':clean'
        expected=['ACTION: READ_FILE test_public.py','ACTION: WRITE_FILE '+case['target_module']+'\n'+case['reference_solution'],
                  'ACTION: RUN_TESTS','ACTION: FINISH Implementation verified']
        assert [step['action'] for step in record['steps']]==expected
        assert all(step['kind']=='teacher' and step['feedback']['success'] for step in record['steps'])
        assert record['steps'][-1]['feedback']['verified_completion']
        assert record['final_files_sha256']==digest_record({**case['initial_files'],case['target_module']:case['reference_solution']})
        encoded=encode_executed_trajectory(record,tokenizer,max_tokens=8192)
        assert encoded['record_sha256']==row['record_sha256']
        assert int(encoded['reset_mask'].sum())==1 and encoded['prompt_tokens'].shape[1]<=4096
        inputs=encoded['input_ids'].numel();supervised=int((encoded['labels']!=-100).sum())
        assert inputs==row['input_tokens'] and supervised==row['supervised_tokens']
        input_count+=inputs;supervised_count+=supervised
    assert input_count==report['input_tokens'] and supervised_count==report['supervised_tokens']
    result={'status':'complete','records':357,'input_tokens':input_count,'supervised_tokens':supervised_count,
            'gzip_sha256':report['gzip_sha256'],'report_sha256':hashlib.sha256((corpus/'report.json').read_bytes()).hexdigest(),
            'repeated_sandbox_executions':0,'model_updates':0}
    (args.root/'collection-audit.json').write_text(json.dumps(result,indent=2))
    print('MBPP_CLEAN_AUDIT',json.dumps(result),flush=True)


if __name__=='__main__':main()
