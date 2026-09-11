"""Bounded post-training evaluation with an unchanged fixed task denominator."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_provenance import provenance
from evaluate_tool_policy import BANK, aggregate, digest, write_json
from evaluate_trained_tool_policy import resolve_completed_checkpoint


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','training-run','corpus','tokenizer','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--training-report-sha256',required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=args.root/'post-training-source-hashes.json'
    for relative,expected in json.loads(manifest.read_text()).items():
        assert digest(args.root/relative)==expected,relative
    for name,expected in BANK.items():
        assert digest(args.corpus/name)==expected,name
    checkpoints={name:resolve_completed_checkpoint(args.training_run,args.training_report_sha256,name)[1]
                 for name in ('recurrent','transformer')}
    requests=[json.loads(line) for line in (args.corpus/'requests.jsonl').read_text().splitlines()]
    plan=[{'source_id':row['source_id'],'mode':mode} for row in requests for mode in ('from_scratch','repair')]
    assert len(plan)==48
    report={'status':'running','models':{},'source_manifest_sha256':digest(manifest),
            'training_report_sha256':args.training_report_sha256,'checkpoint_hashes':checkpoints,
            'provenance':provenance(train_seed=1986,eval_seed=198,bank_digest=BANK['requests.jsonl'],deterministic=False),
            'controller_note':'No model operations in controller; each child applies strict deterministic mode.'}
    write_json(args.output/'plan.json',plan)
    write_json(args.output/'report.json',report)
    started=time.monotonic()
    for name in ('recurrent','transformer'):
        write_json(args.output/'status.json',{'stage':'evaluating','model':name})
        folder=args.output/name
        command=[sys.executable,str(args.root/'evaluate_trained_tool_policy.py'),'--model',name,
                 '--training-run',str(args.training_run),'--training-report-sha256',args.training_report_sha256,
                 '--corpus',str(args.corpus),'--tokenizer',str(args.tokenizer),'--output',str(folder)]
        timed_out=False
        model_started=time.monotonic()
        with (args.output/(name+'.log')).open('x') as log:
            try:
                result=subprocess.run(command,cwd=args.root,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,timeout=1800)
                returncode=result.returncode
            except subprocess.TimeoutExpired:
                timed_out,returncode=True,None
        cases=[]
        for index,task in enumerate(plan):
            path=folder/f'case-{index:02d}.json'
            if path.exists():
                row=json.loads(path.read_text())
                assert all(row[key]==value for key,value in task.items())
                cases.append(row)
        child=json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else None
        complete=not timed_out and returncode==0 and child is not None and child['status']=='complete' and len(cases)==48
        entry={'status':'complete' if complete else 'incomplete','returncode':returncode,'timed_out':timed_out,
               'elapsed_seconds':time.monotonic()-model_started,'scores':aggregate(plan,cases),'child_report':child,
               'case_hashes':{path.name:digest(path) for path in folder.glob('case-*.json')}}
        report['models'][name]=entry
        write_json(args.output/'report.json',report)
        print('TRAINED_TOOL_POLICY_MODEL_DONE',name,entry['status'],entry['scores']['completed_tasks'],flush=True)
    report['status']='complete' if all(row['status']=='complete' for row in report['models'].values()) else 'incomplete'
    report['elapsed_seconds']=time.monotonic()-started
    write_json(args.output/'report.json',report)
    write_json(args.output/'status.json',{'stage':report['status']})
    print('TRAINED_TOOL_POLICY_PAIR_DONE',report['status'],flush=True)


if __name__=='__main__': main()
