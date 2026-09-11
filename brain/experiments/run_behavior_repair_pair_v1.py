"""Sequential bounded actor collection; no optimizer or model in controller."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_tool_policy import write_json,digest
from collect_behavior_repair_v1 import select_cases,TRAINING_REPORT,CHECKPOINTS
from evaluate_trained_tool_policy import resolve_completed_checkpoint


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','bank','training-run','tokenizer','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=False)
    plan=[{'source_id':c['source_id'],'case_sha256':c['case_sha256']} for c in select_cases(args.bank)]
    assert plan==json.loads((args.root/'selection.json').read_text())
    for name,expected in CHECKPOINTS.items():
        assert resolve_completed_checkpoint(args.training_run,TRAINING_REPORT,name)[1]==expected
    report={'status':'running','models':{},'planned_actor_attempts':64,'model_updates':0,
            'source_manifest_sha256':digest(args.root/'source-manifest.json'),'collection_is_training_data_not_benchmark':True}
    write_json(args.output/'report.json',report)
    started=time.monotonic()
    for name in ('recurrent','transformer'):
        write_json(args.output/'status.json',{'stage':'collecting','model':name})
        command=[sys.executable,str(args.root/'collect_behavior_repair_v1.py'),'--model',name]
        for key in ('root','bank','training-run','tokenizer'):
            command+=['--'+key,str(getattr(args,key.replace('-','_')))]
        folder=args.output/name
        command+=['--output',str(folder)]
        timed_out=False
        with (args.output/(name+'.log')).open('x') as log:
            try:
                result=subprocess.run(command,cwd=args.root,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,timeout=960)
                returncode=result.returncode
            except subprocess.TimeoutExpired:
                returncode=None
                timed_out=True
        child=json.loads((folder/'report.json').read_text()) if (folder/'report.json').exists() else None
        observed=[]
        for index,expected in enumerate(plan):
            path=folder/f'actor-{index:02d}.json'
            if path.exists():
                row=json.loads(path.read_text())
                assert all(row[key]==value for key,value in expected.items())
                observed.append(index)
        complete=not timed_out and returncode==0 and child is not None and child['status']=='complete' and len(observed)==32
        report['models'][name]={'status':'complete' if complete else 'incomplete','returncode':returncode,'timed_out':timed_out,
                               'recorded_attempts':len(observed),'missing_indices':[i for i in range(32) if i not in observed],'child_report':child}
        write_json(args.output/'report.json',report)
        print('BEHAVIOR_ACTOR_DONE',name,report['models'][name]['status'],flush=True)
        if not complete: break
    report['status']='complete' if len(report['models'])==2 and all(row['status']=='complete' for row in report['models'].values()) else 'incomplete'
    report['elapsed_seconds']=time.monotonic()-started
    write_json(args.output/'report.json',report)
    print('BEHAVIOR_PAIR_DONE',report['status'],flush=True)


if __name__=='__main__':main()
