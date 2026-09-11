"""Bounded sequential execution of the registered policy training/preflight pair."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_provenance import provenance
from train_tool_policy_v1 import digest,write_json,SELECTION


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','policy-corpus','foundation-corpus','tokenizer','output','recurrent-checkpoint','transformer-checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-preflight',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    source_manifest=args.root/'training-source-hashes.json'
    for relative,expected in json.loads(source_manifest.read_text()).items():
        assert digest(args.root/relative)==expected,relative
    report={'status':'running','models':{},'native_preflight':args.native_preflight,
            'source_manifest_sha256':digest(source_manifest),
            'provenance':provenance(train_seed=1986,eval_seed=1986,bank_digest=SELECTION,deterministic=False),
            'note':'Controller performs no model operations; strict deterministic execution is applied in each child.'}
    started=time.monotonic()
    try:
        for name in ('recurrent','transformer'):
            write_json(args.output/'status.json',{'stage':'native_preflight' if args.native_preflight else 'training','model':name})
            command=[sys.executable,str(args.root/'train_tool_policy_v1.py'),'--model',name,
                '--checkpoint',str(getattr(args,name+'_checkpoint')),'--policy-corpus',str(args.policy_corpus),
                '--foundation-corpus',str(args.foundation_corpus),'--tokenizer',str(args.tokenizer),
                '--output',str(args.output/name),'--source-manifest',str(source_manifest),
                '--source-manifest-sha256',digest(source_manifest)]
            if args.native_preflight: command.append('--native-preflight')
            timeout=300 if args.native_preflight else 1800
            timed_out=False
            model_started=time.monotonic()
            with (args.output/(name+'.log')).open('x') as log:
                try:
                    result=subprocess.run(command,env=os.environ.copy(),cwd=args.root,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                    returncode=result.returncode
                except subprocess.TimeoutExpired:
                    timed_out,returncode=True,None
            child_path=args.output/name/'report.json'
            child=json.loads(child_path.read_text()) if child_path.exists() else None
            expected='passed' if args.native_preflight else 'complete'
            passed=not timed_out and returncode==0 and child is not None and child['status']==expected
            entry={'status':expected if passed else 'incomplete','returncode':returncode,
                   'timed_out':timed_out,'elapsed_seconds':time.monotonic()-model_started,'child_report':child}
            report['models'][name]=entry
            write_json(args.output/'report.json',report)
            print('POLICY_PAIR_MODEL_DONE',name,entry['status'],flush=True)
            if not passed: raise RuntimeError(name+' did not complete the registered gate/run')
        report['status']='passed' if args.native_preflight else 'complete'
    except BaseException as exc:
        report.update(status='incomplete',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        write_json(args.output/'report.json',report)
        write_json(args.output/'status.json',{'stage':report['status']})
        print('POLICY_PAIR_RESULT',json.dumps(report),flush=True)


if __name__=='__main__': main()
