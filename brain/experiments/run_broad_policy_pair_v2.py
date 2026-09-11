"""Sequential bounded native preflight or full matched training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from train_tool_policy_v1 import digest,write_json


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-preflight',action='store_true')
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    manifest=args.root/'training-source-hashes.json';source_sha=digest(manifest)
    for relative,expected in json.loads(manifest.read_text()).items():assert digest(args.root/relative)==expected,relative
    selection=json.loads((args.root/'bank-manifest.json').read_text())['selection_sha256']
    if not args.native_preflight:
        preflight=json.loads((args.root/'native-preflight/report.json').read_text())
        assert preflight['status']=='passed' and preflight['source_manifest_sha256']==source_sha and preflight['selection_sha256']==selection
        for name in ('recurrent','transformer'):
            entry=preflight['models'][name]
            assert entry['status']=='passed' and entry['returncode']==0 and entry['child_report']['discarded_preflight_updates']==8
    started=time.monotonic()
    report={'status':'running','protocol':'broad_policy_v2','native_preflight':args.native_preflight,'models':{},
            'source_manifest_sha256':source_sha,'selection_sha256':selection}
    try:
        for name in ('recurrent','transformer'):
            write_json(args.output/'status.json',{'stage':'native_preflight' if args.native_preflight else 'training','model':name})
            checkpoint=Path('/content/pb-policy-training-v1-r2/training-v1')/name/'checkpoint-3520-trained.pt'
            command=[sys.executable,str(args.root/'train_broad_policy_v2.py'),'--model',name,'--root',str(args.root),
                     '--checkpoint',str(checkpoint),'--foundation-corpus','/content/pb-foundation-v5/corpus',
                     '--tokenizer','/content/pb-foundation-v5-training-r1/brain/data/tokenizers/pseudo_brain_bpe_32000.json',
                     '--output',str(args.output/name),'--source-sha256',source_sha]
            if args.native_preflight:command.append('--native-preflight')
            with (args.output/(name+'.log')).open('x') as log:
                result=subprocess.run(command,cwd=args.root,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,
                                      timeout=300 if args.native_preflight else 1800)
            path=args.output/name/'report.json'
            child=json.loads(path.read_text()) if path.exists() else None
            expected='passed' if args.native_preflight else 'complete'
            passed=result.returncode==0 and child is not None and child['status']==expected
            report['models'][name]={'status':expected if passed else 'incomplete','returncode':result.returncode,'child_report':child}
            write_json(args.output/'report.json',report)
            print('BROAD_POLICY_PAIR_MODEL',name,report['models'][name]['status'],flush=True)
            if not passed:raise RuntimeError(name+' did not pass the registered run')
        report['status']='passed' if args.native_preflight else 'complete'
    except BaseException as exc:
        report.update(status='incomplete',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        write_json(args.output/'report.json',report);write_json(args.output/'status.json',{'stage':report['status']})
        print('BROAD_POLICY_PAIR_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
