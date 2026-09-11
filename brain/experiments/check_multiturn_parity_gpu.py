"""Native trained multi-turn mechanical gate; no optimization or task scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch
from run_provenance import apply_deterministic_mode,provenance
from irene_brain.data.executed_trajectory import encode_executed_trajectory
from irene_brain.evaluation.multiturn_parity import validate_multiturn_parity
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.unified.parallel_depth_model import ParallelDepthModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    expected={'checkpoint':'7feee0e9688dd4fb9014abedf677d09e071986c57ea4acdd8b8a97b61a2c28bf',
              'tokenizer':'760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e',
              'record':'1f8dab043cd5e65315a0c8063939b1692dee70ba1d0944f0de683348f315c139'}
    for name,digest in expected.items():
        assert hashlib.sha256(getattr(args,name).read_bytes()).hexdigest()==digest
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','inputs':expected,'gpu':torch.cuda.get_device_name()}
    start=time.monotonic()
    def deadline(*unused): raise TimeoutError('Registered300-second multi-turn gate limit')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(300)
    try:
        torch.set_num_threads(1)
        apply_deterministic_mode()
        model=ParallelDepthModel.from_payload(torch.load(args.checkpoint,map_location='cpu',weights_only=True)).cuda().eval()
        tokenizer=BpeSemanticTokenizer(tokenizer_file=args.tokenizer,auto_build_if_missing=False)
        record=json.loads(args.record.read_text())
        encoded=encode_executed_trajectory(record,tokenizer,max_tokens=8192)
        report['provenance']=provenance(model=model,train_seed=198,eval_seed=None,
            bank_digest=expected['record'],deterministic=True)
        report['result']=validate_multiturn_parity(model,encoded,resp_id=tokenizer.resp_id,eos_id=tokenizer.eos_id,generation_limit=16)
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed' if isinstance(exc,AssertionError) else 'incomplete',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('MULTITURN_GPU_RESULT',json.dumps(report),flush=True)


if __name__=='__main__': main()
