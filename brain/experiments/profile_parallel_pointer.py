"""Isolated parallel-readout equivalence and long-prompt cost; no model updates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import time

import torch
from run_provenance import apply_deterministic_mode, provenance
from irene_brain.unified.parallel_depth_model import ParallelDepthModel


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    checksum=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert checksum=='7feee0e9688dd4fb9014abedf677d09e071986c57ea4acdd8b8a97b61a2c28bf'
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    apply_deterministic_mode()
    report={'status':'running','checkpoint_sha256':checksum,'cases':[],
            'scope':'Isolated readout only; does not parallelize autoregressive decisions or establish end-to-end generation speed.',
            'key_cache_used':False}
    started=time.monotonic()
    def deadline(*unused): raise TimeoutError('Registered300-second pointer profile limit')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(300)
    try:
        model=ParallelDepthModel.from_payload(torch.load(args.checkpoint,map_location='cpu',weights_only=True)).cuda().eval()
        generator=torch.Generator(device='cuda').manual_seed(1065)
        bank=hashlib.sha256()
        def timed(function):
            function()
            torch.cuda.synchronize()
            durations=[]
            torch.cuda.reset_peak_memory_stats()
            for _ in range(3):
                torch.cuda.synchronize()
                start=time.perf_counter()
                result=function()
                torch.cuda.synchronize()
                durations.append(time.perf_counter()-start)
                del result
            return {'median_seconds':statistics.median(durations),'samples_seconds':durations,
                    'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
        with torch.no_grad():
            for length in (31,4096,32768):
                prompt=torch.randint(1,32000,(1,length),device='cuda',generator=generator)
                prompt[:,::11]=53
                prompt[:,-1]=0
                for count in (1,32):
                    features=torch.randn(1,count,model.config.width,device='cuda',dtype=torch.float64,generator=generator)
                    ids=prompt[:,:count].clone() if count<=length else torch.full((1,count),53,device='cuda',dtype=torch.long)
                    ids[:,::3]=53
                    mask=torch.ones_like(ids)
                    if count>1: mask[:,::7]=0
                    for tensor in (prompt,features,ids,mask): bank.update(tensor.cpu().numpy().tobytes())
                    original_prompt=prompt.clone()
                    def parallel(): return model.readout(features,ids,prompt,mask)
                    def serial():
                        return torch.cat([model.readout(features[:,index:index+1],ids[:,index:index+1],prompt,mask[:,index:index+1])
                                          for index in range(count)],dim=1)
                    actual,reference=parallel(),serial()
                    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
                    error=(actual-reference).abs().max().item()
                    assert error<1e-6 and torch.equal(prompt,original_prompt)
                    del actual,reference
                    parallel_stats=timed(parallel)
                    serial_stats=timed(serial)
                    no_pointer_stats=timed(lambda:model.readout(features,ids,None))
                    case={'prompt_tokens':length,'query_rows':count,'max_absolute_logit_error':error,
                          'parallel_readout':parallel_stats,'serial_readout':serial_stats,
                          'readout_without_pointer':no_pointer_stats,
                          'isolated_row_batching_speedup':serial_stats['median_seconds']/parallel_stats['median_seconds'],
                          'pointer_overhead_seconds':parallel_stats['median_seconds']-no_pointer_stats['median_seconds'],
                          'status':'passed'}
                    report['cases'].append(case)
                    (args.output/'progress.json').write_text(json.dumps(report,indent=2))
                    print('POINTER_PROFILE_CASE',json.dumps(case),flush=True)
        report['provenance']=provenance(model=model,train_seed=198,eval_seed=1065,bank_digest=bank.hexdigest(),deterministic=True)
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed' if isinstance(exc,AssertionError) else 'incomplete',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('POINTER_PROFILE_RESULT',json.dumps(report),flush=True)


if __name__=='__main__': main()
