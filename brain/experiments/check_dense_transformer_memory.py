"""Discarded maximum-length transformer backward/update resource check."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch
from run_provenance import apply_deterministic_mode,provenance
from irene_brain.training.action_trajectory_loss import action_trajectory_loss
from irene_brain.unified.comparison_transformer import ComparisonTransformer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    checksum=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert checksum=='ca080bd8c6562656073dc90b64341cd624c931c1972c07c19ea9a2b4a59d1160'
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','checkpoint_sha256':checksum,'tokens':8192,
            'scope':'Synthetic maximum-length resource preflight; updated weights are discarded.'}
    started=time.monotonic()
    def deadline(*unused): raise TimeoutError('Registered300-second memory preflight')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(300)
    try:
        torch.set_num_threads(1)
        apply_deterministic_mode()
        torch.manual_seed(1066)
        model=ComparisonTransformer.from_payload(torch.load(args.checkpoint,map_location='cpu',weights_only=True)).cuda().train()
        generator=torch.Generator(device='cuda').manual_seed(1066)
        ids=torch.randint(1,32000,(1,8192),generator=generator,device='cuda')
        labels=torch.full_like(ids,-100)
        labels[:,:-1]=ids[:,1:]
        resets=torch.zeros_like(ids)
        resets[:,0]=1
        example={'input_ids':ids,'labels':labels,'reset_mask':resets,'prompt_tokens':ids[:,:128].clone()}
        report['provenance']=provenance(model=model,train_seed=198,eval_seed=1066,
            bank_digest=hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(),deterministic=True)
        report['cuda_total_memory_bytes']=torch.cuda.get_device_properties(0).total_memory
        report['cuda_free_before_optimizer_bytes']=torch.cuda.mem_get_info()[0]
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.1)
        torch.cuda.reset_peak_memory_stats()
        loss=action_trajectory_loss(model,example,chunk_size=256)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        assert torch.isfinite(loss) and torch.isfinite(norm)
        optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        torch.cuda.synchronize()
        report.update(status='passed',loss=loss.item(),gradient_norm=norm.item(),supervised_tokens=8191,
                      peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),discarded_optimizer_updates=1)
    except Exception as exc:
        report.update(status='incomplete',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('DENSE_TRANSFORMER_MEMORY_RESULT',json.dumps(report),flush=True)


if __name__=='__main__': main()
