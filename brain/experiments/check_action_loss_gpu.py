"""Native paired action-loss/gradient gate on two frozen training records."""
import argparse
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch
from torch.nn import functional as F
from run_provenance import apply_deterministic_mode,provenance
from irene_brain.data.executed_trajectory import encode_executed_trajectory
from irene_brain.training.action_trajectory_loss import action_trajectory_loss
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.unified.parallel_depth_model import ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer


def main():
    parser=argparse.ArgumentParser()
    for name in ('recurrent_checkpoint','transformer_checkpoint','corpus','tokenizer','output'):
        parser.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    digests={'recurrent':'7feee0e9688dd4fb9014abedf677d09e071986c57ea4acdd8b8a97b61a2c28bf',
             'transformer':'ca080bd8c6562656073dc90b64341cd624c931c1972c07c19ea9a2b4a59d1160'}
    raw=(args.corpus/'train.jsonl.gz').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='896a39d478aaa674903c27eb9c7af0d96017bc721843c860a3d56daa056d73a0'
    rows=[json.loads(line) for line in gzip.decompress(raw).split(b'\n') if line][:2]
    assert [row['repair'] for row in rows]==[False,True]
    assert hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()=='760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
    args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','models':{},'records':[row['record_sha256'] for row in rows]}
    started=time.monotonic()
    def deadline(*unused): raise TimeoutError('Registered300-second action-gradient gate limit')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(300)
    try:
        torch.set_num_threads(1)
        apply_deterministic_mode()
        tokenizer=BpeSemanticTokenizer(tokenizer_file=args.tokenizer,auto_build_if_missing=False)
        examples=[encode_executed_trajectory(row,tokenizer,max_tokens=8192) for row in rows]
        for name,cls in (('recurrent',ParallelDepthModel),('transformer',ComparisonTransformer)):
            checkpoint=getattr(args,name+'_checkpoint')
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==digests[name]
            model=cls.from_payload(torch.load(checkpoint,map_location='cpu',weights_only=True)).cuda().train()
            entry={'checkpoint_sha256':digests[name],'cases':[],
                   'provenance':provenance(model=model,train_seed=198,eval_seed=None,
                       bank_digest=hashlib.sha256(json.dumps(report['records']).encode()).hexdigest(),deterministic=True)}
            report['models'][name]=entry
            for source,example in zip(rows,examples):
                encoded={key:value.cuda() if isinstance(value,torch.Tensor) else value for key,value in example.items()}
                model.zero_grad(set_to_none=True)
                logits=model(encoded['input_ids'],reset_mask=encoded['reset_mask'],prompt_tokens=encoded['prompt_tokens'])
                assert torch.isfinite(logits).all()
                active=encoded['labels']!=-100
                reference=F.cross_entropy(logits[active],encoded['labels'][active])
                reference.backward()
                reference_value=reference.item()
                gradients={key:param.grad.detach().clone() for key,param in model.named_parameters() if param.grad is not None}
                del logits,reference
                assert all(torch.isfinite(value).all() for value in gradients.values())
                model.zero_grad(set_to_none=True)
                actual=action_trajectory_loss(model,encoded,chunk_size=256)
                actual.backward()
                atol,rtol=(1e-10,1e-8) if name=='recurrent' else (1e-6,1e-4)
                torch.testing.assert_close(actual,torch.tensor(reference_value,device='cuda',dtype=actual.dtype),atol=atol,rtol=rtol)
                maximum=0.
                participating=0
                for key,param in model.named_parameters():
                    assert (param.grad is not None)==(key in gradients)
                    if param.grad is not None:
                        assert torch.isfinite(param.grad).all()
                        torch.testing.assert_close(param.grad,gradients[key],atol=atol,rtol=rtol,msg=key)
                        maximum=max(maximum,(param.grad-gradients[key]).abs().max().item())
                        participating+=1
                case={'record_sha256':source['record_sha256'],'input_tokens':encoded['input_ids'].numel(),
                      'supervised_tokens':int(active.sum()),'reference_loss':reference_value,'chunked_loss':actual.item(),
                      'max_gradient_absolute_error':maximum,'parameter_gradients_checked':participating,'status':'passed'}
                entry['cases'].append(case)
                print('ACTION_LOSS_GPU_CASE',name,json.dumps(case),flush=True)
                del actual,gradients,encoded
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed' if isinstance(exc,AssertionError) else 'incomplete',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('ACTION_LOSS_GPU_RESULT',json.dumps(report),flush=True)


if __name__=='__main__': main()
