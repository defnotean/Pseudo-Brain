"""Read-only diagnosis of the failed native transformer action-gradient gate."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import types

import torch
from torch.nn import functional as F
from run_provenance import apply_deterministic_mode,provenance
from irene_brain.data.executed_trajectory import encode_executed_trajectory
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.training.action_trajectory_loss import action_trajectory_loss
from irene_brain.unified.comparison_transformer import ComparisonTransformer


def compare(actual,reference):
    details=[]
    assert set(actual)==set(reference)
    total_error,total_reference=0.,0.
    for name in actual:
        left,right=actual[name],reference[name]
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        error=(left-right).abs()
        outside=error>(1e-6+1e-4*right.abs())
        location=int(error.flatten().argmax())
        square=float((left.double()-right.double()).square().sum())
        ref_square=float(right.double().square().sum())
        total_error+=square
        total_reference+=ref_square
        details.append({'parameter':name,'max_absolute_error':error.max().item(),
                        'elements_outside_registered_tolerance':int(outside.sum()),'elements':left.numel(),
                        'reference_at_max_error':right.flatten()[location].item(),
                        'actual_at_max_error':left.flatten()[location].item(),
                        'relative_l2_error':(square/max(ref_square,1e-300))**.5})
    return {'bit_exact':all(torch.equal(actual[name],reference[name]) for name in actual),
            'max_absolute_error':max(row['max_absolute_error'] for row in details),
            'parameters_outside_registered_tolerance':sum(row['elements_outside_registered_tolerance']>0 for row in details),
            'relative_l2_error':(total_error/max(total_reference,1e-300))**.5,'parameters':details}


def main():
    parser=argparse.ArgumentParser()
    for name in ('checkpoint','corpus','tokenizer','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()=='ca080bd8c6562656073dc90b64341cd624c931c1972c07c19ea9a2b4a59d1160'
    raw=(args.corpus/'train.jsonl.gz').read_bytes()
    assert hashlib.sha256(raw).hexdigest()=='896a39d478aaa674903c27eb9c7af0d96017bc721843c860a3d56daa056d73a0'
    assert hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()=='760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report={'status':'running','cases':[],'optimizer_updates':0,'scope':'Diagnostic only; original failed gate remains failed.'}
    def deadline(*unused): raise TimeoutError('Registered300-second gradient diagnosis limit')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(300)
    try:
        torch.set_num_threads(1)
        apply_deterministic_mode()
        model=ComparisonTransformer.from_payload(torch.load(args.checkpoint,map_location='cpu',weights_only=True)).cuda().train()
        rows=[json.loads(line) for line in gzip.decompress(raw).split(b'\n') if line][:2]
        tokenizer=BpeSemanticTokenizer(tokenizer_file=args.tokenizer,auto_build_if_missing=False)
        report['provenance']=provenance(model=model,train_seed=198,eval_seed=None,
            bank_digest=hashlib.sha256(json.dumps([row['record_sha256'] for row in rows]).encode()).hexdigest(),deterministic=True)
        captured=[]
        original_features=model.features_parallel
        def capture(self,*arguments,**keywords):
            features=original_features(*arguments,**keywords)
            features.retain_grad()
            captured.append(features)
            return features
        model.features_parallel=types.MethodType(capture,model)
        for row in rows:
            example=encode_executed_trajectory(row,tokenizer,max_tokens=8192)
            example={key:value.cuda() if isinstance(value,torch.Tensor) else value for key,value in example.items()}
            ids,labels,prompt,resets=(example[key] for key in ('input_ids','labels','prompt_tokens','reset_mask'))
            modes={}
            for mode in ('full_selected','full_selected_repeat','full_dense','chunked','chunked_repeat','chunked_no_checkpoint'):
                model.zero_grad(set_to_none=True)
                captured.clear()
                if mode.startswith('full'):
                    logits=model(ids,reset_mask=resets,prompt_tokens=prompt)
                    if mode=='full_dense':
                        loss=F.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=-100,reduction='sum')/(labels!=-100).sum()
                    else:
                        active=labels!=-100
                        loss=F.cross_entropy(logits[active],labels[active])
                elif mode=='chunked_no_checkpoint':
                    features=model.features_parallel(ids,resets)
                    total=features.new_zeros(())
                    for start in range(0,ids.shape[1],256):
                        logits=model.readout(features[:,start:start+256],ids[:,start:start+256],prompt)
                        total=total+F.cross_entropy(logits.flatten(0,1),labels[:,start:start+256].flatten(),ignore_index=-100,reduction='sum')
                    loss=total/(labels!=-100).sum()
                else:
                    loss=action_trajectory_loss(model,example,chunk_size=256)
                assert torch.isfinite(loss)
                loss.backward()
                assert len(captured)==1 and captured[0].grad is not None
                modes[mode]={'loss':loss.item(),'gradients':{name:parameter.grad.detach().clone() for name,parameter in model.named_parameters() if parameter.grad is not None},
                             'features_gradient':captured[0].grad.detach().clone()}
                del loss
                captured.clear()
            comparisons={}
            for actual,reference in (('full_selected_repeat','full_selected'),('full_dense','full_selected'),
                                     ('chunked','full_selected'),('chunked_repeat','chunked'),('chunked_no_checkpoint','chunked')):
                comparisons[actual+'_vs_'+reference]={'parameters':compare(modes[actual]['gradients'],modes[reference]['gradients']),
                    'features':compare({'features':modes[actual]['features_gradient']},{'features':modes[reference]['features_gradient']})}
            case={'record_sha256':row['record_sha256'],'input_tokens':ids.numel(),
                  'losses':{name:value['loss'] for name,value in modes.items()},'comparisons':comparisons}
            report['cases'].append(case)
            (args.output/'progress.json').write_text(json.dumps(report,indent=2))
            print('ACTION_GRADIENT_DIAG_CASE',json.dumps({'record_sha256':case['record_sha256'],'losses':case['losses'],
                'comparisons':{name:{kind:{key:value for key,value in stats.items() if key!='parameters'} for kind,stats in entry.items()} for name,entry in comparisons.items()}}),flush=True)
            del modes,example
        report['status']='complete'
    except Exception as exc:
        report.update(status='incomplete',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('ACTION_GRADIENT_DIAG_RESULT',report['status'],report['elapsed_seconds'],flush=True)


if __name__=='__main__': main()
