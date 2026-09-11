"""Registered deterministic language training with hash-bound resume receipts."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
    raise RuntimeError('Start Python with registered CUBLAS_WORKSPACE_CONFIG=:4096:8')

from run_provenance import apply_deterministic_mode, provenance
apply_deterministic_mode()
import torch
from irene_brain.runtime.policy import ResourcePolicy
from irene_brain.training.checkpoint import TrainerCursor
from irene_brain.training.language_checkpoint import save_language_boundary, restore_language_boundary
from irene_brain.evaluation.independent_pilot import validate_streaming_parity
from train_foundation_v4 import load_foundation_partition, training


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):
            value.update(block)
    return value.hexdigest()


def publish_json(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2)
        stream.flush()
        os.fsync(stream.fileno())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--model',choices=('recurrent','transformer'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-receipt',type=Path)
    parser.add_argument('--resume-receipt-sha256')
    parser.add_argument('--stop-after-updates',type=int)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    assert torch.cuda.is_available(), 'Requires authorized remote CUDA'
    torch.set_num_threads(1)
    if bool(args.resume_receipt) != bool(args.resume_receipt_sha256):
        raise ValueError('Resume requires both receipt and expected digest')
    assert digest(args.corpus/'manifest.json')==args.manifest_sha256
    assert digest(args.source_manifest)==args.source_manifest_sha256
    for relative,expected in json.loads(args.source_manifest.read_text()).items():
        assert digest(args.source_manifest.parent/relative)==expected
    manifest=json.loads((args.corpus/'manifest.json').read_text())
    assert manifest['status']=='complete'
    assert manifest['partitions']['train']['rows']==69632
    assert manifest['partitions']['development']['rows']==1536
    rows=load_foundation_partition(args.corpus,'train',manifest)
    dev_rows=load_foundation_partition(args.corpus,'development',manifest)
    training.assert_isolated(rows,dev_rows)
    total=len(rows)
    stop=args.stop_after_updates if args.stop_after_updates is not None else total
    if not 1<=stop<=total or (args.preflight and stop not in (32,64)):
        raise ValueError('Invalid registered stop boundary')
    token_path=Path(__file__).resolve().parents[1]/'data/tokenizers/pseudo_brain_bpe_32000.json'
    assert digest(token_path)==manifest['tokenizer_sha256']
    tokenizer=training.BpeSemanticTokenizer(tokenizer_file=token_path,auto_build_if_missing=False)
    # The preflight loads only its fixed first64 examples; its LR horizon stays total.
    examples=training.encode_rows(rows[:64] if args.preflight else rows,tokenizer)
    dev=[] if args.preflight else training.encode_rows(dev_rows,tokenizer)
    config={'version':1,'model':args.model,'total_updates':total,'seed':198,'lr':3e-4,
            'final_lr':3e-5,'warmup':64,'betas':[.9,.95],'eps':1e-8,'decay':.1,
            'clip':1.,'checkpoint_every':4096,'time_limit':3600.,'preflight':args.preflight,
            'loss':'response_mean','recurrent_bucket':256,'strict_deterministic':True}
    identity={'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
              'data_sha256':args.manifest_sha256,'code_sha256':args.source_manifest_sha256}
    cls=training.ParallelDepthModel if args.model=='recurrent' else training.ComparisonTransformer
    random.seed(198)
    torch.manual_seed(198)
    model=cls().cuda().train()
    assert sum(parameter.numel() for parameter in model.parameters())<36000000
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),eps=1e-8,weight_decay=.1)
    initial=provenance(model=model,train_seed=198,eval_seed=734,bank_digest=args.manifest_sha256,deterministic=True)
    completed,base_seconds,input_tokens,supervised_tokens=0,0.,0,0
    if args.resume_receipt:
        assert digest(args.resume_receipt)==args.resume_receipt_sha256
        receipt=json.loads(args.resume_receipt.read_text())
        assert receipt['identity']==identity and receipt['model']==args.model
        assert receipt['status']=='paused', 'Only registered planned pauses auto-resume'
        assert args.resume_receipt.parent.resolve()==args.output.resolve()
        assert json.loads((args.output/'run.json').read_text())['identity']==identity
        checkpoint=args.resume_receipt.parent/receipt['checkpoint_file']
        assert checkpoint.parent.resolve()==args.resume_receipt.parent.resolve()
        cursor=restore_language_boundary(checkpoint,model=model,optimizer=optimizer,
            checkpoint_sha256=receipt['checkpoint_sha256'],**identity)
        completed=cursor.optimizer_step
        assert cursor.epoch==0 and cursor.next_batch==completed and completed==receipt['completed_updates']
        base_seconds=receipt['training_seconds']
        input_tokens=receipt['input_tokens']
        supervised_tokens=receipt['supervised_tokens']
        assert input_tokens==sum(row['token_count'] for row in rows[:completed])
        assert supervised_tokens==sum(row['supervised_tokens'] for row in rows[:completed])
        assert args.output.is_dir()
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        publish_json(args.output/'run.json',{'config':config,'identity':identity,'initial_provenance':initial})
    assert completed<stop and base_seconds<3600
    attempt=args.output/('attempt-'+uuid.uuid4().hex)
    attempt.mkdir()
    status=attempt/'status.json'
    if completed==0 and not args.preflight:
        initial_dev=training.evaluate(model,dev,status,args.model,'initial_development')
        publish_json(args.output/'initial-development.json',initial_dev)
    start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    last_receipt=args.resume_receipt
    def boundary(reason):
        nonlocal last_receipt
        checkpoint=args.output/f'boundary-{completed:08d}-{reason}.pt'
        checksum=save_language_boundary(checkpoint,model=model,optimizer=optimizer,
            cursor=TrainerCursor(next_batch=completed,optimizer_step=completed),
            policy=ResourcePolicy(allow_artifact_write=True),**identity)
        record={'schema_version':1,'status':reason,'model':args.model,'identity':identity,
                'completed_updates':completed,'input_tokens':input_tokens,'supervised_tokens':supervised_tokens,
                'training_seconds':base_seconds+time.monotonic()-start,
                'checkpoint_file':checkpoint.name,'checkpoint_sha256':checksum,'attempt':attempt.name}
        if record['training_seconds']>=3600:
            record['status']='time_limit'
        last_receipt=checkpoint.with_suffix('.json')
        publish_json(last_receipt,record)
        training.write_json(args.output/'latest.json',{'receipt_file':last_receipt.name,'sha256':digest(last_receipt)})
        return record
    reason='paused'
    try:
        with (attempt/'updates.jsonl').open('x') as log:
            for index in range(completed,stop):
                if base_seconds+time.monotonic()-start>=3600:
                    reason='time_limit'
                    break
                row,ids,labels,prompt=examples[index]
                lr=3e-4*(index+1)/64 if index<64 else 3e-5+.5*(3e-4-3e-5)*(1+math.cos(math.pi*(index-64)/(total-65)))
                for group in optimizer.param_groups:
                    group['lr']=lr
                optimizer.zero_grad(set_to_none=True)
                loss=training.pilot_loss(model,ids.cuda(),labels.cuda(),prompt.cuda())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed=index+1
                input_tokens+=row['token_count']
                supervised_tokens+=row['supervised_tokens']
                log.write(json.dumps({'update':completed,'text_sha256':row['text_sha256'],'task':row['task'],
                    'loss':loss.item(),'lr':lr,'grad_norm':grad_norm.item()})+'\n')
                if completed%32==0:
                    log.flush()
                    training.write_json(status,{'stage':'training','updates':completed,'total':total,
                        'training_seconds':base_seconds+time.monotonic()-start})
                    print('FOUNDATION_V5_TRAIN',args.model,completed,loss.item(),flush=True)
                if completed%4096==0 and completed<stop:
                    log.flush()
                    os.fsync(log.fileno())
                    boundary('running')
            log.flush()
            os.fsync(log.fileno())
        if base_seconds+time.monotonic()-start>=3600:
            reason='time_limit'
        elif completed==total:
            reason='complete'
        record=boundary(reason)
        reason=record['status']
        publish_json(attempt/'completion.json',{**record,'peak_cuda_bytes':torch.cuda.max_memory_allocated()})
        if reason=='complete':
            payload=model.checkpoint_payload()
            payload.update(tokenizer_json=tokenizer._hf_tokenizer.to_str(),pilot_metadata=record)
            with (args.output/'inference.pt').open('xb') as stream:
                torch.save(payload,stream)
            final={**record,'inference_checkpoint_sha256':digest(args.output/'inference.pt')}
            final['development']=training.evaluate(model,dev,status,args.model,'final_development')
            final['samples']=training.generate_samples(model,tokenizer,args.model=='recurrent')
            if args.model=='recurrent':
                final['parity']=validate_streaming_parity(model)
            publish_json(args.output/'final.json',final)
        training.write_json(status,{'stage':reason,'completed_updates':completed})
        print('FOUNDATION_V5_BOUNDARY',json.dumps(record),flush=True)
    except Exception as exc:
        training.write_json(status,{'stage':'failed','completed_updates':completed,
            'error_type':type(exc).__name__,'error':str(exc),'last_receipt':None if last_receipt is None else str(last_receipt)})
        raise


if __name__=='__main__':
    main()
