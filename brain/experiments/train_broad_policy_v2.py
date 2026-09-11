"""New matched broader policy experiment, with no automatic resume or promotion."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from run_provenance import apply_deterministic_mode,provenance
from train_tool_policy_v1 import digest,write_json,cuda_example,evaluate,DEV_HASH,TOKENIZER
from irene_brain.data.broad_policy_training_bank import prepare_broad_policy_bank,UPDATES
from irene_brain.data.policy_training_bank import load_rows,encode_foundation_row
from irene_brain.evaluation.independent_pilot import validate_streaming_parity
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.training.action_trajectory_loss import action_trajectory_loss
from irene_brain.unified.parallel_depth_model import ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer

CHECKPOINTS={'recurrent':'0b3edca04de150068c81fb8f71f05cd0aad8f42e423b9989bc789b919ffb85b7',
             'transformer':'346c9d4c9d334490cb2e93c54f4a1f0f66020a3833b4e7565f015b6d08c44dfe'}


def learning_rate(index):
    return 3e-5*(index+1)/64 if index<64 else 3e-6+.5*(3e-5-3e-6)*(1+math.cos(math.pi*(index-64)/(UPDATES-65)))


def update(model,optimizer,example,index):
    for group in optimizer.param_groups:group['lr']=learning_rate(index)
    optimizer.zero_grad(set_to_none=True)
    loss=action_trajectory_loss(model,cuda_example(example),chunk_size=256)
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return {'loss':loss.item(),'gradient_norm':grad.item(),'lr':learning_rate(index)}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=tuple(CHECKPOINTS),required=True)
    for name in ('root','checkpoint','foundation-corpus','tokenizer','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-sha256',required=True)
    parser.add_argument('--native-preflight',action='store_true')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();status=args.output/'status.json'
    report={'status':'incomplete','model':args.model,'protocol':'broad_policy_v2','native_preflight':args.native_preflight,
            'planned_updates':UPDATES,'completed_updates':0,'source_manifest_sha256':args.source_sha256}
    try:
        assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        torch.set_num_threads(1);apply_deterministic_mode()
        random.seed(1987);np.random.seed(1987);torch.manual_seed(1987);torch.cuda.manual_seed_all(1987)
        manifest=args.root/'training-source-hashes.json'
        assert digest(manifest)==args.source_sha256
        for relative,expected in json.loads(manifest.read_text()).items():assert digest(args.root/relative)==expected,relative
        assert digest(args.checkpoint)==CHECKPOINTS[args.model] and digest(args.tokenizer)==TOKENIZER
        tokenizer=BpeSemanticTokenizer(tokenizer_file=args.tokenizer,auto_build_if_missing=False)
        write_json(status,{'stage':'encoding'})
        bank=prepare_broad_policy_bank(args.root/'corpus',args.foundation_corpus,tokenizer)
        expected=json.loads((args.root/'bank-manifest.json').read_text())
        assert bank['manifest']==expected
        assert bank['schedule']==json.loads((args.root/'selection.json').read_text())
        selection=expected['selection_sha256'];report['selection_sha256']=selection
        cls=ParallelDepthModel if args.model=='recurrent' else ComparisonTransformer
        model=cls.from_payload(torch.load(args.checkpoint,map_location='cpu',weights_only=True)).cuda().train()
        assert sum(p.numel() for p in model.parameters())<36000000
        report['initial_provenance']=provenance(model=model,train_seed=1987,eval_seed=1987,bank_digest=selection,deterministic=True)
        report['checkpoint_sha256']=CHECKPOINTS[args.model]
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,betas=(.9,.95),eps=1e-8,weight_decay=.1)
        if args.native_preflight:
            checks=[]
            for index,row in enumerate(expected['preflight_selection']):
                example=bank['train'][row['index']] if row['kind']=='policy' else bank['replay'][row['index']]
                stats=update(model,optimizer,example,index)
                assert all(math.isfinite(value) for value in stats.values()) and stats['gradient_norm']>0
                assert all(torch.isfinite(p).all() for p in model.parameters())
                checks.append({**row,**stats})
            assert len(checks)==8
            if args.model=='recurrent':
                model.eval();report['parity']=validate_streaming_parity(model)
            report.update(status='passed',discarded_preflight_updates=8,checks=checks)
            return
        dev_rows=load_rows(args.foundation_corpus/'development.jsonl.gz',DEV_HASH)
        assert len(dev_rows)==1536
        foundation_dev=[encode_foundation_row(row,tokenizer) for row in dev_rows]
        for name,examples in (('policy',bank['development']),('foundation',foundation_dev)):
            write_json(args.output/('initial-'+name+'-development.json'),evaluate(model,examples,status,'initial_'+name))
        train_started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        counts={}
        def checkpoint(reason):
            path=args.output/f'checkpoint-{report["completed_updates"]:04d}-{reason}.pt'
            assert not path.exists()
            payload=model.checkpoint_payload()
            payload['model_state_dict']={key:value.detach().cpu().clone() for key,value in payload['model_state_dict'].items()}
            torch.save(payload,path)
            receipt={'checkpoint':path.name,'sha256':digest(path),'status':reason,'completed_updates':report['completed_updates'],
                     'counts':counts,'selection_sha256':selection,'training_seconds':time.monotonic()-train_started,'resume_supported':False}
            write_json(path.with_suffix('.json'),receipt)
            return receipt
        with (args.output/'updates.jsonl').open('x') as log:
            for index,row in enumerate(bank['schedule']):
                if time.monotonic()-train_started>=1200:
                    report.update(status='time_limit',checkpoint=checkpoint('time_limit'));return
                example=bank['train'][row['index']] if row['kind']=='policy' else bank['replay'][row['index']]
                stats=update(model,optimizer,example,index)
                report['completed_updates']=index+1
                stat=counts.setdefault(row['group'],{'updates':0,'input_tokens':0,'supervised_tokens':0})
                stat['updates']+=1
                for key in ('input_tokens','supervised_tokens'):stat[key]+=row[key]
                log.write(json.dumps({'update':index+1,**row,**stats})+'\n');log.flush()
                if (index+1)%64==0:
                    write_json(status,{'stage':'training','updates':index+1,'total':UPDATES,'training_seconds':time.monotonic()-train_started})
                    print('BROAD_POLICY_TRAIN',args.model,index+1,stats['loss'],flush=True)
                if index+1==3948:checkpoint('epoch')
        receipt=checkpoint('trained')
        report.update(checkpoint=receipt,counts=counts,training_seconds=time.monotonic()-train_started,peak_cuda_bytes=torch.cuda.max_memory_allocated())
        if report['training_seconds']>=1200:report['status']='time_limit';return
        assert counts==expected['groups']
        if args.model=='recurrent':model.eval();report['parity']=validate_streaming_parity(model)
        for name,examples in (('policy',bank['development']),('foundation',foundation_dev)):
            write_json(args.output/('final-'+name+'-development.json'),evaluate(model,examples,status,'final_'+name))
        report['status']='complete'
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        write_json(args.output/'report.json',report)
        write_json(status,{'stage':report['status'],'completed_updates':report['completed_updates']})
        print('BROAD_POLICY_RESULT',json.dumps(report),flush=True)


if __name__=='__main__':main()
