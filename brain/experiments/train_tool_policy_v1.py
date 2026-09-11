"""Registered paired-policy fine-tuning; no automatic resume or model selection."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from run_provenance import apply_deterministic_mode, provenance
from irene_brain.data.policy_training_bank import prepare_policy_training_bank, encode_foundation_row, load_rows
from irene_brain.evaluation.independent_pilot import validate_streaming_parity
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.training.action_trajectory_loss import action_trajectory_loss
from irene_brain.unified.parallel_depth_model import ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer


CHECKPOINTS={'recurrent':'7feee0e9688dd4fb9014abedf677d09e071986c57ea4acdd8b8a97b61a2c28bf',
             'transformer':'ca080bd8c6562656073dc90b64341cd624c931c1972c07c19ea9a2b4a59d1160'}
SELECTION='5933901753e5416e75ae2fac765e3d21746de29d3e7437415a0fd12420ca98cd'
DEV_HASH='da54b4cb7bcaabaea216e1db7b4f7965e96393d99dd1d20207f7096f245ad90d'
TOKENIZER='760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''): value.update(block)
    return value.hexdigest()


def write_json(path,value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)


def cuda_example(example):
    return {key:value.cuda() if isinstance(value,torch.Tensor) else value for key,value in example.items()}


def learning_rate(index):
    return 1e-4*(index+1)/32 if index<32 else 1e-5+.5*(1e-4-1e-5)*(1+math.cos(math.pi*(index-32)/(3520-33)))


def update(model,optimizer,example,index):
    lr=learning_rate(index)
    for group in optimizer.param_groups: group['lr']=lr
    optimizer.zero_grad(set_to_none=True)
    loss=action_trajectory_loss(model,cuda_example(example),chunk_size=256)
    loss.backward()
    grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    return {'loss':loss.item(),'gradient_norm':grad.item(),'lr':lr}


def evaluate(model,examples,status_path,stage):
    model.eval()
    totals,details={},[]
    with torch.no_grad():
        for index,example in enumerate(examples):
            loss=action_trajectory_loss(model,cuda_example(example),chunk_size=256).item()
            tokens=int((example['labels']!=-100).sum())
            group=example.get('task','policy')
            stat=totals.setdefault(group,{'nll_sum':0.,'supervised_tokens':0,'examples':0})
            stat['nll_sum']+=loss*tokens
            stat['supervised_tokens']+=tokens
            stat['examples']+=1
            details.append({'record_sha256':example.get('record_sha256',example.get('text_sha256')),
                            'nll':loss,'supervised_tokens':tokens,'group':group})
            if (index+1)%32==0:
                write_json(status_path,{'stage':stage,'evaluated':index+1,'total':len(examples)})
    for stat in totals.values(): stat['token_weighted_nll']=stat['nll_sum']/stat['supervised_tokens']
    model.train()
    return {'groups':totals,'examples':details}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=tuple(CHECKPOINTS),required=True)
    for name in ('checkpoint','policy-corpus','foundation-corpus','tokenizer','output','source-manifest'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-manifest-sha256',required=True)
    parser.add_argument('--native-preflight',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    report={'status':'incomplete','model':args.model,'completed_updates':0,'planned_updates':3520,
            'native_preflight':args.native_preflight,'selection_sha256':SELECTION,
            'loss_paths':{'recurrent':'chunk256 with recurrent bucket padding','transformer':'full reference readout'},
            'protocol_amendment':'policy_training_v1_amendment_2'}
    status=args.output/'status.json'
    try:
        assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'
        torch.set_num_threads(1)
        apply_deterministic_mode()
        random.seed(1986)
        np.random.seed(1986)
        torch.manual_seed(1986)
        torch.cuda.manual_seed_all(1986)
        assert digest(args.source_manifest)==args.source_manifest_sha256
        for relative,expected in json.loads(args.source_manifest.read_text()).items():
            assert digest(args.source_manifest.parent/relative)==expected,relative
        assert digest(args.checkpoint)==CHECKPOINTS[args.model]
        assert digest(args.tokenizer)==TOKENIZER
        tokenizer=BpeSemanticTokenizer(tokenizer_file=args.tokenizer,auto_build_if_missing=False)
        write_json(status,{'stage':'encoding'})
        bank=prepare_policy_training_bank(args.policy_corpus,args.foundation_corpus,tokenizer)
        assert bank['manifest']['selection_sha256']==SELECTION
        write_json(args.output/'bank-manifest.json',bank['manifest'])
        write_json(args.output/'selection.json',bank['schedule'])
        cls=ParallelDepthModel if args.model=='recurrent' else ComparisonTransformer
        model=cls.from_payload(torch.load(args.checkpoint,map_location='cpu',weights_only=True)).cuda().train()
        assert sum(parameter.numel() for parameter in model.parameters())<36000000
        report['initial_provenance']=provenance(model=model,train_seed=1986,eval_seed=1986,bank_digest=SELECTION,deterministic=True)
        report['inputs']={'checkpoint_sha256':CHECKPOINTS[args.model],'tokenizer_sha256':TOKENIZER,
                          'source_manifest_sha256':args.source_manifest_sha256}
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,betas=(.9,.95),eps=1e-8,weight_decay=.1)
        if args.native_preflight:
            # Two discarded updates exercise both data routes on native model shapes.
            selected=[bank['schedule'][0],bank['schedule'][4]]
            assert [row['kind'] for row in selected]==['policy','foundation']
            checks=[]
            for index,row in enumerate(selected):
                example=bank['train'][row['index']] if row['kind']=='policy' else bank['replay'][row['index']]
                stats=update(model,optimizer,example,index)
                assert all(math.isfinite(value) for value in stats.values())
                assert stats['gradient_norm']>0
                assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
                checks.append({**row,**stats})
            report.update(status='passed',discarded_preflight_updates=2,checks=checks)
            return
        dev_rows=load_rows(args.foundation_corpus/'development.jsonl.gz',DEV_HASH)
        assert len(dev_rows)==1536
        foundation_dev=[encode_foundation_row(row,tokenizer) for row in dev_rows]
        for name,examples in (('policy',bank['development']),('foundation',foundation_dev)):
            write_json(args.output/('initial-'+name+'-development.json'),evaluate(model,examples,status,'initial_'+name))
        train_started=time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        counts={'policy':{'input_tokens':0,'supervised_tokens':0,'updates':0},
                'foundation':{'input_tokens':0,'supervised_tokens':0,'updates':0}}
        def checkpoint(reason):
            path=args.output/f'checkpoint-{report["completed_updates"]:04d}-{reason}.pt'
            assert not path.exists()
            payload=model.checkpoint_payload()
            payload['model_state_dict']={key:value.detach().cpu().clone() for key,value in payload['model_state_dict'].items()}
            torch.save(payload,path)
            receipt={'checkpoint':path.name,'sha256':digest(path),'status':reason,
                     'completed_updates':report['completed_updates'],'counts':counts,
                     'selection_sha256':SELECTION,'training_seconds':time.monotonic()-train_started,
                     'resume_supported':False}
            write_json(path.with_suffix('.json'),receipt)
            return receipt
        with (args.output/'updates.jsonl').open('x') as log:
            for index,row in enumerate(bank['schedule']):
                if time.monotonic()-train_started>=900:
                    report.update(status='time_limit',checkpoint=checkpoint('time_limit'))
                    return
                example=bank['train'][row['index']] if row['kind']=='policy' else bank['replay'][row['index']]
                stats=update(model,optimizer,example,index)
                report['completed_updates']=index+1
                count=counts[row['kind']]
                count['updates']+=1
                for key in ('input_tokens','supervised_tokens'): count[key]+=row[key]
                log.write(json.dumps({'update':index+1,**row,**stats})+'\n')
                log.flush()
                if (index+1)%32==0:
                    write_json(status,{'stage':'training','updates':index+1,'total':3520,
                        'training_seconds':time.monotonic()-train_started})
                    print('POLICY_TRAIN',args.model,index+1,stats['loss'],flush=True)
                if (index+1)%880==0 and index+1<3520: checkpoint('progress')
        receipt=checkpoint('trained')
        report.update(checkpoint=receipt,training_seconds=time.monotonic()-train_started,counts=counts,
                      peak_cuda_bytes=torch.cuda.max_memory_allocated())
        if report['training_seconds']>=900:
            report['status']='time_limit'
            return
        assert sum(count['input_tokens'] for count in counts.values())==bank['manifest']['input_tokens']
        assert sum(count['supervised_tokens'] for count in counts.values())==bank['manifest']['supervised_tokens']
        if args.model=='recurrent':
            model.eval()
            report['parity']=validate_streaming_parity(model)
        for name,examples in (('policy',bank['development']),('foundation',foundation_dev)):
            write_json(args.output/('final-'+name+'-development.json'),evaluate(model,examples,status,'final_'+name))
        report['status']='complete'
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        write_json(args.output/'report.json',report)
        write_json(status,{'stage':report['status'],'completed_updates':report['completed_updates']})
        print('POLICY_TRAIN_RESULT',json.dumps(report),flush=True)


if __name__=='__main__': main()
