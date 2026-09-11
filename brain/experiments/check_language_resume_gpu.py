"""One fresh-process phase of the registered native language restart gate."""
import argparse
import hashlib
import json
from pathlib import Path
import random

import torch

from run_provenance import provenance
from irene_brain.data.sequence_buckets import bucket_training_example
from irene_brain.runtime.policy import ResourcePolicy
from irene_brain.training.checkpoint import TrainerCursor
from irene_brain.training.language_checkpoint import save_language_boundary, restore_language_boundary
from irene_brain.unified.parallel_depth_model import ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key:cpu(item) for key,item in value.items()}
    if isinstance(value, (list,tuple)):
        return type(value)(cpu(item) for item in value)
    return value


def same(actual, expected, path='root'):
    if isinstance(expected, torch.Tensor):
        assert actual.dtype == expected.dtype and torch.equal(actual.cpu(),expected.cpu()), path
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys(), path
        for key in expected:
            same(actual[key],expected[key],path+'/'+str(key))
    elif isinstance(expected, (list,tuple)):
        assert len(actual)==len(expected),path
        for index,(left,right) in enumerate(zip(actual,expected)):
            same(left,right,path+'/'+str(index))
    else:
        assert actual==expected,path


def draws():
    return [random.random(),torch.rand(5),torch.rand(5,device='cuda').cpu()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=('recurrent','transformer'),required=True)
    parser.add_argument('--phase',choices=('baseline','interrupted','resumed'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'Requires authorized remote CUDA'
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    bank, digest = [], hashlib.sha256()
    generator = torch.Generator().manual_seed(965)
    for length in (31,257,2049,31,257,2049):
        ids = torch.randint(1,32000,(1,length),generator=generator)
        labels = ids.roll(-1,1)
        labels[:,:length//2] = -100
        labels[:,-1] = -100
        prompt = ids[:,:length//2].clone()
        bank.append((ids,labels,prompt))
        for tensor in bank[-1]:
            digest.update(str(list(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    bank_hash = digest.hexdigest()
    random.seed(198)
    torch.manual_seed(198)
    cls = ParallelDepthModel if args.model=='recurrent' else ComparisonTransformer
    model = cls().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),eps=1e-8,weight_decay=.1)
    record = provenance(model=model,train_seed=198,eval_seed=965,bank_digest=bank_hash,
                        deterministic=torch.are_deterministic_algorithms_enabled())
    source_hashes = json.loads((Path(__file__).parent/'source-hashes.json').read_text())
    for relative,expected in source_hashes.items():
        assert hashlib.sha256((Path(__file__).parent/relative).read_bytes()).hexdigest()==expected
    identity = {'config_sha256':hashlib.sha256((args.model+'|lr=3e-4/(i+1)|steps=6|adamw=.9,.95,1e-8,.1|clip=1|bucket=256').encode()).hexdigest(),
                'data_sha256':bank_hash,
                'code_sha256':hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()}
    path = args.output/'boundary.pt'
    first,last = 0,3 if args.phase=='interrupted' else 6
    if args.phase=='resumed':
        saved = json.loads((args.output/'interrupted.json').read_text())
        random.seed(3)
        torch.manual_seed(3)
        cursor = restore_language_boundary(path,model=model,optimizer=optimizer,
                    checkpoint_sha256=saved['boundary_sha256'],**identity)
        assert cursor==TrainerCursor(next_batch=3,optimizer_step=3)
        first = cursor.next_batch
    losses,sentinels = [],[]
    for index in range(first,last):
        ids,labels,prompt = (tensor.cuda() for tensor in bank[index])
        if args.model=='recurrent':
            ids,labels = bucket_training_example(ids,labels)
        for group in optimizer.param_groups:
            group['lr'] = 3e-4/(index+1)
        loss = model.language_loss(ids,labels,prompt_tokens=prompt)
        assert torch.isfinite(loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
        sentinels.append(draws())
        print('RESUME_UPDATE',args.model,args.phase,index+1,loss.item(),flush=True)
    report = {'status':'passed','model':args.model,'phase':args.phase,'losses':losses,
              'provenance':record,'identity':identity}
    if args.phase=='interrupted':
        report['boundary_sha256'] = save_language_boundary(path,model=model,optimizer=optimizer,
            cursor=TrainerCursor(next_batch=3,optimizer_step=3),policy=ResourcePolicy(allow_artifact_write=True),**identity)
    elif args.phase=='baseline':
        torch.save({'model':cpu(model.state_dict()),'optimizer':cpu(optimizer.state_dict()),
                    'losses':losses,'sentinels':sentinels,'next_draws':draws()},args.output/'expected.pt')
    else:
        expected = torch.load(args.output/'expected.pt',map_location='cpu',weights_only=True)
        interrupted = json.loads((args.output/'interrupted.json').read_text())
        baseline = json.loads((args.output/'baseline.json').read_text())
        same(interrupted['losses'],expected['losses'][:3],'prefix_losses')
        same(record['init_param_digest'],baseline['provenance']['init_param_digest'],'initialization')
        same(losses,expected['losses'][3:],'losses')
        same(sentinels,expected['sentinels'][3:],'sentinels')
        same(draws(),expected['next_draws'],'next_draws')
        same(model.state_dict(),expected['model'],'model')
        same(optimizer.state_dict(),expected['optimizer'],'optimizer')
        report['exact_restart_verified'] = True
    (args.output/(args.phase+'.json')).write_text(json.dumps(report,indent=2))
    print('RESUME_PHASE_COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':
    main()
