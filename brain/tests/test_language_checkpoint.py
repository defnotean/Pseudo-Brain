import copy
import random

import pytest
import torch

from irene_brain.runtime.policy import ResourcePolicy
from irene_brain.training.checkpoint import TrainerCursor
from irene_brain.training.language_checkpoint import save_language_boundary, restore_language_boundary
from irene_brain.unified.parallel_depth_model import ParallelDepthConfig, ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer


IDENTITY = dict(config_sha256='a'*64, data_sha256='b'*64, code_sha256='c'*64)


def build(cls):
    random.seed(991)
    torch.manual_seed(991)
    model = cls(ParallelDepthConfig(vocab_size=128, width=64))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9,.95))
    return model, optimizer


def advance(model, optimizer, first, last):
    losses = []
    for step in range(first, last):
        length = random.choice((11, 17, 31))
        ids = torch.randint(1,128,(1,length))
        labels = ids.roll(-1,1)
        labels[:,:length//2] = -100
        labels[:,-1] = -100
        for group in optimizer.param_groups:
            group['lr'] = 3e-4/(step+1)
        loss = model.language_loss(ids, labels, prompt_tokens=ids[:,:length//2])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return losses


def same(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            same(left[key], right[key])
    elif isinstance(left, (list,tuple)):
        assert len(left) == len(right)
        for a,b in zip(left,right):
            same(a,b)
    else:
        assert left == right


@pytest.mark.parametrize('cls',[ParallelDepthModel, ComparisonTransformer])
def test_interruption_preserves_future_losses_weights_moments_and_rng(tmp_path, cls):
    model, optimizer = build(cls)
    baseline = advance(model,optimizer,0,7)
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_random = (random.random(), torch.rand(5))
    model, optimizer = build(cls)
    prefix = advance(model,optimizer,0,3)
    path = tmp_path/'boundary.pt'
    digest = save_language_boundary(path, model=model, optimizer=optimizer,
        cursor=TrainerCursor(epoch=0,next_batch=3,optimizer_step=3),
        policy=ResourcePolicy(allow_artifact_write=True), **IDENTITY)
    del model, optimizer
    # Reconstruction and unrelated work consume RNG, as in a new interpreter.
    model, optimizer = build(cls)
    random.seed(5)
    torch.manual_seed(5)
    torch.rand(100)
    cursor = restore_language_boundary(path, model=model, optimizer=optimizer,
        checkpoint_sha256=digest, **IDENTITY)
    assert cursor == TrainerCursor(epoch=0,next_batch=3,optimizer_step=3)
    suffix = advance(model,optimizer,cursor.next_batch,7)
    assert prefix+suffix == baseline
    same(model.state_dict(), expected_model)
    same(optimizer.state_dict(), expected_optimizer)
    same((random.random(),torch.rand(5)), expected_random)


def test_boundary_rejects_pending_gradients_and_never_overwrites(tmp_path):
    model, optimizer = build(ParallelDepthModel)
    path = tmp_path/'boundary.pt'
    kwargs = dict(model=model,optimizer=optimizer,cursor=TrainerCursor(),
                  policy=ResourcePolicy(allow_artifact_write=True),**IDENTITY)
    next(model.parameters()).sum().backward()
    with pytest.raises(ValueError,match='cleared gradients'):
        save_language_boundary(path,**kwargs)
    assert not path.exists()
    optimizer.zero_grad(set_to_none=True)
    save_language_boundary(path,**kwargs)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_language_boundary(path,**kwargs)
    assert path.read_bytes() == original


@pytest.mark.parametrize('mismatch',['data','source','optimizer_order'])
def test_incompatible_resume_rejected_before_loading_model(tmp_path,mismatch):
    model, optimizer = build(ParallelDepthModel)
    advance(model,optimizer,0,2)
    path = tmp_path/'boundary.pt'
    digest = save_language_boundary(path,model=model,optimizer=optimizer,
        cursor=TrainerCursor(next_batch=2,optimizer_step=2),
        policy=ResourcePolicy(allow_artifact_write=True),**IDENTITY)
    model, optimizer = build(ParallelDepthModel)
    before = copy.deepcopy(model.state_dict())
    identity = dict(IDENTITY)
    if mismatch == 'optimizer_order':
        optimizer = torch.optim.AdamW(list(reversed(list(model.parameters()))),lr=3e-4)
    else:
        identity['data_sha256' if mismatch=='data' else 'code_sha256'] = 'd'*64
    with pytest.raises(ValueError,match='identity'):
        restore_language_boundary(path,model=model,optimizer=optimizer,
            checkpoint_sha256=digest,**identity)
    same(model.state_dict(),before)
