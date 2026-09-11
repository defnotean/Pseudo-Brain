import pytest
import torch
from torch.nn import functional as F
from irene_brain.training.action_trajectory_loss import action_trajectory_loss
from irene_brain.unified.parallel_depth_model import ParallelDepthModel,ParallelDepthConfig
from irene_brain.unified.comparison_transformer import ComparisonTransformer


def example():
    ids=torch.randint(10,128,(1,81))
    labels=torch.full_like(ids,-100)
    # Three action spans; the first is an injected fault and has no targets.
    for start,eos in ((9,20),(33,45),(58,69)):
        ids[0,start]=5
        ids[0,eos]=2
        if start!=9: labels[0,start:eos]=ids[0,start+1:eos+1]
    reset=torch.zeros_like(ids)
    reset[0,0]=1
    return {'input_ids':ids,'labels':labels,'prompt_tokens':ids[:,:9].clone(),'reset_mask':reset}


@pytest.mark.parametrize('kind',['recurrent','transformer'])
def test_action_only_loss_and_every_parameter_gradient_match_full_reference(kind):
    torch.manual_seed(1987)
    cls=ParallelDepthModel if kind=='recurrent' else ComparisonTransformer
    model=cls(ParallelDepthConfig(width=64,vocab_size=128,pointer=True)).train()
    encoded=example()
    full=model(encoded['input_ids'],reset_mask=encoded['reset_mask'],prompt_tokens=encoded['prompt_tokens'])
    active=encoded['labels']!=-100
    expected=F.cross_entropy(full[active],encoded['labels'][active])
    expected.backward()
    gradients={name:parameter.grad.clone() for name,parameter in model.named_parameters() if parameter.grad is not None}
    assert any(name.startswith('layers.') and value.abs().max()>0 for name,value in gradients.items())
    assert any(name.startswith('ptr_') and value.abs().max()>0 for name,value in gradients.items())
    model.zero_grad(set_to_none=True)
    actual=action_trajectory_loss(model,encoded,chunk_size=7)
    actual.backward()
    atol,rtol=(1e-10,1e-8) if kind=='recurrent' else (1e-6,1e-4)
    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
    for name,parameter in model.named_parameters():
        assert (parameter.grad is not None)==(name in gradients)
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad,gradients[name],atol=atol,rtol=rtol)


def test_interior_resets_and_changed_pointer_source_are_rejected():
    torch.manual_seed(1987)
    model=ParallelDepthModel(ParallelDepthConfig(width=64,vocab_size=128))
    encoded=example()
    encoded['reset_mask'][0,33]=1
    with pytest.raises(ValueError,match='Only task start'):
        action_trajectory_loss(model,encoded)
    encoded['reset_mask'][0,33]=0
    encoded['prompt_tokens'][0,0]+=1
    with pytest.raises(ValueError,match='Pointer source'):
        action_trajectory_loss(model,encoded)
