import pytest
import torch
from irene_brain.evaluation.independent_pilot import greedy_decode
from irene_brain.unified.parallel_depth_model import ParallelDepthModel, ParallelDepthConfig


@pytest.mark.parametrize('length',[31,257])
@pytest.mark.parametrize('pointer',[False,True])
def test_opt_in_preserves_tokens_stopping_and_fresh_task_state(length,pointer):
    torch.manual_seed(833)
    model = ParallelDepthModel(ParallelDepthConfig(vocab_size=128,width=32,pointer=pointer)).eval()
    context = torch.randint(1,128,(1,length))
    prompt = context[:,:-1].clone()
    original_prompt = prompt.clone()
    kwargs = dict(recurrent=True,eos_id=-1,limit=16)
    serial = greedy_decode(model,context,prompt,**kwargs)
    parallel = greedy_decode(model,context,prompt,parallel_prefill=True,**kwargs)
    assert parallel == serial
    assert parallel['recurrent_state_bytes']==4096 and parallel['stop_reason']=='token_limit'
    assert torch.equal(prompt,original_prompt)
    greedy_decode(model,context.flip(1),prompt.flip(1),parallel_prefill=True,**kwargs)
    assert greedy_decode(model,context,prompt,parallel_prefill=True,**kwargs)==serial
    kwargs['eos_id'] = serial['token_ids'][0]
    assert greedy_decode(model,context,prompt,parallel_prefill=True,**kwargs)=={
        'token_ids':[],'stop_reason':'eos','recurrent_state_bytes':4096}


def test_parallel_prefill_rejects_transformer_route_before_model_execution():
    with pytest.raises(ValueError,match='only for the recurrent'):
        greedy_decode(None,torch.ones(1,2,dtype=torch.long),torch.ones(1,1,dtype=torch.long),
                      recurrent=False,eos_id=0,parallel_prefill=True)
