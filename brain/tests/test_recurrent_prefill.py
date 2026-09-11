import pytest
import torch

from irene_brain.evaluation.recurrent_prefill import parallel_recurrent_prefill
from irene_brain.unified.parallel_depth_model import ParallelDepthConfig, ParallelDepthModel


@pytest.mark.parametrize('length',[1,31,257])
@pytest.mark.parametrize('pointer',[False,True])
def test_prefill_and_continued_generation_match_streaming(length,pointer):
    torch.manual_seed(411+length)
    model = ParallelDepthModel(ParallelDepthConfig(vocab_size=128,width=32,pointer=pointer)).eval()
    ids = torch.randint(1,128,(2,length))
    prompt = ids[:,:min(length,11)].clone() if pointer else None
    resets = torch.zeros_like(ids)
    if length > 1:
        resets[0,length//2] = 1
        resets[1,length//3] = 1
    with torch.no_grad():
        state = model.init_state(2)
        for index in range(length):
            state *= 1-resets[:,index,None,None]
            expected,state = model.step(ids[:,index],state,prompt)
        actual,parallel_state = parallel_recurrent_prefill(model,ids,prompt,resets)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
        torch.testing.assert_close(parallel_state[:,:8].double()+parallel_state[:,8:].double(),
                                   state[:,:8].double()+state[:,8:].double(),atol=1e-12,rtol=0)
        assert parallel_state.shape == (2,16,64) and parallel_state.dtype == torch.float32
        assert parallel_state.numel()*parallel_state.element_size() == 2*4096
        assert not actual.requires_grad and not parallel_state.requires_grad
        # The returned state must work for ordinary subsequent token generation.
        continuation = torch.randint(1,128,(2,7))
        for index in range(7):
            expected,state = model.step(continuation[:,index],state,prompt)
            actual,parallel_state = model.step(continuation[:,index],parallel_state,prompt)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)


def test_prefill_rejects_empty_context():
    model = ParallelDepthModel(ParallelDepthConfig(vocab_size=128,width=32)).eval()
    with pytest.raises(ValueError,match='nonempty'):
        parallel_recurrent_prefill(model,torch.empty(1,0,dtype=torch.long))


def test_prefill_rejects_non_fp64_model():
    model = ParallelDepthModel(ParallelDepthConfig(vocab_size=128,width=32)).float().eval()
    with pytest.raises(ValueError,match='FP64'):
        parallel_recurrent_prefill(model,torch.ones(1,1,dtype=torch.long))
