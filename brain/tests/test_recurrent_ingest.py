import pytest
import torch
from irene_brain.evaluation.recurrent_ingest import parallel_recurrent_ingest
from irene_brain.evaluation.recurrent_prefill import parallel_recurrent_prefill
from irene_brain.unified.parallel_depth_model import ParallelDepthModel,ParallelDepthConfig


def decoded(state):
    return state[:,:8].double()+state[:,8:].double()


def model_and_state(pointer=True,batch=2):
    torch.manual_seed(510)
    model=ParallelDepthModel(ParallelDepthConfig(vocab_size=128,width=32,pointer=pointer)).eval()
    warm=torch.randint(1,128,(batch,17))
    with torch.no_grad():
        _,state=parallel_recurrent_prefill(model,warm,warm)
    return model,state,warm


@pytest.mark.parametrize('length',[1,31,257])
@pytest.mark.parametrize('pointer',[False,True])
def test_nonzero_state_observation_and_continuation_match_serial(length,pointer):
    model,state,prompt=model_and_state(pointer)
    original=state.clone()
    ids=torch.randint(1,128,(2,length))
    resets=torch.zeros_like(ids)
    if length>1:
        resets[0,length//2]=1
        resets[1,length-1]=1
    expected_state=state.clone()
    with torch.no_grad():
        for index in range(length):
            expected_state*=1-resets[:,index,None,None]
            expected,expected_state=model.step(ids[:,index],expected_state,prompt)
        actual,next_state=parallel_recurrent_ingest(model,ids,state,prompt,resets)
        assert torch.equal(state,original)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
        torch.testing.assert_close(decoded(next_state),decoded(expected_state),atol=1e-12,rtol=0)
        assert next_state.numel()*next_state.element_size()==2*4096
        assert next_state.dtype==torch.float32 and not next_state.requires_grad
        for token in torch.randint(1,128,(7,2)):
            expected,expected_state=model.step(token,expected_state,prompt)
            actual,next_state=model.step(token,next_state,prompt)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)


def test_chunked_ingestion_uses_only_new_tokens_and_retains_state():
    model,state,prompt=model_and_state(batch=1)
    serial_state=state.clone()
    ids=torch.randint(1,128,(1,109))
    offset=0
    with torch.no_grad():
        for size in (1,3,13,31,61):
            chunk=ids[:,offset:offset+size]
            for index in range(size):
                expected,serial_state=model.step(chunk[:,index],serial_state,prompt)
            actual,state=parallel_recurrent_ingest(model,chunk,state,prompt)
            torch.testing.assert_close(actual,expected,atol=1e-6,rtol=0)
            torch.testing.assert_close(decoded(state),decoded(serial_state),atol=1e-12,rtol=0)
            offset+=size
    assert offset==ids.shape[1]


@pytest.mark.parametrize('bad',['shape','dtype','nonfinite','empty','reset'])
def test_invalid_continuation_inputs_are_rejected(bad):
    model,state,prompt=model_and_state(batch=1)
    ids=torch.ones(1,3,dtype=torch.long)
    resets=None
    if bad=='shape': state=state[:,:8]
    elif bad=='dtype': state=state.double()
    elif bad=='nonfinite': state[0,0,0]=float('nan')
    elif bad=='empty': ids=ids[:,:0]
    elif bad=='reset': resets=torch.full_like(ids,2)
    with pytest.raises(ValueError):
        parallel_recurrent_ingest(model,ids,state,prompt,resets)
