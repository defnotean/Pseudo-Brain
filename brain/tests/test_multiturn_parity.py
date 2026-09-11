import pytest
import torch
from irene_brain.evaluation.multiturn_parity import validate_multiturn_parity
from irene_brain.unified.parallel_depth_model import ParallelDepthModel,ParallelDepthConfig


def fixture():
    torch.manual_seed(1986)
    model=ParallelDepthModel(ParallelDepthConfig(width=32,vocab_size=128,pointer=True)).eval()
    ids=torch.randint(10,128,(1,81))
    spans=[{'start':9,'eos_position':20,'observation_end':33},
           {'start':33,'eos_position':45,'observation_end':58},
           {'start':58,'eos_position':69,'observation_end':81}]
    for span in spans:
        ids[0,span['start']]=5
        ids[0,span['eos_position']]=2
    reset=torch.zeros_like(ids)
    reset[0,0]=1
    return model,{'input_ids':ids,'prompt_tokens':ids[:,:9].clone(),'reset_mask':reset,'action_spans':spans}


def test_multiturn_validator_covers_teacher_boundaries_and_generated_continuation():
    model,encoded=fixture()
    result=validate_multiturn_parity(model,encoded,resp_id=5,eos_id=2,generation_limit=7)
    assert result['status']=='passed' and result['tokens']==81 and result['state_bytes']==4096
    assert len(result['generated_turns'])==3


def test_validator_rejects_nonfinite_outputs(monkeypatch):
    model,encoded=fixture()
    readout=model.readout
    def broken(*args,**kwargs): return readout(*args,**kwargs)*float('nan')
    monkeypatch.setattr(model,'readout',broken)
    with pytest.raises(AssertionError,match='Nonfinite'):
        validate_multiturn_parity(model,encoded,resp_id=5,eos_id=2)
