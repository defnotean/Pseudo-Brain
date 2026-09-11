import pytest
import torch

from irene_brain.agent.parallel_depth_session import ParallelDepthSession
from irene_brain.unified.parallel_depth_model import ParallelDepthModel, ParallelDepthConfig


def setup(pointer=True):
    torch.manual_seed(1907)
    model = ParallelDepthModel(ParallelDepthConfig(width=32, vocab_size=128, pointer=pointer)).eval()
    prompt = torch.randint(1,128,(1,31))
    return model, prompt


@torch.no_grad()
def serial_ingest(model, ids, state, prompt):
    for token in ids.unbind(1):
        logits, state = model.step(token, state, prompt)
    return logits, state


def assert_state(actual, expected):
    assert actual.shape == (1,16,64) and actual.dtype == torch.float32
    assert actual.numel()*actual.element_size() == 4096 and not actual.requires_grad
    decode = lambda s: s[:,:8].double()+s[:,8:].double()
    torch.testing.assert_close(decode(actual), decode(expected), atol=1e-12, rtol=0)


@pytest.mark.parametrize('pointer', [False,True])
def test_multiple_actions_observations_match_serial_without_history_replay(pointer):
    model, prompt = setup(pointer)
    session = ParallelDepthSession(model)
    session.start(prompt)
    _, reference = serial_ingest(model, prompt, model.init_state(1), prompt)
    for length in (31,257):
        observation = torch.randint(1,128,(1,length))
        expected, reference = serial_ingest(model, observation, reference, prompt)
        actual = session.ingest(observation)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0)
        prefix = torch.tensor([[5]])
        logits, reference = serial_ingest(model, prefix, reference, prompt)
        emitted = []
        stop = 'token_limit'
        with torch.no_grad():
            for _ in range(12):
                token = logits.argmax(dim=-1)
                logits, reference = model.step(token, reference, prompt)
                if token.item() == 2:
                    stop = 'eos'
                    break
                emitted.append(token.item())
        result = session.generate(prefix, eos_id=2, max_new_tokens=12)
        assert result.token_ids == tuple(emitted) and result.stop_reason == stop
        assert result.state_bytes == 4096
        assert_state(session.state, reference)
    assert set(vars(session)) == {'model','_state','_prompt'}
    assert torch.equal(session._prompt, prompt)


def test_emitted_eos_enters_state_before_next_observation():
    model, prompt = setup()
    session = ParallelDepthSession(model)
    session.start(prompt)
    prefix = torch.tensor([[5]])
    logits, reference = serial_ingest(model, prefix, session.state, prompt)
    eos = int(logits.argmax().item())
    with torch.no_grad():
        _, reference = model.step(torch.tensor([eos]), reference, prompt)
    result = session.generate(prefix, eos_id=eos, max_new_tokens=4)
    assert result.token_ids == () and result.stop_reason == 'eos'
    assert_state(session.state, reference)
    observation = torch.tensor([[17,23,29]])
    expected, reference = serial_ingest(model, observation, reference, prompt)
    torch.testing.assert_close(session.ingest(observation), expected, atol=1e-6, rtol=0)
    assert_state(session.state, reference)


def test_token_limit_consumes_last_token_without_inventing_eos():
    model, prompt = setup()
    session = ParallelDepthSession(model)
    session.start(prompt)
    prefix = torch.tensor([[5]])
    logits, reference = serial_ingest(model, prefix, session.state, prompt)
    selected = logits.argmax(dim=-1)
    eos = (int(selected.item()) + 1) % model.config.vocab_size
    with torch.no_grad():
        _, reference = model.step(selected, reference, prompt)
    result = session.generate(prefix, eos_id=eos, max_new_tokens=1)
    assert result.token_ids == (int(selected.item()),) and result.stop_reason == 'token_limit'
    assert_state(session.state, reference)


def test_fresh_start_and_external_mutation_do_not_leak_between_tasks():
    model, prompt = setup()
    session = ParallelDepthSession(model)
    session.start(prompt)
    original = session.state
    frozen_prompt = prompt.clone()
    prompt.fill_(0)
    assert torch.equal(session._prompt, frozen_prompt)
    exported = session.state
    exported.zero_()
    assert torch.equal(session.state, original)
    session.generate(torch.tensor([[5]]), eos_id=2, max_new_tokens=7)
    session.ingest(torch.tensor([[19,31,77]]))
    session.start(frozen_prompt)
    assert torch.equal(session.state, original)


def test_invalid_arguments_fail_before_changing_state():
    model, prompt = setup()
    session = ParallelDepthSession(model)
    with pytest.raises(RuntimeError):
        session.ingest(prompt)
    session.start(prompt)
    before = session.state
    for bad in (torch.empty(1,0,dtype=torch.long), torch.tensor([[128]]), prompt.float(), prompt.repeat(2,1)):
        with pytest.raises(ValueError):
            session.ingest(bad)
        assert torch.equal(session.state,before)
    for kwargs in ({'eos_id':2,'max_new_tokens':0}, {'eos_id':128,'max_new_tokens':4}):
        with pytest.raises(ValueError):
            session.generate(torch.tensor([[5]]), **kwargs)
        assert torch.equal(session.state,before)
