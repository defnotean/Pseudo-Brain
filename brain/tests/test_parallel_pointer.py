"""Behavioral gates for the independent-time pointer candidate."""
import pytest
import torch
from irene_brain.unified.unified_model import make_unified_model


def candidate():
    torch.manual_seed(91)
    return make_unified_model(tier="tier1", vocab_size=1000,
        use_pointer_copy=True, use_token_skip=True, use_gated_token_skip=True,
        compensated_state=True, pointer_mode="parallel_predecessor").eval()


@pytest.mark.parametrize("length", [31, 257])
def test_parallel_streaming_parity_with_resets_and_masks(length):
    model = candidate()
    x = torch.randint(1, 1000, (2, length))
    prompt = torch.tensor([[17, 23, 17, 29, 0], [0, 0, 0, 0, 0]])
    x[:, 5] = 17
    mask = torch.ones_like(x)
    mask[:, 8:13] = 0
    resets = torch.zeros_like(x)
    resets[:, 18] = 1
    with torch.no_grad():
        par = model(x, prompt_tokens=prompt, pointer_mask=mask, reset_mask=resets)
        seq = model(x, parallel=False, prompt_tokens=prompt, pointer_mask=mask, reset_mask=resets)
    assert (par['logits'] - seq['logits']).abs().max() < 1e-6
    assert torch.count_nonzero(par['ptr_attn'][1]) == 0
    assert torch.count_nonzero(par['ptr_attn'][:, 8:13]) == 0
    state = model.init_state(2, torch.device('cpu'))
    _, state = model.step(model.encode_sensory(token_ids=x[:, 0]), state,
                          token_id=x[:, 0], prompt_tokens=prompt)
    assert state.ptr_prev_alpha is None and state.ptr_prev_gamma is None
    assert state.hierarchical_state.fast_state_bytes() == 4096


def test_copy_continuation_and_duplicate_ambiguity():
    model = candidate()
    prompt = torch.tensor([[11, 41, 12, 51, 0]])
    scores = torch.zeros(1, 2, 5, dtype=torch.float64)
    attn = model._parallel_pointer_attention(scores, torch.tensor([[11, 12]]), prompt)
    assert attn.argmax(-1).tolist() == [[1, 3]]
    assert (attn.max(-1).values > .999).all()
    # A repeated predecessor cannot be resolved from token identity alone.
    # Both successors remain eligible and the content score disambiguates.
    prompt = torch.tensor([[11, 41, 11, 51, 0]])
    scores[0, 0, 3] = 3
    attn = model._parallel_pointer_attention(scores, torch.tensor([[11, 11]]), prompt)
    assert attn[0, 0, 3] > attn[0, 0, 1]
    assert attn[0, 1, 3] == attn[0, 1, 1]
    assert torch.count_nonzero(attn[..., -1]) == 0


def test_no_future_token_influence_and_sparse_readout_gradients():
    model = candidate()
    x = torch.randint(1, 1000, (2, 31))
    prompt = x[:, :9].clone()
    changed = x.clone()
    changed[:, 17:] = 91
    with torch.no_grad():
        first = model(x, prompt_tokens=prompt)['logits']
        second = model(changed, prompt_tokens=prompt)['logits']
    assert torch.equal(first[:, :17], second[:, :17])
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, 3:15] = True
    sparse = model(x, prompt_tokens=prompt, language_mask=mask)
    assert torch.allclose(first[mask], sparse['logits'], atol=1e-10, rtol=0)
    loss = sparse['logits'].square().mean()
    loss.backward()
    for parameter in (model.ptr_q.weight, model.ptr_k.weight,
                      model.ptr_gate.weight, model.ptr_seq_boost):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_mode_validation_and_parameter_budget():
    model = candidate()
    baseline = make_unified_model(tier='tier1', vocab_size=1000,
        use_pointer_copy=True, use_token_skip=True, use_gated_token_skip=True)
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in baseline.parameters())
    with pytest.raises(ValueError, match='Unknown pointer mode'):
        make_unified_model(tier='tier1', vocab_size=1000, pointer_mode='unknown')


@pytest.mark.parametrize('mode', ['sequential', 'parallel_predecessor'])
def test_checkpoint_restores_pointer_mode(tmp_path, mode):
    from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
    model = candidate()
    checkpoint = dict(tier='tier1', vocab_size=1000, use_pointer_copy=True,
        use_token_skip=True, use_gated_token_skip=True, compensated_state=True,
        model_state_dict=model.state_dict())
    if mode != 'sequential':
        checkpoint['pointer_mode'] = mode
    path = tmp_path / 'candidate.pt'
    torch.save(checkpoint, path)
    agent = RecurrentSoftwareAgent(checkpoint_path=path, device=torch.device('cpu'))
    assert agent.model.pointer_mode == mode
    assert agent.model.compensated_state
