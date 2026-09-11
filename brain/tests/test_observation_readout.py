import torch
from irene_brain.unified.unified_model import make_unified_model
from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent


def test_omitting_unused_readout_preserves_all_state_and_next_prediction():
    torch.manual_seed(72)
    model = make_unified_model(tier='tier1', vocab_size=1000, use_pointer_copy=True,
        use_token_skip=True, use_gated_token_skip=True, compensated_state=True).eval()
    tokens = torch.tensor([[13, 19, 29, 41]])
    full = model.init_state(1, torch.device('cpu'))
    cheap = model.init_state(1, torch.device('cpu'))
    with torch.no_grad():
        for i in range(tokens.shape[1]):
            token = tokens[:, i]
            sensory = model.encode_sensory(token_ids=token)
            a, full = model.step(sensory, full, token_id=token)
            b, cheap = model.step(sensory, cheap, token_id=token, read_language=i == 3)
            for key, value in vars(full.hierarchical_state).items():
                other = getattr(cheap.hierarchical_state, key)
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, other), key
            assert torch.equal(full.last_action, cheap.last_action)
            assert torch.equal(full.active_thread, cheap.active_thread)
            assert cheap.ptr_prev_alpha is None and cheap.ptr_prev_gamma is None
            if i < 3:
                assert b['logits'] is None
            else:
                assert torch.equal(a['logits'], b['logits'])


def test_agent_ingestion_calls_language_head_only_once():
    torch.manual_seed(11)
    model = make_unified_model(tier='tier1', vocab_size=32000)
    agent = RecurrentSoftwareAgent(model=model)
    calls = []
    hook = model.slot_head.register_forward_hook(lambda *args: calls.append(1))
    try:
        result = agent._ingest_text_into_slot('An observation with several different input tokens.')
    finally:
        hook.remove()
    assert len(calls) == 1
    assert result is not None and result.shape == (32000,)


def test_active_legacy_pointer_state_is_not_skipped():
    model = make_unified_model(tier='tier1', vocab_size=1000, use_pointer_copy=True).eval()
    state = model.init_state(1, torch.device('cpu'))
    token = torch.tensor([7])
    with torch.no_grad():
        output, state = model.step(model.encode_sensory(token_ids=token), state,
            token_id=token, prompt_tokens=torch.tensor([[7, 11]]), read_language=False)
    assert output['logits'] is not None
    assert state.ptr_prev_alpha is not None
