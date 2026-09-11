"""Behavioral checks for multi-turn boundaries, copying, supervision and proof."""
from types import SimpleNamespace

import pytest
import torch

from irene_brain.agent.procedural_evaluator import verified_repair_evidence
from irene_brain.agent.procedural_training_generator import ProceduralTrainingGenerator
from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
from irene_brain.agent.software_environment import NeuralSoftwareEnvironment
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.unified.unified_model import make_unified_model
from train_and_eval_target_relevance import build_pomdp_batches_calibrated


@pytest.mark.parametrize("masked", [False, True])
def test_pointer_and_skip_streaming_parity(masked):
    torch.manual_seed(71)
    model = make_unified_model(tier="tier1", vocab_size=1000, use_token_skip=True,
                               use_gated_token_skip=True, use_pointer_copy=True).eval()
    tokens = torch.randint(1, 1000, (2, 24))
    prompt = tokens[:, :7].clone()
    prompt[1, -2:] = 0
    mask = torch.ones_like(tokens, dtype=torch.float32) if masked else None
    if mask is not None:
        mask[:, :5] = 0
        mask[:, 10:15] = 0
    with torch.no_grad():
        par = model(tokens, parallel=True, prompt_tokens=prompt, pointer_mask=mask)
        seq = model(tokens, parallel=False, prompt_tokens=prompt, pointer_mask=mask)
    assert (par["logits"] - seq["logits"]).abs().max().item() < 1e-6
    assert model.init_state(1, torch.device("cpu")).hierarchical_state.fast_state_bytes() == 4096


def test_training_targets_are_causal_and_faults_are_context():
    tokenizer = BpeSemanticTokenizer()
    tasks = ProceduralTrainingGenerator().generate_training_tasks(count=6)
    batches = build_pomdp_batches_calibrated(tasks, tokenizer, batch_size=6)
    b = batches[0]
    for row in range(6):
        x = b["X"][row]
        resp = (x == tokenizer.resp_id).nonzero().flatten().tolist()
        eos = (x == tokenizer.eos_id).nonzero().flatten().tolist()
        for r, e in zip(resp, eos):
            assert b["M"][row, e] == 0  # never teach the observation after EOS
            if r == resp[0] and row in (1, 2, 3, 4):
                assert b["M"][row, r:e].sum() == 0
            else:
                assert b["M"][row, e - 1] == 1  # do teach EOS itself
        valid = b["ptr_targets"][row] >= 0
        assert valid.any()
        assert torch.equal(b["Y"][row, valid], b["prompt_tokens"][row, b["ptr_targets"][row, valid]])
        assert torch.all(b["target_masks"][row] <= b["M"][row])


def test_observations_do_not_change_copy_source(tmp_path):
    torch.manual_seed(12)
    model = make_unified_model(tier="tier1", vocab_size=32000)
    agent = RecurrentSoftwareAgent(model=model)
    env = NeuralSoftwareEnvironment(tmp_path)
    agent.execute_pomdp_episode("Implement `f` in `a.py`", env,
        action_plan=["ACTION: WRITE_FILE a.py\ndef f()\n return None"],
        target_function="f", target_module="a.py")
    prompt = list(agent.active_prompt_tokens)
    agent._ingest_text_into_slot("[OBSERVATION: SyntaxError repeated]", slot_id=1)
    assert agent.active_prompt_tokens == prompt
    assert "SyntaxError" not in agent.tokenizer.decode(prompt)
    agent.reset()
    assert not agent.active_prompt_tokens
    assert agent.cognitive_state.ptr_prev_alpha is None


def test_generated_eos_is_consumed_once(monkeypatch):
    model = make_unified_model(tier="tier1", vocab_size=32000)
    agent = RecurrentSoftwareAgent(model=model)
    seen = []
    original = model.step
    def step(*args, **kwargs):
        seen.append(int(kwargs["token_id"].item()))
        out, state = original(*args, **kwargs)
        out["logits"].fill_(-100)
        out["logits"][:, agent.tokenizer.eos_id] = 100
        return out, state
    monkeypatch.setattr(model, "step", step)
    agent.generate_action_autoregressive()
    assert seen == [agent.tokenizer.resp_id, agent.tokenizer.eos_id]


@pytest.mark.parametrize("success,source,validator,expected", [
    (False, "autonomous", True, False), (True, "scripted", True, False),
    (True, "autonomous", False, False), (True, "autonomous", True, True),
])
def test_repair_proof_requires_autonomous_oracle_pass(success, source, validator, expected):
    feedback = [
        dict(cycle=1, action_type="FINISH", success=False, transition="[PHASE: REPAIR_LOGIC]"),
        dict(cycle=2, action_type="WRITE_FILE", success=True),
        dict(cycle=3, action_type="FINISH", success=success),
    ]
    res = SimpleNamespace(success=success, policy_source=source, action_feedback=feedback)
    assert bool(verified_repair_evidence(res, validator)) is expected


def test_actual_champion_configuration_budget():
    model = make_unified_model(tier="tier2", vocab_size=32000, use_token_skip=True,
                              use_gated_token_skip=True, use_pointer_copy=True)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) < 36_000_000
    assert model.init_state(1, torch.device("cpu")).hierarchical_state.fast_state_bytes() == 4096


@pytest.mark.parametrize("length", [31, 257, 513])
def test_compensated_checkpoint_long_batched_parity(length):
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    agent = RecurrentSoftwareAgent(checkpoint_path=root / "checkpoints/pb_pomdp_champion.pt")
    model = agent.model
    model.enable_compensated_state()
    torch.manual_seed(123 + length)
    tokens = torch.randint(1, 32000, (2, length))
    prompts = tokens[:, :11].clone()
    prompts[1, -3:] = 0
    mask = torch.ones_like(tokens, dtype=torch.float32)
    mask[:, :11] = 0
    mask[:, 15:23] = 0
    with torch.no_grad():
        parallel = model(tokens, prompt_tokens=prompts, pointer_mask=mask, parallel=True)
        streaming = model(tokens, prompt_tokens=prompts, pointer_mask=mask, parallel=False)
    assert (parallel["logits"] - streaming["logits"]).abs().max().item() < 1e-6
    state = model.init_state(1, torch.device("cpu"))
    assert state.hierarchical_state.working_thoughts.dtype == torch.float32
    assert state.hierarchical_state.working_thoughts.numel() * 4 == 4096
    assert model.logical_slots == 8
    assert torch.count_nonzero(state.hierarchical_state.working_thoughts[:, 8:]) == 0


def test_compensated_mode_rejects_residual_slot_addressing():
    model = make_unified_model(tier="tier1", vocab_size=1000, compensated_state=True)
    state = model.init_state(1, torch.device("cpu"))
    sensory = model.encode_sensory(token_ids=torch.tensor([5]))
    with pytest.raises(ValueError, match="residual slots"):
        model.step(sensory, state, thread_id=torch.tensor([8]))
    with pytest.raises(ValueError, match="routing"):
        model.step(sensory, state, allow_routing=True)


def test_training_bank_is_repeatable_despite_temporary_workspaces():
    tokenizer = BpeSemanticTokenizer()
    tasks = ProceduralTrainingGenerator().generate_training_tasks(count=6)
    a = build_pomdp_batches_calibrated(tasks, tokenizer, batch_size=6)
    b = build_pomdp_batches_calibrated(tasks, tokenizer, batch_size=6)
    for key in a[0]:
        assert torch.equal(a[0][key], b[0][key]), key


def test_selective_readout_preserves_supervised_logits_and_gradients():
    torch.manual_seed(191)
    model = make_unified_model(tier="tier0", vocab_size=1000, use_token_skip=True,
                               use_gated_token_skip=True, use_pointer_copy=True,
                               compensated_state=True)
    tokens = torch.randint(1, 1000, (2, 24))
    prompt = tokens[:, :7]
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    mask[:, 8:12] = True
    mask[:, 18:22] = True
    pointer_mask = torch.ones_like(tokens, dtype=torch.float32)
    full = model(tokens, prompt_tokens=prompt, pointer_mask=pointer_mask)
    reference = full["logits"][mask].detach()
    full["logits"][mask].square().mean().backward()
    gradients = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    selective = model(tokens, prompt_tokens=prompt, pointer_mask=pointer_mask, language_mask=mask)
    assert selective["logits"].shape == (int(mask.sum()), 1000)
    torch.testing.assert_close(selective["logits"], reference, atol=1e-10, rtol=1e-10)
    selective["logits"].square().mean().backward()
    for name, p in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], atol=1e-10, rtol=1e-9)
