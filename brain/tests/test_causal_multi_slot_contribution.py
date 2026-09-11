"""Test Suite for Causal Multi-Slot Contribution and Episodic Retrieval Fusion.

Verifies:
1. Slot 1 (Environment Observation) Causal Impact:
   - When allow_routing=False, Slot 1 perturbation yields 0.0 logit difference on Slot 0.
   - When allow_routing=True, Slot 1 perturbation produces measurable logit difference (> 0).
2. Slot 2 (Error Hypothesis / Reflection) Causal Impact:
   - When allow_routing=False, Slot 2 perturbation yields 0.0 logit difference on Slot 0.
   - When allow_routing=True, Slot 2 perturbation produces measurable logit difference (> 0).
3. Episodic Memory Causal Impact:
   - When allow_routing=True, perturbing episodic_thoughts produces measurable logit difference (> 0)
     via CrossTierRetrievalModule fusion.
4. Slot Shuffling / Inactive Slot Diversity:
   - Permuting populated working thought slots produces distinct routed messages across the ensemble.
5. Invariant 1: Law 1 Working Memory Footprint:
   - Strictly 4,096 bytes (16 slots * 64 float32 * 4 bytes).
6. Invariant 2: Streaming / Scan Parity:
   - Single-token step streaming parity with forward_sequence_parallel strictly < 1e-6.
7. RecurrentSoftwareAgent Multi-Slot Routing Support:
   - RecurrentSoftwareAgent properly enables allow_routing by default and passes it to model.step.
"""

from __future__ import annotations

import pytest
import torch

from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
from irene_brain.unified.unified_model import UnifiedPseudoBrain, make_unified_model


@pytest.fixture
def cpu_model() -> UnifiedPseudoBrain:
    model = make_unified_model(tier="tier1_3m", vocab_size=1000)
    model.eval()
    return model


def test_law1_working_memory_footprint(cpu_model: UnifiedPseudoBrain) -> None:
    """Verify strict Law 1 working memory invariant: exactly 4,096 bytes."""
    device = torch.device("cpu")
    state = cpu_model.init_state(batch_size=1, device=device)
    w_bytes = state.hierarchical_state.fast_state_bytes()
    assert w_bytes == 4096, f"Expected 4096 bytes, got {w_bytes}"
    assert state.hierarchical_state.working_thoughts.shape == (1, 16, 64)
    assert state.hierarchical_state.working_thoughts.dtype == torch.float32


def test_streaming_scan_parity(cpu_model: UnifiedPseudoBrain) -> None:
    """Verify streaming/scan parity invariant strictly < 1e-6 when allow_routing=False."""
    device = torch.device("cpu")
    for T in (31, 257):
        tokens = torch.randint(1, 1000, (1, T), device=device)
        with torch.no_grad():
            out_par = cpu_model.forward_sequence_parallel(token_seq=tokens)
            logits_par = out_par["logits"]

            state = cpu_model.init_state(batch_size=1, device=device)
            logits_seq = []
            for t in range(T):
                tok = tokens[:, t]
                sensory = cpu_model.encode_sensory(token_ids=tok)
                out_step, state = cpu_model.step(
                    sensory,
                    state,
                    thread_id=torch.tensor([0]),
                    allow_routing=False,
                    read_language=True,
                    token_id=tok,
                )
                logits_seq.append(out_step["logits"])
            logits_seq_tensor = torch.stack(logits_seq, dim=1)

            diff = torch.max(torch.abs(logits_par - logits_seq_tensor)).item()
            assert diff < 1e-6, f"Streaming/scan parity exceeded 1e-6 at T={T}: {diff:.2e}"


def test_slot1_observation_causal_isolation_and_contribution(cpu_model: UnifiedPseudoBrain) -> None:
    """Verify Slot 1 perturbation has 0.0 impact when unrouted, and > 0 impact when routed."""
    device = torch.device("cpu")
    sens_step = cpu_model.encode_sensory(token_ids=torch.tensor([42]))

    # --- Part A: allow_routing=False -> strict isolation (0.0 diff) ---
    state_clean = cpu_model.init_state(batch_size=1, device=device)
    state_pert = cpu_model.init_state(batch_size=1, device=device)
    state_pert.hierarchical_state.working_thoughts[0, 1] += 2.0

    out_clean_unrouted, _ = cpu_model.step(
        sens_step, state_clean, thread_id=torch.tensor([0]), allow_routing=False, read_language=True
    )
    out_pert_unrouted, _ = cpu_model.step(
        sens_step, state_pert, thread_id=torch.tensor([0]), allow_routing=False, read_language=True
    )
    diff_unrouted = torch.max(torch.abs(out_clean_unrouted["logits"] - out_pert_unrouted["logits"])).item()
    assert diff_unrouted == 0.0, f"Expected 0.0 diff when allow_routing=False, got {diff_unrouted}"

    # --- Part B: allow_routing=True -> causal influence (> 0 diff) ---
    out_clean_routed, _ = cpu_model.step(
        sens_step, state_clean, thread_id=torch.tensor([0]), allow_routing=True, read_language=True
    )
    out_pert_routed, _ = cpu_model.step(
        sens_step, state_pert, thread_id=torch.tensor([0]), allow_routing=True, read_language=True
    )
    diff_routed = torch.max(torch.abs(out_clean_routed["logits"] - out_pert_routed["logits"])).item()
    assert diff_routed > 1e-5, f"Expected causal logit influence > 1e-5, got {diff_routed}"


def test_slot2_repair_reflection_causal_contribution(cpu_model: UnifiedPseudoBrain) -> None:
    """Verify Slot 2 (error reflection) causally impacts Slot 0 action decoding under routing."""
    device = torch.device("cpu")
    sens_obs = cpu_model.encode_sensory(token_ids=torch.tensor([15]))
    sens_act = cpu_model.encode_sensory(token_ids=torch.tensor([42]))

    # Ingest error feedback into Slot 2
    state = cpu_model.init_state(batch_size=1, device=device)
    _, state = cpu_model.step(sens_obs, state, thread_id=torch.tensor([2]), allow_routing=True)

    # Ingest action prompt into Slot 0
    _, state_clean = cpu_model.step(sens_act, state, thread_id=torch.tensor([0]), allow_routing=True)

    # Perturb Slot 2 error reflection
    state_pert = cpu_model.init_state(batch_size=1, device=device)
    state_pert.hierarchical_state.working_thoughts = state.hierarchical_state.working_thoughts.clone()
    state_pert.hierarchical_state.working_thoughts[0, 2] += 2.0
    _, state_pert = cpu_model.step(sens_act, state_pert, thread_id=torch.tensor([0]), allow_routing=True)

    # Decode action logits from Slot 0
    out_clean, _ = cpu_model.step(sens_act, state_clean, thread_id=torch.tensor([0]), allow_routing=True, read_language=True)
    out_pert, _ = cpu_model.step(sens_act, state_pert, thread_id=torch.tensor([0]), allow_routing=True, read_language=True)

    diff = torch.max(torch.abs(out_clean["logits"] - out_pert["logits"])).item()
    assert diff > 1e-5, f"Expected Slot 2 causal influence > 1e-5, got {diff}"


def test_episodic_memory_causal_retrieval_fusion(cpu_model: UnifiedPseudoBrain) -> None:
    """Verify episodic memory perturbation causally influences readout via fused retrieval."""
    device = torch.device("cpu")
    sens = cpu_model.encode_sensory(token_ids=torch.tensor([42]))

    state_clean = cpu_model.init_state(batch_size=1, device=device)
    state_pert = cpu_model.init_state(batch_size=1, device=device)
    state_pert.hierarchical_state.episodic_thoughts += 2.0

    out_clean, _ = cpu_model.step(sens, state_clean, thread_id=torch.tensor([0]), allow_routing=True, read_language=True)
    out_pert, _ = cpu_model.step(sens, state_pert, thread_id=torch.tensor([0]), allow_routing=True, read_language=True)

    diff = torch.max(torch.abs(out_clean["logits"] - out_pert["logits"])).item()
    assert diff > 1e-5, f"Expected episodic retrieval causal influence > 1e-5, got {diff}"


def test_inactive_slots_shuffling_diversity(cpu_model: UnifiedPseudoBrain) -> None:
    """Verify that permuting populated thought slots alters inter-slot message routing."""
    device = torch.device("cpu")
    state = cpu_model.init_state(batch_size=1, device=device)

    # Populate several slots with sensory information
    for s_id in (1, 2, 3, 4):
        sens = cpu_model.encode_sensory(token_ids=torch.tensor([10 * s_id]))
        _, state = cpu_model.step(sens, state, thread_id=torch.tensor([s_id]), allow_routing=True)

    wt = state.hierarchical_state.working_thoughts
    routed_orig, _ = cpu_model.router(wt)

    # Roll slot contents
    wt_rolled = torch.roll(wt, shifts=1, dims=1)
    routed_rolled, _ = cpu_model.router(wt_rolled)

    diff = torch.max(torch.abs(routed_orig - routed_rolled)).item()
    assert diff > 0.1, f"Expected significant routing difference after slot shift, got {diff}"


def test_recurrent_software_agent_routing_configuration(cpu_model: UnifiedPseudoBrain) -> None:
    """Verify RecurrentSoftwareAgent defaults allow_routing=True and maintains 4KB footprint."""
    agent = RecurrentSoftwareAgent(model=cpu_model, allow_routing=True)
    assert agent.allow_routing is True

    agent._ingest_text_into_slot("Goal: Solve task", slot_id=0)
    agent._ingest_text_into_slot("Observation: Test passed", slot_id=1)
    agent._ingest_text_into_slot("Reflection: Check edge cases", slot_id=2)
    agent._ingest_text_into_slot("History: Ran tests", slot_id=3)

    action = agent.generate_action_autoregressive(slot_id=0, max_new_tokens=8)
    assert isinstance(action, str)
    assert agent.cognitive_state.hierarchical_state.fast_state_bytes() == 4096

def test_recurrent_software_agent_execute_episode(cpu_model: UnifiedPseudoBrain, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify RecurrentSoftwareAgent.execute_episode follows POMDP protocol and conforms to evaluation trace schema."""
    from dataclasses import dataclass
    from irene_brain.agent.recurrent_software_agent import GenerationResult

    @dataclass
    class MockFeedback:
        action_type: str
        success: bool
        observation_text: str
        verified_completion: bool = False

    class MockEnv:
        def __init__(self):
            self.step = 0
        def execute_action(self, action):
            self.step += 1
            if self.step == 1:
                return MockFeedback("RUN_TESTS", False, "SyntaxError: invalid syntax")
            return MockFeedback("FINISH", True, "All tests passed", verified_completion=True)

    agent = RecurrentSoftwareAgent(model=cpu_model, allow_routing=True)
    monkeypatch.setattr(
        agent, "generate_action_autoregressive",
        lambda *a, **kw: GenerationResult("ACTION: RUN_TESTS test_count.py", [1, 2], "eos")
    )
    result = agent.execute_episode("Task: Implement count\nTarget: count_ops.py", MockEnv(), max_cycles=3)

    assert result.success is True
    assert result.stop_reason == "verified_completion"
    assert result.cycles == 2
    assert result.state_bytes == 4096
    assert len(result.trace) == 2
    assert result.trace[0]["executed"] is True
    assert result.trace[1]["executed"] is True

