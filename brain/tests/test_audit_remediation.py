"""Comprehensive unit and contract verification for audit remediations.

Covers:
1. Recurrent state persistence (step_count, P_t, working_salience) across 100+ steps.
2. Truthful trace generation: rejection of truncated generations, zero fabricated EOS,
   and exact emitted token preservation.
3. Observation token limit enforcement before state ingestion.
4. Fail-fast checkpoint loading with FileNotFoundError on missing files.
5. Strict metric boundedness in [0.0, 1.0] for target token accuracy.
6. Mathematical parity between forward_sequence_sequential and sequential step() unrolling.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import torch

from irene_brain.agent.pomdp_protocol import task_header, observation_transition
from irene_brain.agent.procedural_evaluator import compute_span_similarities
from irene_brain.agent.recurrent_software_agent import GenerationResult, RecurrentSoftwareAgent
from irene_brain.agent.software_environment import EnvironmentObservation, NeuralSoftwareEnvironment
from irene_brain.unified.unified_model import UnifiedPseudoBrain, make_unified_model


def test_state_persistence_across_steps():
    """Verify that step_count, P_t, active_thread, and working_salience persist across 100 steps."""
    model = make_unified_model(tier="tier1_3m", vocab_size=1000)
    model.eval()

    state = model.init_state(batch_size=1, device=torch.device("cpu"))
    assert int(state.hierarchical_state.step_count.item()) == 0
    assert state.hierarchical_state.working_salience is not None

    p_t_dummy = torch.randn(1, 16, 64)
    state.P_t = p_t_dummy
    state.hierarchical_state.P_t = p_t_dummy

    for step_idx in range(100):
        token = torch.tensor([42], dtype=torch.long)
        sensory = model.encode_sensory(token_ids=token)
        outputs, state = model.step(sensory, state, token_id=token, allow_routing=False)
        assert int(state.hierarchical_state.step_count.item()) == step_idx + 1
        assert state.P_t is not None
        assert state.hierarchical_state.P_t is not None
        assert state.hierarchical_state.working_salience is not None


def test_fail_fast_checkpoint_loading():
    """Verify that RecurrentSoftwareAgent raises FileNotFoundError on missing checkpoint."""
    with pytest.raises(FileNotFoundError, match="Checkpoint file does not exist"):
        RecurrentSoftwareAgent(checkpoint_path="nonexistent_champion_weights.pt")


def test_metric_boundedness_and_exact_match():
    """Verify that compute_span_similarities never exceeds 1.0 on pathological inputs."""
    # Pathological repetition test from Audit Finding #35
    tok_acc, char_sim = compute_span_similarities("a_a_a_a_a_a_a.py", "a_b.py")
    assert 0.0 <= tok_acc <= 1.0
    assert 0.0 <= char_sim <= 1.0

    # Exact match test
    tok_acc_exact, char_sim_exact = compute_span_similarities("data_loader.py", "data_loader.py")
    assert tok_acc_exact == 1.0
    assert char_sim_exact == 1.0

    # Completely disjoint test
    tok_acc_zero, _ = compute_span_similarities("foo", "bar")
    assert tok_acc_zero == 0.0


def test_generation_result_string_compatibility():
    """Verify GenerationResult behaves as str while preserving metadata."""
    res = GenerationResult("ACTION: WRITE_FILE mod.py", [10, 20, 30], "eos")
    assert isinstance(res, str)
    assert res == "ACTION: WRITE_FILE mod.py"
    assert res.startswith("ACTION: ")
    assert res.stop_reason == "eos"
    assert res.token_ids == [10, 20, 30]


def test_rejection_of_truncated_actions_without_fabricated_eos():
    """Verify that truncated actions are NOT executed and no EOS is fabricated."""
    model = make_unified_model(tier="tier1_3m", vocab_size=1000)
    agent = RecurrentSoftwareAgent(model=model)

    executed_actions = []

    class MockEnv:
        def execute_action(self, action):
            executed_actions.append(action)
            return EnvironmentObservation("WRITE_FILE", True, "ok")

    # Monkeypatch generate_action_autoregressive to simulate token-limit cutoff
    def mock_truncated_generation(*args, **kwargs):
        return GenerationResult("ACTION: WRITE_FILE incomplete_", [1, 2, 3], "token_limit")

    agent.generate_action_autoregressive = mock_truncated_generation

    res = agent.execute_episode("Task: Implement foo\nTarget: foo.py", MockEnv(), max_cycles=3)

    assert res.success is False
    assert res.stop_reason == "action_token_limit"
    assert len(executed_actions) == 0, "Truncated action must NOT be executed in environment!"
    assert len(res.trace) == 1
    assert res.trace[0]["executed"] is False
    assert res.trace[0]["generation_stop"] == "token_limit"
    assert res.trace[0]["token_ids"] == [1, 2, 3]


def test_observation_token_limit_enforcement():
    """Verify that observations exceeding max_observation_tokens trigger rejection before ingestion."""
    model = make_unified_model(tier="tier1_3m", vocab_size=1000)
    agent = RecurrentSoftwareAgent(model=model)

    class GiantObsEnv:
        def execute_action(self, action):
            # Return an observation with 500 words which exceeds a max limit of 10
            return EnvironmentObservation("READ_FILE", True, "word " * 500)

    # Valid initial action
    def mock_valid_generation(*args, **kwargs):
        return GenerationResult("ACTION: READ_FILE foo.py", [1, 2], "eos")

    agent.generate_action_autoregressive = mock_valid_generation

    res = agent.execute_episode(
        "Task: Read foo\nTarget: foo.py",
        GiantObsEnv(),
        max_cycles=3,
        max_observation_tokens=10,
    )

    assert res.success is False
    assert res.stop_reason == "observation_token_limit"


def test_forward_sequence_sequential_parity():
    """Verify mathematical parity between forward_sequence_sequential and step() unrolling."""
    torch.manual_seed(42)
    model = make_unified_model(tier="tier1_3m", vocab_size=500, use_routing=True)
    model.eval()

    tokens = torch.randint(1, 500, (1, 16))

    with torch.no_grad():
        # 1. Unroll via forward_sequence_sequential with allow_routing=True
        seq_out = model.forward_sequence_sequential(token_seq=tokens, allow_routing=True)

        # 2. Unroll manually step-by-step with allow_routing=True
        state = model.init_state(1, torch.device("cpu"))
        step_logits = []
        for t in range(tokens.shape[1]):
            tok_t = tokens[:, t]
            sensory = model.encode_sensory(token_ids=tok_t)
            outputs, state = model.step(sensory, state, allow_routing=True, token_id=tok_t)
            step_logits.append(outputs["logits"])
        manual_logits = torch.stack(step_logits, dim=1)

        diff = (seq_out["logits"] - manual_logits).abs().max().item()
        assert diff < 1e-6, f"Sequential unrolling discrepancy: {diff}"


class LogitForcer:
    """Forces model.step output logits to choose specific tokens in the real generator loop.

    Exercises generate_text_autoregressive unrolling and stopping conditions without monkeypatching
    the generator method itself. Only intercepts generation steps (where prompt_tokens is present in kwargs).
    """
    def __init__(self, agent: RecurrentSoftwareAgent, token_sequence: List[int], cycle: bool = False):
        self.agent = agent
        self.token_sequence = list(token_sequence)
        self.cycle = cycle
        self.idx = 0
        self.original_step = agent.model.step

    def __enter__(self):
        def forced_step(sensory, state, **kwargs):
            outputs, new_state = self.original_step(sensory, state, **kwargs)
            if "prompt_tokens" in kwargs:
                if self.cycle:
                    target_tok = self.token_sequence[self.idx % len(self.token_sequence)]
                elif self.idx < len(self.token_sequence):
                    target_tok = self.token_sequence[self.idx]
                else:
                    target_tok = self.token_sequence[-1]
                self.idx += 1
                forced_logits = torch.full((1, self.agent.model.vocab_size), -1000.0, device=self.agent.device)
                forced_logits[0, target_tok] = 1000.0
                outputs["logits"] = forced_logits
            return outputs, new_state

        self.agent.model.step = forced_step
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.agent.model.step = self.original_step


def test_forced_terminal_tokens_eos_pad_sep():
    """Verify explicit terminal token handling in the REAL generation loop.

    Exercises the unrolled generate_text_autoregressive loop via logit forcing.
    Verifies:
    1. Forced EOS: enters state exactly once, stop_reason is 'eos', actual_terminal_token_id is eos_id.
    2. Forced PAD: reported as 'pad', does not fabricate EOS, actual_terminal_token_id is pad_id.
    3. Forced SEP: reported as 'sep', does not fabricate EOS, actual_terminal_token_id is sep_id.
    4. Token limit: ordinary tokens unroll until max_new_tokens without inserting EOS.
    5. Repetition cutoff: repeating semantic sequences trigger cutoff without inserting EOS.
    6. Non-EOS/truncated actions are never executed by execute_episode.
    """
    model = make_unified_model(tier="tier1_3m", vocab_size=1000)
    agent = RecurrentSoftwareAgent(model=model)

    eos_id = agent.tokenizer.eos_id
    pad_id = agent.tokenizer.pad_id
    sep_id = agent.tokenizer.sep_id

    # 1. Forced EOS: real generator loop selects EOS, ingests [EOS] into state exactly once
    action_text = "ACTION: FINISH done"
    action_toks = agent.tokenizer.encode(action_text) or [0]
    c0 = int(agent.cognitive_state.hierarchical_state.step_count.item())
    with LogitForcer(agent, action_toks + [eos_id]):
        res_eos = agent.generate_action_autoregressive()
    c1 = int(agent.cognitive_state.hierarchical_state.step_count.item())

    assert res_eos.stop_reason == "eos"
    assert res_eos.actual_terminal_token_id == eos_id
    assert res_eos.decoded_raw_text == action_text
    prefix_len = len(agent.tokenizer.encode("[RESP]"))
    eos_token_len = len(agent.tokenizer.encode("[EOS]"))
    expected_steps = prefix_len + len(action_toks) + eos_token_len
    assert (c1 - c0) == expected_steps, f"EOS must step prefix + tokens + [EOS] once, expected {expected_steps}, got {c1 - c0}"

    # 2. Forced PAD: stop_reason must be 'pad', NOT 'eos', and must NOT ingest [EOS]
    pad_text = "ACTION: WRITE_FILE foo.py"
    pad_toks = agent.tokenizer.encode(pad_text) or [0]
    c0 = int(agent.cognitive_state.hierarchical_state.step_count.item())
    with LogitForcer(agent, pad_toks + [pad_id]):
        res_pad = agent.generate_action_autoregressive()
    c1 = int(agent.cognitive_state.hierarchical_state.step_count.item())

    assert res_pad.stop_reason == "pad"
    assert res_pad.actual_terminal_token_id == pad_id
    assert (c1 - c0) == prefix_len + len(pad_toks)

    # In execute_episode, PAD must halt without execution
    class DummyEnv:
        def execute_action(self, a):
            raise AssertionError("Environment must NOT be called for non-EOS cutoff!")

    with LogitForcer(agent, pad_toks + [pad_id]):
        ep_pad = agent.execute_episode("Task: foo\nTarget: foo.py", DummyEnv(), max_cycles=2)
    assert ep_pad.success is False
    assert ep_pad.stop_reason == "action_pad"
    assert ep_pad.trace[0]["executed"] is False
    assert ep_pad.trace[0]["actual_terminal_token_id"] == pad_id

    # 3. Forced SEP: stop_reason must be 'sep', NOT 'eos', and must NOT ingest [EOS]
    sep_text = "ACTION: WRITE_FILE foo.py"
    sep_toks = agent.tokenizer.encode(sep_text) or [0]
    with LogitForcer(agent, sep_toks + [sep_id]):
        res_sep = agent.generate_action_autoregressive()
    assert res_sep.stop_reason == "sep"
    assert res_sep.actual_terminal_token_id == sep_id

    with LogitForcer(agent, sep_toks + [sep_id]):
        ep_sep = agent.execute_episode("Task: foo\nTarget: foo.py", DummyEnv(), max_cycles=2)
    assert ep_sep.success is False
    assert ep_sep.stop_reason == "action_sep"
    assert ep_sep.trace[0]["executed"] is False
    assert ep_sep.trace[0]["actual_terminal_token_id"] == sep_id

    # 4. Token limit: sequence without terminal tokens reaches max_new_tokens
    non_syntax = [tok for tok in range(60, 200) if tok not in set(agent.tokenizer.encode(" \n\t_():=,.-'\"[]{}0123456789") or [])]
    lim_toks = (non_syntax[:5]) * 10
    with LogitForcer(agent, lim_toks):
        res_lim = agent.generate_action_autoregressive(max_new_tokens=10)
    assert res_lim.stop_reason == "token_limit"
    assert res_lim.actual_terminal_token_id is None

    with LogitForcer(agent, lim_toks):
        ep_lim = agent.execute_episode("Task: foo\nTarget: foo.py", DummyEnv(), max_cycles=2, max_action_tokens=10)
    assert ep_lim.success is False
    assert ep_lim.stop_reason == "action_token_limit"
    assert ep_lim.trace[0]["executed"] is False

    # 5. Repetition cutoff: 4 repeating non-syntax tokens trigger repetition cutoff
    rep_toks = (non_syntax[:4]) * 5
    with LogitForcer(agent, rep_toks):
        res_rep = agent.generate_action_autoregressive(max_new_tokens=50)
    assert res_rep.stop_reason == "repetition_cutoff"
    assert res_rep.actual_terminal_token_id is None

    with LogitForcer(agent, rep_toks):
        ep_rep = agent.execute_episode("Task: foo\nTarget: foo.py", DummyEnv(), max_cycles=2, max_action_tokens=50)
    assert ep_rep.success is False
    assert ep_rep.stop_reason == "action_repetition_cutoff"
    assert ep_rep.trace[0]["executed"] is False


def test_raw_vs_executed_action_decoupling():
    """Verify raw decoded text and executed text are strictly separated."""
    model = make_unified_model(tier="tier1_3m", vocab_size=1000)
    agent = RecurrentSoftwareAgent(model=model)

    raw_output = "ACTION: WRITE_FILE\ndef solve(): return 42"
    raw_toks = agent.tokenizer.encode(raw_output) + [agent.tokenizer.eos_id]

    # Unassisted mode (raw policy benchmark)
    with LogitForcer(agent, raw_toks):
        unassisted_res = agent.generate_action_autoregressive(
            target_module="solution.py",
            allow_assisted_transformations=False,
        )
    assert unassisted_res.decoded_raw_text == raw_output
    assert unassisted_res.executed_text == raw_output
    assert unassisted_res.transformation_applied is None

    # Assisted mode: target module insertion permitted but explicitly recorded
    with LogitForcer(agent, raw_toks):
        assisted_res = agent.generate_action_autoregressive(
            target_module="solution.py",
            allow_assisted_transformations=True,
        )
    assert assisted_res.decoded_raw_text == raw_output
    assert "solution.py" in assisted_res.executed_text
    assert assisted_res.transformation_applied == "target_module_binding"

    # Verify trace entries preserve all audit fields
    class RecordingEnv:
        def __init__(self):
            self.actions = []
        def execute_action(self, a):
            self.actions.append(a)
            return EnvironmentObservation("UNKNOWN", False, "unrecognized")

    env = RecordingEnv()
    with LogitForcer(agent, raw_toks):
        ep_res = agent.execute_episode(
            "Task: solve problem\nTarget: solution.py",
            env,
            max_cycles=1,
            allow_assisted_transformations=False,
        )
    entry = ep_res.trace[0]
    assert "generated_token_ids" in entry
    assert "decoded_raw_text" in entry
    assert "environment_action_text" in entry
    assert "executed_text" in entry
    assert "raw_action" in entry
    assert "transformation_applied" in entry
    assert "actual_terminal_token_id" in entry
    assert entry["decoded_raw_text"] == raw_output
    assert entry["executed_text"] == raw_output
    assert entry["raw_action"] == raw_output
    assert entry["environment_action_text"] == raw_output
    assert entry["transformation_applied"] is None


def test_parallel_forward_routing_fail_fast():
    """Verify that parallel=True raises ValueError when routing or thread_seq is requested."""
    model = make_unified_model(tier="tier1_3m", vocab_size=500, use_routing=True)
    tokens = torch.randint(1, 500, (1, 8))

    # allow_routing=True with parallel=True must fail fast
    with pytest.raises(ValueError, match="Parallel sequence forward does not support dynamic slot routing"):
        model(tokens, allow_routing=True, parallel=True)

    # thread_seq with parallel=True must fail fast
    thread_seq = torch.zeros((1, 8), dtype=torch.long)
    with pytest.raises(ValueError, match="Parallel sequence forward does not support dynamic slot routing"):
        model(tokens, thread_seq=thread_seq, parallel=True)


def test_full_task_prompt_ingestion_into_slot_1():
    """Verify that the full task specification is ingested into Slot 1 recurrent memory."""
    model = make_unified_model(tier="tier1_3m", vocab_size=1000)
    agent = RecurrentSoftwareAgent(model=model)

    class DummyEnv:
        def execute_action(self, a):
            return EnvironmentObservation("FINISH", True, "done")

    agent.generate_action_autoregressive = lambda *args, **kwargs: GenerationResult(
        text="ACTION: FINISH done",
        token_ids=[1, 2],
        stop_reason="eos",
        actual_terminal_token_id=agent.tokenizer.eos_id,
    )

    prompt = (
        "Task: Complex arithmetic\n"
        "Target: math_ops.py\n"
        "Detailed specification: The function must handle negative numbers, "
        "overflow protection, and return float precision with epsilon=1e-7."
    )

    agent.execute_episode(prompt, DummyEnv(), max_cycles=1)

    # Slot 1 must not be zero: it has ingested the detailed specification
    slot_1_state = agent.cognitive_state.hierarchical_state.working_thoughts[0, 1]
    assert slot_1_state.norm().item() > 0.01, "Slot 1 must have ingested task specification!"


def test_evaluator_control_correct_workspace_and_finish():
    """Evaluator control 1: correct workspace + FINISH + passing validator -> verified completion."""
    tmp = Path(tempfile.mkdtemp(prefix="test_ctrl_correct_"))
    try:
        (tmp / "math_utils.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
        (tmp / "test_math.py").write_text("from math_utils import add\ndef test_add(): assert add(2, 3) == 5\n", encoding="utf-8")
        def validator(env: NeuralSoftwareEnvironment):
            res = env._handle_run_tests("test_math.py")
            return res.success, "tests passed"

        env = NeuralSoftwareEnvironment(tmp, task_validator=validator)
        obs = env.execute_action("ACTION: FINISH repaired addition operator")
        assert obs.action_type == "FINISH"
        assert obs.success is True
        assert obs.verified_completion is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluator_control_incorrect_workspace_and_finish():
    """Evaluator control 2: incorrect workspace + FINISH -> rejected (success=False, verified_completion=False)."""
    tmp = Path(tempfile.mkdtemp(prefix="test_ctrl_incorrect_"))
    try:
        (tmp / "math_utils.py").write_text("def add(a, b): return a - b\n", encoding="utf-8")
        (tmp / "test_math.py").write_text("from math_utils import add\ndef test_add(): assert add(2, 3) == 5\n", encoding="utf-8")
        def validator(env: NeuralSoftwareEnvironment):
            res = env._handle_run_tests("test_math.py")
            return res.success, "tests failed"

        env = NeuralSoftwareEnvironment(tmp, task_validator=validator)
        obs = env.execute_action("ACTION: FINISH claim done")
        assert obs.action_type == "FINISH"
        assert obs.success is False
        assert obs.verified_completion is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluator_control_finish_without_validator():
    """Evaluator control 3: FINISH with no validator configured -> rejected."""
    tmp = Path(tempfile.mkdtemp(prefix="test_ctrl_novalidator_"))
    try:
        (tmp / "math_utils.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
        env = NeuralSoftwareEnvironment(tmp, task_validator=None)
        obs = env.execute_action("ACTION: FINISH done")
        assert obs.action_type == "FINISH"
        assert obs.success is False
        assert obs.verified_completion is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evaluator_control_write_file_not_verified_completion():
    """Evaluator control 4: WRITE_FILE success -> not verified completion."""
    tmp = Path(tempfile.mkdtemp(prefix="test_ctrl_write_"))
    try:
        env = NeuralSoftwareEnvironment(tmp)
        obs = env.execute_action("ACTION: WRITE_FILE math_utils.py\ndef add(a, b): return a + b\n")
        assert obs.action_type == "WRITE_FILE"
        assert obs.success is True
        assert obs.verified_completion is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_training_deployment_trajectory_equivalence():
    """Verify that training demonstration matches deployment ingestion token-by-token and slot-by-slot."""
    tmp1 = Path(tempfile.mkdtemp(prefix="test_traj_eq_train_"))
    tmp2 = Path(tempfile.mkdtemp(prefix="test_traj_eq_deploy_"))
    try:
        for tdir in (tmp1, tmp2):
            (tdir / "math_utils.py").write_text("def add(a, b): return a - b\n", encoding="utf-8")
            (tdir / "test_math.py").write_text("from math_utils import add\ndef test_add(): assert add(2, 3) == 5\n", encoding="utf-8")

        def validator(e: NeuralSoftwareEnvironment):
            return e._handle_run_tests("test_math.py").success, "validated"

        env_train = NeuralSoftwareEnvironment(tmp1, task_validator=validator)
        env_deploy = NeuralSoftwareEnvironment(tmp2, task_validator=validator)

        teacher_actions = [
            "ACTION: RUN_TESTS test_math.py",
            "ACTION: WRITE_FILE math_utils.py\ndef add(a, b): return a + b\n",
            "ACTION: RUN_TESTS test_math.py",
            "ACTION: FINISH repaired addition operator",
        ]
        goal = "Fix add in math_utils.py so tests pass"
        target_module = "math_utils.py"

        model = make_unified_model("tier1_3m", vocab_size=1000)
        agent = RecurrentSoftwareAgent(model=model)
        tok = agent.tokenizer

        # 1. Generate demonstration via trajectory builder
        header = task_header(goal, None, target_module)
        initial_prompt = f"Task: {goal}\nTarget: {target_module}\n"
        chunks = [(header, 0, False), (initial_prompt, 1, False)]
        for act in teacher_actions:
            chunks.append(("[RESP]", 0, False))
            chunks.append((act, 0, True))
            chunks.append(("[EOS]", 0, True))
            fb = env_train.execute_action(act)
            if fb.action_type == "FINISH" and fb.success and getattr(fb, "verified_completion", False):
                break
            obs_text = getattr(fb, "observation_text", str(fb))
            trans = observation_transition(fb, target_module)
            chunks.append((trans, 0, False))
            chunks.append((obs_text, 1, False))
            chunks.append((f"[ACTION_TAKEN: {fb.action_type}]", 3, False))
            if fb.action_type == "RUN_TESTS" and not fb.success:
                err = getattr(fb, "stderr", "") or obs_text
                chunks.append((f"[ERROR_OBSERVED: {err[:120]}]", 2, False))
            elif fb.action_type == "FINISH" and not fb.success:
                chunks.append((f"[FINISH_REJECTED: {obs_text[:160]}]", 2, False))

        d_tokens: List[int] = []
        d_threads: List[int] = []
        for text, slot_id, _ in chunks:
            toks = tok.encode(text) or [0]
            d_tokens.extend(toks)
            d_threads.extend([slot_id] * len(toks))

        # 2. Record tokens stepped during real deployment execution
        deployment_tokens: List[int] = []
        deployment_threads: List[int] = []

        orig_ingest = agent._ingest_text_into_slot
        def tracing_ingest(text, slot_id=0):
            t_list = agent.tokenizer.encode(text) or [0]
            deployment_tokens.extend(t_list)
            deployment_threads.extend([slot_id] * len(t_list))
            return orig_ingest(text, slot_id=slot_id)
        agent._ingest_text_into_slot = tracing_ingest

        act_iter = iter(teacher_actions)
        def teacher_action_gen(*args, **kwargs):
            act = next(act_iter)
            tracing_ingest("[RESP]", slot_id=0)
            tracing_ingest(act, slot_id=0)
            tracing_ingest("[EOS]", slot_id=0)
            act_tokens = agent.tokenizer.encode(act) or [0]
            return GenerationResult(
                text=act,
                token_ids=act_tokens,
                stop_reason="eos",
                actual_terminal_token_id=agent.tokenizer.eos_id,
                decoded_raw_text=act,
                executed_text=act,
                transformation_applied=None,
            )
        agent.generate_action_autoregressive = teacher_action_gen

        res = agent.execute_episode(initial_prompt, env_deploy, max_cycles=6)
        assert res.success is True
        assert res.stop_reason == "verified_completion"

        # Assert 100% token-by-token and slot-by-slot identity
        assert len(d_tokens) == len(deployment_tokens), f"Token count mismatch: {len(d_tokens)} vs {len(deployment_tokens)}"
        assert d_tokens == deployment_tokens, "Tokens must match identically between teacher dataset and deployment!"
        assert d_threads == deployment_threads, "Slot IDs must match identically between teacher dataset and deployment!"
    finally:
        shutil.rmtree(tmp1, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)


def test_nonfinite_logits_safeguard():
    """Verify that nonfinite logits fail fast with stop_reason='nonfinite_logits' and zero environment execution."""
    model = make_unified_model("tier1_3m", vocab_size=1000)
    agent = RecurrentSoftwareAgent(model=model)

    class DummyEnv:
        def __init__(self):
            self.executed_actions = []
        def execute_action(self, a):
            self.executed_actions.append(a)
            return EnvironmentObservation("FINISH", True, "done")

    env = DummyEnv()
    # Force step to return NaN logits when read_language=True
    orig_step = model.step
    def nan_step(*args, **kwargs):
        outputs, state = orig_step(*args, **kwargs)
        if outputs.get("logits") is not None:
            outputs["logits"] = torch.full_like(outputs["logits"], float("nan"))
        return outputs, state
    model.step = nan_step

    res = agent.execute_episode("Task: do something\nTarget: foo.py\n", env, max_cycles=1)
    assert res.success is False
    assert res.stop_reason == "action_nonfinite_logits"
    assert len(env.executed_actions) == 0, "No action should be executed in environment on nonfinite logits!"

