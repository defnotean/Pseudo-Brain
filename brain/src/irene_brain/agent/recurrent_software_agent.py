"""Recurrent Software Agent for Pseudo-Brain.

Coordinates software engineering, testing, and debugging entirely through
Pseudo-Brain's persistent recurrent core (UnifiedPseudoBrain) in a true
Action-Observation POMDP loop:

1. Slot 0 (Goal Intent): Ingests the top-level user goal.
2. Slot 1 (Perception/World Model): Ingests environmental observations (file contents, test feedback, memory chunks).
3. Slot 2 (Active Subproblem / Action Candidate): Formulates next tool action and repairs test errors.
4. Slot 3 (Action / History): Records executed action history.

The environment contains ZERO domain-specific branching or presets; the core
drives the sequence of decisions through recurrent state transitions.
"""

from __future__ import annotations

import re
import hashlib
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from irene_brain.unified.unified_model import UnifiedCognitiveState, UnifiedPseudoBrain, make_unified_model
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.agent.pomdp_protocol import task_header, observation_transition
from irene_brain.agent.software_environment import EnvironmentObservation, NeuralSoftwareEnvironment


@dataclass
class AgentPOMDPEpisodeResult:
    """Outcome of an end-to-end recurrent POMDP software engineering episode."""
    goal: str
    success: bool
    cycles_completed: int
    actions_taken: List[str]
    observations: List[str]
    final_summary: str
    working_memory_bytes: int = 4096
    elapsed_ms: float = 0.0
    slot_0_delta: float = 0.0
    slot_1_delta: float = 0.0
    slot_2_delta: float = 0.0
    slot_3_delta: float = 0.0
    action_feedback: List[Dict[str, Any]] = field(default_factory=list)
    policy_source: str = "autonomous"


class GenerationResult(str):
    """Structured outcome of autoregressive recurrent generation.

    Subclasses str for 100% backward compatibility with string operations
    while preserving authentic emitted token IDs, actual terminal token ID,
    decoded raw text, executed text, transformation provenance, and verified stop reason.
    """
    text: str
    token_ids: List[int]
    stop_reason: str
    actual_terminal_token_id: Optional[int]
    decoded_raw_text: str
    executed_text: str
    transformation_applied: Optional[str]

    def __new__(
        cls,
        text: str,
        token_ids: List[int],
        stop_reason: str,
        actual_terminal_token_id: Optional[int] = None,
        decoded_raw_text: Optional[str] = None,
        executed_text: Optional[str] = None,
        transformation_applied: Optional[str] = None,
    ):
        obj = str.__new__(cls, text)
        obj.text = text
        obj.token_ids = list(token_ids)
        obj.stop_reason = stop_reason
        obj.actual_terminal_token_id = actual_terminal_token_id
        obj.decoded_raw_text = text if decoded_raw_text is None else decoded_raw_text
        obj.executed_text = text if executed_text is None else executed_text
        obj.transformation_applied = transformation_applied
        return obj


class RecurrentSoftwareAgent:
    """Persistent recurrent cognitive agent driving software tasks via POMDP actions."""

    def __init__(
        self,
        model: Optional[UnifiedPseudoBrain] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        tier: str = "tier2_35m",
        vocab_size: int = 32000,
        device: Optional[torch.device] = None,
        allow_routing: bool = True,
        allow_assisted_transformations: bool = False,
    ):
        self.device = device or torch.device("cpu")
        self.allow_routing = allow_routing
        self.allow_assisted_transformations = allow_assisted_transformations
        self.checkpoint_loaded = False
        self.active_checkpoint: Optional[str] = None
        tokenizer_json = None

        if model is None:
            if checkpoint_path is not None:
                ckpt_p = Path(checkpoint_path)
                if not ckpt_p.is_file():
                    raise FileNotFoundError(f"Checkpoint file does not exist: {checkpoint_path}")
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                tokenizer_json = ckpt.get("tokenizer_json")
                model_tier = ckpt.get("tier", tier)
                model_vocab = ckpt.get("vocab_size", vocab_size)
                use_skip = ckpt.get("use_token_skip", False)
                use_gated = ckpt.get("use_gated_token_skip", False)
                use_ptr = ckpt.get("use_pointer_copy", False)
                model = make_unified_model(
                    tier=model_tier,
                    vocab_size=model_vocab,
                    use_token_skip=use_skip,
                    use_gated_token_skip=use_gated,
                    use_pointer_copy=use_ptr,
                    compensated_state=ckpt.get("compensated_state", False),
                    pointer_mode=ckpt.get("pointer_mode", "sequential"),
                    retention_profile=ckpt.get("retention_profile", "legacy"),
                )
                model.load_state_dict(ckpt["model_state_dict"], strict=True)
                self.checkpoint_loaded = True
                self.active_checkpoint = str(checkpoint_path)
            else:
                model = make_unified_model(tier=tier, vocab_size=vocab_size)

        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = (BpeSemanticTokenizer.from_str(tokenizer_json, vocab_size=self.model.vocab_size)
                          if tokenizer_json is not None
                          else BpeSemanticTokenizer(vocab_size=self.model.vocab_size))
        self.cognitive_state: UnifiedCognitiveState = self.model.init_state(batch_size=1, device=self.device)
        self.active_prompt_tokens: List[int] = []

    def reset(self) -> None:
        """Reset internal cognitive state to clean initial state."""
        self.cognitive_state = self.model.init_state(batch_size=1, device=self.device)
        self.active_prompt_tokens = []

    def _ingest_text_into_slot(self, text: str, slot_id: int = 0) -> Optional[Tensor]:
        """Project text sequentially into sensory features and step the designated recurrent slot."""
        tokens = self.tokenizer.encode(text) or [0]
        tid = torch.tensor([slot_id % self.model.logical_slots], dtype=torch.long, device=self.device)
        last_logits: Optional[Tensor] = None

        with torch.no_grad():
            for token_index, tok in enumerate(tokens):
                tok_t = torch.tensor([tok], dtype=torch.long, device=self.device)
                sensory = self.model.encode_sensory(token_ids=tok_t)
                outputs, self.cognitive_state = self.model.step(
                    sensory, self.cognitive_state, thread_id=tid, allow_routing=self.allow_routing, token_id=tok_t,
                    read_language=token_index == len(tokens) - 1,
                )
                if outputs["logits"] is not None:
                    last_logits = outputs["logits"][0]
        return last_logits

    def generate_text_autoregressive(
        self,
        slot_id: int = 0,
        prompt_prefix: str = "[RESP]",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Autoregressively decode raw text directly from the recurrent state.

        Steps prompt prefix into the designated slot, then sequentially unrolls single-token
        step transitions from the recurrent core until [EOS] or [PAD], sampling next tokens
        from the vocabulary logits.
        """
        tid = torch.tensor([slot_id % self.model.logical_slots], dtype=torch.long, device=self.device)
        last_logits: Optional[Tensor] = None
        prompt_t = (
            torch.tensor([self.active_prompt_tokens], dtype=torch.long, device=self.device)
            if self.active_prompt_tokens
            else None
        )
        # Reset pointer sequential state so that new action generation starts with clean global span acquisition
        if hasattr(self.cognitive_state, "ptr_prev_alpha"):
            self.cognitive_state.ptr_prev_alpha = None
            self.cognitive_state.ptr_prev_gamma = None

        if prompt_prefix:
            prefix_tokens = self.tokenizer.encode(prompt_prefix) or [0]
            with torch.no_grad():
                for tok in prefix_tokens:
                    tok_t = torch.tensor([tok], dtype=torch.long, device=self.device)
                    sensory = self.model.encode_sensory(token_ids=tok_t)
                    outputs, self.cognitive_state = self.model.step(
                        sensory, self.cognitive_state, thread_id=tid, allow_routing=self.allow_routing, token_id=tok_t,
                        prompt_tokens=prompt_t,
                    )
                    last_logits = outputs["logits"][0]

        if last_logits is None:
            return GenerationResult(
                text="",
                token_ids=[0],
                stop_reason="token_limit",
                actual_terminal_token_id=None,
                decoded_raw_text="",
                executed_text="",
                transformation_applied=None,
            )

        gen_tokens: List[int] = []
        syntax_exempt = set(self.tokenizer.encode(" \n\t_():=,.-'\"[]{}0123456789") or [])
        stop_reason = "token_limit"
        actual_terminal_token_id: Optional[int] = None

        with torch.no_grad():
            for _ in range(max_new_tokens):
                step_logits = last_logits.clone()
                # Apply repetition penalty only when sampling with temperature > 0 and only to non-syntax tokens
                if temperature > 1e-4 and len(gen_tokens) >= 1:
                    for prev_tok in set(gen_tokens[-8:]):
                        if prev_tok not in syntax_exempt:
                            if step_logits[prev_tok] > 0:
                                step_logits[prev_tok] = step_logits[prev_tok] / 1.2

                if temperature <= 1e-4:
                    next_tok = int(step_logits.argmax().item())
                else:
                    probs = torch.softmax(step_logits / temperature, dim=-1)
                    next_tok = int(torch.multinomial(probs, num_samples=1).item())

                # Explicit handling of terminal tokens: never conflate PAD or SEP with EOS!
                if next_tok == self.tokenizer.eos_id:
                    stop_reason = "eos"
                    actual_terminal_token_id = next_tok
                    break
                elif next_tok == self.tokenizer.pad_id:
                    stop_reason = "pad"
                    actual_terminal_token_id = next_tok
                    break
                elif next_tok == self.tokenizer.sep_id:
                    stop_reason = "sep"
                    actual_terminal_token_id = next_tok
                    break

                gen_tokens.append(next_tok)

                tok_t = torch.tensor([next_tok], dtype=torch.long, device=self.device)
                sensory = self.model.encode_sensory(token_ids=tok_t)
                outputs, self.cognitive_state = self.model.step(
                    sensory, self.cognitive_state, thread_id=tid, allow_routing=self.allow_routing, token_id=tok_t,
                    prompt_tokens=prompt_t,
                )
                last_logits = outputs["logits"][0]

                # The returned token must enter the state even when it triggers
                # the loop cutoff, before the closing EOS is consumed.
                if len(gen_tokens) >= 8 and gen_tokens[-4:] == gen_tokens[-8:-4]:
                    stop_reason = "repetition_cutoff"
                    break

        # ONLY ingest [EOS] if the generation stopped due to natural [EOS]!
        # Do NOT ingest [EOS] on PAD, SEP, token_limit, or repetition_cutoff!
        if stop_reason == "eos":
            self._ingest_text_into_slot("[EOS]", slot_id=slot_id)

        candidate = self.tokenizer.decode(gen_tokens, skip_special=True).strip()
        return GenerationResult(
            text=candidate,
            token_ids=gen_tokens,
            stop_reason=stop_reason,
            actual_terminal_token_id=actual_terminal_token_id,
            decoded_raw_text=candidate,
            executed_text=candidate,
            transformation_applied=None,
        )

    def generate_action_autoregressive(
        self,
        slot_id: int = 0,
        prompt_prefix: str = "[RESP]",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        target_module: Optional[str] = None,
        allow_assisted_transformations: bool = False,
    ) -> GenerationResult:
        """Decode a policy action without converting invalid output into success.

        When allow_assisted_transformations is False (raw policy benchmark mode):
            Zero assistance is applied. No module name insertion, no regex patching.
            The raw text and executed text are strictly identical.

        When allow_assisted_transformations is True:
            Formatting adaptation may be applied, and transformation_applied records
            the exact assistance applied.
        """
        gen_res = self.generate_text_autoregressive(
            slot_id=slot_id, prompt_prefix=prompt_prefix,
            max_new_tokens=max_new_tokens, temperature=temperature,
        )
        raw_text = gen_res.text
        transformation: Optional[str] = None
        executed_text = raw_text

        if allow_assisted_transformations:
            candidate = raw_text
            if "WRITE_:FILE" in candidate:
                candidate = candidate.replace("WRITE_:FILE", "WRITE_FILE ")
                transformation = "colon_typo_fix"

            candidate = re.sub(r"^ACTION:\s*([A-Z_]+)", r"ACTION: \1", candidate)

            # If generated action starts with a recognized actuator verb, use it directly
            valid_verbs = ("READ_FILE", "WRITE_FILE", "EDIT_FILE", "RUN_TESTS", "RETRIEVE_MEMORY", "FINISH")
            parts = candidate.split()
            if len(parts) >= 2 and parts[1] in valid_verbs:
                verb = parts[1]
                if verb == "WRITE_FILE" and target_module:
                    # If model emitted code but target module name had formatting artifact, ensure target module is bound
                    first_line, sep, body = candidate.partition("\n")
                    if not sep:
                        m_code = re.search(r"(def\s+|class\s+|return\s+)", candidate)
                        if m_code:
                            code = candidate[m_code.start():]
                            candidate = f"ACTION: WRITE_FILE {target_module}\n{code}"
                            transformation = "target_module_reconstruction"
                    elif not first_line.replace("ACTION: WRITE_FILE", "").strip().endswith(".py"):
                        candidate = f"ACTION: WRITE_FILE {target_module}\n{body}"
                        transformation = "target_module_binding"
                executed_text = candidate
            elif candidate.startswith("ACTION: "):
                executed_text = candidate
            else:
                executed_text = f"ACTION: UNPARSED {candidate}"
                transformation = "unparsed_prefix_wrap"
        else:
            # Raw benchmark: zero heuristic rewriting of verbs or target modules
            if raw_text.startswith("ACTION: "):
                executed_text = raw_text
                transformation = None
            else:
                executed_text = f"ACTION: UNPARSED {raw_text}"
                transformation = "unparsed_prefix_wrap"

        return GenerationResult(
            text=executed_text,
            token_ids=gen_res.token_ids,
            stop_reason=gen_res.stop_reason,
            actual_terminal_token_id=gen_res.actual_terminal_token_id,
            decoded_raw_text=raw_text,
            executed_text=executed_text,
            transformation_applied=transformation,
        )

    def execute_pomdp_episode(
        self,
        goal: str,
        env: NeuralSoftwareEnvironment,
        action_plan: Optional[List[str]] = None,
        max_cycles: int = 10,
        reset_state: bool = True,
        target_module: Optional[str] = None,
        target_function: Optional[str] = None,
    ) -> AgentPOMDPEpisodeResult:
        """Execute a full Action-Observation POMDP software engineering cycle.

        If action_plan is provided, steps through the sequence of candidate actions,
        monitoring environment observations and state updates. If action_plan is None,
        unrolls the policy autoregressively from the recurrent core.

        If reset_state is True, internal state is cleared to ensure pure zero-shot isolation.
        If reset_state is False, state persists across episodes for lifelong learning.
        """
        import re
        t0 = time.perf_counter()
        if reset_state:
            self.reset()

        # Extract target module and function if not explicitly passed
        if not target_module:
            m_mod = re.search(r'([a-zA-Z0-9_]+\.py)', goal)
            if m_mod:
                target_module = m_mod.group(1)
        if not target_function:
            m_fn = re.search(r'`([a-zA-Z0-9_]+)(?:\(.*?\)|`)', goal)
            if m_fn:
                target_function = m_fn.group(1)
            else:
                m_fn2 = re.search(r'(?:class|function|Implement|Write)\s+`?([a-zA-Z0-9_]+)', goal)
                if m_fn2:
                    target_function = m_fn2.group(1)

        # Capture initial slot tensors to verify state transitions
        init_slots = self.cognitive_state.hierarchical_state.working_thoughts.clone()
        slot_0_init = init_slots[0, 0].clone()
        slot_1_init = init_slots[0, 1].clone()
        slot_2_init = init_slots[0, 2].clone()
        slot_3_init = init_slots[0, 3].clone()

        # 1. Step 0: Ingest Goal and Target Specification into Slot 0 (Goal Intent)
        init_prompt = task_header(goal, target_function, target_module)
        self.active_prompt_tokens = self.tokenizer.encode(init_prompt) or [0]
        self._ingest_text_into_slot(init_prompt, slot_id=0)

        actions_taken: List[str] = []
        observations: List[str] = []
        action_feedback: List[Dict[str, Any]] = []
        episode_success = False
        final_summary = ""

        def target_snapshot():
            # Evaluation provenance only; never fed into model inputs or state.
            path = env._resolve_safe_path(target_module) if target_module else None
            if path is None:
                return None
            if not path.is_file():
                return {"exists": False}
            return {"exists": True, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

        for cycle in range(max_cycles):
            if action_plan is not None:
                if cycle >= len(action_plan):
                    break
                current_action = action_plan[cycle]
            else:
                current_action = self.generate_action_autoregressive(slot_id=0, prompt_prefix="[RESP]")

            actions_taken.append(current_action)

            # 2. Environment executes atomic action deterministically
            target_before = target_snapshot()
            obs: EnvironmentObservation = env.execute_action(current_action)
            observations.append(obs.observation_text)
            transition = observation_transition(obs, target_module)
            action_feedback.append({
                "cycle": cycle + 1, "action_type": obs.action_type,
                "success": obs.success, "return_code": obs.return_code,
                "observation": obs.observation_text, "transition": transition,
                "target_before": target_before, "target_after": target_snapshot(),
            })

            # 3. Step observation dynamically into Slot 0 (Continuous POMDP trajectory)
            obs_text = obs.observation_text.strip()
            self._ingest_text_into_slot(transition, slot_id=0)
            self._ingest_text_into_slot(obs_text, slot_id=1)

            # 4. Record action execution into Slot 3 (Action History)
            self._ingest_text_into_slot(f"[ACTION_TAKEN: {obs.action_type}]", slot_id=3)

            # 5. Handle Terminal Action
            if obs.action_type == "FINISH":
                if obs.success:
                    episode_success = True
                    final_summary = obs.observation_text
                    break
                else:
                    # Model claimed completion, but task validation rejected it!
                    self._ingest_text_into_slot(f"[FINISH_REJECTED: {obs.observation_text[:160]}]", slot_id=2)
                    final_summary = obs.observation_text

            # 6. If test failure occurred, core observes the error in state and handles repair
            if obs.action_type == "RUN_TESTS" and not obs.success:
                self._ingest_text_into_slot(f"[ERROR_OBSERVED: {obs.stderr[:120]}]", slot_id=2)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Calculate slot deltas across active slots
        current_slots = self.cognitive_state.hierarchical_state.working_thoughts
        slot_0_delta = float(torch.norm(current_slots[0, 0] - slot_0_init).item())
        slot_1_delta = float(torch.norm(current_slots[0, 1] - slot_1_init).item())
        slot_2_delta = float(torch.norm(current_slots[0, 2] - slot_2_init).item())
        slot_3_delta = float(torch.norm(current_slots[0, 3] - slot_3_init).item())

        return AgentPOMDPEpisodeResult(
            goal=goal,
            success=episode_success,
            cycles_completed=len(actions_taken),
            actions_taken=actions_taken,
            observations=observations,
            final_summary=final_summary,
            working_memory_bytes=self.cognitive_state.hierarchical_state.fast_state_bytes(),
            elapsed_ms=elapsed_ms,
            slot_0_delta=slot_0_delta,
            slot_1_delta=slot_1_delta,
            slot_2_delta=slot_2_delta,
            slot_3_delta=slot_3_delta,
            action_feedback=action_feedback,
            policy_source="autonomous" if action_plan is None else "scripted",
        )

    def execute_episode(
        self,
        initial_prompt: str,
        environment: Any,
        *,
        max_cycles: int = 8,
        max_action_tokens: int = 512,
        max_observation_tokens: int = 4096,
        max_prompt_tokens: int = 4096,
        allow_assisted_transformations: Optional[bool] = None,
    ) -> Any:
        """Execute episode adhering to the audited tool policy evaluation protocol."""
        from dataclasses import asdict
        from irene_brain.agent.parallel_depth_episode import ParallelEpisodeResult

        if allow_assisted_transformations is None:
            allow_assisted_transformations = getattr(self, "allow_assisted_transformations", False)

        for value in (max_cycles, max_action_tokens, max_observation_tokens, max_prompt_tokens):
            if type(value) is not int or value < 1:
                raise ValueError("Episode limits must be positive integers")
        if not isinstance(initial_prompt, str) or not initial_prompt:
            raise ValueError("A nonempty initial task prompt is required")

        self.reset()

        # Extract target module, target function, and goal if present
        targets = re.findall(r"^Target: (.+)$", initial_prompt, re.MULTILINE)
        target_module = targets[0].strip() if targets else None
        if not target_module:
            m_mod = re.search(r"([a-zA-Z0-9_]+\.py)", initial_prompt)
            if m_mod:
                target_module = m_mod.group(1)

        target_function = None
        m_fn = re.search(r"Implement [`']?([a-zA-Z0-9_]+)\(", initial_prompt)
        if m_fn:
            target_function = m_fn.group(1).strip()

        goal = ""
        m_goal = re.search(r"Task: (.+)", initial_prompt)
        if m_goal:
            goal = m_goal.group(1).strip()
        else:
            goal = initial_prompt.strip()

        header = task_header(goal, target_function, target_module)
        conditioned_prompt = f"{header}\n{initial_prompt}"

        # Enforce prompt limit against the actual full conditioned prompt
        conditioned_tokens = self.tokenizer.encode(conditioned_prompt) or [0]
        if len(conditioned_tokens) > max_prompt_tokens:
            raise ValueError("Conditioned prompt exceeds the registered token limit")

        self.active_prompt_tokens = conditioned_tokens
        prompt_sha = hashlib.sha256(conditioned_prompt.encode()).hexdigest()

        # Ingest goal header into Slot 0 (Goal Intent)
        self._ingest_text_into_slot(header, slot_id=0)

        # Ingest full task specification into Slot 1 (Perception/World Model)
        # Ensures recurrent core receives every constraint and instruction needed to solve the task!
        self._ingest_text_into_slot(initial_prompt, slot_id=1)

        trace = []
        for cycle in range(max_cycles):
            action_res = self.generate_action_autoregressive(
                slot_id=0,
                prompt_prefix="[RESP]",
                max_new_tokens=max_action_tokens,
                target_module=target_module,
                allow_assisted_transformations=allow_assisted_transformations,
            )
            state_bytes = self.cognitive_state.hierarchical_state.fast_state_bytes()

            if action_res.stop_reason != "eos":
                # Incomplete action generated (token limit, repetition cutoff, pad, or sep).
                # Reject action: DO NOT execute in the environment! DO NOT fabricate EOS!
                entry = {
                    "cycle": cycle + 1,
                    "generated_token_ids": action_res.token_ids,
                    "decoded_raw_text": action_res.decoded_raw_text,
                    "executed_text": None,
                    "raw_action": action_res.decoded_raw_text,
                    "transformation_applied": action_res.transformation_applied,
                    "actual_terminal_token_id": action_res.actual_terminal_token_id,
                    "token_ids": action_res.token_ids,
                    "generation_stop": action_res.stop_reason,
                    "executed": False,
                    "state_bytes": state_bytes,
                    "feedback": None,
                }
                trace.append(entry)
                stop_reason = f"action_{action_res.stop_reason}"
                return ParallelEpisodeResult(False, stop_reason, len(trace), state_bytes, prompt_sha, tuple(trace))

            action = action_res.executed_text
            feedback = environment.execute_action(action)
            entry = {
                "cycle": cycle + 1,
                "generated_token_ids": action_res.token_ids,
                "decoded_raw_text": action_res.decoded_raw_text,
                "executed_text": action,
                "raw_action": action,
                "transformation_applied": action_res.transformation_applied,
                "actual_terminal_token_id": action_res.actual_terminal_token_id,
                "token_ids": action_res.token_ids,
                "generation_stop": "eos",
                "executed": True,
                "state_bytes": state_bytes,
                "feedback": asdict(feedback),
            }
            trace.append(entry)

            if feedback.action_type == "FINISH" and feedback.success and getattr(feedback, "verified_completion", False):
                return ParallelEpisodeResult(True, "verified_completion", len(trace), state_bytes, prompt_sha, tuple(trace))

            obs_text = getattr(feedback, "observation_text", str(feedback))
            transition = observation_transition(feedback, target_module)
            obs_tokens = self.tokenizer.encode(obs_text) or [0]
            trans_tokens = self.tokenizer.encode(transition) or [0]
            if len(obs_tokens) > max_observation_tokens or len(trans_tokens) > max_observation_tokens:
                return ParallelEpisodeResult(False, "observation_token_limit", len(trace), state_bytes, prompt_sha, tuple(trace))

            self._ingest_text_into_slot(transition, slot_id=0)
            self._ingest_text_into_slot(obs_text, slot_id=1)
            self._ingest_text_into_slot(f"[ACTION_TAKEN: {feedback.action_type}]", slot_id=3)

            if feedback.action_type == "FINISH" and not feedback.success:
                self._ingest_text_into_slot(f"[FINISH_REJECTED: {obs_text[:160]}]", slot_id=2)
            if feedback.action_type == "RUN_TESTS" and not feedback.success:
                err_snippet = getattr(feedback, "stderr", "") or obs_text
                self._ingest_text_into_slot(f"[ERROR_OBSERVED: {err_snippet[:120]}]", slot_id=2)

        state_bytes = self.cognitive_state.hierarchical_state.fast_state_bytes()
        return ParallelEpisodeResult(False, "cycle_limit", len(trace), state_bytes, prompt_sha, tuple(trace))

