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


class RecurrentSoftwareAgent:
    """Persistent recurrent cognitive agent driving software tasks via POMDP actions."""

    def __init__(
        self,
        model: Optional[UnifiedPseudoBrain] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        tier: str = "tier2_35m",
        vocab_size: int = 32000,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cpu")
        self.checkpoint_loaded = False
        self.active_checkpoint: Optional[str] = None

        if model is None:
            if checkpoint_path is not None and Path(checkpoint_path).exists():
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                model_tier = ckpt.get("tier", tier)
                model_vocab = ckpt.get("vocab_size", vocab_size)
                use_skip = ckpt.get("use_token_skip", False)
                model = make_unified_model(tier=model_tier, vocab_size=model_vocab, use_token_skip=use_skip)
                model.load_state_dict(ckpt["model_state_dict"], strict=False)
                self.checkpoint_loaded = True
                self.active_checkpoint = str(checkpoint_path)
            else:
                model = make_unified_model(tier=tier, vocab_size=vocab_size)

        self.model = model.to(self.device)
        self.model.eval()

        self.tokenizer = BpeSemanticTokenizer(vocab_size=self.model.vocab_size)
        self.cognitive_state: UnifiedCognitiveState = self.model.init_state(batch_size=1, device=self.device)

    def reset(self) -> None:
        """Reset internal cognitive state to clean initial state."""
        self.cognitive_state = self.model.init_state(batch_size=1, device=self.device)

    def _ingest_text_into_slot(self, text: str, slot_id: int = 0) -> Optional[Tensor]:
        """Project text sequentially into sensory features and step the designated recurrent slot."""
        tokens = self.tokenizer.encode(text) or [0]
        tid = torch.tensor([slot_id % self.model.K_fast], dtype=torch.long, device=self.device)
        last_logits: Optional[Tensor] = None

        with torch.no_grad():
            for tok in tokens:
                tok_t = torch.tensor([tok], dtype=torch.long, device=self.device)
                sensory = self.model.encode_sensory(token_ids=tok_t)
                outputs, self.cognitive_state = self.model.step(
                    sensory, self.cognitive_state, thread_id=tid, allow_routing=False, token_id=tok_t
                )
                last_logits = outputs["logits"][0]
        return last_logits

    def generate_action_autoregressive(
        self,
        slot_id: int = 0,
        prompt_prefix: str = "[RESP]",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> str:
        """Autoregressively generate an action string directly from the recurrent state.

        Steps prompt prefix into the designated slot, then sequentially unrolls single-token
        step transitions from the recurrent core until [EOS] or [PAD], sampling next tokens
        from the vocabulary logits.
        """
        tid = torch.tensor([slot_id % self.model.K_fast], dtype=torch.long, device=self.device)
        last_logits: Optional[Tensor] = None

        if prompt_prefix:
            prefix_tokens = self.tokenizer.encode(prompt_prefix) or [0]
            with torch.no_grad():
                for tok in prefix_tokens:
                    tok_t = torch.tensor([tok], dtype=torch.long, device=self.device)
                    sensory = self.model.encode_sensory(token_ids=tok_t)
                    outputs, self.cognitive_state = self.model.step(
                        sensory, self.cognitive_state, thread_id=tid, allow_routing=False, token_id=tok_t
                    )
                    last_logits = outputs["logits"][0]

        if last_logits is None:
            return "ACTION: FINISH Goal evaluated"

        gen_tokens: List[int] = []
        syntax_exempt = set(self.tokenizer.encode(" \n\t_():=,.-'\"[]{}0123456789") or [])
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

                if next_tok in (self.tokenizer.eos_id, self.tokenizer.pad_id, self.tokenizer.sep_id):
                    break

                gen_tokens.append(next_tok)

                # Attractor loop safeguard: if same 4-token sequence repeats twice consecutively, break
                if len(gen_tokens) >= 8 and gen_tokens[-4:] == gen_tokens[-8:-4]:
                    break

                tok_t = torch.tensor([next_tok], dtype=torch.long, device=self.device)
                sensory = self.model.encode_sensory(token_ids=tok_t)
                outputs, self.cognitive_state = self.model.step(
                    sensory, self.cognitive_state, thread_id=tid, allow_routing=False, token_id=tok_t
                )
                last_logits = outputs["logits"][0]

        candidate = self.tokenizer.decode(gen_tokens, skip_special=True).strip()

        # If generated action starts with a recognized actuator verb, use it directly
        valid_verbs = ("READ_FILE", "WRITE_FILE", "EDIT_FILE", "RUN_TESTS", "RETRIEVE_MEMORY", "FINISH")
        parts = candidate.split()
        if len(parts) >= 2 and parts[1] in valid_verbs:
            return candidate

        # DO NOT convert invalid or uncalibrated tokens into ACTION: FINISH!
        if candidate.startswith("ACTION: "):
            return candidate
        return f"ACTION: UNPARSED {candidate}"

    def execute_pomdp_episode(
        self,
        goal: str,
        env: NeuralSoftwareEnvironment,
        action_plan: Optional[List[str]] = None,
        max_cycles: int = 10,
        reset_state: bool = True,
    ) -> AgentPOMDPEpisodeResult:
        """Execute a full Action-Observation POMDP software engineering cycle.

        If action_plan is provided, steps through the sequence of candidate actions,
        monitoring environment observations and state updates. If action_plan is None,
        unrolls the policy autoregressively from the recurrent core.

        If reset_state is True, internal state is cleared to ensure pure zero-shot isolation.
        If reset_state is False, state persists across episodes for lifelong learning.
        """
        t0 = time.perf_counter()
        if reset_state:
            self.reset()

        # Capture initial slot tensors to verify state transitions
        init_slots = self.cognitive_state.hierarchical_state.working_thoughts.clone()
        slot_0_init = init_slots[0, 0].clone()
        slot_1_init = init_slots[0, 1].clone()
        slot_2_init = init_slots[0, 2].clone()
        slot_3_init = init_slots[0, 3].clone()

        # 1. Step 0: Ingest Goal into Slot 0 (Goal Intent) with initial code writing phase
        self._ingest_text_into_slot(f"[GOAL: {goal}]\n[PHASE: WRITE_CODE]\n", slot_id=0)

        actions_taken: List[str] = []
        observations: List[str] = []
        episode_success = False
        final_summary = ""

        for cycle in range(max_cycles):
            if action_plan is not None:
                if cycle >= len(action_plan):
                    break
                current_action = action_plan[cycle]
            else:
                current_action = self.generate_action_autoregressive(slot_id=0, prompt_prefix="[RESP]")

            actions_taken.append(current_action)

            # 2. Environment executes atomic action deterministically
            obs: EnvironmentObservation = env.execute_action(current_action)
            observations.append(obs.observation_text)

            # 3. Step observation into Slot 0 (Continuous POMDP trajectory) and Slot 1 (Perception)
            self._ingest_text_into_slot(
                f"\n[OBSERVATION: {obs.observation_text}]\n[PHASE: VERIFY_AND_FINISH]\n", slot_id=0
            )
            self._ingest_text_into_slot(obs.observation_text, slot_id=1)

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
                    # Ingest rejection details into Slot 1 (Perception) and Slot 2 (Self-Repair)
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
        )
