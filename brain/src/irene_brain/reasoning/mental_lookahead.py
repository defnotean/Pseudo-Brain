"""Adaptive Mental Lookahead Engine for Test-Time Thought Simulation.

Provides test-time compute scaling for Pseudo-Brain:
1. Dynamic Entropy Trigger: Steps greedily (<1 ms) when top-1 token probability is high;
   triggers latent mental rollouts when token ambiguity / entropy is elevated.
2. Latent State-Space Rollouts: Unrolls B candidate trajectories forward H steps (H in [2, 8])
   purely in compact 4.0 KB working memory without allocating transformer KV-caches.
3. Value-Guided Trajectory Selection: Evaluates candidate thought branches using model.value_head
   and cumulative log-probabilities to pick optimal cognitive paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from irene_brain.unified.unified_model import UnifiedCognitiveState, UnifiedPseudoBrain


@dataclass
class TrajectoryCandidate:
    """A simulated mental thought branch in working memory."""
    tokens: List[int]
    cumulative_logprob: float
    final_state: UnifiedCognitiveState
    predicted_value: float
    total_score: float


class MentalLookaheadEngine:
    """Test-Time Mental Lookahead and Cognitive Simulation Engine."""

    def __init__(
        self,
        model: UnifiedPseudoBrain,
        branch_factor: int = 4,
        horizon: int = 4,
        entropy_threshold: float = 1.2,
        value_weight: float = 0.5,
    ) -> None:
        self.model = model
        self.branch_factor = branch_factor
        self.horizon = horizon
        self.entropy_threshold = entropy_threshold
        self.value_weight = value_weight

    def evaluate_entropy(self, logits: Tensor) -> float:
        """Compute Shannon entropy of next-token distribution."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).item()
        return float(entropy)

    def simulate_lookahead(
        self,
        current_state: UnifiedCognitiveState,
        candidate_tokens: List[int],
        device: torch.device,
    ) -> TrajectoryCandidate:
        """Roll out a single candidate branch forward H steps in latent mental state."""
        # Deep-copy compact 4.0 KB state
        state = UnifiedCognitiveState(
            hierarchical_state=current_state.hierarchical_state.clone(),
            active_thread=current_state.active_thread.clone(),
            last_action=current_state.last_action.clone() if current_state.last_action is not None else None,
        )

        tokens_unrolled: List[int] = []
        cum_logprob = 0.0

        first_tok = candidate_tokens[0]
        tokens_unrolled.append(first_tok)

        # Step 1: execute chosen candidate token
        sensory = self.model.encode_sensory(token_ids=torch.tensor([first_tok], device=device))
        out, state = self.model.step(sensory, state)
        logits = out["logits"]
        log_probs = F.log_softmax(logits, dim=-1)

        # Step 2..H: greedy rollout forward in latent state
        for _ in range(1, self.horizon):
            next_t = int(logits.argmax(dim=-1).item())
            if next_t == self.model.vocab_size - 1 or next_t == 2:  # EOS or PAD
                break
            tokens_unrolled.append(next_t)
            cum_logprob += log_probs[0, next_t].item()

            sensory = self.model.encode_sensory(token_ids=torch.tensor([next_t], device=device))
            out, state = self.model.step(sensory, state)
            logits = out["logits"]
            log_probs = F.log_softmax(logits, dim=-1)

        # Evaluate final state using value_head
        pred_val = out["predicted_value"].item()
        total_score = cum_logprob + self.value_weight * pred_val

        return TrajectoryCandidate(
            tokens=tokens_unrolled,
            cumulative_logprob=cum_logprob,
            final_state=state,
            predicted_value=pred_val,
            total_score=total_score,
        )

    def select_next_token_adaptive(
        self,
        current_logits: Tensor,
        current_state: UnifiedCognitiveState,
        device: torch.device,
    ) -> Tuple[int, bool, float]:
        """Adaptive selection: greedy if confident, mental lookahead if ambiguous."""
        entropy = self.evaluate_entropy(current_logits)

        # Fast path: low entropy / high confidence -> greedy update (<0.5 ms)
        if entropy < self.entropy_threshold:
            greedy_token = int(current_logits.argmax(dim=-1).item())
            return greedy_token, False, entropy

        # Adaptive path: high ambiguity -> simulate top-K branches in mental state
        top_k = torch.topk(current_logits[0], self.branch_factor)
        top_indices = top_k.indices.tolist()

        candidates: List[TrajectoryCandidate] = []
        for cand_tok in top_indices:
            cand = self.simulate_lookahead(
                current_state=current_state,
                candidate_tokens=[cand_tok],
                device=device,
            )
            candidates.append(cand)

        # Pick candidate branch with highest total score (log-prob + future state value)
        candidates.sort(key=lambda c: c.total_score, reverse=True)
        best_token = candidates[0].tokens[0]
        return best_token, True, entropy

    def generate_with_lookahead(
        self,
        prompt_tokens: List[int],
        max_new_tokens: int = 64,
        eos_id: int = 2,
        pad_id: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Generate response with adaptive test-time mental lookahead."""
        dev = device or torch.device("cpu")
        state = self.model.init_state(batch_size=1, device=dev)

        # 1. Prime state on prompt
        out = None
        for t_id in prompt_tokens:
            sensory = self.model.encode_sensory(token_ids=torch.tensor([t_id], device=dev))
            out, state = self.model.step(sensory, state)

        generated_tokens: List[int] = []
        lookahead_activations = 0
        current_logits = out["logits"].clone()

        # 2. Adaptive streaming generation
        for _ in range(max_new_tokens):
            next_token, used_lookahead, ent = self.select_next_token_adaptive(
                current_logits=current_logits,
                current_state=state,
                device=dev,
            )
            if used_lookahead:
                lookahead_activations += 1

            if next_token == eos_id or next_token == pad_id:
                break

            generated_tokens.append(next_token)
            sensory = self.model.encode_sensory(token_ids=torch.tensor([next_token], device=dev))
            out, state = self.model.step(sensory, state)
            current_logits = out["logits"].clone()

        return {
            "tokens": generated_tokens,
            "lookahead_activations": lookahead_activations,
            "total_tokens": len(generated_tokens),
            "final_state": state,
        }
