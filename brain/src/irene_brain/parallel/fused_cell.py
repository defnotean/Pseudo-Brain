"""Parallel Recurrent Blocks and Fused Cells for Pseudo-Brain.

Bypasses the sequential unrolling wall (for t in range(T):) by reformulating the
recurrent core into an associative linear recurrence evaluated via parallel
associative prefix scan in O(log T) steps during training, while preserving
exact O(1) single-token streaming during inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from irene_brain.model.brain_cell import FactorizedLowRankProjection
from irene_brain.model.torch_model import deterministic_thought_identity_codes
from irene_brain.parallel.parallel_scan import parallel_scan


@dataclass
class FusedCognitiveState:
    """Persistent cognitive state for O(1) single-step streaming inference."""
    thoughts: Tensor  # [B, K, W] or [B, W]
    prev_thoughts: Tensor  # [B, K, W]
    active_thread: Tensor  # [B] integer thread indices


class FusedBrainCellCore(nn.Module):
    """Parallel Recurrent BrainCell Core with Associative Linear Recurrence.

    Formulates the recurrence as:
        h_t = a_t * h_{t-1} + b_t
    where:
        a_t = sigmoid(W_iz(x_eff) + bias_z)   (retention gate in [0, 1])
        b_t = (1.0 - a_t) * tanh(W_in(x_eff)) (candidate state update)

    During training:
        Input sequence X of shape [B, T, D] is projected in parallel across all T tokens.
        The recurrence is resolved in O(log T) steps via parallel associative prefix scan.
    During inference:
        step(thought, x) evaluates the exact single-step transition in O(1) time.
    """

    def __init__(
        self,
        input_size: int = 512,
        thought_size: int = 48,
        rank: Optional[int] = None,
        proj_dim: Optional[int] = None,
        tier: Optional[str] = None,
        num_deep_layers: int = 3,
        retention_bias_init: float = 1.5,
        use_reset_gate: bool = False,
    ) -> None:
        super().__init__()
        self.tier = tier

        # Tier 2 scaling laws (Law 1: W=64, Law 2: rank r=32, proj_dim=4096)
        if tier == "tier2":
            thought_size = 64
            rank = 32 if rank is None else rank
            proj_dim = 4096 if proj_dim is None else proj_dim
            input_size = 4096 if input_size in (48, 256, 512) else input_size

        self.input_size = input_size
        self.thought_size = thought_size
        self.rank = rank
        self.proj_dim = proj_dim
        self.num_deep_layers = num_deep_layers
        self.use_reset_gate = use_reset_gate

        # Deep projection parameter core (Law 2 for capacity scaling)
        if tier == "tier2" and proj_dim is not None and proj_dim > 0:
            deep_layers: list[nn.Module] = []
            for _ in range(num_deep_layers):
                deep_layers.append(nn.Linear(proj_dim, proj_dim))
                deep_layers.append(nn.GELU())
            self.deep_proj: Optional[nn.Module] = nn.Sequential(*deep_layers)
        else:
            self.deep_proj = None

        # Input projections: Dense or Factorized Low-Rank Projections
        if rank is not None and rank > 0:
            self.W_iz: nn.Module = FactorizedLowRankProjection(input_size, thought_size, rank=rank)
            self.W_in: nn.Module = FactorizedLowRankProjection(input_size, thought_size, rank=rank)
            if self.use_reset_gate:
                self.W_ir: Optional[nn.Module] = FactorizedLowRankProjection(input_size, thought_size, rank=rank)
            else:
                self.W_ir = None
        else:
            self.W_iz = nn.Linear(input_size, thought_size)
            self.W_in = nn.Linear(input_size, thought_size)
            if self.use_reset_gate:
                self.W_ir = nn.Linear(input_size, thought_size)
            else:
                self.W_ir = None

        # Retention gate bias initialization (positive bias promotes persistent memory)
        self._init_biases(retention_bias_init)

        # Telemetry
        self.eval_count: int = 0
        self.total_flops: int = 0

    def _init_biases(self, retention_bias: float) -> None:
        if isinstance(self.W_iz, nn.Linear):
            nn.init.constant_(self.W_iz.bias, retention_bias)
        elif hasattr(self.W_iz, "up") and isinstance(self.W_iz.up, nn.Linear):
            if self.W_iz.up.bias is not None:
                nn.init.constant_(self.W_iz.up.bias, retention_bias)

    def state_bytes(self, K: int = 128, bytes_per_element: int = 4) -> int:
        """Law 1 Recurrent State Footprint: K * W * 4 bytes."""
        return K * self.thought_size * bytes_per_element

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def _compute_coefficients(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes associative scan coefficients (a, b) from input x in parallel.

        Returns:
            a: Retention / decay tensor in (0, 1)
            b: Scaled candidate injection tensor
        """
        x_eff = self.deep_proj(x) if self.deep_proj is not None else x
        z = torch.sigmoid(self.W_iz(x_eff))
        n = torch.tanh(self.W_in(x_eff))

        if self.W_ir is not None:
            r = torch.sigmoid(self.W_ir(x_eff))
            n = n * r

        a = z
        b = (1.0 - z) * n
        return a, b

    def step(self, thought: Tensor, x: Tensor) -> Tensor:
        """Single-step O(1) streaming recurrent update for inference.

        Args:
            thought: Previous hidden state [..., thought_size]
            x: Current input [..., input_size]

        Returns:
            Next hidden state [..., thought_size]
        """
        batch_slots = thought.numel() // self.thought_size
        self.eval_count += batch_slots

        a, b = self._compute_coefficients(x)
        return a * thought + b

    def forward_sequential(self, x: Tensor, h0: Optional[Tensor] = None) -> Tensor:
        """Sequential unrolling loop over sequence x: [B, T, ...].

        Used as an exact mathematical reference to verify parallel scan correctness.
        """
        B, T = x.shape[:2]
        trailing_shape = x.shape[2:-1] if x.ndim > 3 else ()
        state_shape = (B, *trailing_shape, self.thought_size)

        if h0 is not None:
            h = h0
        else:
            h = torch.zeros(state_shape, dtype=x.dtype, device=x.device)

        states: List[Tensor] = []
        for t in range(T):
            x_t = x[:, t]
            h = self.step(h, x_t)
            states.append(h)

        return torch.stack(states, dim=1)

    def forward(
        self,
        x: Tensor,
        h0: Optional[Tensor] = None,
        method: Literal["work_efficient", "hillis_steele", "sequential"] = "work_efficient",
    ) -> Tensor:
        """Parallel sequence forward evaluation.

        Args:
            x: Input tensor sequence of shape [B, T, input_size] or [B, T, K, input_size].
            h0: Optional initial state [B, thought_size] or [B, K, thought_size].
            method: Scan algorithm ('work_efficient', 'hillis_steele', 'sequential').

        Returns:
            Hidden states tensor of shape [B, T, ..., thought_size] computed in O(log T) steps.
        """
        if method == "sequential":
            return self.forward_sequential(x, h0=h0)

        # Track telemetry
        num_tokens = x.shape[0] * x.shape[1]
        self.eval_count += num_tokens

        # Parallel feed-forward projection across all T tokens simultaneously
        a, b = self._compute_coefficients(x)

        # O(log T) parallel associative prefix scan along sequence dim=1
        return parallel_scan(a, b, h0=h0, dim=1, method=method)


class FusedSlotRecurrentBlock(nn.Module):
    """Multi-Slot Parallel Cognitive Recurrent Block.

    Maintains K cognitive threads/slots in parallel.
    Uses Cognitive Input Gating (CIG) to direct inputs into slots and computes
    all slot recurrent updates across the entire sequence via Parallel Associative Scan.
    """

    def __init__(
        self,
        K: int = 16,
        thought_size: int = 32,
        proj_dim: int = 128,
        tier: Optional[str] = None,
        rank: Optional[int] = None,
        num_deep_layers: int = 3,
    ) -> None:
        super().__init__()
        self.K = K
        self.thought_size = thought_size
        self.proj_dim = proj_dim

        # Slot identity orthogonal representations
        self.register_buffer(
            "slot_identities",
            deterministic_thought_identity_codes(thoughtlets=K, width=thought_size),
            persistent=False,
        )

        # Cognitive Input Gating (CIG) gate projected from input and slot identity
        self.cig_gate = nn.Sequential(
            nn.Linear(proj_dim + thought_size, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.cig_gate[0].bias, 1.0)

        # Shared core recurrent cell across all K slots
        self.core = FusedBrainCellCore(
            input_size=proj_dim,
            thought_size=thought_size,
            rank=rank,
            proj_dim=proj_dim,
            tier=tier,
            num_deep_layers=num_deep_layers,
        )

    def _compute_salience(self, x_exp: Tensor, slot_codes: Tensor) -> Tensor:
        """Computes CIG salience scores with T=0.5 sharpening.

        Args:
            x_exp: [..., K, proj_dim]
            slot_codes: [..., K, thought_size]

        Returns:
            salience: [..., K, 1]
        """
        gate_in = torch.cat([x_exp, slot_codes], dim=-1)
        raw_gate = self.cig_gate(gate_in)
        clamped = raw_gate.clamp(1e-4, 1.0 - 1e-4)
        logit_gate = torch.log(clamped / (1.0 - clamped))
        return torch.sigmoid(logit_gate / 0.5)

    def step(
        self,
        x_proj: Tensor,  # [B, proj_dim]
        thoughts: Tensor,  # [B, K, W]
    ) -> Tensor:
        """Single streaming step in O(1) time.

        Args:
            x_proj: Projected sensory/token input [B, proj_dim]
            thoughts: Previous slot thought states [B, K, W]

        Returns:
            next_thoughts: Updated slot thoughts [B, K, W]
        """
        B = x_proj.shape[0]
        x_exp = x_proj.unsqueeze(1).expand(-1, self.K, -1)  # [B, K, proj_dim]
        slots = self.slot_identities.unsqueeze(0).expand(B, -1, -1)

        # Input salience gate
        salience = self._compute_salience(x_exp, slots)  # [B, K, 1]

        # Candidate thought update from core
        new_t = self.core.step(thoughts, x_exp)  # [B, K, W]

        # Gated state blend
        return (1.0 - salience) * thoughts + salience * new_t

    def forward(
        self,
        x_seq: Tensor,  # [B, T, proj_dim]
        h0: Optional[Tensor] = None,  # [B, K, W]
        method: Literal["work_efficient", "hillis_steele", "sequential"] = "work_efficient",
    ) -> Tensor:
        """Parallel sequence forward across all T tokens and K slots simultaneously.

        Args:
            x_seq: Projected input sequence [B, T, proj_dim]
            h0: Optional initial states [B, K, thought_size]
            method: Scan algorithm

        Returns:
            all_thoughts: Tensor [B, T, K, thought_size] computed in O(log T) steps.
        """
        B, T, _ = x_seq.shape
        x_exp = x_seq.unsqueeze(2).expand(-1, -1, self.K, -1)  # [B, T, K, proj_dim]
        slots = self.slot_identities.view(1, 1, self.K, self.thought_size).expand(B, T, -1, -1)

        # 1. Compute CIG salience for all T tokens and K slots in 1 batched operation
        salience = self._compute_salience(x_exp, slots)  # [B, T, K, 1]

        # 2. Compute candidate recurrence coefficients from core
        # Core has: h_core = a_core * h_{t-1} + b_core
        # With CIG gating: h_t = (1 - salience) * h_{t-1} + salience * (a_core * h_{t-1} + b_core)
        #                      = [(1 - salience) + salience * a_core] * h_{t-1} + [salience * b_core]
        # This is strictly of the form: h_t = a_net * h_{t-1} + b_net!
        a_core, b_core = self.core._compute_coefficients(x_exp)  # [B, T, K, W]

        a_net = (1.0 - salience) + salience * a_core
        b_net = salience * b_core

        # 3. Parallel associative scan along sequence dim=1
        return parallel_scan(a_net, b_net, h0=h0, dim=1, method=method)


class ParallelNativeSemanticPseudoBrain(nn.Module):
    """Pseudo-Brain Native Semantic Cognitive Architecture with Parallel Associative Scan.

    Fully parallel drop-in replacement for NativeSemanticPseudoBrain.
    Replaces token-by-token Python loop unrolling with O(log T) parallel prefix scan,
    breaking the sequential training wall for 1B parameter scaling.
    """

    def __init__(
        self,
        vocab_size: int = 344,
        K: int = 16,
        thought_size: int = 32,
        embed_dim: int = 64,
        proj_dim: int = 128,
        tier: Optional[str] = None,
        rank: Optional[int] = None,
        num_deep_layers: int = 3,
    ) -> None:
        super().__init__()
        self.tier = tier
        if tier == "tier2":
            thought_size = 64
            rank = 32 if rank is None else rank
            proj_dim = 4096 if proj_dim == 128 else proj_dim
            embed_dim = 128 if embed_dim == 64 else embed_dim

        self.vocab_size = vocab_size
        self.K = K
        self.thought_size = thought_size
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.rank = rank
        self.num_deep_layers = num_deep_layers

        # 1. Semantic Token Embedding & Sensory Projection
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Linear(embed_dim, proj_dim)

        # 2. Multi-Slot Parallel Recurrent Block
        self.recurrent_block = FusedSlotRecurrentBlock(
            K=K,
            thought_size=thought_size,
            proj_dim=proj_dim,
            tier=tier,
            rank=rank,
            num_deep_layers=num_deep_layers,
        )

        # 3. Thread-Targeted Semantic Readout Head
        head_hidden = max(64, min(proj_dim // 2, 512))
        self.slot_head = nn.Sequential(
            nn.Linear(thought_size, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, vocab_size),
        )

        # Slot Orthogonality Codes
        self.register_buffer(
            "slot_identities",
            deterministic_thought_identity_codes(thoughtlets=K, width=thought_size),
            persistent=False,
        )

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}

    def state_bytes(self, bytes_per_element: int = 4) -> int:
        return self.K * self.thought_size * bytes_per_element

    def init_state(self, batch_size: int, device: torch.device) -> FusedCognitiveState:
        """Initialize persistent state for O(1) streaming inference."""
        slots = self.slot_identities.to(device=device)
        thoughts = slots.unsqueeze(0).expand(batch_size, -1, -1).clone()
        active_thread = torch.zeros(batch_size, dtype=torch.long, device=device)
        return FusedCognitiveState(
            thoughts=thoughts,
            prev_thoughts=thoughts.clone(),
            active_thread=active_thread,
        )

    def step(
        self,
        token_ids: Tensor,  # [B]
        state: FusedCognitiveState,
        thread_ids: Optional[Tensor] = None,  # [B]
    ) -> Tuple[Tensor, FusedCognitiveState]:
        """Single-step O(1) streaming update for inference."""
        B = token_ids.shape[0]

        # Update active thread index
        if thread_ids is not None:
            state.active_thread = thread_ids.clamp(0, self.K - 1)
        else:
            is_thread_tok = (token_ids >= 9) & (token_ids < 9 + self.K)
            if is_thread_tok.any():
                new_tids = (token_ids - 9).clamp(0, self.K - 1)
                state.active_thread = torch.where(is_thread_tok, new_tids, state.active_thread)

        # Embed & Project
        emb = self.embedding(token_ids)
        x_proj = self.proj(emb)

        # Recurrent slot update
        next_thoughts = self.recurrent_block.step(x_proj, state.thoughts)

        # Readout from active thread
        batch_idx = torch.arange(B, device=token_ids.device)
        active_k = state.active_thread.clamp(0, self.K - 1)
        queried = next_thoughts[batch_idx, active_k]
        logits = self.slot_head(queried)

        next_state = FusedCognitiveState(
            thoughts=next_thoughts,
            prev_thoughts=state.thoughts,
            active_thread=state.active_thread,
        )
        return logits, next_state

    def forward_sequential(
        self,
        token_seq: Tensor,  # [B, T]
        thread_seq: Optional[Tensor] = None,  # [B, T]
    ) -> Tensor:
        """Sequential unrolling forward loop for mathematical validation."""
        B, T = token_seq.shape
        state = self.init_state(B, token_seq.device)
        logits_list: List[Tensor] = []

        for t in range(T):
            tid = thread_seq[:, t] if thread_seq is not None else None
            logits_t, state = self.step(token_seq[:, t], state, thread_ids=tid)
            logits_list.append(logits_t)

        return torch.stack(logits_list, dim=1)

    def forward(
        self,
        token_seq: Tensor,  # [B, T]
        thread_seq: Optional[Tensor] = None,  # [B, T]
        method: Literal["work_efficient", "hillis_steele", "sequential"] = "work_efficient",
    ) -> Tensor:
        """Parallel sequence forward in O(log T) steps using parallel associative scan.

        Args:
            token_seq: Token index sequence [B, T]
            thread_seq: Optional explicit active thread IDs [B, T]
            method: Scan algorithm ('work_efficient', 'hillis_steele', 'sequential')

        Returns:
            logits: Output next-token prediction logits [B, T, vocab_size]
        """
        if method == "sequential":
            return self.forward_sequential(token_seq, thread_seq=thread_seq)

        B, T = token_seq.shape
        dev = token_seq.device

        # 1. Embed & Project all T tokens in parallel
        emb = self.embedding(token_seq)  # [B, T, embed_dim]
        x_proj = self.proj(emb)  # [B, T, proj_dim]

        # 2. Initial state from slot identities
        h0 = self.slot_identities.unsqueeze(0).expand(B, -1, -1).to(dev)  # [B, K, W]

        # 3. Parallel recurrent block across all T tokens and K slots in O(log T) steps
        all_thoughts = self.recurrent_block(x_seq=x_proj, h0=h0, method=method)  # [B, T, K, W]

        # 4. Determine active thread for each token step t
        if thread_seq is not None:
            active_threads = thread_seq.clamp(0, self.K - 1)
        else:
            # Detect thread markers [THREAD:k] across the sequence
            is_thread = (token_seq >= 9) & (token_seq < 9 + self.K)
            thread_targets = (token_seq - 9).clamp(0, self.K - 1)
            # Scan / propagate active thread index across time
            active_threads = torch.zeros(B, T, dtype=torch.long, device=dev)
            cur_th = torch.zeros(B, dtype=torch.long, device=dev)
            for t in range(T):
                th_t = is_thread[:, t]
                cur_th = torch.where(th_t, thread_targets[:, t], cur_th)
                active_threads[:, t] = cur_th

        # 5. Gather queried slot for all B and T in parallel
        # active_threads: [B, T] -> expand to gather index [B, T, 1, W]
        idx = active_threads.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.thought_size)
        queried = torch.gather(all_thoughts, 2, idx).squeeze(2)  # [B, T, W]

        # 6. Readout head across all B and T in a single batched GEMM
        logits = self.slot_head(queried)  # [B, T, vocab_size]
        return logits
