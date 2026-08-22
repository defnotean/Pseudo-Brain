"""Phase 2.6 Workstream G: Episodic Memory v0.

A deliberately minimal, fully vectorized episodic store:

  write: gated key/value commit per step (learned write gate, top-1 salience)
  retrieve: cosine-similarity key match -> value readout
  inject: retrieved evidence enters cognition via the EXISTING
          `retrieved_memory` interface of IreneBrainModel (no new bypass:
          the same path the architecture contract already defines)

Guardrails (owner directive):
  - everything else frozen: same BrainCell, K, heads, training recipe, seeds,
    torture protocol;
  - no target leakage: writes happen only from frames BEFORE the decision frame,
    and t17's value is written only when flashed (before the delay);
  - memory cost reported separately (params/bytes/latency/write & retrieval rates);
  - control-matched comparison vs a parameter-matched widened no-memory model;
  - six causal interventions at evaluation time.

The store is an nn.Module wrapper AROUND the frozen model; model weights are
loaded from a baseline checkpoint and remain trainable only via the normal
recipe (memory params are additional and counted separately).
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn


class EpisodicMemoryV0(nn.Module):
    """Vectorized key-value episodic store with learned gated writes."""

    def __init__(self, *, key_width: int, value_width: int, entries: int = 64,
                 write_temperature: float = 1.0):
        super().__init__()
        self.key_width = key_width
        self.value_width = value_width
        self.entries = entries
        # keys/values stored as buffers so they persist in checkpoints but are
        # not "learned parameters" (capacity is reported as bytes, not params).
        self.register_buffer("keys", torch.zeros(entries, key_width))
        self.register_buffer("values", torch.zeros(entries, value_width))
        self.register_buffer("ages", torch.full((entries,), float("inf")))
        self.write_ptr = 0
        # Learned write gate: should this frame's content be committed?
        self.write_gate = nn.Sequential(
            nn.Linear(key_width + value_width, key_width),
            nn.SiLU(),
            nn.Linear(key_width, 1),
            nn.Sigmoid(),
        )
        # Key encoder: maps a frame summary to lookup key space.
        self.key_encoder = nn.Sequential(
            nn.Linear(key_width * 2, key_width),
            nn.LayerNorm(key_width),
        )
        self.write_temperature = write_temperature
        # Telemetry counters (reset by caller).
        self.writes_attempted = 0
        self.writes_committed = 0
        self.retrievals = 0
        self.last_gate_mean = None
        self.last_retrieval_sims = None
        self.last_retrieval_idx = None

    @torch.no_grad()
    def _commit(self, key: Tensor, value: Tensor) -> None:
        idx = self.write_ptr % self.entries
        self.keys[idx] = key.squeeze(0)
        self.values[idx] = value.squeeze(0)
        self.ages[idx] = 0.0
        self.write_ptr += 1

    def write_step(self, frame_summary: Tensor, content: Tensor) -> Tensor:
        """Gated write. frame_summary/content: [B, W]. Returns gate values."""
        b = frame_summary.shape[0]
        gate_in = torch.cat((frame_summary, content), dim=-1)
        g = self.write_gate(gate_in)  # [B,1]
        self.last_gate_mean = g.mean().detach()
        self.writes_attempted += b
        committed = (g > 0.5).float().mean().item()
        self.writes_committed += int(round(committed * b))
        if self.training or True:
            # soft-write during training for differentiability: blend into slot
            idx = self.write_ptr % self.entries
            self.keys[idx] = (1 - g[0]) * self.keys[idx] + g[0] * frame_summary[0]
            self.values[idx] = (1 - g[0]) * self.values[idx] + g[0] * content[0]
            self.ages[idx] = 0.0
            self.write_ptr += 1
        return g

    def retrieve(self, query: Tensor, k: int = 1) -> tuple[Tensor, Tensor, Tensor]:
        """query: [B, W]. Returns (values [B,k,V], sims [B,k], indices [B,k])."""
        q = torch.nn.functional.normalize(query, dim=-1)
        keys_n = torch.nn.functional.normalize(self.keys, dim=-1)
        sims = q @ keys_n.T  # [B, entries]
        sims_masked = torch.where(torch.isfinite(self.ages), sims,
                                  torch.full_like(sims, -2.0))
        vals, idx = torch.topk(sims_masked, k=min(k, self.entries), dim=-1)
        retrieved = self.values[idx]  # [B, k, V]
        self.retrievals += query.shape[0]
        self.last_retrieval_sims = vals.detach()
        self.last_retrieval_idx = idx.detach()
        return retrieved, vals, idx

    def gate_sparsity_loss(self) -> Tensor:
        """L2 pressure toward closed gate; task decides how open it must be."""
        if self.last_gate_mean is None:
            return torch.zeros(1, device=self.keys.device).squeeze()
        return self.last_gate_mean

    def reset_store(self) -> None:
        self.keys.zero_(); self.values.zero_()
        self.ages.fill_(float("inf")); self.write_ptr = 0

    def parameter_count(self) -> int:
        """Learned parameters added by memory (gates/encoders only)."""
        return sum(p.numel() for p in self.parameters())

    def storage_bytes(self) -> int:
        return self.keys.numel() * self.keys.element_size() + \
               self.values.numel() * self.values.element_size()
