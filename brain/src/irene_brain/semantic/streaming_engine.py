"""Streaming Cognitive Inference Engine.

Executes continuous, real-time, token-by-token state updates and autoregressive
generation through Pseudo-Brain's persistent recurrent slots with zero conversation-history replay.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .native_semantic_model import NativeSemanticPseudoBrain, SemanticCognitiveState
from .tokenizer import SemanticTokenizer


class StreamingCognitiveSession:
    """Manages an active conversational stream using persistent cognitive state."""

    def __init__(
        self,
        model: NativeSemanticPseudoBrain,
        tokenizer: Optional[SemanticTokenizer] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer or SemanticTokenizer(max_threads=model.K)
        self.device = device or next(model.parameters()).device
        self.model.eval()

        self.state = self.model.init_state(batch_size=1, device=self.device)
        self.step_latencies_ms: List[float] = []
        self.total_tokens_processed = 0

    def reset(self) -> None:
        """Reset internal cognitive state to fresh identity initialization."""
        self.state = self.model.init_state(batch_size=1, device=self.device)
        self.step_latencies_ms.clear()
        self.total_tokens_processed = 0

    def step_token(
        self,
        token_id: int,
        thread_id: Optional[int] = None,
        allow_routing: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        """Ingest a single token and update cognitive state. Returns (logits [V], latency_ms)."""
        t_in = torch.tensor([token_id], dtype=torch.long, device=self.device)
        th_in = torch.tensor([thread_id], dtype=torch.long, device=self.device) if thread_id is not None else None

        t0 = time.perf_counter()
        with torch.no_grad():
            logits, self.state = self.model.step(t_in, self.state, thread_ids=th_in, allow_routing=allow_routing)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        self.step_latencies_ms.append(dt_ms)
        self.total_tokens_processed += 1
        return logits.squeeze(0), dt_ms

    def ingest_text(
        self,
        text: str,
        thread_id: Optional[int] = None,
        allow_routing: bool = False,
    ) -> List[float]:
        """Ingest a sequence of text without generating a response."""
        token_ids = self.tokenizer.encode(text, thread_id=thread_id)
        latencies = []
        for tid in token_ids:
            _, dt = self.step_token(tid, thread_id=thread_id, allow_routing=allow_routing)
            latencies.append(dt)
        return latencies

    def generate_response(
        self,
        prompt_text: Optional[str] = None,
        thread_id: Optional[int] = None,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        stop_on_eos: bool = True,
        stop_on_sep: bool = True,
        allow_routing: bool = False,
    ) -> Dict[str, Any]:
        """Generate response tokens autoregressively from current persistent state."""
        prompt_latencies = []
        last_logits = None

        if prompt_text:
            token_ids = self.tokenizer.encode(prompt_text, thread_id=thread_id)
            for tid in token_ids:
                last_logits, dt = self.step_token(tid, thread_id=thread_id, allow_routing=allow_routing)
                prompt_latencies.append(dt)

        # Ingest [RESP] token marker to switch to response phase
        resp_tok = self.tokenizer.resp_id
        last_logits, dt = self.step_token(resp_tok, thread_id=thread_id, allow_routing=allow_routing)
        prompt_latencies.append(dt)

        generated_token_ids: List[int] = []
        gen_latencies = []

        for _ in range(max_new_tokens):
            if last_logits is None:
                break

            if temperature <= 1e-4:
                next_tok = int(last_logits.argmax().item())
            else:
                probs = F.softmax(last_logits / temperature, dim=-1)
                next_tok = int(torch.multinomial(probs, num_samples=1).item())

            if stop_on_eos and next_tok == self.tokenizer.eos_id:
                break
            if stop_on_sep and next_tok == self.tokenizer.sep_id:
                break

            generated_token_ids.append(next_tok)
            last_logits, dt = self.step_token(next_tok, thread_id=thread_id, allow_routing=allow_routing)
            gen_latencies.append(dt)

        generated_text = self.tokenizer.decode(generated_token_ids, skip_special=True)
        return {
            "response_text": generated_text,
            "generated_tokens": generated_token_ids,
            "prompt_latencies_ms": prompt_latencies,
            "generation_latencies_ms": gen_latencies,
            "mean_step_latency_ms": float(np.mean(gen_latencies)) if gen_latencies else 0.0,
            "active_thread": int(self.state.active_thread.item()),
            "total_tokens_session": self.total_tokens_processed,
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Compute live session telemetry and latency percentiles."""
        lats = np.array(self.step_latencies_ms) if self.step_latencies_ms else np.array([0.0])
        active_tid = int(self.state.active_thread.item())
        thoughts = self.state.thoughts[0]  # [K, W]
        P_t = self.state.P_t[0]  # [K, V]

        return {
            "active_thread": active_tid,
            "total_tokens": self.total_tokens_processed,
            "latency_mean_ms": float(np.mean(lats)),
            "latency_p50_ms": float(np.percentile(lats, 50)),
            "latency_p90_ms": float(np.percentile(lats, 90)),
            "latency_p99_ms": float(np.percentile(lats, 99)),
            "latency_under_16_67ms": bool(np.all(lats <= 16.67)),
            "active_slot_norm": float(torch.norm(thoughts[active_tid]).item()),
            "active_plastic_norm": float(torch.norm(P_t[active_tid]).item()),
            "all_slots_mean_norm": float(torch.norm(thoughts, dim=-1).mean().item()),
        }
