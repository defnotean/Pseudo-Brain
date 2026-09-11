"""Opt-in token session for the parallel-depth action/observation core.

The session retains only the ordinary 4 KB recurrent state and the immutable
initial pointer prompt. Output tokens are returned to the caller, never replayed
as model context. This adapter does not execute tools or judge task completion.
"""
from dataclasses import dataclass

import torch

from irene_brain.evaluation.recurrent_ingest import parallel_recurrent_ingest
from irene_brain.evaluation.recurrent_prefill import parallel_recurrent_prefill


@dataclass(frozen=True)
class SessionGeneration:
    token_ids: tuple[int, ...]
    stop_reason: str
    state_bytes: int


class ParallelDepthSession:
    """One stream; eight logical layers packed in 16x64 float32 values.

Call start for every independent task. New observation tokens and fixed response
prefixes enter through ingest; generated tokens enter once through model.step.
No logits, action history, pointer alignments, or past observations are cached.
"""

    def __init__(self, model):
        if getattr(model, 'architecture', None) != 'parallel_depth_v1':
            raise ValueError('Session requires the parallel-depth recurrent model')
        if model.embedding.weight.dtype != torch.float64:
            raise ValueError('Session requires the FP64 parity model')
        self.model = model.eval()
        self._state = None
        self._prompt = None

    def _tokens(self, tokens):
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.shape[1] < 1:
            raise ValueError('Tokens must be a nonempty [1,tokens] tensor')
        if tokens.dtype != torch.long or tokens.device != self.model.embedding.weight.device:
            raise ValueError('Tokens must be long and on the model device')
        if not ((tokens >= 0) & (tokens < self.model.config.vocab_size)).all():
            raise ValueError('Token IDs must be within the model vocabulary')
        return tokens

    def _started(self):
        if self._state is None:
            raise RuntimeError('Start a task before ingesting observations or generating')

    @property
    def state(self):
        """A detached copy for auditing; changing it cannot alter this session."""
        self._started()
        return self._state.clone()

    @property
    def state_bytes(self):
        self._started()
        return self._state.numel() * self._state.element_size()

    def start(self, prompt_tokens):
        """Replace task state with a fresh initial prefill; freeze pointer input."""
        prompt = self._tokens(prompt_tokens).clone()
        _, state = parallel_recurrent_prefill(self.model, prompt, prompt)
        if state.shape != (1,16,64) or state.dtype != torch.float32 or not torch.isfinite(state).all():
            raise FloatingPointError('Invalid initial session state')
        self._prompt, self._state = prompt, state

    def ingest(self, new_tokens):
        """Consume a new chunk once; return transient final logits to the caller."""
        self._started()
        logits, state = parallel_recurrent_ingest(self.model, self._tokens(new_tokens), self._state, self._prompt)
        self._state = state
        return logits

    @torch.no_grad()
    def generate(self, prefix_tokens, *, eos_id, max_new_tokens):
        """Greedy raw output, without repair, repetition heuristics or fake EOS.

Every selected token, including EOS, enters state exactly once. A token-limit
stop consumes its final emitted token but does not insert a synthetic terminator.
The caller must distinguish truncated output before executing an action.
"""
        self._started()
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError('max_new_tokens must be a positive integer')
        if type(eos_id) is not int or not 0 <= eos_id < self.model.config.vocab_size:
            raise ValueError('EOS must be a model vocabulary ID')
        logits = self.ingest(prefix_tokens)
        generated = []
        for _ in range(max_new_tokens):
            token = logits.argmax(dim=-1)
            token_id = int(token.item())
            next_logits, next_state = self.model.step(token, self._state, self._prompt)
            if not torch.isfinite(next_logits).all() or not torch.isfinite(next_state).all():
                raise FloatingPointError('Nonfinite generated transition')
            self._state = next_state
            if token_id == eos_id:
                return SessionGeneration(tuple(generated), 'eos', self.state_bytes)
            generated.append(token_id)
            logits = next_logits
        return SessionGeneration(tuple(generated), 'token_limit', self.state_bytes)
