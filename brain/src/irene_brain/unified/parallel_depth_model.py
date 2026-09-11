"""Experimental deep affine recurrence with a fixed 16 x 64 float32 state.

Eight layers use eight value slots and eight numerical-residual slots. Each
layer scans time in parallel; layer depth is sequential. No token history is
stored. An optional immutable task prompt belongs to the caller, not the state.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from irene_brain.parallel.triton_scan import triton_scan


@dataclass(frozen=True)
class ParallelDepthConfig:
    vocab_size: int = 32000
    width: int = 384
    layers: int = 8
    state_width: int = 64
    ff_multiplier: int = 4
    pointer: bool = True


class RMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class AffineDepthLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d, w = config.width, config.state_width
        self.input_norm = RMSNorm(d)
        self.gates = nn.Linear(d, w)
        self.candidate = nn.Linear(d, w)
        self.state_out = nn.Linear(w, d, bias=False)
        self.ff_norm = RMSNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * config.ff_multiplier), nn.GELU(),
                                nn.Linear(d * config.ff_multiplier, d))
        # Learnable initial time scales, not hard retention floors.
        retention = 1 - torch.logspace(math.log10(.5), math.log10(.001), w)
        with torch.no_grad():
            self.gates.bias.copy_(torch.logit(retention))
        self.residual_scale = 1 / math.sqrt(2 * config.layers)

    def coefficients(self, x):
        u = self.input_norm(x)
        a = torch.sigmoid(self.gates(u))
        return a, (1 - a) * torch.tanh(self.candidate(u))

    def output(self, x, h):
        x = x + self.residual_scale * self.state_out(h)
        return x + self.residual_scale * self.ff(self.ff_norm(x))


class ParallelDepthModel(nn.Module):
    """Unpromoted language/reasoning candidate; not the existing POMDP adapter."""
    architecture = 'parallel_depth_v1'

    def __init__(self, config=ParallelDepthConfig()):
        super().__init__()
        if config.layers != 8 or config.state_width != 64:
            raise ValueError('This candidate requires eight logical 64-wide layers')
        if config.width < 1 or config.ff_multiplier < 1 or config.vocab_size < 2:
            raise ValueError('Invalid model dimensions')
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        nn.init.normal_(self.embedding.weight, std=.02)
        self.layers = nn.ModuleList([AffineDepthLayer(config) for _ in range(config.layers)])
        self.output_norm = RMSNorm(config.width)
        if config.pointer:
            self.ptr_q = nn.Linear(config.width, 64, bias=False)
            self.ptr_k = nn.Linear(config.width, 64, bias=False)
            self.ptr_gate = nn.Linear(config.width, 1)
            self.ptr_seq_boost = nn.Parameter(torch.tensor(15.))
            nn.init.constant_(self.ptr_gate.bias, -2.)
        self.double()
        if sum(p.numel() for p in self.parameters()) >= 36_000_000:
            raise ValueError('Candidate exceeds the 36M parameter ceiling')

    def init_state(self, batch_size, device=None):
        return torch.zeros(batch_size, 16, 64, dtype=torch.float32,
                           device=device or self.embedding.weight.device)

    def features_parallel(self, token_ids, reset_mask=None):
        if self.embedding.weight.dtype != torch.float64:
            raise ValueError('Strict parity candidate requires float64 computation')
        x = self.embedding(token_ids)
        for layer in self.layers:
            a, b = layer.coefficients(x)
            if reset_mask is not None:
                a = a * (1 - reset_mask.to(a.dtype).unsqueeze(-1))
            h = triton_scan(a, b)
            x = layer.output(x, h)
        return x

    def readout(self, features, token_ids, prompt_tokens=None, pointer_mask=None):
        logits = F.linear(self.output_norm(features), self.embedding.weight)
        if not self.config.pointer or prompt_tokens is None:
            return logits
        if prompt_tokens.ndim != 2 or prompt_tokens.shape[1] == 0:
            raise ValueError('Prompt must be a nonempty [batch, positions] tensor')
        # Both training and inference use independent rows; no attention recurrence.
        keys = self.ptr_k(self.embedding(prompt_tokens))
        scores = torch.einsum('btd,bnd->btn', self.ptr_q(features), keys) / 8
        valid = prompt_tokens != 0
        predecessor = torch.zeros_like(scores, dtype=torch.bool)
        predecessor[..., 1:] = ((token_ids[..., None] == prompt_tokens[:, None, :-1])
                                & valid[:, None, :-1] & valid[:, None, 1:])
        scores = scores + self.ptr_seq_boost * predecessor
        scores = scores.masked_fill(~valid[:, None], -1e9)
        attention = scores.softmax(-1) * valid[:, None]
        if pointer_mask is not None:
            attention = attention * pointer_mask[..., None]
        boost = attention * torch.sigmoid(self.ptr_gate(features)) * 25
        return logits.scatter_add(-1, prompt_tokens[:, None].expand(-1, token_ids.shape[1], -1), boost)

    def forward(self, token_ids, reset_mask=None, prompt_tokens=None, pointer_mask=None):
        features = self.features_parallel(token_ids, reset_mask)
        return self.readout(features, token_ids, prompt_tokens, pointer_mask)

    def step(self, token_ids, state, prompt_tokens=None):
        if state.shape != (token_ids.shape[0], 16, 64) or state.dtype != torch.float32:
            raise ValueError('Fast state must be [batch, 16, 64] float32')
        if self.embedding.weight.dtype != torch.float64:
            raise ValueError('Strict parity candidate requires float64 computation')
        x = self.embedding(token_ids)
        updated = []
        for index, layer in enumerate(self.layers):
            previous = state[:, index].double() + state[:, index + 8].double()
            a, b = layer.coefficients(x)
            h = a * previous + b
            updated.append(h)
            x = layer.output(x, h)
        values = torch.stack(updated, dim=1)
        high = values.float()
        low = (values - high.double()).float()
        next_state = torch.cat((high, low), dim=1)
        logits = self.readout(x[:, None], token_ids[:, None], prompt_tokens)[:, 0]
        return logits, next_state

    def language_loss(self, token_ids, labels, reset_mask=None, prompt_tokens=None,
                      pointer_mask=None, chunk_size=256):
        """Exact summed-token objective with recomputed, bounded vocabulary logits.

        Chunking changes readout workspace, not recurrent context or loss weights.
        All nonignored labels are next-token targets, as emitted by SequencePacker.
        """
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        count = (labels != -100).sum()
        if count.item() == 0:
            raise ValueError('No supervised tokens')
        features = self.features_parallel(token_ids, reset_mask)
        total = features.new_zeros(())
        for start in range(0, token_ids.shape[1], chunk_size):
            end = start + chunk_size
            local_mask = None if pointer_mask is None else pointer_mask[:, start:end]
            def loss_piece(hidden, ids, targets, mask=local_mask):
                logits = self.readout(hidden, ids, prompt_tokens, mask)
                return F.cross_entropy(logits.flatten(0, 1), targets.flatten(),
                                       ignore_index=-100, reduction='sum')
            total = total + checkpoint(loss_piece, features[:, start:end], token_ids[:, start:end],
                                       labels[:, start:end], use_reentrant=False)
        return total / count

    def checkpoint_payload(self):
        return {'architecture': self.architecture, 'config': asdict(self.config),
                'model_state_dict': self.state_dict()}

    @classmethod
    def from_payload(cls, payload):
        if payload.get('architecture') != cls.architecture:
            raise ValueError('Unsupported architecture')
        model = cls(ParallelDepthConfig(**payload['config']))
        model.load_state_dict(payload['model_state_dict'], strict=True)
        return model
