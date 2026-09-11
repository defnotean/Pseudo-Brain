"""Small causal transformer control for the broad learning experiment.

This baseline is allowed attention over its full prefix; it is not a 4 KB model.
Float32 parameters, bfloat16 Flash attention, six 384-wide layers, tied readout.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from irene_brain.unified.parallel_depth_model import ParallelDepthConfig, ParallelDepthModel, RMSNorm


class CausalLayer(nn.Module):
    def __init__(self, width, layers=6):
        super().__init__()
        if width % 64:
            raise ValueError('Width must be divisible by 64')
        self.heads = width // 64
        self.norm = RMSNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.projection = nn.Linear(width, width, bias=False)
        self.ff_norm = RMSNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))
        self.scale = 1 / math.sqrt(2 * layers)
        self.register_buffer('frequencies', 10000 ** (-torch.arange(0, 64, 2).float() / 64), persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm(x)).view(batch, length, 3, self.heads, 64).unbind(2)
        angle = torch.arange(length, device=x.device).float()[:, None] * self.frequencies[None]
        cosine, sine = angle.cos()[None, :, None], angle.sin()[None, :, None]
        def rotate(t):
            even, odd = t[..., ::2], t[..., 1::2]
            return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), -1).flatten(-2)
        q, k = rotate(q), rotate(k)
        if x.is_cuda:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attention = F.scaled_dot_product_attention(q.transpose(1, 2).to(torch.bfloat16),
                    k.transpose(1, 2).to(torch.bfloat16), v.transpose(1, 2).to(torch.bfloat16),
                    dropout_p=0., is_causal=True).to(x.dtype)
        else:
            attention = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                v.transpose(1, 2), dropout_p=0., is_causal=True)
        x = x + self.scale * self.projection(attention.transpose(1, 2).reshape(batch, length, width))
        return x + self.scale * self.ff(self.ff_norm(x))


class ComparisonTransformer(ParallelDepthModel):
    architecture = 'comparison_transformer_v1'

    def __init__(self, config=ParallelDepthConfig()):
        super().__init__(config)
        self.layers = nn.ModuleList([CausalLayer(config.width) for _ in range(6)])
        self.float()
        if sum(p.numel() for p in self.parameters()) >= 36_000_000:
            raise ValueError('Comparison model exceeds parameter ceiling')

    def features_parallel(self, token_ids, reset_mask=None):
        if reset_mask is not None and bool(reset_mask[:, 1:].any()):
            raise ValueError('Transformer comparison uses one whole document per batch row')
        x = self.embedding(token_ids)
        for layer in self.layers:
            x = layer(x)
        return x

    def init_state(self, *args, **kwargs):
        raise NotImplementedError('Transformer comparison has no 4 KB recurrent state')

    def step(self, *args, **kwargs):
        raise NotImplementedError('Transformer comparison evaluates complete prefixes')
