"""Distributional reward utilities using two-hot targets in symlog space."""
from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def symlog(value: Tensor) -> Tensor:
    """Bi-symmetric logarithm with linear behaviour around zero."""
    return torch.sign(value) * torch.log1p(value.abs())


def symexp(value: Tensor) -> Tensor:
    """Inverse of :func:`symlog`."""
    return torch.sign(value) * torch.expm1(value.abs())


class SymlogTwoHotReward(nn.Module):
    """Encode scalar rewards on evenly spaced symlog bins.

    Training uses two neighbouring bins and soft-target cross entropy.  Decode
    computes the expectation of each bin's value after converting that bin to
    raw reward units.  It intentionally does *not* apply ``symexp`` after
    averaging in symlog space, because that would not be an expected reward.
    """

    def __init__(
        self,
        *,
        num_bins: int,
        symlog_min: float,
        symlog_max: float,
    ) -> None:
        super().__init__()
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
            raise ValueError("num_bins must be an integer >= 2")
        if not symlog_min < symlog_max:
            raise ValueError("symlog_min must be less than symlog_max")
        bins = torch.linspace(float(symlog_min), float(symlog_max), num_bins)
        self.register_buffer("symlog_bins", bins)
        self.register_buffer("raw_bins", symexp(bins))

    @property
    def num_bins(self) -> int:
        return int(self.symlog_bins.numel())

    def targets(self, reward: Tensor) -> Tensor:
        """Return normalized two-hot targets with shape ``reward.shape + [N]``.

        Rewards outside the configured support are clipped to the nearest end
        bin.  Rewards inside the support are linearly interpolated in symlog
        space between their two neighbouring bins.
        """
        if not torch.is_floating_point(reward):
            reward = reward.to(dtype=self.symlog_bins.dtype)
        bins = self.symlog_bins.to(device=reward.device, dtype=reward.dtype)
        encoded = symlog(reward).clamp(min=bins[0], max=bins[-1])
        step = bins[1] - bins[0]
        position = (encoded - bins[0]) / step
        lower = position.floor().to(torch.long).clamp(0, self.num_bins - 1)
        upper = (lower + 1).clamp(max=self.num_bins - 1)
        upper_weight = position.sub(lower.to(position.dtype)).clamp(0.0, 1.0)
        lower_weight = 1.0 - upper_weight

        target = reward.new_zeros(reward.shape + (self.num_bins,))
        target.scatter_add_(-1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
        target.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
        return target

    def soft_cross_entropy(
        self,
        logits: Tensor,
        target: Tensor,
        *,
        reduction: str = "mean",
    ) -> Tensor:
        """Cross entropy from logits to a normalized soft target distribution."""
        if logits.shape != target.shape or logits.shape[-1] != self.num_bins:
            raise ValueError("logits and target must have equal shape ending in num_bins")
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError("reduction must be none, mean, or sum")
        per_item = -(target * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        if reduction == "none":
            return per_item
        if reduction == "sum":
            return per_item.sum()
        return per_item.mean()

    def loss(
        self,
        logits: Tensor,
        reward: Tensor,
        *,
        reduction: str = "mean",
    ) -> Tensor:
        """Encode raw scalar rewards and compute soft-target cross entropy."""
        if logits.shape[:-1] != reward.shape or logits.shape[-1] != self.num_bins:
            raise ValueError("logits must have shape reward.shape + [num_bins]")
        return self.soft_cross_entropy(
            logits,
            self.targets(reward).to(dtype=logits.dtype),
            reduction=reduction,
        )

    def decode(self, logits: Tensor) -> Tensor:
        """Decode logits to expected reward in raw environment units."""
        if logits.shape[-1] != self.num_bins:
            raise ValueError("logits must end in num_bins")
        probabilities = F.softmax(logits, dim=-1)
        raw_bins = self.raw_bins.to(device=logits.device, dtype=logits.dtype)
        return (probabilities * raw_bins).sum(dim=-1)


__all__ = ["SymlogTwoHotReward", "symlog", "symexp"]
