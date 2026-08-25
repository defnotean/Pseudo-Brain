"""CPU contracts for symlog-space two-hot reward prediction."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from irene_brain.v2.reward_distribution import SymlogTwoHotReward, symexp


class SymlogTwoHotRewardContract(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.reward = SymlogTwoHotReward(
            num_bins=5,
            symlog_min=-2.0,
            symlog_max=2.0,
        )

    def test_targets_are_nonnegative_and_normalized(self) -> None:
        values = torch.tensor([-100.0, -2.0, -0.2, 0.0, 0.7, 5.0, 100.0])
        target = self.reward.targets(values)

        self.assertEqual(tuple(target.shape), (7, 5))
        self.assertTrue(bool((target >= 0.0).all()))
        self.assertTrue(torch.allclose(target.sum(dim=-1), torch.ones(7)))
        self.assertTrue(bool((target.ne(0.0).sum(dim=-1) <= 2).all()))

    def test_exact_bin_reward_produces_one_hot_target(self) -> None:
        value = symexp(torch.tensor(1.0))
        target = self.reward.targets(value)

        self.assertTrue(torch.allclose(target, torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])))

    def test_interpolated_reward_produces_two_hot_target(self) -> None:
        value = symexp(torch.tensor(0.5))
        target = self.reward.targets(value)

        self.assertTrue(torch.allclose(target, torch.tensor([0.0, 0.0, 0.5, 0.5, 0.0])))

    def test_out_of_support_rewards_clip_to_endpoint_bins(self) -> None:
        target = self.reward.targets(torch.tensor([-1.0e6, 1.0e6]))

        self.assertTrue(torch.equal(target[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])))
        self.assertTrue(torch.equal(target[1], torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0])))

    def test_decode_is_expected_raw_reward_not_symexp_of_mean_symlog(self) -> None:
        probabilities = torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.5]])
        logits = probabilities.clamp_min(1.0e-30).log()
        decoded = self.reward.decode(logits)
        expected_raw = 0.5 * symexp(torch.tensor(2.0))
        incorrect_transform_of_mean = symexp(torch.tensor(1.0))

        self.assertTrue(torch.allclose(decoded, expected_raw.unsqueeze(0)))
        self.assertFalse(torch.allclose(decoded, incorrect_transform_of_mean.unsqueeze(0)))
        self.assertGreater(float(decoded.item()), 0.0)

        negative_logits = torch.full((1, 5), -100.0)
        negative_logits[0, 0] = 100.0
        self.assertLess(float(self.reward.decode(negative_logits).item()), 0.0)

    def test_soft_target_cross_entropy_has_finite_nonzero_gradients(self) -> None:
        logits = torch.randn(4, 5, requires_grad=True)
        values = torch.tensor([-4.0, -0.3, 0.4, 5.0])
        loss = self.reward.loss(logits, values)
        loss.backward()

        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
