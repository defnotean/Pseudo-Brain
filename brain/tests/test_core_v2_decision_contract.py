"""Mandatory CPU contracts for the corrected Core V2 deployed action path."""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from irene_brain.v2 import CONFIG_A_DECISION_ONLY, CONFIG_C_FULL, CoreV2Config, CoreV2Model
from irene_brain.v2.losses import deployed_decision_loss


def _color_batch(batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(1702)
    observations = torch.rand(batch_size, 3, 32, 32, generator=generator) * 0.05
    targets = torch.arange(batch_size, dtype=torch.long).remainder(2) * 2
    red = targets.eq(2).float().view(-1, 1, 1)
    blue = targets.eq(0).float().view(-1, 1, 1)
    observations[:, 0] += red * 0.9
    observations[:, 2] += blue * 0.9
    return observations, targets


class CoreV2DecisionContract(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_legacy_scalar_utility_reproduces_zero_gradient_failure(self) -> None:
        torch.manual_seed(42)
        config = CoreV2Config(decision_aggregation="legacy_scalar_utility_v0")
        model = CoreV2Model(config=config, flags=CONFIG_A_DECISION_ONLY)
        observations, targets = _color_batch(4)
        output, _ = model(observations, model.init_state(4, torch.device("cpu")))
        loss = deployed_decision_loss(output, targets)
        loss.backward()

        self.assertTrue(torch.equal(output.decision.action_values, torch.zeros_like(output.decision.action_values)))
        self.assertEqual(
            sum(int(p.grad is not None and bool(torch.any(p.grad != 0))) for p in model.parameters()),
            0,
        )

    def test_direct_mean_has_nonzero_end_to_end_gradient(self) -> None:
        torch.manual_seed(42)
        model = CoreV2Model(flags=CONFIG_A_DECISION_ONLY)
        observations, targets = _color_batch(4)
        output, _ = model(observations, model.init_state(4, torch.device("cpu")))
        loss = deployed_decision_loss(output, targets)
        loss.backward()

        self.assertGreater(model.degenerate_action_head.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.encoder.conv1.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.thought_field.brain_cell.in_proj.weight.grad.abs().sum().item(), 0.0)

    def test_direct_mean_fixed_batch_learning_is_stable(self) -> None:
        torch.manual_seed(42)
        model = CoreV2Model(flags=CONFIG_C_FULL)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        observations, targets = _color_batch()
        losses = []

        for step in range(60):
            torch.manual_seed(10_000 + step)
            output, _ = model(observations, model.init_state(8, torch.device("cpu")))
            loss = deployed_decision_loss(output, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        self.assertTrue(all(torch.isfinite(torch.tensor(losses))))
        self.assertLess(sum(losses[-10:]) / 10, 0.20)
        self.assertLess(sum(losses[-10:]) / 10, (sum(losses[:10]) / 10) * 0.20)

    def test_direct_mean_is_slot_permutation_invariant(self) -> None:
        torch.manual_seed(7)
        model = CoreV2Model(flags=CONFIG_C_FULL).eval()
        observations, _ = _color_batch(2)
        output, _ = model(observations, model.init_state(2, torch.device("cpu")))
        hypotheses = output.hypotheses
        permutation = torch.randperm(hypotheses.action_logits.shape[1])
        permuted = type(hypotheses)(**{
            field: getattr(hypotheses, field)[:, permutation]
            for field in hypotheses.__dataclass_fields__
        })
        decision = model.aggregator(hypotheses)
        permuted_decision = model.aggregator(permuted)
        self.assertTrue(torch.allclose(decision.action_values, permuted_decision.action_values, atol=1e-7, rtol=0.0))

    def test_normalized_recurrences_stay_bounded_on_long_sequences(self) -> None:
        torch.manual_seed(42)
        config = CoreV2Config(
            braincell_dynamics="normalized_mixture_v1",
            belief_dynamics="convex_gated_v1",
        )
        model = CoreV2Model(config=config, flags=CONFIG_C_FULL)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
        observations, targets = _color_batch(4)

        for step in range(12):
            torch.manual_seed(20_000 + step)
            state = model.init_state(4, torch.device("cpu"))
            output = None
            for _ in range(25):
                output, state = model(observations, state)
                self.assertTrue(torch.isfinite(output.thoughts).all())
                self.assertTrue(torch.isfinite(output.belief).all())
                self.assertLessEqual(output.belief.abs().max().item(), 1.0 + 1e-6)
                self.assertLessEqual(output.thoughts.abs().max().item(), 11.0)

            loss = deployed_decision_loss(output, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.assertTrue(all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            self.assertTrue(torch.isfinite(norm))
            optimizer.step()

    def test_deployed_loss_accepts_exact_macro_class_weights(self) -> None:
        torch.manual_seed(9)
        model = CoreV2Model(flags=CONFIG_A_DECISION_ONLY)
        observations, targets = _color_batch(5)
        targets = torch.arange(5, dtype=torch.long)
        output, _ = model(observations, model.init_state(5, torch.device("cpu")))
        weights = torch.tensor([6.2666667, 0.6372881, 0.5611940, 1.3549550, 1.3309735])
        actual = deployed_decision_loss(output, targets, weights)
        expected = torch.nn.functional.nll_loss(
            torch.log_softmax(output.decision.action_values, dim=-1),
            targets,
            weight=weights,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_fixed_exposure_mean_does_not_renormalize_each_batch(self) -> None:
        torch.manual_seed(11)
        model = CoreV2Model(flags=CONFIG_A_DECISION_ONLY)
        observations, _ = _color_batch(5)
        targets = torch.tensor([0, 1, 1, 1, 1], dtype=torch.long)
        output, _ = model(observations, model.init_state(5, torch.device("cpu")))
        weights = torch.tensor([6.2666667, 0.6372881, 0.5611940, 1.3549550, 1.3309735])
        actual = deployed_decision_loss(
            output,
            targets,
            weights,
            weight_normalization="fixed_exposure_mean_v1",
        )
        per_sample = torch.nn.functional.nll_loss(
            torch.log_softmax(output.decision.action_values, dim=-1),
            targets,
            reduction="none",
        )
        expected = (per_sample * weights[targets]).mean()
        legacy = deployed_decision_loss(output, targets, weights)
        self.assertTrue(torch.equal(actual, expected))
        self.assertFalse(torch.equal(actual, legacy))

    def test_unknown_weight_normalization_fails_closed(self) -> None:
        torch.manual_seed(13)
        model = CoreV2Model(flags=CONFIG_A_DECISION_ONLY)
        observations, targets = _color_batch(2)
        output, _ = model(observations, model.init_state(2, torch.device("cpu")))
        with self.assertRaisesRegex(ValueError, "unknown deployed-loss"):
            deployed_decision_loss(
                output,
                targets,
                torch.ones(5),
                weight_normalization="not-a-loss-contract",
            )

    def test_fixed_exposure_mean_preserves_turn_sample_weights(self) -> None:
        torch.manual_seed(17)
        model = CoreV2Model(flags=CONFIG_A_DECISION_ONLY)
        observations, _ = _color_batch(5)
        targets = torch.arange(5, dtype=torch.long)
        output, _ = model(observations, model.init_state(5, torch.device("cpu")))
        sample_weights = torch.tensor([0.1, 1.0, 0.1, 1.0, 0.1])
        actual = deployed_decision_loss(
            output,
            targets,
            weight_normalization="fixed_exposure_mean_v1",
            sample_weights=sample_weights,
        )
        per_sample = torch.nn.functional.nll_loss(
            torch.log_softmax(output.decision.action_values, dim=-1),
            targets,
            reduction="none",
        )
        self.assertTrue(torch.equal(actual, (per_sample * sample_weights).mean()))


if __name__ == "__main__":
    unittest.main()
