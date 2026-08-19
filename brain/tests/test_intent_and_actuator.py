from __future__ import annotations

import unittest
from dataclasses import replace

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.model.intent import (
        NUM_INTENTS,
        DirectionalAction,
        DirectionalActionHead,
        DirectionalActionPrediction,
        IntentEncoder,
        IntentKind,
        LatentIntentHead,
        LatentIntentPrediction,
        LatentIntentPredictor,
    )
    from irene_brain.model.spec import ThoughtFieldConfig
    from irene_brain.model.torch_model import IreneBrainModel
    from irene_brain.types import HidKey


def tiny_config() -> ThoughtFieldConfig:
    """Return tiny topology-faithful configuration for CPU tests."""
    return replace(
        ThoughtFieldConfig.smoke(),
        core_width=16,
        sensor_tokens=4,
        belief_tokens=2,
        working_memory_tokens=1,
        thoughtlets=4,
        goal_context_tokens=2,
        cognitive_cycles=2,
        brain_cell_blocks=1,
        attention_heads=2,
        routed_neighbors=1,
        episodic_memory_entries=8,
        retrieved_entries_per_thoughtlet=1,
    )


@unittest.skipUnless(torch is not None, "PyTorch is required for intent and actuator tests")
class IntentAndActuatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        assert torch is not None
        torch.set_num_threads(1)
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass

    def setUp(self) -> None:
        assert torch is not None
        torch.manual_seed(42)

    def test_intent_kind_enum_and_string_parsing(self) -> None:
        self.assertEqual(int(IntentKind.UNSTICK), 0)
        self.assertEqual(int(IntentKind.EVADE), 1)
        self.assertEqual(int(IntentKind.NAVIGATE), 2)
        self.assertEqual(int(IntentKind.EXPLORE), 3)
        self.assertEqual(int(IntentKind.NEUTRAL), 4)
        self.assertEqual(NUM_INTENTS, 5)
        self.assertEqual(IntentKind.count(), 5)

        self.assertEqual(IntentKind.from_string("[UNSTICK]"), IntentKind.UNSTICK)
        self.assertEqual(IntentKind.from_string("EVADE"), IntentKind.EVADE)
        self.assertEqual(IntentKind.from_string("navigate"), IntentKind.NAVIGATE)
        self.assertEqual(IntentKind.from_string("[explore]"), IntentKind.EXPLORE)
        self.assertEqual(IntentKind.from_string("NEUTRAL"), IntentKind.NEUTRAL)

        with self.assertRaises(ValueError):
            IntentKind.from_string("UNKNOWN_INTENT")

    def test_intent_encoder_discrete_and_continuous(self) -> None:
        assert torch is not None
        width = 16
        tokens = 8
        encoder = IntentEncoder(width=width, tokens=tokens, num_intents=NUM_INTENTS)

        # 1. Single IntentKind / int
        gc_single = encoder(IntentKind.UNSTICK)
        self.assertEqual(gc_single.shape, (1, tokens, width))

        # 2. Sequence of IntentKinds
        gc_seq = encoder([IntentKind.UNSTICK, IntentKind.NAVIGATE, IntentKind.EXPLORE])
        self.assertEqual(gc_seq.shape, (3, tokens, width))

        # 3. 1D Tensor of integer IDs
        tensor_ids = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        gc_tensor = encoder(tensor_ids)
        self.assertEqual(gc_tensor.shape, (5, tokens, width))

        # 4. 2D Tensor of shape [batch, 1]
        gc_2d = encoder(tensor_ids.unsqueeze(1))
        self.assertEqual(gc_2d.shape, (5, tokens, width))
        self.assertTrue(torch.equal(gc_tensor, gc_2d))

        # 5. One-hot continuous vector should match discrete integer output
        one_hot = F.one_hot(tensor_ids, num_classes=NUM_INTENTS).float()
        gc_one_hot = encoder(one_hot)
        self.assertEqual(gc_one_hot.shape, (5, tokens, width))
        self.assertTrue(torch.allclose(gc_tensor, gc_one_hot, atol=1e-5))

        # 6. Soft probability distribution
        probs = torch.softmax(torch.randn(4, NUM_INTENTS), dim=-1)
        gc_soft = encoder(probs)
        self.assertEqual(gc_soft.shape, (4, tokens, width))
        self.assertTrue(torch.isfinite(gc_soft).all())

        # 7. Out of range validation
        with self.assertRaises(ValueError):
            encoder(torch.tensor([5], dtype=torch.long))
        with self.assertRaises(ValueError):
            encoder(torch.tensor([-1], dtype=torch.long))

    def test_latent_intent_head_prediction(self) -> None:
        assert torch is not None
        batch = 3
        thoughtlets = 4
        registers = 3
        width = 16
        belief_tokens = 2

        head = LatentIntentHead(width=width, num_intents=NUM_INTENTS, heads=2)
        thoughts = torch.randn(batch, thoughtlets, registers, width, requires_grad=True)
        belief = torch.randn(batch, belief_tokens, width, requires_grad=True)

        prediction = head(thoughts=thoughts, belief=belief)
        self.assertIsInstance(prediction, LatentIntentPrediction)
        self.assertEqual(prediction.logits.shape, (batch, NUM_INTENTS))
        self.assertEqual(prediction.probabilities.shape, (batch, NUM_INTENTS))
        self.assertEqual(prediction.predicted_intent.shape, (batch,))
        self.assertIsNotNone(prediction.attention_weights)

        # Probabilities sum to 1.0
        prob_sum = prediction.probabilities.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sum, torch.ones(batch), atol=1e-5))

        # Predicted intent matches argmax of logits
        self.assertTrue(torch.equal(prediction.predicted_intent, prediction.logits.argmax(dim=-1)))

        # Gradients flow back to inputs
        target = torch.tensor([0, 2, 4], dtype=torch.long)
        loss = F.cross_entropy(prediction.logits, target)
        loss.backward()
        self.assertIsNotNone(thoughts.grad)
        self.assertIsNotNone(belief.grad)
        assert thoughts.grad is not None
        assert belief.grad is not None
        self.assertTrue(torch.isfinite(thoughts.grad).all())
        self.assertTrue(torch.isfinite(belief.grad).all())

    def test_latent_intent_predictor_end_to_end(self) -> None:
        assert torch is not None
        batch = 2
        thoughtlets = 4
        registers = 3
        width = 16
        tokens = 8

        predictor = LatentIntentPredictor(
            width=width,
            tokens=tokens,
            num_intents=NUM_INTENTS,
            heads=2,
        )
        thoughts = torch.randn(batch, thoughtlets, registers, width)
        belief = torch.randn(batch, 2, width)

        # Soft routing (default)
        goal_context_soft, pred_soft = predictor(thoughts, belief, hard=False)
        self.assertEqual(goal_context_soft.shape, (batch, tokens, width))
        self.assertTrue(torch.isfinite(goal_context_soft).all())

        # Hard discrete routing
        goal_context_hard, pred_hard = predictor(thoughts, belief, hard=True)
        self.assertEqual(goal_context_hard.shape, (batch, tokens, width))
        self.assertTrue(torch.isfinite(goal_context_hard).all())

    def test_directional_action_head_shape_and_mapping(self) -> None:
        assert torch is not None
        batch = 4
        width = 16
        head = DirectionalActionHead(width=width)

        # 2D features [batch, width]
        features = torch.randn(batch, width)
        pred = head(features)
        self.assertIsInstance(pred, DirectionalActionPrediction)
        self.assertEqual(pred.logits.shape, (batch, 5))
        self.assertEqual(pred.probabilities.shape, (batch, 5))
        self.assertEqual(pred.action.shape, (batch,))
        self.assertEqual(pred.wasd_binary.shape, (batch, 4))

        # 3D features [batch, seq, width]
        features_3d = torch.randn(batch, 10, width)
        pred_3d = head(features_3d)
        self.assertEqual(pred_3d.logits.shape, (batch, 5))
        self.assertEqual(pred_3d.wasd_binary.shape, (batch, 4))

        # Full keyboard tensor mapping [batch, 256]
        keyboard = head.to_keyboard_tensor(pred.logits)
        self.assertEqual(keyboard.shape, (batch, 256))

        for i in range(batch):
            action = int(pred.action[i])
            if action == DirectionalAction.NONE:
                self.assertEqual(float(keyboard[i].sum()), 0.0)
            elif action == DirectionalAction.W:
                self.assertEqual(float(keyboard[i, int(HidKey.W)]), 1.0)
                self.assertEqual(float(keyboard[i].sum()), 1.0)
            elif action == DirectionalAction.A:
                self.assertEqual(float(keyboard[i, int(HidKey.A)]), 1.0)
                self.assertEqual(float(keyboard[i].sum()), 1.0)
            elif action == DirectionalAction.S:
                self.assertEqual(float(keyboard[i, int(HidKey.S)]), 1.0)
                self.assertEqual(float(keyboard[i].sum()), 1.0)
            elif action == DirectionalAction.D:
                self.assertEqual(float(keyboard[i, int(HidKey.D)]), 1.0)
                self.assertEqual(float(keyboard[i].sum()), 1.0)

    def test_directional_mutual_exclusion(self) -> None:
        """Mechanically verify that W+S and A+D can NEVER be simultaneously active."""
        assert torch is not None
        width = 16
        head = DirectionalActionHead(width=width)

        # 1. Adversarial edge cases: conflicting large logits for opposite directions
        # Case: W=100.0, S=99.0 (W wins, S must be 0)
        conflicting_ws = torch.tensor([[0.0, 100.0, 0.0, 99.0, 0.0]])
        wasd_ws = head.to_wasd_binary(conflicting_ws)
        self.assertEqual(wasd_ws[0, 0].item(), 1.0)  # W is active
        self.assertEqual(wasd_ws[0, 2].item(), 0.0)  # S is NOT active
        self.assertEqual((wasd_ws[0, 0] * wasd_ws[0, 2]).item(), 0.0)

        # Case: A=100.0, D=99.0 (A wins, D must be 0)
        conflicting_ad = torch.tensor([[0.0, 0.0, 100.0, 0.0, 99.0]])
        wasd_ad = head.to_wasd_binary(conflicting_ad)
        self.assertEqual(wasd_ad[0, 1].item(), 1.0)  # A is active
        self.assertEqual(wasd_ad[0, 3].item(), 0.0)  # D is NOT active
        self.assertEqual((wasd_ad[0, 1] * wasd_ad[0, 3]).item(), 0.0)

        # Case: NONE is largest -> all WASD are 0
        none_logits = torch.tensor([[100.0, 10.0, 10.0, 10.0, 10.0]])
        wasd_none = head.to_wasd_binary(none_logits)
        self.assertEqual(float(wasd_none.sum()), 0.0)

        # 2. Stress test with 10,000 random logit batches
        random_logits = torch.randn(10000, 5) * 20.0
        wasd_random = head.to_wasd_binary(random_logits)

        # Invariant 1: W and S are NEVER simultaneously 1
        ws_conflict = (wasd_random[:, 0] > 0.5) & (wasd_random[:, 2] > 0.5)
        self.assertEqual(int(ws_conflict.sum()), 0)

        # Invariant 2: A and D are NEVER simultaneously 1
        ad_conflict = (wasd_random[:, 1] > 0.5) & (wasd_random[:, 3] > 0.5)
        self.assertEqual(int(ad_conflict.sum()), 0)

        # Invariant 3: sum of active directions per sample is <= 1.0
        self.assertTrue((wasd_random.sum(dim=-1) <= 1.0).all())

        # 3. Gumbel-Softmax sampling maintains mutual exclusion
        sampled = head.sample_wasd(random_logits[:100], temperature=1.0, hard=True)
        self.assertEqual(int(((sampled[:, 0] > 0.5) & (sampled[:, 2] > 0.5)).sum()), 0)
        self.assertEqual(int(((sampled[:, 1] > 0.5) & (sampled[:, 3] > 0.5)).sum()), 0)
        self.assertTrue((sampled.sum(dim=-1) <= 1.0).all())

    def test_model_integration_with_goal_context(self) -> None:
        assert torch is not None
        config = tiny_config()
        model = IreneBrainModel(config, input_resolution=(8, 8), plan_steps=2).cpu()
        model.eval()

        encoder = IntentEncoder(
            width=config.core_width,
            tokens=config.goal_context_tokens,
            num_intents=NUM_INTENTS,
        )

        pixels = torch.rand(1, 3, 8, 8)
        previous_control = torch.zeros(1, config.actuator.total_queries)
        elapsed_seconds = torch.tensor([1.0 / 30.0])

        # Encode specific UNSTICK intent into goal_context
        goal_context = encoder(IntentKind.UNSTICK)
        self.assertEqual(goal_context.shape, (1, config.goal_context_tokens, config.core_width))

        # Set specific goal_context on BrainState
        state = model.initial_state(1)
        state = replace(state, goal_context=goal_context)
        self.assertTrue(torch.equal(state.goal_context, goal_context))

        # Forward with explicit goal_context
        with torch.no_grad():
            output = model(
                pixels,
                previous_control,
                elapsed_seconds,
                state=state,
            )
        self.assertEqual(output.action.control.shape, (1, config.actuator.total_queries))
        self.assertEqual(output.next_state.goal_context.shape, (1, config.goal_context_tokens, config.core_width))


if __name__ == "__main__":
    unittest.main()
