"""Unit tests for the Structural Action Grouping module."""

from __future__ import annotations

import math
import unittest

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]

if torch is not None:
    from irene_brain.model.action_groups import (
        CONTINUOUS_DEADZONE_LIMIT,
        CONTROL_VECTOR_SIZE,
        DEFAULT_MODIFIER_KEYS,
        DEFAULT_MOVEMENT_KEYS,
        DEFAULT_TRIGGER_KEYS,
        DEFAULT_TRIGGER_MOUSE_BUTTONS,
        MOUSE_AXIS_DX,
        MOUSE_AXIS_DY,
        MOUSE_BUTTONS_START,
        SCROLL_AXIS,
        SQUASH_LIMIT,
        MovementDirection5,
        MovementDirection9,
        MovementMode,
        StructuredActionLoss,
        StructuredActionPrediction,
        StructuredActionSpec,
        StructuredActuatorHead,
        StructuredLossOutput,
    )
    from irene_brain.training.batches import control_to_vector
    from irene_brain.types import GenericControl, HidKey


@unittest.skipUnless(torch is not None, "PyTorch is required for action group tests")
class ActionGroupsTests(unittest.TestCase):
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

    def test_movement_direction_enums_and_mappings(self) -> None:
        # 5-way enum tests
        self.assertEqual(int(MovementDirection5.NONE), 0)
        self.assertEqual(int(MovementDirection5.W), 1)
        self.assertEqual(int(MovementDirection5.A), 2)
        self.assertEqual(int(MovementDirection5.S), 3)
        self.assertEqual(int(MovementDirection5.D), 4)

        self.assertEqual(MovementDirection5.NONE.keys, ())
        self.assertEqual(MovementDirection5.W.keys, (int(HidKey.W),))
        self.assertEqual(MovementDirection5.A.keys, (int(HidKey.A),))
        self.assertEqual(MovementDirection5.S.keys, (int(HidKey.S),))
        self.assertEqual(MovementDirection5.D.keys, (int(HidKey.D),))

        self.assertEqual(MovementDirection5.W.name_lower, "w")
        self.assertEqual(MovementDirection5.NONE.name_lower, "none")

        # 5-way from_keys
        self.assertEqual(MovementDirection5.from_keys([]), MovementDirection5.NONE)
        self.assertEqual(MovementDirection5.from_keys([int(HidKey.W)]), MovementDirection5.W)
        self.assertEqual(MovementDirection5.from_keys([int(HidKey.A)]), MovementDirection5.A)
        self.assertEqual(MovementDirection5.from_keys([int(HidKey.S)]), MovementDirection5.S)
        self.assertEqual(MovementDirection5.from_keys([int(HidKey.D)]), MovementDirection5.D)
        # Conflicting/multiple keys fall back to NONE in 5-way
        self.assertEqual(
            MovementDirection5.from_keys([int(HidKey.W), int(HidKey.S)]),
            MovementDirection5.NONE,
        )

        # 9-way enum tests
        self.assertEqual(int(MovementDirection9.NONE), 0)
        self.assertEqual(int(MovementDirection9.W), 1)
        self.assertEqual(int(MovementDirection9.A), 2)
        self.assertEqual(int(MovementDirection9.S), 3)
        self.assertEqual(int(MovementDirection9.D), 4)
        self.assertEqual(int(MovementDirection9.WA), 5)
        self.assertEqual(int(MovementDirection9.WD), 6)
        self.assertEqual(int(MovementDirection9.SA), 7)
        self.assertEqual(int(MovementDirection9.SD), 8)

        self.assertEqual(MovementDirection9.WA.keys, (int(HidKey.W), int(HidKey.A)))
        self.assertEqual(MovementDirection9.WD.keys, (int(HidKey.W), int(HidKey.D)))
        self.assertEqual(MovementDirection9.SA.keys, (int(HidKey.S), int(HidKey.A)))
        self.assertEqual(MovementDirection9.SD.keys, (int(HidKey.S), int(HidKey.D)))

        # 9-way from_keys
        self.assertEqual(
            MovementDirection9.from_keys([int(HidKey.W), int(HidKey.A)]),
            MovementDirection9.WA,
        )
        self.assertEqual(
            MovementDirection9.from_keys([int(HidKey.W), int(HidKey.D)]),
            MovementDirection9.WD,
        )
        self.assertEqual(
            MovementDirection9.from_keys([int(HidKey.S), int(HidKey.A)]),
            MovementDirection9.SA,
        )
        self.assertEqual(
            MovementDirection9.from_keys([int(HidKey.S), int(HidKey.D)]),
            MovementDirection9.SD,
        )
        # Opposite conflict in 9-way falls back to NONE
        self.assertEqual(
            MovementDirection9.from_keys([int(HidKey.W), int(HidKey.S)]),
            MovementDirection9.NONE,
        )
        self.assertEqual(
            MovementDirection9.from_keys([int(HidKey.A), int(HidKey.D)]),
            MovementDirection9.NONE,
        )

    def test_structured_action_spec_validation(self) -> None:
        spec5 = StructuredActionSpec(width=16, movement_mode="5way")
        self.assertEqual(spec5.num_movement_classes, 5)
        self.assertEqual(spec5.num_modifiers, len(DEFAULT_MODIFIER_KEYS))
        self.assertEqual(
            spec5.num_triggers,
            len(DEFAULT_TRIGGER_KEYS) + len(DEFAULT_TRIGGER_MOUSE_BUTTONS),
        )
        self.assertEqual(spec5.num_continuous, 3)  # 2 aim + 1 scroll

        spec9 = StructuredActionSpec(
            width=32,
            movement_mode="9way",
            include_scroll=True,
            include_gamepad_axes=True,
        )
        self.assertEqual(spec9.num_movement_classes, 9)
        self.assertEqual(spec9.num_continuous, 11)  # 2 aim + 1 scroll + 8 gamepad

        # Validation errors
        with self.assertRaises(ValueError):
            StructuredActionSpec(width=0)
        with self.assertRaises(ValueError):
            StructuredActionSpec(width=16, movement_mode="invalid_mode")
        with self.assertRaises(ValueError):
            StructuredActionSpec(width=16, movement_keys=(26, 4, 22))  # need 4 keys
        with self.assertRaises(ValueError):
            StructuredActionSpec(width=16, movement_keys=(26, 4, 22, 26))  # duplicate
        with self.assertRaises(ValueError):
            # Overlapping movement and modifier
            StructuredActionSpec(width=16, modifier_keys=(26, 44))
        with self.assertRaises(ValueError):
            # Overlapping modifier and trigger
            StructuredActionSpec(width=16, modifier_keys=(44,), trigger_keys=(44,))
        with self.assertRaises(ValueError):
            StructuredActionSpec(width=16, continuous_squash="unknown_squash")
        with self.assertRaises(ValueError):
            StructuredActionSpec(width=16, continuous_deadzone=-0.01)

    def test_structured_actuator_head_unpooled_forward(self) -> None:
        assert torch is not None
        batch = 3
        width = 16
        spec = StructuredActionSpec(width=width, movement_mode="5way")
        head = StructuredActuatorHead(width=width, spec=spec, heads=2)

        sensors = torch.randn(batch, 4, width)
        belief = torch.randn(batch, 2, width)
        thoughts = torch.randn(batch, 4, 3, width)
        working_memory = torch.randn(batch, 1, width)
        retrieved_memory = torch.randn(batch, 4, 2, width)
        goal_context = torch.randn(batch, 2, width)

        pred = head(
            sensors=sensors,
            belief=belief,
            thoughts=thoughts,
            working_memory=working_memory,
            retrieved_memory=retrieved_memory,
            goal_context=goal_context,
        )

        self.assertIsInstance(pred, StructuredActionPrediction)
        self.assertEqual(pred.movement_logits.shape, (batch, 5))
        self.assertEqual(pred.movement_probs.shape, (batch, 5))
        self.assertEqual(pred.movement_action.shape, (batch,))
        self.assertEqual(pred.modifier_logits.shape, (batch, spec.num_modifiers))
        self.assertEqual(pred.modifier_probs.shape, (batch, spec.num_modifiers))
        self.assertEqual(pred.trigger_logits.shape, (batch, spec.num_triggers))
        self.assertEqual(pred.trigger_probs.shape, (batch, spec.num_triggers))
        self.assertEqual(pred.mouse_aim.shape, (batch, 2))
        self.assertIsNotNone(pred.scroll)
        assert pred.scroll is not None
        self.assertEqual(pred.scroll.shape, (batch, 1))
        self.assertEqual(pred.control_vector.shape, (batch, CONTROL_VECTOR_SIZE))
        self.assertIsNotNone(pred.query_features)
        self.assertIsNotNone(pred.state_attention)
        self.assertIsNotNone(pred.thought_attention)

        # Probabilities sum to 1 for categorical movement
        self.assertTrue(
            torch.allclose(pred.movement_probs.sum(dim=-1), torch.ones(batch), atol=1e-5)
        )
        # Bernoulli probabilities are in [0, 1]
        self.assertTrue((pred.modifier_probs >= 0.0).all() and (pred.modifier_probs <= 1.0).all())
        self.assertTrue((pred.trigger_probs >= 0.0).all() and (pred.trigger_probs <= 1.0).all())

    def test_structured_actuator_head_9way_forward(self) -> None:
        assert torch is not None
        batch = 2
        width = 16
        spec = StructuredActionSpec(
            width=width,
            movement_mode="9way",
            include_scroll=True,
            include_gamepad_axes=True,
        )
        head = StructuredActuatorHead(width=width, spec=spec, heads=2)

        features_3d = torch.randn(batch, 8, width)
        pred = head(features=features_3d)

        self.assertEqual(pred.movement_logits.shape, (batch, 9))
        self.assertEqual(pred.movement_probs.shape, (batch, 9))
        self.assertEqual(pred.movement_action.shape, (batch,))
        self.assertIsNotNone(pred.gamepad_axes)
        assert pred.gamepad_axes is not None
        self.assertEqual(pred.gamepad_axes.shape, (batch, 8))
        self.assertEqual(pred.control_vector.shape, (batch, CONTROL_VECTOR_SIZE))

    def test_structured_actuator_head_2d_features(self) -> None:
        assert torch is not None
        batch = 4
        width = 16
        head = StructuredActuatorHead(width=width)
        features_2d = torch.randn(batch, width)
        pred = head(features=features_2d)

        self.assertEqual(pred.movement_logits.shape, (batch, 5))
        self.assertEqual(pred.control_vector.shape, (batch, CONTROL_VECTOR_SIZE))

    def test_gradients_flow_through_all_groups(self) -> None:
        assert torch is not None
        batch = 2
        width = 16
        spec = StructuredActionSpec(width=width, movement_mode="5way")
        head = StructuredActuatorHead(width=width, spec=spec, heads=2)
        loss_fn = StructuredActionLoss(spec=spec)

        thoughts = torch.randn(batch, 4, 2, width, requires_grad=True)
        belief = torch.randn(batch, 2, width, requires_grad=True)
        sensors = torch.randn(batch, 2, width, requires_grad=True)

        pred = head(thoughts=thoughts, belief=belief, sensors=sensors)

        target = torch.zeros(batch, CONTROL_VECTOR_SIZE)
        target[0, int(HidKey.W)] = 1.0
        target[0, DEFAULT_MODIFIER_KEYS[0]] = 1.0
        target[0, DEFAULT_TRIGGER_KEYS[0]] = 1.0
        target[0, MOUSE_AXIS_DX] = 0.5
        target[0, MOUSE_AXIS_DY] = -0.5

        loss_output = loss_fn(pred, target)
        self.assertIsInstance(loss_output, StructuredLossOutput)
        self.assertTrue(torch.isfinite(loss_output.loss))

        loss_output.loss.backward()

        self.assertIsNotNone(thoughts.grad)
        self.assertIsNotNone(belief.grad)
        self.assertIsNotNone(sensors.grad)
        assert thoughts.grad is not None
        assert belief.grad is not None
        assert sensors.grad is not None
        self.assertTrue(torch.isfinite(thoughts.grad).all())
        self.assertTrue(torch.isfinite(belief.grad).all())
        self.assertTrue(torch.isfinite(sensors.grad).all())

        # Parameters receive gradients
        for p in head.movement_head.parameters():
            self.assertIsNotNone(p.grad)
        for p in head.modifier_head.parameters():
            self.assertIsNotNone(p.grad)
        for p in head.aim_head.parameters():
            self.assertIsNotNone(p.grad)
        for p in head.trigger_head.parameters():
            self.assertIsNotNone(p.grad)

    def test_wire_format_exact_channel_reconstruction(self) -> None:
        assert torch is not None
        batch = 1
        width = 16
        spec = StructuredActionSpec(width=width, movement_mode="5way")
        head = StructuredActuatorHead(width=width, spec=spec)

        movement_action = torch.tensor([int(MovementDirection5.W)], dtype=torch.long)
        modifier_probs = torch.zeros(batch, spec.num_modifiers)
        modifier_probs[0, 0] = 0.9  # LShift (225) active
        modifier_probs[0, 1] = 0.9  # Space (44) active

        trigger_probs = torch.zeros(batch, spec.num_triggers)
        trigger_probs[0, 0] = 0.9  # E (8) active
        trigger_probs[0, len(spec.trigger_keys)] = 0.9  # LMB (256) active

        mouse_aim = torch.tensor([[0.03, -0.04]])
        scroll = torch.tensor([[0.02]])

        ctrl = head.reconstruct_control_vector(
            movement_action=movement_action,
            modifier_probs=modifier_probs,
            trigger_probs=trigger_probs,
            mouse_aim=mouse_aim,
            scroll=scroll,
            threshold=0.5,
        )

        self.assertEqual(ctrl.shape, (batch, CONTROL_VECTOR_SIZE))

        # Check exact channels
        self.assertEqual(ctrl[0, int(HidKey.W)].item(), 1.0)
        self.assertEqual(ctrl[0, int(HidKey.A)].item(), 0.0)
        self.assertEqual(ctrl[0, int(HidKey.S)].item(), 0.0)
        self.assertEqual(ctrl[0, int(HidKey.D)].item(), 0.0)

        self.assertEqual(ctrl[0, 225].item(), 1.0)  # LShift
        self.assertEqual(ctrl[0, 44].item(), 1.0)   # Space
        self.assertEqual(ctrl[0, 224].item(), 0.0)  # LCtrl

        self.assertEqual(ctrl[0, 8].item(), 1.0)    # E
        self.assertEqual(ctrl[0, 21].item(), 0.0)   # R
        self.assertEqual(ctrl[0, 256].item(), 1.0)  # LMB (mouse button 0)
        self.assertEqual(ctrl[0, 257].item(), 0.0)  # RMB (mouse button 1)

        self.assertAlmostEqual(ctrl[0, MOUSE_AXIS_DX].item(), 0.03, places=5)
        self.assertAlmostEqual(ctrl[0, MOUSE_AXIS_DY].item(), -0.04, places=5)
        self.assertAlmostEqual(ctrl[0, SCROLL_AXIS].item(), 0.02, places=5)

    def test_generic_control_roundtrip(self) -> None:
        assert torch is not None
        batch = 2
        width = 16
        spec = StructuredActionSpec(width=width, movement_mode="5way")
        head = StructuredActuatorHead(width=width, spec=spec)

        features = torch.randn(batch, width)
        pred = head(features=features)

        controls = head.to_generic_controls(pred, threshold=0.5)
        self.assertEqual(len(controls), batch)
        for ctrl in controls:
            self.assertIsInstance(ctrl, GenericControl)
            vec = control_to_vector(ctrl)
            self.assertEqual(len(vec), CONTROL_VECTOR_SIZE)

        single_ctrl = head.to_generic_control(pred, index=0)
        self.assertEqual(single_ctrl, controls[0])

    def test_adversarial_mutual_exclusion_5way(self) -> None:
        """Adversarial stress test: opposite movement pairs (W+S, A+D) can NEVER be simultaneous in 5-way."""
        assert torch is not None
        width = 16
        head = StructuredActuatorHead(
            width=width, spec=StructuredActionSpec(width=width, movement_mode="5way")
        )

        # 1. Edge Case: Adversarial extreme logits competing for opposite pairs
        # Extreme W vs S conflict
        competing_ws = torch.tensor([[0.0, 1e6, 0.0, 1e6 - 1.0, 0.0]])
        action_ws = competing_ws.argmax(dim=-1)
        ctrl_ws = head.reconstruct_control_vector(
            movement_action=action_ws,
            modifier_probs=torch.zeros(1, head.spec.num_modifiers),
            trigger_probs=torch.zeros(1, head.spec.num_triggers),
            mouse_aim=torch.zeros(1, 2),
        )
        self.assertEqual(ctrl_ws[0, int(HidKey.W)].item(), 1.0)
        self.assertEqual(ctrl_ws[0, int(HidKey.S)].item(), 0.0)
        self.assertEqual(
            (ctrl_ws[0, int(HidKey.W)] * ctrl_ws[0, int(HidKey.S)]).item(), 0.0
        )

        # Extreme A vs D conflict
        competing_ad = torch.tensor([[0.0, 0.0, 1e6, 0.0, 1e6 - 1.0]])
        action_ad = competing_ad.argmax(dim=-1)
        ctrl_ad = head.reconstruct_control_vector(
            movement_action=action_ad,
            modifier_probs=torch.zeros(1, head.spec.num_modifiers),
            trigger_probs=torch.zeros(1, head.spec.num_triggers),
            mouse_aim=torch.zeros(1, 2),
        )
        self.assertEqual(ctrl_ad[0, int(HidKey.A)].item(), 1.0)
        self.assertEqual(ctrl_ad[0, int(HidKey.D)].item(), 0.0)
        self.assertEqual(
            (ctrl_ad[0, int(HidKey.A)] * ctrl_ad[0, int(HidKey.D)]).item(), 0.0
        )

        # 2. Saturated 10,000 random logit batch stress test
        random_logits = torch.randn(10000, 5) * 50.0
        actions = random_logits.argmax(dim=-1)
        ctrl_batch = head.reconstruct_control_vector(
            movement_action=actions,
            modifier_probs=torch.zeros(10000, head.spec.num_modifiers),
            trigger_probs=torch.zeros(10000, head.spec.num_triggers),
            mouse_aim=torch.zeros(10000, 2),
        )

        # Invariant 1: W and S are NEVER simultaneously active
        w_active = ctrl_batch[:, int(HidKey.W)] > 0.5
        s_active = ctrl_batch[:, int(HidKey.S)] > 0.5
        ws_conflicts = (w_active & s_active).sum().item()
        self.assertEqual(ws_conflicts, 0)

        # Invariant 2: A and D are NEVER simultaneously active
        a_active = ctrl_batch[:, int(HidKey.A)] > 0.5
        d_active = ctrl_batch[:, int(HidKey.D)] > 0.5
        ad_conflicts = (a_active & d_active).sum().item()
        self.assertEqual(ad_conflicts, 0)

        # Invariant 3: Total active movement keys <= 1
        total_wasd = (
            w_active.float() + a_active.float() + s_active.float() + d_active.float()
        )
        self.assertTrue((total_wasd <= 1.0).all())

    def test_adversarial_mutual_exclusion_9way(self) -> None:
        """Adversarial stress test: opposite movement pairs (W+S, A+D) can NEVER be simultaneous in 9-way."""
        assert torch is not None
        width = 16
        head = StructuredActuatorHead(
            width=width, spec=StructuredActionSpec(width=width, movement_mode="9way")
        )

        # Verify all 9 discrete states individually
        for action_idx in range(9):
            act = torch.tensor([action_idx], dtype=torch.long)
            ctrl = head.reconstruct_control_vector(
                movement_action=act,
                modifier_probs=torch.zeros(1, head.spec.num_modifiers),
                trigger_probs=torch.zeros(1, head.spec.num_triggers),
                mouse_aim=torch.zeros(1, 2),
            )
            w = ctrl[0, int(HidKey.W)].item()
            a = ctrl[0, int(HidKey.A)].item()
            s = ctrl[0, int(HidKey.S)].item()
            d = ctrl[0, int(HidKey.D)].item()

            self.assertEqual(w * s, 0.0, f"W and S coactivated in state {action_idx}")
            self.assertEqual(a * d, 0.0, f"A and D coactivated in state {action_idx}")

        # Saturated 10,000 random logit batch stress test
        random_logits = torch.randn(10000, 9) * 50.0
        actions = random_logits.argmax(dim=-1)
        ctrl_batch = head.reconstruct_control_vector(
            movement_action=actions,
            modifier_probs=torch.zeros(10000, head.spec.num_modifiers),
            trigger_probs=torch.zeros(10000, head.spec.num_triggers),
            mouse_aim=torch.zeros(10000, 2),
        )

        w_active = ctrl_batch[:, int(HidKey.W)] > 0.5
        s_active = ctrl_batch[:, int(HidKey.S)] > 0.5
        ws_conflicts = (w_active & s_active).sum().item()
        self.assertEqual(ws_conflicts, 0)

        a_active = ctrl_batch[:, int(HidKey.A)] > 0.5
        d_active = ctrl_batch[:, int(HidKey.D)] > 0.5
        ad_conflicts = (a_active & d_active).sum().item()
        self.assertEqual(ad_conflicts, 0)

        total_wasd = (
            w_active.float() + a_active.float() + s_active.float() + d_active.float()
        )
        self.assertTrue((total_wasd <= 2.0).all())

    def test_action_sampling(self) -> None:
        assert torch is not None
        batch = 100
        width = 16
        head = StructuredActuatorHead(width=width)
        features = torch.randn(batch, width)
        pred = head(features=features)

        samples = head.sample_actions(pred, temperature=1.0, hard=True)
        self.assertIn("movement_action", samples)
        self.assertIn("modifier_sample", samples)
        self.assertIn("trigger_sample", samples)

        # Movement sampling maintains mutual exclusion
        movement_actions = samples["movement_action"]
        ctrl_sampled = head.reconstruct_control_vector(
            movement_action=movement_actions,
            modifier_probs=samples["modifier_sample"],
            trigger_probs=samples["trigger_sample"],
            mouse_aim=samples["mouse_aim"],
        )
        w_active = ctrl_sampled[:, int(HidKey.W)] > 0.5
        s_active = ctrl_sampled[:, int(HidKey.S)] > 0.5
        self.assertEqual(int((w_active & s_active).sum()), 0)

        a_active = ctrl_sampled[:, int(HidKey.A)] > 0.5
        d_active = ctrl_sampled[:, int(HidKey.D)] > 0.5
        self.assertEqual(int((a_active & d_active).sum()), 0)

    def test_continuous_squash_and_deadzones(self) -> None:
        assert torch is not None
        width = 16
        batch = 5

        # 1. deadzone_tanh squash limit
        spec_deadzone = StructuredActionSpec(
            width=width, continuous_squash="deadzone_tanh"
        )
        head_deadzone = StructuredActuatorHead(width=width, spec=spec_deadzone)
        # Force large continuous outputs
        with torch.no_grad():
            for p in head_deadzone.aim_head.parameters():
                p.fill_(100.0)
        pred_deadzone = head_deadzone(features=torch.ones(batch, width))
        self.assertTrue((pred_deadzone.mouse_aim.abs() <= SQUASH_LIMIT).all())
        self.assertTrue((pred_deadzone.mouse_aim.abs() < CONTINUOUS_DEADZONE_LIMIT).all())

        # 2. tanh bounding
        spec_tanh = StructuredActionSpec(
            width=width, continuous_squash="tanh", max_mouse_aim=2.0
        )
        head_tanh = StructuredActuatorHead(width=width, spec=spec_tanh)
        pred_tanh = head_tanh(features=torch.randn(batch, width))
        self.assertTrue((pred_tanh.mouse_aim.abs() <= 2.0 + 1e-5).all())

        # 3. hard deadzone
        spec_hard = StructuredActionSpec(
            width=width, continuous_squash="hard", continuous_deadzone=0.1
        )
        head_hard = StructuredActuatorHead(width=width, spec=spec_hard)
        raw = torch.tensor([[0.05, 0.2]])  # 0.05 is inside 0.1, 0.2 is outside
        aim_processed, _, _ = head_hard._process_continuous(raw)
        self.assertEqual(aim_processed[0, 0].item(), 0.0)
        self.assertNotEqual(aim_processed[0, 1].item(), 0.0)

        # 4. smooth deadzone
        spec_smooth = StructuredActionSpec(
            width=width, continuous_squash="smooth", continuous_deadzone=0.1
        )
        head_smooth = StructuredActuatorHead(width=width, spec=spec_smooth)
        raw_smooth = torch.tensor([[0.05, 0.3]])
        aim_smooth, _, _ = head_smooth._process_continuous(raw_smooth)
        self.assertEqual(aim_smooth[0, 0].item(), 0.0)
        expected_excess = math.tanh(0.3) - 0.1
        self.assertAlmostEqual(aim_smooth[0, 1].item(), expected_excess, places=5)

    def test_structured_action_loss_computation(self) -> None:
        assert torch is not None
        batch = 4
        width = 16
        spec = StructuredActionSpec(width=width, movement_mode="5way")
        head = StructuredActuatorHead(width=width, spec=spec)
        loss_fn = StructuredActionLoss(
            spec=spec,
            movement_weight=2.0,
            modifier_weight=1.5,
            trigger_weight=1.0,
            continuous_weight=0.5,
            deadzone_hinge_weight=0.1,
            deadzone_hinge_margin=0.04,
        )

        pred = head(features=torch.randn(batch, width))

        # Target tensor
        target = torch.zeros(batch, CONTROL_VECTOR_SIZE)
        target[0, int(HidKey.W)] = 1.0
        target[1, int(HidKey.A)] = 1.0
        target[2, int(HidKey.S)] = 1.0
        target[3, int(HidKey.D)] = 1.0
        target[:, DEFAULT_MODIFIER_KEYS[0]] = 1.0
        target[:, DEFAULT_TRIGGER_KEYS[0]] = 1.0
        target[:, MOUSE_AXIS_DX] = 0.2
        target[:, MOUSE_AXIS_DY] = -0.2

        loss_out = loss_fn(pred, target)
        self.assertIsInstance(loss_out, StructuredLossOutput)
        self.assertEqual(loss_out.samples, batch)
        self.assertGreater(float(loss_out.loss.detach()), 0.0)

        # Check all metric keys
        expected_metrics = {
            "loss",
            "movement_loss",
            "movement_accuracy",
            "modifier_loss",
            "trigger_loss",
            "continuous_loss",
            "deadzone_hinge_loss",
        }
        self.assertEqual(set(loss_out.metrics.keys()), expected_metrics)

        # Test loss with GenericControl objects
        gen_controls = [
            GenericControl(keys_down=(int(HidKey.W),)),
            GenericControl(keys_down=(int(HidKey.A),)),
            GenericControl(keys_down=(int(HidKey.S),)),
            GenericControl(keys_down=(int(HidKey.D),)),
        ]
        loss_gc = loss_fn(pred, gen_controls)
        self.assertIsInstance(loss_gc, StructuredLossOutput)
        self.assertTrue(torch.isfinite(loss_gc.loss))


if __name__ == "__main__":
    unittest.main()
