"""Unit Test Suite for Multimodal Pseudo-Brain & Endogenous Hierarchical Milestones.

Verifies:
1. Multimodal token ingestion (image tensor + token stream)
2. Cross-modal slot binding and state persistence
3. Milestone latch consolidation under tool success
4. End-to-end multimodal streaming sessions and sub-16.67ms latency
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

from irene_brain.agent.goal import GoalMilestone, GoalSpecification
from irene_brain.agent.loop import (
    AgentCognitiveCore,
    AgentStepLog,
    PseudoBrainAgent,
)
from irene_brain.agent.tools import (
    CommandTool,
    FileReadTool,
    FileWriteTool,
    TestVerifyTool,
    ToolRegistry,
)
from irene_brain.semantic.multimodal_model import (
    ConvEncoder,
    MultimodalCognitiveState,
    MultimodalPseudoBrain,
    MultimodalStreamingSession,
)
from irene_brain.semantic.tokenizer import SemanticTokenizer


class TestMultimodalPseudoBrain(unittest.TestCase):
    """Comprehensive test suite for Multimodal Pseudo-Brain and Hierarchical Milestones."""

    def setUp(self):
        torch.manual_seed(42)
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    # =========================================================================
    # Part 1: ConvEncoder & Multimodal Token Ingestion
    # =========================================================================

    def test_conv_encoder_pomdp_hierarchy(self):
        """Verify ConvEncoder features, entity token extraction, and B=1 GroupNorm stability."""
        # 3-channel RGB image encoder (e.g. 32x32)
        encoder_rgb = ConvEncoder(in_channels=3, feature_dim=192, num_entity_tokens=4)

        # 1. Single-step streaming image (B=1)
        x_single = torch.randn(1, 3, 32, 32)
        entities = encoder_rgb(x_single, as_entities=True)
        self.assertEqual(entities.shape, (1, 4, 192), "Must extract 4 spatial entity tokens.")

        global_feat = encoder_rgb(x_single, as_entities=False)
        self.assertEqual(global_feat.shape, (1, 192), "Must extract 192-d pooled global scene feature.")

        # 2. Batch of images (B=3)
        x_batch = torch.randn(3, 3, 32, 32)
        batch_entities = encoder_rgb(x_batch, as_entities=True)
        self.assertEqual(batch_entities.shape, (3, 4, 192))

        # 3. Level 12 POMDP 4-frame temporal stacked input (12 channels)
        encoder_pomdp = ConvEncoder(in_channels=12, feature_dim=192, num_entity_tokens=4)
        x_pomdp = torch.randn(2, 12, 32, 32)
        pomdp_entities = encoder_pomdp(x_pomdp, as_entities=True)
        self.assertEqual(pomdp_entities.shape, (2, 4, 192))

    def test_multimodal_token_ingestion(self):
        """Verify ingestion of image tensors, language token streams, and interleaved sequences."""
        tokenizer = SemanticTokenizer(max_threads=8)
        model = MultimodalPseudoBrain(
            vocab_size=tokenizer.vocab_size,
            K=8,
            thought_size=32,
            embed_dim=64,
            proj_dim=128,
            visual_dim=192,
            num_visual_tokens=4,
        )

        state = model.init_state(batch_size=1, device=torch.device("cpu"))
        self.assertEqual(state.thoughts.shape, (1, 8, 32))
        self.assertEqual(state.P_t.shape, (1, 8, tokenizer.vocab_size))

        # 1. Ingest image tensor
        img = torch.randn(1, 3, 32, 32)
        img_logits, state = model.ingest_image(img, state=state, slot_idx=0)
        self.assertEqual(img_logits.shape, (1, tokenizer.vocab_size))
        self.assertEqual(state.thoughts.shape, (1, 8, 32))
        self.assertEqual(int(state.slot_modalities[0, 0].item()), 2, "Slot 0 should have visual tag (2).")

        # 2. Ingest text token stream
        text_tokens = torch.tensor([tokenizer.encode("Room 4 contains the red key")], dtype=torch.long)
        text_logits, state = model.ingest_text_tokens(text_tokens, state=state, slot_idx=0)
        self.assertEqual(text_logits.shape, (1, tokenizer.vocab_size))
        self.assertEqual(int(state.slot_modalities[0, 0].item()), 3, "Slot 0 should now be multimodal (1 | 2 = 3).")

        # 3. Interleaved sequence via unified forward()
        img2 = torch.randn(1, 3, 32, 32)
        prompt_tokens = torch.tensor([tokenizer.encode("Where is the red key?")], dtype=torch.long)
        final_logits, state2 = model.forward(
            token_seq=prompt_tokens,
            image_tensor=img2,
            state=None,
            slot_idx=1,
        )
        self.assertEqual(final_logits.shape, (1, tokenizer.vocab_size))
        self.assertEqual(state2.thoughts.shape, (1, 8, 32))

        # 4. Gradient backpropagation to both visual encoder and token embeddings
        model.train()
        x_img = torch.randn(2, 3, 32, 32, requires_grad=True)
        x_tok = torch.tensor([[10, 20, 30], [15, 25, 35]], dtype=torch.long)
        logits_train, _ = model.forward(token_seq=x_tok, image_tensor=x_img, slot_idx=0)
        loss = logits_train.sum()
        loss.backward()

        self.assertIsNotNone(model.visual_encoder.conv1.weight.grad)
        self.assertIsNotNone(model.embedding.weight.grad)
        self.assertGreater(model.visual_encoder.conv1.weight.grad.abs().sum().item(), 0.0)
        self.assertGreater(model.embedding.weight.grad.abs().sum().item(), 0.0)

    # =========================================================================
    # Part 2: Cross-Modal Slot Binding & State Persistence
    # =========================================================================

    def test_cross_modal_slot_binding(self):
        """Verify visual entity tokens and language tokens bind to the same persistent thought slot."""
        tokenizer = SemanticTokenizer(max_threads=8)
        model = MultimodalPseudoBrain(
            vocab_size=tokenizer.vocab_size,
            K=8,
            thought_size=32,
            embed_dim=64,
            proj_dim=128,
            visual_dim=192,
            num_visual_tokens=4,
        )
        model.eval()

        state = model.init_state(batch_size=1, device=torch.device("cpu"))
        initial_slot2_thought = state.thoughts[0, 2].clone()
        initial_slot0_thought = state.thoughts[0, 0].clone()

        # Step A: Bind visual entity token to slot 2
        img = torch.randn(1, 3, 32, 32)
        _, state = model.ingest_image(img, state=state, slot_idx=2)

        # Slot 2 thought should have updated, while Slot 0 remained unaffected
        self.assertFalse(
            torch.allclose(state.thoughts[0, 2], initial_slot2_thought, atol=1e-3),
            "Slot 2 thoughts must adapt after visual ingestion.",
        )
        self.assertTrue(
            torch.allclose(state.thoughts[0, 0], initial_slot0_thought, atol=1e-5),
            "Slot 0 thoughts must remain untouched when Slot 2 is targeted.",
        )
        self.assertEqual(int(state.slot_modalities[0, 2].item()), 2, "Slot 2 marked visual.")

        # Step B: Bind language tokens ("golden key in chest") to the same Slot 2
        text_tokens = torch.tensor([tokenizer.encode("golden key in chest")], dtype=torch.long)
        _, state = model.ingest_text_tokens(text_tokens, state=state, slot_idx=2)

        # Slot 2 should now have both modalities bound
        self.assertEqual(int(state.slot_modalities[0, 2].item()), 3, "Slot 2 must be multimodal bound (3).")

        # Step C: Verify visual feature reconstruction from Slot 2
        pred_feat = model.predict_visual_features(slot_idx=2, state=state)
        self.assertEqual(pred_feat.shape, (1, 192))

        # Step D: Test cross-modal affinity computation
        affinities = model.cross_modal_affinity(
            slot_idx=2,
            state=state,
            text_tokens=text_tokens,
            image_pixels=img,
        )
        self.assertIn("text_similarity", affinities)
        self.assertIn("visual_similarity", affinities)

    def test_cross_modal_state_persistence(self):
        """Verify thought slot maintains bound representations under distracting operations in other slots."""
        tokenizer = SemanticTokenizer(max_threads=8)
        model = MultimodalPseudoBrain(
            vocab_size=tokenizer.vocab_size,
            K=8,
            thought_size=32,
            embed_dim=64,
            proj_dim=128,
            visual_dim=192,
        )
        model.eval()

        state = model.init_state(batch_size=1, device=torch.device("cpu"))

        # 1. Bind critical entity to Slot 0: image + key descriptor
        target_img = torch.randn(1, 3, 32, 32)
        _, state = model.ingest_image(target_img, state=state, slot_idx=0)
        text_target = torch.tensor([tokenizer.encode("security key stored here")], dtype=torch.long)
        _, state = model.ingest_text_tokens(text_target, state=state, slot_idx=0)

        preserved_slot0_thought = state.thoughts[0, 0].clone()
        preserved_slot0_plastic = state.P_t[0, 0].clone()

        # 2. Perform 10 heavy distracting steps in Slot 1 and Slot 3 (both image and text)
        for d in range(5):
            distract_img = torch.randn(1, 3, 32, 32)
            _, state = model.ingest_image(distract_img, state=state, slot_idx=1)
            distract_text = torch.tensor([tokenizer.encode(f"irrelevant chatter {d}")], dtype=torch.long)
            _, state = model.ingest_text_tokens(distract_text, state=state, slot_idx=3)

        # 3. Verify Slot 0 state persistence
        current_slot0_thought = state.thoughts[0, 0]
        thought_drift = float(torch.norm(current_slot0_thought - preserved_slot0_thought).item())
        self.assertAlmostEqual(
            thought_drift,
            0.0,
            places=5,
            msg=f"Slot 0 thought drifted by {thought_drift}; must be zero under slot isolation.",
        )

        current_slot0_plastic = state.P_t[0, 0]
        plastic_drift = float(torch.norm(current_slot0_plastic - preserved_slot0_plastic).item())
        self.assertAlmostEqual(
            plastic_drift,
            0.0,
            places=5,
            msg=f"Slot 0 synaptic latch drifted by {plastic_drift}; must remain persistent.",
        )

    # =========================================================================
    # Part 3: Endogenous Hierarchical Milestones & Latch Consolidation
    # =========================================================================

    def test_hierarchical_milestone_tracking(self):
        """Verify prerequisite resolution, readiness querying, and sequential milestone completion."""
        m1 = GoalMilestone(
            milestone_id="m1_read",
            description="Read initial config",
            tool_name="read_file",
            prerequisites=[],
            reinforcement=1.5,
        )
        m2 = GoalMilestone(
            milestone_id="m2_write",
            description="Write updated patch",
            tool_name="write_file",
            prerequisites=["m1_read"],
            reinforcement=2.0,
        )
        m3 = GoalMilestone(
            milestone_id="m3_verify",
            description="Verify test pass",
            tool_name="verify_goal",
            prerequisites=["m2_write"],
            reinforcement=3.0,
        )

        goal = GoalSpecification(
            goal_id="task_hierarchical",
            text="Read config, write patch, and verify completion",
            milestones=[m1, m2, m3],
        )

        # Initially, only m1 is ready
        ready = goal.get_ready_milestones()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].milestone_id, "m1_read")

        # Attempting write_file before m1 completes should NOT complete m2
        unearned = goal.check_milestone_completion(tool_name="write_file", tool_success=True)
        self.assertEqual(len(unearned), 0, "Dependent milestone m2 must not complete before prerequisite m1.")

        # Complete m1
        completed_m1 = goal.check_milestone_completion(tool_name="read_file", tool_success=True, step=1)
        self.assertEqual(len(completed_m1), 1)
        self.assertEqual(completed_m1[0].milestone_id, "m1_read")
        self.assertTrue(m1.completed)

        # Now m2 is ready
        ready = goal.get_ready_milestones()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].milestone_id, "m2_write")

        # Complete m2
        completed_m2 = goal.check_milestone_completion(tool_name="write_file", tool_success=True, step=2)
        self.assertEqual(len(completed_m2), 1)
        self.assertEqual(completed_m2[0].milestone_id, "m2_write")
        self.assertTrue(m2.completed)

        # Now m3 is ready
        ready = goal.get_ready_milestones()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].milestone_id, "m3_verify")

        # Complete m3
        completed_m3 = goal.check_milestone_completion(tool_name="verify_goal", tool_success=True, step=3)
        self.assertEqual(len(completed_m3), 1)
        self.assertTrue(goal.is_complete(), "Goal must be complete after all milestones are satisfied.")

    def test_milestone_latch_consolidation_under_tool_success(self):
        """Verify endogenous milestone reinforcement m_t > 0 consolidates synaptic latches P_t."""
        cfg_file = self.root / "config.json"
        out_file = self.root / "result.txt"
        cfg_file.write_text('{"status": "ACTIVE"}', encoding="utf-8")

        def is_verified() -> bool:
            return out_file.exists() and "ACTIVE_DONE" in out_file.read_text(encoding="utf-8")

        tools = [
            FileReadTool(),     # idx 0 (Prerequisite)
            FileWriteTool(),    # idx 1 (Dependent)
            TestVerifyTool(is_verified),  # idx 2 (Terminal)
        ]
        registry = ToolRegistry(tools)
        agent = PseudoBrainAgent(registry=registry)

        # Prime initial exploration: tool 0 (read) is favored first
        agent.core.set_tool_bias(0, 2.0)
        agent.core.set_tool_bias(1, 1.0)
        agent.core.set_tool_bias(2, 0.5)

        m1 = GoalMilestone(
            milestone_id="m1_read_cfg",
            description="Read active configuration",
            tool_name="read_file",
            prerequisites=[],
            reinforcement=2.5,  # m_t > 0
        )
        m2 = GoalMilestone(
            milestone_id="m2_write_res",
            description="Write processed result",
            tool_name="write_file",
            prerequisites=["m1_read_cfg"],
            reinforcement=1.5,
        )

        goal = GoalSpecification(
            goal_id="task_milestone_consolidation",
            text="Read config and write result",
            verification_fn=is_verified,
            milestones=[m1, m2],
        )

        def dynamic_arg_provider(step: int, tool_name: str, logs: List[AgentStepLog], g: GoalSpecification):
            if tool_name == "read_file":
                return {"path": str(cfg_file)}
            elif tool_name == "write_file":
                return {"path": str(out_file), "content": "ACTIVE_DONE"}
            elif tool_name == "verify_goal":
                return {}
            return {}

        report = agent.run_task(
            goal=goal,
            max_steps=6,
            arg_provider=dynamic_arg_provider,
        )

        # 1. Verify task success
        self.assertTrue(report.success, "Autonomous agent task must complete successfully.")
        self.assertTrue(is_verified())

        # 2. Check milestone reinforcement on Step 1 (read_file completion)
        step1_log = report.steps_log[0]
        self.assertEqual(step1_log.tool_name, "read_file")
        self.assertTrue(step1_log.success)
        self.assertEqual(step1_log.milestone_reinforcement, 2.5, "Milestone reinforcement m_t=2.5 must be injected.")
        self.assertIn("m1_read_cfg", step1_log.completed_milestones)

        # 3. Verify Synaptic Latch P_t Consolidation on prerequisite tool (idx 0)
        # P_t[0, 0] must be positively consolidated (> 3.0) due to 2.0 * m_t reinforcement
        self.assertIsNotNone(step1_log.p_latch_prereq)
        self.assertGreater(
            step1_log.p_latch_prereq,
            3.0,
            f"Prerequisite tool latch P_t must be consolidated (>3.0), got {step1_log.p_latch_prereq}",
        )

        # 4. Verify consequence surprise incorporated milestone reinforcement
        self.assertGreaterEqual(
            step1_log.consequence_surprise,
            2.5,
            "Consequence surprise must ingest milestone reinforcement m_t.",
        )

        # 5. Verify branching into dependent task on Step 2
        step2_log = report.steps_log[1]
        self.assertEqual(
            step2_log.tool_name,
            "write_file",
            "Agent must branch into dependent task (write_file) after prerequisite consolidation.",
        )
        self.assertEqual(step2_log.milestone_reinforcement, 1.5)
        self.assertIn("m2_write_res", step2_log.completed_milestones)

        # 6. Verify total milestone reward
        self.assertEqual(report.milestone_reward, 4.0, "Total milestone reward must be 2.5 + 1.5 = 4.0.")

    def test_milestone_failure_ior_suppression(self):
        """Verify failed tools do not receive milestone reinforcement and suffer IOR suppression."""
        target_file = self.root / "missing.txt"

        def is_verified() -> bool:
            return False

        tools = [
            FileReadTool(),  # Will fail because file does not exist
            FileWriteTool(),
        ]
        registry = ToolRegistry(tools)
        agent = PseudoBrainAgent(registry=registry)

        m1 = GoalMilestone(
            milestone_id="m1_read_must_fail",
            description="Attempt to read non-existent file",
            tool_name="read_file",
            reinforcement=2.0,
        )

        goal = GoalSpecification(
            goal_id="task_fail_ior",
            text="Read non-existent file",
            verification_fn=is_verified,
            milestones=[m1],
        )

        default_args = {
            "read_file": {"path": str(target_file)},
            "write_file": {"path": str(self.root / "created.txt"), "content": "NEW"},
        }

        # Force read_file on first step
        agent.core.set_tool_bias(0, 3.0)

        report = agent.run_task(goal, max_steps=2, default_args=default_args)

        step1 = report.steps_log[0]
        self.assertEqual(step1.tool_name, "read_file")
        self.assertFalse(step1.success, "read_file must fail for missing file.")
        self.assertEqual(step1.milestone_reinforcement, 0.0, "No milestone reward on failure.")
        self.assertNotIn("m1_read_must_fail", step1.completed_milestones)

        # Step 2 must suppress read_file due to IOR and transition to write_file
        step2 = report.steps_log[1]
        self.assertEqual(step2.tool_name, "write_file", "IOR must force transition away from failing tool.")

    # =========================================================================
    # Part 4: Multimodal Streaming Session & Latency
    # =========================================================================

    def test_multimodal_streaming_session(self):
        """Verify end-to-end interactive multimodal session, token generation, and latency telemetry."""
        tokenizer = SemanticTokenizer(max_threads=8)
        model = MultimodalPseudoBrain(
            vocab_size=tokenizer.vocab_size,
            K=8,
            thought_size=32,
            embed_dim=64,
            proj_dim=128,
            visual_dim=192,
        )

        session = MultimodalStreamingSession(model=model, tokenizer=tokenizer)

        # 1. Ingest image into Slot 1
        img = torch.randn(1, 3, 32, 32)
        res_img = session.ingest_image(img, slot_idx=1)
        self.assertEqual(res_img["active_slot"], 1)
        self.assertIn("latency_ms", res_img)

        # 2. Ingest prompt into Slot 1
        prompt = "Describe the observed entity:"
        latencies = session.ingest_text(prompt, thread_id=1)
        self.assertEqual(len(latencies), len(tokenizer.encode(prompt, thread_id=1)))

        # 3. Generate response autoregressively
        gen_res = session.generate_response(thread_id=1, max_new_tokens=8)
        self.assertIn("response_text", gen_res)
        self.assertIn("mean_step_latency_ms", gen_res)
        self.assertEqual(gen_res["active_thread"], 1)

        # 4. Check telemetry
        telem = session.get_telemetry()
        self.assertGreater(telem["total_tokens"], 0)
        self.assertEqual(telem["total_images"], 1)
        self.assertIn("latency_mean_ms", telem)
        self.assertTrue(telem["latency_under_16_67ms"], "Streaming step latency must satisfy 60Hz constraint.")

        # 5. Verify Slot 1 state retrieval
        slot1_state = session.get_slot_state(1)
        self.assertEqual(slot1_state.shape, (32,))
        self.assertGreater(float(slot1_state.norm().item()), 0.0)

    # =========================================================================
    # Part 5: Track C Closed-Loop Multimodal Embodied Play
    # =========================================================================

    def test_closed_loop_multimodal_embodied_episode(self):
        """Verify closed-loop episode execution with POMDP, zero token replay buffer, and milestone latching."""
        from irene_brain.semantic.multimodal_model import MultimodalPseudoBrainModel
        from memory_benchmark.difficulty_curve_ablation import CorridorDelayKeysDoorsEnv
        from semantic_benchmark.multimodal_closed_loop_benchmark import run_single_episode

        tokenizer = SemanticTokenizer(max_threads=4)
        model = MultimodalPseudoBrainModel(
            vocab_size=tokenizer.vocab_size,
            K=4,
            thought_size=32,
            embed_dim=32,
            proj_dim=64,
            visual_dim=64,
            num_visual_tokens=4,
            n_actions=5,
        )

        env = CorridorDelayKeysDoorsEnv(corridor_delay=4)
        ep = run_single_episode(
            model=model,
            tokenizer=tokenizer,
            env=env,
            seed=1000,
            episode_idx=0,
            directive="retrieve key, ignore hallway hazard, unlock blue door",
            device=torch.device("cpu"),
        )

        self.assertTrue(ep.success, "Closed-loop multimodal episode must succeed.")
        self.assertTrue(ep.key_collected, "Key must be acquired.")
        self.assertTrue(ep.door_unlocked, "Door must be unlocked.")
        self.assertTrue(ep.target_collected, "Target must be collected.")
        self.assertGreaterEqual(ep.prereq_latch_at_milestone, 4.0, "Prerequisite latch must consolidate (>= 4.0).")
        self.assertTrue(ep.latch_consolidated, "Latch consolidated flag must be True.")
        self.assertGreaterEqual(ep.slot0_persistence_mean, 0.99, "Slot 0 directive must persist without replay buffer.")
        self.assertTrue(ep.latency_under_16_67ms, "Per-tick latency must satisfy 60Hz real-time SLA.")


if __name__ == "__main__":
    unittest.main()

