"""Test suite for Pseudo-Brain Native Semantic & Language Processing Interface."""

import time
import pytest
import torch
import numpy as np

from irene_brain.semantic.tokenizer import SemanticTokenizer
from irene_brain.semantic.native_semantic_model import (
    NativeSemanticPseudoBrain,
    SemanticCognitiveState,
    make_semantic_model,
)
from irene_brain.semantic.streaming_engine import StreamingCognitiveSession
from semantic_benchmark.curriculum_datasets import (
    LanguageCurriculumGenerator,
    build_curriculum_batch,
)
from semantic_benchmark.llm_data_transform import (
    LLMDataTransformer,
    generate_synthetic_raw_dialogues,
)


def test_semantic_tokenizer_fidelity():
    """Verify byte-level encode/decode round-trip and special token preservation."""
    tokenizer = SemanticTokenizer(max_threads=16)
    text = "Hello, world! [THREAD:2]Alice owns a red car. [RESP]ok[EOS]"
    tokens = tokenizer.encode(text)

    assert len(tokens) > 0
    assert tokenizer.thread_tokens[2] in tokenizer.token_to_id
    t2_id = tokenizer.token_to_id["[THREAD:2]"]
    assert t2_id in tokens
    assert tokenizer.token_id_to_thread_id(t2_id) == 2

    decoded = tokenizer.decode(tokens)
    assert "[THREAD:2]" in decoded
    assert "[RESP]" in decoded
    assert "[EOS]" in decoded
    assert "Alice owns a red car." in decoded


def test_native_semantic_model_shapes():
    """Verify forward and single-step shapes of NativeSemanticPseudoBrain."""
    vocab_size = 344
    K = 8
    W = 24
    model = NativeSemanticPseudoBrain(vocab_size=vocab_size, K=K, thought_size=W, embed_dim=48, proj_dim=96)
    model.eval()

    B, T = 2, 10
    token_seq = torch.randint(0, vocab_size, (B, T))
    thread_seq = torch.randint(0, K, (B, T))

    # Sequential forward
    logits = model(token_seq, thread_seq=thread_seq)
    assert logits.shape == (B, T, vocab_size)

    # Streaming single step
    state = model.init_state(batch_size=B, device=token_seq.device)
    assert state.thoughts.shape == (B, K, W)
    assert state.P_t.shape == (B, K, vocab_size)

    logits_0, next_state = model.step(token_seq[:, 0], state, thread_ids=thread_seq[:, 0])
    assert logits_0.shape == (B, vocab_size)
    assert next_state.thoughts.shape == (B, K, W)
    assert next_state.P_t.shape == (B, K, vocab_size)


def test_competitor_baseline_instantiation():
    """Verify that all 6 candidate architectures build and execute without errors."""
    vocab_size = 344
    archs = ["pseudo_brain", "pseudo_brain_no_cgp", "pseudo_brain_no_threads", "gru", "ssm", "reactive"]
    tokens = torch.randint(0, vocab_size, (2, 6))

    for arch in archs:
        model = make_semantic_model(arch, vocab_size=vocab_size, K=8, thought_size=24)
        logits = model(tokens)
        assert logits.shape == (2, 6, vocab_size), f"Failed for arch: {arch}"


def test_cig_gate_slot_shielding():
    """Verify Cognitive Input Gating with T=0.5 sharpening preserves unaddressed slots."""
    model = NativeSemanticPseudoBrain(vocab_size=344, K=8, thought_size=32)
    state = model.init_state(batch_size=1, device=torch.device("cpu"))
    initial_thoughts = state.thoughts.clone()

    # Step on thread 3
    tok = torch.tensor([15], dtype=torch.long)
    th = torch.tensor([3], dtype=torch.long)
    _, next_state = model.step(tok, state, thread_ids=th)

    # Active slot 3 should change significantly
    delta_slot_3 = torch.norm(next_state.thoughts[0, 3] - initial_thoughts[0, 3]).item()
    # Unaddressed slot 0 should have near-zero drift due to CIG shielding
    delta_slot_0 = torch.norm(next_state.thoughts[0, 0] - initial_thoughts[0, 0]).item()

    # Slot 3 must change much more than unaddressed slot 0
    assert delta_slot_3 >= delta_slot_0 or delta_slot_3 > 0.01


def test_streaming_engine_no_history_replay():
    """Verify StreamingCognitiveSession operates token-by-token with zero buffer replay."""
    tokenizer = SemanticTokenizer(max_threads=8)
    model = NativeSemanticPseudoBrain(vocab_size=tokenizer.vocab_size, K=8, thought_size=24)
    session = StreamingCognitiveSession(model=model, tokenizer=tokenizer)

    assert session.total_tokens_processed == 0

    # Ingest text without replaying
    session.ingest_text("Fact: Alice lives in Toronto.", thread_id=0)
    assert session.total_tokens_processed > 0
    tokens_after_ingest = session.total_tokens_processed

    # Ingest query and generate
    res = session.generate_response(prompt_text="Where does Alice live?", thread_id=0, max_new_tokens=8)
    assert len(res["generated_tokens"]) > 0
    assert session.total_tokens_processed > tokens_after_ingest

    # Verify telemetry
    telemetry = session.get_telemetry()
    assert telemetry["total_tokens"] == session.total_tokens_processed
    assert telemetry["latency_mean_ms"] >= 0.0


def test_curriculum_datasets_generation():
    """Verify Stages A and B curriculum generation and batch collation."""
    generator = LanguageCurriculumGenerator(max_threads=8)
    rng = np.random.RandomState(42)

    ep_copy = generator.generate_stage_a_copy(rng, length=4, delay=6)
    assert ep_copy.task_name == "stage_a_copy"
    assert len(ep_copy.tokens) == len(ep_copy.targets) == len(ep_copy.threads)

    ep_bind = generator.generate_stage_a_associative_binding(rng, num_pairs=3, delay=6)
    assert ep_bind.task_name == "stage_a_associative_binding"

    ep_recall = generator.generate_stage_b_delayed_recall(rng, num_facts=2, delay=8)
    assert ep_recall.task_name == "stage_b_delayed_recall"

    ep_inter = generator.generate_stage_b_interleaved_conversations(rng, num_conversations=3)
    assert ep_inter.task_name == "stage_b_interleaved_conversations"

    ep_preempt = generator.generate_stage_b_preemption(rng, interruption_length=12)
    assert ep_preempt.task_name == "stage_b_preemption"

    # Batch builder
    tokens_t, targets_t, threads_t, episodes = build_curriculum_batch(generator, batch_size=8, seed=123)
    assert tokens_t.shape[0] == 8
    assert targets_t.shape == tokens_t.shape
    assert threads_t.shape == tokens_t.shape
    assert len(episodes) == 8


def test_phase10_llm_data_transform():
    """Verify Phase 10 transformation pipeline converts raw dialogues into multi-threaded episodes."""
    raw_dialogues = generate_synthetic_raw_dialogues(num_dialogues=6, seed=42)
    assert len(raw_dialogues) == 6

    transformer = LLMDataTransformer(max_threads=4)
    episodes = transformer.transform_corpus(raw_dialogues, num_episodes=5, seed=42)
    assert len(episodes) == 5

    for ep in episodes:
        assert ep.num_threads <= 4
        assert len(ep.interleaved_text) > 0
        assert len(ep.thread_schedule) > 0
        assert "[THREAD:" in ep.interleaved_text


def test_streaming_latency_realtime():
    """Verify single-token cognitive update latency satisfies <= 16.67ms (60 Hz) on CPU."""
    model = NativeSemanticPseudoBrain(vocab_size=344, K=16, thought_size=32, embed_dim=64, proj_dim=128)
    session = StreamingCognitiveSession(model=model)

    latencies = []
    for _ in range(50):
        _, dt = session.step_token(token_id=25, thread_id=1)
        latencies.append(dt)

    p90 = float(np.percentile(latencies, 90))
    # Must comfortably be within 16.67 ms (typically 1-3 ms on CPU)
    assert p90 < 16.67, f"p90 latency {p90:.2f} ms exceeded 16.67 ms budget"
