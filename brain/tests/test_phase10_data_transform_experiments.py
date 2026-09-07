"""Test suite for Phase 10: Conventional LLM Dataset Transformation Pipeline."""

import json
from pathlib import Path
import tempfile
import pytest
import torch
import numpy as np

from irene_brain.semantic.tokenizer import SemanticTokenizer
from irene_brain.semantic.native_semantic_model import (
    NativeSemanticPseudoBrain,
    MonolithicGRULanguageModel,
)
from semantic_benchmark.llm_data_transform import (
    LLMDataTransformer,
    TrainingParadigm,
    RawDialogueTurn,
    RawDialogue,
    TransformedCognitiveEpisode,
    generate_synthetic_raw_dialogues,
    export_transformed_dataset,
    collate_episode_batch,
    evaluate_model_on_episodes,
)


def test_parse_sharegpt_format():
    """Verify parsing of ShareGPT conversations format."""
    transformer = LLMDataTransformer(max_threads=8)
    sharegpt_sample = [
        {
            "id": "sharegpt_001",
            "category": "travel",
            "conversations": [
                {"from": "human", "value": "Where is Mount Fuji located?"},
                {"from": "gpt", "value": "Mount Fuji is located on Honshu Island in Japan."},
                {"from": "human", "value": "How tall is it?"},
                {"from": "gpt", "value": "It stands at 3,776 meters tall."},
            ],
        }
    ]

    dialogues = transformer.parse_standard_dialogue_json(sharegpt_sample)
    assert len(dialogues) == 1
    d = dialogues[0]
    assert d.dialogue_id == "sharegpt_001"
    assert d.category == "travel"
    assert len(d.turns) == 4
    assert d.turns[0].role == "user"
    assert "Mount Fuji" in d.turns[0].content
    assert d.turns[1].role == "assistant"
    assert "3,776 meters" in d.turns[3].content


def test_parse_openai_format():
    """Verify parsing of OpenAI messages format with system prompt."""
    transformer = LLMDataTransformer(max_threads=8)
    openai_sample = [
        {
            "id": "openai_001",
            "topic": "coding",
            "messages": [
                {"role": "system", "content": "You are an expert Python developer."},
                {"role": "user", "content": "How do I reverse a string in Python?"},
                {"role": "assistant", "content": "You can use string slicing: s[::-1]."},
            ],
        }
    ]

    dialogues = transformer.parse_standard_dialogue_json(openai_sample)
    assert len(dialogues) == 1
    d = dialogues[0]
    assert d.dialogue_id == "openai_001"
    assert d.category == "coding"
    assert len(d.turns) == 2
    assert d.turns[0].role == "user"
    assert "[System: You are an expert Python developer.]" in d.turns[0].content
    assert "s[::-1]" in d.turns[1].content


def test_parse_alpaca_format():
    """Verify parsing of Alpaca instruction/input/output format."""
    transformer = LLMDataTransformer(max_threads=8)
    alpaca_sample = [
        {
            "instruction": "Calculate the square of the input number.",
            "input": "12",
            "output": "The square of 12 is 144.",
        }
    ]

    dialogues = transformer.parse_standard_dialogue_json(alpaca_sample)
    assert len(dialogues) == 1
    d = dialogues[0]
    assert len(d.turns) == 2
    assert d.turns[0].role == "user"
    assert "Input: 12" in d.turns[0].content
    assert "144" in d.turns[1].content


def test_parse_filtering_and_edge_cases():
    """Verify handling of empty, single-turn, or malformed dialogues."""
    transformer = LLMDataTransformer(max_threads=8)
    invalid_sample = [
        {"conversations": []},  # Empty
        {"conversations": [{"from": "human", "value": "Only user message"}]},  # No assistant turn
        {"conversations": [{"from": "gpt", "value": "Only assistant message"}]},  # No user turn
        {"conversations": [{"from": "human", "value": "   "}, {"from": "gpt", "value": "   "}]},  # Whitespace
    ]

    dialogues = transformer.parse_standard_dialogue_json(invalid_sample)
    assert len(dialogues) == 0


def test_synthetic_dialogue_seen_and_unseen():
    """Verify generation of seen (in-distribution) and unseen (OOD) dialogues."""
    seen = generate_synthetic_raw_dialogues(num_dialogues=8, seed=42, unseen=False)
    assert len(seen) == 8
    seen_categories = {d.category for d in seen}
    assert "travel" in seen_categories or "coding" in seen_categories

    unseen = generate_synthetic_raw_dialogues(num_dialogues=8, seed=42, unseen=True)
    assert len(unseen) == 8
    unseen_categories = {d.category for d in unseen}
    assert "medical" in unseen_categories or "finance" in unseen_categories
    assert not seen_categories.intersection(unseen_categories)


def test_four_training_paradigms_generation():
    """Verify episode formatting for all 4 Training Paradigms."""
    dialogues = generate_synthetic_raw_dialogues(num_dialogues=10, seed=42)
    transformer = LLMDataTransformer(max_threads=8)
    rng = np.random.RandomState(42)

    # 1. Model A (Monolithic Sequential, no thread tokens)
    ep_a = transformer.create_model_a_episode(dialogues, rng)
    assert ep_a.paradigm == TrainingParadigm.MODEL_A.value
    assert ep_a.num_threads == 1
    assert "[THREAD:" not in ep_a.interleaved_text
    assert all(th == 0 for th in ep_a.threads)
    assert ep_a.targets is not None
    assert any(t != -100 for t in ep_a.targets)

    # 2. Model B (Sequential with persistent state)
    ep_b = transformer.create_model_b_episode(dialogues, rng)
    assert ep_b.paradigm == TrainingParadigm.MODEL_B.value
    assert ep_b.num_threads == 1
    assert "[THREAD:0]" in ep_b.interleaved_text
    assert all(th == 0 for th in ep_b.threads)

    # 3. Model C (Interleaved cognitive training, independent threads)
    ep_c = transformer.create_model_c_episode(dialogues, rng, num_threads=4)
    assert ep_c.paradigm == TrainingParadigm.MODEL_C.value
    assert ep_c.num_threads > 1
    assert ep_c.has_preemption is False
    assert ep_c.has_dependency is False
    assert any(th > 0 for th in ep_c.threads)

    # 4. Model D (Interleaved + Preemption + Cross-Thread Dependency)
    ep_d = transformer.create_model_d_episode(dialogues, rng, num_threads=4)
    assert ep_d.paradigm == TrainingParadigm.MODEL_D.value
    assert ep_d.has_preemption is True
    assert ep_d.has_dependency is True
    assert "[INTERRUPT]" in ep_d.interleaved_text
    assert "[RESUME]" in ep_d.interleaved_text
    assert "[DEP]" in ep_d.interleaved_text


def test_long_term_memory_delay_sweep_generation():
    """Verify Long-Term Memory Delay Sweep construction across L in [10, 32, 64, 128, 256, 512]."""
    transformer = LLMDataTransformer(max_threads=8)
    rng = np.random.RandomState(42)
    delays = [10, 32, 64, 128, 256, 512]

    for L in delays:
        ep = transformer.generate_delay_sweep_episode(rng, delay_tokens=L, paradigm=TrainingParadigm.MODEL_D.value)
        assert ep.metadata["delay_tokens"] == L
        assert len(ep.tokens) > L
        assert len(ep.tokens) == len(ep.targets) == len(ep.threads)
        # Verify target is only at the end
        assert ep.targets[-1] != -100
        assert ep.targets[0] == -100


def test_jsonl_dataset_export_and_reload():
    """Verify transformed cognitive dataset serialization to JSONL and round-trip fidelity."""
    dialogues = generate_synthetic_raw_dialogues(num_dialogues=6, seed=42)
    transformer = LLMDataTransformer(max_threads=4)
    episodes = transformer.transform_corpus(dialogues, num_episodes=5, seed=42, paradigm=TrainingParadigm.MODEL_D.value)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        export_transformed_dataset(episodes, tmp_path)
        assert tmp_path.stat().st_size > 0

        with open(tmp_path, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        assert len(lines) == 5
        for item in lines:
            assert "episode_id" in item
            assert "interleaved_text" in item
            assert "num_threads" in item
            assert "has_preemption" in item
            assert "has_dependency" in item
            assert "thread_schedule" in item
            assert "tokens" in item
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def test_model_paradigms_forward_backward_pass():
    """Verify forward and backward gradients for Model A (Monolithic) and Model D (Pseudo-Brain)."""
    tokenizer = SemanticTokenizer(max_threads=8)
    transformer = LLMDataTransformer(max_threads=8, tokenizer=tokenizer)
    dialogues = generate_synthetic_raw_dialogues(num_dialogues=4, seed=42)
    rng = np.random.RandomState(42)

    # 1. Model A forward/backward
    model_a = MonolithicGRULanguageModel(vocab_size=tokenizer.vocab_size, embed_dim=32, hidden_dim=64)
    ep_a = transformer.create_model_a_episode(dialogues, rng)
    tokens_a, targets_a, _ = collate_episode_batch([ep_a], pad_id=tokenizer.pad_id)

    logits_a = model_a(tokens_a)
    loss_a = torch.nn.functional.cross_entropy(logits_a.view(-1, logits_a.shape[-1]), targets_a.view(-1), ignore_index=-100)
    loss_a.backward()

    for p in model_a.parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any()

    # 2. Model D forward/backward
    model_d = NativeSemanticPseudoBrain(
        vocab_size=tokenizer.vocab_size,
        K=8,
        thought_size=16,
        embed_dim=32,
        proj_dim=64,
        use_cgp=True,
        use_routing=True,
    )
    ep_d = transformer.create_model_d_episode(dialogues, rng, num_threads=4)
    tokens_d, targets_d, threads_d = collate_episode_batch([ep_d], pad_id=tokenizer.pad_id)

    logits_d = model_d(tokens_d, thread_seq=threads_d, allow_routing=True)
    loss_d = torch.nn.functional.cross_entropy(logits_d.view(-1, logits_d.shape[-1]), targets_d.view(-1), ignore_index=-100)
    loss_d.backward()

    for p in model_d.parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any()


def test_phase10_benchmark_evaluation_smoke():
    """Smoke test of evaluation metrics on candidate models."""
    tokenizer = SemanticTokenizer(max_threads=4)
    transformer = LLMDataTransformer(max_threads=4, tokenizer=tokenizer)
    model = NativeSemanticPseudoBrain(vocab_size=tokenizer.vocab_size, K=4, thought_size=16, embed_dim=32, proj_dim=64)

    dialogues = generate_synthetic_raw_dialogues(num_dialogues=4, seed=42)
    rng = np.random.RandomState(42)
    episodes = [transformer.create_model_c_episode(dialogues, rng, num_threads=2) for _ in range(3)]

    metrics = evaluate_model_on_episodes(model, episodes, tokenizer, batch_size=2)
    assert "token_accuracy" in metrics
    assert "sequence_accuracy" in metrics
    assert "latency_ms" in metrics
    assert 0.0 <= metrics["token_accuracy"] <= 100.0
    assert 0.0 <= metrics["sequence_accuracy"] <= 100.0
    assert metrics["latency_ms"] >= 0.0
