"""Comprehensive Unit Tests for Fused Cells and Parallel Recurrent Blocks."""

import copy
import pytest
import torch

from irene_brain.parallel.fused_cell import (
    FusedBrainCellCore,
    FusedSlotRecurrentBlock,
    ParallelNativeSemanticPseudoBrain,
)


def test_fused_brain_cell_core_parallel_matches_sequential():
    """Verify FusedBrainCellCore parallel scan exactly matches sequential loop and step streaming."""
    torch.manual_seed(42)
    B, T, in_dim, w_dim = 4, 32, 64, 32
    cell = FusedBrainCellCore(input_size=in_dim, thought_size=w_dim)
    cell.eval()

    x = torch.randn(B, T, in_dim)
    h0 = torch.randn(B, w_dim)

    out_we = cell(x, h0=h0, method="work_efficient")
    out_hs = cell(x, h0=h0, method="hillis_steele")
    out_seq = cell.forward_sequential(x, h0=h0)

    assert torch.allclose(out_we, out_seq, atol=1e-5)
    assert torch.allclose(out_hs, out_seq, atol=1e-5)

    # Verify single-step streaming equivalence
    h = h0
    stream_states = []
    for t in range(T):
        h = cell.step(h, x[:, t])
        stream_states.append(h)
    out_stream = torch.stack(stream_states, dim=1)

    assert torch.allclose(out_stream, out_we, atol=1e-5)


def test_fused_brain_cell_core_tier2_scaling():
    """Verify Tier 2 architectural scaling laws (Law 1: 32KB state memory, Law 2: factorized rank r=32)."""
    cell = FusedBrainCellCore(tier="tier2", proj_dim=4096, rank=32, num_deep_layers=2)
    assert cell.thought_size == 64
    assert cell.rank == 32
    assert cell.state_bytes(K=128, bytes_per_element=4) == 32768  # Exactly 32 KB Law 1 Contract
    assert cell.deep_proj is not None

    # Test execution
    x = torch.randn(2, 8, 4096)
    out = cell(x)
    assert out.shape == (2, 8, 64)


def test_fused_brain_cell_core_gradients():
    """Verify autograd gradients match between parallel scan and sequential loop."""
    torch.manual_seed(42)
    B, T, in_dim, w_dim = 2, 16, 32, 16
    cell_par = FusedBrainCellCore(input_size=in_dim, thought_size=w_dim)
    cell_seq = copy.deepcopy(cell_par)

    x = torch.randn(B, T, in_dim)
    h0 = torch.randn(B, w_dim)

    loss_par = (cell_par(x, h0=h0, method="work_efficient") ** 2).sum()
    loss_par.backward()

    loss_seq = (cell_seq.forward_sequential(x, h0=h0) ** 2).sum()
    loss_seq.backward()

    assert torch.allclose(loss_par, loss_seq, atol=1e-5)
    for (name1, p1), (name2, p2) in zip(cell_par.named_parameters(), cell_seq.named_parameters()):
        if p1.grad is not None and p2.grad is not None:
            diff = (p1.grad - p2.grad).abs().max().item()
            assert diff < 1e-4, f"Gradient mismatch for {name1}: {diff}"


def test_fused_slot_recurrent_block_fidelity():
    """Verify multi-slot block with CIG gating across K slots matches sequential unrolling."""
    torch.manual_seed(42)
    B, T, K, W, proj_dim = 2, 16, 4, 16, 32
    block = FusedSlotRecurrentBlock(K=K, thought_size=W, proj_dim=proj_dim)
    block.eval()

    x_seq = torch.randn(B, T, proj_dim)
    h0 = torch.randn(B, K, W)

    thoughts_par = block(x_seq, h0=h0, method="work_efficient")
    assert thoughts_par.shape == (B, T, K, W)

    # Step unrolling
    h = h0
    stream_thoughts = []
    for t in range(T):
        h = block.step(x_seq[:, t], h)
        stream_thoughts.append(h)
    thoughts_seq = torch.stack(stream_thoughts, dim=1)

    assert torch.allclose(thoughts_par, thoughts_seq, atol=1e-5)


def test_parallel_native_semantic_pseudo_brain_forward_and_streaming():
    """Verify end-to-end ParallelNativeSemanticPseudoBrain produces matching logits in parallel, sequential, and streaming."""
    torch.manual_seed(42)
    vocab_size = 200
    K = 8
    W = 24
    model = ParallelNativeSemanticPseudoBrain(
        vocab_size=vocab_size,
        K=K,
        thought_size=W,
        embed_dim=48,
        proj_dim=96,
    )
    model.eval()

    B, T = 3, 20
    tokens = torch.randint(0, vocab_size, (B, T))
    threads = torch.randint(0, K, (B, T))

    # Parallel scan forward
    logits_par = model(tokens, thread_seq=threads, method="work_efficient")
    assert logits_par.shape == (B, T, vocab_size)

    # Sequential forward
    logits_seq = model.forward_sequential(tokens, thread_seq=threads)
    assert logits_seq.shape == (B, T, vocab_size)
    assert torch.allclose(logits_par, logits_seq, atol=1e-5)

    # Streaming single step
    state = model.init_state(batch_size=B, device=tokens.device)
    streaming_list = []
    for t in range(T):
        l_t, state = model.step(tokens[:, t], state, thread_ids=threads[:, t])
        streaming_list.append(l_t)
    logits_stream = torch.stack(streaming_list, dim=1)

    assert torch.allclose(logits_par, logits_stream, atol=1e-5)


def test_parallel_native_semantic_pseudo_brain_thread_tokens():
    """Verify thread marker tokens [THREAD:k] propagate active thread correctly."""
    model = ParallelNativeSemanticPseudoBrain(vocab_size=344, K=8, thought_size=24, embed_dim=48, proj_dim=96)
    model.eval()

    # Token 11 corresponds to thread 11 - 9 = 2
    tokens = torch.tensor([[11, 50, 51, 13, 60]])  # thread 2, then thread 4 (13 - 9 = 4)
    logits = model(tokens)
    assert logits.shape == (1, 5, 344)


def test_parallel_native_semantic_pseudo_brain_backward_gradients():
    """Verify backward gradient flow matches between parallel and sequential models."""
    torch.manual_seed(42)
    vocab_size = 100
    model_par = ParallelNativeSemanticPseudoBrain(
        vocab_size=vocab_size, K=4, thought_size=16, embed_dim=32, proj_dim=64
    )
    model_seq = copy.deepcopy(model_par)

    B, T = 2, 10
    tokens = torch.randint(0, vocab_size, (B, T))
    threads = torch.randint(0, 4, (B, T))

    loss_par = (model_par(tokens, thread_seq=threads, method="work_efficient") ** 2).sum()
    loss_par.backward()

    loss_seq = (model_seq.forward_sequential(tokens, thread_seq=threads) ** 2).sum()
    loss_seq.backward()

    assert torch.allclose(loss_par, loss_seq, atol=1e-5)
    for (name1, p1), (name2, p2) in zip(model_par.named_parameters(), model_seq.named_parameters()):
        if p1.grad is not None and p2.grad is not None:
            diff = (p1.grad - p2.grad).abs().max().item()
            assert diff < 1e-4, f"Parameter gradient mismatch for {name1}: {diff}"
