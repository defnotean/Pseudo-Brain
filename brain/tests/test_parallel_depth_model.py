import os
import pytest
import torch
from torch.nn import functional as F
from irene_brain.unified.parallel_depth_model import ParallelDepthConfig, ParallelDepthModel


def candidate():
    torch.manual_seed(483)
    return ParallelDepthModel(ParallelDepthConfig(vocab_size=128, width=32, ff_multiplier=2)).to(
        os.environ.get('PB_DEPTH_TEST_DEVICE', 'cpu'))


@pytest.mark.parametrize('length', [31, 257])
def test_native_parallel_streaming_parity_and_constant_state(length):
    model = candidate().eval()
    device = model.embedding.weight.device
    ids = torch.randint(1, 128, (2, length), device=device)
    prompt = torch.tensor([[11, 17, 11, 29, 0], [0, 0, 0, 0, 0]], device=device)
    ids[:, 3] = 11
    resets = torch.zeros_like(ids)
    resets[0, 13] = resets[1, 19] = 1
    with torch.no_grad():
        parallel = model(ids, resets, prompt)
        state = model.init_state(2)
        steps = []
        for t in range(length):
            state = state * (1 - resets[:, t, None, None])
            output, state = model.step(ids[:, t], state, prompt)
            steps.append(output)
        assert (parallel - torch.stack(steps, 1)).abs().max() < 1e-6
    assert state.shape == (2, 16, 64)
    assert state.dtype == torch.float32 and state[0].numel() * state.element_size() == 4096


def test_causality_all_pad_and_reset_isolation():
    model = candidate().eval()
    device = model.embedding.weight.device
    ids = torch.randint(1, 128, (1, 27), device=device)
    changed = ids.clone()
    changed[:, 14:] = 7
    with torch.no_grad():
        original = model(ids)
        assert torch.equal(original[:, :14], model(changed)[:, :14])
        assert torch.equal(original, model(ids, prompt_tokens=torch.zeros(1, 4, dtype=torch.long, device=device)))
        resets = torch.zeros_like(ids)
        resets[:, 14] = 1
        separate = model(ids[:, 14:])
        assert (model(ids, resets)[:, 14:] - separate).abs().max() < 1e-6


def test_chunked_loss_and_gradients_match_dense_with_pointer_masks():
    model = candidate()
    device = model.embedding.weight.device
    ids = torch.randint(1, 128, (2, 19), device=device)
    labels = torch.roll(ids, -1, 1)
    labels[:, :5] = -100
    labels[0, -2:] = -100
    prompt = ids[:, :5].clone()
    pointer_mask = torch.ones_like(ids)
    pointer_mask[:, 3:8] = 0
    dense = F.cross_entropy(model(ids, prompt_tokens=prompt, pointer_mask=pointer_mask).flatten(0, 1), labels.flatten())
    dense.backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    chunked = model.language_loss(ids, labels, prompt_tokens=prompt, pointer_mask=pointer_mask, chunk_size=4)
    chunked.backward()
    assert abs(dense.item() - chunked.item()) < 1e-8
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert (p.grad - gradients[name]).abs().max() < 1e-8, name
    for index in range(8):
        for module in ('gates', 'candidate', 'state_out'):
            assert gradients[f'layers.{index}.{module}.weight'].abs().sum() > 0


def test_checkpoint_and_invalid_contracts():
    model = candidate().cpu().eval()
    restored = ParallelDepthModel.from_payload(model.checkpoint_payload()).eval()
    ids = torch.tensor([[4, 12, 9]])
    with torch.no_grad():
        assert torch.equal(model(ids), restored(ids))
    with pytest.raises(ValueError, match='No supervised'):
        model.language_loss(ids, torch.full_like(ids, -100))
    with pytest.raises(ValueError, match='Fast state'):
        model.step(ids[:, 0], torch.zeros(1, 8, 64))
    with pytest.raises(ValueError, match='eight logical'):
        ParallelDepthModel(ParallelDepthConfig(layers=4))
