"""GPU-only independent gradient checks for the FP64 recurrent candidate.

Run on an authorized remote GPU. A plain serial recurrence supplies the
reference; it does not call either associative scan implementation.
"""
import copy
import importlib

import pytest
import torch

from irene_brain.unified.parallel_depth_model import ParallelDepthConfig, ParallelDepthModel

scan_module = importlib.import_module('irene_brain.parallel.triton_scan')
depth_module = importlib.import_module('irene_brain.unified.parallel_depth_model')
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not scan_module.is_triton_available(),
    reason='Requires authorized remote CUDA with native Triton',
)


def serial_reference(a, b, h0=None):
    state = torch.zeros_like(b[:, 0]) if h0 is None else h0
    states = []
    for index in range(b.shape[1]):
        state = a[:, index] * state + b[:, index]
        states.append(state)
    return torch.stack(states, dim=1)


@pytest.mark.parametrize('length', [31, 257, 2048, 2049])
@pytest.mark.parametrize('has_initial_state', [False, True])
def test_fp64_scan_matches_serial_gradients(length, has_initial_state, monkeypatch):
    torch.manual_seed(811 + length)
    a = .9 + .0999 * torch.rand(2, length, 64, device='cuda', dtype=torch.float64)
    # Include exact reset boundaries as well as long retention times.
    a[:, length // 2] = 0
    b = torch.randn_like(a) * .05
    h0 = torch.randn(2, 64, device='cuda', dtype=torch.float64)
    inputs = [a, b] + ([h0] if has_initial_state else [])
    native_inputs = [item.detach().clone().requires_grad_() for item in inputs]
    reference_inputs = [item.detach().clone().requires_grad_() for item in inputs]
    calls = []
    original = scan_module._run_triton_scan

    def observed_native(*args, **kwargs):
        assert args[0].is_cuda
        calls.append(args[0].shape[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(scan_module, '_run_triton_scan', observed_native)
    actual = scan_module.triton_scan(*native_inputs)
    expected = serial_reference(*reference_inputs)
    upstream = torch.randn_like(actual)
    native_grads = torch.autograd.grad((actual * upstream).sum(), native_inputs)
    reference_grads = torch.autograd.grad((expected * upstream).sum(), reference_inputs)
    assert calls == ([length, length] if length <= 2048 else [])
    torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    for actual_grad, expected_grad in zip(native_grads, reference_grads):
        assert torch.isfinite(actual_grad).all()
        torch.testing.assert_close(actual_grad, expected_grad, atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize('length', [31, 257])
def test_candidate_parameter_gradients_match_serial_reference(length, monkeypatch):
    torch.manual_seed(931 + length)
    native = ParallelDepthModel(ParallelDepthConfig(vocab_size=128, width=32)).cuda()
    reference = copy.deepcopy(native)
    ids = torch.randint(1, 128, (2, length), device='cuda')
    labels = torch.roll(ids, -1, dims=1)
    labels[:, :7] = -100
    labels[:, -1] = -100
    prompt = ids[:, :7].clone()
    resets = torch.zeros_like(ids, dtype=torch.bool)
    resets[:, length // 2] = True
    pointer_mask = (~resets).double()
    arguments = dict(reset_mask=resets, prompt_tokens=prompt, pointer_mask=pointer_mask, chunk_size=16)
    actual_loss = native.language_loss(ids, labels, **arguments)
    actual_loss.backward()
    monkeypatch.setattr(depth_module, 'triton_scan', serial_reference)
    reference_loss = reference.language_loss(ids, labels, **arguments)
    reference_loss.backward()
    torch.testing.assert_close(actual_loss, reference_loss, atol=1e-8, rtol=1e-8)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in native.named_parameters():
        expected = reference_parameters[name].grad
        assert parameter.grad is not None and expected is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        torch.testing.assert_close(parameter.grad, expected, atol=1e-8, rtol=1e-8, msg=name)
