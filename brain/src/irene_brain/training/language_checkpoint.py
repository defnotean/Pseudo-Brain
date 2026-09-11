"""Optimizer-boundary checkpoints for the parallel language model pilots.

Reuse the existing atomic checkpoint envelope and its run-identity checks.
Call only after optimizer.step() and zero_grad(set_to_none=True). The caller
owns the deterministic data cursor and learning-rate schedule in config_sha256.
This does not retrofit optimizer state into older inference checkpoints.
"""
from dataclasses import asdict
import os
import random

import torch

from .checkpoint import load_checkpoint, save_checkpoint


def language_runtime_fingerprint(model):
    device = next(model.parameters()).device
    return {
        'torch': str(torch.__version__),
        'cuda': str(torch.version.cuda),
        'device_type': device.type,
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'cpu_threads': torch.get_num_threads(),
        'deterministic': torch.are_deterministic_algorithms_enabled(),
        'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
        'matmul_precision': torch.get_float32_matmul_precision(),
        'matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_tf32': torch.backends.cudnn.allow_tf32,
        'cudnn_benchmark': torch.backends.cudnn.benchmark,
        'cudnn_deterministic': torch.backends.cudnn.deterministic,
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG', ''),
    }


def _identity(model, optimizer):
    named = dict(model.named_parameters())
    names = {id(value): key for key, value in named.items()}
    groups, observed = [], []
    for group in optimizer.param_groups:
        group_names = []
        for parameter in group['params']:
            if id(parameter) not in names:
                raise ValueError('Optimizer contains a parameter outside the model')
            group_names.append(names[id(parameter)])
        groups.append(group_names)
        observed.extend(group_names)
    if len(observed) != len(set(observed)) or set(observed) != set(named):
        raise ValueError('Language optimizer must own every model parameter exactly once')
    if len({parameter.device for parameter in named.values()}) != 1:
        raise ValueError('Language checkpoint requires one model device')
    if any(parameter.grad is not None for parameter in named.values()):
        raise ValueError('Checkpoint requires cleared gradients at an optimizer boundary')
    return {
        'architecture': model.architecture,
        'config': asdict(model.config),
        'parameters': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                              'requires_grad': value.requires_grad} for key, value in named.items()},
        'optimizer_class': type(optimizer).__module__ + '.' + type(optimizer).__qualname__,
        'optimizer_parameter_names': groups,
    }


def save_language_boundary(path, *, model, optimizer, cursor, config_sha256,
                           data_sha256, code_sha256, policy):
    identity = _identity(model, optimizer)
    device = next(model.parameters()).device
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return save_checkpoint(
        path, cursor=cursor,
        system_state={'identity': identity, 'model': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'training': model.training},
        rng_state={'python': random.getstate(), 'torch_cpu': torch.get_rng_state(),
                   'torch_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else []},
        config_sha256=config_sha256, data_sha256=data_sha256, code_sha256=code_sha256,
        runtime_fingerprint=language_runtime_fingerprint(model), policy=policy)


def restore_language_boundary(path, *, model, optimizer, config_sha256,
                              data_sha256, code_sha256, checkpoint_sha256):
    """Restore into freshly constructed objects; discard them if restoration fails."""
    identity = _identity(model, optimizer)
    loaded = load_checkpoint(
        path, expected_config_sha256=config_sha256, expected_data_sha256=data_sha256,
        expected_code_sha256=code_sha256, expected_checkpoint_sha256=checkpoint_sha256,
        expected_runtime_fingerprint=language_runtime_fingerprint(model))
    state = loaded.system_state
    if set(state) != {'identity', 'model', 'optimizer', 'training'} or state['identity'] != identity:
        raise ValueError('Language model or optimizer identity mismatch')
    if type(state['training']) is not bool:
        raise ValueError('Invalid saved training mode')
    rng = loaded.rng_state
    if set(rng) != {'python', 'torch_cpu', 'torch_cuda'}:
        raise ValueError('Invalid language RNG state')
    device = next(model.parameters()).device
    expected_cuda_count = torch.cuda.device_count() if device.type == 'cuda' else 0
    if not isinstance(rng['torch_cuda'], list) or len(rng['torch_cuda']) != expected_cuda_count:
        raise ValueError('Language CUDA RNG device count mismatch')
    # Validate CPU/Python states without changing process-global generators.
    random.Random().setstate(rng['python'])
    torch.Generator().set_state(rng['torch_cpu'])
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    model.train(state['training'])
    random.setstate(rng['python'])
    torch.set_rng_state(rng['torch_cpu'])
    if device.type == 'cuda':
        torch.cuda.set_rng_state_all(rng['torch_cuda'])
    return loaded.cursor
