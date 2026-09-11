"""Bounded native gate for ingestion from a carried recurrent state."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import time

import torch
from run_provenance import apply_deterministic_mode, provenance
from irene_brain.evaluation.recurrent_ingest import parallel_recurrent_ingest
from irene_brain.evaluation.recurrent_prefill import parallel_recurrent_prefill
from irene_brain.unified.parallel_depth_model import ParallelDepthModel


def decoded(state):
    return state[:, :8].double() + state[:, 8:].double()


@torch.no_grad()
def compare(model, ids, state, prompt, resets, chunks):
    reference = state.clone()
    original_state = state.clone()
    original_prompt = prompt.clone()
    actual_state = state
    offset = 0
    result = {'tokens': ids.shape[1], 'batch': ids.shape[0], 'chunks': chunks,
              'max_boundary_logit_error': 0., 'max_boundary_state_error': 0.}
    for size in chunks:
        for index in range(offset, offset + size):
            reference *= 1 - resets[:, index, None, None]
            expected, reference = model.step(ids[:, index], reference, prompt)
            assert torch.isfinite(expected).all() and torch.isfinite(reference).all()
        actual, actual_state = parallel_recurrent_ingest(
            model, ids[:, offset:offset+size], actual_state, prompt, resets[:, offset:offset+size])
        assert torch.isfinite(actual).all() and torch.isfinite(actual_state).all()
        assert actual_state.shape == (ids.shape[0], 16, 64) and actual_state.dtype == torch.float32
        result['state_bytes_per_stream'] = actual_state[0].numel() * actual_state.element_size()
        assert result['state_bytes_per_stream'] == 4096
        result['max_boundary_logit_error'] = max(result['max_boundary_logit_error'], (actual-expected).abs().max().item())
        result['max_boundary_state_error'] = max(result['max_boundary_state_error'], (decoded(actual_state)-decoded(reference)).abs().max().item())
        assert result['max_boundary_logit_error'] < 1e-6
        assert result['max_boundary_state_error'] < 1e-12
        offset += size
    assert offset == ids.shape[1]
    assert torch.equal(state, original_state) and torch.equal(prompt, original_prompt)
    result['continuation_logit_error'] = 0.
    # Predetermined continuation, not replayed earlier tokens or a history cache.
    for index in range(7):
        token = torch.full((ids.shape[0],), 53+index, dtype=torch.long, device=ids.device)
        expected, reference = model.step(token, reference, prompt)
        actual, actual_state = model.step(token, actual_state, prompt)
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        assert torch.isfinite(actual_state).all() and torch.isfinite(reference).all()
        result['continuation_logit_error'] = max(result['continuation_logit_error'], (actual-expected).abs().max().item())
    assert result['continuation_logit_error'] < 1e-6
    result['status'] = 'passed'
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available(), 'Requires authorized remote native CUDA'
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == args.checkpoint_sha256
    torch.set_num_threads(1)
    apply_deterministic_mode()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'checkpoint_sha256': args.checkpoint_sha256, 'cases': [],
              'gpu': torch.cuda.get_device_name(), 'source_hashes': {}}
    for name, obj in [('ingest', parallel_recurrent_ingest), ('prefill', parallel_recurrent_prefill), ('model', ParallelDepthModel)]:
        report['source_hashes'][name] = hashlib.sha256(Path(inspect.getfile(inspect.unwrap(obj))).read_bytes()).hexdigest()
    report['source_hashes']['runner'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    started = time.monotonic()
    def deadline(*unused):
        raise TimeoutError('Registered 900-second native ingestion gate limit reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(900)
    try:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model = ParallelDepthModel.from_payload(payload).cuda().eval()
        del payload
        bank = hashlib.sha256()
        generator = torch.Generator(device='cuda').manual_seed(1064)
        with torch.no_grad():
            for length, batch, reset in [(1,1,False), (31,1,False), (257,1,False), (2048,1,False),
                                          (2049,1,False), (32768,1,False), (257,2,True)]:
                warm = torch.randint(1, model.config.vocab_size, (batch,31), device='cuda', generator=generator)
                prompt = warm.clone()
                _, state = parallel_recurrent_prefill(model, warm, prompt)
                assert decoded(state).abs().max().item() > 0
                ids = torch.randint(1, model.config.vocab_size, (batch,length), device='cuda', generator=generator)
                if length == 32768:
                    ids[:,4096:8192] = ids[:,4096:4097].clone()
                resets = torch.zeros_like(ids)
                if reset:
                    resets[0,length//2] = 1
                    resets[1,-1] = 1
                for tensor in (warm, ids, resets):
                    bank.update(tensor.cpu().numpy().tobytes())
                result = compare(model, ids, state, prompt, resets, [length])
                result['resets'] = reset
                report['cases'].append(result)
                if length == 257 and batch == 1:
                    report['cases'].append(compare(model, ids, state, prompt, resets, [1,3,13,31,61,148]))
                (args.output/'progress.json').write_text(json.dumps(report, indent=2))
                print('INGEST_GPU_CASE', json.dumps(result), flush=True)
        report['provenance'] = provenance(model=model, train_seed=198, eval_seed=1064,
                                           bank_digest=bank.hexdigest(), deterministic=True)
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed' if isinstance(exc, AssertionError) else 'incomplete',
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic() - started
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        print('INGEST_GPU_RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
