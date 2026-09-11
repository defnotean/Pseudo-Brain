"""Remote-only, bounded long-context parity diagnostic for the frozen v3 model."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import signal
import time

import torch
from irene_brain.unified.parallel_depth_model import ParallelDepthModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Requires authorized remote CUDA runtime')
    torch.set_num_threads(1)
    expected_hash = '723bf4dcf2db108a17f6864a21e623c02f27ce9f2f62b999c5921eb9aeec13eb'
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert checkpoint_hash == expected_hash
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status':'running','completed_positions':0,'max_abs_logits':0.,
              'checkpoint_sha256':checkpoint_hash,'seed':1063,'tokens':32768,
              'reset_positions':[0,16384],'chunks':[], 'gpu':torch.cuda.get_device_name(),
              'torch':torch.__version__, 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model_sha256':hashlib.sha256(Path(inspect.getfile(ParallelDepthModel)).read_bytes()).hexdigest()}
    start = time.monotonic()
    def deadline(*unused):
        raise TimeoutError('Registered900-second parity limit reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(900)
    try:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        model = ParallelDepthModel.from_payload(payload).cuda().eval()
        del payload
        torch.cuda.reset_peak_memory_stats()
        generator = torch.Generator(device='cuda').manual_seed(1063)
        ids = torch.randint(1, model.config.vocab_size, (1,32768), device='cuda', generator=generator)
        ids[:,4096:8192] = ids[:,4096:4097].clone()
        prompt = ids[:,:31].clone()
        resets = torch.zeros_like(ids)
        resets[:,0] = 1
        resets[:,16384] = 1
        with torch.no_grad():
            features = model.features_parallel(ids, resets)
            state = model.init_state(1)
            for offset in range(0,32768,256):
                end = offset + 256
                expected = model.readout(features[:,offset:end], ids[:,offset:end], prompt)
                assert torch.isfinite(expected).all()
                maximum = torch.zeros((),device='cuda',dtype=torch.float64)
                finite = torch.ones((),device='cuda',dtype=torch.bool)
                for index in range(offset,end):
                    state *= 1 - resets[:,index,None,None]
                    actual,state = model.step(ids[:,index],state,prompt)
                    assert state.shape == (1,16,64) and state.dtype == torch.float32
                    assert state.numel()*state.element_size() == 4096
                    maximum = torch.maximum(maximum,(actual-expected[:,index-offset]).abs().max())
                    finite &= torch.isfinite(actual).all() & torch.isfinite(state).all()
                assert finite.item(), 'Nonfinite streaming logits/state'
                difference = maximum.item()
                report['completed_positions'] = end
                report['max_abs_logits'] = max(report['max_abs_logits'],difference)
                report['chunks'].append({'end':end,'max_abs_logits':difference})
                if difference >= 1e-6:
                    raise AssertionError('Streaming/parallel difference exceeds frozen1e-6 bound')
                if end % 2048 == 0:
                    print('PARITY_PROGRESS',end,report['max_abs_logits'],flush=True)
                    (args.output/'progress.json').write_text(json.dumps(report,indent=2))
            report['status'] = 'passed'
            report['state_bytes'] = state.numel()*state.element_size()
    except Exception as exc:
        report.update(status='failed' if isinstance(exc,AssertionError) else 'incomplete',
                      error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-start
        report['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('MAX_CONTEXT_PARITY_RESULT',json.dumps({k:v for k,v in report.items() if k!='chunks'}),flush=True)


if __name__ == '__main__':
    main()
