"""Bounded native correctness/timing gate for the opt-in prefill path."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import signal
import statistics
import time

import torch
from irene_brain.evaluation.recurrent_prefill import parallel_recurrent_prefill
from irene_brain.unified.parallel_depth_model import ParallelDepthModel


@torch.no_grad()
def serial_prefill(model, ids, prompt, resets):
    state = model.init_state(ids.shape[0])
    for index in range(ids.shape[1]):
        state *= 1-resets[:,index,None,None]
        logits,state = model.step(ids[:,index],state,prompt)
    return logits,state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Requires authorized remote native CUDA')
    torch.set_num_threads(1)
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == args.checkpoint_sha256
    args.output.mkdir(parents=True,exist_ok=False)
    report = {'status':'running','checkpoint_sha256':args.checkpoint_sha256,'cases':[],
              'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'helper_sha256':hashlib.sha256(Path(inspect.getfile(inspect.unwrap(parallel_recurrent_prefill))).read_bytes()).hexdigest(),
              'model_sha256':hashlib.sha256(Path(inspect.getfile(ParallelDepthModel)).read_bytes()).hexdigest()}
    started = time.monotonic()
    def deadline(*unused):
        raise TimeoutError('Registered900-second prefill gate limit reached')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(900)
    try:
        payload = torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        model = ParallelDepthModel.from_payload(payload).cuda().eval()
        del payload
        generator = torch.Generator(device='cuda').manual_seed(1063)
        with torch.no_grad():
            for length in (31,257,2049,32768):
                case = {'tokens':length,'status':'running','max_all_position_logits':0.}
                report['cases'].append(case)
                ids = torch.randint(1,model.config.vocab_size,(1,length),device='cuda',generator=generator)
                if length == 32768:
                    ids[:,4096:8192] = ids[:,4096:4097].clone()
                prompt = ids[:,:31 if length==32768 else length-1].clone()
                resets = torch.zeros_like(ids)
                resets[:,0] = 1
                resets[:,length//2] = 1
                features = model.features_parallel(ids,resets)
                state = model.init_state(1)
                for offset in range(0,length,256):
                    end = min(offset+256,length)
                    reference = model.readout(features[:,offset:end],ids[:,offset:end],prompt)
                    assert torch.isfinite(reference).all()
                    maximum = torch.zeros((),dtype=torch.float64,device='cuda')
                    finite = torch.ones((),dtype=torch.bool,device='cuda')
                    for index in range(offset,end):
                        state *= 1-resets[:,index,None,None]
                        serial_logits,state = model.step(ids[:,index],state,prompt)
                        maximum = torch.maximum(maximum,(serial_logits-reference[:,index-offset]).abs().max())
                        finite &= torch.isfinite(serial_logits).all() & torch.isfinite(state).all()
                    assert finite.item()
                    case['max_all_position_logits'] = max(case['max_all_position_logits'],maximum.item())
                    assert case['max_all_position_logits'] < 1e-6
                    if end%4096==0:
                        print('PREFILL_GATE_POSITIONS',length,end,case['max_all_position_logits'],flush=True)
                del features,reference
                logits,parallel_state = parallel_recurrent_prefill(model,ids,prompt,resets)
                assert torch.isfinite(logits).all() and torch.isfinite(parallel_state).all()
                case['final_logit_error'] = (logits-serial_logits).abs().max().item()
                decoded = lambda value:value[:,:8].double()+value[:,8:].double()
                case['state_error'] = (decoded(parallel_state)-decoded(state)).abs().max().item()
                case['state_bytes'] = parallel_state.numel()*parallel_state.element_size()
                assert parallel_state.dtype==torch.float32 and case['state_bytes']==4096
                assert case['final_logit_error'] < 1e-6 and case['state_error'] < 1e-12
                continuation = torch.randint(1,model.config.vocab_size,(1,7),device='cuda',generator=generator)
                case['continuation_error'] = 0.
                for index in range(7):
                    expected,state = model.step(continuation[:,index],state,prompt)
                    actual,parallel_state = model.step(continuation[:,index],parallel_state,prompt)
                    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
                    case['continuation_error'] = max(case['continuation_error'],(actual-expected).abs().max().item())
                assert case['continuation_error'] < 1e-6
                if length < 32768:
                    timings = {'serial':[],'parallel':[]}
                    for repetition in range(3):
                        order = ('serial','parallel') if repetition%2==0 else ('parallel','serial')
                        for mode in order:
                            torch.cuda.synchronize()
                            start = time.perf_counter()
                            if mode=='serial':
                                serial_prefill(model,ids,prompt,resets)
                            else:
                                parallel_recurrent_prefill(model,ids,prompt,resets)
                            torch.cuda.synchronize()
                            timings[mode].append(time.perf_counter()-start)
                    case['timings_seconds'] = timings
                    case['median_speedup'] = statistics.median(timings['serial'])/statistics.median(timings['parallel'])
                    if length in (257,2049):
                        assert case['median_speedup'] >= 2., 'Registered prefill speed gate failed'
                case['status'] = 'passed'
                (args.output/'progress.json').write_text(json.dumps(report,indent=2))
                print('PREFILL_CASE_COMPLETE',json.dumps(case),flush=True)
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed' if isinstance(exc,AssertionError) else 'incomplete',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('PREFILL_GPU_RESULT',json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
