"""Generate from an explicit pilot checkpoint without opening any grader data."""
import argparse
import hashlib
import inspect
import json
import signal
import time
from pathlib import Path
import torch
from tokenizers import Tokenizer
from irene_brain.evaluation.independent_pilot import load_requests, greedy_decode, validate_streaming_parity
from irene_brain.unified.parallel_depth_model import ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel-prefill', action='store_true',
                        help='Use verified parallel initial-context ingestion for the recurrent model')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Generation requires the authorized remote GPU runtime')
    torch.set_num_threads(1)
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if checkpoint_hash != args.checkpoint_sha256:
        raise ValueError('Checkpoint hash mismatch')
    manifest = json.loads(args.manifest.read_text())
    raw = args.requests.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['requests_sha256']:
        raise ValueError('Request hash mismatch')
    requests = load_requests(raw)
    if [[row['benchmark'], row['task_id']] for row in requests] != manifest['selected']:
        raise ValueError('Request selection differs from manifest')
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    classes = {cls.architecture: cls for cls in (ParallelDepthModel, ComparisonTransformer)}
    model = classes[payload['architecture']].from_payload(payload).cuda().eval()
    tokenizer = Tokenizer.from_str(payload['tokenizer_json'])
    eos_id, resp_id = tokenizer.token_to_id('[EOS]'), tokenizer.token_to_id('[RESP]')
    if eos_id is None or resp_id is None:
        raise ValueError('Missing boundary tokens')
    recurrent = payload['architecture'] == ParallelDepthModel.architecture
    if args.parallel_prefill and not recurrent:
        raise ValueError('--parallel-prefill requires the recurrent model')
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_training_metadata': payload.get('pilot_metadata'),
        'architecture': payload['architecture'], 'parameters': sum(p.numel() for p in model.parameters()),
        'requests_sha256': manifest['requests_sha256'], 'max_new_tokens': 512,
        'decoding': 'greedy', 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
        'completed_tasks': 0, 'correctness_scored': False,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    report['prefill_mode'] = ('parallel' if args.parallel_prefill else 'serial') if recurrent else 'full_prefix'
    if args.parallel_prefill:
        from irene_brain.evaluation.recurrent_prefill import parallel_recurrent_prefill
        helper_path = Path(inspect.getfile(inspect.unwrap(parallel_recurrent_prefill)))
        report['prefill_helper_sha256'] = hashlib.sha256(helper_path.read_bytes()).hexdigest()
    del payload
    start = time.monotonic()
    def timeout(*unused):
        raise TimeoutError('Independent generation exceeded 1800 seconds')
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(1800)
    try:
        if recurrent:
            report['trained_streaming_parity'] = validate_streaming_parity(model)
        with (args.output/'responses.jsonl').open('x') as stream:
            for request in requests:
                text = '[THREAD:0]User: ' + request['question'] + ' [RESP]'
                context = tokenizer.encode(text).ids
                if context[-1] != resp_id or resp_id in context[:-1]:
                    raise ValueError('Ambiguous response boundary')
                if len(context) + 512 > 32768:
                    raise ValueError('Evaluation exceeds registered context limit')
                ids = torch.tensor([context], device='cuda')
                result = greedy_decode(model, ids, ids[:, :-1], recurrent=recurrent,
                                       eos_id=eos_id, limit=512, parallel_prefill=args.parallel_prefill)
                if recurrent and result['recurrent_state_bytes'] != 4096:
                    raise RuntimeError('Recurrent state budget changed')
                row = {key: request[key] for key in ('benchmark', 'task_id')}
                row.update(result, response=tokenizer.decode(result['token_ids'], skip_special_tokens=False))
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                report['completed_tasks'] += 1
                print('GENERATED', row['benchmark'], row['task_id'], len(result['token_ids']), flush=True)
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='incomplete', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic() - start
        (args.output/'generation.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
