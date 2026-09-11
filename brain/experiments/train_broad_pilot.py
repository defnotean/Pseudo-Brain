"""Fixed-order, single-pass recurrent/transformer learning comparison on Colab."""
import argparse
import gc
import gzip
import hashlib
import json
import math
from pathlib import Path
import signal
import time
import traceback
import torch
from irene_brain.data.broad_corpus import assert_isolated
from irene_brain.data.streaming_loader import SequencePacker, RawDocument, TaskType
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.unified.parallel_depth_model import ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer

PROMPTS = [
    'Explain why the Moon has phases in two sentences.',
    'Write a Python function count_unique(items) that returns the number of distinct hashable items.',
    'A box contains 17 red marbles and 9 blue marbles. Four red marbles are removed. How many marbles remain?',
    'Every wug is a dax. No dax is a mip. Can a wug be a mip? Explain briefly.',
    'Write a Python function clamp_value(x, low, high) that limits x to the inclusive interval.',
    'Explain the difference between a necessary condition and a sufficient condition.',
]


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def load_partition(corpus, partition, manifest):
    packed = (corpus/(partition + '.jsonl.gz')).read_bytes()
    expected = manifest['partitions'][partition]
    assert hashlib.sha256(packed).hexdigest() == expected['gzip_sha256']
    raw = gzip.decompress(packed)
    assert hashlib.sha256(raw).hexdigest() == expected['jsonl_sha256']
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    assert len(rows) == expected['rows']
    return rows


def encode_rows(rows, tokenizer):
    result = []
    for row in rows:
        packer = SequencePacker(tokenizer, seq_len=row['token_count'], max_threads=1,
            supervised_only_loss=True, document_policy='whole')
        packer.append_document(RawDocument(row['text'], TaskType(row['task'])))
        block = packer.emit_block()
        assert not packer._token_buffer
        supervised = int((block.labels != -100).sum())
        assert supervised == row['supervised_tokens']
        prompt_end = (block.input_ids == tokenizer.resp_id).nonzero()[0].item()
        assert prompt_end > 0
        result.append((row, block.input_ids[None], block.labels[None], block.input_ids[None, :prompt_end]))
    return result


def evaluate(model, examples, status, name, stage):
    model.eval()
    stats, details = {}, []
    with torch.no_grad():
        for index, (row, ids, labels, prompt) in enumerate(examples):
            loss = model.language_loss(ids.cuda(), labels.cuda(), prompt_tokens=prompt.cuda()).item()
            if not math.isfinite(loss):
                raise FloatingPointError('Nonfinite evaluation loss')
            group = stats.setdefault(row['task'], {'nll_sum': 0., 'supervised_tokens': 0, 'examples': 0})
            group['nll_sum'] += loss * row['supervised_tokens']
            group['supervised_tokens'] += row['supervised_tokens']
            group['examples'] += 1
            details.append({'text_sha256': row['text_sha256'], 'task': row['task'], 'nll': loss,
                            'supervised_tokens': row['supervised_tokens']})
            if (index + 1) % 32 == 0:
                write_json(status, {'stage': stage, 'model': name, 'evaluated': index + 1})
                print(name, stage, index + 1, flush=True)
    for group in stats.values():
        group['token_weighted_nll'] = group['nll_sum'] / group['supervised_tokens']
    model.train()
    return {'domains': stats, 'examples': details}


def generate_samples(model, tokenizer, recurrent):
    model.eval()
    samples = []
    with torch.no_grad():
        for question in PROMPTS:
            context = tokenizer.encode('[THREAD:0]User: ' + question + ' [RESP]')
            ids = torch.tensor([context], device='cuda')
            prompt = ids[:, :-1].clone()
            generated = []
            if recurrent:
                state = model.init_state(1)
                for token in ids[0]:
                    logits, state = model.step(token.reshape(1), state, prompt)
            else:
                logits = model(ids, prompt_tokens=prompt)[:, -1]
            for index in range(128):
                token = logits.argmax(-1)
                if token.item() == tokenizer.eos_id:
                    break
                generated.append(token.item())
                if recurrent:
                    logits, state = model.step(token, state, prompt)
                else:
                    ids = torch.cat((ids, token[:, None]), 1)
                    logits = model(ids, prompt_tokens=prompt)[:, -1]
            samples.append({'prompt': question, 'response': tokenizer.decode(generated),
                            'generated_tokens': len(generated), 'limit': 128})
    return samples


def check_trained_parity(model):
    model.eval()
    generator = torch.Generator(device='cuda').manual_seed(734)
    receipt = []
    with torch.no_grad():
        for length in (31, 257):
            ids = torch.randint(1, 32000, (1, length), device='cuda', generator=generator)
            prompt = ids[:, :11].clone()
            resets = torch.zeros_like(ids)
            resets[:, 13] = 1
            expected = model(ids, resets, prompt)
            state = model.init_state(1)
            worst = 0.
            for t in range(length):
                state *= 1 - resets[:, t, None, None]
                actual, state = model.step(ids[:, t], state, prompt)
                worst = max(worst, (expected[:, t] - actual).abs().max().item())
            receipt.append({'tokens': length, 'max_abs_logits': worst})
    return receipt


def run_model(name, cls, train, dev, tokenizer, manifest, output, status):
    destination = output/name
    destination.mkdir()
    torch.manual_seed(198)
    model = cls().cuda()
    report = {'architecture': model.architecture, 'parameters': sum(p.numel() for p in model.parameters()),
              'completed_updates': 0, 'training_tokens': 0, 'training_supervised_tokens': 0,
              'corpus_partitions': manifest['partitions'], 'status': 'running'}
    report['initial_development'] = evaluate(model, dev, status, name, 'initial_development')
    write_json(destination/'metrics.json', report)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), eps=1e-8, weight_decay=.1)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    def deadline(*args):
        raise TimeoutError('Registered 1800-second training limit reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(1800)
    training_error = None
    try:
        with (destination/'training.jsonl').open('x') as log:
            for index, (row, ids, labels, prompt) in enumerate(train):
                if index < 64:
                    lr = 3e-4 * (index + 1) / 64
                else:
                    progress = (index - 64) / (len(train) - 65)
                    lr = 3e-5 + .5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * progress))
                for group in optimizer.param_groups:
                    group['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                loss = model.language_loss(ids.cuda(), labels.cuda(), prompt_tokens=prompt.cuda())
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                report['completed_updates'] = index + 1
                report['training_tokens'] += row['token_count']
                report['training_supervised_tokens'] += row['supervised_tokens']
                log.write(json.dumps({'update': index + 1, 'text_sha256': row['text_sha256'],
                    'task': row['task'], 'loss': loss.item(), 'lr': lr, 'grad_norm': grad_norm.item(),
                    'elapsed_seconds': time.monotonic() - start}) + '\n')
                if (index + 1) % 32 == 0:
                    log.flush()
                    write_json(status, {'stage': 'training', 'model': name, 'updates': index + 1,
                        'total_updates': len(train), 'elapsed_seconds': time.monotonic() - start})
                    print(name, 'TRAIN', index + 1, 'loss', loss.item(), flush=True)
        report['status'] = 'trained'
    except Exception as exc:
        training_error = exc
        report.update(status='incomplete', error_type=type(exc).__name__, error=str(exc))
    finally:
        signal.alarm(0)
        torch.cuda.synchronize()
        report['training_seconds'] = time.monotonic() - start
        report['training_peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        payload = model.checkpoint_payload()
        payload['pilot_metadata'] = {k: v for k, v in report.items() if k != 'initial_development'}
        payload['tokenizer_json'] = tokenizer._hf_tokenizer.to_str()
        torch.save(payload, destination/'checkpoint.pt')
        report['checkpoint_sha256'] = hashlib.sha256((destination/'checkpoint.pt').read_bytes()).hexdigest()
        write_json(destination/'metrics.json', report)
    if training_error is not None:
        raise training_error
    del optimizer, loss, payload
    gc.collect()
    torch.cuda.empty_cache()
    report['final_development'] = evaluate(model, dev, status, name, 'final_development')
    report['diagnostic_samples'] = generate_samples(model, tokenizer, name == 'recurrent')
    if name == 'recurrent':
        report['trained_parity'] = check_trained_parity(model)
        report['trained_parity_passed'] = all(r['max_abs_logits'] < 1e-6 for r in report['trained_parity'])
        if not report['trained_parity_passed']:
            report['status'] = 'failed_parity'
            write_json(destination/'metrics.json', report)
            raise RuntimeError('Trained recurrent model failed parity')
    report['status'] = 'complete'
    write_json(destination/'metrics.json', report)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    status = args.output/'status.json'
    try:
        assert torch.cuda.is_available()
        torch.set_num_threads(1)
        manifest = json.loads((args.corpus/'manifest.json').read_text())
        token_path = Path(__file__).resolve().parents[1]/'data/tokenizers/pseudo_brain_bpe_32000.json'
        assert hashlib.sha256(token_path.read_bytes()).hexdigest() == manifest['tokenizer_sha256']
        tokenizer = BpeSemanticTokenizer(tokenizer_file=token_path, auto_build_if_missing=False)
        train_rows = load_partition(args.corpus, 'train', manifest)
        dev_rows = load_partition(args.corpus, 'development', manifest)
        assert_isolated(train_rows, dev_rows)
        counts = {}
        for name, cls in [('recurrent', ParallelDepthModel), ('transformer', ComparisonTransformer)]:
            temporary = cls()
            counts[name] = sum(p.numel() for p in temporary.parameters())
            del temporary
        assert max(counts.values()) < 36_000_000
        assert max(counts.values()) / min(counts.values()) < 1.05
        source_root = Path(__file__).resolve().parents[1]
        provenance = {'parameters': counts, 'gpu': torch.cuda.get_device_name(), 'torch_version': torch.__version__,
            'corpus_manifest_sha256': hashlib.sha256((args.corpus/'manifest.json').read_bytes()).hexdigest(),
            'source_hashes': {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((source_root/'src').rglob('*.py'))},
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        provenance['registration_hashes'] = {name: hashlib.sha256((source_root/'experiments'/name).read_bytes()).hexdigest()
            for name in ('broad_pilot_v1_registration.md', 'broad_pilot_v1_precision_addendum.md')}
        write_json(args.output/'provenance.json', provenance)
        print('ENCODING_FROZEN_CORPUS', len(train_rows), len(dev_rows), flush=True)
        train, dev = encode_rows(train_rows, tokenizer), encode_rows(dev_rows, tokenizer)
        for name, cls in [('recurrent', ParallelDepthModel), ('transformer', ComparisonTransformer)]:
            report = run_model(name, cls, train, dev, tokenizer, manifest, args.output, status)
            print('MODEL_COMPLETE', name, report['training_seconds'], flush=True)
        write_json(status, {'stage': 'complete', 'models': ['recurrent', 'transformer']})
    except Exception as exc:
        write_json(status, {'stage': 'failed', 'error_type': type(exc).__name__, 'error': str(exc)})
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
