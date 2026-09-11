"""Fixed-prefix pointer ablation after policy training; no optimizer or generation."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import torch
from torch.nn import functional as F
from run_provenance import apply_deterministic_mode
from irene_brain.data.executed_trajectory import encode_executed_trajectory
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.unified.parallel_depth_model import ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer


CHECKPOINTS = {'recurrent': '0b3edca04de150068c81fb8f71f05cd0aad8f42e423b9989bc789b919ffb85b7',
               'transformer': '346c9d4c9d334490cb2e93c54f4a1f0f66020a3833b4e7565f015b6d08c44dfe'}
BANK = {'train': '896a39d478aaa674903c27eb9c7af0d96017bc721843c860a3d56daa056d73a0',
        'development': '0381456052591928c18debafd5e86bf6dc1c304c1793bec2740e0c30f7cccae8'}
TOKENIZER = '760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def selected_records(corpus):
    selected = []
    for split, expected in BANK.items():
        path = corpus / (split + '.jsonl.gz')
        if digest(path) != expected:
            raise ValueError('Policy bank hash mismatch')
        rows = [json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
        if len(rows) != (352 if split == 'train' else 24):
            raise ValueError('Policy bank cardinality mismatch')
        indices = range(0, 352, 11) if split == 'train' else range(24)
        selected.extend((split, index, rows[index]) for index in indices)
    return selected


def action_regions(record, encoded, tokenizer):
    """Map each supervised next-token target to header, body or EOS."""
    regions = torch.full_like(encoded['labels'], -1)
    for step, span in zip(record['steps'], encoded['action_spans']):
        if span['kind'] != 'teacher':
            continue
        action = tokenizer._hf_tokenizer.encode(step['action'])
        start = span['start']
        if action.ids != encoded['labels'][0, start:span['eos_position'] - 1].tolist():
            raise ValueError('Action token offsets differ from training labels')
        newline = step['action'].find('\n')
        for index, (left, right) in enumerate(action.offsets):
            # A token touching the first newline belongs to the header. Its
            # characters cannot be divided without changing tokenization.
            regions[0, start + index] = 0 if newline < 0 or left <= newline else 1
        regions[0, span['eos_position'] - 1] = 2
    if not torch.equal(regions >= 0, encoded['labels'] != -100):
        raise ValueError('Region layout does not cover exactly supervised targets')
    return regions


def score_readouts(normal, base, targets, gates, present):
    if normal.shape != base.shape or normal.ndim != 2 or targets.shape != normal.shape[:1]:
        raise ValueError('Invalid readout metric shapes')
    if not len(targets) or not torch.isfinite(normal).all() or not torch.isfinite(base).all():
        raise ValueError('Readout metrics require nonempty finite logits')
    correct_normal, correct_base = normal.argmax(-1) == targets, base.argmax(-1) == targets
    return {'tokens': len(targets), 'normal_nll_sum': F.cross_entropy(normal, targets, reduction='sum').item(),
            'base_nll_sum': F.cross_entropy(base, targets, reduction='sum').item(),
            'normal_correct': int(correct_normal.sum()), 'base_correct': int(correct_base.sum()),
            'pointer_helped': int((correct_normal & ~correct_base).sum()),
            'pointer_hurt': int((correct_base & ~correct_normal).sum()),
            'argmax_changed': int((normal.argmax(-1) != base.argmax(-1)).sum()),
            'target_present_in_prompt': int(present.sum()), 'gate_sum': gates.double().sum().item()}


def add_group(groups, name, values):
    group = groups.setdefault(name, {key: 0 for key in values})
    for key, value in values.items():
        group[key] += value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=tuple(CHECKPOINTS), required=True)
    for name in ('checkpoint', 'corpus', 'tokenizer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    assert digest(args.checkpoint) == CHECKPOINTS[args.model] and digest(args.tokenizer) == TOKENIZER
    torch.set_num_threads(1)
    apply_deterministic_mode()
    torch.manual_seed(1988)
    torch.cuda.manual_seed_all(1988)
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'model': args.model, 'checkpoint_sha256': CHECKPOINTS[args.model],
              'optimizer_updates': 0, 'generated_tokens': 0, 'records': [], 'groups': {},
              'scope': 'Fixed teacher prefixes only; an isolated readout counterfactual, not a task score or generation ablation.'}
    started = time.monotonic()
    def deadline(*unused):
        raise TimeoutError('Registered 300-second model diagnostic bound')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(300)
    try:
        tokenizer = BpeSemanticTokenizer(tokenizer_file=args.tokenizer, auto_build_if_missing=False)
        cls = ParallelDepthModel if args.model == 'recurrent' else ComparisonTransformer
        model = cls.from_payload(torch.load(args.checkpoint, map_location='cpu', weights_only=True)).cuda().eval()
        report['parameter_count'] = sum(parameter.numel() for parameter in model.parameters())
        report['pointer_sequence_boost'] = model.ptr_seq_boost.item()
        assert report['parameter_count'] < 36000000
        selection = selected_records(args.corpus)
        report['selection'] = [{'split': split, 'index': index, 'record_sha256': row['record_sha256']}
                               for split, index, row in selection]
        with torch.inference_mode():
            for split, index, row in selection:
                encoded = encode_executed_trajectory(row, tokenizer, max_tokens=8192)
                regions = action_regions(row, encoded, tokenizer).cuda()
                ids, targets = encoded['input_ids'].cuda(), encoded['labels'].cuda()
                prompt = encoded['prompt_tokens'].cuda()
                features = model.features_parallel(ids, encoded['reset_mask'].cuda())
                groups = {}
                for start in range(0, ids.shape[1], 256):
                    end = start + 256
                    active = targets[:, start:end] != -100
                    if not bool(active.any()):
                        continue
                    hidden, inputs = features[:, start:end], ids[:, start:end]
                    normal = model.readout(hidden, inputs, prompt)[active]
                    base = model.readout(hidden, inputs, None)[active]
                    target = targets[:, start:end][active]
                    region = regions[:, start:end][active]
                    gates = torch.sigmoid(model.ptr_gate(hidden)).squeeze(-1)[active]
                    present = torch.isin(target, prompt[prompt != tokenizer.pad_id])
                    for name, mask in [('all', torch.ones_like(target, dtype=torch.bool)),
                                       ('header', region == 0), ('body', region == 1), ('eos', region == 2),
                                       ('target_in_prompt', present), ('target_not_in_prompt', ~present)]:
                        if bool(mask.any()):
                            add_group(groups, name, score_readouts(normal[mask], base[mask], target[mask], gates[mask], present[mask]))
                assert groups['all']['tokens'] == int((targets != -100).sum())
                assert sum(groups.get(name, {}).get('tokens', 0) for name in ('header', 'body', 'eos')) == groups['all']['tokens']
                for name, values in groups.items():
                    add_group(report['groups'], split + '/' + name, values)
                report['records'].append({'split': split, 'index': index, 'record_sha256': row['record_sha256'], 'groups': groups})
                (args.output / 'status.json').write_text(json.dumps({'stage': 'diagnosing', 'records': len(report['records']), 'total': 56}))
                del features, ids, targets, prompt, regions
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='incomplete', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        (args.output / 'status.json').write_text(json.dumps({'stage': report['status']}))
        print('POLICY_COPY_DIAGNOSTIC', report['status'], args.model, len(report['records']), flush=True)


if __name__ == '__main__':
    main()
