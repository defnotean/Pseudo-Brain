"""Evaluate completed policy weights on the unchanged observed coding/math bank."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def run_child(command, env, log_path, timeout):
    started = time.monotonic()
    with log_path.open('x') as log:
        try:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
            return {'returncode': result.returncode, 'timed_out': False,
                    'elapsed_seconds': time.monotonic() - started}
        except subprocess.TimeoutExpired:
            return {'returncode': None, 'timed_out': True, 'elapsed_seconds': time.monotonic() - started}


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'binding-root', 'training-run', 'bank', 'tokenizer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--training-report-sha256', required=True)
    args = parser.parse_args()
    for manifest, expected in ((args.binding_root / 'post-training-source-hashes.json',
                               '98b85f02412d4b8bf02f8cea4b50322fe4815087b8eed8ec17d4f467653d9cee'),
                              (args.root / 'retention-source-hashes.json', None)):
        if expected is not None:
            assert digest(manifest) == expected
        for relative, checksum in json.loads(manifest.read_text()).items():
            assert digest(manifest.parent / relative) == checksum, relative
    sys.path[:0] = [str(args.binding_root), str(args.binding_root / 'src')]
    from evaluate_trained_tool_policy import resolve_completed_checkpoint
    checkpoints = {name: resolve_completed_checkpoint(args.training_run, args.training_report_sha256, name)
                   for name in ('recurrent', 'transformer')}
    for filename, checksum in {
        'requests.jsonl': '1afbb0d6b1f8c3ba4f78a4b50d1040c67ee0701d97660ef96b267728f913642a',
        'graders.jsonl': 'acd9144610d3b1fddd80e956aede28d90a7f266b82e9f2a85d2226ff8b89b63b',
    }.items():
        assert digest(args.bank / filename) == checksum
    tokenizer_sha = '760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
    assert digest(args.tokenizer) == tokenizer_sha
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'models': {}, 'planned_tasks_per_model': 64,
              'training_report_sha256': args.training_report_sha256,
              'source_manifest_sha256': digest(args.root / 'retention-source-hashes.json'),
              'tokenizer_sha256': tokenizer_sha,
              'scope': 'Observed development bank; retention diagnostic, not frontier qualification',
              'v4_reference': {'recurrent': {'HumanEval': 0, 'GSM8K': 1},
                               'transformer': {'HumanEval': 0, 'GSM8K': 0}}}
    started = time.monotonic()
    write_json(args.output / 'training-binding.json', {
        name: {'checkpoint': str(path), 'sha256': checksum, 'child_report': child}
        for name, (path, checksum, child) in checkpoints.items()})
    write_json(args.output / 'report.json', report)
    try:
        for name, (checkpoint, checksum, _) in checkpoints.items():
            write_json(args.output / 'status.json', {'stage': 'generation', 'model': name})
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(args.root / 'brain/src'))
            generation = run_child([sys.executable, str(args.root / 'brain/experiments/generate_policy_retention.py'),
                '--checkpoint', str(checkpoint), '--checkpoint-sha256', checksum,
                '--tokenizer', str(args.tokenizer), '--tokenizer-sha256', tokenizer_sha,
                '--requests', str(args.bank / 'requests.jsonl'), '--manifest', str(args.bank / 'manifest.json'),
                '--output', str(args.output / name)], env, args.output / (name + '-generation.log'), 1900)
            entry = {'status': 'incomplete', 'checkpoint_sha256': checksum,
                     'generation': generation, 'planned_tasks': {'HumanEval': 32, 'GSM8K': 32}}
            report['models'][name] = entry
            if (args.output / name / 'generation.json').exists() and (args.output / name / 'responses.jsonl').exists():
                env['CUDA_VISIBLE_DEVICES'] = '-1'
                write_json(args.output / 'status.json', {'stage': 'scoring', 'model': name})
                scoring = run_child([sys.executable, str(args.root / 'brain/experiments/score_independent_math.py'),
                    '--include-code', '--bank', str(args.bank), '--generation', str(args.output / name),
                    '--output', str(args.output / (name + '-scores.json'))], env,
                    args.output / (name + '-scoring.log'), 180)
                entry['scoring'] = scoring
                if scoring['returncode'] == 0:
                    scores = json.loads((args.output / (name + '-scores.json')).read_text())
                    entry['scores'] = {benchmark: {key: scores[benchmark][key] for key in ('correct', 'total', 'missing')}
                                       for benchmark in ('HumanEval', 'GSM8K')}
                    assert all(row['total'] == 32 for row in entry['scores'].values())
                    if (generation['returncode'] == 0 and scores['status'] == 'complete'
                            and scores['HumanEval']['status'] == 'complete'
                            and all(row['missing'] == 0 for row in entry['scores'].values())):
                        entry['status'] = 'complete'
            write_json(args.output / 'report.json', report)
            print('POLICY_RETENTION_MODEL_DONE', name, json.dumps(entry), flush=True)
        report['status'] = 'complete' if all(row['status'] == 'complete' for row in report['models'].values()) else 'incomplete'
    except BaseException as exc:
        report.update(status='incomplete', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(args.output / 'report.json', report)
        write_json(args.output / 'status.json', {'stage': report['status']})
        print('POLICY_RETENTION_DONE', report['status'], flush=True)


if __name__ == '__main__':
    main()
