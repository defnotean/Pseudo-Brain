"""Bounded sequential controller for the fixed-prefix pointer diagnostic."""
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


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'training-run', 'corpus', 'tokenizer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    training_report = args.training_run / 'report.json'
    assert digest(training_report) == '1b5cf6883fcc3ad536c96ac0c06bcac354c1c5c96622ad8426953106b605a0b4'
    assert json.loads(training_report.read_text())['status'] == 'complete'
    manifest = args.root / 'diagnostic-source-hashes.json'
    for relative, expected in json.loads(manifest.read_text()).items():
        assert digest(args.root / relative) == expected, relative
    args.output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'running', 'models': {}, 'source_manifest_sha256': digest(manifest),
              'training_report_sha256': digest(training_report), 'optimizer_updates': 0, 'generated_tokens': 0}
    started = time.monotonic()
    try:
        for name in ('recurrent', 'transformer'):
            (args.output / 'status.json').write_text(json.dumps({'stage': 'diagnosing', 'model': name}))
            command = [sys.executable, str(args.root / 'diagnose_policy_copy_readout.py'), '--model', name,
                '--checkpoint', str(args.training_run / name / 'checkpoint-3520-trained.pt'),
                '--corpus', str(args.corpus), '--tokenizer', str(args.tokenizer), '--output', str(args.output / name)]
            with (args.output / (name + '.log')).open('x') as log:
                try:
                    process = subprocess.run(command, env=os.environ.copy(), cwd=args.root,
                                             stdout=log, stderr=subprocess.STDOUT, timeout=330)
                    returncode, timed_out = process.returncode, False
                except subprocess.TimeoutExpired:
                    returncode, timed_out = None, True
            path = args.output / name / 'report.json'
            child = json.loads(path.read_text()) if path.exists() else None
            complete = returncode == 0 and child is not None and child['status'] == 'complete' and len(child['records']) == 56
            report['models'][name] = {'status': 'complete' if complete else 'incomplete',
                                      'returncode': returncode, 'timed_out': timed_out,
                                      'report_sha256': digest(path) if path.exists() else None,
                                      'groups': child.get('groups', {}) if child else {}}
            (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        report['status'] = 'complete' if all(row['status'] == 'complete' for row in report['models'].values()) else 'incomplete'
    except BaseException as exc:
        report.update(status='incomplete', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - started
        (args.output / 'report.json').write_text(json.dumps(report, indent=2))
        (args.output / 'status.json').write_text(json.dumps({'stage': report['status']}))
        print('POLICY_COPY_DIAGNOSTIC_PAIR', report['status'], flush=True)


if __name__ == '__main__':
    main()
