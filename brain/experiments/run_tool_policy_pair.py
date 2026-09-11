"""Sequential bounded controller for the frozen two-model V4 policy baseline."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_provenance import provenance
from evaluate_tool_policy import BANK, aggregate, write_json


def main():
    parser = argparse.ArgumentParser()
    for name in ('root', 'corpus', 'tokenizer', 'recurrent-checkpoint', 'transformer-checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source_manifest = args.root / 'native-source-hashes.json'
    for relative, expected in json.loads(source_manifest.read_text()).items():
        assert hashlib.sha256((args.root / relative).read_bytes()).hexdigest() == expected, relative
    for name, expected in BANK.items():
        assert hashlib.sha256((args.corpus / name).read_bytes()).hexdigest() == expected, name
    requests = [json.loads(line) for line in (args.corpus / 'requests.jsonl').read_text().splitlines()]
    plan = [{'source_id': row['source_id'], 'mode': mode} for row in requests for mode in ('from_scratch', 'repair')]
    assert len(plan) == 48
    report = {'status': 'running', 'models': {}, 'source_manifest_sha256': hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
              'provenance': provenance(train_seed=198, eval_seed=198, bank_digest=BANK['requests.jsonl'], deterministic=False),
              'controller_note': 'Controller performs no model operations; each child applies strict deterministic settings.'}
    write_json(args.output / 'plan.json', plan)
    write_json(args.output / 'report.json', report)
    started = time.monotonic()
    for name in ('recurrent', 'transformer'):
        write_json(args.output / 'status.json', {'stage': 'evaluating', 'model': name})
        folder = args.output / name
        command = [sys.executable, str(args.root / 'evaluate_tool_policy.py'), '--model', name,
                   '--checkpoint', str(getattr(args, name + '_checkpoint')), '--corpus', str(args.corpus),
                   '--tokenizer', str(args.tokenizer), '--output', str(folder)]
        model_started = time.monotonic()
        timed_out = False
        with (args.output / (name + '.log')).open('x') as log:
            try:
                process = subprocess.run(command, cwd=args.root, env=os.environ.copy(), stdout=log,
                                         stderr=subprocess.STDOUT, timeout=1800)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                # subprocess.run kills and waits for the owned child before returning.
                timed_out, returncode = True, None
        cases = []
        for index, task in enumerate(plan):
            path = folder / f'case-{index:02d}.json'
            if path.exists():
                row = json.loads(path.read_text())
                assert all(row[key] == value for key, value in task.items())
                cases.append(row)
        child = json.loads((folder / 'report.json').read_text()) if (folder / 'report.json').exists() else None
        complete = returncode == 0 and not timed_out and child is not None and child['status'] == 'complete' and len(cases) == 48
        entry = {'status': 'complete' if complete else 'incomplete', 'returncode': returncode,
                 'timed_out': timed_out, 'elapsed_seconds': time.monotonic()-model_started,
                 'scores': aggregate(plan, cases), 'child_report': child,
                 'case_hashes': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in folder.glob('case-*.json')}}
        report['models'][name] = entry
        write_json(args.output / 'report.json', report)
        print('TOOL_POLICY_MODEL_DONE', name, entry['status'], entry['scores']['completed_tasks'], flush=True)
    report['status'] = 'complete' if all(row['status'] == 'complete' for row in report['models'].values()) else 'incomplete'
    report['elapsed_seconds'] = time.monotonic()-started
    write_json(args.output / 'report.json', report)
    write_json(args.output / 'status.json', {'stage': report['status']})
    print('TOOL_POLICY_PAIR_DONE', report['status'], flush=True)


if __name__ == '__main__':
    main()
