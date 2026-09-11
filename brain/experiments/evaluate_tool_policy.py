"""Frozen V4 tool-policy baseline; all code execution uses the isolated backend."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from run_provenance import apply_deterministic_mode, provenance
from irene_brain.agent.parallel_depth_episode import ParallelDepthSoftwareAgent, TransformerSoftwareAgent
from irene_brain.evaluation.tool_policy import run_tool_policy_case
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.unified.parallel_depth_model import ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer


CHECKPOINTS = {
    'recurrent': '7feee0e9688dd4fb9014abedf677d09e071986c57ea4acdd8b8a97b61a2c28bf',
    'transformer': 'ca080bd8c6562656073dc90b64341cd624c931c1972c07c19ea9a2b4a59d1160',
}
BANK = {
    'requests.jsonl': 'c306d8e0399531c1157fed1abb09314013f577cf4be0ca564aaeebaa05f561a1',
    'graders.jsonl': '28d7ed1677ace53899058c56047253ee109523594e0e1c42328979fc480c468d',
}
TOKENIZER = '760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
MODES = ('from_scratch', 'repair')
VERBS = {'WRITE_FILE', 'READ_FILE', 'EDIT_FILE', 'RUN_TESTS', 'RETRIEVE_MEMORY', 'FINISH'}


class EvaluationDeadline(Exception):
    pass


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def aggregate(plan, results):
    """Missing tasks remain explicit failures in the predeclared denominator."""
    keys = [(row['source_id'], row['mode']) for row in plan]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate planned task')
    observed = {}
    for row in results:
        key = (row['source_id'], row['mode'])
        if key not in keys or key in observed:
            raise ValueError('Duplicate or unplanned result')
        if row['observed_repair'] and not row['success']:
            raise ValueError('Repair cannot be credited without completion')
        observed[key] = row
    stop_reasons = Counter(row['episode']['stop_reason'] for row in results)
    stop_reasons['missing'] = len(keys) - len(results)
    events = [event for row in results for event in row['events']]
    return {
        'planned_tasks': len(keys), 'recorded_tasks': len(results),
        'missing_tasks': [row for row in plan if (row['source_id'], row['mode']) not in observed],
        'completed_tasks': sum(row['success'] for row in results),
        'observed_repairs': sum(row['observed_repair'] for row in results),
        'completion_rate': sum(row['success'] for row in results) / len(keys),
        'repair_rate': sum(row['observed_repair'] for row in results) / len(keys),
        'stop_reasons': dict(stop_reasons),
        'executed_actions': len(events),
        'recognized_action_verbs': sum(event['action_type'] in VERBS for event in events),
        'successful_environment_actions': sum(event['success'] for event in events),
        'action_types': dict(Counter(event['action_type'] for event in events)),
        'conditions': {
            mode: {
                'planned_tasks': sum(row['mode'] == mode for row in plan),
                'recorded_tasks': sum(row['mode'] == mode for row in results),
                'completed_tasks': sum(row['success'] for row in results if row['mode'] == mode),
                'observed_repairs': sum(row['observed_repair'] for row in results if row['mode'] == mode),
            } for mode in MODES
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=tuple(CHECKPOINTS), required=True)
    for name in ('checkpoint', 'corpus', 'tokenizer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'status': 'incomplete', 'model': args.model, 'registered_seconds': 1800,
              'denominator': 48, 'trained_on_policy_bank': False,
              'limit': 'Procedural development only. Same-interpreter tests are not adversarially tamper-proof.',
              'memory': '4096-byte recurrent state' if args.model == 'recurrent' else
                        'Full prefix, 32768-token cap; no optimized generation-speed claim'}
    results, plan = [], []

    def deadline(*unused):
        raise EvaluationDeadline('Registered1800-second model evaluation limit')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(1800)
    try:
        assert torch.cuda.is_available()
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        torch.set_num_threads(1)
        random.seed(198)
        np.random.seed(198)
        torch.manual_seed(198)
        torch.cuda.manual_seed_all(198)
        apply_deterministic_mode()
        for name, expected in BANK.items():
            assert digest(args.corpus / name) == expected, name
        assert digest(args.tokenizer) == TOKENIZER
        assert digest(args.checkpoint) == CHECKPOINTS[args.model]
        requests = [json.loads(line) for line in (args.corpus / 'requests.jsonl').read_text().splitlines()]
        private = [json.loads(line) for line in (args.corpus / 'graders.jsonl').read_text().splitlines()]
        graders = {row['source_id']: row for row in private}
        assert len(requests) == len(private) == len(graders) == 24
        assert len({row['source_id'] for row in requests}) == 24
        assert set(graders) == {row['source_id'] for row in requests}
        assert len({row['family_sha256'] for row in requests}) == 6
        plan = [{'source_id': row['source_id'], 'mode': mode} for row in requests for mode in MODES]
        write_json(args.output / 'plan.json', plan)
        report['inputs'] = {**BANK, 'tokenizer': TOKENIZER, 'checkpoint': CHECKPOINTS[args.model]}
        tokenizer = BpeSemanticTokenizer(tokenizer_file=args.tokenizer, auto_build_if_missing=False)
        cls, agent_cls = (ParallelDepthModel, ParallelDepthSoftwareAgent) if args.model == 'recurrent' else (ComparisonTransformer, TransformerSoftwareAgent)
        model = cls.from_payload(torch.load(args.checkpoint, map_location='cpu', weights_only=True)).cuda().eval()
        report['provenance'] = provenance(model=model, train_seed=198, eval_seed=198,
            bank_digest=BANK['requests.jsonl'], deterministic=True)
        report['parameter_count'] = sum(parameter.numel() for parameter in model.parameters())
        assert report['parameter_count'] < 36000000
        agent = agent_cls(model, tokenizer)
        with torch.inference_mode():
            for request in requests:
                for mode in MODES:
                    index = len(results)
                    write_json(args.output / 'status.json', {'stage': 'evaluating', 'index': index,
                        'source_id': request['source_id'], 'mode': mode, 'elapsed_seconds': time.monotonic()-started})
                    row = run_tool_policy_case(agent, request, graders[request['source_id']], mode=mode)
                    if args.model == 'recurrent':
                        assert row['episode']['state_bytes'] == 4096
                        assert row['episode']['prefix_tokens'] is None
                    else:
                        assert row['episode']['state_bytes'] is None
                        assert 0 < row['episode']['prefix_tokens'] <= 32768
                    case_path = args.output / f'case-{index:02d}.json'
                    assert not case_path.exists()
                    write_json(case_path, row)
                    results.append(row)
                    report['scores'] = aggregate(plan, results)
                    write_json(args.output / 'report.json', report)
                    print('TOOL_POLICY_CASE', index, row['source_id'], mode, row['success'],
                          row['observed_repair'], row['episode']['stop_reason'], flush=True)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        if plan:
            report['scores'] = aggregate(plan, results)
        report['elapsed_seconds'] = time.monotonic() - started
        report['case_hashes'] = {path.name: digest(path) for path in args.output.glob('case-*.json')}
        write_json(args.output / 'report.json', report)
        print('TOOL_POLICY_RESULT', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
