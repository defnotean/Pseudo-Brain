"""Post-training tool evaluation using the unchanged registered policy scorer."""
import argparse
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
from run_provenance import apply_deterministic_mode, provenance
from evaluate_tool_policy import BANK, TOKENIZER, MODES, aggregate, digest, write_json
from irene_brain.agent.parallel_depth_episode import ParallelDepthSoftwareAgent, TransformerSoftwareAgent
from irene_brain.evaluation.tool_policy import run_tool_policy_case
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from irene_brain.unified.parallel_depth_model import ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer


SELECTION = '5933901753e5416e75ae2fac765e3d21746de29d3e7437415a0fd12420ca98cd'
TRAINING_SOURCE = '3be24612bfb31fac5376bb5c33ca33b5514906a5d0e1186659cea82616183005'


def resolve_completed_checkpoint(training_run, report_sha256, model):
    """Bind evaluation to the completed registered pair, never a selected partial."""
    path = training_run / 'report.json'
    if digest(path) != report_sha256:
        raise ValueError('Training report hash mismatch')
    pair = json.loads(path.read_text())
    if pair['status'] != 'complete' or pair['native_preflight'] or pair['source_manifest_sha256'] != TRAINING_SOURCE:
        raise ValueError('Expected the completed registered policy training pair')
    if set(pair['models']) != {'recurrent', 'transformer'}:
        raise ValueError('Training pair is incomplete')
    for name, entry in pair['models'].items():
        child = entry['child_report']
        if entry['status'] != 'complete' or child['status'] != 'complete' or child['completed_updates'] != 3520:
            raise ValueError('Missing or partial training result')
        if child['selection_sha256'] != SELECTION or child['native_preflight']:
            raise ValueError('Wrong training selection or diagnostic weights')
        if name == 'recurrent' and len(child.get('parity', [])) != 3:
            raise ValueError('Required recurrent parity evidence is missing')
    child = pair['models'][model]['child_report']
    receipt = child['checkpoint']
    filename = receipt['checkpoint']
    if filename != 'checkpoint-3520-trained.pt' or receipt['completed_updates'] != 3520:
        raise ValueError('Only the registered final training boundary is eligible')
    checkpoint = training_run / model / filename
    if digest(checkpoint) != receipt['sha256']:
        raise ValueError('Trained checkpoint hash mismatch')
    return checkpoint, receipt['sha256'], child


class EvaluationDeadline(Exception):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('recurrent', 'transformer'), required=True)
    for name in ('training-run', 'corpus', 'tokenizer', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--training-report-sha256', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'status': 'incomplete', 'model': args.model, 'denominator': 48,
              'trained_on_policy_bank': True, 'training_report_sha256': args.training_report_sha256,
              'registered_seconds': 1800,
              'limit': 'Procedural development only; same-interpreter tests are not adversarially tamper-proof.',
              'memory': '4096-byte recurrent state' if args.model == 'recurrent' else
                        'Full prefix, 32768-token cap; no optimized generation-speed claim'}
    plan, results = [], []

    def deadline(*unused):
        raise EvaluationDeadline('Registered1800-second model evaluation limit')

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(1800)
    try:
        assert torch.cuda.is_available() and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
        torch.set_num_threads(1)
        random.seed(198)
        np.random.seed(198)
        torch.manual_seed(198)
        torch.cuda.manual_seed_all(198)
        apply_deterministic_mode()
        for filename, expected in BANK.items():
            assert digest(args.corpus / filename) == expected
        assert digest(args.tokenizer) == TOKENIZER
        requests = [json.loads(line) for line in (args.corpus / 'requests.jsonl').read_text().splitlines()]
        private = [json.loads(line) for line in (args.corpus / 'graders.jsonl').read_text().splitlines()]
        graders = {row['source_id']: row for row in private}
        assert len(requests) == len(private) == len(graders) == 24
        assert len({row['source_id'] for row in requests}) == 24
        assert set(graders) == {row['source_id'] for row in requests}
        assert len({row['family_sha256'] for row in requests}) == 6
        plan = [{'source_id': row['source_id'], 'mode': mode} for row in requests for mode in MODES]
        write_json(args.output / 'plan.json', plan)
        checkpoint, checksum, trained = resolve_completed_checkpoint(args.training_run, args.training_report_sha256, args.model)
        report['inputs'] = {**BANK, 'tokenizer': TOKENIZER, 'checkpoint': checksum}
        report['training_selection_sha256'] = SELECTION
        report['training_completed_updates'] = trained['completed_updates']
        tokenizer = BpeSemanticTokenizer(tokenizer_file=args.tokenizer, auto_build_if_missing=False)
        cls, agent_cls = (ParallelDepthModel, ParallelDepthSoftwareAgent) if args.model == 'recurrent' else (ComparisonTransformer, TransformerSoftwareAgent)
        model = cls.from_payload(torch.load(checkpoint, map_location='cpu', weights_only=True)).cuda().eval()
        report['provenance'] = provenance(model=model, train_seed=1986, eval_seed=198,
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
                        assert row['episode']['state_bytes'] == 4096 and row['episode']['prefix_tokens'] is None
                    else:
                        assert row['episode']['state_bytes'] is None and 0 < row['episode']['prefix_tokens'] <= 32768
                    case_path = args.output / f'case-{index:02d}.json'
                    assert not case_path.exists()
                    write_json(case_path, row)
                    results.append(row)
                    report['scores'] = aggregate(plan, results)
                    write_json(args.output / 'report.json', report)
                    print('TRAINED_TOOL_POLICY_CASE', index, row['source_id'], mode, row['success'],
                          row['observed_repair'], row['episode']['stop_reason'], flush=True)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        if plan:
            report['scores'] = aggregate(plan, results)
        report['elapsed_seconds'] = time.monotonic()-started
        report['case_hashes'] = {path.name: digest(path) for path in args.output.glob('case-*.json')}
        write_json(args.output / 'report.json', report)
        print('TRAINED_TOOL_POLICY_RESULT', json.dumps(report), flush=True)


if __name__ == '__main__': main()
