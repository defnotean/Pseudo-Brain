"""Separate answer-bearing process: score frozen model output, never generate."""
import argparse
import hashlib
import json
from pathlib import Path
from irene_brain.evaluation.independent_pilot import load_requests, score_math_bank


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--include-code', action='store_true', help='Run HumanEval in the verified remote sandbox after grader controls')
    args = parser.parse_args()
    manifest = json.loads((args.bank/'manifest.json').read_text())
    raw_requests = (args.bank/'requests.jsonl').read_bytes()
    raw_graders = (args.bank/'graders.jsonl').read_bytes()
    for name, raw in (('requests', raw_requests), ('graders', raw_graders)):
        if hashlib.sha256(raw).hexdigest() != manifest[name + '_sha256']:
            raise ValueError(name + ' hash mismatch')
    requests = load_requests(raw_requests)
    if [[row['benchmark'], row['task_id']] for row in requests] != manifest['selected']:
        raise ValueError('Registered task selection mismatch')
    generation = json.loads((args.generation/'generation.json').read_text())
    if generation['requests_sha256'] != manifest['requests_sha256']:
        raise ValueError('Generation used different requests')
    raw_responses = (args.generation/'responses.jsonl').read_bytes()
    responses = [json.loads(line) for line in raw_responses.decode().splitlines()]
    if len(responses) != generation['completed_tasks']:
        raise ValueError('Generation receipt count mismatch')
    graders = [json.loads(line) for line in raw_graders.decode().splitlines()]
    report = score_math_bank(requests, graders, responses)
    if args.include_code:
        from irene_brain.evaluation.independent_code import score_code_bank
        report['HumanEval'] = score_code_bank(requests, graders, responses)
    if generation['status'] != 'complete':
        report['status'] = 'incomplete'
    report.update(checkpoint_sha256=generation['checkpoint_sha256'],
                  checkpoint_training_metadata=generation['checkpoint_training_metadata'],
                  requests_sha256=manifest['requests_sha256'],
                  graders_sha256=manifest['graders_sha256'],
                  responses_sha256=hashlib.sha256(raw_responses).hexdigest(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print('MATH_SCORED', report['status'], report['GSM8K']['correct'], '/', report['GSM8K']['total'])
    if args.include_code:
        print('CODE_SCORED', report['HumanEval']['status'], report['HumanEval']['correct'], '/', report['HumanEval']['total'])


if __name__ == '__main__':
    main()
