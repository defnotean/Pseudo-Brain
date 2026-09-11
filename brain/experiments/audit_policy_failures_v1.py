"""Audit completed behavior traces and training feedback; never execute model code."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(episodes):
    counts, errors, syntax_errors = Counter(), Counter(), Counter()
    for prompt, steps in episodes:
        targets = re.findall(r'^Target: (.+)$', prompt, re.MULTILINE)
        assert len(targets) == 1
        counts['episodes'] += 1
        for step in steps:
            action, feedback = step['action'], step['feedback']
            counts['actions'] += 1
            counts['recognized_actions'] += feedback['action_type'] != 'UNKNOWN'
            counts['successful_environment_actions'] += bool(feedback['success'])
            if feedback['action_type'] == 'RUN_TESTS' and not feedback['success']:
                counts['failed_run_tests'] += 1
                names = re.findall(r'(?m)^\s*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(?::(?:\s|$)|\s*$)', feedback['observation_text'])
                classes = {name for name in names if name.endswith(('Error', 'Exception'))}
                errors.update(classes or {'no_exception_class_in_feedback'})
            header, separator, body = action.partition('\n')
            match = re.fullmatch(r'ACTION: WRITE_FILE (.+)', header)
            if match:
                counts['write_actions'] += 1
                counts['writes_to_other_paths'] += match.group(1) != targets[0]
                counts['writes_with_edit_markers'] += '<<<TARGET\n' in body
                counts['writes_without_body'] += not separator or not body.strip()
                try:
                    parsed = ast.parse(body)
                except SyntaxError as exc:
                    counts['write_bodies_invalid_python'] += 1
                    syntax_errors[exc.msg] += 1
                else:
                    counts['write_bodies_valid_python'] += 1
                    counts['write_bodies_without_function'] += not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in parsed.body)
    assert counts['write_actions'] == counts['write_bodies_invalid_python'] + counts['write_bodies_valid_python']
    return {'counts': dict(counts), 'failed_test_exception_mentions': dict(errors),
            'write_syntax_error_messages': dict(syntax_errors)}


def main():
    parser = argparse.ArgumentParser()
    for name in ('corpus', 'evaluation', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    train_path = args.corpus / 'train.jsonl.gz'
    assert checksum(train_path) == '896a39d478aaa674903c27eb9c7af0d96017bc721843c860a3d56daa056d73a0'
    pair_path = args.evaluation / 'report.json'
    assert checksum(pair_path) == 'd2cc4c44abad7c736093864f375088e2fbccad1eac35c26ee7e055db45cd9ecd'
    pair = json.loads(pair_path.read_text())
    assert pair['status'] == 'complete'
    train = [json.loads(line) for line in gzip.decompress(train_path.read_bytes()).splitlines()]
    assert len(train) == 352
    requests = [json.loads(line) for line in (args.corpus / 'requests.jsonl').read_text().splitlines()]
    assert checksum(args.corpus / 'requests.jsonl') == 'c306d8e0399531c1157fed1abb09314013f577cf4be0ca564aaeebaa05f561a1'
    prompts = {row['source_id']: row['initial_prompt'] for row in requests}
    report = {'status': 'complete', 'source_sha256': checksum(Path(__file__)),
              'training_bank_sha256': checksum(train_path), 'evaluation_report_sha256': checksum(pair_path),
              'training': summarize((row['initial_prompt'], row['steps']) for row in train), 'models': {},
              'scope': 'Descriptive counts of observed feedback and AST syntax; no model/code execution or causal attribution. Exception categories may overlap within feedback.'}
    for name, entry in pair['models'].items():
        episodes = []
        for filename, expected in sorted(entry['case_hashes'].items()):
            path = args.evaluation / name / filename
            assert checksum(path) == expected
            row = json.loads(path.read_text())
            steps = [{'action': step['raw_action'], 'feedback': step['feedback']}
                     for step in row['episode']['trace'] if step['executed']]
            episodes.append((prompts[row['source_id']], steps))
        assert len(episodes) == 48
        result = summarize(episodes)
        for field, score in (('actions', 'executed_actions'), ('recognized_actions', 'recognized_action_verbs'),
                             ('successful_environment_actions', 'successful_environment_actions')):
            assert result['counts'][field] == entry['scores'][score]
        report['models'][name] = result
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print('POLICY_FAILURE_DISTRIBUTION', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
