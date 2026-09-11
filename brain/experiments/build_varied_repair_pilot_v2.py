"""Execute the registered training-only fault-coverage pilot; never a model score."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import signal
import time

from reconstruct_policy_training_cases import reconstruct
from irene_brain.agent.isolated_software_environment import IsolatedSoftwareEnvironment, ToolFeedback
from irene_brain.agent.parallel_depth_episode import observation_frame
from irene_brain.data.executed_trajectory import TeacherAction, digest_record, record_executed_trajectory, encode_executed_trajectory
from irene_brain.evaluation.workspace_sandbox import run_workspace_tests
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer

SCENARIOS = ('clean', 'syntax', 'import_symbol', 'missing_module', 'write_edit_body',
             'missing_write_body', 'invalid_header', 'edit_mismatch', 'runtime', 'module_assertion')
EXPECTED = {'syntax': 'SyntaxError', 'import_symbol': 'ImportError', 'missing_module': 'ModuleNotFoundError',
            'write_edit_body': 'SyntaxError', 'missing_write_body': 'ModuleNotFoundError',
            'invalid_header': 'ModuleNotFoundError', 'edit_mismatch': 'ImportError',
            'runtime': 'RuntimeError', 'module_assertion': 'AssertionError'}
TOKENIZER_SHA256 = '760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'


def plan(case, scenario):
    target, reference = case['target_module'], case['reference_solution']
    if target in case['initial_files']:
        raise ValueError('Registered pilot requires initially absent target')
    imports = [node for node in ast.walk(ast.parse(case['initial_files']['test_public.py']))
               if isinstance(node, ast.ImportFrom) and node.module == target[:-3]]
    if len(imports) != 1 or len(imports[0].names) != 1:
        raise ValueError('Expected one public import identifying target symbol')
    symbol = imports[0].names[0].name
    actions = [TeacherAction('ACTION: READ_FILE test_public.py')]
    faulty = None
    if scenario == 'syntax':
        faulty = 'def broken(:\n    pass'
    elif scenario == 'import_symbol':
        faulty = reference + '\ndel ' + symbol
    elif scenario == 'write_edit_body':
        faulty = '<<<TARGET\npass\n===\n' + reference + '\n>>>'
    elif scenario == 'runtime':
        faulty = 'raise RuntimeError("Injected training fault")'
    elif scenario == 'module_assertion':
        faulty = 'raise AssertionError("Injected module-load assertion fault")'
    elif scenario == 'edit_mismatch':
        faulty = 'pass'
    if faulty is not None:
        actions.append(TeacherAction('ACTION: WRITE_FILE ' + target + '\n' + faulty, 'injected_fault'))
    if scenario == 'missing_module':
        wrong = 'misplaced_' + target
        if wrong in case['initial_files']:
            raise ValueError('Wrong-path control would overwrite existing file')
        actions.append(TeacherAction('ACTION: WRITE_FILE ' + wrong + '\n' + reference, 'injected_fault'))
    elif scenario == 'missing_write_body':
        actions.append(TeacherAction('ACTION: WRITE_FILE ' + target, 'injected_fault'))
    elif scenario == 'invalid_header':
        actions.append(TeacherAction('WRITE_FILE ' + target + '\n' + reference, 'injected_fault'))
    elif scenario == 'edit_mismatch':
        actions.append(TeacherAction('ACTION: EDIT_FILE ' + target + '\n<<<TARGET\nabsent exact target\n===\n' + reference + '\n>>>', 'injected_fault'))
    if scenario != 'clean':
        actions.extend((TeacherAction('ACTION: RUN_TESTS'), TeacherAction('ACTION: READ_FILE ' + target)))
    if scenario in ('syntax', 'import_symbol', 'runtime', 'module_assertion'):
        actions.append(TeacherAction('ACTION: EDIT_FILE ' + target + '\n<<<TARGET\n' + faulty + '\n===\n' + reference + '\n>>>'))
    else:
        actions.append(TeacherAction('ACTION: WRITE_FILE ' + target + '\n' + reference))
    actions.extend((TeacherAction('ACTION: RUN_TESTS'), TeacherAction('ACTION: FINISH Implementation verified')))
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    started = time.monotonic()
    report = {'status': 'running', 'protocol': 'varied_repair_v2_pilot', 'model_updates': 0,
              'generated_tokens': 0, 'attempted': 0, 'accepted': 0, 'scenarios': {}, 'records': []}
    def deadline(*unused):
        raise TimeoutError('Registered 900-second pilot deadline')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(900)
    try:
        cases, reconstruction = reconstruct(args.corpus)
        (args.output/'reconstruction.json').write_text(json.dumps(reconstruction, indent=2))
        first = {}
        for case in cases:
            first.setdefault(case['family_sha256'], case)
        chosen = list(first.values())
        if len(chosen) != 22:
            raise ValueError('Expected exactly 22 training families')
        report['selection_sha256'] = digest_record([{'source_id': c['source_id'], 'family_sha256': c['family_sha256']} for c in chosen])
        if hashlib.sha256(args.tokenizer.read_bytes()).hexdigest() != TOKENIZER_SHA256:
            raise ValueError('Tokenizer identity changed')
        tokenizer = BpeSemanticTokenizer(tokenizer_file=args.tokenizer, auto_build_if_missing=False)
        records = []
        counts = {scenario: Counter() for scenario in SCENARIOS}
        for case in chosen:
            def private(files):
                return run_workspace_tests({**files, '__private_check.py': case['private_check']}, '__private_check.py')
            good = {**case['initial_files'], case['target_module']: case['reference_solution']}
            bad = {**case['initial_files'], case['target_module']: 'raise RuntimeError("Injected training fault")'}
            controls = {'public_good': run_workspace_tests(good, 'test_public.py'), 'private_good': private(good),
                        'public_bad': run_workspace_tests(bad, 'test_public.py'), 'private_bad': private(bad)}
            with (args.output/'controls.jsonl').open('a') as stream:
                stream.write(json.dumps({'source_id': case['source_id'], 'controls': controls})+'\n')
            if not (all(controls[k]['passed'] for k in ('public_good', 'private_good')) and
                    all(not controls[k]['passed'] and 'Injected training fault' in controls[k]['output'] for k in ('public_bad', 'private_bad'))):
                raise ValueError('Original positive/negative control failed')
            def validator(files):
                result = private(files)
                return result['passed'], 'External checks passed' if result['passed'] else 'External checks failed'
            for scenario in SCENARIOS:
                report['current'] = {'source_id': case['source_id'], 'scenario': scenario}
                report['attempted'] += 1
                env = IsolatedSoftwareEnvironment(case['initial_files'], validator=validator)
                record = record_executed_trajectory(case['initial_prompt'], env, plan(case, scenario),
                    source_id=case['source_id']+':'+scenario, validator_id=hashlib.sha256(case['private_check'].encode()).hexdigest())
                del record['record_sha256']
                record.update(base_source_id=case['source_id'], partition='train', family_sha256=case['family_sha256'],
                              scenario=scenario, protocol='varied_repair_v2_pilot')
                record['record_sha256'] = digest_record(record)
                with (args.output/'attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(record, ensure_ascii=True)+'\n')
                tests = [step['feedback'] for step in record['steps'] if step['feedback']['action_type']=='RUN_TESTS']
                expected_outcomes = [True] if scenario=='clean' else [False, True]
                if [step['success'] for step in tests] != expected_outcomes:
                    raise ValueError('Unexpected actual public test outcomes')
                if scenario != 'clean' and EXPECTED[scenario] not in tests[0]['observation_text']:
                    raise ValueError('Injected fault did not produce registered exception')
                if scenario in ('missing_write_body', 'invalid_header', 'edit_mismatch'):
                    faults = [s for s in record['steps'] if s['kind']=='injected_fault']
                    if faults[-1]['feedback']['success']:
                        raise ValueError('Malformed or unmatched action was not rejected')
                missing_read = scenario in ('missing_module', 'missing_write_body', 'invalid_header')
                for step in record['steps']:
                    feedback = step['feedback']
                    if step['kind']=='teacher' and feedback['action_type'] in ('READ_FILE', 'WRITE_FILE', 'EDIT_FILE'):
                        expected_success = not (missing_read and step['action']=='ACTION: READ_FILE '+case['target_module'])
                        if feedback['success'] != expected_success:
                            raise ValueError('Unexpected teacher file operation outcome')
                encoded = encode_executed_trajectory(record, tokenizer, max_tokens=8192)
                if encoded['prompt_tokens'].shape[1] > 4096 or int(encoded['reset_mask'].sum()) != 1:
                    raise ValueError('Prompt or state reset contract failed')
                if env.files[case['target_module']] != case['reference_solution']:
                    raise ValueError('Final target differs from canonical reference')
                for step, span in zip(record['steps'], encoded['action_spans']):
                    if len(tokenizer.encode(step['action']))+1 > 1024 or len(tokenizer.encode(observation_frame(ToolFeedback(**step['feedback'])))) > 4096:
                        raise ValueError('Action or observation token limit exceeded')
                    if step['kind']=='injected_fault' and (encoded['labels'][0,span['start']:span['eos_position']+1] != -100).any():
                        raise ValueError('Injected action entered supervision')
                stats = {'source_id': record['source_id'], 'scenario': scenario, 'record_sha256': record['record_sha256'],
                         'input_tokens': encoded['input_ids'].numel(), 'supervised_tokens': int((encoded['labels']!=-100).sum()),
                         'failed_test_exception': EXPECTED.get(scenario), 'verified_completion': record['status']=='complete'}
                records.append(record)
                report['records'].append(stats)
                counts[scenario].update(records=1, input_tokens=stats['input_tokens'], supervised_tokens=stats['supervised_tokens'])
                report['accepted'] += 1
            report['scenarios'] = {key: dict(value) for key, value in counts.items()}
            (args.output/'progress.json').write_text(json.dumps({k: v for k, v in report.items() if k!='records'}, indent=2))
            print('VARIED_REPAIR_PROGRESS', report['accepted'], 220, flush=True)
        if len(records) != 220:
            raise ValueError('Registered attempted record count changed')
        compressed = gzip.compress(('\n'.join(json.dumps(r, sort_keys=True, ensure_ascii=True) for r in records)+'\n').encode(), mtime=0)
        (args.output/'train-pilot.jsonl.gz').write_bytes(compressed)
        report.update(status='complete', gzip_sha256=hashlib.sha256(compressed).hexdigest(), gzip_bytes=len(compressed))
    except Exception as exc:
        report.update(status='incomplete', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-started
        (args.output/'report.json').write_text(json.dumps(report, indent=2))
        print('VARIED_REPAIR_RESULT', json.dumps({k:v for k,v in report.items() if k!='records'}), flush=True)


if __name__ == '__main__':
    main()
