"""Audit every official MBPP training candidate before making a case bank."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import signal
import time

from irene_brain.data.broad_corpus import assign_components
from irene_brain.data.foundation_corpus import registered_benchmark_keys
from irene_brain.data.mbpp_training import adapt_case, row_keys, structural_code_key
from irene_brain.data.executed_trajectory import digest_record
from irene_brain.agent.isolated_software_environment import ToolFeedback
from irene_brain.agent.parallel_depth_episode import observation_frame
from irene_brain.evaluation.workspace_sandbox import run_workspace_tests
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer

INPUTS = {
    'mbpp': ('/content/pb-mbpp-source-v1/mbpp/mbpp.jsonl', 'ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f'),
    'mbpp_train': ('/content/pb-mbpp-source-v1/official-train.jsonl', '97f660b820f3d75a99a424ec249b9bce04e33517789584c05d6e2a0e84995d97'),
    'development_192': ('/content/pb-broad-pilot/corpus/development.jsonl.gz', '7c3cdcde6475c900640ea89494018a5c7eff4792be38e17d27748930dbbc010f'),
    'development_384': ('/content/pb-foundation-v4/corpus/development.jsonl.gz', 'f95f6a2f491db9df37f8682283a49605af68da62e5d0274e5d33a7e7815388b5'),
    'development_1536': ('/content/pb-foundation-v5/corpus/development.jsonl.gz', 'da54b4cb7bcaabaea216e1db7b4f7965e96393d99dd1d20207f7096f245ad90d'),
    'benchmark_requests': ('/content/pb-broad-pilot/independent_eval/requests.jsonl', '1afbb0d6b1f8c3ba4f78a4b50d1040c67ee0701d97660ef96b267728f913642a'),
    'procedural_development': ('/content/pb-procedural-repair-bank-v1/corpus/graders.jsonl', '28d7ed1677ace53899058c56047253ee109523594e0e1c42328979fc480c468d'),
}


def load_inputs():
    loaded = {}
    for name, (path, expected) in INPUTS.items():
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('Frozen input hash changed: '+name)
        if path.endswith('.gz'):
            raw = gzip.decompress(raw)
        loaded[name] = [json.loads(line) for line in raw.splitlines()]
    for name, count in (('mbpp',974), ('mbpp_train',374), ('development_192',192), ('development_384',384),
                        ('development_1536',1536), ('benchmark_requests',64), ('procedural_development',24)):
        if len(loaded[name]) != count:
            raise ValueError('Frozen input count changed: '+name)
    return loaded


def isolated_candidates(inputs):
    all_rows = inputs['mbpp']
    training = sorted((row for row in all_rows if 601 <= row['task_id'] <= 974), key=lambda row:row['task_id'])
    if training != inputs['mbpp_train'] or [r['task_id'] for r in training] != list(range(601,975)):
        raise ValueError('Official training extraction changed')
    candidates = [{'row':row, 'isolation_keys':row_keys(row)} for row in training]
    seeds = [{'isolation_keys':row_keys(row), 'quarantine_seed':True} for row in all_rows if not 601 <= row['task_id'] <= 974]
    for name in ('development_192', 'development_384', 'development_1536'):
        for row in inputs[name]:
            keys = set(row['isolation_keys'])
            if row.get('task') == 'code' and '[RESP]' in row.get('text',''):
                answer = row['text'].split('[RESP]',1)[1].removesuffix('[EOS]').strip()
                try:
                    keys.add(structural_code_key(answer))
                except (SyntaxError, ValueError):
                    pass
            seeds.append({'isolation_keys':sorted(keys), 'quarantine_seed':True})
    for row in inputs['procedural_development']:
        seeds.append({'isolation_keys':[structural_code_key(row['reference_solution'])], 'quarantine_seed':True})
    for key in registered_benchmark_keys(inputs['benchmark_requests']):
        seeds.append({'isolation_keys':[key], 'quarantine_seed':True})
    joined = assign_components(candidates+seeds)
    blocked = {row['component_id'] for row in joined if row.get('quarantine_seed')}
    return candidates, blocked, len(seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    start = time.monotonic()
    report = {'status':'running', 'protocol':'mbpp_training_bank_v1', 'model_updates':0,
              'candidates_processed':0, 'eligible':0, 'dispositions':{}, 'inputs':INPUTS, 'records':[]}
    def deadline(*unused):
        raise TimeoutError('Registered 900-second MBPP preparation limit')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(900)
    try:
        inputs = load_inputs()
        candidates, blocked, seed_count = isolated_candidates(inputs)
        report['quarantine_seed_records'] = seed_count
        token_raw = args.tokenizer.read_bytes()
        if hashlib.sha256(token_raw).hexdigest() != '760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e':
            raise ValueError('Frozen tokenizer changed')
        tokenizer = BpeSemanticTokenizer(tokenizer_file=args.tokenizer, auto_build_if_missing=False)
        records, seen, dispositions = [], set(), Counter()
        for candidate in candidates:
            row, component = candidate['row'], candidate['component_id']
            result = {'task_id':row['task_id'], 'component_id':component, 'isolation_keys':candidate['isolation_keys']}
            case = None
            if component in blocked:
                result['disposition'] = 'quarantined_component'
            elif component in seen:
                result['disposition'] = 'duplicate_training_component'
            else:
                seen.add(component)
                try:
                    case = adapt_case(row)
                except (SyntaxError, ValueError) as exc:
                    result.update(disposition='format_rejection', error_type=type(exc).__name__, error=str(exc))
                if case is not None:
                    good = {**case['initial_files'], case['target_module']:case['reference_solution']}
                    bad = {**case['initial_files'], case['target_module']:'raise RuntimeError("Injected training fault")'}
                    controls = {'public_good':run_workspace_tests(good, 'test_public.py'),
                                'private_good':run_workspace_tests({**good,'__private_check.py':case['private_check']}, '__private_check.py'),
                                'public_bad':run_workspace_tests(bad, 'test_public.py'),
                                'private_bad':run_workspace_tests({**bad,'__private_check.py':case['private_check']}, '__private_check.py')}
                    result['controls'] = controls
                    if not all(controls[key]['passed'] for key in ('public_good','private_good')):
                        result['disposition'] = 'canonical_control_failure'
                    elif not all(not controls[key]['passed'] and 'Injected training fault' in controls[key]['output'] for key in ('public_bad','private_bad')):
                        result['disposition'] = 'negative_control_failure'
                    else:
                        sizes = {'prompt_tokens':len(tokenizer.encode(case['initial_prompt'])),
                                 'write_tokens_with_eos':len(tokenizer.encode('ACTION: WRITE_FILE '+case['target_module']+'\n'+case['reference_solution']))+1,
                                 'public_observation_tokens':len(tokenizer.encode(observation_frame(ToolFeedback('READ_FILE',True,case['initial_files']['test_public.py']))))}
                        result['sizes'] = sizes
                        if sizes['prompt_tokens'] > 4096 or sizes['write_tokens_with_eos'] > 1024 or sizes['public_observation_tokens'] > 4096:
                            result['disposition'] = 'token_limit_exclusion'
                        else:
                            case.update(component_id=component, dataset='google-research/mbpp', source_revision='08a8d6736475776f42ffac23b2c13111a28e5795',
                                        validator_sha256=hashlib.sha256(case['private_check'].encode()).hexdigest(), sizes=sizes)
                            case['case_sha256'] = digest_record(case)
                            result.update(disposition='eligible', case_sha256=case['case_sha256'])
                            records.append(case)
            dispositions[result['disposition']] += 1
            report['candidates_processed'] += 1
            report['eligible'] = len(records)
            report['dispositions'] = dict(dispositions)
            report['records'].append({key:value for key,value in result.items() if key!='controls'})
            with (args.output/'candidate-audit.jsonl').open('a') as stream:
                stream.write(json.dumps(result,ensure_ascii=True)+'\n')
            if report['candidates_processed'] % 32 == 0:
                (args.output/'progress.json').write_text(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
                print('MBPP_PROGRESS', report['candidates_processed'], 374, 'eligible',len(records), flush=True)
        if report['candidates_processed'] != 374 or sum(dispositions.values()) != 374:
            raise ValueError('Candidate accounting is incomplete')
        if len({record['component_id'] for record in records}) != len(records):
            raise ValueError('Duplicate eligible components')
        if any(record['component_id'] in blocked for record in records):
            raise ValueError('Quarantined component admitted')
        raw = ('\n'.join(json.dumps(record,sort_keys=True,ensure_ascii=True) for record in records)+'\n').encode()
        compressed = gzip.compress(raw,mtime=0)
        (args.output/'training-cases.jsonl.gz').write_bytes(compressed)
        report.update(status='complete', gzip_sha256=hashlib.sha256(compressed).hexdigest(), raw_sha256=hashlib.sha256(raw).hexdigest(),
                      gzip_bytes=len(compressed), unique_structural_families=len({r['family_sha256'] for r in records}))
    except Exception as exc:
        report.update(status='incomplete', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-start
        (args.output/'report.json').write_text(json.dumps(report,indent=2))
        print('MBPP_RESULT', json.dumps({k:v for k,v in report.items() if k!='records'}), flush=True)


if __name__ == '__main__':
    main()
