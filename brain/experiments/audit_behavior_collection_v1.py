"""Verify completed real-prefix records, source identities and admission counts."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

from collect_behavior_repair_v1 import CHECKPOINTS,select_cases
from evaluate_tool_policy import TOKENIZER,digest
from irene_brain.data.executed_trajectory import digest_record
from irene_brain.data.behavior_repair_trajectory import encode_behavior_repair
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--bank',type=Path,required=True)
    parser.add_argument('--tokenizer',type=Path,required=True)
    args=parser.parse_args()
    pair=json.loads((args.root/'collection/report.json').read_text())
    assert pair['status']=='complete' and set(pair['models'])==set(CHECKPOINTS)
    assert digest(args.tokenizer)==TOKENIZER
    tokenizer=BpeSemanticTokenizer(tokenizer_file=args.tokenizer,auto_build_if_missing=False)
    cases=select_cases(args.bank)
    source_sha=digest(args.root/'source-manifest.json')
    summary={'status':'complete','model_updates':0,'planned_actor_attempts':64,'recorded_actor_attempts':0,
             'accepted_trajectories':0,'models':{},'source_manifest_sha256':source_sha}
    for model,expected_checkpoint in CHECKPOINTS.items():
        folder=args.root/'collection'/model
        child=json.loads((folder/'report.json').read_text())
        assert child==pair['models'][model]['child_report'] and child['status']=='complete'
        assert child['checkpoint_sha256']==expected_checkpoint and child['recorded_attempts']==32
        compressed=(folder/'training-trajectories.jsonl.gz').read_bytes()
        assert hashlib.sha256(compressed).hexdigest()==child['trajectory_gzip_sha256']
        records=[json.loads(line) for line in gzip.decompress(compressed).splitlines()]
        assert len(records)==child['accepted']
        by_hash={r['record_sha256']:r for r in records}
        assert len(by_hash)==len(records)
        counts=Counter()
        noncanonical=0
        exceptions=Counter()
        input_tokens=supervised_tokens=repaired=0
        for index,(case,entry) in enumerate(zip(cases,child['records'])):
            assert entry['source_id']==case['source_id'] and entry['case_sha256']==case['case_sha256']
            path=folder/f'actor-{index:02d}.json'
            assert digest(path)==entry['raw_attempt_sha256']
            raw=json.loads(path.read_text())
            assert raw['case_sha256']==case['case_sha256'] and raw['checkpoint_sha256']==expected_checkpoint
            assert raw['episode']['initial_prompt_sha256']==hashlib.sha256(case['initial_prompt'].encode()).hexdigest()
            for action in raw['episode']['trace']:
                assert tokenizer.decode(action['token_ids'],skip_special=False)==action['raw_action']
                noncanonical+=tokenizer.encode(action['raw_action'])!=action['token_ids']
            counts[entry['disposition']]+=1
            if entry['disposition']=='accepted':
                record=by_hash[entry['record_sha256']]
                assert record==json.loads((folder/f'teacher-{index:02d}.json').read_text())
                assert record['actor_trace']==raw['episode']['trace']
                assert record['initial_prompt']==case['initial_prompt'] and record['initial_files']==case['initial_files']
                assert record['raw_attempt_sha256']==entry['raw_attempt_sha256']
                assert record['behavior_final_files_sha256']==digest_record(raw['actor_final_files'])
                actor=record['actor']
                assert actor['provenance_kind']=='model_checkpoint' and actor['checkpoint_sha256']==expected_checkpoint
                assert actor['source_manifest_sha256']==source_sha and actor['initial_case_sha256']==case['case_sha256'] and actor['tokenizer_sha256']==TOKENIZER
                encoded=encode_behavior_repair(record,tokenizer)
                assert encoded['input_ids'].numel()==entry['input_tokens']
                assert int((encoded['labels']!=-100).sum())==entry['supervised_tokens']
                input_tokens+=entry['input_tokens'];supervised_tokens+=entry['supervised_tokens']
                teacher_tests=[s['feedback'] for s in record['steps'][record['behavior_steps']:] if s['feedback']['action_type']=='RUN_TESTS']
                repaired+=entry['teacher_repaired_observed_failure']
                assert entry['teacher_repaired_observed_failure']==(not teacher_tests[0]['success'] and teacher_tests[1]['success'])
                if not teacher_tests[0]['success']:
                    text=teacher_tests[0]['observation_text']
                    for name in ('SyntaxError','ImportError','ModuleNotFoundError','NameError','AssertionError','TypeError','ValueError','RuntimeError'):
                        if name in text:exceptions[name]+=1
        assert len(child['records'])==32 and dict(counts)==child['dispositions'] and sum(counts.values())==32
        assert input_tokens==child['input_tokens'] and supervised_tokens==child['supervised_tokens'] and repaired==child['teacher_repaired_observed_failures']
        summary['models'][model]={key:child[key] for key in ('recorded_attempts','accepted','dispositions','actor_stops','actor_action_types','parameter_count',
            'teacher_repaired_observed_failures','input_tokens','supervised_tokens','elapsed_seconds','trajectory_gzip_sha256')}
        summary['models'][model].update(noncanonical_actor_token_sequences=noncanonical,first_teacher_test_exception_mentions=dict(exceptions),
                                       actor_verified_completions=sum(e['actor_verified_completion'] for e in child['records']))
        summary['recorded_actor_attempts']+=32
        summary['accepted_trajectories']+=len(records)
    summary['pair_elapsed_seconds']=pair['elapsed_seconds']
    summary['evidence_limit']='Actual training-task prefixes plus teacher repairs; teacher completions are not autonomous model success.'
    with (args.root/'collection-audit.json').open('x') as stream:json.dump(summary,stream,indent=2)
    print('BEHAVIOR_AUDIT',json.dumps(summary),flush=True)


if __name__=='__main__':main()
