"""Recover only the frozen v1 training cases, checking every recorded identity."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from irene_brain.agent.procedural_training_generator import ProceduralTrainingGenerator
from irene_brain.data.procedural_repair_bank import partition_tasks
from irene_brain.data.executed_trajectory import digest_record

TRAIN_SHA256 = '896a39d478aaa674903c27eb9c7af0d96017bc721843c860a3d56daa056d73a0'


def reconstruct(corpus):
    compressed = (Path(corpus) / 'train.jsonl.gz').read_bytes()
    if hashlib.sha256(compressed).hexdigest() != TRAIN_SHA256:
        raise ValueError('Frozen training corpus identity changed')
    records = [json.loads(line) for line in gzip.decompress(compressed).splitlines()]
    tasks = ProceduralTrainingGenerator(seed=1985).generate_training_tasks(count=2048, diversify_indices=False)
    selected, mapping = partition_tasks(tasks)
    cases = selected['train']
    if len(cases) != 352 or len(records) != 352:
        raise ValueError('Frozen training selection count changed')
    if mapping != json.loads((Path(corpus) / 'family-map.json').read_text()):
        raise ValueError('Frozen family assignment changed')
    identities = []
    for case, record in zip(cases, records):
        if digest_record({k: v for k, v in record.items() if k != 'record_sha256'}) != record['record_sha256']:
            raise ValueError('Original executed record is corrupt')
        for key in ('source_id', 'family_sha256', 'initial_prompt', 'initial_files', 'repair'):
            if case[key] != record[key]:
                raise ValueError('Reconstructed case mismatch: ' + key)
        if record['partition'] != 'train' or mapping[case['family_sha256']]['partition'] != 'train':
            raise ValueError('Nontraining case selected')
        validator_id = hashlib.sha256(case['private_check'].encode()).hexdigest()
        final_hash = digest_record({**case['initial_files'], case['target_module']: case['reference_solution']})
        if validator_id != record['validator_id'] or final_hash != record['final_files_sha256']:
            raise ValueError('Private validator or canonical final workspace changed')
        identities.append({'source_id': case['source_id'], 'record_sha256': record['record_sha256'],
                           'validator_id': validator_id, 'final_files_sha256': final_hash})
    return cases, {'status': 'complete', 'records': len(cases), 'training_families': len({c['family_sha256'] for c in cases}),
                   'train_gzip_sha256': TRAIN_SHA256, 'identities_sha256': digest_record(identities), 'identities': identities}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases, report = reconstruct(args.corpus)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != 'identities'}), flush=True)


if __name__ == '__main__':
    main()
