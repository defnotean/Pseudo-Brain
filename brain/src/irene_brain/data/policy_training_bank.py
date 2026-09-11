"""Fixed action-demonstration/replay schedule for the paired policy pilot."""
import gzip
import hashlib
import json
import random

import torch
from irene_brain.data.executed_trajectory import encode_executed_trajectory
from irene_brain.data.jsonl_records import load_jsonl_bytes
from irene_brain.data.streaming_loader import SequencePacker, RawDocument, TaskType


POLICY_HASHES = {
    'train.jsonl.gz': '896a39d478aaa674903c27eb9c7af0d96017bc721843c860a3d56daa056d73a0',
    'development.jsonl.gz': '0381456052591928c18debafd5e86bf6dc1c304c1793bec2740e0c30f7cccae8',
}
FOUNDATION_HASH = '66210c987c41ced5ff1f92d1365b1e27e5ad7f960d8b97263c2de89fb053ad48'


def load_rows(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Frozen bank hash mismatch: ' + path.name)
    return load_jsonl_bytes(gzip.decompress(raw))


def make_schedule():
    """Eight complete policy epochs; one distinct replay row every four actions."""
    replay = list(range(69632))
    random.Random(1986).shuffle(replay)
    schedule, replay_index = [], 0
    for epoch in range(8):
        order = list(range(352))
        random.Random(1986 + epoch).shuffle(order)
        for position, index in enumerate(order):
            schedule.append({'kind': 'policy', 'index': index, 'epoch': epoch})
            if (position + 1) % 4 == 0:
                schedule.append({'kind': 'foundation', 'index': replay[replay_index], 'epoch': epoch})
                replay_index += 1
    assert len(schedule) == 3520 and replay_index == 704
    return schedule


def encode_foundation_row(row, tokenizer):
    if not 0 < row['token_count'] <= 4096 or not 0 < row['supervised_tokens'] <= 1024:
        raise ValueError('Foundation row exceeds whole-document bounds')
    packer = SequencePacker(tokenizer, seq_len=row['token_count'], max_threads=1,
                            supervised_only_loss=True, document_policy='whole')
    packer.append_document(RawDocument(row['text'], TaskType(row['task'])))
    block = packer.emit_block()
    if packer._token_buffer or int((block.labels != -100).sum()) != row['supervised_tokens']:
        raise ValueError('Foundation token/label count mismatch')
    responses = (block.input_ids == tokenizer.resp_id).nonzero()
    if len(responses) != 1 or int(responses[0]) < 1:
        raise ValueError('Foundation replay must have one response boundary')
    ids, labels = block.input_ids[None], block.labels[None]
    resets = torch.zeros_like(ids)
    resets[0, 0] = 1
    return {'input_ids': ids, 'labels': labels, 'prompt_tokens': ids[:, :int(responses[0])].clone(),
            'reset_mask': resets, 'text_sha256': row['text_sha256'], 'task': row['task']}


def prepare_policy_training_bank(policy_corpus, foundation_corpus, tokenizer):
    train = load_rows(policy_corpus / 'train.jsonl.gz', POLICY_HASHES['train.jsonl.gz'])
    dev = load_rows(policy_corpus / 'development.jsonl.gz', POLICY_HASHES['development.jsonl.gz'])
    if len(train) != 352 or len(dev) != 24:
        raise ValueError('Policy partition size mismatch')
    for key, expected_train, expected_dev in (('source_id', 352, 24), ('family_sha256', 22, 6)):
        left, right = {row[key] for row in train}, {row[key] for row in dev}
        if len(left) != expected_train or len(right) != expected_dev or left & right:
            raise ValueError('Policy partition overlap or cardinality mismatch')
    foundation = load_rows(foundation_corpus / 'train.jsonl.gz', FOUNDATION_HASH)
    if len(foundation) != 69632:
        raise ValueError('Foundation training size mismatch')
    schedule = make_schedule()
    replay_indices = [row['index'] for row in schedule if row['kind'] == 'foundation']
    encoded_train = [encode_executed_trajectory(row, tokenizer, max_tokens=8192) for row in train]
    encoded_dev = [encode_executed_trajectory(row, tokenizer, max_tokens=8192) for row in dev]
    encoded_replay = {index: encode_foundation_row(foundation[index], tokenizer) for index in replay_indices}
    selection = []
    input_tokens, supervised_tokens = 0, 0
    for step in schedule:
        if step['kind'] == 'policy':
            encoded = encoded_train[step['index']]
            record_hash = train[step['index']]['record_sha256']
        else:
            encoded = encoded_replay[step['index']]
            record_hash = foundation[step['index']]['text_sha256']
        inputs = encoded['input_ids'].numel()
        supervised = int((encoded['labels'] != -100).sum())
        input_tokens += inputs
        supervised_tokens += supervised
        selection.append({**step, 'record_sha256': record_hash, 'input_tokens': inputs, 'supervised_tokens': supervised})
    serialized = json.dumps(selection, sort_keys=True, separators=(',', ':')).encode()
    manifest = {'version': 'policy_training_bank_v1', 'updates': len(selection), 'policy_updates': 2816,
                'replay_updates': 704, 'unique_replay_rows': len(set(replay_indices)),
                'policy_train_families': 22, 'policy_development_families': 6,
                'input_tokens': input_tokens, 'supervised_tokens': supervised_tokens,
                'policy_hashes': POLICY_HASHES, 'foundation_train_sha256': FOUNDATION_HASH,
                'selection_sha256': hashlib.sha256(serialized).hexdigest(),
                'limitation': 'Policy-family isolation is local to this procedural bank; no global semantic novelty claim.'}
    return {'train': encoded_train, 'development': encoded_dev, 'replay': encoded_replay,
            'schedule': selection, 'manifest': manifest}
