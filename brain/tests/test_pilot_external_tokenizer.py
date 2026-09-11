"""Tokenizer identity must be checked before interpreting model token indices."""
import hashlib
import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from generate_policy_retention import resolve_tokenizer


def fixture(tmp_path):
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '[RESP]': 1, '[EOS]': 2, 'hello': 3}, unk_token='[UNK]'))
    path = tmp_path / 'tokenizer.json'
    path.write_text(tokenizer.to_str())
    return tokenizer, path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_external_and_embedded_tokenizers_preserve_ids(tmp_path):
    tokenizer, path, digest = fixture(tmp_path)
    payload = {'tokenizer_json': json.dumps(json.loads(tokenizer.to_str()), indent=2)}
    for candidate in (resolve_tokenizer(payload), resolve_tokenizer(payload, path, digest),
                      resolve_tokenizer({}, path, digest)):
        assert candidate.encode('hello').ids == [3]
        assert candidate.token_to_id('[RESP]') == 1
        assert candidate.token_to_id('[EOS]') == 2


def test_missing_or_unverified_external_tokenizer_is_rejected(tmp_path):
    _, path, digest = fixture(tmp_path)
    for arguments in (({},), ({}, path), ({}, None, digest), ({}, path, '0' * 64)):
        with pytest.raises(ValueError):
            resolve_tokenizer(*arguments)


def test_verified_but_incompatible_tokenizer_is_rejected(tmp_path):
    tokenizer, path, digest = fixture(tmp_path)
    changed = json.loads(tokenizer.to_str())
    changed['model']['vocab']['hello'] = 4
    with pytest.raises(ValueError, match='differs from embedded'):
        resolve_tokenizer({'tokenizer_json': json.dumps(changed)}, path, digest)
