"""CPU-only tokenizer identity check against the two original V4 checkpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from generate_policy_retention import resolve_tokenizer


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    for name in ('tokenizer', 'recurrent-checkpoint', 'transformer-checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    torch.set_num_threads(1)
    tokenizer_sha = '760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
    expected = {'recurrent': '7feee0e9688dd4fb9014abedf677d09e071986c57ea4acdd8b8a97b61a2c28bf',
                'transformer': 'ca080bd8c6562656073dc90b64341cd624c931c1972c07c19ea9a2b4a59d1160'}
    report = {'status': 'passed', 'tokenizer_sha256': tokenizer_sha, 'models': {}, 'model_operations': 0}
    examples = ['[THREAD:0]User: Write a function. [RESP]', 'def f(x):\n    return x + 1\n',
                'Reason carefully.\n#### 42', 'Unicode: café λ 🧠', '[EOS]', '[OBSERVATION: {}]']
    for name, checksum in expected.items():
        path = getattr(args, name + '_checkpoint')
        assert digest(path) == checksum
        payload = torch.load(path, map_location='cpu', weights_only=True)
        embedded = resolve_tokenizer(payload)
        external = resolve_tokenizer(payload, args.tokenizer, tokenizer_sha)
        assert json.loads(embedded.to_str()) == json.loads(external.to_str())
        for text in examples:
            assert embedded.encode(text).ids == external.encode(text).ids
        assert embedded.get_vocab() == external.get_vocab()
        report['models'][name] = {'checkpoint_sha256': checksum, 'vocab_size': embedded.get_vocab_size(),
                                  'encoded_cases': len(examples), 'entire_tokenizer_json_equal': True}
        del payload, embedded, external
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print('POLICY_RETENTION_TOKENIZERS', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
