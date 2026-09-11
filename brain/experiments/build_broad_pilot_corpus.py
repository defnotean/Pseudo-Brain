"""Read pinned public sources and freeze an isolated, bounded learning corpus."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import time
import datasets
from datasets import load_dataset
from irene_brain.data.broad_corpus import digest, isolation_keys, assign_components, assert_isolated
from irene_brain.data.streaming_loader import HFStreamIterator, TaskType, SequencePacker, RawDocument
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer

SOURCES = [
    ('HuggingFaceH4/ultrachat_200k', 'default', 'train_sft', '8049631c405ae6576f93f445c6b8166f76f5505a', 'language'),
    ('code-search-net/code_search_net', 'python', 'train', 'bd0cf261e357a3eb5c8fba490d23ec1a1cd59555', 'code'),
    ('open-r1/OpenR1-Math-220k', 'default', 'train', 'e4e141ec9dea9f8326f4d347be56105859b2bd68', 'reasoning'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    token_path = Path(__file__).resolve().parents[1]/'data/tokenizers/pseudo_brain_bpe_32000.json'
    tokenizer = BpeSemanticTokenizer(tokenizer_file=token_path, auto_build_if_missing=False)
    rows, excluded, seen = [], [], set()
    started = time.monotonic()
    for name, config, split, revision, task in SOURCES:
        ds = load_dataset(name, name=config, split=split, revision=revision, streaming=True, token=False)
        ds = ds.shuffle(seed=198, buffer_size=4096).take(4096)
        reader = HFStreamIterator(TaskType(task), name, revision=revision, dataset_config=config,
                                  allow_synthetic_fallback=False)
        for index, raw in enumerate(ds):
            base = dict(dataset=name, config=config, split=split, revision=revision,
                        task=task, shuffled_stream_index=index)
            text = reader._extract_text(raw).strip()
            try:
                if not text or '[RESP]' not in text:
                    raise ValueError('no_response')
                keys = isolation_keys(raw, task, text)
                text_hash = digest(text)
                if text_hash in seen:
                    raise ValueError('duplicate_text')
                formatted = '[THREAD:0]' + text
                if not formatted.endswith('[EOS]'):
                    formatted += '[EOS]'
                tokens = tokenizer.encode(formatted)
                if len(tokens) > 32768:
                    raise ValueError('over_32768_tokens')
                packer = SequencePacker(tokenizer, seq_len=len(tokens), max_threads=1,
                                       supervised_only_loss=True, document_policy='whole')
                packer.append_document(RawDocument(text, TaskType(task)))
                block = packer.emit_block()
                supervised = int((block.labels != -100).sum())
                if supervised == 0:
                    raise ValueError('no_supervised_tokens')
                seen.add(text_hash)
                rows.append({**base, 'text': text, 'text_sha256': text_hash, 'isolation_keys': keys,
                             'token_count': len(tokens), 'supervised_tokens': supervised})
            except ValueError as exc:
                excluded.append({**base, 'reason': str(exc), 'text_sha256': digest(text)})
            if (index + 1) % 512 == 0:
                print('CORPUS_SOURCE', task, index + 1, 'retained_total', len(rows), flush=True)
    assign_components(rows)
    partitions = {}
    for partition, limit in [('train', 1024), ('development', 64)]:
        selected = []
        for task in ('language', 'code', 'reasoning'):
            available = [r for r in rows if r['partition'] == partition and r['task'] == task]
            available.sort(key=lambda r: r['text_sha256'])
            if len(available) < limit:
                raise RuntimeError(f'Insufficient {partition}/{task}: {len(available)} < {limit}')
            selected.extend(available[:limit])
        random.Random(198).shuffle(selected)
        partitions[partition] = selected
    assert_isolated(partitions['train'], partitions['development'])
    manifest = {'sources': SOURCES, 'datasets_version': datasets.__version__, 'seed': 198,
        'shuffle_buffer': 4096, 'candidate_rows_per_source': 4096, 'max_context': 32768,
        'tokenizer_sha256': hashlib.sha256(token_path.read_bytes()).hexdigest(),
        'excluded_counts': dict(Counter(r['reason'] for r in excluded)),
        'related_record_isolation': 'transitive normalized question / repository / exact normalized code / text components',
        'limits': 'Development is source-derived, not an independent frontier benchmark. Semantic paraphrase or fork equivalence beyond these keys is not certified.',
        'partitions': {}, 'elapsed_seconds': time.monotonic() - started}
    for partition, selected in partitions.items():
        payload = ''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in selected).encode()
        compressed = gzip.compress(payload, mtime=0)
        (args.output/(partition + '.jsonl.gz')).write_bytes(compressed)
        manifest['partitions'][partition] = {
            'rows': len(selected), 'jsonl_sha256': hashlib.sha256(payload).hexdigest(),
            'gzip_sha256': hashlib.sha256(compressed).hexdigest(),
            'tokens': sum(r['token_count'] for r in selected),
            'supervised_tokens': sum(r['supervised_tokens'] for r in selected),
            'domain_tokens': {task: sum(r['token_count'] for r in selected if r['task'] == task)
                              for task in ('language', 'code', 'reasoning')}}
    (args.output/'exclusions.jsonl.gz').write_bytes(gzip.compress(
        ''.join(json.dumps(r) + '\n' for r in excluded).encode(), mtime=0))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('CORPUS_FROZEN', json.dumps(manifest['partitions']), flush=True)


if __name__ == '__main__':
    main()
