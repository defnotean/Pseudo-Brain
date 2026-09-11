"""Run the registered foundation comparison through the frozen v3 functions."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import traceback

HELPER_HASH = '1b4f833db439a2031d2f085085fe7082baeebd084ea9f2281c4d08a4f7b8e547'
helper_path = Path(__file__).with_name('train_broad_recurrent_bucket_v3.py')
if hashlib.sha256(helper_path.read_bytes()).hexdigest() != HELPER_HASH:
    raise RuntimeError('Frozen v3 training helper hash mismatch')
import train_broad_recurrent_bucket_v3 as training
from irene_brain.data.jsonl_records import load_jsonl_bytes


def load_foundation_partition(corpus, partition, manifest):
    compressed = (corpus/(partition+'.jsonl.gz')).read_bytes()
    expected = manifest['partitions'][partition]
    assert hashlib.sha256(compressed).hexdigest() == expected['gzip_sha256']
    raw = gzip.decompress(compressed)
    assert hashlib.sha256(raw).hexdigest() == expected['jsonl_sha256']
    rows = load_jsonl_bytes(raw)
    assert len(rows) == expected['rows']
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    status = args.output/'status.json'
    try:
        if not training.torch.cuda.is_available():
            raise RuntimeError('Requires authorized remote CUDA runtime')
        training.torch.set_num_threads(1)
        raw = (args.corpus/'manifest.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == args.manifest_sha256
        manifest = json.loads(raw)
        assert manifest['status'] == 'complete'
        assert manifest['partitions']['train']['rows'] == 20480
        assert manifest['partitions']['development']['rows'] == 384
        source = Path(__file__).resolve().parents[1]
        token_path = source/'data/tokenizers/pseudo_brain_bpe_32000.json'
        assert hashlib.sha256(token_path.read_bytes()).hexdigest() == manifest['tokenizer_sha256']
        tokenizer = training.BpeSemanticTokenizer(tokenizer_file=token_path,auto_build_if_missing=False)
        train_rows = load_foundation_partition(args.corpus,'train',manifest)
        dev_rows = load_foundation_partition(args.corpus,'development',manifest)
        training.assert_isolated(train_rows,dev_rows)
        assert all(row['token_count'] <= 4096 and 0 < row['supervised_tokens'] <= 1024 for row in train_rows+dev_rows)
        counts = {}
        for name,cls in [('recurrent',training.ParallelDepthModel),('transformer',training.ComparisonTransformer)]:
            temporary = cls()
            counts[name] = sum(parameter.numel() for parameter in temporary.parameters())
            del temporary
        assert max(counts.values()) < 36_000_000 and max(counts.values())/min(counts.values()) < 1.05
        provenance = {'parameters':counts,'gpu':training.torch.cuda.get_device_name(),'torch_version':training.torch.__version__,
            'corpus_manifest_sha256':args.manifest_sha256,'helper_sha256':HELPER_HASH,
            'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'registration_sha256':hashlib.sha256(Path(__file__).with_name('foundation_training_v4_registration.md').read_bytes()).hexdigest(),
            'data_registration_sha256':hashlib.sha256(Path(__file__).with_name('foundation_corpus_v4_registration.md').read_bytes()).hexdigest(),
            'source_hashes':{str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted((source/'src').rglob('*.py'))}}
        training.write_json(args.output/'provenance.json',provenance)
        print('FOUNDATION_ENCODING',len(train_rows),len(dev_rows),flush=True)
        train,dev = training.encode_rows(train_rows,tokenizer),training.encode_rows(dev_rows,tokenizer)
        for name,cls in [('recurrent',training.ParallelDepthModel),('transformer',training.ComparisonTransformer)]:
            report = training.run_model(name,cls,train,dev,tokenizer,manifest,args.output,status)
            print('FOUNDATION_MODEL_COMPLETE',name,report['training_seconds'],flush=True)
        training.write_json(status,{'stage':'complete','models':['recurrent','transformer']})
    except Exception as exc:
        training.write_json(status,{'stage':'failed','error_type':type(exc).__name__,'error':str(exc)})
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
