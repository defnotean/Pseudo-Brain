"""Prepare the registered v4 bank remotely; this script never trains a model."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import signal
import time

import datasets
import torch
from datasets import load_dataset
from irene_brain.data.broad_corpus import assert_isolated
from irene_brain.data.foundation_corpus import complete_exchange, quarantine_related, registered_benchmark_keys
from irene_brain.data.jsonl_records import load_jsonl_bytes
from irene_brain.data.streaming_loader import SequencePacker, RawDocument, TaskType
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer

SOURCES = [
    ('HuggingFaceH4/ultrachat_200k','default','train_sft','8049631c405ae6576f93f445c6b8166f76f5505a','language'),
    ('code-search-net/code_search_net','python','train','bd0cf261e357a3eb5c8fba490d23ec1a1cd59555','code'),
]
TOKENIZER_HASH = '760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
GSM_HASH = '17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465'
REQUEST_HASH = '1afbb0d6b1f8c3ba4f78a4b50d1040c67ee0701d97660ef96b267728f913642a'
OLD_DEV_HASH = '7c3cdcde6475c900640ea89494018a5c7eff4792be38e17d27748930dbbc010f'


def verified_bytes(path, expected):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('Input hash mismatch: '+str(path))
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--math-source',type=Path,required=True)
    parser.add_argument('--requests',type=Path,required=True)
    parser.add_argument('--prior-development',type=Path,required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=False)
    report = {'status':'running','sources':SOURCES,'datasets_version':datasets.__version__,
              'seed':198,'candidate_rows_per_hf_source':32768,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'tokenizer_sha256':TOKENIZER_HASH,'gsm_train_sha256':GSM_HASH,
              'requests_sha256':REQUEST_HASH,'prior_development_gzip_sha256':OLD_DEV_HASH,
              'partitions':{},'limits':'Exact normalized isolation only; not semantic paraphrase certification or frontier qualification.'}
    start = time.monotonic()
    def deadline(*unused):
        raise TimeoutError('Registered1800-second corpus preparation limit reached')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(1800)
    excluded = Counter()
    try:
        token_path = Path(__file__).resolve().parents[1]/'data/tokenizers/pseudo_brain_bpe_32000.json'
        verified_bytes(token_path,TOKENIZER_HASH)
        tokenizer = BpeSemanticTokenizer(tokenizer_file=token_path,auto_build_if_missing=False)
        requests = load_jsonl_bytes(verified_bytes(args.requests,REQUEST_HASH))
        benchmark_keys = registered_benchmark_keys(requests)
        old_dev = load_jsonl_bytes(gzip.decompress(verified_bytes(args.prior_development,OLD_DEV_HASH)))
        math_rows = load_jsonl_bytes(verified_bytes(args.math_source,GSM_HASH))
        assert len(math_rows) == 7473
        rows,seen = [],set()
        with (args.output/'exclusions.jsonl').open('x') as exclusion_log:
            def consider(raw,task,metadata):
                record = None
                try:
                    record = complete_exchange(raw,task)
                    ids = tokenizer.encode('[THREAD:0]'+record['text'])
                    if ids.count(tokenizer.resp_id) != 1 or ids.count(tokenizer.eos_id) != 1 or ids[-1] != tokenizer.eos_id:
                        raise ValueError('ambiguous_encoded_boundary')
                    response_tokens = len(ids)-ids.index(tokenizer.resp_id)-1
                    if response_tokens > 1024:
                        raise ValueError('response_over_1024')
                    if len(ids) > 4096:
                        raise ValueError('document_over_4096')
                    if record['text_sha256'] in seen:
                        raise ValueError('duplicate_text')
                    seen.add(record['text_sha256'])
                    record.update(metadata,token_count=len(ids),supervised_tokens=response_tokens)
                    # Question and answer are already retained in the complete text.
                    del record['question'],record['answer']
                    rows.append(record)
                except ValueError as exc:
                    reason = str(exc)
                    excluded[task+':'+reason] += 1
                    exclusion_log.write(json.dumps({**metadata,'task':task,'reason':reason,
                        'text_sha256':record.get('text_sha256') if record else None})+'\n')
            for name,config,split,revision,task in SOURCES:
                stream = load_dataset(name,name=config,split=split,revision=revision,streaming=True,token=False)
                observed = 0
                for index,raw in enumerate(stream.shuffle(seed=198,buffer_size=4096).take(32768)):
                    consider(raw,task,dict(dataset=name,config=config,split=split,revision=revision,shuffled_stream_index=index))
                    observed += 1
                    if observed % 2048 == 0:
                        print('FOUNDATION_CANDIDATES',task,observed,'retained',len(rows),flush=True)
                assert observed == 32768, 'Source exhausted before registered candidate count'
            for index,raw in enumerate(math_rows):
                consider(raw,'reasoning',dict(dataset='openai/grade-school-math',split='train',
                    revision='3101c7d5072418e28b9008a6636bde82a006892c',source_index=index))
            retained,quarantined = quarantine_related(rows,old_dev,benchmark_keys)
            for record in quarantined:
                excluded[record['task']+':quarantined_component'] += 1
                exclusion_log.write(json.dumps({'task':record['task'],'reason':'quarantined_component',
                    'text_sha256':record['text_sha256'],'component_id':record['component_id']})+'\n')
        availability = {partition:{task:sum(row['task']==task and row['partition']==partition for row in retained)
                                  for task in ('language','code','reasoning')} for partition in ('train','development')}
        report['availability'] = availability
        print('FOUNDATION_AVAILABILITY',json.dumps(availability),flush=True)
        partitions = {}
        for partition in ('train','development'):
            selected = []
            for task in ('language','code','reasoning'):
                quota = (4096 if task=='reasoning' else 8192) if partition=='train' else 128
                available = sorted((row for row in retained if row['partition']==partition and row['task']==task),key=lambda row:row['text_sha256'])
                if len(available) < quota:
                    raise RuntimeError(f'Insufficient {partition}/{task}: {len(available)} < {quota}')
                selected.extend(available[:quota])
            random.Random(198).shuffle(selected)
            partitions[partition] = selected
        assert_isolated(partitions['train'],partitions['development'])
        assert_isolated(partitions['train']+partitions['development'],old_dev)
        for partition,selected in partitions.items():
            for index,row in enumerate(selected):
                packer = SequencePacker(tokenizer,seq_len=row['token_count'],max_threads=1,
                    supervised_only_loss=True,document_policy='whole')
                packer.append_document(RawDocument(row['text'],TaskType(row['task'])))
                block = packer.emit_block()
                assert not packer._token_buffer
                assert int((block.labels != -100).sum()) == row['supervised_tokens']
                response_position = (block.input_ids == tokenizer.resp_id).nonzero()[0].item()
                assert (block.labels[:response_position] == -100).all()
                assert torch.equal(block.labels[response_position:-1],block.input_ids[response_position+1:])
                assert block.labels[-1] == -100 and block.labels[-2] == tokenizer.eos_id
                if (index+1)%2048 == 0:
                    print('FOUNDATION_PACKER_VERIFIED',partition,index+1,flush=True)
            raw = ''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in selected).encode()
            assert load_jsonl_bytes(raw) == selected
            compressed = gzip.compress(raw,mtime=0)
            (args.output/(partition+'.jsonl.gz')).write_bytes(compressed)
            report['partitions'][partition] = {'rows':len(selected),'jsonl_sha256':hashlib.sha256(raw).hexdigest(),
                'gzip_sha256':hashlib.sha256(compressed).hexdigest(),'tokens':sum(row['token_count'] for row in selected),
                'supervised_tokens':sum(row['supervised_tokens'] for row in selected),
                'domain_tokens':{task:sum(row['token_count'] for row in selected if row['task']==task) for task in ('language','code','reasoning')},
                'response_lengths':{task:dict(Counter(row['supervised_tokens'] for row in selected if row['task']==task)) for task in ('language','code','reasoning')}}
        report['status'] = 'complete'
    except Exception as exc:
        report.update(status='incomplete',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds'] = time.monotonic()-start
        report['excluded_counts'] = dict(excluded)
        (args.output/'manifest.json').write_text(json.dumps(report,indent=2))
        print('FOUNDATION_PREPARATION_RESULT',json.dumps({key:value for key,value in report.items() if key!='partitions'}),flush=True)


if __name__ == '__main__':
    main()
