"""CPU-only full-bank integrity and frozen foundation-encoding equivalence."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import signal
import time

import torch
from run_provenance import provenance
from irene_brain.data.policy_training_bank import prepare_policy_training_bank, load_rows, FOUNDATION_HASH
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
from train_broad_recurrent_bucket_v3 import encode_rows


def main():
    root=Path(__file__).resolve().parent
    assert not torch.cuda.is_available(), 'Run this preflight CUDA-hidden'
    torch.set_num_threads(1)
    started=time.monotonic()
    report={'status':'running','gpu_tests_run':False}
    def deadline(*unused): raise TimeoutError('Registered300-second CPU bank preflight')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(300)
    try:
        tokenizer_path=Path('/content/pb-foundation-v5-training-r1/brain/data/tokenizers/pseudo_brain_bpe_32000.json')
        assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()=='760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
        tokenizer=BpeSemanticTokenizer(tokenizer_file=tokenizer_path,auto_build_if_missing=False)
        foundation=Path('/content/pb-foundation-v5/corpus')
        bank=prepare_policy_training_bank(Path('/content/pb-procedural-repair-bank-v1/corpus'),foundation,tokenizer)
        schedule=bank['schedule']
        for epoch in range(8):
            selected=[row for row in schedule if row['epoch']==epoch]
            assert len(selected)==440
            assert Counter(row['index'] for row in selected if row['kind']=='policy')==Counter(range(352))
            for start in range(0,440,5):
                assert [row['kind'] for row in selected[start:start+5]]==['policy']*4+['foundation']
        assert len(bank['replay'])==704
        all_rows=load_rows(foundation/'train.jsonl.gz',FOUNDATION_HASH)
        indices=list(bank['replay'])
        expected=encode_rows([all_rows[index] for index in indices],tokenizer)
        for index,(_,ids,labels,prompt) in zip(indices,expected):
            actual=bank['replay'][index]
            for key,value in (('input_ids',ids),('labels',labels),('prompt_tokens',prompt)):
                assert torch.equal(actual[key],value),(index,key)
        for example in bank['train']+bank['development']+list(bank['replay'].values()):
            assert example['reset_mask'][0,0]==1 and torch.count_nonzero(example['reset_mask'][:,1:])==0
            assert torch.equal(example['input_ids'][:,:example['prompt_tokens'].shape[1]],example['prompt_tokens'])
            assert torch.isfinite(example['input_ids']).all()
            assert 0<int((example['labels']!=-100).sum())
        assert sum(example['input_ids'].numel() for example in bank['train'])==402615
        assert sum(int((example['labels']!=-100).sum()) for example in bank['train'])==61340
        report.update(status='passed',manifest=bank['manifest'],foundation_encodings_compared=704,
                      policy_train_encoded=352,policy_development_encoded=24,
                      provenance=provenance(train_seed=1986,eval_seed=1986,bank_digest=bank['manifest']['selection_sha256'],deterministic=False))
        (root/'selection.json').write_text(json.dumps(schedule,indent=2))
        (root/'bank-manifest.json').write_text(json.dumps(bank['manifest'],indent=2))
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-started
        (root/'bank-preflight.json').write_text(json.dumps(report,indent=2))
        print('POLICY_BANK_PREFLIGHT_RESULT',json.dumps(report),flush=True)


if __name__=='__main__': main()
