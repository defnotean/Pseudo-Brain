"""CPU preparation freezes the exact schedule before any native model updates."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import signal
import torch
from irene_brain.data.broad_policy_training_bank import prepare_broad_policy_bank
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();started=time.monotonic()
    report={'status':'incomplete','model_updates':0}
    def deadline(*unused):raise TimeoutError('300-second CPU preparation deadline')
    signal.signal(signal.SIGALRM,deadline);signal.alarm(300)
    try:
        assert not torch.cuda.is_available();torch.set_num_threads(1)
        for name,expected in json.loads((args.root/'preparation-source-manifest.json').read_text()).items():
            assert hashlib.sha256((args.root/name).read_bytes()).hexdigest()==expected,name
        tokenizer_path=Path('/content/pb-foundation-v5-training-r1/brain/data/tokenizers/pseudo_brain_bpe_32000.json')
        assert hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()=='760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
        tokenizer=BpeSemanticTokenizer(tokenizer_file=tokenizer_path,auto_build_if_missing=False)
        bank=prepare_broad_policy_bank(args.root/'corpus',Path('/content/pb-foundation-v5/corpus'),tokenizer)
        (args.root/'bank-manifest.json').write_text(json.dumps(bank['manifest'],indent=2))
        (args.root/'selection.json').write_text(json.dumps(bank['schedule'],indent=2))
        report.update(status='complete',manifest=bank['manifest'])
    except BaseException as exc:
        report.update(error_type=type(exc).__name__,error=str(exc));raise
    finally:
        signal.alarm(0)
        report['elapsed_seconds']=time.monotonic()-started
        (args.root/'preparation-report.json').write_text(json.dumps(report,indent=2))
        print('BROAD_POLICY_PREPARATION',json.dumps(report),flush=True)


if __name__=='__main__':main()
