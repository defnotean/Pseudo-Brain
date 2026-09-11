"""Bind post-training evaluation to both completed v2 arms and parity receipts."""
import json
import math
from evaluate_tool_policy import digest

SELECTION='16cbc46c6fe8b380024634520cfc210e2879737278ce8d1b2a5eeed3087cd949'
TRAINING_SOURCE='442afc835cef0b4ae9f269f48ef353c16401e0ad1e5511fae9587ccdb28baf3d'
INITIAL={'recurrent':'0b3edca04de150068c81fb8f71f05cd0aad8f42e423b9989bc789b919ffb85b7',
         'transformer':'346c9d4c9d334490cb2e93c54f4a1f0f66020a3833b4e7565f015b6d08c44dfe'}


def resolve_completed_checkpoint(training_run,report_sha256,model):
    if model not in INITIAL:raise ValueError('Unknown comparison arm')
    path=training_run/'report.json'
    if digest(path)!=report_sha256:raise ValueError('Training report hash mismatch')
    pair=json.loads(path.read_text())
    if (pair.get('status')!='complete' or pair.get('native_preflight') is not False or
        pair.get('protocol')!='broad_policy_v2' or pair.get('source_manifest_sha256')!=TRAINING_SOURCE or
        pair.get('selection_sha256')!=SELECTION or set(pair.get('models',{}))!=set(INITIAL)):
        raise ValueError('Expected completed registered broader training pair')
    manifest=training_run.parent/'training-source-hashes.json'
    if digest(manifest)!=TRAINING_SOURCE:raise ValueError('Training source identity changed')
    for relative,expected in json.loads(manifest.read_text()).items():
        if digest(training_run.parent/relative)!=expected:raise ValueError('Frozen source changed: '+relative)
    bank=json.loads((training_run.parent/'bank-manifest.json').read_text())
    for name,entry in pair['models'].items():
        child=entry['child_report']
        if (entry.get('returncode')!=0 or entry.get('status')!='complete' or child.get('status')!='complete' or
            child.get('protocol')!='broad_policy_v2' or child.get('native_preflight') is not False or
            child.get('completed_updates')!=7896 or child.get('planned_updates')!=7896 or
            child.get('selection_sha256')!=SELECTION or child.get('source_manifest_sha256')!=TRAINING_SOURCE or
            child.get('checkpoint_sha256')!=INITIAL[name] or child.get('counts')!=bank['groups']):
            raise ValueError('Missing, partial or mismatched training arm')
        if json.loads((training_run/name/'report.json').read_text())!=child:raise ValueError('Child report differs from paired receipt')
        receipt=child['checkpoint']
        if (receipt.get('checkpoint')!='checkpoint-7896-trained.pt' or receipt.get('completed_updates')!=7896 or
            receipt.get('status')!='trained' or receipt.get('selection_sha256')!=SELECTION or receipt.get('counts')!=bank['groups']):
            raise ValueError('Only the final registered boundary is eligible')
        if digest(training_run/name/receipt['checkpoint'])!=receipt['sha256']:raise ValueError('Final checkpoint hash mismatch')
        if name=='recurrent':
            parity=child.get('parity',[])
            if [row.get('tokens') for row in parity]!=[31,257,2049]:raise ValueError('Missing recurrent parity lengths')
            if any(row.get('state_bytes')!=4096 or not math.isfinite(row['max_abs_logits']) or not 0<=row['max_abs_logits']<1e-6 for row in parity):
                raise ValueError('Recurrent memory or numerical gate failed')
    child=pair['models'][model]['child_report'];receipt=child['checkpoint']
    return training_run/model/receipt['checkpoint'],receipt['sha256'],child
