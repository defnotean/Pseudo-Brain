"""Hash-bound broader policy mixture; keeps historical data/encoders unchanged."""
from collections import Counter
import hashlib
import json
import random

import torch
from irene_brain.data.policy_training_bank import load_rows,encode_foundation_row,FOUNDATION_HASH,POLICY_HASHES
from irene_brain.data.behavior_repair_trajectory import encode_training_trajectory
from irene_brain.agent.parallel_depth_episode import observation_frame
from irene_brain.agent.isolated_software_environment import ToolFeedback

SOURCES={
    'procedural':('procedural.jsonl.gz','896a39d478aaa674903c27eb9c7af0d96017bc721843c860a3d56daa056d73a0',352),
    'varied':('varied.jsonl.gz','323aae211d6467e2805db728dc16492d5c48a662e822822579eb7406bd6158a2',220),
    'clean':('clean.jsonl.gz','04fe6f20b48a922391906469fc72d413dc42ab3ece109a88ca56b5b54db5567c',357),
    'behavior_recurrent':('behavior-recurrent.jsonl.gz','81a77eeb060ad0110be56647bd83617eb7ebf28ab8333e36b9e22f2781a28569',27),
    'behavior_transformer':('behavior-transformer.jsonl.gz','41c3b21a5e3a7edcb05c8fe46e63d8cbc181477350fd8054ebd59bc2ae69f8c8',31),
}
DOMAINS=('code','reasoning','language')
UPDATES=7896


def check_policy_encoding(record,encoded,tokenizer):
    ids,labels=encoded['input_ids'],encoded['labels']
    assert encoded['reset_mask'][0,0]==1 and int(encoded['reset_mask'].sum())==1
    prompt=encoded['prompt_tokens']
    assert 0<prompt.shape[1]<=4096 and torch.equal(ids[:,:prompt.shape[1]],prompt)
    assert torch.all(labels[:,:prompt.shape[1]]==-100)
    assert ids.numel()<=8192
    spans=encoded['action_spans']
    assert len(spans)==len(record['steps'])
    for step,span in zip(record['steps'],spans):
        start,eos,end=(span[key] for key in ('start','eos_position','observation_end'))
        assert ids[0,start]==tokenizer.resp_id and ids[0,eos]==tokenizer.eos_id
        assert eos-start<=1024 and end-eos-1<=4096
        assert torch.all(labels[0,eos:end]==-100)
        if step['kind']=='teacher':
            assert torch.equal(labels[0,start:eos],ids[0,start+1:eos+1])
        else:assert torch.all(labels[0,start:eos]==-100)
        assert len(tokenizer.encode(observation_frame(ToolFeedback(**step['feedback']))))==end-eos-1
    assert record['steps'][-1]['feedback']['verified_completion']


def prepare_broad_policy_bank(corpus,foundation_corpus,tokenizer):
    records=[];groups=[]
    for group,(filename,expected,count) in SOURCES.items():
        rows=load_rows(corpus/filename,expected)
        assert len(rows)==count
        assert all(row['partition']=='train' and row['status']=='complete' for row in rows)
        if group.startswith('behavior_'):
            assert all(row['version']=='behavior_repair_trajectory_v2' and row['actor']['provenance_kind']=='model_checkpoint' for row in rows)
        else:assert all(row['version']=='executed_trajectory_v1' for row in rows)
        records.extend(rows);groups.extend([group]*count)
    assert len(records)==987 and len({r['record_sha256'] for r in records})==987
    development=load_rows(corpus/'development.jsonl.gz',POLICY_HASHES['development.jsonl.gz'])
    assert len(development)==24
    for key in ('source_id','family_sha256'):
        assert not {r[key] for r in records}&{r[key] for r in development},key
    train=[]
    for record in records:
        encoded=encode_training_trajectory(record,tokenizer,max_tokens=8192)
        check_policy_encoding(record,encoded,tokenizer)
        train.append(encoded)
    dev=[encode_training_trajectory(row,tokenizer,max_tokens=8192) for row in development]
    foundation=load_rows(foundation_corpus/'train.jsonl.gz',FOUNDATION_HASH)
    assert len(foundation)==69632
    assert Counter(row['task'] for row in foundation)=={'code':32768,'reasoning':4096,'language':32768}
    pools={}
    for offset,domain in enumerate(DOMAINS):
        pool=[i for i,row in enumerate(foundation) if row['task']==domain]
        random.Random(1987+offset).shuffle(pool)
        pools[domain]=pool[:1974]
    replay_indices=[i for domain in DOMAINS for i in pools[domain]]
    assert len(set(replay_indices))==5922
    assert len({foundation[i]['text_sha256'] for i in replay_indices})==5922
    replay={i:encode_foundation_row(foundation[i],tokenizer) for i in replay_indices}
    schedule=[];position=0
    for epoch in range(2):
        order=list(range(987));random.Random(1987+epoch).shuffle(order)
        for i in order:
            schedule.append({'kind':'policy','group':groups[i],'index':i,'epoch':epoch})
            for domain in DOMAINS:
                schedule.append({'kind':'foundation','group':domain,'index':pools[domain][position],'epoch':epoch})
            position+=1
    assert len(schedule)==UPDATES and position==1974
    totals={}
    for row in schedule:
        encoded=train[row['index']] if row['kind']=='policy' else replay[row['index']]
        identity=records[row['index']]['record_sha256'] if row['kind']=='policy' else foundation[row['index']]['text_sha256']
        row.update(record_sha256=identity,input_tokens=encoded['input_ids'].numel(),supervised_tokens=int((encoded['labels']!=-100).sum()))
        stat=totals.setdefault(row['group'],{'updates':0,'input_tokens':0,'supervised_tokens':0})
        stat['updates']+=1
        for key in ('input_tokens','supervised_tokens'):stat[key]+=row[key]
    selection=json.dumps(schedule,sort_keys=True,separators=(',',':')).encode()
    preflight=[]
    for group in tuple(SOURCES)+DOMAINS:
        candidates=[row for row in schedule if row['group']==group]
        preflight.append(max(candidates,key=lambda row:row['input_tokens']))
    manifest={'version':'broad_policy_training_bank_v2','updates':UPDATES,'policy_records':987,'policy_updates':1974,
              'foundation_updates':5922,'unique_foundation_rows':5922,'groups':totals,
              'source_hashes':{name:value[1] for name,value in SOURCES.items()},'foundation_sha256':FOUNDATION_HASH,
              'development_sha256':POLICY_HASHES['development.jsonl.gz'],
              'selection_sha256':hashlib.sha256(selection).hexdigest(),
              'input_tokens':sum(row['input_tokens'] for row in schedule),'supervised_tokens':sum(row['supervised_tokens'] for row in schedule),
              'preflight_selection':preflight,
              'limitations':'Repeated cases across demonstrations; structural isolation only; not 987 independent tasks.'}
    return {'train':train,'development':dev,'replay':replay,'schedule':schedule,'manifest':manifest}
