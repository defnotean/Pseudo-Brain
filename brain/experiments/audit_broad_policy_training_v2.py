"""Audit complete update logs, exact schedule and development measurements."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def read(path):return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','previous','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run=args.root/'training'
    manifest=args.root/'training-source-hashes.json'
    assert hashlib.sha256(manifest.read_bytes()).hexdigest()=='442afc835cef0b4ae9f269f48ef353c16401e0ad1e5511fae9587ccdb28baf3d'
    for name,expected in read(manifest).items():assert hashlib.sha256((args.root/name).read_bytes()).hexdigest()==expected,name
    bank=read(args.root/'bank-manifest.json');schedule=read(args.root/'selection.json')
    selection=hashlib.sha256(json.dumps(schedule,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert selection==bank['selection_sha256']=='16cbc46c6fe8b380024634520cfc210e2879737278ce8d1b2a5eeed3087cd949'
    assert len(schedule)==7896
    for epoch in range(2):
        rows=[row for row in schedule if row['epoch']==epoch]
        assert len(rows)==3948 and Counter(row['index'] for row in rows if row['kind']=='policy')==Counter(range(987))
        for start in range(0,len(rows),4):
            assert [row['kind'] for row in rows[start:start+4]]==['policy','foundation','foundation','foundation']
            assert [row['group'] for row in rows[start+1:start+4]]==['code','reasoning','language']
    foundation=[row for row in schedule if row['kind']=='foundation']
    assert len({row['index'] for row in foundation})==len({row['record_sha256'] for row in foundation})==5922
    pair=read(run/'report.json');assert pair['status']=='complete' and set(pair['models'])=={'recurrent','transformer'}
    results={}
    for name in ('recurrent','transformer'):
        child=read(run/name/'report.json')
        assert pair['models'][name]['child_report']==child and pair['models'][name]['returncode']==0
        assert child['status']=='complete' and child['completed_updates']==child['planned_updates']==7896
        log=[json.loads(line) for line in (run/name/'updates.jsonl').read_text().splitlines()]
        assert len(log)==7896
        totals={}
        for index,(actual,expected) in enumerate(zip(log,schedule)):
            assert actual['update']==index+1 and all(actual[key]==value for key,value in expected.items())
            for key in ('loss','gradient_norm','lr'):assert math.isfinite(actual[key]) and actual[key]>=0
            lr=3e-5*(index+1)/64 if index<64 else 3e-6+.5*(3e-5-3e-6)*(1+math.cos(math.pi*(index-64)/(7896-65)))
            assert actual['lr']==lr
            stat=totals.setdefault(actual['group'],{'updates':0,'input_tokens':0,'supervised_tokens':0})
            stat['updates']+=1
            for key in ('input_tokens','supervised_tokens'):stat[key]+=actual[key]
        assert totals==child['counts']==bank['groups']
        measurements={}
        for domain in ('policy','foundation'):
            initial=read(run/name/('initial-'+domain+'-development.json'))
            final=read(run/name/('final-'+domain+'-development.json'))
            previous=read(args.previous/name/('final-'+domain+'-development.json'))
            assert initial==previous
            assert len(initial['examples'])==len(final['examples'])==(24 if domain=='policy' else 1536)
            assert [{k:v for k,v in row.items() if k!='nll'} for row in initial['examples']]==[{k:v for k,v in row.items() if k!='nll'} for row in final['examples']]
            assert all(math.isfinite(row['nll']) and row['nll']>=0 for row in final['examples'])
            for group,stat in initial['groups'].items():
                after=final['groups'][group]
                assert stat['examples']==after['examples'] and stat['supervised_tokens']==after['supervised_tokens']
                measurements[group]={'initial_nll':stat['token_weighted_nll'],'final_nll':after['token_weighted_nll'],
                                     'delta_nll':after['token_weighted_nll']-stat['token_weighted_nll']}
        results[name]={'updates_verified':7896,'initial_evaluation_equals_previous_final':True,'counts':totals,'development':measurements,
                       'checkpoint':child['checkpoint'],'parity':child.get('parity')}
    result={'status':'complete','selection_sha256':selection,'training_report_sha256':hashlib.sha256((run/'report.json').read_bytes()).hexdigest(),
            'models':results,'model_updates_during_audit':0,'autonomous_evaluation':'not performed by this audit'}
    args.output.write_text(json.dumps(result,indent=2))
    print('BROAD_POLICY_TRAINING_AUDIT',json.dumps(result),flush=True)


if __name__=='__main__':main()
