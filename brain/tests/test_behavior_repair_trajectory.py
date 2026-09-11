import copy
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from irene_brain.agent.isolated_software_environment import IsolatedSoftwareEnvironment
from irene_brain.agent.parallel_depth_episode import ParallelEpisodeResult, observation_frame
from irene_brain.data.behavior_repair_trajectory import record_behavior_repair, encode_behavior_repair, encode_training_trajectory
from irene_brain.data.executed_trajectory import TeacherAction, digest_record
from irene_brain.evaluation.workspace_sandbox import run_workspace_tests
from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer

PROMPT='Repair m.py: implement twice(x) and verify it using test_m.py.'
FILES={'test_m.py':'from m import twice\ndef test_twice():\n    assert twice(3) == 6\n'}
REFERENCE='def twice(x): return 2*x'
BAD='ACTION: WRITE_FILE m.py\ndef twice(x): return 3*x'


@pytest.fixture(scope='module')
def tokenizer():
    path=Path(os.environ['PB_TOKENIZER_PATH'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
    return BpeSemanticTokenizer(tokenizer_file=path,auto_build_if_missing=False)


def metadata(architecture='parallel_depth_v1'):
    return {'architecture':architecture,'provenance_kind':'codec_fixture','collection_protocol':'synthetic_codec_fixture_not_model_generation',
            **{key:'0'*64 for key in ('checkpoint_sha256','tokenizer_sha256','source_manifest_sha256','initial_case_sha256')}}


def prefix(tokenizer, actions=(BAD,'ACTION: RUN_TESTS'), architecture='parallel_depth_v1'):
    def validator(files):
        result=run_workspace_tests({**files,'private.py':'from m import twice\ndef test_private():\n    assert twice(-8) == -16\n'},'private.py')
        return result['passed'],'Verified by private check' if result['passed'] else 'Private check failed'
    env=IsolatedSoftwareEnvironment(FILES,validator=validator)
    trace=[]
    consumed=tokenizer.encode(PROMPT)
    for index,action in enumerate(actions):
        # Character encoding intentionally preserves a legal but noncanonical
        # emitted sequence, exposing accidental decode/re-encode conversions.
        ids=[token for character in action for token in tokenizer.encode(character)]
        assert tokenizer.decode(ids,skip_special=False)==action
        if action:
            assert ids!=tokenizer.encode(action)
        feedback=env.execute_action(action)
        observation=tokenizer.encode(observation_frame(feedback))
        consumed += [tokenizer.resp_id]+ids+[tokenizer.eos_id]+observation
        trace.append({'cycle':index+1,'raw_action':action,'token_ids':ids,'generation_stop':'eos',
                      'executed':True,'state_bytes':4096 if architecture=='parallel_depth_v1' else None,
                      'feedback':asdict(feedback),'observation_tokens':len(observation),'observation_ingested':True})
    episode=ParallelEpisodeResult(False,'cycle_limit',len(trace),4096 if architecture=='parallel_depth_v1' else None,
                                  hashlib.sha256(PROMPT.encode()).hexdigest(),tuple(trace),
                                  None if architecture=='parallel_depth_v1' else len(consumed))
    return env,episode,consumed


def continuation():
    return [TeacherAction('ACTION: READ_FILE test_m.py'),TeacherAction('ACTION: WRITE_FILE m.py\n'+REFERENCE),
            TeacherAction('ACTION: RUN_TESTS'),TeacherAction('ACTION: FINISH done')]


def record(tokenizer, *, actions=(BAD,'ACTION: RUN_TESTS'), architecture='parallel_depth_v1'):
    env,episode,consumed=prefix(tokenizer,actions,architecture)
    result=record_behavior_repair(PROMPT,FILES,episode,env,continuation(),tokenizer,
        actor=metadata(architecture),source_id='codec-fixture',validator_id='separate-private-double-check')
    return result,consumed,env


@pytest.mark.parametrize('architecture',['parallel_depth_v1','comparison_transformer_v1'])
def test_exact_actor_tokens_teacher_only_labels_and_real_repair(tokenizer,architecture):
    result,consumed,env=record(tokenizer,architecture=architecture)
    encoded=encode_behavior_repair(result,tokenizer,allow_fixture=True)
    boundary=encoded['action_spans'][result['behavior_steps']]['start']
    assert encoded['input_ids'][0,:boundary].tolist()==consumed
    assert encoded['labels'][0,:boundary].eq(-100).all()
    assert result['steps'][1]['feedback']['success'] is False
    assert result['steps'][-2]['feedback']['success'] is True
    assert result['steps'][-1]['feedback']['verified_completion'] is True
    assert env.files['m.py']==REFERENCE
    assert encoded['reset_mask'].sum()==1 and encoded['reset_mask'][0,0]==1
    assert encoded['prompt_tokens'].tolist()==[tokenizer.encode(PROMPT)]
    for span in encoded['action_spans']:
        start,eos,end=span['start'],span['eos_position'],span['observation_end']
        assert encoded['input_ids'][0,eos]==tokenizer.eos_id
        assert encoded['labels'][0,eos:end].eq(-100).all()
        if span['kind']=='teacher':
            assert torch.equal(encoded['labels'][0,start:eos],encoded['input_ids'][0,start+1:eos+1])
    Path('fixture-'+architecture+'.json').write_text(json.dumps(result,indent=2))


@pytest.mark.parametrize('action',['','not a valid action'])
def test_immediate_eos_and_malformed_complete_behavior_remain_masked(tokenizer,action):
    result,consumed,_=record(tokenizer,actions=(action,))
    assert result['steps'][0]['feedback']['action_type']=='UNKNOWN'
    encoded=encode_behavior_repair(result,tokenizer,allow_fixture=True)
    assert encoded['input_ids'][0,:len(consumed)].tolist()==consumed
    assert encoded['labels'][0,:len(consumed)].eq(-100).all()


@pytest.mark.parametrize('stop',['action_token_limit','context_limit','observation_token_limit','verified_completion'])
def test_caps_and_finished_episodes_are_rejected_before_teacher_writes(tokenizer,stop):
    env,episode,_=prefix(tokenizer,actions=('bad action',))
    before=env.files
    episode=replace(episode,stop_reason=stop,success=stop=='verified_completion')
    with pytest.raises(ValueError,match='complete bounded actor prefixes'):
        record_behavior_repair(PROMPT,FILES,episode,env,continuation(),tokenizer,actor=metadata(),source_id='s',validator_id='v')
    assert env.files==before


@pytest.mark.parametrize('change',['partial_action','unconsumed_observation','wrong_prompt','wrong_state','wrong_workspace','protected_test','invalid_teacher_token'])
def test_prefix_mismatch_rejected_without_mutation(tokenizer,change):
    actions=('ACTION: WRITE_FILE test_m.py\npass',) if change=='protected_test' else ('bad action',)
    env,episode,_=prefix(tokenizer,actions=actions)
    teachers=continuation()
    if change=='partial_action':
        episode.trace[0]['generation_stop']='token_limit'
    elif change=='unconsumed_observation':
        episode.trace[0]['observation_ingested']=False
    elif change=='wrong_prompt':
        episode=replace(episode,initial_prompt_sha256='0'*64)
    elif change=='wrong_state':
        episode=replace(episode,state_bytes=8192)
    elif change=='wrong_workspace':
        env.execute_action('ACTION: WRITE_FILE unexpected.py\nx=1')
    elif change=='invalid_teacher_token':
        teachers.append(TeacherAction('ACTION: WRITE_FILE x.py\nx="[EOS]"'))
    before=env.files
    with pytest.raises(ValueError):
        record_behavior_repair(PROMPT,FILES,episode,env,teachers,tokenizer,actor=metadata(),source_id='s',validator_id='v')
    assert env.files==before


def test_fixture_cannot_enter_training_and_limits_never_truncate(tokenizer):
    result,_,_=record(tokenizer,actions=('bad action',))
    with pytest.raises(ValueError,match='fixtures cannot enter'):
        encode_training_trajectory(result,tokenizer)
    for limit in ('max_tokens','max_action_tokens','max_observation_tokens','max_prompt_tokens'):
        with pytest.raises(ValueError,match='limit'):
            encode_behavior_repair(result,tokenizer,allow_fixture=True,**{limit:1})


@pytest.mark.parametrize('change',['digest','actor_trace','eos','observation','prefix_hash','provenance'])
def test_record_coherence_rejected_even_after_rehash(tokenizer,change):
    result,_,_=record(tokenizer,actions=('bad action',))
    if change=='digest':
        result['source_id']='tampered'
    else:
        if change=='actor_trace': result['actor_trace'][0]['raw_action']='different'
        elif change=='eos': result['steps'][0]['terminated_by_eos']=False
        elif change=='observation': result['steps'][0]['observation_token_ids']=[123]
        elif change=='prefix_hash': result['behavior_prefix_sha256']='0'*64
        elif change=='provenance': result['actor']['provenance_kind']='model_checkpoint'
        result['record_sha256']=digest_record({k:v for k,v in result.items() if k!='record_sha256'})
    with pytest.raises(ValueError):
        encode_behavior_repair(result,tokenizer,allow_fixture=True)
