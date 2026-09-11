"""Colab-only actual execution and causal supervision controls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import pytest
import torch

from irene_brain.data.executed_trajectory import TeacherAction, record_executed_trajectory, encode_executed_trajectory, digest_record
from irene_brain.agent.isolated_software_environment import IsolatedSoftwareEnvironment
from irene_brain.agent.parallel_depth_episode import observation_frame
from irene_brain.agent.isolated_software_environment import ToolFeedback
from irene_brain.evaluation.code_sandbox import check_program


class Tokenizer:
    resp_id=5
    eos_id=2
    pad_id=0
    def encode(self,text):
        if text=='[RESP]': return [5]
        if text=='[EOS]': return [2]
        return [byte+10 for byte in text.encode()]


@pytest.fixture(scope='module')
def recorded():
    def validator(files):
        result=check_program(files.get('m.py',''),'def check(candidate):\n    assert candidate(-8)==-16','twice')
        return result['passed'],'Verified' if result['passed'] else 'Incorrect'
    env=IsolatedSoftwareEnvironment({'test_m.py':'from m import twice\ndef test_twice(): assert twice(3)==6'},validator=validator)
    actions=[TeacherAction('ACTION: WRITE_FILE m.py\ndef twice(x): return 3*x','injected_fault'),
             TeacherAction('ACTION: RUN_TESTS'),
             TeacherAction('ACTION: EDIT_FILE m.py\n<<<TARGET\n3*x\n===\n2*x\n>>>'),
             TeacherAction('ACTION: RUN_TESTS'),TeacherAction('ACTION: FINISH done')]
    record=record_executed_trajectory('Repair m.py and verify it',env,actions,
        source_id='infrastructure-control-1984',validator_id='isolated-double-check-v1')
    Path(os.environ['PB_TRACE_OUTPUT']).write_text(json.dumps(record,indent=2))
    return record


def test_actual_outcomes_and_teacher_source_are_recorded(recorded):
    assert recorded['status']=='complete' and recorded['policy_source']=='scripted_teacher'
    assert not recorded['steps'][1]['feedback']['success']
    assert recorded['steps'][3]['feedback']['success']
    assert recorded['steps'][-1]['feedback']['verified_completion']


def test_exact_frames_and_only_causal_teacher_action_labels(recorded):
    tokenizer=Tokenizer()
    encoded=encode_executed_trajectory(recorded,tokenizer)
    ids,labels=encoded['input_ids'][0],encoded['labels'][0]
    prompt=tokenizer.encode(recorded['initial_prompt'])
    assert encoded['prompt_tokens'].tolist()==[prompt]
    expected=list(prompt)
    for step in recorded['steps']:
        expected += [5]+tokenizer.encode(step['action'])+[2]
        expected += tokenizer.encode(observation_frame(ToolFeedback(**step['feedback'])))
    assert ids.tolist()==expected
    assert (labels[:len(prompt)]==-100).all()
    for span in encoded['action_spans']:
        start,eos,end=span['start'],span['eos_position'],span['observation_end']
        assert ids[eos]==2 and (labels[eos:end]==-100).all()
        if span['kind']=='injected_fault':
            assert (labels[start:eos]==-100).all()
        else:
            assert torch.equal(labels[start:eos],ids[start+1:eos+1])
            assert labels[eos-1]==2
    assert encoded['reset_mask'].sum()==1 and encoded['reset_mask'][0,0]==1
    assert encoded['prompt_tokens'].shape[1]==len(prompt)


def test_tampered_record_and_silent_truncation_are_rejected(recorded):
    changed=copy.deepcopy(recorded)
    changed['steps'][1]['feedback']['success']=True
    with pytest.raises(ValueError,match='digest'):
        encode_executed_trajectory(changed,Tokenizer())
    with pytest.raises(ValueError,match='no truncation'):
        encode_executed_trajectory(recorded,Tokenizer(),max_tokens=30)


def test_unverified_record_cannot_become_training_data():
    env=IsolatedSoftwareEnvironment(validator=lambda files:(False,'Incomplete'))
    record=record_executed_trajectory('goal',env,[TeacherAction('ACTION: FINISH done')],source_id='source',validator_id='validator')
    assert record['status']=='incomplete'
    with pytest.raises(ValueError,match='externally completed'):
        encode_executed_trajectory(record,Tokenizer())


def test_missing_validator_and_invalid_plan_fail_before_writes():
    env=IsolatedSoftwareEnvironment()
    with pytest.raises(ValueError,match='external validation'):
        record_executed_trajectory('goal',env,[TeacherAction('ACTION: WRITE_FILE x.py\nx')],source_id='s',validator_id='v')
    assert env.files=={}
    env=IsolatedSoftwareEnvironment(validator=lambda files:(False,'Incomplete'))
    with pytest.raises(ValueError,match='Invalid teacher'):
        record_executed_trajectory('goal',env,[TeacherAction('ACTION: WRITE_FILE x.py\nx','unknown')],source_id='s',validator_id='v')
    assert env.files=={}


def test_real_frozen_tokenizer_frames_and_embedded_eos_rejection(recorded):
    from irene_brain.semantic.bpe_tokenizer import BpeSemanticTokenizer
    path=Path(os.environ['PB_TOKENIZER_PATH'])
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='760ff8f4ad5b81f529e64bf5486cc658fd90d9aea8120e7e63930ed500a5f92e'
    tokenizer=BpeSemanticTokenizer(tokenizer_file=path,auto_build_if_missing=False)
    encoded=encode_executed_trajectory(recorded,tokenizer)
    assert encoded['prompt_tokens'].tolist()==[tokenizer.encode(recorded['initial_prompt'])]
    for span in encoded['action_spans']:
        assert encoded['input_ids'][0,span['start']]==tokenizer.resp_id
        assert encoded['input_ids'][0,span['eos_position']]==tokenizer.eos_id
        assert (encoded['labels'][0,span['eos_position']:span['observation_end']]==-100).all()
    changed=copy.deepcopy(recorded)
    changed['steps'][0]['action']='ACTION: WRITE_FILE m.py\nx = "[EOS]"'
    del changed['record_sha256']
    changed['record_sha256']=digest_record(changed)
    with pytest.raises(ValueError,match='ambiguous'):
        encode_executed_trajectory(changed,tokenizer)
