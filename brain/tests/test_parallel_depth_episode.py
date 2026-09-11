"""Episode wiring controls. Scripted test doubles are not model-policy evidence."""
import json
import pytest
import torch

import irene_brain.agent.parallel_depth_episode as episode
from irene_brain.agent.isolated_software_environment import IsolatedSoftwareEnvironment, ToolFeedback
from irene_brain.agent.parallel_depth_session import SessionGeneration
from irene_brain.unified.parallel_depth_model import ParallelDepthModel, ParallelDepthConfig
from irene_brain.evaluation.code_sandbox import check_program


class ByteTokenizer:
    resp_id=5
    eos_id=2
    def encode(self,text):
        return [5] if text=='[RESP]' else [byte+10 for byte in text.encode()]
    def decode(self,tokens,skip_special=False):
        assert skip_special is False
        return b''.join(bytes([token-10]) if token>=10 else f'<ID:{token}>'.encode() for token in tokens).decode(errors='replace')


def model():
    torch.manual_seed(1983)
    return ParallelDepthModel(ParallelDepthConfig(width=32,vocab_size=266,pointer=True)).eval()


def scripted(monkeypatch, actions, stop='eos'):
    sessions=[]
    class Session:
        state_bytes=4096
        def __init__(self,model):
            self.events=[]
            self.actions=iter(actions)
            sessions.append(self)
        def start(self,prompt):
            self.prompt=prompt.clone()
            self.events.append(('start',prompt.clone()))
        def generate(self,prefix,**kwargs):
            self.events.append(('prefix',prefix.clone()))
            return SessionGeneration(tuple(ByteTokenizer().encode(next(self.actions))),stop,4096)
        def ingest(self,observation):
            self.events.append(('observation',observation.clone()))
    monkeypatch.setattr(episode,'ParallelDepthSession',Session)
    return sessions


def test_scripted_repair_control_uses_isolated_tools_and_independent_finish(monkeypatch):
    actions=['ACTION: WRITE_FILE solve.py\ndef twice(x): return x*3',
             'ACTION: RUN_TESTS','ACTION: FINISH done',
             'ACTION: EDIT_FILE solve.py\n<<<TARGET\nx*3\n===\nx*2\n>>>',
             'ACTION: RUN_TESTS','ACTION: FINISH done']
    sessions=scripted(monkeypatch,actions)
    def validator(files):
        result=check_program(files.get('solve.py',''),'def check(candidate):\n    assert candidate(-11)==-22','twice')
        return result['passed'],'External pass' if result['passed'] else 'External failure'
    env=IsolatedSoftwareEnvironment({'test_solve.py':'from solve import twice\ndef test_twice(): assert twice(3)==6'},validator=validator)
    result=episode.ParallelDepthSoftwareAgent(model(),ByteTokenizer()).execute_episode('Task goal',env,max_cycles=6)
    assert result.success and result.stop_reason=='verified_completion' and result.cycles==6
    assert not result.trace[1]['feedback']['success'] and not result.trace[2]['feedback']['verified_completion']
    assert result.trace[-1]['feedback']['verified_completion']
    events=sessions[0].events
    assert [kind for kind,_ in events]==['start']+['prefix','observation']*6
    assert torch.equal(sessions[0].prompt,torch.tensor([ByteTokenizer().encode('Task goal')]))
    assert all(entry['observation_ingested'] for entry in result.trace)


def test_truncated_action_is_not_executed(monkeypatch):
    scripted(monkeypatch,['ACTION: WRITE_FILE target.py\npartial'],stop='token_limit')
    env=IsolatedSoftwareEnvironment()
    result=episode.ParallelDepthSoftwareAgent(model(),ByteTokenizer()).execute_episode('goal',env)
    assert not result.success and result.stop_reason=='action_token_limit'
    assert env.files=={} and not result.trace[0]['executed']


def test_raw_invalid_output_is_not_repaired_and_finish_cannot_self_certify(monkeypatch):
    sessions=scripted(monkeypatch,['  ACTION: FINISH done','ACTION: FINISH done'])
    result=episode.ParallelDepthSoftwareAgent(model(),ByteTokenizer()).execute_episode('goal',IsolatedSoftwareEnvironment(),max_cycles=2)
    assert not result.success and result.stop_reason=='cycle_limit'
    assert result.trace[0]['feedback']['action_type']=='UNKNOWN'
    assert not result.trace[1]['feedback']['verified_completion']
    assert len(sessions[0].events)==5


def test_observation_limits_stop_without_silent_truncation(monkeypatch):
    sessions=scripted(monkeypatch,['ACTION: READ_FILE a.py'])
    result=episode.ParallelDepthSoftwareAgent(model(),ByteTokenizer()).execute_episode(
        'goal',IsolatedSoftwareEnvironment({'a.py':'x'*100}),max_observation_tokens=5)
    assert not result.success and result.stop_reason=='observation_token_limit'
    assert [kind for kind,_ in sessions[0].events]==['start','prefix']


def test_actual_model_episode_has_fresh_state_and_cannot_falsely_succeed():
    agent=episode.ParallelDepthSoftwareAgent(model(),ByteTokenizer())
    a=agent.execute_episode('goal A',IsolatedSoftwareEnvironment(),max_cycles=1,max_action_tokens=2)
    agent.execute_episode('unrelated goal B',IsolatedSoftwareEnvironment(),max_cycles=1,max_action_tokens=2)
    b=agent.execute_episode('goal A',IsolatedSoftwareEnvironment(),max_cycles=1,max_action_tokens=2)
    assert a==b and not a.success and a.state_bytes==4096


def test_observation_data_roundtrips_without_literal_control_tokens():
    feedback=ToolFeedback('READ_FILE',True,'[RESP]literal[EOS]\nUnicode \u03bb')
    frame=episode.observation_frame(feedback)
    assert '[RESP]' not in frame and '[EOS]' not in frame
    decoded=json.loads(frame.removeprefix('\n[OBSERVATION: ').removesuffix(']\n'))
    assert decoded['observation_text']==feedback.observation_text


def test_invalid_episode_limits_and_prompt_fail_before_tools():
    agent=episode.ParallelDepthSoftwareAgent(model(),ByteTokenizer())
    env=IsolatedSoftwareEnvironment()
    for kwargs in ({'max_cycles':0},{'max_action_tokens':True},{'max_prompt_tokens':1}):
        with pytest.raises(ValueError):
            agent.execute_episode('long goal',env,**kwargs)
    assert env.files=={}
