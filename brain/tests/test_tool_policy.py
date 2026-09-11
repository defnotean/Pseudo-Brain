from dataclasses import asdict
import hashlib
from types import SimpleNamespace
import pytest
from irene_brain.evaluation.tool_policy import repair_evidence,run_tool_policy_case


def event(kind,success,before='bad',after='bad',attempted=False,verified=False):
    return {'action_type':kind,'success':success,'target_before':before,'target_after':after,
            'tests_attempted':attempted if kind=='RUN_TESTS' else False,
            'validation_attempted':attempted if kind=='FINISH' else False,'verified_completion':verified}


@pytest.mark.parametrize('events,success,expected',[
    ([event('RUN_TESTS',False,attempted=True),event('EDIT_FILE',True,after='good'),event('FINISH',True,verified=True)],True,True),
    ([event('RUN_TESTS',False),event('EDIT_FILE',True,after='good'),event('FINISH',True,verified=True)],True,False),
    ([event('UNKNOWN',False),event('WRITE_FILE',True,after='good'),event('FINISH',True,verified=True)],True,False),
    ([event('FINISH',False,attempted=True),event('WRITE_FILE',True),event('FINISH',True,verified=True)],True,False),
    ([event('RUN_TESTS',False,attempted=True),event('EDIT_FILE',True,after='good'),event('FINISH',False)],False,False),
])
def test_repair_requires_actual_attempt_changed_target_and_verified_success(events,success,expected):
    assert (repair_evidence(events,success) is not None)==expected


def test_scripted_control_preserves_public_private_separation_and_actual_repairs():
    request={'source_id':'control','family_sha256':'control-family',
             'initial_prompt':'Task: Implement twice.\nTarget: m.py\n',
             'initial_files':{'test_public.py':'from m import twice\ndef test_public(): assert twice(3)==6'}}
    grader={'source_id':'control','target_module':'m.py','reference_solution':'def twice(x): return 2*x',
            'private_check':'def test_private():\n    from m import twice\n    assert twice(-17)==-34'}
    class ScriptedControl:
        def execute_episode(self,prompt,env,**limits):
            assert '-17' not in prompt and '-17' not in str(env.files)
            assert '__private_check.py' not in env.files
            assert limits['max_action_tokens']==1024
            actions=['ACTION: RUN_TESTS','ACTION: WRITE_FILE m.py\ndef twice(x): return 2*x','ACTION: FINISH done']
            trace=[]
            for action in actions:
                feedback=env.execute_action(action)
                trace.append({'raw_action':action,'executed':True,'feedback':asdict(feedback)})
            # Dataclass is required by the scoring interface; no actual model is
            # involved in this explicitly scripted infrastructure control.
            from irene_brain.agent.parallel_depth_episode import ParallelEpisodeResult
            return ParallelEpisodeResult(True,'verified_completion',3,4096,hashlib.sha256(prompt.encode()).hexdigest(),tuple(trace))
    result=run_tool_policy_case(ScriptedControl(),request,grader,mode='repair')
    assert result['success'] and result['observed_repair']
    assert result['repair_evidence']=={'failure_event':0,'target_change_event':1,'verified_finish_event':2}
    assert request['initial_files']=={'test_public.py':'from m import twice\ndef test_public(): assert twice(3)==6'}


def test_request_with_private_fields_is_rejected_before_execution():
    with pytest.raises(ValueError,match='non-public'):
        run_tool_policy_case(None,{'reference_solution':'leak'}, {},mode='from_scratch')
