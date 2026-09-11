from types import SimpleNamespace
import pytest
from irene_brain.agent.procedural_evaluator import recovery_kind, verified_repair_evidence


@pytest.mark.parametrize('before,after,verb,expected', [
    ({'exists': False}, {'exists': True, 'sha256': 'new'}, 'FINISH', 'missing_file_recovery'),
    ({'exists': True, 'sha256': 'bad'}, {'exists': True, 'sha256': 'good'}, 'FINISH', 'existing_code_repair'),
    ({'exists': True, 'sha256': 'same'}, {'exists': True, 'sha256': 'same'}, 'FINISH', 'unclassified_recovery'),
    ({'exists': False}, {'exists': True, 'sha256': 'new'}, 'WRITE_FILE', 'rejected_mutation_recovery'),
    ({}, {}, 'FINISH', 'unclassified_recovery'),
])
def test_recovery_evidence_categories(before, after, verb, expected):
    failed = dict(action_type=verb, target_before=before)
    repaired = dict(target_after=after)
    assert recovery_kind(failed, repaired) == expected


def test_category_does_not_bypass_final_validation_or_autonomy():
    feedback = [
        dict(action_type='FINISH', success=False, cycle=1, transition='[PHASE: REPAIR_LOGIC]',
             target_before={'exists': True, 'sha256': 'bad'}),
        dict(action_type='WRITE_FILE', success=True, cycle=2, target_after={'exists': True, 'sha256': 'good'}),
        dict(action_type='FINISH', success=True, cycle=3),
    ]
    result = SimpleNamespace(action_feedback=feedback, success=True, policy_source='autonomous')
    assert verified_repair_evidence(result, True)['recovery_kind'] == 'existing_code_repair'
    assert not verified_repair_evidence(result, False)
    result.policy_source = 'scripted'
    assert not verified_repair_evidence(result, True)
    result.policy_source = 'autonomous'
    feedback[-1]['success'] = False
    assert not verified_repair_evidence(result, True)


def test_episode_records_file_change_without_feeding_hash_into_model(tmp_path, monkeypatch):
    from irene_brain.unified.unified_model import make_unified_model
    from irene_brain.agent.recurrent_software_agent import RecurrentSoftwareAgent
    from irene_brain.agent.software_environment import NeuralSoftwareEnvironment
    agent = RecurrentSoftwareAgent(model=make_unified_model(tier='tier1', vocab_size=32000))
    observed_inputs = []
    monkeypatch.setattr(agent, '_ingest_text_into_slot', lambda text, slot_id=0: observed_inputs.append(text))
    actions = iter(['ACTION: FINISH done', 'ACTION: WRITE_FILE a.py\ndef f():\n return 1', 'ACTION: FINISH done'])
    monkeypatch.setattr(agent, 'generate_action_autoregressive', lambda **kwargs: next(actions))
    def validate(env):
        path = env.workspace_dir / 'a.py'
        passed = path.is_file() and 'return 1' in path.read_text()
        return passed, 'passed' if passed else 'file missing'
    env = NeuralSoftwareEnvironment(tmp_path, task_validator=validate)
    result = agent.execute_pomdp_episode('Implement f in a.py', env, max_cycles=3,
        target_module='a.py', target_function='f')
    before = result.action_feedback[0]['target_before']
    after = result.action_feedback[1]['target_after']
    assert before == {'exists': False}
    assert after['exists'] and len(after['sha256']) == 64
    assert all(after['sha256'] not in text for text in observed_inputs)
    assert verified_repair_evidence(result, True)['recovery_kind'] == 'missing_file_recovery'
