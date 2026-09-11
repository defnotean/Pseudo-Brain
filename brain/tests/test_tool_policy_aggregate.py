import pytest

from evaluate_tool_policy import aggregate


def test_interrupted_comparison_keeps_missing_tasks_in_denominator():
    plan = [{'source_id': str(index), 'mode': mode} for index in range(24)
            for mode in ('from_scratch', 'repair')]
    result = {'source_id': '0', 'mode': 'repair', 'success': True,
              'observed_repair': True, 'episode': {'stop_reason': 'verified_completion'},
              'events': [{'action_type': 'UNKNOWN', 'success': False},
                         {'action_type': 'FINISH', 'success': True}]}
    scores = aggregate(plan, [result])
    assert scores['completion_rate'] == scores['repair_rate'] == 1/48
    assert len(scores['missing_tasks']) == 47
    assert scores['conditions']['repair']['planned_tasks'] == 24
    assert scores['recognized_action_verbs'] == scores['successful_environment_actions'] == 1
    assert scores['stop_reasons'] == {'verified_completion': 1, 'missing': 47}
    with pytest.raises(ValueError, match='Duplicate or unplanned'):
        aggregate(plan, [result, result])
    with pytest.raises(ValueError, match='Duplicate or unplanned'):
        aggregate(plan, [{**result, 'source_id': 'unexpected'}])
    with pytest.raises(ValueError, match='without completion'):
        aggregate(plan, [{**result, 'success': False}])
