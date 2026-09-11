import pytest
from irene_brain.evaluation.independent_code import extract_program, score_code_bank


def test_complete_code_or_single_fence_only():
    source = 'def twice(x):\n    return x * 2'
    assert extract_program(source, 'twice') == source
    assert extract_program('```python\n' + source + '\n```', 'twice') == source
    for bad in ('return x * 2', 'Here is the code:\n' + source,
                '```python\n' + source, '```javascript\n' + source + '\n```'):
        with pytest.raises(ValueError):
            extract_program(bad, 'twice')


def test_actual_code_scoring_keeps_bad_and_missing_tasks():
    requests = [{'benchmark': 'HumanEval', 'task_id': str(i)} for i in range(4)]
    graders = [dict(row, entry_point='twice', prompt='def twice(x):\n',
                    canonical_solution='    return x * 2\n',
                    test='def check(candidate):\n    assert candidate(3) == 6\n') for row in requests]
    responses = [dict(requests[0], response='def twice(x): return x * 2'),
                 dict(requests[1], response='def twice(x): return x * 3'),
                 dict(requests[2], response='unfinished gibberish')]
    report = score_code_bank(requests, graders, responses)
    assert report['correct'] == 1
    assert report['total'] == 4
    assert report['missing'] == 1
    assert report['status'] == 'incomplete'
    with pytest.raises(ValueError, match='duplicate'):
        score_code_bank(requests, graders, responses + responses[:1])
