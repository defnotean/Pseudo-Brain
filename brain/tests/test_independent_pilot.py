import json
import pytest
import torch
from irene_brain.evaluation.independent_pilot import load_requests, grade_math, greedy_decode, score_math_bank, validate_streaming_parity


def test_rejects_grader_data_and_duplicate_tasks():
    row = {'benchmark': 'GSM8K', 'task_id': '0', 'question': 'How many?'}
    assert load_requests(json.dumps(row).encode()) == [row]
    with pytest.raises(ValueError, match='grader'):
        load_requests(json.dumps(dict(row, answer='#### 3')).encode())
    with pytest.raises(ValueError, match='Duplicate'):
        load_requests((json.dumps(row) + '\n' + json.dumps(row)).encode())


def test_math_requires_exact_final_answer_without_incidental_number_credit():
    assert grade_math('Calculations\n#### 1,200.00', 'reason\n#### 1200')['correct']
    assert grade_math('#### -12.5', '#### -12.50')['correct']
    for response in ('1200', 'We have 1200 things.', '#### 1201', '#### NaN',
                     '#### 1,20', '#### 1200\nBut actually 5', '#### 1200 or 2'):
        assert not grade_math(response, '#### 1200')['correct']
    with pytest.raises(ValueError, match='reference'):
        grade_math('#### 2', 'bad reference')


def test_math_bank_keeps_missing_tasks_and_rejects_duplicate_or_foreign_outputs():
    requests = [{'benchmark': 'GSM8K', 'task_id': str(i), 'question': 'question'} for i in range(3)]
    graders = [dict(row, answer='#### 2') for row in requests]
    responses = [{'benchmark': 'GSM8K', 'task_id': '0', 'response': '#### 2'},
                 {'benchmark': 'GSM8K', 'task_id': '1', 'response': '#### 3'}]
    report = score_math_bank(requests, graders, responses)
    assert report['status'] == 'incomplete'
    assert report['GSM8K']['correct'] == 1
    assert report['GSM8K']['total'] == 3
    assert report['GSM8K']['missing'] == 1
    assert report['HumanEval']['status'] == 'unscored'
    for bad in (responses + responses[:1], responses + [dict(responses[0], task_id='99')]):
        with pytest.raises(ValueError, match='Unexpected or duplicate'):
            score_math_bank(requests, graders, bad)
    with pytest.raises(ValueError, match='Grader bank'):
        score_math_bank(requests, graders[:2], responses)


class StreamingSpy:
    def __init__(self):
        self.initializations = 0
        self.prompts = []
        self.inputs = []

    def init_state(self, batch):
        self.initializations += 1
        return torch.zeros(batch, 16, 64, dtype=torch.float32)

    def step(self, token, state, prompt):
        self.prompts.append(prompt.clone())
        self.inputs.append(token.item())
        next_token = 4 if state[0, 0, 0] < 3 else 2
        state = state.clone()
        state[0, 0, 0] += 1
        logits = torch.zeros(1, 8)
        logits[0, next_token] = 1
        return logits, state


def test_each_task_resets_state_and_pointer_never_grows_with_generated_history():
    model = StreamingSpy()
    ids = torch.tensor([[6, 5]])
    prompt = ids[:, :1].clone()
    first = greedy_decode(model, ids, prompt, recurrent=True, eos_id=2)
    second = greedy_decode(model, ids, prompt, recurrent=True, eos_id=2)
    assert first == second
    assert first == {'token_ids': [4, 4], 'stop_reason': 'eos', 'recurrent_state_bytes': 4096}
    assert model.initializations == 2
    assert model.inputs == [6, 5, 4, 4] * 2
    assert all(torch.equal(seen, prompt) for seen in model.prompts)
    assert ids.tolist() == [[6, 5]]


def test_transformer_uses_full_prefix_but_immutable_pointer_and_respects_limit():
    class PrefixSpy:
        def __init__(self):
            self.contexts = []
        def __call__(self, ids, prompt_tokens):
            self.contexts.append(ids.clone())
            assert prompt_tokens.tolist() == [[6]]
            logits = torch.zeros(1, ids.shape[1], 8)
            logits[..., 4] = 1
            return logits
    model = PrefixSpy()
    result = greedy_decode(model, torch.tensor([[6, 5]]), torch.tensor([[6]]),
                           recurrent=False, eos_id=2, limit=3)
    assert result['token_ids'] == [4, 4, 4]
    assert result['stop_reason'] == 'token_limit'
    assert result['recurrent_state_bytes'] is None
    assert [ids.tolist() for ids in model.contexts] == [[[6, 5]], [[6, 5, 4]], [[6, 5, 4, 4]]]


def test_checkpoint_parity_gate_checks_real_model_and_rejects_nonfinite_logits(monkeypatch):
    from irene_brain.unified.parallel_depth_model import ParallelDepthConfig, ParallelDepthModel
    model = ParallelDepthModel(ParallelDepthConfig(width=64, vocab_size=128))
    receipt = validate_streaming_parity(model, lengths=(31, 65))
    assert all(row['state_bytes'] == 4096 and row['max_abs_logits'] < 1e-6 for row in receipt)
    monkeypatch.setattr(model, 'forward', lambda *args, **kwargs: torch.full((1, 31, 128), float('nan')))
    with pytest.raises(FloatingPointError, match='Nonfinite parallel'):
        validate_streaming_parity(model, lengths=(31,))
