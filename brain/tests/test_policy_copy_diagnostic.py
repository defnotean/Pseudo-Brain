from types import SimpleNamespace

import pytest
import torch
from diagnose_policy_copy_readout import action_regions, score_readouts


def test_help_and_harm_are_distinct_from_changed_predictions():
    base = torch.tensor([[5., 0., 0.], [5., 4., 0.], [0., 1., 5.]], dtype=torch.float64)
    normal = torch.tensor([[5., 8., 0.], [9., 4., 0.], [0., 6., 5.]], dtype=torch.float64)
    targets = torch.tensor([1, 0, 2])
    result = score_readouts(normal, base, targets, torch.tensor([.2, .3, .4]), torch.tensor([False, True, False]))
    assert result['tokens'] == 3 and result['normal_correct'] == result['base_correct'] == 2
    assert result['pointer_helped'] == result['pointer_hurt'] == 1
    assert result['argmax_changed'] == 2 and result['target_present_in_prompt'] == 1
    assert result['normal_nll_sum'] == pytest.approx(float(-normal.log_softmax(-1)[torch.arange(3), targets].sum()))
    assert result['gate_sum'] == pytest.approx(.9)


def test_nonfinite_and_empty_measurements_are_rejected():
    with pytest.raises(ValueError):
        score_readouts(torch.empty(0, 3), torch.empty(0, 3), torch.empty(0, dtype=torch.long), torch.empty(0), torch.empty(0))
    with pytest.raises(ValueError):
        score_readouts(torch.tensor([[float('nan'), 0.]]), torch.zeros(1, 2), torch.tensor([0]), torch.ones(1), torch.ones(1))


def test_regions_preserve_masked_fault_and_next_token_eos_alignment():
    class Characters:
        def encode(self, text):
            return SimpleNamespace(ids=[ord(char) for char in text], offsets=[(i, i + 1) for i in range(len(text))])
    action = 'ACTION: WRITE_FILE a.py\nx=1'
    start = 4
    eos = start + len(action) + 1
    labels = torch.full((1, eos + 3), -100, dtype=torch.long)
    labels[0, start:eos - 1] = torch.tensor([ord(char) for char in action])
    labels[0, eos - 1] = 2
    encoded = {'labels': labels, 'action_spans': [{'start': 0, 'eos_position': 3, 'kind': 'injected_fault'},
                                                {'start': start, 'eos_position': eos, 'kind': 'teacher'}]}
    row = {'steps': [{'action': 'bad'}, {'action': action}]}
    regions = action_regions(row, encoded, SimpleNamespace(_hf_tokenizer=Characters()))
    assert regions[0, :start].tolist() == [-1] * start
    assert regions[0, start:start + action.index('\n') + 1].eq(0).all()
    assert regions[0, eos - 4:eos - 1].tolist() == [1, 1, 1]
    assert regions[0, eos - 1] == 2 and regions[0, eos:].eq(-1).all()
