import pytest
import torch
from train_and_eval_target_relevance import pointer_attention_supervision


def test_duplicate_positions_contribute_to_correct_output_token():
    attention = torch.tensor([[[.4, .4, .2]]], dtype=torch.float64, requires_grad=True)
    prompt = torch.tensor([[17, 17, 23]])
    target = torch.tensor([[0]])
    position_loss, position_acc, token_acc = pointer_attention_supervision(attention, target, prompt)
    token_loss, _, _ = pointer_attention_supervision(attention, target, prompt, 'token_mass')
    assert torch.allclose(position_loss, -torch.log(torch.tensor(.4 + 1e-6, dtype=torch.float64)) / (1 + 1e-6))
    assert token_loss < position_loss
    assert position_acc == 0 and token_acc > .99
    token_loss.backward()
    assert torch.isfinite(attention.grad).all()
    assert attention.grad[0, 0, 0] == attention.grad[0, 0, 1] < 0
    assert attention.grad[0, 0, 2] == 0


def test_token_mass_is_invariant_to_source_position_permutation():
    attention = torch.tensor([[[.2, .3, .5], [.6, .1, .3]]], dtype=torch.float64)
    prompt = torch.tensor([[7, 7, 9]])
    targets = torch.tensor([[0, 2]])
    loss, _, _ = pointer_attention_supervision(attention, targets, prompt, 'token_mass')
    permutation = torch.tensor([2, 0, 1])
    remapped = torch.tensor([[1, 0]])
    permuted, _, _ = pointer_attention_supervision(attention[:, :, permutation], remapped, prompt[:, permutation], 'token_mass')
    assert torch.equal(loss, permuted)


@pytest.mark.parametrize('mode', ['position', 'token_mass'])
def test_unsupervised_rows_do_not_affect_loss_or_gradients(mode):
    attention = torch.tensor([[[.1, .9], [.3, .7]]], requires_grad=True)
    targets = torch.tensor([[1, -1]])
    prompt = torch.tensor([[4, 5]])
    loss, _, _ = pointer_attention_supervision(attention, targets, prompt, mode)
    loss.backward()
    assert attention.grad[0, 0, 1] != 0
    assert torch.count_nonzero(attention.grad[0, 1]) == 0
    empty, _, _ = pointer_attention_supervision(attention, torch.full_like(targets, -1), prompt, mode)
    assert empty == 0 and torch.isfinite(empty)


def test_legacy_position_loss_matches_previous_formula():
    torch.manual_seed(17)
    attention = torch.randn(2, 7, 5, dtype=torch.float64).softmax(-1)
    targets = torch.randint(-1, 5, (2, 7))
    prompt = torch.tensor([[1, 2, 2, 3, 4], [3, 2, 2, 1, 4]])
    p = attention.gather(2, targets.clamp(min=0).unsqueeze(2)).squeeze(2)
    valid = (targets >= 0).float()
    expected = -(torch.log(p + 1e-6) * valid).sum() / (valid.sum() + 1e-6)
    actual, _, _ = pointer_attention_supervision(attention, targets, prompt)
    assert torch.equal(actual, expected)
