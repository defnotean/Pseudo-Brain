import pytest
import torch
from irene_brain.data.sequence_buckets import bucket_training_example
from irene_brain.unified.parallel_depth_model import ParallelDepthConfig, ParallelDepthModel
from irene_brain.unified.comparison_transformer import ComparisonTransformer


def test_bucket_preserves_all_inputs_and_targets_without_truncation():
    ids = torch.arange(31)[None]
    labels = ids + 1
    labels[:, :5] = -100
    padded, targets = bucket_training_example(ids, labels, multiple=16, max_length=64)
    assert padded.shape == (1, 32)
    assert torch.equal(padded[:, :31], ids)
    assert torch.equal(targets[:, :31], labels)
    assert targets[0, 31] == -100
    assert (targets != -100).sum() == (labels != -100).sum()
    with pytest.raises(ValueError, match='context bound'):
        bucket_training_example(ids, labels, multiple=16, max_length=16)
    with pytest.raises(ValueError, match='divisible'):
        bucket_training_example(ids, labels, multiple=16, max_length=31)


@pytest.mark.parametrize('model_type', [ParallelDepthModel, ComparisonTransformer])
def test_causal_padding_preserves_supervised_loss_and_gradients(model_type):
    torch.manual_seed(9107)
    model = model_type(ParallelDepthConfig(width=64, vocab_size=128))
    ids = torch.randint(1, 128, (1, 31))
    labels = torch.roll(ids, -1, 1)
    labels[:, :5] = labels[:, -1:] = -100
    prompt = ids[:, :5].clone()
    ordinary = model.language_loss(ids, labels, prompt_tokens=prompt, chunk_size=16)
    ordinary.backward()
    grads = {name: p.grad.clone() for name, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    padded, targets = bucket_training_example(ids, labels, multiple=64, max_length=128)
    bucketed = model.language_loss(padded, targets, prompt_tokens=prompt, chunk_size=16)
    bucketed.backward()
    tolerance = 1e-8 if model_type is ParallelDepthModel else 1e-5
    assert abs(ordinary.item() - bucketed.item()) < tolerance
    for name, p in model.named_parameters():
        assert torch.isfinite(p.grad).all(), name
        assert (grads[name] - p.grad).abs().max() < tolerance, name
