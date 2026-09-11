import os
import pytest
import torch
from irene_brain.data.broad_corpus import assign_components, assert_isolated, isolation_keys
from irene_brain.unified.parallel_depth_model import ParallelDepthConfig
from irene_brain.unified.comparison_transformer import ComparisonTransformer


def test_transitive_related_records_cannot_cross_split():
    rows = [dict(isolation_keys=k, text_sha256=str(i)) for i, k in enumerate([
        ['repository:a', 'code:x'], ['repository:b', 'code:x'], ['repository:b', 'code:y']])]
    assign_components(rows)
    assert len({r['component_id'] for r in rows}) == 1
    with pytest.raises(ValueError, match='overlap'):
        assert_isolated(rows[:1], rows[1:])
    question_a = isolation_keys({'problem': '  What IS 2+2? '}, 'reasoning', 'first')[-1]
    question_b = isolation_keys({'prompt': 'What is 2+2?'}, 'language', 'second')[-1]
    assert question_a == question_b


def test_transformer_causality_and_chunked_loss():
    torch.manual_seed(198)
    model = ComparisonTransformer(ParallelDepthConfig(width=64, vocab_size=128)).to(
        os.environ.get('PB_COMPARISON_DEVICE', 'cpu'))
    device = model.embedding.weight.device
    ids = torch.randint(1, 128, (1, 23), device=device)
    changed = ids.clone()
    changed[:, 13:] = 4
    with torch.no_grad():
        assert torch.equal(model(ids)[:, :13], model(changed)[:, :13])
    labels = torch.roll(ids, -1, 1)
    labels[:, :5] = -100
    dense = torch.nn.functional.cross_entropy(model(ids).flatten(0, 1), labels.flatten())
    dense.backward()
    gradient = model.layers[0].qkv.weight.grad.clone()
    model.zero_grad(set_to_none=True)
    chunked = model.language_loss(ids, labels, chunk_size=4)
    chunked.backward()
    assert abs(dense.item() - chunked.item()) < 1e-5
    relative_error = (gradient - model.layers[0].qkv.weight.grad).norm() / gradient.norm()
    print('CHUNK_GRADIENT_RELATIVE_ERROR', relative_error.item())
    # Flash attention rounds its adjoints to bfloat16. Different loss-reduction
    # order can cross a rounding boundary even though the objective is identical.
    tolerance = torch.finfo(torch.bfloat16).eps if device.type == 'cuda' else 1e-5
    assert relative_error < tolerance
    assert model.layers[-1].qkv.weight.grad.abs().sum() > 0
    resets = torch.zeros_like(ids)
    resets[:, 12] = 1
    with pytest.raises(ValueError, match='whole document'):
        model(ids, resets)
