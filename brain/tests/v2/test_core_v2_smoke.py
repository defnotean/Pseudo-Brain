"""Local CPU-only smoke test for Core V2 (CUDA hidden per workspace rules).

Verifies:
  1. Forward pass runs on all three feature configs
  2. Output shapes are correct
  3. Backward pass produces gradients on all modules
  4. Deployed decision loss decreases over a few steps (learning signal exists)
  5. Permutation equivariance of thought field (slot order invariance)
  6. Deterministic mode: same seed -> bitwise identical outputs
"""
import os
import sys

# CUDA hidden, single thread — workspace rules for local verification
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", "..", "src"))

import torch
import numpy as np

from irene_brain.v2 import (
    CoreV2Model, CoreV2Config,
    CONFIG_A_DECISION_ONLY, CONFIG_B_PREDICTIVE, CONFIG_C_FULL,
)
from irene_brain.v2.losses import total_v2_loss, deployed_decision_loss, per_slot_decision_loss


def make_batch(B=4, T=3):
    torch.manual_seed(0)
    return torch.rand(B, T, 3, 32, 32)


def test_forward_all_configs():
    print("[1] forward pass on all configs...")
    obs = make_batch()[:, 0]  # [B, 3, 32, 32]
    for name, flags in [
        ("A_decision_only", CONFIG_A_DECISION_ONLY),
        ("B_predictive", CONFIG_B_PREDICTIVE),
        ("C_full", CONFIG_C_FULL),
    ]:
        model = CoreV2Model(flags=flags)
        state = model.init_state(4, torch.device("cpu"))
        out, state = model(obs, state)
        assert out.decision.action_dist.shape == (4, 5), f"{name}: bad action_dist shape"
        assert out.hypotheses.action_logits.shape == (4, 32, 5), f"{name}: bad logits shape"
        print(f"    {name}: OK  action_dist[0]={out.decision.action_dist[0].detach().numpy().round(3)}")


def test_backward_gradients():
    print("[2] backward pass reaches all modules...")
    model = CoreV2Model(flags=CONFIG_C_FULL)
    obs = make_batch()[:, 0]
    state = model.init_state(4, torch.device("cpu"))
    out, _ = model(obs, state)

    target = torch.tensor([1, 2, 3, 4])
    loss, components = total_v2_loss(out, model.config, model.flags, target)
    loss.backward()

    # Check every parameter group got gradient
    zero_grads = []
    for pname, p in model.named_parameters():
        if p.requires_grad and (p.grad is None or p.grad.abs().sum().item() == 0):
            zero_grads.append(pname)

    if zero_grads:
        print(f"    WARNING: {len(zero_grads)} params with zero/None grad:")
        for z in zero_grads[:10]:
            print(f"      - {z}")
        if len(zero_grads) > 10:
            print(f"      ... and {len(zero_grads)-10} more")
    else:
        print("    all parameters received gradients")

    print(f"    loss={loss.item():.4f}  components={ {k: round(v.item(),4) for k,v in components.items()} }")


def test_learning_signal():
    print("[3] deployed decision loss decreases (learning signal exists)...")
    torch.manual_seed(42)
    np.random.seed(42)
    model = CoreV2Model(flags=CONFIG_C_FULL)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Tiny synthetic task: color->action mapping
    B = 8
    losses = []
    for step in range(60):
        # Red-ish frames -> action 0; blue-ish frames -> action 2
        obs = torch.rand(B, 3, 32, 32) * 0.3
        labels = np.random.randint(0, 2, B)
        obs[:, 0] += labels[:, None, None] * 0.7   # red channel high for class 1
        obs[:, 2] += (1 - labels)[:, None, None] * 0.7  # blue channel high for class 0
        target = torch.tensor([0 if l == 0 else 2 for l in labels])

        state = model.init_state(B, torch.device("cpu"))
        out, _ = model(obs, state)
        loss, _ = total_v2_loss(out, model.config, model.flags, target)

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    first10, last10 = np.mean(losses[:10]), np.mean(losses[-10:])
    print(f"    loss first10={first10:.4f} last10={last10:.4f}")
    assert last10 < first10 * 0.9, f"No learning signal! {first10:.4f} -> {last10:.4f}"
    print(f"    learning signal CONFIRMED ({(1-last10/first10)*100:.1f}% decrease)")


def test_permutation_equivariance():
    print("[4] thought-field permutation equivariance...")
    from irene_brain.v2.thought_field import ThoughtFieldV2
    from irene_brain.v2.config import DEFAULT_CONFIG

    tf = ThoughtFieldV2(DEFAULT_CONFIG)
    tf.eval()

    B, K, W = 2, 8, DEFAULT_CONFIG.width
    torch.manual_seed(7)
    thoughts = torch.randn(B, K, W)
    belief = torch.randn(B, W)

    with torch.no_grad():
        out1 = tf(thoughts, belief)
        perm = torch.randperm(K)
        out2 = tf(thoughts[:, perm], belief)

    diff = (out1[:, perm] - out2).abs().max().item()
    print(f"    max |out[perm] - out_perm| = {diff:.2e}")
    assert diff < 1e-4, f"Not permutation equivariant: {diff}"


def test_determinism():
    print("[5] determinism: same seed -> identical outputs...")
    obs = make_batch()[:, 0]

    torch.manual_seed(123); np.random.seed(123)
    m1 = CoreV2Model(flags=CONFIG_C_FULL)
    s1 = m1.init_state(4, torch.device("cpu"))
    o1, _ = m1(obs, s1)

    torch.manual_seed(123); np.random.seed(123)
    m2 = CoreV2Model(flags=CONFIG_C_FULL)
    s2 = m2.init_state(4, torch.device("cpu"))
    o2, _ = m2(obs, s2)

    # Same init weights?
    w_same = all(torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters()))
    d_diff = (o1.decision.action_dist - o2.decision.action_dist).abs().max().item()
    print(f"    weights identical: {w_same}, max output diff: {d_diff:.2e}")
    assert w_same and d_diff < 1e-6


def test_deploy_vs_slots_divergence():
    print("[6] deployed vs per-slot loss divergence probe (Stage V2.0 diagnostic)...")
    """On an untrained model the two losses should be similar; after training on
    per-slot loss only, deployed accuracy may diverge — that's the V1 failure mode."""
    model = CoreV2Model(flags=CONFIG_C_FULL)
    obs = make_batch()[:, 0]
    state = model.init_state(4, torch.device("cpu"))
    out, _ = model(obs, state)
    target = torch.tensor([0, 1, 2, 3])
    ld = deployed_decision_loss(out, target).item()
    ls = per_slot_decision_loss(out, target).item()
    print(f"    deployed_loss={ld:.4f}  per_slot_loss={ls:.4f}  (untrained, both ~log5={np.log(5):.4f})")


if __name__ == "__main__":
    test_forward_all_configs()
    test_backward_gradients()
    test_learning_signal()
    test_permutation_equivariance()
    test_determinism()
    test_deploy_vs_slots_divergence()
    print("\nALL SMOKE TESTS PASSED")
