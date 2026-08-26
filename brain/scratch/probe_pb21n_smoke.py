"""Smoke-test PB21N runner: compile, load parent, graft, init-identity,
dimension check. No full evidence collection / training (those run in the
real trial)."""
from __future__ import annotations
import sys
from pathlib import Path

BRAIN = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BRAIN / "scripts"), str(BRAIN / "src")]

import importlib.util
SCRIPT = BRAIN / "scripts" / "v21n_prior_action_hazard_conditioning_v1.py"
spec = importlib.util.spec_from_file_location("pb21n", SCRIPT)
pb21n = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pb21n
spec.loader.exec_module(pb21n)
print("import OK; RUN_ID =", pb21n.RUN_ID)

model, result, prov = pb21n.load_parent()
print("parent loaded; state sha matches sealed:",
      pb21n._state_dict_sha256(model.state_dict()) == pb21n.PARENT_STATE_SHA256)

om = model.world_model.outcome_model
# dimension checks
cfg = model.config
print("config.width =", cfg.width, " world_model_hidden =", cfg.world_model_hidden)
assert cfg.width == pb21n.WIDTH, "width mismatch"
assert cfg.world_model_hidden == pb21n.HIDDEN, "hidden mismatch"
print("hazard_state_trunk in-dim =", om.hazard_state_trunk[0].in_features,
      "out =", om.hazard_state_trunk[0].out_features)
print("hazard_action_embedding rows =", om.hazard_action_embedding.num_embeddings,
      "dim =", om.hazard_action_embedding.embedding_dim)
print("hazard_outcome_trunk in =", om.hazard_outcome_trunk[0].in_features,
      "out =", om.hazard_outcome_trunk[0].out_features)
assert om.hazard_outcome_trunk[0].in_features == 2 * pb21n.HIDDEN
assert om.hazard_state_trunk[0].in_features == pb21n.WIDTH

base = pb21n.graft_base(om)
prior = pb21n.graft_prior(om)
pb21n.assert_prior_init_identity(base, prior)
print("graft init-identity OK (prior channel adds nothing at init)")

# parameter counts
print("base trainable params:", sum(p.numel() for p in base.parameters()))
print("prior trainable params:", sum(p.numel() for p in prior.parameters()))

# tiny forward sanity
import torch
g = torch.Generator(device="cpu"); g.manual_seed(0)
belief = torch.randn(8, pb21n.WIDTH, generator=g)
ids = torch.arange(pb21n.ACTION_COUNT).unsqueeze(0).expand(8, -1)
pid = torch.randint(0, pb21n.ACTION_COUNT, (8,), generator=g)
with torch.no_grad():
    out_base = base(belief.detach(), ids, None)
    out_prior = prior(belief.detach(), ids, pid)
print("base out shape:", tuple(out_base.shape), "prior out shape:", tuple(out_prior.shape))
assert out_base.shape == (8, pb21n.ACTION_COUNT)
# float32 GEMM over zero-padded cols -> sub-1e-6 gap (recorded, not equal)
gap = float((out_base - out_prior).abs().max())
assert torch.allclose(out_base, out_prior, atol=1.0e-6, rtol=1.0e-5)
print("init-identity gap:", gap, "(under the runner's 1e-6 gate)")
print("SMOKE OK")
