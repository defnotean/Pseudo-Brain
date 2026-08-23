import sys, torch
sys.path.insert(0, "/workspace/repo/brain/src")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "esc", "/workspace/repo/brain/scripts/dgx_phase2_definitive_escalation.py")
esc = importlib.util.module_from_spec(spec)
sys.modules["esc"] = esc
spec.loader.exec_module(esc)

import types
src = open("/workspace/repo/brain/scripts/braincell_candidates_screening.py").read()
src = src.replace(
    'os.path.join(_here, "..", "..", "dgx_phase2_definitive_escalation.py")',
    '"/workspace/repo/brain/scripts/dgx_phase2_definitive_escalation.py"')
mod = types.ModuleType("bcs")
mod.__dict__["__name__"] = "bcs"
mod.__dict__["__file__"] = "/workspace/repo/brain/scripts/braincell_candidates_screening.py"
exec(compile(src, "bcs", "exec"), mod.__dict__)

base = esc.VectorizedPseudoBrain()
m = mod.CandidateCore(base, "current", torch.device("cpu"))
rgb = torch.rand(1, 3, 32, 32)
ctrl = torch.zeros(1, 307)
dt = torch.tensor([0.016667])
out, h = m(rgb, ctrl, dt)
print("wrapper logits:", tuple(out.proposals.action_logits.shape))
print("wrapper dist:", tuple(out.action_dist.shape))
outb, hb = base(rgb, ctrl, dt)
print("base logits:", tuple(outb.proposals.action_logits.shape))
for name in dir(out.proposals):
    if not name.startswith("_"):
        v = getattr(out.proposals, name)
        if torch.is_tensor(v):
            print("  proposals." + name, tuple(v.shape))
print("PROBE DONE")
