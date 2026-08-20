"""Test sequential 2-step recurrent training unroll."""

import unittest
import torch
import torch.nn.functional as F
import numpy as np

from irene_brain.model.thought_mediated_model import build_resource_matched_thought_model
from irene_brain.environments.phase2_suite import Phase2TaskEnvironment, make_family_suite, TaskFamily
from irene_brain.training.staged_branch_curriculum import generate_comprehensive_branch_bundle, MultiHypothesisBranchLoss
from irene_brain.types import GenericControl, HidKey

class TestSequentialTraining(unittest.TestCase):
    def test_2step_recurrent_unroll(self):
        device = torch.device("cpu")
        model = build_resource_matched_thought_model(4)
        loss_fn = MultiHypothesisBranchLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        env = Phase2TaskEnvironment(make_family_suite(TaskFamily.FAMILY_B_PURSUIT_EVASION)[0])
        obs0 = env.reset(123)
        
        # Step 0
        raw0 = np.frombuffer(obs0.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
        rgb0 = F.interpolate(torch.from_numpy(raw0.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0, size=(32, 32))
        ctrl = torch.zeros((1, 307))
        dt = torch.tensor([0.016667])
        
        state0 = model.initial_state(1)
        out0 = model(rgb0, ctrl, dt, state=state0)
        bundle0 = generate_comprehensive_branch_bundle(env, obs0)
        l0, _ = loss_fn(out0.action.proposals, [bundle0])
        
        # Env step with WAIT
        step_out = env.step(GenericControl(mouse_dx=0.0, mouse_dy=0.0, keys_down=()))
        obs1 = step_out.observation
        
        # Step 1 with preserved recurrent state
        raw1 = np.frombuffer(obs1.rgb.pixels, dtype=np.uint8).reshape((16, 16, 3))
        rgb1 = F.interpolate(torch.from_numpy(raw1.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0, size=(32, 32))
        out1 = model(rgb1, ctrl, dt, state=out0.next_state)
        bundle1 = generate_comprehensive_branch_bundle(env, obs1)
        l1, _ = loss_fn(out1.action.proposals, [bundle1])
        
        total_loss = l0 + l1
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        self.assertTrue(torch.isfinite(total_loss))

if __name__ == "__main__":
    unittest.main()
