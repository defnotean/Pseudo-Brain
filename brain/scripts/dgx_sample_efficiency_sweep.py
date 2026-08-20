"""Sample Efficiency & Training Horizon Scaling Benchmark on DGX Spark.

Evaluates how K=1, 4, 8, 16, 32 vs Proposal-GRU scale across training steps (200, 500, 1000, 1500)
under 2-hazard, 4-joint-hypothesis staged uncertainty across 5 independent seeds.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor

from irene_brain.model.consequence_thought_actuator import ConsequenceThoughtActuator, ConsequenceActionOutput


class ProposalGRUBaseline(nn.Module):
    def __init__(self, hidden_dim: int = 112, num_proposals: int = 21) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_proposals = num_proposals
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
        )
        self.encoder = nn.Sequential(
            nn.Linear(64 * 8 * 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.proposal_expander = nn.Linear(hidden_dim, num_proposals * hidden_dim)
        self.actuator = ConsequenceThoughtActuator(
            core_width=hidden_dim,
            num_buttons=296,
            max_reflex_delta=0.02,
        )

    def initial_state(self, batch_size: int = 1) -> Tensor:
        return torch.zeros(batch_size, self.hidden_dim)

    def forward(self, rgb: Tensor, ctrl: Tensor, dt: Tensor, state: Tensor | None = None, **kwargs) -> tuple[ConsequenceActionOutput, Tensor]:
        batch = rgb.shape[0]
        if state is None:
            state = self.initial_state(batch).to(rgb.device)
        conv_feats = self.conv(rgb)
        enc_feats = self.encoder(conv_feats)
        next_h = self.gru(enc_feats, state)
        expanded_slots = self.proposal_expander(next_h).view(batch, self.num_proposals, 1, self.hidden_dim)
        pseudo_sensors = enc_feats.unsqueeze(1)
        action_out = self.actuator(sensors=pseudo_sensors, thoughts=expanded_slots)
        return action_out, next_h


class VectorizedPseudoBrain(nn.Module):
    def __init__(self, thoughtlets: int = 32, width: int = 120, heads: int = 4, cycles: int = 3) -> None:
        super().__init__()
        self.k = thoughtlets
        self.w = width
        self.heads = heads
        self.cycles = cycles

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, width),
            nn.LayerNorm(width),
        )

        self.slot_gru = nn.GRUCell(width * 2, width)
        self.slot_norm = nn.LayerNorm(width)
        
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.attn_out = nn.Linear(width, width)
        self.attn_norm = nn.LayerNorm(width)

        gen = torch.Generator().manual_seed(42 + thoughtlets)
        slot_identities = torch.randn(1, thoughtlets, width, generator=gen) * 0.1
        self.register_buffer("slot_identities", slot_identities)

        self.actuator = ConsequenceThoughtActuator(
            core_width=width,
            num_buttons=296,
            max_reflex_delta=0.02,
        )

    def initial_state(self, batch_size: int = 1) -> Tensor:
        return self.slot_identities.expand(batch_size, -1, -1)

    def forward(self, rgb: Tensor, ctrl: Tensor, dt: Tensor, state: Tensor | None = None, **kwargs) -> tuple[ConsequenceActionOutput, Tensor]:
        batch = rgb.shape[0]
        device = rgb.device
        if state is None:
            state = self.initial_state(batch).to(device)

        sens = self.conv(rgb)
        sens_expanded = sens.unsqueeze(1).expand(batch, self.k, self.w)

        h = state
        for c in range(self.cycles):
            gru_in = torch.cat((sens_expanded, h), dim=-1).reshape(batch * self.k, self.w * 2)
            h_flat = h.reshape(batch * self.k, self.w)
            h_next = self.slot_gru(gru_in, h_flat)
            h = self.slot_norm(h_next.reshape(batch, self.k, self.w))

            q = self.q_proj(h).reshape(batch, self.k, self.heads, self.w // self.heads).transpose(1, 2)
            k = self.k_proj(h).reshape(batch, self.k, self.heads, self.w // self.heads).transpose(1, 2)
            v = self.v_proj(h).reshape(batch, self.k, self.heads, self.w // self.heads).transpose(1, 2)
            
            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).reshape(batch, self.k, self.w)
            h = self.attn_norm(h + self.attn_out(attn))

        act_out = self.actuator(sensors=sens.unsqueeze(1), thoughts=h)
        return act_out, h


def train_step_batch(model: nn.Module, optimizer: optim.Optimizer, device: torch.device, step: int) -> float:
    model.train()
    is_pb = isinstance(model, VectorizedPseudoBrain)
    k_slots = model.k if is_pb else model.num_proposals

    haz_a_left = ((step // 2) % 2 == 0)
    haz_b_left = (step % 2 == 0)
    world_idx = (0 if haz_a_left else 2) + (0 if haz_b_left else 1)

    rgb0 = torch.randn(1, 3, 32, 32, device=device) * 0.1 + 0.5
    ctrl0 = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)

    out0, h0 = model(rgb0, ctrl0, dt, state=None)
    p0 = out0.proposals.branch_probability.squeeze(-1)
    target_p0 = torch.full_like(p0, 0.25)
    loss_p0 = F.binary_cross_entropy(p0, target_p0)

    target_act0 = torch.zeros((1, k_slots), dtype=torch.long, device=device)
    loss_act0 = F.cross_entropy(out0.proposals.action_logits.view(-1, 5), target_act0.view(-1))

    rgb1 = rgb0.clone()
    if haz_a_left: rgb1[:, 0, :16, :16] += 0.8
    else: rgb1[:, 0, :16, 16:] += 0.8

    out1, h1 = model(rgb1, ctrl0, dt, state=h0)
    p1 = out1.proposals.branch_probability.squeeze(-1)

    qtr = k_slots // 4 if k_slots >= 4 else 1
    target_p1 = torch.zeros_like(p1)
    if haz_a_left: target_p1[:, :2*qtr] = 0.50
    else: target_p1[:, 2*qtr:] = 0.50
    loss_p1 = F.binary_cross_entropy(p1, target_p1)
    target_act1 = torch.zeros((1, k_slots), dtype=torch.long, device=device)
    loss_act1 = F.cross_entropy(out1.proposals.action_logits.view(-1, 5), target_act1.view(-1))

    rgb2 = rgb1.clone()
    if haz_b_left: rgb2[:, 1, 16:, :16] += 0.8
    else: rgb2[:, 1, 16:, 16:] += 0.8

    out2, h2 = model(rgb2, ctrl0, dt, state=h1)
    p2 = out2.proposals.branch_probability.squeeze(-1)

    target_p2 = torch.zeros_like(p2)
    if world_idx == 0: target_p2[:, :qtr] = 1.0
    elif world_idx == 1: target_p2[:, qtr:2*qtr] = 1.0
    elif world_idx == 2: target_p2[:, 2*qtr:3*qtr] = 1.0
    else: target_p2[:, 3*qtr:] = 1.0
    loss_p2 = F.binary_cross_entropy(p2, target_p2)

    opt_act = 1 + (world_idx % 4)
    target_act2 = torch.full((1, k_slots), opt_act, dtype=torch.long, device=device)
    loss_act2 = F.cross_entropy(out2.proposals.action_logits.view(-1, 5), target_act2.view(-1))

    total_loss = loss_p0 + loss_act0 + loss_p1 + loss_act1 + 2.0 * loss_p2 + loss_act2
    optimizer.zero_grad()
    total_loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return total_loss.item()


def evaluate_model_seeds(
    model: nn.Module,
    device: torch.device,
    seeds: list[int] = [100, 200, 300, 400, 500],
    delay: int = 15,
) -> tuple[float, float, float]:
    model.eval()
    ctrl0 = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)

    returns = []
    stage1_survivals = []
    final_accs = []

    with torch.no_grad():
        for seed in seeds:
            ep_return = 0.0
            surv_s1 = 0
            final_res = 0
            episodes = 20

            for ep in range(episodes):
                np.random.seed(seed + ep * 7)
                torch.manual_seed(seed + ep * 7)

                haz_a_left = (np.random.rand() > 0.5)
                haz_b_left = (np.random.rand() > 0.5)
                world_idx = (0 if haz_a_left else 2) + (0 if haz_b_left else 1)
                state = None

                rgb0 = torch.randn(1, 3, 32, 32, device=device) * 0.1 + 0.5
                for t in range(delay):
                    rgb_t = rgb0 + torch.randn_like(rgb0) * 0.02
                    out, state = model(rgb_t, ctrl0, dt, state=state)
                    act = int(torch.argmax(out.action_dist[0]).item())
                    if act != 0: ep_return -= 15.0

                rgb1 = rgb0.clone()
                if haz_a_left: rgb1[:, 0, :16, :16] += 0.8
                else: rgb1[:, 0, :16, 16:] += 0.8

                for t in range(delay):
                    rgb_t = rgb1 + torch.randn_like(rgb1) * 0.02
                    out, state = model(rgb_t, ctrl0, dt, state=state)
                    act = int(torch.argmax(out.action_dist[0]).item())
                    if act != 0: ep_return -= 10.0

                if act == 0: surv_s1 += 1

                rgb2 = rgb1.clone()
                if haz_b_left: rgb2[:, 1, 16:, :16] += 0.8
                else: rgb2[:, 1, 16:, 16:] += 0.8

                out_final, state = model(rgb2, ctrl0, dt, state=state)
                final_act = int(torch.argmax(out_final.action_dist[0]).item())
                opt_act = 1 + (world_idx % 4)

                if final_act == opt_act:
                    final_res += 1
                    ep_return += 20.0
                else:
                    ep_return -= 10.0

            returns.append(ep_return / episodes)
            stage1_survivals.append(surv_s1 / episodes * 100.0)
            final_accs.append(final_res / episodes * 100.0)

    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(stage1_survivals))


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print(f"=== SAMPLE EFFICIENCY & TRAINING HORIZON SWEEP (DGX Spark {device}) ===")
    print("=" * 115)

    checkpoints = [200, 500, 1000, 1500]
    seeds = [100, 200, 300, 400, 500]

    # Instantiate models and optimizers
    models = {
        "Proposal-GRU": ProposalGRUBaseline(hidden_dim=112, num_proposals=21).to(device),
        "PB K=1": VectorizedPseudoBrain(thoughtlets=1, width=120, heads=4, cycles=3).to(device),
        "PB K=4": VectorizedPseudoBrain(thoughtlets=4, width=120, heads=4, cycles=3).to(device),
        "PB K=8": VectorizedPseudoBrain(thoughtlets=8, width=120, heads=4, cycles=3).to(device),
        "PB K=16": VectorizedPseudoBrain(thoughtlets=16, width=120, heads=4, cycles=3).to(device),
        "PB K=32": VectorizedPseudoBrain(thoughtlets=32, width=120, heads=4, cycles=3).to(device),
    }
    optimizers = {name: optim.AdamW(m.parameters(), lr=5e-4, weight_decay=1e-4) for name, m in models.items()}

    print(f"\nEvaluating Learning Dynamics across {checkpoints} steps (5 Seeds):")
    print(f"{'Training Steps':16s} | {'Model':14s} | {'Stage 1 Survival':18s} | {'Mean Episode Return':22s}")
    print("-" * 115)

    curr_step = 0
    for target_step in checkpoints:
        steps_to_run = target_step - curr_step
        for step_idx in range(steps_to_run):
            step = curr_step + step_idx
            for name, m in models.items():
                train_step_batch(m, optimizers[name], device, step)
        curr_step = target_step

        print(f"\n--- Checkpoint @ {target_step} steps ---")
        for name, m in models.items():
            mean_ret, std_ret, s1_surv = evaluate_model_seeds(m, device, seeds=seeds, delay=15)
            print(f"{f'Step {target_step}':16s} | {name:14s} | {s1_surv:16.1f}% | {mean_ret:+12.2f} +/- {std_ret:.2f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
