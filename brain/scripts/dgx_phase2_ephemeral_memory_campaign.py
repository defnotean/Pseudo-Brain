"""Ephemeral Memory & Hypothesis Persistence Benchmark on DGX Spark.

In this rigorous test:
1. Cues A, B, C are strictly EPHEMERAL (flash for 3 frames, then vanish completely).
2. At final execution time (t=60), the sensory input is 100% BLANK.
3. If memory/thought state is RESET, the model is blind and must fail.
4. Evaluates across 5 independent TRAINING seeds for Proposal-GRU and PB K in {1, 8, 16, 32}.
"""

import time
import math
from typing import Any, Sequence
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

    def forward(
        self,
        rgb: Tensor,
        ctrl: Tensor,
        dt: Tensor,
        state: Tensor | None = None,
        reset_state: bool = False,
        **kwargs,
    ) -> tuple[ConsequenceActionOutput, Tensor]:
        batch = rgb.shape[0]
        if state is None or reset_state:
            state = self.initial_state(batch).to(rgb.device)
        conv_feats = self.conv(rgb)
        enc_feats = self.encoder(conv_feats)
        next_h = self.gru(enc_feats, state)
        expanded_slots = self.proposal_expander(next_h).view(batch, self.num_proposals, 1, self.hidden_dim)
        pseudo_sensors = enc_feats.unsqueeze(1)
        action_out = self.actuator(sensors=pseudo_sensors, thoughts=expanded_slots)
        return action_out, next_h


class VectorizedPseudoBrain(nn.Module):
    def __init__(self, thoughtlets: int = 32, width: int = 120, heads: int = 4, cycles: int = 3, init_seed: int = 42) -> None:
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

        gen = torch.Generator().manual_seed(init_seed + thoughtlets)
        slot_identities = torch.randn(1, thoughtlets, width, generator=gen) * 0.1
        self.register_buffer("slot_identities", slot_identities)

        self.actuator = ConsequenceThoughtActuator(
            core_width=width,
            num_buttons=296,
            max_reflex_delta=0.02,
        )

    def initial_state(self, batch_size: int = 1) -> Tensor:
        return self.slot_identities.expand(batch_size, -1, -1)

    def forward(
        self,
        rgb: Tensor,
        ctrl: Tensor,
        dt: Tensor,
        state: Tensor | None = None,
        reset_state: bool = False,
        slot_permutation: Sequence[int] | None = None,
        **kwargs,
    ) -> tuple[ConsequenceActionOutput, Tensor]:
        batch = rgb.shape[0]
        device = rgb.device
        if state is None or reset_state:
            state = self.initial_state(batch).to(device)

        if slot_permutation is not None:
            state = state[:, slot_permutation]

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


# ==============================================================================
# Ephemeral Multi-Stage Training
# ==============================================================================

def train_ephemeral_model(
    model: nn.Module,
    device: torch.device,
    training_steps: int = 1500,
    lr: float = 5e-4,
) -> None:
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    is_pb = isinstance(model, VectorizedPseudoBrain)
    k_slots = model.k if is_pb else model.num_proposals

    for step in range(training_steps):
        haz_a = (step % 2 == 0)
        haz_b = ((step // 2) % 2 == 0)
        haz_c = ((step // 4) % 2 == 0)
        world_idx = (0 if haz_a else 4) + (0 if haz_b else 2) + (0 if haz_c else 1)

        blank_rgb = torch.randn(1, 3, 32, 32, device=device) * 0.05 + 0.5
        ctrl0 = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)

        # Stage 0: Ambiguity -> 8 hypotheses (p=0.125), Action: WAIT
        out0, h0 = model(blank_rgb, ctrl0, dt, state=None)
        p0 = out0.proposals.branch_probability.squeeze(-1)
        loss_p0 = F.binary_cross_entropy(p0, torch.full_like(p0, 0.125))
        loss_act0 = F.cross_entropy(out0.proposals.action_logits.view(-1, 5), torch.zeros((1, k_slots), dtype=torch.long, device=device).view(-1))

        # Stage 1: Ephemeral Flash A (3 frames), then vanished -> 4 hypotheses (p=0.25)
        rgb_flash_a = blank_rgb.clone()
        if haz_a: rgb_flash_a[:, 0, :16, :16] += 0.8
        else: rgb_flash_a[:, 0, :16, 16:] += 0.8

        _, h_flash_a = model(rgb_flash_a, ctrl0, dt, state=h0)
        # Next frame: Cue A has vanished! Observation is blank again
        out1, h1 = model(blank_rgb, ctrl0, dt, state=h_flash_a)
        p1 = out1.proposals.branch_probability.squeeze(-1)
        eighth = max(1, k_slots // 8)
        target_p1 = torch.zeros_like(p1)
        if haz_a: target_p1[:, :4*eighth] = 0.25
        else: target_p1[:, 4*eighth:] = 0.25
        loss_p1 = F.binary_cross_entropy(p1, target_p1)
        loss_act1 = F.cross_entropy(out1.proposals.action_logits.view(-1, 5), torch.zeros((1, k_slots), dtype=torch.long, device=device).view(-1))

        # Stage 2: Ephemeral Flash B (3 frames), then vanished -> 2 hypotheses (p=0.50)
        rgb_flash_b = blank_rgb.clone()
        if haz_b: rgb_flash_b[:, 1, :16, :16] += 0.8
        else: rgb_flash_b[:, 1, :16, 16:] += 0.8

        _, h_flash_b = model(rgb_flash_b, ctrl0, dt, state=h1)
        out2, h2 = model(blank_rgb, ctrl0, dt, state=h_flash_b)
        p2 = out2.proposals.branch_probability.squeeze(-1)
        half_block = (0 if haz_a else 4*eighth) + (0 if haz_b else 2*eighth)
        target_p2 = torch.zeros_like(p2)
        target_p2[:, half_block:half_block + 2*eighth] = 0.50
        loss_p2 = F.binary_cross_entropy(p2, target_p2)
        loss_act2 = F.cross_entropy(out2.proposals.action_logits.view(-1, 5), torch.zeros((1, k_slots), dtype=torch.long, device=device).view(-1))

        # Stage 3: Ephemeral Flash C (3 frames), then vanished -> 1 ground truth (p=1.00)
        rgb_flash_c = blank_rgb.clone()
        if haz_c: rgb_flash_c[:, 2, 16:, :16] += 0.8
        else: rgb_flash_c[:, 2, 16:, 16:] += 0.8

        _, h_flash_c = model(rgb_flash_c, ctrl0, dt, state=h2)
        out3, h3 = model(blank_rgb, ctrl0, dt, state=h_flash_c)
        p3 = out3.proposals.branch_probability.squeeze(-1)
        target_p3 = torch.zeros_like(p3)
        final_block = world_idx * eighth
        target_p3[:, final_block:final_block + eighth] = 1.00
        loss_p3 = F.binary_cross_entropy(p3, target_p3)

        opt_act = 1 + (world_idx % 4)
        target_act3 = torch.full((1, k_slots), opt_act, dtype=torch.long, device=device)
        loss_act3 = F.cross_entropy(out3.proposals.action_logits.view(-1, 5), target_act3.view(-1))

        total_loss = loss_p0 + loss_act0 + loss_p1 + loss_act1 + loss_p2 + loss_act2 + 2.0 * loss_p3 + loss_act3
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


# ==============================================================================
# Ephemeral Evaluation Function
# ==============================================================================

def evaluate_ephemeral_benchmark(
    model: nn.Module,
    device: torch.device,
    eval_seeds: list[int] = [1000, 2000, 3000, 4000, 5000],
    delay: int = 15,
    reset_state: bool = False,
) -> dict[str, float]:
    model.eval()
    ctrl0 = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)

    returns = []
    gamble_avoids = []
    stage1_survs = []
    stage2_survs = []
    final_accs = []

    with torch.no_grad():
        for seed in eval_seeds:
            ep_return = 0.0
            g_avoid = 0
            s1_surv = 0
            s2_surv = 0
            f_acc = 0
            episodes = 20

            for ep in range(episodes):
                np.random.seed(seed + ep * 13)
                torch.manual_seed(seed + ep * 13)

                haz_a = (np.random.rand() > 0.5)
                haz_b = (np.random.rand() > 0.5)
                haz_c = (np.random.rand() > 0.5)
                world_idx = (0 if haz_a else 4) + (0 if haz_b else 2) + (0 if haz_c else 1)
                state = None

                blank_rgb = torch.randn(1, 3, 32, 32, device=device) * 0.05 + 0.5

                # Stage 0: Ambiguity (t = 0 .. delay)
                for t in range(delay):
                    out, state = model(blank_rgb, ctrl0, dt, state=state, reset_state=reset_state)
                    act = int(torch.argmax(out.action_dist[0]).item())
                    if act != 0: ep_return -= 15.0

                if act == 0: g_avoid += 1

                # Stage 1: Cue A flashes for 3 frames, then vanishes
                flash_a = blank_rgb.clone()
                if haz_a: flash_a[:, 0, :16, :16] += 0.8
                else: flash_a[:, 0, :16, 16:] += 0.8

                for t in range(3):
                    out, state = model(flash_a, ctrl0, dt, state=state, reset_state=reset_state)
                    act = int(torch.argmax(out.action_dist[0]).item())
                    if act != 0: ep_return -= 10.0

                # Then blank for remaining delay
                for t in range(delay - 3):
                    out, state = model(blank_rgb, ctrl0, dt, state=state, reset_state=reset_state)
                    act = int(torch.argmax(out.action_dist[0]).item())
                    if act != 0: ep_return -= 10.0

                if act == 0: s1_surv += 1

                # Stage 2: Cue B flashes for 3 frames, then vanishes
                flash_b = blank_rgb.clone()
                if haz_b: flash_b[:, 1, :16, :16] += 0.8
                else: flash_b[:, 1, :16, 16:] += 0.8

                for t in range(3):
                    out, state = model(flash_b, ctrl0, dt, state=state, reset_state=reset_state)
                    act = int(torch.argmax(out.action_dist[0]).item())
                    if act != 0: ep_return -= 10.0

                for t in range(delay - 3):
                    out, state = model(blank_rgb, ctrl0, dt, state=state, reset_state=reset_state)
                    act = int(torch.argmax(out.action_dist[0]).item())
                    if act != 0: ep_return -= 10.0

                if act == 0: s2_surv += 1

                # Stage 3: Cue C flashes for 3 frames, then vanishes
                flash_c = blank_rgb.clone()
                if haz_c: flash_c[:, 2, 16:, :16] += 0.8
                else: flash_c[:, 2, 16:, 16:] += 0.8

                for t in range(3):
                    out, state = model(flash_c, ctrl0, dt, state=state, reset_state=reset_state)

                # Final Execution at t=60: Observation is 100% BLANK!
                out_final, state = model(blank_rgb, ctrl0, dt, state=state, reset_state=reset_state)
                final_act = int(torch.argmax(out_final.action_dist[0]).item())
                opt_act = 1 + (world_idx % 4)

                if final_act == opt_act:
                    f_acc += 1
                    ep_return += 20.0
                else:
                    ep_return -= 10.0

            returns.append(ep_return / episodes)
            gamble_avoids.append(g_avoid / episodes * 100.0)
            stage1_survs.append(s1_surv / episodes * 100.0)
            stage2_survs.append(s2_surv / episodes * 100.0)
            final_accs.append(f_acc / episodes * 100.0)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "median_return": float(np.median(returns)),
        "gamble_avoid_pct": float(np.mean(gamble_avoids)),
        "stage1_surv_pct": float(np.mean(stage1_survs)),
        "stage2_surv_pct": float(np.mean(stage2_survs)),
        "final_acc_pct": float(np.mean(final_accs)),
    }


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print(f"=== STRICT EPHEMERAL MEMORY & PERSISTENCE BENCHMARK (DGX Spark {device}) ===")
    print("=" * 115)

    training_seeds = [42, 142, 242, 342, 442]
    eval_seeds = [1000, 2000, 3000, 4000, 5000]

    model_configs = [
        ("Proposal-GRU Baseline", "gru", None),
        ("Pseudo-Brain K=1", "pb", 1),
        ("Pseudo-Brain K=8", "pb", 8),
        ("Pseudo-Brain K=16", "pb", 16),
        ("Pseudo-Brain K=32", "pb", 32),
    ]

    all_normal_results = {}
    all_reset_results = {}

    print(f"\n--- 1. Training across 5 Independent Training Seeds (Seeds={training_seeds}, 1500 steps) ---")
    for name, m_type, k_val in model_configs:
        seed_returns = []
        seed_resets = []
        seed_s1 = []
        seed_s2 = []
        seed_acc = []

        t0_model = time.perf_counter()
        for t_seed in training_seeds:
            torch.manual_seed(t_seed)
            np.random.seed(t_seed)

            if m_type == "gru":
                m = ProposalGRUBaseline(hidden_dim=112, num_proposals=21).to(device)
            else:
                m = VectorizedPseudoBrain(thoughtlets=k_val, width=120, heads=4, cycles=3, init_seed=t_seed).to(device)

            train_ephemeral_model(m, device=device, training_steps=1500)

            # Normal Persistent Evaluation
            res_norm = evaluate_ephemeral_benchmark(m, device=device, eval_seeds=eval_seeds, delay=15, reset_state=False)
            seed_returns.append(res_norm["mean_return"])
            seed_s1.append(res_norm["stage1_surv_pct"])
            seed_s2.append(res_norm["stage2_surv_pct"])
            seed_acc.append(res_norm["final_acc_pct"])

            # Reset Frame-by-Frame Evaluation (Must completely fail if task is strictly non-Markovian)
            res_reset = evaluate_ephemeral_benchmark(m, device=device, eval_seeds=eval_seeds, delay=15, reset_state=True)
            seed_resets.append(res_reset["mean_return"])

        elapsed = time.perf_counter() - t0_model
        all_normal_results[name] = {
            "mean_return": float(np.mean(seed_returns)),
            "std_return": float(np.std(seed_returns)),
            "median_return": float(np.median(seed_returns)),
            "s1": float(np.mean(seed_s1)),
            "s2": float(np.mean(seed_s2)),
            "acc": float(np.mean(seed_acc)),
        }
        all_reset_results[name] = {
            "mean_return": float(np.mean(seed_resets)),
            "std_return": float(np.std(seed_resets)),
        }
        print(f"  {name:26s} (5 training seeds in {elapsed:.1f}s): Normal Return = {np.mean(seed_returns):+6.2f} +/- {np.std(seed_returns):4.2f} | Reset Return = {np.mean(seed_resets):+6.2f} | Final Acc = {np.mean(seed_acc):5.1f}%")

    print("\n" + "=" * 115)
    print("--- 2. Audited Ephemeral Memory Comparison Table (5 Independent Training Seeds) ---")
    print(f"{'Model Architecture':26s} | {'Stage 1 Surv':14s} | {'Stage 2 Surv':14s} | {'Final Acc':12s} | {'Normal Return':20s} | {'Frame Reset Return':20s}")
    print("-" * 115)
    for name, _, _ in model_configs:
        nr = all_normal_results[name]
        rr = all_reset_results[name]
        print(f"{name:26s} | {nr['s1']:12.1f}% | {nr['s2']:12.1f}% | {nr['acc']:10.1f}% | {nr['mean_return']:+10.2f} +/- {nr['std_return']:4.2f} | {rr['mean_return']:+10.2f} +/- {rr['std_return']:4.2f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
