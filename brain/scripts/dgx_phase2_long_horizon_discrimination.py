"""Long-Horizon Multi-Stage Uncertainty Discrimination Benchmark.

Tests whether parallel thoughtlets outperform a monolithic recurrent GRU when:
1. Multiple independent stochastic entities create 4-8 joint hypotheses (e.g. 2 hazards branching 50/50).
2. Evidence arrives in stages over extended delays (D1=15, D2=30 frames).
3. Early commitment is fatal (75% penalty), requiring sustained hypothesis coexistence and sequential posterior collapse.
4. Validates whole-slot permutation invariance with identity codes.
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


# ==============================================================================
# 1. Architecture Implementations
# ==============================================================================

class ProposalGRUBaseline(nn.Module):
    """Monolithic GRU baseline with identical multi-hypothesis consequence proposal decoder."""
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
    """Vectorized Slot-Recurrent Pseudo-Brain with exchangeable hypothesis identity codes."""
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

        # Exchangeable slot identity codes (transported together with whole slot)
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

    def forward(
        self,
        rgb: Tensor,
        ctrl: Tensor,
        dt: Tensor,
        state: Tensor | None = None,
        slot_permutation: Sequence[int] | None = None,
        **kwargs,
    ) -> tuple[ConsequenceActionOutput, Tensor]:
        batch = rgb.shape[0]
        device = rgb.device
        if state is None:
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
# 2. Permutation Invariance Regression Test
# ==============================================================================

def test_permutation_invariance(model: VectorizedPseudoBrain, device: torch.device) -> float:
    """Verify that whole-slot permutation produces numerical invariance (< 1e-5)."""
    model.eval()
    rgb = torch.randn(1, 3, 32, 32, device=device)
    ctrl = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)

    with torch.no_grad():
        out_normal, h_normal = model(rgb, ctrl, dt, state=None)
        
        # Random permutation of slots
        perm = np.random.permutation(model.k).tolist()
        out_perm, h_perm = model(rgb, ctrl, dt, state=None, slot_permutation=perm)

        # Difference in action distribution
        act_diff = (out_normal.action_dist - out_perm.action_dist).abs().max().item()
    return act_diff


# ==============================================================================
# 3. Multi-Stage Long-Horizon Training Curriculum
# ==============================================================================

def train_multistage_stochastic(
    model: nn.Module,
    device: torch.device,
    training_steps: int = 500,
    lr: float = 5e-4,
) -> None:
    """Train on 2-hazard, 4-joint-hypothesis staged resolution curriculum."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    is_pb = isinstance(model, VectorizedPseudoBrain)
    k_slots = model.k if is_pb else model.num_proposals

    for step in range(training_steps):
        # Hazard A (Left=0 / Right=1) and Hazard B (Left=0 / Right=1)
        # Joint world state: 0=LL, 1=LR, 2=RL, 3=RR
        haz_a_left = ((step // 2) % 2 == 0)
        haz_b_left = (step % 2 == 0)
        world_idx = (0 if haz_a_left else 2) + (0 if haz_b_left else 1)

        # Stage 0: Initial Ambiguity (t=0..15) -> 4 equally likely hypotheses
        rgb0 = torch.randn(1, 3, 32, 32, device=device) * 0.1 + 0.5
        ctrl0 = torch.zeros(1, 307, device=device)
        dt = torch.tensor([0.016667], device=device)

        out0, h0 = model(rgb0, ctrl0, dt, state=None)
        p0 = out0.proposals.branch_probability.squeeze(-1)
        target_p0 = torch.full_like(p0, 0.25)
        loss_p0 = F.binary_cross_entropy(p0, target_p0)

        # Action condition at t=0: WAIT (action 0)
        act_logits0 = out0.proposals.action_logits
        target_act0 = torch.zeros((1, k_slots), dtype=torch.long, device=device)
        loss_act0 = F.cross_entropy(act_logits0.view(-1, 5), target_act0.view(-1))

        # Stage 1: Partial Reveal at t=15 (Hazard A direction is revealed, B still hidden)
        rgb1 = rgb0.clone()
        if haz_a_left:
            rgb1[:, 0, :16, :16] += 0.8  # Cue A reveals Left
        else:
            rgb1[:, 0, :16, 16:] += 0.8  # Cue A reveals Right

        out1, h1 = model(rgb1, ctrl0, dt, state=h0)
        p1 = out1.proposals.branch_probability.squeeze(-1)

        # Target branch probabilities collapse from 4 to 2 active hypotheses (p=0.50)
        qtr = k_slots // 4 if k_slots >= 4 else 1
        target_p1 = torch.zeros_like(p1)
        if haz_a_left:
            target_p1[:, :2*qtr] = 0.50
        else:
            target_p1[:, 2*qtr:] = 0.50
        loss_p1 = F.binary_cross_entropy(p1, target_p1)
        target_act1 = torch.zeros((1, k_slots), dtype=torch.long, device=device)  # WAIT (0)
        loss_act1 = F.cross_entropy(out1.proposals.action_logits.view(-1, 5), target_act1.view(-1))

        # Stage 2: Final Reveal at t=30 (Hazard B direction is revealed)
        rgb2 = rgb1.clone()
        if haz_b_left:
            rgb2[:, 1, 16:, :16] += 0.8  # Cue B reveals Left
        else:
            rgb2[:, 1, 16:, 16:] += 0.8  # Cue B reveals Right

        out2, h2 = model(rgb2, ctrl0, dt, state=h1)
        p2 = out2.proposals.branch_probability.squeeze(-1)

        # Target branch probability collapses to single ground truth (p=1.0)
        target_p2 = torch.zeros_like(p2)
        if world_idx == 0: target_p2[:, :qtr] = 1.0
        elif world_idx == 1: target_p2[:, qtr:2*qtr] = 1.0
        elif world_idx == 2: target_p2[:, 2*qtr:3*qtr] = 1.0
        else: target_p2[:, 3*qtr:] = 1.0
        loss_p2 = F.binary_cross_entropy(p2, target_p2)

        # Final action decision: Directional escape key matching joint world resolution
        opt_act = 1 + (world_idx % 4)  # Actions 1..4
        target_act2 = torch.full((1, k_slots), opt_act, dtype=torch.long, device=device)
        loss_act2 = F.cross_entropy(out2.proposals.action_logits.view(-1, 5), target_act2.view(-1))

        total_loss = loss_p0 + loss_act0 + loss_p1 + loss_act1 + 2.0 * loss_p2 + loss_act2
        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


# ==============================================================================
# 4. Long-Horizon Multi-Stage Discrimination Benchmark
# ==============================================================================

def run_long_horizon_benchmark(
    model: nn.Module,
    device: torch.device,
    seeds: list[int] = [100, 200, 300, 400, 500],
    delay_stage1: int = 15,
    delay_stage2: int = 15,
) -> dict[str, float]:
    """Evaluates 5-seed performance on multi-stage delayed uncertainty resolution."""
    model.eval()
    ctrl0 = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)

    returns = []
    gamble_avoid_rates = []
    stage1_survival_rates = []
    final_resolution_accs = []
    entropy_reductions_stage1 = []
    entropy_reductions_stage2 = []

    with torch.no_grad():
        for seed in seeds:
            ep_return = 0.0
            gamble_avoid_count = 0
            stage1_survivals = 0
            final_resolutions = 0
            episodes = 25

            for ep in range(episodes):
                np.random.seed(seed + ep * 7)
                torch.manual_seed(seed + ep * 7)

                haz_a_left = (np.random.rand() > 0.5)
                haz_b_left = (np.random.rand() > 0.5)
                world_idx = (0 if haz_a_left else 2) + (0 if haz_b_left else 1)
                state = None

                # -------------------------------------------------------------
                # Stage 0: Initial Delay (Ticks 0 .. D1)
                # -------------------------------------------------------------
                rgb0 = torch.randn(1, 3, 32, 32, device=device) * 0.1 + 0.5
                p0_norm = None
                for t in range(delay_stage1):
                    # Unobserved sensory drift
                    rgb_t = rgb0 + torch.randn_like(rgb0) * 0.02
                    out, state = model(rgb_t, ctrl0, dt, state=state)
                    act = int(torch.argmax(out.action_dist[0]).item())
                    if t == 0:
                        p0 = out.proposals.branch_probability.squeeze(-1)[0].cpu().numpy()
                        p0_norm = np.clip(p0, 1e-6, 1.0) / np.sum(np.clip(p0, 1e-6, 1.0))

                    if act == 0:
                        ep_return += 0.0  # Safe probe
                    else:
                        # Fatal blind gamble before information arrives
                        ep_return -= 15.0

                # Check if agent successfully waited through Stage 0
                if act == 0:
                    gamble_avoid_count += 1

                # -------------------------------------------------------------
                # Stage 1: Partial Reveal (Cue A appears) (Ticks D1 .. D1+D2)
                # -------------------------------------------------------------
                rgb1 = rgb0.clone()
                if haz_a_left:
                    rgb1[:, 0, :16, :16] += 0.8
                else:
                    rgb1[:, 0, :16, 16:] += 0.8

                p1_norm = None
                for t in range(delay_stage2):
                    rgb_t = rgb1 + torch.randn_like(rgb1) * 0.02
                    out, state = model(rgb_t, ctrl0, dt, state=state)
                    act = int(torch.argmax(out.action_dist[0]).item())
                    if t == 0:
                        p1 = out.proposals.branch_probability.squeeze(-1)[0].cpu().numpy()
                        p1_norm = np.clip(p1, 1e-6, 1.0) / np.sum(np.clip(p1, 1e-6, 1.0))

                    if act == 0:
                        ep_return += 0.0  # Correct second probe
                    else:
                        ep_return -= 10.0

                if act == 0:
                    stage1_survivals += 1

                # -------------------------------------------------------------
                # Stage 2: Final Reveal (Cue B appears) & Execution
                # -------------------------------------------------------------
                rgb2 = rgb1.clone()
                if haz_b_left:
                    rgb2[:, 1, 16:, :16] += 0.8
                else:
                    rgb2[:, 1, 16:, 16:] += 0.8

                out_final, state = model(rgb2, ctrl0, dt, state=state)
                final_act = int(torch.argmax(out_final.action_dist[0]).item())
                opt_act = 1 + (world_idx % 4)

                if final_act == opt_act:
                    final_resolutions += 1
                    ep_return += 20.0  # Full task completion
                else:
                    ep_return -= 10.0

                # Entropy reductions
                h0 = -np.sum(p0_norm * np.log2(p0_norm))
                h1 = -np.sum(p1_norm * np.log2(p1_norm))
                p2 = out_final.proposals.branch_probability.squeeze(-1)[0].cpu().numpy()
                p2_norm = np.clip(p2, 1e-6, 1.0) / np.sum(np.clip(p2, 1e-6, 1.0))
                h2 = -np.sum(p2_norm * np.log2(p2_norm))

                entropy_reductions_stage1.append(max(0.0, (h0 - h1) / max(1e-6, h0) * 100.0))
                entropy_reductions_stage2.append(max(0.0, (h0 - h2) / max(1e-6, h0) * 100.0))

            returns.append(ep_return / episodes)
            gamble_avoid_rates.append(gamble_avoid_count / episodes * 100.0)
            stage1_survival_rates.append(stage1_survivals / episodes * 100.0)
            final_resolution_accs.append(final_resolutions / episodes * 100.0)

    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "gamble_avoid_pct": float(np.mean(gamble_avoid_rates)),
        "stage1_survival_pct": float(np.mean(stage1_survival_rates)),
        "final_acc_pct": float(np.mean(final_resolution_accs)),
        "ent_red_stage1_pct": float(np.mean(entropy_reductions_stage1)),
        "ent_red_stage2_pct": float(np.mean(entropy_reductions_stage2)),
    }


# ==============================================================================
# 5. Main Campaign Runner
# ==============================================================================

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 115)
    print(f"=== LONG-HORIZON MULTI-STAGE UNCERTAINTY DISCRIMINATION (DGX Spark {device}) ===")
    print("=" * 115)

    # 1. Permutation Invariance Regression Test
    print("\n--- 1. Whole-Slot Permutation Invariance Regression Check ---")
    for k in [4, 8, 16, 32]:
        m = VectorizedPseudoBrain(thoughtlets=k, width=120, heads=4, cycles=3).to(device)
        diff = test_permutation_invariance(m, device=device)
        print(f"  Pseudo-Brain K={k:2d} Slot Permutation Invariance Diff: {diff:.2e} ({'PASSED' if diff < 1e-5 else 'FAILED'})")

    # 2. Build Models
    print("\n--- 2. Model Initialization & Resource Accounting ---")
    print(f"{'Model Architecture':26s} | {'Width':5s} | {'Slots K':8s} | {'Parameters':14s} | {'Forward Latency':18s}")
    print("-" * 115)

    rgb = torch.randn(1, 3, 32, 32, device=device)
    ctrl = torch.zeros(1, 307, device=device)
    dt = torch.tensor([0.016667], device=device)
    n_iters = 500

    gru = ProposalGRUBaseline(hidden_dim=112, num_proposals=21).to(device)
    gru_p = sum(p.numel() for p in gru.parameters())
    for _ in range(50): _ = gru(rgb, ctrl, dt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters): _ = gru(rgb, ctrl, dt)
    torch.cuda.synchronize()
    gru_lat = (time.perf_counter() - t0) / n_iters * 1000.0
    print(f"{'Proposal-GRU Baseline':26s} | {'112':5s} | {'—':8s} | {gru_p:14,d} | {gru_lat:14.3f} ms")

    k_models = {}
    for k in [1, 4, 8, 16, 32]:
        m = VectorizedPseudoBrain(thoughtlets=k, width=120, heads=4, cycles=3).to(device)
        total_p = sum(p.numel() for p in m.parameters())
        err = (total_p - gru_p) / gru_p * 100.0
        for _ in range(50): _ = m(rgb, ctrl, dt)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters): _ = m(rgb, ctrl, dt)
        torch.cuda.synchronize()
        m_lat = (time.perf_counter() - t0) / n_iters * 1000.0
        print(f"{f'Vectorized PB (K={k})':26s} | {'120':5s} | {k:8d} | {total_p:10,d} ({err:+.1f}%) | {m_lat:14.3f} ms")
        k_models[k] = m
    print("=" * 115)

    # 3. Train all models on staged 4-joint-hypothesis curriculum
    print("\n--- 3. Training Models on Staged 4-Hypothesis Multi-Horizon Curriculum (500 steps) ---")
    t_train0 = time.perf_counter()
    train_multistage_stochastic(gru, device=device, training_steps=500)
    for k in [1, 4, 8, 16, 32]:
        train_multistage_stochastic(k_models[k], device=device, training_steps=500)
    print(f"  Training finished in {(time.perf_counter() - t_train0):.1f}s.")

    # 4. Long-Horizon Multi-Stage Evaluation across 5 Independent Seeds
    print("\n--- 4. Staged Multi-Uncertainty Discrimination (D1=15, D2=15 ticks, 5 Seeds: [100,200,300,400,500]) ---")
    print(f"{'Model Architecture':26s} | {'Gamble Avoid':14s} | {'Stage 1 Surv':14s} | {'Final Res Acc':14s} | {'Stage 1 H Red':14s} | {'Final H Red':12s} | {'Mean Return':20s}")
    print("-" * 115)

    seeds = [100, 200, 300, 400, 500]
    for name, mod in [("Proposal-GRU Baseline", gru)] + [(f"Pseudo-Brain K={k}", k_models[k]) for k in [1, 4, 8, 16, 32]]:
        res = run_long_horizon_benchmark(mod, device=device, seeds=seeds, delay_stage1=15, delay_stage2=15)
        print(f"{name:26s} | {res['gamble_avoid_pct']:12.1f}% | {res['stage1_survival_pct']:12.1f}% | {res['final_acc_pct']:12.1f}% | {res['ent_red_stage1_pct']:12.1f}% | {res['ent_red_stage2_pct']:10.1f}% | {res['mean_return']:+10.2f} +/- {res['std_return']:.2f}")
    print("=" * 115)


if __name__ == "__main__":
    main()
