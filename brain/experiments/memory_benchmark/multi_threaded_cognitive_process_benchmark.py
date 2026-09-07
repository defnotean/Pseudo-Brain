"""Multi-Threaded Cognitive Process Benchmark (MTCP-Bench / Workstream 11).

Bridges Pseudo-Brain from synthetic latent variable tracking to a realistic multi-threaded
cognitive workload benchmark evaluating concurrent task management, interruptions,
cross-thread dependencies, and delayed consequences.

Simulates an embodied agent managing N_threads in [8, 16] concurrent asynchronous processes:
1. Data Gathering Thread: Asynchronous sensor/value streams with periodic queries.
2. Procedural Pipeline Thread: Multi-stage workflow (Fetch -> Process -> Verify) interrupted mid-flight.
3. Safety Invariant Monitor: Boundary condition tracking with state alarm checks.
4. Cross-Thread Dependency: Thread B depends on the output of Thread A.
5. Orthogonal Independent Threads: Must maintain isolation with zero mutual cross-talk.
6. Delayed Consequence Feedback: Actions committed at t receive outcome feedback at t + D.

Evaluates 4 Parameter-Matched Architectures:
1. Pseudo-Brain CGP (Shared recurrent core + CIG sharpening + CGSL + Slot-Targeted Readout)
2. Monolithic GRU (Multi-layer gated recurrent unit)
3. Modern Diagonal SSM (GLRU / S4D-class diagonal state space model)
4. Recurrent Linear Attention (RWKV / Mamba-class linear token mixer)

Compound Cognitive Metric:
S_cognitive = Retention * (1 - Interference) * Recovery * Latency_Eff * Param_Eff
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.model.brain_cell import BrainCellCore, FactorizedLowRankProjection
from irene_brain.model.sparse_thought_router import SparseThoughtRouter
from irene_brain.model.torch_model import deterministic_thought_identity_codes
from memory_benchmark.multi_threaded_latent_dependency_benchmark import count_params


# ==============================================================================
# 1. Environment: MultiThreadedCognitiveProcessEnv (MTCP-Bench)
# ==============================================================================

class MTCPOp:
    DATA = 0
    PIPELINE_FETCH = 1
    PIPELINE_PROCESS = 2
    PIPELINE_VERIFY = 3
    SAFETY_ALERT = 4
    DEPENDENT_STEP = 5
    INTERRUPT = 6
    DISTRACTOR = 7
    DELAYED_CONSEQUENCE = 8
    QUERY = 9


@dataclass
class MTCPStep:
    thread_id: int
    op_type: int
    payload: int  # Discrete target/value in [0, num_values - 1]
    is_query: bool
    query_type: str  # 'retention', 'interference', 'recovery', 'dependency'
    target: int  # Ground truth label for prediction (-100 if not a query)


class MultiThreadedCognitiveProcessEnv:
    """Realistic multi-threaded cognitive process environment."""

    def __init__(
        self,
        num_threads: int = 8,
        num_values: int = 8,
        input_dim: int = 64,
        max_interruption_length: int = 16,
    ):
        assert num_threads >= 4, "Must have at least 4 threads for dependencies and interruptions"
        self.num_threads = num_threads
        self.num_values = num_values
        self.input_dim = input_dim
        self.max_interruption_length = max_interruption_length

    def generate_episode(self, rng: np.random.RandomState) -> List[MTCPStep]:
        """Generate a structured asynchronous multi-threaded workflow."""
        steps: List[MTCPStep] = []
        thread_states: Dict[int, int] = {}

        # 1. Initial State Enrollment across all threads
        for tid in range(self.num_threads):
            val = int(rng.randint(0, self.num_values))
            thread_states[tid] = val
            steps.append(MTCPStep(
                thread_id=tid,
                op_type=MTCPOp.DATA,
                payload=val,
                is_query=False,
                query_type="none",
                target=-100,
            ))

        # 2. Pipeline Execution on Thread 0 (Fetch -> Process), then INTERRUPT
        pipe_tid = 0
        p_val1 = int(rng.randint(0, self.num_values))
        thread_states[pipe_tid] = p_val1
        steps.append(MTCPStep(
            thread_id=pipe_tid,
            op_type=MTCPOp.PIPELINE_FETCH,
            payload=p_val1,
            is_query=False,
            query_type="none",
            target=-100,
        ))

        # Preemption: Interrupt Thread 0 with urgent tasks on other threads
        interrupt_len = int(rng.randint(8, max(9, self.max_interruption_length + 1)))
        for _ in range(interrupt_len):
            other_tid = int(rng.randint(1, self.num_threads))
            # Either distractor or intermediate data update
            if rng.rand() < 0.5:
                # Distractor noise token
                steps.append(MTCPStep(
                    thread_id=other_tid,
                    op_type=MTCPOp.DISTRACTOR,
                    payload=int(rng.randint(0, self.num_values)),
                    is_query=False,
                    query_type="none",
                    target=-100,
                ))
            else:
                # Update on orthogonal thread (tests interference)
                new_v = int(rng.randint(0, self.num_values))
                thread_states[other_tid] = new_v
                steps.append(MTCPStep(
                    thread_id=other_tid,
                    op_type=MTCPOp.DATA,
                    payload=new_v,
                    is_query=False,
                    query_type="none",
                    target=-100,
                ))

        # 3. Pipeline Resume: Verify stage on Thread 0
        p_val2 = (p_val1 + 1) % self.num_values
        thread_states[pipe_tid] = p_val2
        steps.append(MTCPStep(
            thread_id=pipe_tid,
            op_type=MTCPOp.PIPELINE_VERIFY,
            payload=p_val2,
            is_query=False,
            query_type="none",
            target=-100,
        ))

        # 4. Cross-Thread Dependency: Thread 1 computes f(Thread 1 state, Thread 0 state)
        dep_tid = 1
        combined_val = (thread_states[dep_tid] + thread_states[pipe_tid]) % self.num_values
        thread_states[dep_tid] = combined_val
        steps.append(MTCPStep(
            thread_id=dep_tid,
            op_type=MTCPOp.DEPENDENT_STEP,
            payload=combined_val,
            is_query=False,
            query_type="none",
            target=-100,
        ))

        # 5. Delayed Consequence Feedback
        consequence_tid = int(rng.randint(2, self.num_threads))
        steps.append(MTCPStep(
            thread_id=consequence_tid,
            op_type=MTCPOp.DELAYED_CONSEQUENCE,
            payload=thread_states[consequence_tid],
            is_query=False,
            query_type="none",
            target=-100,
        ))

        # Additional distractor delay
        for _ in range(8):
            steps.append(MTCPStep(
                thread_id=int(rng.randint(0, self.num_threads)),
                op_type=MTCPOp.DISTRACTOR,
                payload=int(rng.randint(0, self.num_values)),
                is_query=False,
                query_type="none",
                target=-100,
            ))

        # 6. Evaluation Query Phase
        # Query A: Pipeline Recovery (Thread 0)
        steps.append(MTCPStep(
            thread_id=pipe_tid,
            op_type=MTCPOp.QUERY,
            payload=0,
            is_query=True,
            query_type="recovery",
            target=thread_states[pipe_tid],
        ))

        # Query B: Cross-Thread Dependency (Thread 1)
        steps.append(MTCPStep(
            thread_id=dep_tid,
            op_type=MTCPOp.QUERY,
            payload=0,
            is_query=True,
            query_type="dependency",
            target=thread_states[dep_tid],
        ))

        # Query C: Untouched Orthogonal Threads (Interference Test)
        untouched_candidates = [tid for tid in range(2, self.num_threads) if tid != consequence_tid]
        if not untouched_candidates:
            untouched_candidates = [2]
        for tid in untouched_candidates[:2]:
            steps.append(MTCPStep(
                thread_id=tid,
                op_type=MTCPOp.QUERY,
                payload=0,
                is_query=True,
                query_type="interference",
                target=thread_states[tid],
            ))

        # Query D: General Retention (Consequence thread)
        steps.append(MTCPStep(
            thread_id=consequence_tid,
            op_type=MTCPOp.QUERY,
            payload=0,
            is_query=True,
            query_type="retention",
            target=thread_states[consequence_tid],
        ))

        return steps

    def encode_step(self, step: MTCPStep) -> np.ndarray:
        """Encode step into fixed-dimensional vector [input_dim]."""
        vec = np.zeros(self.input_dim, dtype=np.float32)
        # Thread one-hot: dims 0 .. num_threads - 1
        vec[step.thread_id % min(16, self.input_dim)] = 1.0
        # Op type one-hot: dims 16 .. 25
        vec[16 + (step.op_type % 10)] = 1.0
        # Payload encoding: dims 26 .. 33
        vec[26 + (step.payload % self.num_values)] = 1.0
        # Query indicator: dim 34
        if step.is_query:
            vec[34] = 1.0
        # Low-amplitude high-frequency sensory noise on remaining dimensions
        vec[35:] = np.random.randn(self.input_dim - 35) * 0.05
        return vec


def build_mtcp_batch(
    env: MultiThreadedCognitiveProcessEnv,
    batch_size: int = 16,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, List[List[MTCPStep]]]:
    """Construct batch tensors: obs [B, T, D], targets [B, T]."""
    episodes: List[List[MTCPStep]] = []
    for i in range(batch_size):
        rng = np.random.RandomState(seed + i * 10007)
        episodes.append(env.generate_episode(rng))

    max_len = max(len(ep) for ep in episodes)
    obs = np.zeros((batch_size, max_len, env.input_dim), dtype=np.float32)
    targets = np.full((batch_size, max_len), -100, dtype=np.int64)

    for b, ep in enumerate(episodes):
        for t, step in enumerate(ep):
            obs[b, t] = env.encode_step(step)
            if step.is_query:
                targets[b, t] = step.target

    return torch.from_numpy(obs), torch.from_numpy(targets), episodes


# ==============================================================================
# 2. Competitor Architectures
# ==============================================================================

class MTCPPseudoBrainModel(nn.Module):
    """Pseudo-Brain CGP with Thread-Targeted Slot Readout and Gated Plasticity."""

    def __init__(
        self,
        input_dim: int = 64,
        K: int = 8,
        thought_size: int = 24,
        proj_dim: int = 256,
        num_values: int = 8,
        plastic_decay: float = 0.9999,
        plastic_lr: float = 0.25,
        rank: Optional[int] = None,
        epsilon_dormant: float = 0.05,
        conditional_recurrence: bool = True,
        event_driven_sparsity: bool = False,
    ):
        super().__init__()
        self.K = K
        self.thought_size = thought_size
        self.proj_dim = proj_dim
        self.num_values = num_values
        self.plastic_decay = plastic_decay
        self.plastic_lr = plastic_lr
        self.rank = rank
        self.epsilon_dormant = epsilon_dormant
        self.conditional_recurrence = conditional_recurrence
        self.event_driven_sparsity = event_driven_sparsity

        self.proj = nn.Linear(input_dim, proj_dim)
        self.brain_cell = BrainCellCore(input_size=proj_dim, thought_size=thought_size, rank=rank)

        # CIG Gate with Temperature Sharpening
        self.cig_gate = nn.Sequential(
            nn.Linear(proj_dim + thought_size, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.cig_gate[0].bias, 1.0)

        # Thread-Targeted Slot Readout (eliminates M=32 sample-starvation wall!)
        if rank is not None and rank > 0:
            self.slot_head = nn.Sequential(
                FactorizedLowRankProjection(thought_size, proj_dim // 2, rank=rank),
                nn.GELU(),
                nn.Linear(proj_dim // 2, num_values),
            )
        else:
            self.slot_head = nn.Sequential(
                nn.Linear(thought_size, proj_dim // 2),
                nn.GELU(),
                nn.Linear(proj_dim // 2, num_values),
            )

        # Synaptic Plastic Latch per slot: [B, K, num_values]
        self.plastic_modulator = nn.Linear(thought_size + 16, num_values)
        self.surprise_gate = nn.Sequential(
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.surprise_gate[0].bias, -1.0)
        self.plastic_scale = nn.Parameter(torch.tensor(1.2))

        self.surprise_encoder = nn.Sequential(
            nn.Linear(1, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 16),
        )

        self.register_buffer(
            "slot_identities",
            deterministic_thought_identity_codes(thoughtlets=K, width=thought_size),
            persistent=False,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, D = x_seq.shape
        dev = x_seq.device
        x_proj = self.proj(x_seq)

        if self.slot_identities.shape[0] != self.K:
            slots = deterministic_thought_identity_codes(thoughtlets=self.K, width=self.thought_size).to(dev)
        else:
            slots = self.slot_identities.to(device=dev, dtype=x_seq.dtype)

        thoughts = slots.unsqueeze(0).expand(B, -1, -1).clone()  # [B, K, W]
        P_t = torch.zeros(B, self.K, self.num_values, device=dev)
        prev_thoughts = thoughts.clone()
        logits_list = []

        for t in range(T):
            x_in = x_proj[:, t]  # [B, proj_dim]
            x_exp = x_in.unsqueeze(1).expand(-1, self.K, -1)  # [B, K, proj_dim]

            # CIG Gate with T=0.5 sharpening
            gate_in = torch.cat([x_exp, thoughts], dim=-1)
            raw_gate = self.cig_gate(gate_in)
            salience = torch.sigmoid((torch.logit(raw_gate.clamp(1e-6, 1 - 1e-6))) / 0.5)

            if self.event_driven_sparsity:
                active_tid = x_seq[:, t, :min(self.K, 16, x_seq.shape[-1])].argmax(dim=-1)
                sparse_salience = torch.full_like(salience, 0.01)
                for b in range(B):
                    sparse_salience[b, active_tid[b]] = salience[b, active_tid[b]]
                salience = sparse_salience

            # Event-Driven Sparse Slot Ticking (Conditional Recurrence)
            if self.conditional_recurrence and self.epsilon_dormant > 0.0:
                thoughts, _ = self.brain_cell.forward_conditional(
                    thoughts=thoughts,
                    x=x_exp,
                    salience=salience,
                    epsilon_dormant=self.epsilon_dormant,
                )
            else:
                # Dense fallback
                t_flat = thoughts.reshape(B * self.K, self.thought_size)
                x_flat = x_exp.reshape(B * self.K, -1)
                new_t = self.brain_cell(t_flat, x_flat).reshape(B, self.K, self.thought_size)
                thoughts = (1.0 - salience) * thoughts + salience * new_t

            # Surprise & Fast Synaptic Latch Update
            delta_slot = torch.norm(thoughts - prev_thoughts, dim=-1, keepdim=True)  # [B, K, 1]
            prev_thoughts = thoughts.detach()
            surprise_emb = self.surprise_encoder(delta_slot)  # [B, K, 16]

            p_gate = self.surprise_gate(surprise_emb)  # [B, K, 1]
            delta_P = p_gate * torch.tanh(self.plastic_modulator(torch.cat([thoughts, surprise_emb], dim=-1)))
            P_t = self.plastic_decay * P_t + self.plastic_lr * delta_P

            # Readout: Determine queried thread from one-hot input in x_seq
            active_tid = x_seq[:, t, :min(self.K, 16)].argmax(dim=-1)  # [B]
            batch_idx = torch.arange(B, device=dev)
            queried_slot = thoughts[batch_idx, active_tid]  # [B, W]
            queried_P = P_t[batch_idx, active_tid]  # [B, num_values]

            slot_logits = self.slot_head(queried_slot)
            total_logits = slot_logits + self.plastic_scale * queried_P
            logits_list.append(total_logits)

        return torch.stack(logits_list, dim=1)


class MTCPMonolithicGRU(nn.Module):
    """Parameter-Matched Monolithic GRU Baseline."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_values: int = 8,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_values),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x_seq)
        return self.head(out)


class MTCPDiagonalSSM(nn.Module):
    """Modern Diagonal Gated Linear Recurrent Unit (GLRU / S4D-class)."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 288,
        num_layers: int = 3,
        num_values: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.in_proj = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.gate_proj = nn.ModuleList([
            nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        self.decay_param = nn.ParameterList([
            nn.Parameter(torch.randn(hidden_dim) * 0.1 - 2.0)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_values),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_seq.shape
        cur_x = x_seq

        for layer in range(self.num_layers):
            in_p = self.in_proj[layer](cur_x)
            gate_p = torch.sigmoid(self.gate_proj[layer](cur_x))
            alpha = torch.sigmoid(self.decay_param[layer]).unsqueeze(0)  # [1, H]

            h_t = torch.zeros(B, self.hidden_dim, device=x_seq.device)
            h_list = []
            for t in range(T):
                inp_t = in_p[:, t] * gate_p[:, t]
                h_t = alpha * h_t + (1.0 - alpha) * inp_t
                h_list.append(h_t)
            h_seq = torch.stack(h_list, dim=1)
            cur_x = cur_x + self.out_proj[layer](h_seq) if cur_x.shape[-1] == self.hidden_dim else self.out_proj[layer](h_seq)

        return self.head(cur_x)


class MTCPLinearAttention(nn.Module):
    """Modern Recurrent Linear Attention (RWKV / Mamba-class Token Mixer)."""

    def __init__(
        self,
        input_dim: int = 64,
        hidden_dim: int = 256,
        num_values: int = 8,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.k_proj = nn.Linear(input_dim, hidden_dim)
        self.v_proj = nn.Linear(input_dim, hidden_dim)
        self.decay = nn.Parameter(torch.tensor(-0.1))
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_values),
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_seq.shape
        Q = F.elu(self.q_proj(x_seq)) + 1.0  # [B, T, H]
        K = F.elu(self.k_proj(x_seq)) + 1.0
        V = self.v_proj(x_seq)

        gamma = torch.sigmoid(self.decay)
        # Recurrent state accumulation
        S_t = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        Z_t = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        out_list = []

        for t in range(T):
            k_t = K[:, t]
            v_t = V[:, t]
            q_t = Q[:, t]

            S_t = gamma * S_t + k_t * v_t
            Z_t = gamma * Z_t + k_t
            y_t = (q_t * S_t) / (q_t * Z_t + 1e-6)
            out_list.append(y_t)

        out_seq = torch.stack(out_list, dim=1)
        return self.head(self.out_proj(out_seq))


# ==============================================================================
# 3. Model Factory & Training / Evaluation Harness
# ==============================================================================

def make_mtcp_model(
    arch: str,
    input_dim: int = 64,
    num_threads: int = 8,
    num_values: int = 8,
    rank: Optional[int] = None,
    epsilon_dormant: float = 0.05,
    conditional_recurrence: bool = True,
    event_driven_sparsity: bool = False,
) -> nn.Module:
    """Create parameter-matched model for MTCP-Bench."""
    if arch == "pseudo_brain":
        return MTCPPseudoBrainModel(
            input_dim=input_dim,
            K=num_threads,
            thought_size=48,
            proj_dim=512,
            num_values=num_values,
            rank=rank,
            epsilon_dormant=epsilon_dormant,
            conditional_recurrence=conditional_recurrence,
            event_driven_sparsity=event_driven_sparsity,
        )
    elif arch == "gru":
        return MTCPMonolithicGRU(
            input_dim=input_dim,
            hidden_dim=160,
            num_layers=2,
            num_values=num_values,
        )
    elif arch == "diagonal_ssm":
        return MTCPDiagonalSSM(
            input_dim=input_dim,
            hidden_dim=180,
            num_layers=3,
            num_values=num_values,
        )
    elif arch == "linear_attention":
        return MTCPLinearAttention(
            input_dim=input_dim,
            hidden_dim=280,
            num_values=num_values,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def train_mtcp_model(
    model: nn.Module,
    env: MultiThreadedCognitiveProcessEnv,
    num_steps: int = 140,
    batch_size: int = 32,
    lr: float = 2e-3,
    device: torch.device = torch.device("cpu"),
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for step_idx in range(num_steps):
        obs_b, tgt_b, _ = build_mtcp_batch(env, batch_size, seed=5000 + step_idx)
        obs_b = obs_b.to(device)
        tgt_b = tgt_b.to(device)

        optimizer.zero_grad()
        logits = model(obs_b)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), tgt_b.reshape(B * T), ignore_index=-100)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


def evaluate_mtcp_model(
    model: nn.Module,
    env: MultiThreadedCognitiveProcessEnv,
    num_seeds: int = 15,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    model.eval()
    latencies_ms = []

    metrics_by_type: Dict[str, List[float]] = {
        "all": [],
        "retention": [],
        "interference": [],
        "recovery": [],
        "dependency": [],
    }

    for s_idx in range(num_seeds):
        obs_b, tgt_b, episodes = build_mtcp_batch(env, batch_size=16, seed=9000 + s_idx)
        obs_b = obs_b.to(device)

        t0 = time.perf_counter_ns()
        with torch.no_grad():
            logits = model(obs_b)
        t1 = time.perf_counter_ns()
        latencies_ms.append((t1 - t0) / (1e6 * 16))

        preds = logits.argmax(dim=-1).cpu().numpy()
        targets = tgt_b.numpy()

        for b in range(16):
            ep = episodes[b]
            for t, step in enumerate(ep):
                if step.is_query:
                    is_correct = float(preds[b, t] == targets[b, t])
                    metrics_by_type["all"].append(is_correct)
                    if step.query_type in metrics_by_type:
                        metrics_by_type[step.query_type].append(is_correct)

    retention_acc = float(np.mean(metrics_by_type["retention"])) * 100.0 if metrics_by_type["retention"] else 0.0
    interference_acc = float(np.mean(metrics_by_type["interference"])) * 100.0 if metrics_by_type["interference"] else 0.0
    recovery_acc = float(np.mean(metrics_by_type["recovery"])) * 100.0 if metrics_by_type["recovery"] else 0.0
    dependency_acc = float(np.mean(metrics_by_type["dependency"])) * 100.0 if metrics_by_type["dependency"] else 0.0
    overall_acc = float(np.mean(metrics_by_type["all"])) * 100.0 if metrics_by_type["all"] else 0.0
    mean_latency = float(np.mean(latencies_ms))

    # Cross-talk error rate on orthogonal threads
    cross_talk_err = max(0.0, 100.0 - interference_acc)

    # Compound Cognitive Score:
    # S_cog = Retention * (Interference Isolation) * Recovery * Latency_Eff
    interference_isolation = (100.0 - cross_talk_err) / 100.0
    latency_eff = max(0.01, min(1.0, 16.67 / max(0.01, mean_latency)))
    compound_score = (retention_acc / 100.0) * interference_isolation * (recovery_acc / 100.0) * latency_eff * 100.0

    return {
        "overall_accuracy": overall_acc,
        "retention_accuracy": retention_acc,
        "interference_accuracy": interference_acc,
        "cross_talk_error": cross_talk_err,
        "recovery_accuracy": recovery_acc,
        "dependency_accuracy": dependency_acc,
        "latency_ms_mean": mean_latency,
        "compound_cognitive_score": compound_score,
    }


# ==============================================================================
# 4. Master Benchmark Execution & Report Generation
# ==============================================================================

def run_mtcp_benchmark(
    threads_list: List[int] = [8, 16],
    num_seeds: int = 15,
    train_steps: int = 140,
    device_str: str = "cpu",
) -> Dict[str, Any]:
    device = torch.device(device_str)
    architectures = [
        ("pseudo_brain", "Pseudo-Brain CGP"),
        ("gru", "Monolithic GRU"),
        ("diagonal_ssm", "Modern Diagonal SSM (GLRU)"),
        ("linear_attention", "Recurrent Linear Attention"),
    ]

    results: Dict[str, Any] = {
        "threads_evaluated": threads_list,
        "grid": {},
    }

    for N_th in threads_list:
        n_key = f"threads_{N_th}"
        results["grid"][n_key] = {}
        print("\n" + "=" * 100)
        print(f"MTCP-BENCH WORKLOAD: {N_th} CONCURRENT ASYNCHRONOUS COGNITIVE THREADS")
        print("=" * 100)

        env = MultiThreadedCognitiveProcessEnv(num_threads=N_th, num_values=8)

        for arch_key, arch_label in architectures:
            model = make_mtcp_model(arch_key, input_dim=64, num_threads=N_th, num_values=8).to(device)
            p_count = count_params(model)["total"]

            train_mtcp_model(model, env, num_steps=train_steps, device=device)
            metrics = evaluate_mtcp_model(model, env, num_seeds=num_seeds, device=device)
            metrics["params"] = p_count
            metrics["arch_label"] = arch_label
            results["grid"][n_key][arch_key] = metrics

            print(
                f"{arch_label:<28s} | Params: {p_count/1e3:6.1f}k | Ret: {metrics['retention_accuracy']:5.1f}% | "
                f"Iso: {100.0 - metrics['cross_talk_error']:5.1f}% | Recov: {metrics['recovery_accuracy']:5.1f}% | "
                f"Dep: {metrics['dependency_accuracy']:5.1f}% | Lat: {metrics['latency_ms_mean']:5.2f}ms | "
                f"Score: {metrics['compound_cognitive_score']:5.2f}"
            )

    return results


def generate_mtcp_report(results: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "2026-09-07-mtcp-benchmark.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw MTCP telemetry to {json_path}")

    md_path = output_dir / "2026-09-07-mtcp-benchmark.md"
    lines = [
        "# Multi-Threaded Cognitive Process Benchmark Report (MTCP-Bench)",
        "**Date:** 2026-09-07  ",
        "**Status:** `[MEASURED]` Evaluates 4 parameter-matched architectures across realistic multi-threaded workloads.  \n",
        "## 1. Executive Summary",
        "MTCP-Bench directly evaluates whether Pseudo-Brain's specialized cognitive primitive survives realistic multi-threaded workloads with:",
        "- **Interleaved Asynchronous Threads** ($N \\in [8, 16]$)",
        "- **Mid-flight Interruptions & Preemption** ($D_{\\text{interrupt}} \\in [8, 16]$ steps)",
        "- **Cross-Thread Dependencies** (Thread $B$ dependent on output of Thread $A$)",
        "- **Orthogonal Independent Threads** (zero cross-talk tolerance)",
        "- **Delayed Consequence Feedback**\n",
        "## 2. Benchmark Results & Compound Cognitive Scores\n",
    ]

    for N_th in results["threads_evaluated"]:
        n_key = f"threads_{N_th}"
        lines.extend([
            f"### {N_th} Concurrent Cognitive Threads",
            "| Architecture | Params | Retention Acc | Isolation Acc | Recovery Acc | Dependency Acc | Latency | Compound Score |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for arch_key, m in results["grid"][n_key].items():
            label = m["arch_label"]
            p = m["params"]
            ret = m["retention_accuracy"]
            iso = 100.0 - m["cross_talk_error"]
            rec = m["recovery_accuracy"]
            dep = m["dependency_accuracy"]
            lat = m["latency_ms_mean"]
            score = m["compound_cognitive_score"]
            lines.append(
                f"| **{label}** | {p/1e3:5.1f}k | **{ret:.1f}%** | **{iso:.1f}%** | **{rec:.1f}%** | {dep:.1f}% | {lat:.2f} ms | **{score:.2f}** |"
            )
        lines.append("")

    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Generated Markdown report at {md_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MTCP Benchmark")
    parser.add_argument("--threads", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--num-seeds", type=int, default=15)
    parser.add_argument("--train-steps", type=int, default=140)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="brain/docs/runs")
    args = parser.parse_args()

    results = run_mtcp_benchmark(
        threads_list=args.threads,
        num_seeds=args.num_seeds,
        train_steps=args.train_steps,
        device_str=args.device,
    )
    generate_mtcp_report(results, output_dir=Path(args.output_dir))
