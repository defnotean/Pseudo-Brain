"""Statistical Replication Battery: Grounded Planner vs Direct Actuator with Anti-Collapse Loss.

Evaluates 5 training seeds (42, 43, 44, 45, 46) across 20 held-out evaluation seeds (2001-2020)
under:
1. Direct Actuator Policy (Model's native unpooled WASD readout)
2. Grounded Lookahead Planner (Lookahead rollout using trained counterfactual foresight)
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time
from typing import Any
import torch

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from irene_brain.data.curriculum_dataset import CurriculumDataset, CurriculumDatasetConfig
from irene_brain.environments.maze_chase import MazeChaseEnv
from irene_brain.evaluation.closed_loop_play import (
    EXCLUSIVE_ARGMAX_WASD_V1,
    decode_closed_loop_control,
)
from irene_brain.evaluation.diagnostic_policies import ScriptedMazeChasePlannerPolicy
from irene_brain.evaluation.latent_lookahead_policy import LatentLookaheadPolicy
from irene_brain.evaluation.spatial_cognitive_diagnostics import (
    compute_model_param_digest,
    run_instrumented_diagnostic_episode,
)
from irene_brain.model.lookahead_planner import LatentLookaheadPlanner
from irene_brain.model.spec import ThoughtFieldConfig
from irene_brain.model.torch_model import IreneBrainModel
from irene_brain.training.batches import (
    BUTTON_TARGET_INDICES,
    CONTINUOUS_TARGET_INDICES,
    control_to_vector,
)
from irene_brain.training.cognitive_losses import CognitiveAuxiliaryLoss
from irene_brain.training.config import (
    DatasetConfig,
    DeterminismConfig,
    LoggingConfig,
    OptimizationConfig,
    PrecisionConfig,
    ResourceConfig,
    RunConfig,
    TrainingConfig,
)
from irene_brain.training.dagger_distill import DAggerConfig, DAggerDistiller
from irene_brain.training.objective import ThoughtFieldObjective, _rgb_tensor
from irene_brain.training.torch_system import TorchTrainingSystem
from irene_brain.types import HidKey

TRAIN_SEEDS = (42, 43, 44, 45, 46)
HELD_OUT_SEEDS = tuple(range(2001, 2021))


class DirectModelPolicy:
    """Direct policy evaluation using model's own actuator readout without lookahead planner."""
    identity = "model.direct_actuator.v1"
    uses_privileged_state = False

    def __init__(self, model: IreneBrainModel) -> None:
        self.model = model
        self.state = None
        self.last_control = None

    def reset(self, seed: int | None = None) -> None:
        self.state = self.model.initial_state(batch_size=1)
        self.last_control = torch.zeros((1, 307), dtype=torch.float32)

    def decide(self, observation: Any, elapsed_seconds: float = 1.0 / 60.0) -> tuple[Any, dict[str, Any], float]:
        device = next(self.model.parameters()).device
        rgb = _rgb_tensor((observation.rgb,), device=device, resolution=getattr(self.model, "input_resolution", None))
        dt = torch.tensor([elapsed_seconds], dtype=torch.float32, device=device)
        if self.state is None:
            self.reset()
        with torch.no_grad():
            out = self.model(rgb, self.last_control, dt, self.state)
            self.state = out.next_state
            self.last_control = out.action.control
            logits = out.action.button_logits[0].float().cpu()
            continuous_values = out.action.control[0].float().cpu()
            control, stats = decode_closed_loop_control(
                logits.tolist(),
                [float(continuous_values[idx]) for idx in CONTINUOUS_TARGET_INDICES],
                decode_kind=EXCLUSIVE_ARGMAX_WASD_V1,
            )
            val = float(out.value[0].item()) if hasattr(out, "value") and out.value is not None else 0.0
        return control, stats, val

    def act(self, observation: Any) -> Any:
        ctrl, _, _ = self.decide(observation)
        return ctrl


def train_seed_model(
    train_seed: int,
    *,
    dagger_iters: int = 2,
    updates_per_iter: int = 12,
) -> tuple[IreneBrainModel, float, float]:
    torch.manual_seed(train_seed)
    t0 = time.perf_counter()

    config = ThoughtFieldConfig(
        thoughtlets=4,
        registers_per_thoughtlet=2,
        core_width=32,
        attention_heads=4,
        sensor_tokens=4,
        belief_tokens=4,
        working_memory_tokens=4,
        goal_context_tokens=8,
        cognitive_cycles=3,
    )

    model = IreneBrainModel(
        config=config,
        enable_adaptive_cognition=True,
    )

    training_config = TrainingConfig(
        schema_version=1,
        run=RunConfig(
            name=f"grounded-seed{train_seed}",
            seed=train_seed,
            model_factory="irene_brain.model.torch_model:IreneBrainModel",
            max_optimizer_steps=10000,
        ),
        dataset=DatasetConfig(
            kind="curriculum",
            curriculum_scenario="mixed",
            train_sequences=64,
            validation_sequences=8,
            test_sequences=8,
            sequence_length=8,
            burn_in_steps=2,
            seed_offset=0,
            hazard_count=2,
            tick_period_ns=16_666_667,
            discount=0.99,
        ),
        optimization=OptimizationConfig(
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,
            weight_decay=1e-4,
            max_gradient_norm=1.0,
            warmup_steps=0,
        ),
        precision=PrecisionConfig(
            device="cpu",
            mode="float32",
            allow_tf32=False,
        ),
        determinism=DeterminismConfig(
            enabled=True,
            num_workers=0,
            compile_model=False,
        ),
        logging=LoggingConfig(
            log_every_steps=1,
            evaluate_every_steps=1,
            validation_batches=1,
            checkpoint_every_steps=10,
            keep_last_checkpoints=2,
        ),
        resources=ResourceConfig(
            allow_gpu=False,
            allow_capture=False,
            allow_hid_output=False,
            allow_background_threads=False,
            allow_network=False,
            allow_subprocess=False,
            write_artifacts=True,
            cpu_threads=1,
        ),
    )

    objective = ThoughtFieldObjective(model)
    system = TorchTrainingSystem(objective, training_config)

    dagger_config = DAggerConfig(
        iterations=dagger_iters,
        episodes_per_iteration=2,
        max_ticks_per_episode=60,
        initial_beta=0.8,
        beta_decay=0.85,
        sequence_length=8,
        burn_in_steps=2,
        batch_size=2,
        updates_per_iteration=updates_per_iter,
        ghost_count=2,
        ghost_period=2,
        extra_loops=8,
    )

    distiller = DAggerDistiller(
        config=dagger_config,
        student_model=model,
        training_system=system,
    )

    curriculum_cfg = CurriculumDatasetConfig(
        sequence_length=8,
        sequence_count=40,
        scenario="mixed",
    )
    curriculum_dataset = CurriculumDataset(curriculum_cfg)
    for i in range(min(20, len(curriculum_dataset))):
        distiller.buffer.add_sequence(curriculum_dataset[i])

    # Supervised Counterfactual Foresight + Anti-Collapse Orthogonality (Pillar 2)
    cog_loss_fn = CognitiveAuxiliaryLoss(
        future_weight=1.0,
        gate_surprise_weight=1.0,
        halting_weight=0.5,
        counterfactual_weight=1.0,
        topological_goal_weight=1.0,
        diversity_weight=0.1,  # Pillar 2: Dynamic Thoughtlet Anti-Collapse Loss
    )
    optimizer = system.optimizer if hasattr(system, "optimizer") else None

    final_loss = 0.0
    for it in range(dagger_iters):
        res = distiller.run_dagger_iteration(iteration=it)
        final_loss = res.mean_train_loss

        if optimizer is not None and len(distiller.buffer) > 0:
            model.train()
            for _ in range(updates_per_iter):
                batch = distiller.buffer.sample_batch(batch_size=2, burn_in_steps=2)
                for seq in batch.sequences:
                    if len(seq.transitions) < 6:
                        continue
                    t0_obs = seq.transitions[2]
                    device = next(model.parameters()).device
                    rgb_0 = _rgb_tensor(
                        (t0_obs.observation.rgb,),
                        device=device,
                        resolution=getattr(model, "input_resolution", None),
                    )
                    prev_c = torch.tensor([control_to_vector(t0_obs.observation.previous_control)], dtype=torch.float32, device=device)
                    dt = torch.tensor([0.016], dtype=torch.float32, device=device)
                    st = model.initial_state(batch_size=1)

                    out = model(rgb_0, prev_c, dt, st)

                    act_vec = control_to_vector(t0_obs.action_target)
                    wasd = [act_vec[int(k)] for k in (HidKey.W, HidKey.A, HidKey.S, HidKey.D)]
                    if any(w > 0.5 for w in wasd):
                        act_idx = 1 + wasd.index(max(wasd))
                    else:
                        act_idx = 0

                    haz_flags = []
                    for h_off in (1, 3, 5):
                        idx = min(len(seq.transitions) - 1, 2 + h_off)
                        fut_t = seq.transitions[idx]
                        is_haz = 1.0 if any(ev in ("collision", "caught") for ev in fut_t.event_targets) else 0.0
                        haz_flags.append([is_haz])

                    actual_collisions = torch.tensor([haz_flags], dtype=torch.float32)
                    executed_actions = torch.tensor([act_idx], dtype=torch.long)
                    future_disp_targets = torch.zeros((1, 3, 2), dtype=torch.float32)

                    vec_targets = torch.zeros((1, 2), dtype=torch.float32)
                    if act_idx == 1:
                        vec_targets[0, 1] = -1.0
                    elif act_idx == 2:
                        vec_targets[0, 0] = -1.0
                    elif act_idx == 3:
                        vec_targets[0, 1] = 1.0
                    elif act_idx == 4:
                        vec_targets[0, 0] = 1.0

                    is_hazard = any(ev in ("collision", "caught") for ev in t0_obs.event_targets) or (actual_collisions[0, 0, 0].item() > 0.5)
                    hazard_mask = torch.tensor([is_hazard], dtype=torch.bool)

                    cog_out = cog_loss_fn(
                        diagnostics=out.diagnostics,
                        future_disp_targets=future_disp_targets,
                        executed_actions=executed_actions,
                        actual_collisions=actual_collisions,
                        hazard_mask=hazard_mask,
                        topological_goal_targets=vec_targets,
                        topological_exit_targets=executed_actions,
                    )

                    if cog_out.total_loss.requires_grad:
                        optimizer.zero_grad()
                        cog_out.total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

    train_sec = time.perf_counter() - t0
    return model, float(final_loss), train_sec


def evaluate_policy_across_seeds(
    policy: Any,
    eval_seeds: tuple[int, ...] = HELD_OUT_SEEDS,
    eval_ticks: int = 120,
) -> tuple[float, int]:
    env = MazeChaseEnv()
    expert = ScriptedMazeChasePlannerPolicy()
    val_pellets = 0
    val_catches = 0
    count = len(eval_seeds)

    for s in eval_seeds:
        rep = run_instrumented_diagnostic_episode(
            policy=policy,
            env=env,
            seed=s,
            max_ticks=eval_ticks,
            expert_policy=expert,
        )
        val_pellets += rep.pellets_eaten
        val_catches += rep.ghost_collisions

    return (
        val_pellets / count,
        val_catches,
    )


def compute_thoughtlet_metrics(model: IreneBrainModel) -> tuple[float, float]:
    st = model.initial_state(batch_size=1)
    t_mat = st.thoughts[0, :, 0, :]  # [4, width]
    t_norm = torch.nn.functional.normalize(t_mat, p=2, dim=-1)
    sim_mat = torch.mm(t_norm, t_norm.t())
    mask = ~torch.eye(4, dtype=torch.bool)
    mean_sim = float(sim_mat[mask].mean().item())

    _, s, _ = torch.svd(t_mat.float())
    s_norm = s / (s.sum() + 1e-8)
    entropy = -(s_norm * torch.log(s_norm + 1e-8)).sum().item()
    eff_rank = float(math.exp(entropy))
    return round(mean_sim, 4), round(eff_rank, 3)


def stats(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)
    else:
        std = 0.0
    return round(mean, 2), round(std, 2)


def main() -> None:
    print("=" * 80)
    print("REPLICATION BATTERY: GROUNDED PLANNER VS DIRECT ACTUATOR (P2 + SUPERVISED P1)")
    print("Protocol: 5 Distinct Training Seeds x 20 Held-Out Evaluation Seeds (2001-2020)")
    print("=" * 80)

    direct_results = []
    grounded_planner_results = []

    for seed in TRAIN_SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        model, final_loss, train_sec = train_seed_model(seed)
        digest = compute_model_param_digest(model)
        sim, rank = compute_thoughtlet_metrics(model)
        print(f"  Training finished in {train_sec:.2f}s | Final Loss: {final_loss:.4f} | Digest: {digest}")
        print(f"  Thoughtlet Metrics: Similarity = {sim}, Effective Rank = {rank} / 4.0")

        # 1. Direct Actuator Evaluation
        direct_policy = DirectModelPolicy(model)
        d_pellets, d_catches = evaluate_policy_across_seeds(direct_policy)
        print(f"  [Direct Actuator]    Pellets: {d_pellets:.2f} | Total Catches: {d_catches} ({d_catches/20:.2f}/ep)")

        # 2. Grounded Lookahead Planner Evaluation
        planner = LatentLookaheadPlanner(
            model=model,
            horizon=3,
            gamma=0.95,
            hazard_weight=5.0,
            hazard_prune_threshold=0.50,
            cognitive_cycles_per_step=model.config.cognitive_cycles,
        )
        grounded_policy = LatentLookaheadPolicy(model=model, planner=planner)
        g_pellets, g_catches = evaluate_policy_across_seeds(grounded_policy)
        print(f"  [Grounded Planner]   Pellets: {g_pellets:.2f} | Total Catches: {g_catches} ({g_catches/20:.2f}/ep)")

        direct_results.append({
            "seed": seed,
            "pellets": d_pellets,
            "catches": d_catches,
            "loss": final_loss,
            "digest": digest,
            "sim": sim,
            "rank": rank,
        })
        grounded_planner_results.append({
            "seed": seed,
            "pellets": g_pellets,
            "catches": g_catches,
            "loss": final_loss,
            "digest": digest,
            "sim": sim,
            "rank": rank,
        })

    # Summary Statistics
    d_p_mean, d_p_std = stats([r["pellets"] for r in direct_results])
    d_c_mean, d_c_std = stats([r["catches"] for r in direct_results])
    g_p_mean, g_p_std = stats([r["pellets"] for r in grounded_planner_results])
    g_c_mean, g_c_std = stats([r["catches"] for r in grounded_planner_results])

    print("\n" + "=" * 80)
    print("FINAL 5-SEED REPLICATION SUMMARY")
    print("=" * 80)
    print(f"Direct Actuator:    Pellets = {d_p_mean} +/- {d_p_std} | Catches = {d_c_mean} +/- {d_c_std}")
    print(f"Grounded Planner:   Pellets = {g_p_mean} +/- {g_p_std} | Catches = {g_c_mean} +/- {g_c_std}")
    print("=" * 80)

    summary_data = {
        "direct_actuator": {
            "mean_pellets": d_p_mean,
            "std_pellets": d_p_std,
            "mean_catches": d_c_mean,
            "std_catches": d_c_std,
            "seeds": direct_results,
        },
        "grounded_planner": {
            "mean_pellets": g_p_mean,
            "std_pellets": g_p_std,
            "mean_catches": g_c_mean,
            "std_catches": g_c_std,
            "seeds": grounded_planner_results,
        },
    }
    out_path = pathlib.Path("docs/runs/2026-08-19-grounded-replication-battery.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print("Saved results to docs/runs/2026-08-19-grounded-replication-battery.json")


if __name__ == "__main__":
    main()
