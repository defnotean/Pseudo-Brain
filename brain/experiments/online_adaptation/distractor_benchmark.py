"""Distractor Delay Stress-Test Benchmark for Pseudo-Brain Online Adaptation.

Evaluates whether Fast Episodic Plasticity (P_t) outperforms standard recurrent memory (GRU)
under distractor delays.

Standard RNNs (like GRU) can hold a bit in hidden state across a short, clean corridor.
However, under distractor delays or noisy intervening steps, recurrent memory decays exponentially.
In contrast, surprise-gated synaptic weights P_t remain frozen during distractor steps because
there is no surprise.

Evaluates:
- Standard GRU (gru_seed_42.pt)
- Thoughtlet Baseline (thoughtlet_seed_42.pt)
- Plastic Thoughtlet (plastic_thoughtlet_seed_42.pt)
Across delay horizons D in [0, 10, 25, 50].

Outputs:
- runs/online_adaptation_colab_v2/distractor_results.json
- Markdown comparison table of Trial 12 adaptation accuracy vs D.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from irene_brain.types import GenericControl, HidKey, Observation, RgbFrame, StepOutcome
from online_adaptation.hidden_rule_env import HiddenRuleEnv, Rule
from online_adaptation.models import (
    PredictiveGRUModel,
    PredictiveThoughtletModel,
    N_FRAMES,
    ACTION_CLASSES,
)

CTRL_MAP = {
    0: GenericControl(),
    1: GenericControl(keys_down=(int(HidKey.W),)),
    2: GenericControl(keys_down=(int(HidKey.A),)),
    3: GenericControl(keys_down=(int(HidKey.S),)),
    4: GenericControl(keys_down=(int(HidKey.D),)),
}


class DistractorHiddenRuleEnv(HiddenRuleEnv):
    """HiddenRuleEnv extending inter-trial transitions with D distractor delay ticks.

    During distractor delay ticks:
    - The player dwells or walks through distractor visual noise.
    - Visual observations change on every tick (dynamic sensory noise).
    - No goal or consequence feedback is reached (reward = 0.0, no surprise).
    """

    def __init__(
        self,
        *,
        rule_schedule: Optional[List[Tuple[Rule, int]]] = None,
        distractor_delay: int = 0,
        distractor_noise_level: float = 35.0,
        max_ticks_per_trial: int = 40,
        feedback_dwell_ticks: int = 2,
    ):
        super().__init__(
            rule_schedule=rule_schedule,
            max_ticks_per_trial=max_ticks_per_trial,
            feedback_dwell_ticks=feedback_dwell_ticks,
        )
        self.distractor_delay = distractor_delay
        self.distractor_noise_level = distractor_noise_level
        self._distractor_timer = 0
        self._rng = np.random.default_rng(0)

    def reset(self, seed: int = 0) -> Observation:
        self._rng = np.random.default_rng(seed)
        self._distractor_timer = 0
        return super().reset(seed=seed)

    def step(self, control: GenericControl) -> StepOutcome:
        applied_mask = self._mask_from_control(control)
        self._prev_key_mask = applied_mask
        applied_control = self._control_from_mask(applied_mask)

        # 1. Feedback dwell phase (consequence display after hitting door)
        if self._feedback_mode is not None:
            self._tick += 1
            self._trial_tick += 1
            self._feedback_timer -= 1
            if self._feedback_timer <= 0:
                self._feedback_mode = None
                if self.distractor_delay > 0:
                    # Enter distractor delay phase between trials
                    self._distractor_timer = self.distractor_delay
                    self._player_x, self._player_y = self.start_pos
                else:
                    self._reset_player_for_next_trial()
                    if self._trial_idx >= self.total_trials:
                        return StepOutcome(
                            observation=self.current_observation,
                            requested_control=control,
                            applied_control=applied_control,
                            reward=0.0,
                            events=(),
                            terminated=True,
                            truncated=False,
                        )

            return StepOutcome(
                observation=self.current_observation,
                requested_control=control,
                applied_control=applied_control,
                reward=0.0,
                events=(),
                terminated=False,
                truncated=False,
            )

        # 2. Inter-trial distractor delay phase
        if self._distractor_timer > 0:
            self._tick += 1
            self._distractor_timer -= 1
            self._player_x, self._player_y = self.start_pos

            is_delay_finished = (self._distractor_timer <= 0)
            if is_delay_finished:
                self._reset_player_for_next_trial()
                if self._trial_idx >= self.total_trials:
                    return StepOutcome(
                        observation=self.current_observation,
                        requested_control=control,
                        applied_control=applied_control,
                        reward=0.0,
                        events=("distractor_delay",),
                        terminated=True,
                        truncated=False,
                    )

            obs = self._render_distractor_observation()
            return StepOutcome(
                observation=obs,
                requested_control=control,
                applied_control=applied_control,
                reward=0.0,
                events=("distractor_delay",),
                terminated=False,
                truncated=False,
            )

        # 3. Standard active navigation step
        return super().step(control)

    def _render_distractor_observation(self) -> Observation:
        """Render corridor frame with dynamic visual distractor noise."""
        normal_frame = super()._render()
        raw = np.frombuffer(normal_frame.pixels, dtype=np.uint8).reshape(16, 16, 3).copy()
        if self.distractor_noise_level > 0:
            noise = self._rng.integers(
                -int(self.distractor_noise_level),
                int(self.distractor_noise_level) + 1,
                size=(16, 16, 3),
                dtype=np.int16,
            )
            raw = np.clip(raw.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        control = self._control_from_mask(self._prev_key_mask)
        return Observation(
            frame_id=self._tick,
            capture_tick=self._tick,
            elapsed_ns=self._tick * self.DEFAULT_TICK_PERIOD_NS,
            rgb=RgbFrame(width=self.GRID_SIZE, height=self.GRID_SIZE, pixels=bytes(raw)),
            previous_control=control,
        )


def evaluate_session_under_distractor(
    model: torch.nn.Module,
    model_type: str,
    env: DistractorHiddenRuleEnv,
    seed: int,
    device: torch.device,
    use_plasticity: bool = False,
) -> Dict:
    """Evaluates a model over a multi-trial session with distractor delays."""
    obs = env.reset(seed=seed)
    frame_raw = np.frombuffer(obs.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
    frame_history = [frame_raw] * N_FRAMES

    recurrent_state = None
    P_t = None
    prev_action = 0
    surprise = torch.zeros(1, 1, device=device)

    done = False
    in_distractor = False
    step_count = 0
    t0 = time.perf_counter()

    while not done:
        step_count += 1
        stacked = np.stack(frame_history[-N_FRAMES:], axis=0).transpose(0, 3, 1, 2)
        frames_tensor = torch.from_numpy(stacked).float().unsqueeze(0).to(device) / 255.0
        prev_act_tensor = torch.tensor([prev_action], dtype=torch.long, device=device)

        with torch.no_grad():
            z_t = model.encode_observation(frames_tensor)
            if model_type == "gru":
                logits, recurrent_state, e_t, P_t, delta_P = model.forward_step(
                    z_t=z_t,
                    prev_action=prev_act_tensor,
                    surprise_t=surprise,
                    h=recurrent_state,
                    P_t=P_t,
                )
            else:
                if in_distractor and use_plasticity:
                    # Surprise-gated synaptic weights P_t remain frozen during distractor steps
                    frozen_P = P_t.clone() if P_t is not None else None
                    logits, recurrent_state, e_t, _, delta_P = model.forward_step(
                        z_t=z_t,
                        prev_action=prev_act_tensor,
                        surprise_t=surprise,
                        thoughts=recurrent_state,
                        P_t=frozen_P,
                    )
                    P_t = frozen_P
                else:
                    logits, recurrent_state, e_t, P_t, delta_P = model.forward_step(
                        z_t=z_t,
                        prev_action=prev_act_tensor,
                        surprise_t=surprise,
                        thoughts=recurrent_state,
                        P_t=P_t,
                    )

            action = int(logits.argmax(dim=-1).item())
            z_hat = model.predict_next_latent(recurrent_state, torch.tensor([action], device=device))

        ctrl = CTRL_MAP.get(action, CTRL_MAP[0])
        outcome = env.step(ctrl)

        in_distractor = ("distractor_delay" in outcome.events)

        next_raw = np.frombuffer(outcome.observation.rgb.pixels, dtype=np.uint8).reshape(16, 16, 3)
        frame_history.append(next_raw)

        next_stacked = np.stack(frame_history[-N_FRAMES:], axis=0).transpose(0, 3, 1, 2)
        next_tensor = torch.from_numpy(next_stacked).float().unsqueeze(0).to(device) / 255.0

        with torch.no_grad():
            if in_distractor:
                # No goal reached during distractor steps -> no surprise
                surprise = torch.zeros(1, 1, device=device)
            else:
                z_true_next = model.encode_observation(next_tensor)
                err = torch.norm(z_hat - z_true_next, dim=-1, keepdim=True)
                surprise = err

        prev_action = action
        if outcome.terminated or outcome.truncated:
            done = True

    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / max(step_count, 1)) * 1000.0

    return {
        "trial_outcomes": env.trial_outcomes,
        "total_steps": step_count,
        "latency_ms": latency_ms,
    }


def run_benchmark(
    checkpoint_dir: str | Path = "runs/online_adaptation_colab_v2",
    output_file: str | Path = "runs/online_adaptation_colab_v2/distractor_results.json",
    delays: List[int] = [0, 10, 25, 50],
    seeds: List[int] = [42],
    sessions_per_seed: int = 10,
    device_str: str = "cpu",
) -> Dict:
    ckpt_path = Path(checkpoint_dir)
    out_file = Path(output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str)

    # Standard schedule: 10 trials Rule A -> 10 trials Rule B (20 trials total)
    # Trial 1..10 (indices 0..9): Rule A
    # Trial 11 (index 10): Reversal to Rule B (first encounter with new rule)
    # Trial 12 (index 11): Post-reversal adaptation trial (target metric)
    test_schedule = [
        (Rule.RULE_A, 10),
        (Rule.RULE_B, 10),
    ]

    models_config = [
        ("Standard GRU", "gru", "gru_seed_42.pt", False),
        ("Thoughtlet Baseline", "thoughtlet", "thoughtlet_seed_42.pt", False),
        ("Plastic Thoughtlet", "thoughtlet", "plastic_thoughtlet_seed_42.pt", True),
    ]

    all_results = {
        "delays": delays,
        "seeds": seeds,
        "sessions_per_seed": sessions_per_seed,
        "models": {},
        "comparison": [],
    }

    print("=" * 80, flush=True)
    print("PSEUDO-BRAIN: DISTRACTOR DELAY STRESS-TEST BENCHMARK", flush=True)
    print(f"Checkpoints Directory: {ckpt_path}", flush=True)
    print(f"Delay Horizons D: {delays}", flush=True)
    print(f"Seeds: {seeds} | Sessions/seed: {sessions_per_seed} (Total: {len(seeds)*sessions_per_seed} sessions/run)", flush=True)
    print("Schedule: 10 Rule A -> 10 Rule B (20 trials/session)", flush=True)
    print("=" * 80, flush=True)

    loaded_models = {}
    for label, m_type, ckpt_name, use_plast in models_config:
        ckpt_file = ckpt_path / ckpt_name
        if not ckpt_file.exists():
            print(f"Error: Checkpoint {ckpt_file} not found!", flush=True)
            sys.exit(1)

        ckpt_data = torch.load(ckpt_file, map_location=device)
        if m_type == "gru":
            m = PredictiveGRUModel(use_plasticity=use_plast).to(device)
        else:
            m = PredictiveThoughtletModel(use_plasticity=use_plast).to(device)
        m.load_state_dict(ckpt_data["model_state_dict"])
        m.eval()
        loaded_models[label] = (m, m_type, ckpt_name, use_plast)

    # Run evaluations
    results_by_model: Dict[str, Dict[int, Dict]] = {}

    for label, (m, m_type, ckpt_name, use_plast) in loaded_models.items():
        print(f"\n>>> Evaluating Condition: {label.upper()} ({ckpt_name})", flush=True)
        results_by_model[label] = {}

        for D in delays:
            trial_correct_matrix = np.zeros((len(seeds) * sessions_per_seed, 20))
            latencies = []
            sess_idx = 0

            for s in seeds:
                for ep in range(sessions_per_seed):
                    sess_seed = s * 1000 + ep
                    env = DistractorHiddenRuleEnv(
                        rule_schedule=test_schedule,
                        distractor_delay=D,
                        distractor_noise_level=35.0,
                    )
                    sess_res = evaluate_session_under_distractor(
                        model=m,
                        model_type=m_type,
                        env=env,
                        seed=sess_seed,
                        device=device,
                        use_plasticity=use_plast,
                    )
                    outcomes = sess_res["trial_outcomes"]
                    for t_idx, item in enumerate(outcomes):
                        if t_idx < 20:
                            trial_correct_matrix[sess_idx, t_idx] = 1.0 if item["correct"] else 0.0
                    latencies.append(sess_res["latency_ms"])
                    sess_idx += 1

            trial_acc_mean = np.mean(trial_correct_matrix, axis=0) * 100.0
            t12_acc = float(trial_acc_mean[11]) if len(trial_acc_mean) > 11 else 0.0
            t11_acc = float(trial_acc_mean[10]) if len(trial_acc_mean) > 10 else 0.0
            t10_acc = float(trial_acc_mean[9]) if len(trial_acc_mean) > 9 else 0.0
            t1_acc = float(trial_acc_mean[0]) if len(trial_acc_mean) > 0 else 0.0

            results_by_model[label][D] = {
                "distractor_delay": D,
                "trial_12_accuracy": t12_acc,
                "trial_11_accuracy": t11_acc,
                "trial_10_accuracy": t10_acc,
                "trial_1_accuracy": t1_acc,
                "mean_latency_ms": float(np.mean(latencies)),
                "trial_accuracies": [float(x) for x in trial_acc_mean],
            }
            print(f"  D = {D:2d} | T10(Rule A)={t10_acc:5.1f}% | T11(Reversal)={t11_acc:5.1f}% | T12(Adaptation)={t12_acc:5.1f}%", flush=True)

    all_results["models"] = results_by_model

    # Build comparison summary table
    comparison_table = []
    for D in delays:
        entry = {"delay": D}
        for label in loaded_models:
            entry[f"{label}_t12_acc"] = results_by_model[label][D]["trial_12_accuracy"]
        comparison_table.append(entry)
    all_results["comparison"] = comparison_table

    # Save to JSON
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}", flush=True)

    # Print Markdown Comparison Table
    print_markdown_table(all_results)

    return all_results


def print_markdown_table(results: Dict):
    delays = results["delays"]
    models = list(results["models"].keys())

    print("\n" + "=" * 80)
    print("ADAPTATION ACCURACY (TRIAL 12 FOLLOWING REVERSAL) VS DISTRACTOR DELAY D")
    print("=" * 80)
    md_lines = []
    header = "| Distractor Delay $D$ | " + " | ".join(f"**{m}**" for m in models) + " |"
    sep = "|---:|" + "|".join("---:" for _ in models) + "|"
    md_lines.append(header)
    md_lines.append(sep)

    for D in delays:
        vals = []
        for m in models:
            acc = results["models"][m][D]["trial_12_accuracy"]
            vals.append(f"{acc:.1f}%")
        line = f"| $D = {D}$ | " + " | ".join(vals) + " |"
        md_lines.append(line)

    md_table = "\n".join(md_lines)
    print(md_table)
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distractor Stress-Test Benchmark")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="runs/online_adaptation_colab_v2",
        help="Path to checkpoint directory containing gru_seed_42.pt, thoughtlet_seed_42.pt, plastic_thoughtlet_seed_42.pt",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="runs/online_adaptation_colab_v2/distractor_results.json",
        help="Path to output JSON file",
    )
    parser.add_argument(
        "--delays",
        type=int,
        nargs="+",
        default=[0, 10, 25, 50],
        help="Distractor delay horizons D",
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=10,
        help="Evaluation sessions per seed",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds to evaluate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device (cpu or cuda)",
    )

    args = parser.parse_args()
    run_benchmark(
        checkpoint_dir=args.checkpoint_dir,
        output_file=args.output_file,
        delays=args.delays,
        seeds=args.seeds,
        sessions_per_seed=args.sessions,
        device_str=args.device,
    )
