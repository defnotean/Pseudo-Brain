"""Live Local Terminal Watcher for Pseudo-Brain Remote Training.

Continuously monitors Google Drive or local run directories, rendering
real-time progress, loss trends, rates, and hardware utilization.

Usage:
  # Monitor the latest active run in brain/runs:
  python -m brain.experiments.monitor

  # Monitor a specific run directory:
  python -m brain.experiments.monitor --path brain/runs/online_adaptation/thoughtlet_seed_42

  # Monitor remote Google Drive on Windows (if Drive Desktop is installed):
  python -m brain.experiments.monitor --path "G:\My Drive\PseudoBrain\runs"

  # One-shot status check:
  python -m brain.experiments.monitor --once
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def find_latest_heartbeat(root_path: Path) -> Optional[Path]:
    """Finds the most recently modified heartbeat.json under root_path."""
    if (root_path / "heartbeat.json").exists():
        return root_path / "heartbeat.json"

    candidates = list(root_path.rglob("heartbeat.json"))
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def render_progress_bar(pct: float, width: int = 30) -> str:
    filled = int(width * (pct / 100.0))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct:5.1f}%"


def sparkline(values: List[float], steps: int = 8) -> str:
    """Generates a compact ASCII sparkline for loss trends."""
    if not values:
        return ""
    recent = values[-steps:]
    min_v, max_v = min(recent), max(recent)
    if max_v == min_v:
        return "―" * len(recent)

    chars = [" ", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
    res = []
    for v in recent:
        idx = int((v - min_v) / (max_v - min_v) * (len(chars) - 1))
        res.append(chars[idx])
    return "".join(res)


def render_dashboard(data: dict, loss_history: List[float], clear: bool = True) -> None:
    # Clear screen or ANSI reset if interactive
    if clear and sys.stdout.isatty():
        print("\033[H\033[J", end="")

    exp = data.get("experiment", "unknown")
    model = data.get("model", "unknown")
    seed = data.get("seed", 0)
    step = data.get("step", 0)
    total_steps = data.get("total_steps", 0)
    pct = data.get("progress_pct", 0.0)
    act_loss = data.get("act_loss", 0.0)
    pred_loss = data.get("pred_loss", 0.0)
    surprise = data.get("surprise", 0.0)
    elapsed_s = data.get("elapsed_s", 0.0)
    rate = data.get("rate_steps_per_sec", 0.0)
    eta_str = data.get("eta_str", "--")
    gpu = data.get("gpu_name") or "CPU"
    vram = data.get("vram_allocated_gb")
    vram_str = f"{vram} GB" if vram is not None else "N/A"
    status = data.get("status", "training").upper()

    status_icon = "🟢" if status == "TRAINING" else ("✅" if status == "COMPLETED" else "❌")

    print("=" * 65)
    print("🧠 PSEUDO-BRAIN REAL-TIME TELEMETRY MONITOR")
    print("=" * 65)
    print(f"  Experiment:      {exp.upper()}")
    print(f"  Model & Seed:    {model.upper()} (Seed {seed})")
    print(f"  Status:          {status_icon} {status}")
    print(f"  Hardware:        {gpu} | VRAM: {vram_str}")
    print("-" * 65)
    print(f"  Progress:        {render_progress_bar(pct)} ({step:,}/{total_steps:,} steps)")
    print(f"  Rate:            {rate:.1f} steps/s | Elapsed: {int(elapsed_s)}s | ETA: {eta_str}")
    print("-" * 65)
    print(f"  Action Loss:     {act_loss:.4f}  {sparkline(loss_history)}")
    print(f"  Prediction Loss: {pred_loss:.4f}")
    print(f"  Latent Surprise: {surprise:.4f}")
    print("=" * 65)


def monitor_loop(target_path: Path, poll_interval_s: float = 1.5, once: bool = False):
    print(f"Searching for active runs under: {target_path.resolve()}...")
    loss_history = []
    last_step = -1

    while True:
        hb_file = find_latest_heartbeat(target_path)
        if hb_file is None:
            print(f"\rWaiting for heartbeat.json under {target_path.resolve()}... ", end="", flush=True)
            if once:
                print("\nNo heartbeat found.")
                return
            time.sleep(poll_interval_s)
            continue

        try:
            with open(hb_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            curr_step = data.get("step", 0)
            if curr_step != last_step:
                last_step = curr_step
                loss_history.append(data.get("act_loss", 0.0))

            render_dashboard(data, loss_history, clear=not once)

            # Check if finished
            if data.get("status") in ("completed", "failed"):
                run_dir = hb_file.parent
                eval_file = run_dir / "eval_results.json"
                if eval_file.exists():
                    print("\n📈 FINAL BENCHMARK EVALUATION RESULTS:")
                    try:
                        with open(eval_file, "r", encoding="utf-8") as ef:
                            eval_data = json.load(ef)
                        print(json.dumps(eval_data.get("eval", {}), indent=2))
                    except Exception:
                        pass
                print(f"\nRun reached terminal status: {data.get('status').upper()}.")
                if once:
                    return
                print("Monitoring paused. Press Ctrl+C to exit.")
                return

        except Exception as e:
            pass

        if once:
            return

        time.sleep(poll_interval_s)


def main():
    parser = argparse.ArgumentParser(description="Pseudo-Brain Live Terminal Watcher")
    parser.add_argument(
        "--path",
        type=str,
        default="brain/runs",
        help="Path to run directory or root directory containing heartbeat.json",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.5,
        help="Polling interval in seconds",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print status once and exit immediately",
    )

    args = parser.parse_args()
    try:
        monitor_loop(Path(args.path), poll_interval_s=args.interval, once=args.once)
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


if __name__ == "__main__":
    main()
