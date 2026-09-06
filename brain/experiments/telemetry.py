"""Real-time telemetry and monitoring engine for Pseudo-Brain experiments.

Features:
- Heartbeat file persistence (`heartbeat.json`) for remote watching.
- Universal Webhook dispatch (Discord, Slack, or generic HTTP POST) using standard library.
- Optional Weights & Biases (`wandb`) integration.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

import torch


class TelemetryLogger:
    """Manages periodic heartbeat writing, webhook alerts, and cloud logging."""

    def __init__(
        self,
        run_dir: str | Path,
        experiment: str,
        model_name: str,
        seed: int,
        total_steps: int,
        webhook_url: Optional[str] = None,
        use_wandb: bool = False,
        heartbeat_interval_s: float = 2.0,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_file = self.run_dir / "heartbeat.json"

        self.experiment = experiment
        self.model_name = model_name
        self.seed = seed
        self.total_steps = total_steps
        self.webhook_url = webhook_url or os.environ.get("WEBHOOK_URL")
        self.use_wandb = use_wandb
        self.heartbeat_interval_s = heartbeat_interval_s

        self.start_time = time.time()
        self.last_heartbeat_time = 0.0
        self.last_milestone_pct = 0

        # Attempt optional wandb initialization
        self.wandb_run = None
        if self.use_wandb:
            try:
                import wandb  # type: ignore
                self.wandb_run = wandb.init(
                    project=f"pseudobrain-{self.experiment}",
                    name=f"{self.model_name}_seed_{self.seed}",
                    config={
                        "experiment": self.experiment,
                        "model": self.model_name,
                        "seed": self.seed,
                        "total_steps": self.total_steps,
                    },
                )
            except Exception as e:
                print(f"[TELEMETRY] wandb init skipped / failed: {e}")

        # Post start alert via webhook
        self.send_webhook(
            event="START",
            message=f"🚀 **Training Started**: `{self.experiment}` | Model: `{self.model_name}` | Seed: `{self.seed}` | Steps: `{self.total_steps}`",
        )

    def log_step(
        self,
        step: int,
        act_loss: float,
        pred_loss: float = 0.0,
        surprise: float = 0.0,
        force: bool = False,
    ) -> None:
        """Called at each training step to update telemetry and heartbeat."""
        now = time.time()
        if not force and (now - self.last_heartbeat_time < self.heartbeat_interval_s) and step < self.total_steps:
            return

        self.last_heartbeat_time = now
        elapsed = now - self.start_time
        steps_done = max(step, 1)
        rate = steps_done / max(elapsed, 1e-5)
        steps_left = max(0, self.total_steps - step)
        eta_s = steps_left / max(rate, 1e-5)
        eta_str = f"{int(eta_s // 60)}m{int(eta_s % 60):02d}s"
        progress_pct = round((step / max(self.total_steps, 1)) * 100.0, 1)

        # Hardware VRAM check if CUDA active
        vram_gb = None
        gpu_name = None
        if torch.cuda.is_available():
            try:
                vram_gb = round(torch.cuda.memory_allocated() / (1024**3), 2)
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                pass

        payload = {
            "experiment": self.experiment,
            "model": self.model_name,
            "seed": self.seed,
            "step": step,
            "total_steps": self.total_steps,
            "progress_pct": progress_pct,
            "act_loss": round(float(act_loss), 4),
            "pred_loss": round(float(pred_loss), 4),
            "surprise": round(float(surprise), 4),
            "elapsed_s": round(elapsed, 1),
            "rate_steps_per_sec": round(rate, 2),
            "eta_s": round(eta_s, 1),
            "eta_str": eta_str,
            "gpu_name": gpu_name,
            "vram_allocated_gb": vram_gb,
            "timestamp": now,
            "status": "completed" if step >= self.total_steps else "training",
        }

        # Write atomic heartbeat file
        tmp_file = self.run_dir / "heartbeat.tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            tmp_file.replace(self.heartbeat_file)
        except Exception:
            pass

        # Log to wandb if enabled
        if self.wandb_run:
            try:
                self.wandb_run.log({
                    "step": step,
                    "act_loss": act_loss,
                    "pred_loss": pred_loss,
                    "surprise": surprise,
                    "rate": rate,
                    "vram_gb": vram_gb,
                })
            except Exception:
                pass

        # Check milestone alerts (every 25%)
        milestone = int(progress_pct // 25) * 25
        if milestone > self.last_milestone_pct and milestone < 100:
            self.last_milestone_pct = milestone
            self.send_webhook(
                event="MILESTONE",
                message=f"⏳ **Progress Update ({milestone}%)**: `{self.model_name}` (Seed {self.seed}) | Step {step}/{self.total_steps} | Act Loss: `{act_loss:.4f}` | ETA: `{eta_str}`",
            )

    def log_completion(self, eval_results: Optional[Dict[str, Any]] = None) -> None:
        """Called upon successful run completion."""
        elapsed = time.time() - self.start_time
        self.log_step(self.total_steps, act_loss=0.0, force=True)

        eval_summary = ""
        if eval_results:
            eval_summary = "\n**Evaluation Results:**\n```json\n" + json.dumps(eval_results, indent=2) + "\n```"

        self.send_webhook(
            event="COMPLETE",
            message=f"✅ **Training Completed**: `{self.experiment}` | Model: `{self.model_name}` (Seed `{self.seed}`) in `{elapsed:.1f}s`.{eval_summary}",
        )

        if self.wandb_run:
            try:
                self.wandb_run.finish()
            except Exception:
                pass

    def log_failure(self, error_message: str) -> None:
        """Called upon run failure or exception."""
        # Update heartbeat status to failed
        try:
            if self.heartbeat_file.exists():
                with open(self.heartbeat_file, "r", encoding="utf-8") as f:
                    hb = json.load(f)
                hb["status"] = "failed"
                hb["error"] = str(error_message)
                with open(self.heartbeat_file, "w", encoding="utf-8") as f:
                    json.dump(hb, f, indent=2)
        except Exception:
            pass

        self.send_webhook(
            event="FAILURE",
            message=f"❌ **Run Failed**: `{self.experiment}` | Model: `{self.model_name}` (Seed `{self.seed}`)\n**Error:** `{error_message}`",
        )

        if self.wandb_run:
            try:
                self.wandb_run.finish(exit_code=1)
            except Exception:
                pass

    def send_webhook(self, event: str, message: str) -> None:
        """Sends an HTTP webhook notification asynchronously without blocking."""
        if not self.webhook_url:
            return

        try:
            # Discord / Slack compatible JSON payload
            payload = json.dumps({"content": message, "text": message}).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json", "User-Agent": "PseudoBrain-Telemetry"},
                method="POST",
            )
            # Short timeout so training is never stalled by network hiccups
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                pass
        except Exception as e:
            # Telemetry errors must never crash scientific training
            pass


__all__ = ["TelemetryLogger"]
