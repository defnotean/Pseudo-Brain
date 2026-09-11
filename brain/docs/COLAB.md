# Google Colab Remote Training & Experimentation Guide

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/defnotean/Pseudo-Brain/blob/main/notebooks/pseudobrain_colab_training.ipynb)

This guide details how to run Pseudo-Brain experiments on **Google Colab** using premium GPU compute (NVIDIA A100, V100, L4, T4) with persistent Google Drive storage, unattended background execution, automated remote dispatch via CLI, and live real-time local monitoring.

---

## 1. First-Time Setup in Google Colab

### A. Open the Notebook
1. Open [Google Colab](https://colab.research.google.com/).
2. Select the **GitHub** tab, enter repository URL:
   ```text
   https://github.com/defnotean/Pseudo-Brain
   ```
3. Open [`notebooks/pseudobrain_colab_training.ipynb`](../notebooks/pseudobrain_colab_training.ipynb).
   *(Or upload `notebooks/pseudobrain_colab_training.ipynb` directly).*

### B. Select GPU Accelerator (Google AI Ultra)
1. In Colab menu, navigate to:
   `Runtime` -> `Change runtime type`
2. Select:
   - **Hardware accelerator**: `GPU`
   - **GPU type**: `A100` (Fastest, recommended for multi-seed sweeps) or `L4 / T4`
   - **Runtime shape**: `High-RAM`
3. Click **Save**.

---

## 2. Google Drive Persistent Storage Layout

The notebook automatically connects to Google Drive and initializes:

```text
MyDrive/
└── PseudoBrain/
    ├── checkpoints/    # Intermediate and latest model/optimizer checkpoints (.pt)
    ├── runs/           # Per-run logs, run_meta.json, eval_results.json
    ├── datasets/       # Generated corpora (keys_doors_corpus_v1, hidden_rule_corpus_v1)
    ├── results/        # Aggregated benchmark tables and markdown reports
    └── logs/           # Session output transcripts
```

> **Persistence requires verification**: Files survive runtime deletion only if they were successfully saved to a real Google Drive mount or downloaded elsewhere. Colab can terminate before a checkpoint finishes writing; there is no zero-data-loss guarantee.

---

## 3. Starting Training & Running Benchmarks

Run the notebook cells sequentially:
- **Cell 1**: Mounts Google Drive (`/content/drive/MyDrive/PseudoBrain`).
- **Cell 2**: Clones/syncs the Pseudo-Brain repository.
- **Cell 3**: Verifies dependencies (`torch`, `numpy`, `scipy`, `pandas`).
- **Cell 4**: Telemetry check — detects active GPU model and VRAM.
- **Cell 5**: Dataset check — automatically generates the benchmark datasets if missing.
- **Cell 6 & 7**: Interactive Launcher — configure and start training.

### Using the Unified CLI in Colab Terminal or Cells

You can launch experiments directly from a Colab cell via the unified runner:

#### Hidden Rule Online Adaptation Benchmark:
```bash
python -m brain.experiments.run \
  --experiment online_adaptation \
  --models reactive gru thoughtlet plastic_thoughtlet \
  --seeds 42 142 242 \
  --steps 1500 \
  --device auto \
  --output_dir /content/drive/MyDrive/PseudoBrain/runs
```

#### KeysDoors Memory Benchmark:
```bash
python -m brain.experiments.run \
  --experiment memory_benchmark \
  --models reactive gru thoughtlet \
  --seeds 42 142 242 \
  --steps 4000 \
  --device auto \
  --output_dir /content/drive/MyDrive/PseudoBrain/runs
```

#### Self-Correction Benchmark:
```bash
python -m brain.experiments.run \
  --experiment self_correction \
  --models gru thoughtlet plastic_thoughtlet \
  --seeds 42 142 242 \
  --steps 2000 \
  --device auto \
  --output_dir /content/drive/MyDrive/PseudoBrain/runs
```

---

## 4. Unattended Execution & Disconnect Safety

With a Google AI Ultra subscription:
1. Launch cell 7 (or your bash command).
2. Once the training loop begins printing steps and ETA, **you can safely close your browser tab or shut down your local machine**.
3. Google Colab will continue running the container remotely in the background.
4. Periodic checkpoints (`checkpoint_latest.pt`) are written to Google Drive every 250 steps.

---

## 5. Resuming an Interrupted Run

If a Colab runtime terminates or disconnects during training:
1. Re-open `notebooks/pseudobrain_colab_training.ipynb`.
2. Run cells 1–5 to mount Drive and check GPU.
3. In cell 6, set:
   ```python
   resume = True
   ```
4. Run cell 7.
   The runner will inspect `MyDrive/PseudoBrain/runs/`, find the latest checkpoint, reload model weights and optimizer state, and resume from the exact step where it was interrupted.

Or via CLI:
```bash
python -m brain.experiments.run \
  --experiment online_adaptation \
  --model thoughtlet \
  --seed 42 \
  --resume \
  --output_dir /content/drive/MyDrive/PseudoBrain/runs
```

---

## 6. Accessing & Syncing Results

### In Google Drive:
All results appear under:
`MyDrive/PseudoBrain/runs/<experiment>/`
- `aggregate_results.json`: Complete machine-readable results across all models and seeds.
- `aggregate_results.csv`: Spreadsheet-ready table comparing parameters, latency, and accuracies.
- `<model>_seed_<seed>/run_meta.json`: Full hardware, seed, git commit, and duration metadata.
- `<model>_seed_<seed>/eval_results.json`: Specific held-out benchmark scores.

### Downloading to Local Machine:
1. **Google Drive Desktop**: If Google Drive is installed locally, results will sync automatically to your local file explorer.
2. **Web Download**: In Google Drive web UI, right-click `PseudoBrain/runs/<experiment>` and select **Download**.
3. **Colab File Browser**: In Colab left sidebar, navigate to `drive/MyDrive/PseudoBrain/runs/` and download any artifact.

---

## 7. Automated CLI Remote Dispatch (`scripts/colab_dispatch.py`)

For a completely browserless workflow, Pseudo-Brain provides a unified dispatch bridge using Google's official `google-colab-cli`:

### A. One-Time Setup
```bash
# Install / verify google-colab-cli (bridges via WSL on Windows automatically)
python scripts/colab_dispatch.py install

# Authenticate with your Google AI Ultra account
python scripts/colab_dispatch.py login
```

### B. Launch Remote Experiment on Colab GPU
Dispatch directly to an NVIDIA A100 GPU (or L4 / T4):
```bash
python scripts/colab_dispatch.py run \
  --experiment online_adaptation \
  --models thoughtlet plastic_thoughtlet \
  --gpu A100 \
  --steps 1500
```
This automatically:
1. Provisions a fresh remote Colab container with the requested GPU.
2. Clones or syncs the Pseudo-Brain repository.
3. Mounts Google Drive (`MyDrive/PseudoBrain/runs`).
4. Executes the unified runner and streams stdout/stderr live to your terminal.
5. Shuts down the runtime upon completion to conserve your Google AI Ultra compute units (or pass `--keep` to retain it).

### C. Session Management
```bash
# Check running sessions
python scripts/colab_dispatch.py status

# Terminate active session
python scripts/colab_dispatch.py stop
```

---

## 8. Real-Time Telemetry, Live Terminal Watcher & Alerts

### A. Live Local Terminal Watcher
While training runs remotely on Google Colab, you can watch live progress, loss curves, rates, and GPU memory from your local terminal:
```bash
# Monitor active runs via Google Drive Desktop:
python -m brain.experiments.monitor --path "G:\My Drive\PseudoBrain\runs"

# Or one-shot status check:
python -m brain.experiments.monitor --once
```

### B. Webhook Notifications (Discord / Slack)
Pass `--webhook <url>` to receive instant mobile and desktop alerts when:
- 🚀 Training starts (with GPU model and run parameters)
- ⏳ Progress reaches milestones (25%, 50%, 75% with ETA and loss)
- ✅ Benchmark completes (with full evaluation accuracy table)
- ❌ Any unhandled error occurs (with full traceback)

Example:
```bash
python scripts/colab_dispatch.py run \
  --experiment online_adaptation \
  --models thoughtlet \
  --webhook "https://discord.com/api/webhooks/..."
```

