# Pseudo-Brain

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/defnotean/Pseudo-Brain/blob/main/notebooks/pseudobrain_colab_training.ipynb)

This is the canonical workspace for the persistent multi-thought sensorimotor
model research project and its long-term expansion into a general-purpose
cognitive agent ([brain/PLAN.md §43](./brain/PLAN.md#43-long-term-research-direction-general-purpose-language-tool-use--long-horizon-agent)).

**Start here:** [CURRENT_WORK.md](./CURRENT_WORK.md). That file says what the
current experiment is, what a pass would mean, and which document to open next.
The step-by-step operator commands are in
[brain/docs/OPERATOR_GUIDE.md](./brain/docs/OPERATOR_GUIDE.md).

The implementation, experiment configurations, tests, and research records are
under [brain](./brain/README.md). The Irene desktop/chat project is separate and
is not a dependency or training-data source for Pseudo-Brain.

## Google Colab Remote Training & Watching

Pseudo-Brain supports remote execution on Google Colab with background execution (Google AI Ultra), automatic Google Drive persistence (`MyDrive/PseudoBrain/`), resumable training, and real-time local monitoring:

- **Turnkey Notebook**: Open [`notebooks/pseudobrain_colab_training.ipynb`](./notebooks/pseudobrain_colab_training.ipynb) in Colab and run with 1-click.
- **Remote CLI Dispatch**:
  ```bash
  python scripts/colab_dispatch.py run --experiment online_adaptation --models thoughtlet --gpu A100 --steps 1500
  ```
- **Real-Time Terminal Watcher**:
  ```bash
  python -m brain.experiments.monitor --path "G:\My Drive\PseudoBrain\runs"
  ```
- **Full Guide**: See [`docs/COLAB.md`](./docs/COLAB.md) for architecture, unattended background execution, and webhook alerts.

The Python namespace `irene_brain` is retained temporarily for compatibility
with immutable historical checkpoints. New releases and runs are produced from
this project directory.

The checked-in `.pseudo-brain-workspace-v2` marker identifies this local
release source. DGX sync also verifies that this directory is the actual Git
top-level before packaging `brain/`.
