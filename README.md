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

## Current evidence (2026-09-11)

Pseudo-Brain is an experimental recurrent architecture. The active POMDP policy
has 34.37M parameters; its latest unchanged-weight audit completes 1/10 Level A,
2/10 Level B and 0/10 Level C tasks, with no verified autonomous repairs. The
current tests do not establish general-purpose or frontier-model competence.
See the [setup audit](brain/docs/runs/2026-09-11-pomdp-setup-audit.md) for numerical,
data-overlap and causal-memory findings and the separate training candidate.
The 4 KB budget describes only the fast working tensor, not total inference memory.

A separate22.39M-parameter parallel recurrent candidate and22.97M-parameter
transformer completed the same3072-update broad training pilot on Colab. Both
scored0/32 HumanEval and0/32 GSM8K, with no missing tasks. Lower development loss
and a completed training run have not produced useful coding/reasoning ability.
The candidate retains the parallel immutable-prompt copy head and4096-byte state;
it has not replaced the POMDP champion. See CURRENT_WORK.md for live diagnostics
and the preserved run evidence.

## Unified Model Architecture & Scaling Tiers

Pseudo-Brain implements a unified architecture spanning 3 model parameter tiers, all strictly complying with **Law 1 (Working Memory Invariance: $K=16, W=64$, strictly 4,096 bytes)**:

| Tier | Trainable Parameters | Working Memory | Target Capabilities | Checkpoint |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1 (Base)** | **3.09M** | 4.0 KB | 60 Hz POMDP Sensory Control, Basic Tool Calling | `pb_unified_champion.pt` |
| **Tier 2 (Mid)** | **36.7M** | 4.0 KB | Deep Recurrence, BPE Vocab, Multi-Turn CoT Reasoning | `pb_35m_champion.pt` |
| **Tier 3 (Large)** | **1.02B** | 4.0 KB | Fused Triton Associative Scan, Hierarchical Memory, Mental Lookahead, Multi-Task Pretraining | `pb_1b_champion.pt` |

For full scaling experiments, benchmarks, and Colab A100 training logs, see [docs/SCALING_1B_WALKTHROUGH.md](./docs/SCALING_1B_WALKTHROUGH.md).

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

## Autonomous Lifelong Cognitive Agent & Live Web Research

The repository includes an experimental `AutonomousLifelongAgent`, retrieval
integrations and a conversational CLI. Component demonstrations and persisted
facts do not establish that catastrophic forgetting, hallucination or open-ended
reasoning have been solved. Those claims require independent capability tests.

- **Interactive CLI with Transparent Telemetry**:
  ```bash
  # Ask a novel question and watch real-time web research & consolidation:
  python brain/ask_agent.py "What is Minecraft?"

  # Ask a follow-up question and observe instant memory recall:
  python brain/ask_agent.py "What do you know about Minecraft?"

  # Start an interactive conversational terminal session:
  python brain/ask_agent.py
  ```

The Python namespace `irene_brain` is retained temporarily for compatibility
with immutable historical checkpoints. New releases and runs are produced from
this project directory.

The checked-in `.pseudo-brain-workspace-v2` marker identifies this local
release source. DGX sync also verifies that this directory is the actual Git
top-level before packaging `brain/`.
