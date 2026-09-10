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

Pseudo-Brain includes a fully autonomous lifelong learning agent (`AutonomousLifelongAgent`) that solves the "catastrophic forgetting" and "born again" bottlenecks:

- **Epistemic Humility Gating**: Detects unfamiliar concepts instead of hallucinating answers. Admits lack of knowledge and initiates live autonomous research.
- **Multi-Tier Autonomous Web Research**: Features live search via Wikipedia REST API, Wikipedia query search, and DuckDuckGo Instant Answer API with zero prompt pre-baking.
- **Lifelong Episodic Consolidation**: Genuinely learns new facts into persistent hierarchical memory on disk (`agent_cli_state.pt`). Once a topic is consolidated, follow-up queries recall in under **10 ms** with **zero network traffic**.
- **Natural Human Conversation**: Speaks like a regular person without robotic meta-talk (*no "I remember this from earlier!" or "in my episodic memory"*).
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
