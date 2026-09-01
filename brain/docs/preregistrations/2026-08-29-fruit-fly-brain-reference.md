# Fruit-Fly Brain Reference for Pseudo-Brain

**Preregistered:** 2026-08-29
**Status:** PROPOSED (not yet started)
**Owner:** Pseudo-Brain project
**Canonical compute:** Local CPU (per AGENTS.md)

---

## 1. Motivation

The current pseudo-brain architecture (VectorizedPseudoBrain, thoughtlet-based slot-recurrent core) was designed top-down. This probe explores whether the *Drosophila melanogaster* (fruit fly) brain connectome can serve as a structural and algorithmic reference for improving robustness, modularity, and biological plausibility.

## 2. Biological Reference Points

### 2.1 Central Complex — Ring Attractor Navigation
- ~160,000 neurons in the adult CNS (120K brain + 14.6K VNC)
- Central complex contains ring attractor neurons that compute heading direction
- Path integration via self-supervised learning during development (Vafidis et al., eLife 2022)
- Quasi-continuous attractor networks maintain stable heading representations even in darkness

### 2.2 Mushroom Body — Sparse Coding & Associative Memory
- ~983 neurons in the MB learning/memory center (Takemura et al., eLife 2017)
- Kenyon cells (KCs) encode sensory information sparsely; each makes multiple en passant synapses to MB output neurons (MBONs)
- Dopaminergic neurons (DANs) modulate learning: only ~6% of KC→MBON synapses receive direct DAN input
- Unanticipated circuit motifs: KC→DAN and DAN→MBON (feedback loops for memory gating)

### 2.3 FLYNN Architecture (Connectome-Derived RNN)
- Directly derives RNN topology from the Drosophila synaptic-resolution connectome
- Demonstrates OOD generalization and sensory-loss tolerance (total vision loss survivable)
- High representational modularity (PCA shows structured latent space)
- Same parameter count as hand-crafted networks but with superior robustness

## 3. Proposed Research Questions

1. **Modularity**: Can the pseudo-brain's thoughtlet slots be organized into fly-inspired functional modules (sensory encoding → integration → memory → action) with structured connectivity priors?
2. **Sparse coding**: Would replacing dense thoughtlet activations with Kenyon-cell-like sparse coding improve robustness and reduce interference?
3. **Local learning rules**: Can self-supervised predictive learning (inspired by the fly path-integration rule) replace or supplement backpropagation for thoughtlet gate training?
4. **Dopaminergic gating**: Can a DAN-inspired modulatory pathway improve credit assignment in the thoughtlet architecture?
5. **Structural inductive biases**: Does embedding fly-like connectivity motifs (feedforward + feedback loops) as architectural priors improve sample efficiency?

## 4. Scope Boundaries

- **Included**: Connectome analysis, architecture design, small-scale probes, ablation comparisons
- **Excluded**: Full biological simulation, genetic algorithms, full 160K-neuron modeling
- **Compute budget**: CPU-only, single-thread, deterministic; no GPU, no capture, no network

## 5. Deliverables

1. Literature review & connectome mapping document
2. Fly-inspired architecture sketch (module topology, sparse coding, modulatory gating)
3. Probe script: `fly_brain_reference_probe.py` in `brain/scratch/`
4. Comparison: fly-inspired vs. canonical VectorizedPseudoBrain on current RCQ evaluation protocol
5. Go/no-go recommendation for deeper integration

## 6. Gate Criteria

- **[MEASURED]** Fly-inspired variant passes RCQ v2 evaluation (no regression vs. canonical)
- **[MEASURED]** Demonstrated improvement in at least one robustness metric (OOD, sensory dropout)
- **[MEASURED]** Sparse coding variant shows lower interference (higher recall precision)

## 7. Related Preregistrations

- `2026-08-25-action-query-thought-consumption-1.md` — core thoughtlet architecture
- `2026-08-28-core-v2-joint-training-v1.md` — Core V2 training
- `2026-08-28-pacman-reactive-baseline-v1.md` — reactive baseline comparison

## 8. Notes

- No prior work on biological reference models exists in this project yet
- This is a fresh exploratory probe, not a continuation of existing work
- All findings to be recorded in `brain/docs/runs/` upon execution
