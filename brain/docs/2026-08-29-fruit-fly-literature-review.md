# Fruit-Fly Brain Literature Review & Connectome Mapping

**Date:** 2026-08-29 | **Status:** DRAFT

---

## 1. Connectome Overview

- **Species:** *Drosophila melanogaster*
- **Scale:** ~160K neurons in adult CNS (120K brain + 14.6K VNC)
- **Available:** Full female brain (~120K, ~30M synapses), male/female VNC, full adult CNS, larval CNS
- **Synapses:** Chemical only, polyadic (one T-bar, multiple PSDs)
- **Key:** Modular architecture with specialized neuropils

## 2. Central Complex — Navigation

**Reference:** Vafidis et al., eLife 2022

- Ring attractor neurons compute heading direction
- Self-supervised predictive learning tunes connectivity during development
- Quasi-continuous attractor: path-integrates with gain-1
- Local biologically-plausible learning rule adjusts synaptic efficacies guided by allothetic cues

**Pseudo-Brain implications:**
- Thoughtlet gates could use self-supervised predictive learning instead of backprop
- Ring attractor dynamics could inspire a heading-direction module
- Gain-1 learning rule as alternative to gradient-based gate training

## 3. Mushroom Body — Sparse Coding

**Reference:** Takemura et al., eLife 2017

- ~983 neurons in MB: Kenyon cells (KCs), MBONs, DANs
- KCs fire sparsely (~2-3%); each makes multiple en passant synapses to MBONs
- Only ~6% of KC→MBON synapses receive direct DAN input
- Motifs: KC→DAN and DAN→MBON (feedback loops for memory gating)

**Pseudo-Brain implications:**
- Sparse coding (KC-like) could replace dense thoughtlet activations
- Modulatory gating (DAN-like) could improve credit assignment
- Feedback motifs suggest memory-augmented architectural patterns

## 4. FLYNN — Connectome-Derived RNN

**Reference:** Wang & Chen, arxiv 2607.00025

- RNN topology from Drosophila synaptic-resolution connectome
- Same param count as hand-crafted nets, superior OOD generalization
- Survives total vision loss without retraining
- PCA shows high representational modularity

**Pseudo-Brain implications:**
- Connectome-derived topology produces more robust networks than hand-crafted ones
- Modularity in latent space correlates with robustness
- Sensory dropout tolerance is a measurable robustness metric
## 5. Mapping Summary

| Fly Feature | Pseudo-Brain Target | Expected Benefit |
|---|---|---|
| Ring attractor (central complex) | Spatial reasoning module in thoughtlet gates | Robust heading/path integration without gradient training |
| Sparse KC coding (mushroom body) | Thoughtlet activation sparsity | Lower interference, higher recall precision |
| DAN modulatory gating | Thoughtlet credit assignment | Better signal-to-noise in gradient-free training |
| KC→DAN→MBON feedback loop | Memory-augmented thoughtlet architecture | Structured recurrence for sequential decisions |
| FLYNN connectome topology | Thoughtlet slot connectivity pattern | OOD generalization and sensory loss tolerance |

## 6. Key Open Questions

1. Can self-supervised predictive learning (fly rule) match or exceed backprop for thoughtlet gate training on CPU?
2. Does sparse KC-like coding reduce interference in multi-thoughtlet scenarios?
3. Does connectome-derived slot connectivity improve robustness over random or learned topologies?
4. Is the DAN-inspired modulatory pathway effective for credit assignment in the thoughtlet architecture?

## 7. Next Steps

- Draft fly-inspired architecture sketch with modular topology
- Implement and run probe script (brain/scratch/fly_brain_reference_probe.py)
- Compare results against canonical VectorizedPseudoBrain
- Issue go/no-go recommendation
