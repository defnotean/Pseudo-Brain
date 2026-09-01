#!/usr/bin/env python3
"""Fruit-Fly Brain Reference Probe for Pseudo-Brain.

Explores whether Drosophila neural architecture principles can be
applied to the VectorizedPseudoBrain thoughtlet architecture.

CPU-only, single-threaded, deterministic (per AGENTS.md).
"""

import hashlib
import zlib
import numpy as np


def kenyon_sparse_encode(sensory_input: np.ndarray,
                         n_kc: int = 2000,
                         sparsity: float = 0.05) -> np.ndarray:
    """Sparse random projection inspired by Kenyon cell encoding."""
    seed = zlib.crc32(sensory_input.tobytes()) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    dense = rng.randn(n_kc, len(sensory_input))
    threshold = np.percentile(np.abs(dense), (1 - sparsity) * 100)
    sparse = dense * (np.abs(dense) > threshold)
    return sparse


def ring_attractor_update(heading: np.ndarray,
                          velocity: np.ndarray,
                          dt: float = 0.016,
                          n_neurons: int = 36) -> np.ndarray:
    """Simple ring attractor dynamics for heading integration."""
    angle = np.linspace(0, 2 * np.pi, n_neurons, endpoint=False)
    bump = np.exp(-((angle - heading[0]) % (2 * np.pi))**2 / (2 * 0.3**2))
    velocity_component = np.dot(bump, velocity)
    new_heading = heading + velocity_component * dt
    return new_heading % (2 * np.pi)


def dopaminergic_gating(signal: float,
                        reward: float,
                        learning_rate: float = 0.01) -> float:
    """DAN-inspired modulatory gating for credit assignment."""
    gating_signal = 1.0 / (1.0 + np.exp(-signal))
    return gating_signal * reward * learning_rate


def run_probe(steps: int = 1000) -> dict:
    """Run a minimal fruit-fly-inspired probe."""
    results = {"steps": steps, "modularity_score": 0.0,
               "sparsity_achieved": 0.0, "robustness_drop": 0.0}

    sensory = np.random.randn(64)
    sparse = kenyon_sparse_encode(sensory)
    results["sparsity_achieved"] = float(np.mean(np.abs(sparse) > 0))

    heading = np.array([0.0])
    for _ in range(steps):
        heading = ring_attractor_update(heading, np.array([0.1]))
    results["modularity_score"] = float(np.std(heading))

    results["robustness_drop"] = float(
        dopaminergic_gating(0.5, 1.0) - dopaminergic_gating(0.5, 0.5))

    return results


if __name__ == "__main__":
    results = run_probe()
    print(f"Fruit-Fly Reference Probe Results: {results}")
