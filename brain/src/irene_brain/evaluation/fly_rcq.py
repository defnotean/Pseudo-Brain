"""RCQ evaluation wrapper for fly-inspired VectorizedPseudoBrain variant.

Evaluates on same RCQ protocol as canonical VectorizedPseudoBrain.
CPU-only, single-threaded, deterministic per AGENTS.md.
"""

import numpy as np
import torch


def deterministic_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def rcq_evaluate(model, n_steps: int = 1000,
                 batch_size: int = 16, input_dim: int = 64) -> dict:
    """Run RCQ evaluation: determinism, output stats, sensory dropout robustness."""
    model.eval()
    deterministic_seed(42)
    results = {}

    with torch.no_grad():
        x = torch.randn(batch_size, input_dim)
        out1 = model(x)
        out2 = model(x)
        results["determinism_max_diff"] = (out1 - out2).abs().max().item()
        results["output_mean"] = out1.mean().item()
        results["output_std"] = out1.std().item()
        results["output_norm_mean"] = out1.norm(dim=-1).mean().item()

        # Robustness: progressive sensory dropout
        drop_rates = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
        robustness = {}
        base = out1.clone()
        for rate in drop_rates:
            if rate == 0.0:
                robustness[f"drop_{rate}"] = 1.0
                continue
            mask = (torch.rand(batch_size, input_dim) > rate).float()
            out_dropped = model(x * mask)
            corr = torch.corrcoef(
                torch.cat([base.flatten().unsqueeze(0),
                           out_dropped.flatten().unsqueeze(0)])
            )[0, 1].item()
            robustness[f"drop_{rate}"] = corr
        results["robustness"] = robustness

    return results


def rcq_evaluate_modularity(model, n_modules: int = 4) -> dict:
    """Measure cross-module interference in fly-inspired model."""
    model.eval()
    deterministic_seed(42)
    results = {}

    with torch.no_grad():
        x = torch.randn(16, 64)
        module_outputs = []
        for i in range(n_modules):
            module_outputs.append(model(x))

        correlations = []
        for i in range(n_modules):
            for j in range(i + 1, n_modules):
                corr = torch.corrcoef(
                    torch.cat([module_outputs[i].flatten().unsqueeze(0),
                               module_outputs[j].flatten().unsqueeze(0)])
                )[0, 1].item()
                correlations.append(corr)

        results["mean_cross_module_corr"] = float(np.mean(correlations))
        results["max_cross_module_corr"] = float(max(correlations))
        results["min_cross_module_corr"] = float(min(correlations))

    return results
