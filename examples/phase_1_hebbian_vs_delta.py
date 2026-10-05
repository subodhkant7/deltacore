"""Comparison of Hebbian outer-product vs. Delta-rule error-correcting updates.

Demonstrates the role of residual error correction in associative memory recall
under non-orthogonal key vectors.

When keys are not mutually orthogonal (i.e. k_i^T k_j != 0 for i != j), naive
Hebbian correlation updates induce severe cross-talk interference because the
previously stored memory content is not subtracted during state transition. In
contrast, the Delta rule calculates the prediction error and corrects the residual,
converging toward minimum mean-squared retrieval error.
"""

import sys
from pathlib import Path

# Ensure repository root is on sys.path for standalone script execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch  # noqa: E402

from deltacore.memory.associative import AssociativeMemory  # noqa: E402
from deltacore.updates.delta import DeltaRule  # noqa: E402
from deltacore.updates.hebbian import HebbianRule  # noqa: E402


def run_comparison():
    torch.manual_seed(42)

    # 1. Define Synthetic Associative Memory Task
    # Key dimension K = 4, Value dimension V = 3
    k_dim = 4
    v_dim = 3

    # Define 3 non-orthogonal key vectors (normalized to unit length)
    k_a = torch.tensor([1.0, 0.5, 0.0, 0.0], dtype=torch.float64)
    k_b = torch.tensor([0.4, 1.0, 0.3, 0.0], dtype=torch.float64)
    k_c = torch.tensor([0.1, 0.4, 1.0, 0.2], dtype=torch.float64)

    k_a = k_a / torch.linalg.norm(k_a)
    k_b = k_b / torch.linalg.norm(k_b)
    k_c = k_c / torch.linalg.norm(k_c)

    # Compute key overlap (cross-talk cosine similarities)
    dot_ab = torch.dot(k_a, k_b).item()
    dot_ac = torch.dot(k_a, k_c).item()
    dot_bc = torch.dot(k_b, k_c).item()

    # Define target value vectors
    v_a = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
    v_b = torch.tensor([0.0, 2.0, 1.0], dtype=torch.float64)
    v_c = torch.tensor([-1.0, 1.0, 0.5], dtype=torch.float64)

    associations = [
        ("A", k_a, v_a),
        ("B", k_b, v_b),
        ("C", k_c, v_c),
    ]

    print("=" * 70)
    print("DeltaCore: Hebbian vs Delta-Rule Update Comparison")
    print("=" * 70)
    print(f"Key Dimension K = {k_dim}, Value Dimension V = {v_dim}")
    print("Key Cross-Talk Dot Products (cosine similarities):")
    print(
        f"  <k_A, k_B> = {dot_ab:.4f}  |  <k_A, k_C> = {dot_ac:.4f}  |  <k_B, k_C> = {dot_bc:.4f}"
    )
    print("=" * 70)

    def evaluate_memory(mem: AssociativeMemory, label: str):
        errors = {}
        total_se = 0.0
        for name, k, v in associations:
            pred = mem.read(k)
            err = v - pred
            se = torch.sum(err**2).item()
            total_se += se
            errors[name] = torch.linalg.norm(err).item()
        mse = total_se / len(associations)
        return errors, mse

    # Evaluate Initial Memory (Zeros)
    init_mem = AssociativeMemory.zeros(v_dim=v_dim, k_dim=k_dim, dtype=torch.float64)
    initial_errors, initial_mse = evaluate_memory(init_mem, "Initial")
    print("Initial State (Zero Memory):")
    for name in ("A", "B", "C"):
        print(f"  Retrieval Error ||v - v_hat|| [{name}]: {initial_errors[name]:.4f}")
    print(f"  Initial Mean Squared Error: {initial_mse:.4f}\n")

    # 2. Sequential Training Setup
    epochs = 20
    eta_hebbian = 0.2
    eta_delta = 0.2

    hebbian_rule = HebbianRule(step_size=eta_hebbian)
    delta_rule = DeltaRule(step_size=eta_delta)

    mem_hebbian = init_mem.clone()
    mem_delta = init_mem.clone()

    total_updates = 0

    for _ in range(1, epochs + 1):
        for _, k, v in associations:
            mem_hebbian = hebbian_rule.update(mem_hebbian, k, v)
            mem_delta = delta_rule.update(mem_delta, k, v)
            total_updates += 1

    # 3. Final Evaluation
    hebbian_errors, hebbian_mse = evaluate_memory(mem_hebbian, "Hebbian")
    delta_errors, delta_mse = evaluate_memory(mem_delta, "Delta")

    print(
        f"Results after {epochs} passes over 3 associations, for {total_updates} total updates:"
    )
    print("-" * 70)
    print(
        f"{'Metric':<30} | {'Hebbian Rule':<16} | {'Delta Rule':<16} | {'Difference':<12}"
    )
    print("-" * 70)
    for name in ("A", "B", "C"):
        diff = hebbian_errors[name] - delta_errors[name]
        print(
            f"Error ||v - v_hat|| [{name}]:{'':<7} | {hebbian_errors[name]:<16.4f} | {delta_errors[name]:<16.4f} | {diff:<+12.4f}"
        )
    mse_diff = hebbian_mse - delta_mse
    print("-" * 70)
    print(
        f"{'Mean Squared Error (MSE)':<30} | {hebbian_mse:<16.4f} | {delta_mse:<16.4f} | {mse_diff:<+12.4f}"
    )
    print("=" * 70)
    print("Observation:")
    print("  Due to non-zero dot products between keys, the Hebbian update accumulates")
    print("  cross-talk interference, plateauing at a high retrieval error.")
    print("  The Delta rule corrects prediction residuals at every step, driving")
    print(f"  the MSE down from {initial_mse:.4f} to {delta_mse:.6f}.")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison()
