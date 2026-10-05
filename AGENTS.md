# Engineering Rules for AI & Human Contributors (AGENTS.md)

This document establishes binding engineering instructions and constraints for any autonomous coding agent, AI assistant, or human engineer contributing to **DeltaCore**.

---

## 1. Prime Directives

1. **Read Relevant Documentation Before Modifying Code**:
   - Before writing or editing any code, read [PROJECT_CONSTITUTION.md](file:///Users/urjasoft/Documents/DeltaCore/docs/PROJECT_CONSTITUTION.md), [ARCHITECTURE.md](file:///Users/urjasoft/Documents/DeltaCore/docs/ARCHITECTURE.md), and [ROADMAP.md](file:///Users/urjasoft/Documents/DeltaCore/docs/ROADMAP.md).
   - Never implement features beyond the scope of the current active phase.

2. **Never Silently Change Mathematical Definitions**:
   - Tensor dimensions, sign conventions, normalization constants, and state transition equations must be documented explicitly.
   - If an equation in code differs in any detail from a cited paper or reference formulation, you must explicitly document the discrepancy in code docstrings and research notes.

3. **Write Tests Alongside Implementations**:
   - No code is accepted without accompanying unit tests.
   - For mathematical primitives, test both forward outputs (shapes, invariants, numerical edge cases) and backward gradients where applicable.

4. **Prefer Small, Composable Modules**:
   - Keep classes and functions focused on a single responsibility.
   - Do not fuse representation, state container, update rule, and normalization into monolithic classes.

5. **Avoid Hidden Global State**:
   - Pass configuration and states explicitly.
   - Do not rely on module-level mutable variables, global state singletons, or implicit environment variables.

6. **Make Randomness Explicit**:
   - Every randomized operation, tensor initialization, or synthetic benchmark must accept an explicit `torch.Generator` or `seed: int`.
   - Never rely on implicit global PRNG states for deterministic experiments.

7. **Record Numerical Assumptions**:
   - State expected tensor shapes, precision requirements (e.g., FP32 vs FP64), condition numbers, and valid parameter ranges (e.g., $\lambda \in [0, 1]$) in docstrings and type annotations.
   - Enforce shape and domain checks at module boundaries with clear descriptive errors.

8. **Do Not Call an Implementation "Stable" Unless Stability Has Been Tested**:
   - The term "stable" is a technical numerical claim, not marketing language.
   - An operator or layer may only be documented as numerically stable if it has passed rigorous automated stress testing (long sequences, extreme eigenvalues, near-zero denominators).

9. **Do Not Claim Equivalence with a Paper Unless Verified**:
   - If implementing an algorithm inspired by literature (e.g., HOPE, DeltaNet, TTT, VisionHOPE), state clearly: *"Inspired by [Paper], implemented according to [DeltaCore Specification]"*.
   - Never assert that an implementation is mathematically or empirically equivalent to a cited paper without verifiable reproduction benchmarks.

10. **Do Not Optimize Prematurely**:
    - Prioritize readable, correct, pure-PyTorch mathematical implementations first.
    - Custom kernels (CUDA, Triton, C++) or memory-layout micro-optimizations may only be introduced after reference implementations have passed full numerical verification and profiling confirms a computational bottleneck.

11. **Keep Public APIs Small**:
    - Expose only essential classes and functions in package `__all__` definitions.
    - Treat everything else as internal (`_private`). Small APIs are easier to test, maintain, and keep stable.

12. **Preserve Backwards Compatibility Once an API is Released**:
    - Avoid breaking changes to established public interfaces across minor versions.
    - If a signature change is unavoidable, provide a deprecation warning and migration pathway.

13. **Every Phase Must Leave the Repository in a Runnable State**:
    - Every commit and phase boundary must pass all existing tests cleanly.
    - The repository must never be left in a broken, half-implemented, or un-importable state.
