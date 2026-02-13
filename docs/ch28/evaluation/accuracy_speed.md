# Accuracy vs Speed Tradeoffs

## Overview

Neural ODEs offer a fundamental tradeoff: tighter solver tolerances give more accurate solutions but require more function evaluations (higher cost).

## Solver Tolerance

Adaptive solvers accept absolute and relative tolerance parameters:

```python
z_T = odeint(f, z_0, t, atol=1e-5, rtol=1e-5)  # default
z_T = odeint(f, z_0, t, atol=1e-3, rtol=1e-3)  # faster, less accurate
z_T = odeint(f, z_0, t, atol=1e-7, rtol=1e-7)  # slower, more accurate
```

| Tolerance | Typical NFE | Use Case |
|-----------|-----------|----------|
| 1e-3 | 20–50 | Training (rough gradients are OK) |
| 1e-5 | 50–150 | Default |
| 1e-7 | 100–500 | Evaluation, density estimation |

## Training vs Evaluation Tolerance

A practical strategy: use loose tolerances during training (faster iterations) and tight tolerances during evaluation (more accurate predictions). This works because the model learns dynamics that are approximately correct even with rough integration.

## Fixed-Step vs Adaptive

| Approach | Pros | Cons |
|----------|------|------|
| Adaptive (dopri5) | Accuracy guarantees, automatic | Variable cost, harder to batch |
| Fixed-step (Euler, RK4) | Predictable cost, easy batching | May be inaccurate, needs tuning |

For deployment, fixed-step solvers with a carefully chosen step count often provide better latency guarantees.

## Benchmarking

Always report: NFE (mean and std across test set), wall-clock time, tolerance settings, and solver choice. Different papers use different settings, making direct comparisons difficult.
