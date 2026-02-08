# 32.4.1 Policy Evaluation (Prediction)

## Overview

**Policy evaluation** (also called the **prediction problem**) computes the state value function $V_\pi$ for a given policy $\pi$. This is the first building block of dynamic programming methods.

## The Problem

Given a policy $\pi$ and MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$, find $V_\pi(s)$ for all $s \in \mathcal{S}$.

## Iterative Policy Evaluation

### Algorithm

Repeatedly apply the Bellman equation as an update rule:

$$V_{k+1}(s) = \sum_a \pi(a|s) \left[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s')\right] \quad \text{for all } s$$

Starting from any initial $V_0$ (typically all zeros), the sequence $\{V_k\}$ converges to $V_\pi$ as $k \to \infty$.

### Convergence

By the contraction mapping theorem, the Bellman operator $\mathcal{T}_\pi$ is a $\gamma$-contraction:

$$\|V_{k+1} - V_\pi\|_\infty \leq \gamma \|V_k - V_\pi\|_\infty \leq \gamma^{k+1} \|V_0 - V_\pi\|_\infty$$

The error decreases geometrically at rate $\gamma$.

### Pseudocode

```
Initialize V(s) = 0 for all s ∈ S
Repeat until convergence (Δ < θ):
    Δ = 0
    For each s ∈ S:
        v = V(s)
        V(s) = Σ_a π(a|s) [R(s,a) + γ Σ_s' P(s'|s,a) V(s')]
        Δ = max(Δ, |v - V(s)|)
Return V ≈ V_π
```

### Complexity

- **Time**: $O(|\mathcal{S}|^2 |\mathcal{A}|)$ per sweep, $O(\frac{1}{1-\gamma} \log \frac{1}{\epsilon})$ sweeps for $\epsilon$-accuracy
- **Space**: $O(|\mathcal{S}|)$ for the value function

## In-Place vs. Two-Array Updates

### Two-array (synchronous)

Maintain separate arrays $V_{\text{old}}$ and $V_{\text{new}}$. All updates use $V_{\text{old}}$, then swap.

### In-place (asynchronous)

Update $V(s)$ immediately, so later states in the sweep use already-updated values. In practice, in-place updates often converge faster.

## Direct Solution

For small MDPs, solve the linear system directly:

$$\mathbf{v}_\pi = (\mathbf{I} - \gamma \mathbf{P}_\pi)^{-1} \mathbf{r}_\pi$$

This has $O(|\mathcal{S}|^3)$ complexity but gives the exact answer in one step.

## Financial Application

Policy evaluation in portfolio management answers: "What is the expected risk-adjusted return of this specific trading strategy $\pi$ across all possible market states?"

This is essentially **backtesting** formalized in the MDP framework — but with the advantage of providing state-level value estimates rather than just aggregate performance.

## Summary

Policy evaluation is the foundation of dynamic programming. It computes $V_\pi$ via repeated application of the Bellman equation, with guaranteed convergence. Both iterative and direct methods are available, with the choice depending on the size of the state space.
