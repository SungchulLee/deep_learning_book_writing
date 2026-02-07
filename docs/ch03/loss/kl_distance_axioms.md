# KL Divergence and Distance Axioms

KL divergence is often informally called a "distance" between distributions, but it fails to satisfy the axioms of a metric. This section rigorously examines each metric axiom, proves which hold and which fail, and introduces symmetrized alternatives that do qualify as proper divergence measures.

## Metric Space Axioms

A function $d: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a **metric** (distance function) on a set $\mathcal{X}$ if for all $x, y, z \in \mathcal{X}$:

1. **Non-negativity:** $d(x, y) \geq 0$
2. **Identity of indiscernibles:** $d(x, y) = 0 \iff x = y$
3. **Symmetry:** $d(x, y) = d(y, x)$
4. **Triangle inequality:** $d(x, z) \leq d(x, y) + d(y, z)$

We now evaluate each axiom for $d(p, q) = D_{\text{KL}}(p \| q)$ over the space of probability distributions.

## Axiom 1: Non-Negativity ✓

**Claim:** $D_{\text{KL}}(p \| q) \geq 0$ for all distributions $p, q$.

**Proof.** This is Gibbs' inequality. By Jensen's inequality applied to the strictly concave function $\log$:

$$\begin{aligned}
-D_{\text{KL}}(p \| q) &= \sum_i p_i \log\frac{q_i}{p_i} \\
&\leq \log\sum_i p_i \cdot \frac{q_i}{p_i} \qquad\text{(Jensen's inequality)} \\
&= \log\sum_i q_i = \log 1 = 0
\end{aligned}$$

Therefore $D_{\text{KL}}(p \| q) \geq 0$. $\square$

## Axiom 2: Identity of Indiscernibles ✓

**Claim:** $D_{\text{KL}}(p \| q) = 0 \iff p = q$ (almost everywhere).

**Proof.** The forward direction ($p = q \Rightarrow D_{\text{KL}} = 0$) is immediate: if $p_i = q_i$ for all $i$, then $\log(p_i/q_i) = 0$ and the sum vanishes.

For the reverse direction ($D_{\text{KL}} = 0 \Rightarrow p = q$), note that equality in Jensen's inequality for a strictly concave function holds if and only if the random variable $q_i/p_i$ is constant $p$-almost surely. Since $\sum_i p_i(q_i/p_i) = \sum_i q_i = 1$ and $\sum_i p_i = 1$, the constant must be 1, giving $q_i = p_i$ for all $i$ where $p_i > 0$. $\square$

## Axiom 3: Symmetry ✗

**Claim:** $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ in general.

**Counterexample.** Let $p = (0.9, 0.1)$ and $q = (0.5, 0.5)$:

```python
import numpy as np

p = np.array([0.9, 0.1])
q = np.array([0.5, 0.5])

kl_pq = np.sum(p * np.log(p / q))
kl_qp = np.sum(q * np.log(q / p))

print(f"D_KL(p || q) = {kl_pq:.6f}")  # 0.368563
print(f"D_KL(q || p) = {kl_qp:.6f}")  # 0.510826
print(f"Difference:    {abs(kl_pq - kl_qp):.6f}")  # 0.142263
```

The asymmetry is not merely technical—it has practical consequences. $D_{\text{KL}}(p \| q)$ heavily penalizes regions where $p$ has mass but $q$ does not (since $\log(p_i/q_i) \to +\infty$ as $q_i \to 0$ while $p_i > 0$).

### Intuition for the Asymmetry

Consider a bimodal distribution $p$ and a unimodal approximation $q$:

- **Forward KL** $D_{\text{KL}}(p \| q)$: The expectation is under $p$, so we sample from both modes. If $q$ misses a mode, $\log(p/q)$ is very large there. So $q$ must cover both modes → **mode-covering**.

- **Reverse KL** $D_{\text{KL}}(q \| p)$: The expectation is under $q$, so we only care about regions where $q$ has mass. If $q$ concentrates on one mode, the penalty is low (as long as $p$ also has mass there). So $q$ locks onto one mode → **mode-seeking**.

## Axiom 4: Triangle Inequality ✗

**Claim:** There exist distributions $p, q, r$ such that $D_{\text{KL}}(p \| r) > D_{\text{KL}}(p \| q) + D_{\text{KL}}(q \| r)$.

**Counterexample.** Let $p = (0.1, 0.9)$, $q = (0.5, 0.5)$, $r = (0.9, 0.1)$:

```python
p = np.array([0.1, 0.9])
q = np.array([0.5, 0.5])
r = np.array([0.9, 0.1])

kl_pq = np.sum(p * np.log(p / q))
kl_qr = np.sum(q * np.log(q / r))
kl_pr = np.sum(p * np.log(p / r))

print(f"D_KL(p || q) = {kl_pq:.6f}")
print(f"D_KL(q || r) = {kl_qr:.6f}")
print(f"D_KL(p || r) = {kl_pr:.6f}")
print(f"D_KL(p||q) + D_KL(q||r) = {kl_pq + kl_qr:.6f}")
print(f"Triangle inequality holds: {kl_pr <= kl_pq + kl_qr}")
```

## Summary of Axiom Verification

| Axiom | Status | Comment |
|-------|--------|---------|
| Non-negativity | ✓ Satisfied | Via Gibbs' inequality (Jensen) |
| Identity of indiscernibles | ✓ Satisfied | Equality in Jensen iff $p = q$ a.e. |
| Symmetry | ✗ Violated | $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ in general |
| Triangle inequality | ✗ Violated | Counterexample with three binary distributions |

KL divergence satisfies 2 of 4 metric axioms and is therefore **not a metric**.

## Symmetrized Alternatives

Several constructions restore symmetry, producing divergences that are "more metric-like" than raw KL.

### Jensen-Shannon Divergence

The Jensen-Shannon divergence is the symmetrized, bounded version of KL divergence:

$$\text{JSD}(p \| q) = \frac{1}{2}D_{\text{KL}}\!\left(p \;\middle\|\; \frac{p+q}{2}\right) + \frac{1}{2}D_{\text{KL}}\!\left(q \;\middle\|\; \frac{p+q}{2}\right)$$

Properties:

- **Symmetric:** $\text{JSD}(p \| q) = \text{JSD}(q \| p)$ by construction
- **Bounded:** $0 \leq \text{JSD}(p \| q) \leq \log 2$ (using natural log)
- **Square root is a metric:** $\sqrt{\text{JSD}(p \| q)}$ satisfies the triangle inequality

JSD is used as the training objective in the original GAN formulation.

```python
def jsd(p, q):
    """Jensen-Shannon Divergence."""
    m = 0.5 * (p + q)
    return 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))

p = np.array([0.9, 0.1])
q = np.array([0.5, 0.5])

print(f"JSD(p || q) = {jsd(p, q):.6f}")
print(f"JSD(q || p) = {jsd(q, p):.6f}")  # Same value — symmetric!
```

### Jeffreys Divergence

The Jeffreys divergence is the arithmetic symmetrization:

$$D_J(p \| q) = D_{\text{KL}}(p \| q) + D_{\text{KL}}(q \| p)$$

This is symmetric by construction but is unbounded and does not satisfy the triangle inequality. It can be interpreted as the expected log-likelihood ratio taken under both distributions.

```python
def jeffreys(p, q):
    """Jeffreys divergence."""
    return np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p))

print(f"Jeffreys(p, q) = {jeffreys(p, q):.6f}")
print(f"Jeffreys(q, p) = {jeffreys(q, p):.6f}")  # Same — symmetric
```

### Comparison

| Divergence | Symmetric | Bounded | Triangle Ineq. | Metric |
|------------|-----------|---------|----------------|--------|
| $D_{\text{KL}}(p \| q)$ | No | No | No | No |
| $D_J(p, q) = D_{\text{KL}}(p\|q) + D_{\text{KL}}(q\|p)$ | Yes | No | No | No |
| $\text{JSD}(p, q)$ | Yes | Yes ($\leq \log 2$) | No | No |
| $\sqrt{\text{JSD}(p, q)}$ | Yes | Yes | Yes | **Yes** |

## When Does the Non-Metric Nature Matter?

**Optimization:** For optimization-based applications (training neural networks, variational inference), the non-metric nature of KL divergence is usually irrelevant. Gradient descent only requires a differentiable objective, not a metric.

**Comparison:** When using divergences to compare or cluster distributions (e.g., "which of $q_1, q_2$ is closer to $p$?"), asymmetry means the answer depends on the direction. Forward KL gives different rankings than reverse KL.

**Geometric reasoning:** Metric properties enable geometric intuitions (e.g., "if $p$ is close to $q$ and $q$ is close to $r$, then $p$ is close to $r$"). The triangle inequality violation means such reasoning fails for KL divergence. However, locally (near $p = q$), KL divergence behaves like a quadratic form via the Fisher information matrix, restoring local metric-like behavior (see [KL and Fisher Information](kl_fisher_information.md)).

## Exercises

### Exercise 1: Maximum Asymmetry

Find the pair of binary distributions $p = (\alpha, 1-\alpha)$ and $q = (\beta, 1-\beta)$ on the simplex that maximizes $|D_{\text{KL}}(p\|q) - D_{\text{KL}}(q\|p)|$. What happens as one distribution approaches the boundary of the simplex?

### Exercise 2: Triangle Inequality Violation Magnitude

For the counterexample $p = (0.1, 0.9)$, $q = (0.5, 0.5)$, $r = (0.9, 0.1)$, compute the ratio $D_{\text{KL}}(p\|r) / (D_{\text{KL}}(p\|q) + D_{\text{KL}}(q\|r))$. How large can this ratio become?

### Exercise 3: JSD as a Metric

Verify numerically that $\sqrt{\text{JSD}}$ satisfies the triangle inequality for the distributions $p, q, r$ above. Then argue informally why the square root is needed (hint: consider the analogy with squared Euclidean distance).

## Key Takeaways

KL divergence satisfies non-negativity and identity of indiscernibles but violates symmetry and the triangle inequality, making it a divergence but not a metric. The asymmetry is not a defect but a feature: forward and reverse KL encode fundamentally different optimization objectives (mode-covering vs mode-seeking). When a true metric is needed, the Jensen-Shannon divergence (specifically $\sqrt{\text{JSD}}$) provides a symmetric, bounded, triangle-inequality-satisfying alternative. For most optimization applications in deep learning, the non-metric nature of KL divergence is irrelevant—what matters is that it is non-negative, differentiable, and zero if and only if the distributions match.
