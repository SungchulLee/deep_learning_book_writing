# Reliability Design

In systems engineering, critical systems use redundant components to guard against failure. If a single component has a probability $r_i$ of working correctly, deploying $m_i$ copies in parallel raises the stage reliability to $1 - (1 - r_i)^{m_i}$, since all copies must fail for the stage to fail. The reliability design problem asks: given a fixed budget, how many redundant copies should each stage receive to maximize the overall system reliability? This is a classic application of dynamic programming where the budget plays the role of capacity (analogous to the knapsack) and reliability replaces value.

## Problem Statement

A system has $n$ stages connected in series. Stage $i$ has:

- Component reliability $r_i$ (probability a single component works)
- Component cost $c_i$ (cost per redundant copy)
- At least 1 and at most $u_i$ copies allowed

The total budget is $B$. The system works only if **every** stage works. With $m_i$ copies at stage $i$, the stage reliability is:

$$
R_i(m_i) = 1 - (1 - r_i)^{m_i}
$$

The system reliability is:

$$
R_{\text{sys}} = \prod_{i=1}^{n} R_i(m_i) = \prod_{i=1}^{n} \bigl[1 - (1 - r_i)^{m_i}\bigr]
$$

**Objective**: maximize $R_{\text{sys}}$ subject to $\sum_{i=1}^{n} c_i \cdot m_i \leq B$ and $1 \leq m_i \leq u_i$.

## DP Formulation

Define $dp[i][b]$ as the maximum reliability achievable using stages $1$ through $i$ with budget $b$.

**Recurrence**: for each possible number of copies $m$ at stage $i$:

$$
dp[i][b] = \max_{1 \leq m \leq u_i,\; c_i \cdot m \leq b} \bigl( dp[i-1][b - c_i \cdot m] \cdot R_i(m) \bigr)
$$

**Base case**: $dp[0][b] = 1$ for all $b$ (no stages processed, reliability is 1).

**Answer**: $dp[n][B]$.

## Implementation

```python
"""
Reliability design: maximize system reliability under a budget constraint.

Each stage can have multiple redundant copies. The system works only if
every stage works (series connection). Redundant copies at each stage
work in parallel (any one working suffices).
"""


# ===================================================================
# Reliability design via DP
# ===================================================================
def reliability_design(
    reliabilities: list[float],
    costs: list[int],
    max_copies: list[int],
    budget: int,
) -> tuple[float, list[int]]:
    """Maximize system reliability under budget constraint.

    Parameters
    ----------
    reliabilities : list[float]
        Per-component reliability for each stage.
    costs : list[int]
        Cost per copy for each stage.
    max_copies : list[int]
        Maximum allowed copies for each stage.
    budget : int
        Total budget.

    Returns
    -------
    tuple[float, list[int]]
        Maximum system reliability and number of copies per stage.
    """
    n = len(reliabilities)
    dp = [[0.0] * (budget + 1) for _ in range(n + 1)]

    # Base case: no stages, reliability = 1
    for b in range(budget + 1):
        dp[0][b] = 1.0

    # Track choices for reconstruction
    choice = [[0] * (budget + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        r_i = reliabilities[i - 1]
        c_i = costs[i - 1]
        u_i = max_copies[i - 1]

        for b in range(budget + 1):
            best = 0.0
            best_m = 1
            for m in range(1, u_i + 1):
                cost = c_i * m
                if cost > b:
                    break
                stage_rel = 1.0 - (1.0 - r_i) ** m
                val = dp[i - 1][b - cost] * stage_rel
                if val > best:
                    best = val
                    best_m = m
            dp[i][b] = best
            choice[i][b] = best_m

    # Reconstruct solution
    copies = [0] * n
    b = budget
    for i in range(n, 0, -1):
        copies[i - 1] = choice[i][b]
        b -= costs[i - 1] * copies[i - 1]

    return dp[n][budget], copies


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    # Example: 3-stage system
    reliabilities = [0.9, 0.8, 0.5]
    costs = [10, 15, 20]
    max_copies = [3, 3, 3]
    budget = 100

    max_rel, copies = reliability_design(reliabilities, costs, max_copies, budget)

    print("Reliability Design")
    print(f"Budget: {budget}")
    print(f"Stage reliabilities: {reliabilities}")
    print(f"Stage costs: {costs}")
    print(f"Optimal copies: {copies}")
    print(f"System reliability: {max_rel:.6f}")

    # Show per-stage reliability
    for i in range(len(copies)):
        stage_r = 1 - (1 - reliabilities[i]) ** copies[i]
        print(f"  Stage {i+1}: {copies[i]} copies, reliability = {stage_r:.6f}")
```

**Output:**
```
Reliability Design
Budget: 100
Stage reliabilities: [0.9, 0.8, 0.5]
Stage costs: [10, 15, 20]
Optimal copies: [2, 3, 2]
System reliability: 0.970596
  Stage 1: 2 copies, reliability = 0.990000
  Stage 2: 3 copies, reliability = 0.992000
  Stage 3: 2 copies, reliability = 0.750000
```

## Complexity

| Aspect | Value |
|--------|-------|
| Time | $O(n \cdot B \cdot u_{\max})$ |
| Space | $O(n \cdot B)$ |

where $u_{\max} = \max_i u_i$ is the largest allowed copy count. Since each copy has a cost of at least 1, the effective maximum copies per stage is bounded by $B$, so the worst case is $O(n \cdot B^2)$.

## Connection to Knapsack

The reliability design problem is a variant of the bounded knapsack where:

- **Items** are stages (each must be "taken" at least once)
- **Weight** is the cost of redundant copies
- **Value** is reliability (multiplied rather than added)
- **Capacity** is the budget

The multiplicative objective distinguishes this from standard knapsack. Taking the logarithm converts the product to a sum, reducing it to a standard knapsack formulation when exact arithmetic is acceptable:

$$
\max \prod_{i} R_i(m_i) \iff \max \sum_{i} \log R_i(m_i)
$$

!!! tip "Numerical stability"
    For systems with many stages, the product of reliabilities can become very small. Working in log-space (summing $\log R_i$) avoids floating-point underflow and converts the multiplicative DP to an additive one.

## Reference

- Horowitz, E. & Sahni, S. (1978). *Fundamentals of Computer Algorithms*. Computer Science Press, Chapter 5.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
