# Knapsack FPTAS

The 0/1 knapsack problem is NP-hard, so no polynomial-time exact algorithm
exists unless P = NP. However, we can get *arbitrarily close* to optimal in
polynomial time using a **Fully Polynomial-Time Approximation Scheme (FPTAS)**.
The key idea is elegantly simple: round item values to reduce the size of the
DP table, trading a small loss in accuracy for a large gain in speed.

## Problem Setup

Given $n$ items with weights $w_1, \dots, w_n$ and values $v_1, \dots, v_n$,
and a knapsack capacity $W$, find a subset $S \subseteq \{1, \dots, n\}$
maximizing $\sum_{i \in S} v_i$ subject to $\sum_{i \in S} w_i \le W$.

The exact dynamic programming solution uses a table indexed by value sums.
Let $v_{\max} = \max_i v_i$. The DP runs over scaled values, so the table
size depends on the magnitude of item values, giving pseudo-polynomial time
$O(n^2 v_{\max})$.

## The FPTAS Algorithm

**Intuition.** If we round all values down to multiples of some granularity
$K$, the DP table shrinks. Choosing $K$ carefully ensures the rounding error
stays within a $(1 - \epsilon)$ factor of optimal.

**Definition.** For a given accuracy parameter $\epsilon > 0$, define the
scaling factor

$$
K = \frac{\epsilon \cdot v_{\max}}{n}
$$

and the scaled values

$$
\hat{v}_i = \left\lfloor \frac{v_i}{K} \right\rfloor
$$

The FPTAS solves the knapsack instance with values $\hat{v}_i$ using exact DP,
then returns the corresponding subset of original items.

**Algorithm steps:**

1. Compute $v_{\max} = \max_i v_i$ and $K = \epsilon \cdot v_{\max} / n$
2. For each item $i$, set $\hat{v}_i = \lfloor v_i / K \rfloor$
3. Solve the exact knapsack DP with values $\hat{v}_i$ and weights $w_i$
4. Return the selected items (using original values)

## Approximation Guarantee

!!! tip "Theorem"
    The FPTAS returns a solution with value at least $(1 - \epsilon) \cdot
    \text{OPT}$.

**Proof.** Let $S^*$ be the optimal solution with value $\text{OPT} =
\sum_{i \in S^*} v_i$. Let $\hat{S}$ be the solution found by the FPTAS
using scaled values. Since $\hat{S}$ is optimal for the scaled instance,

$$
\sum_{i \in \hat{S}} \hat{v}_i \ge \sum_{i \in S^*} \hat{v}_i
$$

By the floor function, $\hat{v}_i \ge v_i / K - 1$ for each item, so

$$
\sum_{i \in S^*} \hat{v}_i \ge \sum_{i \in S^*} \frac{v_i}{K} - |S^*|
\ge \frac{\text{OPT}}{K} - n
$$

Since $\hat{S}$ is optimal for scaled values, the original-value sum satisfies

$$
\sum_{i \in \hat{S}} v_i \ge K \cdot \sum_{i \in \hat{S}} \hat{v}_i
\ge K \left(\frac{\text{OPT}}{K} - n\right)
= \text{OPT} - nK = \text{OPT} - \epsilon \cdot v_{\max}
$$

Because $v_{\max} \le \text{OPT}$, we get

$$
\sum_{i \in \hat{S}} v_i \ge \text{OPT} - \epsilon \cdot \text{OPT}
= (1 - \epsilon) \cdot \text{OPT} \qquad \blacksquare
$$

## Running Time

Each scaled value satisfies $\hat{v}_i \le v_i / K \le v_{\max} / K = n / \epsilon$.
The DP table has $n$ rows and at most $n \cdot (n / \epsilon) = n^2 / \epsilon$
columns, so the total running time is

$$
O\!\left(\frac{n^3}{\epsilon}\right)
$$

This is polynomial in both $n$ and $1/\epsilon$, which is precisely the
definition of an FPTAS.

## Implementation

```python
"""
Knapsack FPTAS: (1-epsilon)-approximation in O(n^3 / epsilon) time.
"""


# === Exact DP (value-indexed) ================================================

def knapsack_exact(W, weights, values):
    """Exact 0/1 knapsack via weight-indexed DP."""
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i - 1][w]
            if weights[i - 1] <= w:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    return dp[n][W]


# === FPTAS ====================================================================

def knapsack_fptas(W, weights, values, epsilon):
    """
    FPTAS for 0/1 knapsack.

    Returns a solution with value >= (1 - epsilon) * OPT.
    Runs in O(n^3 / epsilon) time.
    """
    n = len(weights)
    if n == 0:
        return 0, []

    v_max = max(values)
    K = (epsilon * v_max) / n

    # Scale values down
    scaled = [int(v // K) for v in values]
    V = sum(scaled)

    # DP: minimum weight to achieve each scaled-value sum
    INF = float("inf")
    dp = [INF] * (V + 1)
    dp[0] = 0
    parent = [[] for _ in range(V + 1)]

    for i in range(n):
        for v in range(V, scaled[i] - 1, -1):
            if dp[v - scaled[i]] + weights[i] <= W:
                new_w = dp[v - scaled[i]] + weights[i]
                if new_w < dp[v]:
                    dp[v] = new_w
                    parent[v] = parent[v - scaled[i]] + [i]

    # Find best achievable scaled value
    best_v = 0
    for v in range(V + 1):
        if dp[v] < INF:
            best_v = v

    selected = parent[best_v]
    total = sum(values[i] for i in selected)
    return total, selected


# === Demo =====================================================================

if __name__ == "__main__":
    W = 50
    weights = [10, 20, 30]
    values = [60, 100, 120]

    exact = knapsack_exact(W, weights, values)
    print(f"Exact DP:  {exact}")

    epsilon = 0.1
    approx, items = knapsack_fptas(W, weights, values, epsilon)
    print(f"FPTAS (ε={epsilon}): value={approx}, items={items}")
    print(f"Guarantee: >= {(1 - epsilon) * exact:.1f}")
```

**Output:**
```
Exact DP:  220
FPTAS (ε=0.1): value=220, items=[1, 2]
Guarantee: >= 198.0
```

## Why FPTAS Matters

| Property | Exact DP | FPTAS |
|---|---|---|
| Time | $O(nW)$ or $O(n \cdot v_{\max} \cdot n)$ | $O(n^3 / \epsilon)$ |
| Quality | Optimal | $\ge (1 - \epsilon) \cdot \text{OPT}$ |
| Type | Pseudo-polynomial | Fully polynomial |

The FPTAS is one of the strongest positive results in approximation theory:
it shows that although 0/1 knapsack is NP-hard, it is not hard to approximate.
Not all NP-hard problems admit an FPTAS — for example, the general TSP does not,
unless P = NP.

## Reference

- Vazirani, V. V. *Approximation Algorithms*. Springer, 2001. Chapter 8.
- Ibarra, O. H. and Kim, C. E. "Fast Approximation Algorithms for the
  Knapsack and Sum of Subset Problems." *JACM*, 1975.
