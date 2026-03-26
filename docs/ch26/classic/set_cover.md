# Set Cover Approximation

The **Set Cover** problem is one of the most fundamental NP-hard optimization
problems. Despite its hardness, a simple greedy algorithm achieves an
$O(\ln n)$-approximation ratio — and this is essentially the best any
polynomial-time algorithm can do.

## Problem Definition

Given a universe $U = \{1, 2, \dots, n\}$ and a collection of sets
$\mathcal{S} = \{S_1, S_2, \dots, S_m\}$ where each $S_j \subseteq U$ and
each set has a cost $c_j > 0$, find a subcollection
$\mathcal{C} \subseteq \mathcal{S}$ such that

$$
\bigcup_{S_j \in \mathcal{C}} S_j = U
$$

and the total cost $\sum_{S_j \in \mathcal{C}} c_j$ is minimized.

## Greedy Algorithm

**Intuition.** At each step, pick the set with the lowest *cost per new
element covered*. This cost-effectiveness criterion ensures no set is chosen
that is wasteful relative to what it contributes.

**Algorithm:**

1. Initialize $R \leftarrow U$ (remaining uncovered elements)
2. While $R \neq \emptyset$:
    - For each $S_j$, compute the cost-effectiveness $\frac{c_j}{|S_j \cap R|}$
    - Select $S_j$ minimizing this ratio
    - Add $S_j$ to $\mathcal{C}$; update $R \leftarrow R \setminus S_j$
3. Return $\mathcal{C}$

## Worked Example

Consider $U = \{1, 2, 3, 4, 5, 6\}$ with unit-cost sets:
$S_1 = \{1, 2, 3\}$, $S_2 = \{2, 4, 5\}$, $S_3 = \{3, 5, 6\}$, $S_4 = \{1, 4, 6\}$.

| Step | $R$ | Best set | Cost-effectiveness | Chosen |
|------|-----|----------|-------------------|--------|
| 1 | $\{1,2,3,4,5,6\}$ | All cover 3 elements at cost 1 | $1/3$ | $S_1$ |
| 2 | $\{4,5,6\}$ | $S_2$: 2 new, $S_3$: 2 new, $S_4$: 2 new | $1/2$ | $S_2$ |
| 3 | $\{6\}$ | $S_3$: 1 new, $S_4$: 1 new | $1/1$ | $S_3$ |

Greedy selects $\{S_1, S_2, S_3\}$ with total cost $3$. The optimal solution
is $\{S_1, S_2, S_3\}$ or $\{S_2, S_4, S_1\}$, also cost $3$. In this case
the greedy solution is optimal.

## Approximation Guarantee

!!! tip "Theorem"
    The greedy algorithm achieves an approximation ratio of $H_n$, where
    $H_n = \sum_{k=1}^n \frac{1}{k} \le \ln n + 1$ is the $n$-th harmonic
    number.

**Proof.** Assign a *price* to each element $e$ as it gets covered: when
set $S_j$ is chosen covering $k$ new elements, each new element pays
$c_j / k$.

Let $\text{OPT}$ be the optimal cost. At each step, the remaining elements
can be covered at total cost at most $\text{OPT}$ (since the optimal solution
covers everything). If $|R|$ elements remain, some set in the optimal solution
covers at least $|R| / |\mathcal{S}^*|$ of them at cost-effectiveness at most
$\text{OPT} / |R|$. The greedy choice is at least as good.

More precisely, let $n_t$ be the number of uncovered elements after step $t$.
Setting $n_0 = n$, at step $t + 1$ the greedy choice has cost-effectiveness at
most $\text{OPT} / n_t$. The total greedy cost is

$$
\sum_{t=0}^{T-1} \frac{\text{OPT}}{n_t} \cdot (n_t - n_{t+1})
\le \text{OPT} \sum_{k=1}^{n} \frac{1}{k} = H_n \cdot \text{OPT}
$$

The inequality follows because covering $n_t - n_{t+1}$ elements at price
$\text{OPT}/n_t$ each is at most $\text{OPT} \cdot \sum_{j=n_{t+1}+1}^{n_t} 1/j$.
Summing telescopically gives $H_n$. $\square$

## Inapproximability

Dinur and Steurer (2014) showed that Set Cover cannot be approximated within
$(1 - \epsilon) \ln n$ for any $\epsilon > 0$ unless P = NP. The greedy
algorithm is therefore essentially optimal.

## Implementation

```python
"""
Set Cover: greedy H_n-approximation algorithm.
"""


# === Greedy Set Cover =========================================================

def greedy_set_cover(universe, sets, costs):
    """
    Greedy set cover with cost-effectiveness criterion.

    Args:
        universe: set of elements to cover.
        sets: list of sets.
        costs: list of costs for each set.

    Returns:
        (total_cost, selected_indices).
    Approximation ratio: H_n = O(ln n).
    """
    remaining = set(universe)
    selected = []
    total_cost = 0.0

    while remaining:
        # Find most cost-effective set
        best_idx = -1
        best_ratio = float("inf")
        for j, s in enumerate(sets):
            new_covered = len(s & remaining)
            if new_covered > 0:
                ratio = costs[j] / new_covered
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_idx = j

        if best_idx == -1:
            break  # Cannot cover remaining elements

        selected.append(best_idx)
        total_cost += costs[best_idx]
        remaining -= sets[best_idx]

    return total_cost, selected


# === Harmonic number ==========================================================

def harmonic(n):
    """Compute H_n = 1 + 1/2 + ... + 1/n."""
    return sum(1.0 / k for k in range(1, n + 1))


# === Demo =====================================================================

if __name__ == "__main__":
    universe = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    sets = [
        {1, 2, 3, 4},
        {3, 4, 5, 6},
        {5, 6, 7, 8},
        {7, 8, 9, 10},
        {1, 5, 9},
        {2, 6, 10},
    ]
    costs = [4, 4, 4, 4, 3, 3]

    cost, selected = greedy_set_cover(universe, sets, costs)
    n = len(universe)
    print(f"Selected sets: {selected}")
    print(f"Total cost:    {cost}")
    print(f"H_{n} = {harmonic(n):.3f}")
    print(f"Guarantee:     <= {harmonic(n) * 8:.1f} (if OPT=8)")
```

## Summary

| Property | Value |
|---|---|
| Approximation ratio | $H_n \le \ln n + 1$ |
| Time complexity | $O(n \cdot m)$ per iteration, $O(n^2 m)$ total |
| Tight? | Yes — cannot do better than $(1 - \epsilon)\ln n$ |

## Reference

- Chvatal, V. "A Greedy Heuristic for the Set-Covering Problem." *Math. of
  Operations Research*, 1979.
- Dinur, I. and Steurer, D. "Analytical Approach to Parallel Repetition."
  *STOC*, 2014.
- Vazirani, V. V. *Approximation Algorithms*. Springer, 2001. Chapter 2.
