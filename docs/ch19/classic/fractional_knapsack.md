# Fractional Knapsack

The knapsack problem asks: given a set of items, each with a weight and a value, which items should a thief put in a knapsack of limited capacity to maximize total value? In the **fractional** variant, the thief may take fractions of items --- for example, pouring half a bag of gold dust into the sack. This seemingly small relaxation changes the problem fundamentally: while the 0-1 knapsack requires dynamic programming (and is NP-hard), the fractional knapsack admits an elegant $O(n \log n)$ greedy solution.

## Problem Statement

**Input.**

- $n$ items, each with weight $w_i > 0$ and value $v_i > 0$.
- Knapsack capacity $W > 0$.

**Decision variable.** For each item $i$, choose a fraction $x_i \in [0, 1]$ to take.

**Objective.** Maximize total value:

$$
\max \sum_{i=1}^{n} v_i \cdot x_i
$$

**Constraint.** Total weight must not exceed capacity:

$$
\sum_{i=1}^{n} w_i \cdot x_i \leq W
$$

## Why Greedy Works Here

The key insight is the **value-to-weight ratio** $r_i = v_i / w_i$. Each unit of weight from item $i$ contributes $r_i$ to the total value. Since fractions are allowed, the thief should prioritize items with the highest ratio --- filling the knapsack with the most valuable "density" first.

This works because:

1. Taking a fraction of a high-ratio item is always better than taking the same weight from a low-ratio item.
2. There are no indivisibility constraints that could make a partial fill suboptimal.

## Greedy Algorithm

!!! note "Fractional Knapsack Algorithm"
    1. Compute the value-to-weight ratio $r_i = v_i / w_i$ for each item.
    2. Sort items in decreasing order of $r_i$.
    3. For each item (in sorted order):
        - If the entire item fits, take it all ($x_i = 1$).
        - Otherwise, take the fraction that fills the remaining capacity ($x_i = (W_{\text{remaining}}) / w_i$) and stop.

## Worked Example

**Capacity:** $W = 50$.

| Item | $w_i$ | $v_i$ | $r_i = v_i/w_i$ |
|------|--------|--------|------------------|
| A    | 10     | 60     | 6.0              |
| B    | 20     | 100    | 5.0              |
| C    | 30     | 120    | 4.0              |

**Sorted by ratio:** A (6.0), B (5.0), C (4.0).

**Greedy execution:**

1. Take all of A: weight used = 10, value = 60, remaining capacity = 40.
2. Take all of B: weight used = 30, value = 160, remaining capacity = 20.
3. Take $20/30 = 2/3$ of C: weight used = 50, value = $160 + 80 = 240$.

$$
\text{Total value} = 60 + 100 + \frac{2}{3} \cdot 120 = 240
$$

**Comparison with 0-1 knapsack:** The 0-1 optimal is $v_B + v_C = 220$ (take B and C entirely). The fractional solution achieves a higher value of 240 by splitting item C.

## Correctness Proof

**Theorem.** The greedy algorithm produces an optimal solution to the fractional knapsack problem.

**Proof.** Without loss of generality, assume items are sorted so that $r_1 \geq r_2 \geq \cdots \geq r_n$. Let $G = (x_1^G, \ldots, x_n^G)$ be the greedy solution and $S^* = (x_1^*, \ldots, x_n^*)$ be any optimal solution.

Suppose $G \neq S^*$. Let $j$ be the first index where they differ: $x_j^G \neq x_j^*$.

**Case 1:** $x_j^G > x_j^*$ (greedy takes more of item $j$). Since the greedy algorithm takes as much of item $j$ as possible before moving to item $j+1$, the remaining capacity in $S^*$ allocated to items $j, j+1, \ldots, n$ differs from $G$.

Construct $S'$ by increasing $x_j$ from $x_j^*$ toward $x_j^G$ by some amount $\delta$, and decreasing later items to maintain the capacity constraint. The change in value is:

$$
\Delta = \delta \cdot w_j \cdot r_j - \sum_{k > j} \delta_k \cdot w_k \cdot r_k
$$

Since $r_j \geq r_k$ for all $k > j$ and $\delta \cdot w_j = \sum_{k>j} \delta_k \cdot w_k$ (weight balance), we have $\Delta \geq 0$. So $S'$ is at least as good as $S^*$ and agrees with $G$ on one more item.

Repeating this process transforms $S^*$ into $G$ without decreasing value. $\square$

## Python Implementation

```python
"""
Fractional knapsack solved by the greedy value-to-weight ratio strategy.

Unlike the 0-1 knapsack (which requires dynamic programming), the fractional
variant allows taking fractions of items and admits an O(n log n) greedy solution.
"""


# === Greedy Fractional Knapsack ===

def fractional_knapsack(capacity, items):
    """Solve the fractional knapsack problem.

    Args:
        capacity: maximum weight the knapsack can hold
        items: list of (weight, value) tuples

    Returns:
        Tuple of (max_value, fractions) where fractions[i] is the
        fraction of item i taken
    """
    n = len(items)
    # Compute ratios and sort by decreasing ratio
    indexed = [(v / w, w, v, i) for i, (w, v) in enumerate(items)]
    indexed.sort(reverse=True)

    fractions = [0.0] * n
    total_value = 0.0
    remaining = capacity

    for ratio, weight, value, idx in indexed:
        if remaining <= 0:
            break
        if weight <= remaining:
            # Take the entire item
            fractions[idx] = 1.0
            total_value += value
            remaining -= weight
        else:
            # Take a fraction
            fraction = remaining / weight
            fractions[idx] = fraction
            total_value += value * fraction
            remaining = 0

    return total_value, fractions


if __name__ == "__main__":
    # Example: (weight, value)
    items = [(10, 60), (20, 100), (30, 120)]
    capacity = 50

    max_val, fracs = fractional_knapsack(capacity, items)

    print("Fractional Knapsack Solution:")
    print(f"Capacity: {capacity}")
    print(f"{'Item':>5} {'Weight':>7} {'Value':>7} {'Ratio':>7} {'Taken':>7}")
    print("-" * 36)
    for i, (w, v) in enumerate(items):
        print(f"{i+1:>5} {w:>7} {v:>7} {v/w:>7.1f} {fracs[i]:>7.3f}")
    print(f"\nMaximum value: {max_val}")
```

**Output:**
```
Fractional Knapsack Solution:
Capacity: 50
 Item  Weight   Value   Ratio   Taken
------------------------------------
    1      10      60     6.0   1.000
    2      20     100     5.0   1.000
    3      30     120     4.0   0.667

Maximum value: 240.0
```

## Complexity Analysis

- **Computing ratios:** $O(n)$.
- **Sorting:** $O(n \log n)$.
- **Filling the knapsack:** $O(n)$.
- **Total:** $O(n \log n)$.

**Space:** $O(n)$ for the fractions array.

## Contrast: Fractional vs 0-1 Knapsack

| Property | Fractional Knapsack | 0-1 Knapsack |
|----------|---------------------|--------------|
| Item splitting | Allowed | Not allowed |
| Algorithm | Greedy (by ratio) | Dynamic programming |
| Time complexity | $O(n \log n)$ | $O(nW)$ (pseudo-polynomial) |
| Greedy choice property | Holds | Does not hold |
| NP-hard | No | Yes |

The fractional variant always achieves a value at least as high as the 0-1 variant, since every 0-1 solution is a feasible fractional solution.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 16.2. MIT Press.
- Kleinberg, J. & Tardos, E. (2006). *Algorithm Design*, Chapter 4. Pearson.
