# Fractional Knapsack via Dynamic Programming

The classical 0/1 knapsack problem restricts each item to an all-or-nothing choice. The **fractional knapsack** variant relaxes this constraint: you may take any fraction of an item. While the fractional knapsack is optimally solved by a greedy algorithm in $O(n \log n)$, studying it alongside the 0/1 variant highlights when dynamic programming is necessary and when a simpler strategy suffices.

## Problem Formulation

Given $n$ items, each with weight $w_i > 0$ and value $v_i > 0$, and a knapsack of capacity $W$, choose fractions $x_i \in [0, 1]$ to maximize total value subject to the weight constraint:

$$
\max \sum_{i=1}^{n} v_i \, x_i \quad \text{subject to} \quad \sum_{i=1}^{n} w_i \, x_i \le W, \quad 0 \le x_i \le 1
$$

The key quantity is the **value-to-weight ratio** (or value density) of each item:

$$
r_i = \frac{v_i}{w_i}
$$

Items with higher $r_i$ deliver more value per unit of weight.

## Greedy Algorithm

The greedy strategy sorts items by decreasing value density $r_i$ and greedily fills the knapsack.

**Algorithm:**

1. Compute $r_i = v_i / w_i$ for each item $i$.
2. Sort items so that $r_1 \ge r_2 \ge \cdots \ge r_n$.
3. Initialize remaining capacity $C = W$.
4. For each item $i$ in sorted order:
    - If $w_i \le C$: take the entire item ($x_i = 1$), set $C \leftarrow C - w_i$.
    - Else: take the fraction $x_i = C / w_i$, set $C = 0$, and stop.

## Proof of Optimality

!!! tip "Greedy Choice Property"
    At each step, including the item with the highest remaining value density in the solution is consistent with an optimal solution.

**Proof by exchange argument.** Let $\mathbf{x}^*$ be an optimal solution, and let item $j$ be the first item (in sorted order by $r_i$) where the greedy solution $\mathbf{x}^G$ differs from $\mathbf{x}^*$. By construction, $x_j^G > x_j^*$.

Define a new solution $\mathbf{x}'$ that increases $x_j$ from $x_j^*$ toward $x_j^G$ and decreases some later item $k$ (with $r_k \le r_j$) to maintain feasibility. The change in value is:

$$
\Delta V = (x_j^G - x_j^*) \cdot v_j - \delta_k \cdot v_k = w_\Delta (r_j - r_k) \ge 0
$$

where $w_\Delta$ is the weight shifted. Since $r_j \ge r_k$, the new solution is at least as good. Repeating this exchange for every differing item transforms $\mathbf{x}^*$ into $\mathbf{x}^G$ without decreasing the objective. $\square$

## Complexity Analysis

| Step | Time |
|---|---|
| Compute ratios | $O(n)$ |
| Sort by ratio | $O(n \log n)$ |
| Greedy fill | $O(n)$ |
| **Total** | $O(n \log n)$ |

Space complexity is $O(n)$ for storing items and fractions.

## Comparison with 0/1 Knapsack

| Property | Fractional Knapsack | 0/1 Knapsack |
|---|---|---|
| Fractions allowed | Yes ($x_i \in [0,1]$) | No ($x_i \in \{0,1\}$) |
| Optimal strategy | Greedy | Dynamic programming |
| Time complexity | $O(n \log n)$ | $O(nW)$ pseudo-polynomial |
| Greedy works? | Yes | No |

!!! warning "Why Greedy Fails for 0/1 Knapsack"
    Consider items with $(w, v) = \{(10, 60), (20, 100), (30, 120)\}$ and $W = 50$. The greedy approach by value density selects items 1 and 2 (value 160), but the optimal 0/1 solution takes items 2 and 3 (value 220). Greedy fails because it cannot take a fraction of item 3 to fill the remaining capacity.

## Worked Example

**Items:** $(w_1, v_1) = (10, 60)$, $(w_2, v_2) = (20, 100)$, $(w_3, v_3) = (30, 120)$.
**Capacity:** $W = 50$.

**Step 1.** Compute ratios: $r_1 = 6$, $r_2 = 5$, $r_3 = 4$.

**Step 2.** Sorted order: item 1, item 2, item 3.

**Step 3.** Greedy fill:

- Item 1: $w_1 = 10 \le 50$. Take all. $C = 40$. Value $= 60$.
- Item 2: $w_2 = 20 \le 40$. Take all. $C = 20$. Value $= 160$.
- Item 3: $w_3 = 30 > 20$. Take fraction $x_3 = 20/30 = 2/3$. Value $= 160 + 80 = 240$.

**Result:** Total value $= 240$ with items $[1.0, 1.0, 0.667]$.

## Python Implementation

```python
"""
Fractional Knapsack — Greedy Algorithm.

Solves the fractional knapsack problem by sorting items by value-to-weight
ratio and greedily selecting the most valuable items first.
"""

from typing import List, Tuple


# === Greedy Fractional Knapsack ===

def fractional_knapsack(
    items: List[Tuple[float, float]], capacity: float
) -> Tuple[float, List[float]]:
    """Solve the fractional knapsack problem.

    Args:
        items: List of (weight, value) tuples.
        capacity: Maximum weight capacity.

    Returns:
        Tuple of (max_value, fractions) where fractions[i] is
        the fraction of item i taken.
    """
    n = len(items)
    # Compute (index, weight, value, ratio) and sort by ratio descending
    indexed = [(i, w, v, v / w) for i, (w, v) in enumerate(items)]
    indexed.sort(key=lambda x: x[3], reverse=True)

    fractions = [0.0] * n
    total_value = 0.0
    remaining = capacity

    for i, w, v, ratio in indexed:
        if remaining <= 0:
            break
        if w <= remaining:
            fractions[i] = 1.0
            total_value += v
            remaining -= w
        else:
            frac = remaining / w
            fractions[i] = frac
            total_value += v * frac
            remaining = 0

    return total_value, fractions


# === Main ===

if __name__ == "__main__":
    items = [(10, 60), (20, 100), (30, 120)]
    capacity = 50

    max_val, fracs = fractional_knapsack(items, capacity)
    print(f"Items (weight, value): {items}")
    print(f"Capacity: {capacity}")
    print(f"Fractions taken: {[round(f, 3) for f in fracs]}")
    print(f"Maximum value: {max_val}")
    # Output:
    # Items (weight, value): [(10, 60), (20, 100), (30, 120)]
    # Capacity: 50
    # Fractions taken: [1.0, 1.0, 0.667]
    # Maximum value: 240.0
```

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 15 (Greedy Algorithms).
