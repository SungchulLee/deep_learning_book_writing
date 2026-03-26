# Optimal Binary Search Tree

When searching a binary search tree (BST), the cost of finding a key depends on its depth. If some keys are searched far more frequently than others, placing popular keys near the root reduces the expected search cost. The **optimal BST** problem uses dynamic programming to find the tree structure that minimizes this expected cost, given known access frequencies.

## Problem Statement

Given $n$ keys $k_1 < k_2 < \cdots < k_n$ with search probabilities $p_1, p_2, \dots, p_n$ and $n + 1$ dummy keys $d_0, d_1, \dots, d_n$ representing unsuccessful searches with probabilities $q_0, q_1, \dots, q_n$, where:

$$
\sum_{i=1}^{n} p_i + \sum_{j=0}^{n} q_j = 1
$$

The **expected search cost** of a BST $T$ is:

$$
E[\text{cost}] = \sum_{i=1}^{n} p_i \cdot (\text{depth}_T(k_i) + 1) + \sum_{j=0}^{n} q_j \cdot (\text{depth}_T(d_j) + 1)
$$

The goal is to find a BST that minimizes this expected cost.

## Optimal Substructure

If an optimal BST has $k_r$ as its root, then the left subtree (containing $k_1, \dots, k_{r-1}$) must be an optimal BST for those keys, and similarly for the right subtree. This optimal substructure enables a DP solution.

Define $e[i, j]$ as the expected cost of an optimal BST for keys $k_i, \dots, k_j$ (with dummy keys $d_{i-1}, \dots, d_j$). The weight of this subproblem is:

$$
w[i, j] = \sum_{\ell=i}^{j} p_\ell + \sum_{\ell=i-1}^{j} q_\ell
$$

## Recurrence

When $k_r$ is chosen as the root of the subtree for keys $k_i, \dots, k_j$, the cost increases by $w[i, j]$ (each node's depth increases by 1 when it becomes a child of $k_r$):

$$
e[i, j] = \min_{i \le r \le j} \bigl\{e[i, r{-}1] + e[r{+}1, j] + w[i, j]\bigr\}
$$

Base case: $e[i, i{-}1] = q_{i-1}$ (a subtree containing only the dummy key $d_{i-1}$).

## Implementation

```python
"""
Optimal binary search tree via dynamic programming.

Finds the BST structure that minimizes expected search cost
given key access probabilities, in O(n^3) time.
"""

# === Optimal BST ===

def optimal_bst(
    p: list[float], q: list[float]
) -> tuple[float, list[list[int]]]:
    """Compute the optimal BST cost and root table.

    Args:
        p: Search probabilities for keys k_1..k_n (1-indexed, p[0] unused).
        q: Search probabilities for dummy keys d_0..d_n.

    Returns:
        Tuple of (minimum expected cost, root table) where root[i][j]
        is the index of the optimal root for keys k_i..k_j.
    """
    n = len(p) - 1  # p is 1-indexed

    # e[i][j] = expected cost for keys k_i..k_j
    # w[i][j] = total probability weight for keys k_i..k_j
    e = [[0.0] * (n + 2) for _ in range(n + 2)]
    w = [[0.0] * (n + 2) for _ in range(n + 2)]
    root = [[0] * (n + 1) for _ in range(n + 1)]

    # Base cases: e[i][i-1] = q[i-1]
    for i in range(1, n + 2):
        e[i][i - 1] = q[i - 1]
        w[i][i - 1] = q[i - 1]

    # Fill table for increasing chain lengths
    for length in range(1, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            e[i][j] = float('inf')
            w[i][j] = w[i][j - 1] + p[j] + q[j]

            for r in range(i, j + 1):
                cost = e[i][r - 1] + e[r + 1][j] + w[i][j]
                if cost < e[i][j]:
                    e[i][j] = cost
                    root[i][j] = r

    return e[1][n], root


def print_optimal_bst(root: list[list[int]], i: int, j: int,
                      parent: str = "root") -> None:
    """Print the structure of the optimal BST."""
    if i > j:
        print(f"  d_{j} is {parent}")
        return
    r = root[i][j]
    print(f"  k_{r} is {parent}")
    print_optimal_bst(root, i, r - 1, f"left child of k_{r}")
    print_optimal_bst(root, r + 1, j, f"right child of k_{r}")


# === Demonstration ===

if __name__ == "__main__":
    # Example from CLRS: 5 keys with given probabilities
    p = [0, 0.15, 0.10, 0.05, 0.10, 0.20]  # 1-indexed
    q = [0.05, 0.10, 0.05, 0.05, 0.05, 0.10]

    cost, root = optimal_bst(p, q)
    print(f"Minimum expected search cost: {cost:.2f}")
    print("Optimal BST structure:")
    print_optimal_bst(root, 1, 5)
```

**Output:**

```
Minimum expected search cost: 2.75
Optimal BST structure:
  k_2 is root
  k_1 is left child of k_2
  d_0 is left child of k_1
  d_1 is right child of k_1
  k_5 is right child of k_2
  k_4 is left child of k_5
  k_3 is left child of k_4
  d_2 is left child of k_3
  d_3 is right child of k_3
  d_4 is right child of k_4
  d_5 is right child of k_5
```

Key $k_2$ at the root balances the access frequencies. The most frequently searched key $k_5$ ($p_5 = 0.20$) is at depth 1, minimizing its contribution to expected cost.

## Complexity

| Aspect | Cost |
|--------|:----:|
| Time   | $O(n^3)$ |
| Space  | $O(n^2)$ |

The three nested loops (length, start position, root choice) give $O(n^3)$ time. Knuth's optimization reduces this to $O(n^2)$ by observing that $\text{root}[i, j-1] \le \text{root}[i, j] \le \text{root}[i+1, j]$, which limits the search range for $r$.

## Comparison with Balanced BSTs

| Strategy | Expected cost | Guarantee |
|----------|:------------:|:---------:|
| Optimal BST | Minimum possible | Requires known frequencies |
| Balanced BST | $O(\log n)$ per search | No frequency knowledge needed |
| Splay tree | $O(\log n)$ amortized | Adapts to access patterns |

The optimal BST is a static structure. When access frequencies change over time, self-adjusting trees like splay trees provide a dynamic alternative.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 15: Dynamic Programming.
- Knuth, D. E. (1971). Optimum binary search trees. *Acta Informatica*, 1(1), 14--25.
