# Matrix Chain Multiplication

Multiplying a chain of matrices $A_1 A_2 \cdots A_n$ always produces the same result regardless of parenthesization, but the number of scalar multiplications varies dramatically. Matrix chain multiplication asks: what is the optimal way to parenthesize the product to minimize the total number of scalar multiplications? This classic DP problem demonstrates how the order of operations — not the operations themselves — determines computational cost.

## Problem Statement

Given a chain of $n$ matrices $A_1, A_2, \ldots, A_n$ where matrix $A_i$ has dimensions $p_{i-1} \times p_i$, find the parenthesization that minimizes the total number of scalar multiplications.

Multiplying a $p \times q$ matrix by a $q \times r$ matrix requires $p \cdot q \cdot r$ scalar multiplications. Different parenthesizations of the same chain can differ by orders of magnitude in cost.

**Example.** For three matrices with dimensions $10 \times 30$, $30 \times 5$, $5 \times 60$:

- $(A_1 A_2) A_3$: $10 \cdot 30 \cdot 5 + 10 \cdot 5 \cdot 60 = 1500 + 3000 = 4500$
- $A_1 (A_2 A_3)$: $30 \cdot 5 \cdot 60 + 10 \cdot 30 \cdot 60 = 9000 + 18000 = 27000$

The first parenthesization is 6 times cheaper.

## Optimal Substructure

To parenthesize $A_i A_{i+1} \cdots A_j$ optimally, split it at some position $k$ into $(A_i \cdots A_k)(A_{k+1} \cdots A_j)$. The two subchains must themselves be parenthesized optimally — otherwise, replacing them with better parenthesizations would reduce the overall cost, contradicting optimality.

## Recurrence

Let $m[i][j]$ be the minimum number of scalar multiplications to compute $A_i A_{i+1} \cdots A_j$. For a single matrix, no multiplication is needed:

$$
m[i][i] = 0
$$

For $i < j$, try every split point $k$ and take the minimum:

$$
m[i][j] = \min_{i \le k < j} \bigl\{ m[i][k] + m[k+1][j] + p_{i-1} \cdot p_k \cdot p_j \bigr\}
$$

The term $p_{i-1} \cdot p_k \cdot p_j$ is the cost of multiplying the two resulting matrices of dimensions $p_{i-1} \times p_k$ and $p_k \times p_j$.

## Fill Order

The recurrence for $m[i][j]$ depends on subproblems with shorter chain lengths. Fill the table by increasing chain length $\ell = j - i + 1$:

- $\ell = 1$: all $m[i][i] = 0$
- $\ell = 2$: $m[i][i+1]$ for each $i$
- $\ell = 3, 4, \ldots, n$: progressively longer chains

## Complexity

| Aspect | Value |
|---|---|
| Time | $O(n^3)$ — three nested loops |
| Space | $O(n^2)$ — for tables $m$ and $s$ |
| Subproblems | $\binom{n}{2} + n = O(n^2)$ |

## Python Implementation

```python
"""
Matrix Chain Multiplication — Dynamic Programming.

Finds the optimal parenthesization that minimizes scalar multiplications,
using bottom-up tabulation with solution reconstruction.
"""


# === Bottom-Up Tabulation ===

def matrix_chain_order(dims: list[int]) -> tuple[int, list[list[int]]]:
    """Find minimum multiplications for a matrix chain.

    Args:
        dims: List of dimensions where matrix i has size dims[i-1] x dims[i].
              Length n+1 for n matrices.

    Returns:
        Tuple of (min_cost, split_table) where split_table[i][j] gives
        the optimal split point k.
    """
    n = len(dims) - 1  # number of matrices
    m = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]

    # Fill by increasing chain length
    for chain_len in range(2, n + 1):
        for i in range(n - chain_len + 1):
            j = i + chain_len - 1
            m[i][j] = float("inf")
            for k in range(i, j):
                cost = m[i][k] + m[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < m[i][j]:
                    m[i][j] = cost
                    s[i][j] = k

    return m[0][n - 1], s


# === Reconstruct Parenthesization ===

def print_optimal_parens(s: list[list[int]], i: int, j: int) -> str:
    """Reconstruct the optimal parenthesization as a string."""
    if i == j:
        return f"A{i + 1}"
    k = s[i][j]
    left = print_optimal_parens(s, i, k)
    right = print_optimal_parens(s, k + 1, j)
    return f"({left} x {right})"


# === Main ===

if __name__ == "__main__":
    # Matrices: A1(10x30), A2(30x5), A3(5x60)
    dims = [10, 30, 5, 60]
    min_cost, split = matrix_chain_order(dims)
    n = len(dims) - 1
    parens = print_optimal_parens(split, 0, n - 1)

    print(f"Dimensions: {dims}")
    print(f"Minimum multiplications: {min_cost}")
    print(f"Optimal parenthesization: {parens}")

    # Larger example: A1(30x35), A2(35x15), A3(15x5), A4(5x10), A5(10x20), A6(20x25)
    dims2 = [30, 35, 15, 5, 10, 20, 25]
    min_cost2, split2 = matrix_chain_order(dims2)
    n2 = len(dims2) - 1
    parens2 = print_optimal_parens(split2, 0, n2 - 1)

    print(f"\nDimensions: {dims2}")
    print(f"Minimum multiplications: {min_cost2}")
    print(f"Optimal parenthesization: {parens2}")
    # Output:
    # Dimensions: [10, 30, 5, 60]
    # Minimum multiplications: 4500
    # Optimal parenthesization: ((A1 x A2) x A3)
    #
    # Dimensions: [30, 35, 15, 5, 10, 20, 25]
    # Minimum multiplications: 15125
    # Optimal parenthesization: ((A1 x (A2 x A3)) x ((A4 x A5) x A6))
```

## Worked Example

For dimensions $p = [30, 35, 15, 5, 10, 20, 25]$ (six matrices), the $m$ table fills as:

| $m[i][j]$ | $j=1$ | $j=2$ | $j=3$ | $j=4$ | $j=5$ | $j=6$ |
|---|---|---|---|---|---|---|
| $i=1$ | 0 | 15750 | 7875 | 9375 | 11875 | **15125** |
| $i=2$ | | 0 | 2625 | 4375 | 7125 | 10500 |
| $i=3$ | | | 0 | 750 | 2500 | 5375 |
| $i=4$ | | | | 0 | 1000 | 3500 |
| $i=5$ | | | | | 0 | 5000 |
| $i=6$ | | | | | | 0 |

The optimal cost is $m[1][6] = 15125$ with parenthesization $((A_1(A_2 A_3))((A_4 A_5) A_6))$.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
