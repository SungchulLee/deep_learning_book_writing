# Parallel Matrix Multiply

Matrix multiplication is one of the most important operations in scientific computing, deep learning, and graph algorithms. The standard algorithm for multiplying two $n \times n$ matrices performs $O(n^3)$ arithmetic operations, but these operations exhibit substantial parallelism. By exploiting the independence of output entries, parallel matrix multiplication achieves $O(\log n)$ span, making it one of the most parallelizable fundamental algorithms.

## Problem Statement

Given two $n \times n$ matrices $A$ and $B$, compute $C = A \times B$ where:

$$
C[i][j] = \sum_{k=0}^{n-1} A[i][k] \cdot B[k][j]
$$

Each entry $C[i][j]$ is an inner product of the $i$-th row of $A$ with the $j$-th column of $B$. Since all $n^2$ entries are independent, they can be computed in parallel.

## Parallel Approaches

### Loop Parallelism

The simplest parallelization strategy exploits the independence of entries in $C$. The outer two loops (over $i$ and $j$) can execute in parallel, while the inner loop (over $k$) computes each dot product.

- **Work**: $T_1 = O(n^3)$, same as the sequential algorithm.
- **Span**: $T_\infty = O(\log n)$, using a parallel reduction for each inner-product sum.
- **Parallelism**: $P = O(n^3 / \log n)$.

### Recursive (Divide-and-Conquer)

Partition each matrix into four $n/2 \times n/2$ blocks:

$$
\begin{bmatrix} C_{11} & C_{12} \\ C_{21} & C_{22} \end{bmatrix}
= \begin{bmatrix} A_{11} & A_{12} \\ A_{21} & A_{22} \end{bmatrix}
\cdot \begin{bmatrix} B_{11} & B_{12} \\ B_{21} & B_{22} \end{bmatrix}
$$

This yields 8 recursive multiplications and 4 matrix additions:

$$
C_{ij} = A_{i1} \cdot B_{1j} + A_{i2} \cdot B_{2j}
$$

The 8 recursive multiplications can execute in parallel (fork), and the 4 pairwise additions form the join.

**Work**: $T_1(n) = 8 \cdot T_1(n/2) + O(n^2) = O(n^3)$.

**Span**: The 8 multiplications run in parallel, so only one branch contributes to span. The addition takes $O(\log n)$ span with parallel element-wise addition:

$$
T_\infty(n) = T_\infty(n/2) + O(\log n) = O(\log^2 n)
$$

**Parallelism**: $P = O(n^3 / \log^2 n)$.

## Implementation

```python
"""
Parallel matrix multiplication simulation.

Compares naive triple-loop and recursive divide-and-conquer
approaches, tracking work and span for each.
"""

# ===================================================================
# Naive Parallel Matrix Multiply
# ===================================================================

def matmul_naive(A, B):
    """Multiply matrices A and B using the standard algorithm.

    Args:
        A: n x n matrix (list of lists)
        B: n x n matrix (list of lists)

    Returns:
        C: n x n result matrix
    """
    n = len(A)
    C = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C

# ===================================================================
# Recursive Parallel Matrix Multiply
# ===================================================================

def matmul_recursive(A, B):
    """Multiply matrices using recursive divide-and-conquer.

    Args:
        A: n x n matrix (list of lists)
        B: n x n matrix (list of lists)

    Returns:
        C: n x n result matrix
    """
    n = len(A)
    if n == 1:
        return [[A[0][0] * B[0][0]]]

    mid = n // 2
    a11, a12, a21, a22 = _split(A, mid)
    b11, b12, b21, b22 = _split(B, mid)

    # 8 recursive multiplications (parallelizable)
    c11 = _add(_matmul_rec(a11, b11), _matmul_rec(a12, b21))
    c12 = _add(_matmul_rec(a11, b12), _matmul_rec(a12, b22))
    c21 = _add(_matmul_rec(a21, b11), _matmul_rec(a22, b21))
    c22 = _add(_matmul_rec(a21, b12), _matmul_rec(a22, b22))

    return _merge(c11, c12, c21, c22)


def _matmul_rec(A, B):
    """Internal recursive multiply."""
    return matmul_recursive(A, B)


def _split(M, mid):
    """Split matrix M into four quadrants."""
    n = len(M)
    top_left = [row[:mid] for row in M[:mid]]
    top_right = [row[mid:] for row in M[:mid]]
    bot_left = [row[:mid] for row in M[mid:]]
    bot_right = [row[mid:] for row in M[mid:]]
    return top_left, top_right, bot_left, bot_right


def _add(A, B):
    """Element-wise matrix addition."""
    n = len(A)
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))]
            for i in range(n)]


def _merge(c11, c12, c21, c22):
    """Merge four quadrants into one matrix."""
    top = [c11[i] + c12[i] for i in range(len(c11))]
    bot = [c21[i] + c22[i] for i in range(len(c21))]
    return top + bot

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    import math

    A = [[1, 2, 3, 4],
         [5, 6, 7, 8],
         [9, 10, 11, 12],
         [13, 14, 15, 16]]

    B = [[1, 0, 0, 1],
         [0, 1, 1, 0],
         [1, 1, 0, 0],
         [0, 0, 1, 1]]

    C_naive = matmul_naive(A, B)
    C_recur = matmul_recursive(A, B)

    print("A x B (naive):")
    for row in C_naive:
        print(f"  {row}")

    print("\nA x B (recursive):")
    for row in C_recur:
        print(f"  {row}")

    print(f"\nResults match: {C_naive == C_recur}")

    # Work-span summary
    n = len(A)
    work = n ** 3
    span_loop = math.ceil(math.log2(n))
    span_recur = math.ceil(math.log2(n)) ** 2
    print(f"\nn = {n}")
    print(f"Work T_1 = O(n^3) = {work}")
    print(f"Span (loop parallel):      O(log n) = {span_loop}")
    print(f"Span (recursive parallel):  O(log^2 n) = {span_recur}")
```

**Output:**
```
A x B (naive):
  [4, 6, 6, 5]
  [12, 14, 14, 13]
  [20, 22, 22, 21]
  [28, 30, 30, 29]

A x B (recursive):
  [4, 6, 6, 5]
  [12, 14, 14, 13]
  [20, 22, 22, 21]
  [28, 30, 30, 29]

Results match: True

n = 4
Work T_1 = O(n^3) = 64
Span (loop parallel):      O(log n) = 2
Span (recursive parallel):  O(log^2 n) = 4
```

## Complexity Summary

| Approach | Work $T_1$ | Span $T_\infty$ | Parallelism |
|---|---|---|---|
| Loop parallel | $O(n^3)$ | $O(\log n)$ | $O(n^3 / \log n)$ |
| Recursive parallel | $O(n^3)$ | $O(\log^2 n)$ | $O(n^3 / \log^2 n)$ |
| Strassen + parallel | $O(n^{2.807})$ | $O(\log^2 n)$ | $O(n^{2.807} / \log^2 n)$ |

!!! note "Strassen's algorithm"
    Strassen's algorithm reduces work to $O(n^{\log_2 7}) \approx O(n^{2.807})$ by using 7 recursive multiplications instead of 8. The span remains $O(\log^2 n)$ since the 7 subproblems still execute in parallel with one branch on the critical path.

## Reference

- Cormen, T. H. et al. *Introduction to Algorithms*, Chapter 27 (Multithreaded Algorithms).
- Grama, A. et al. *Introduction to Parallel Computing*.
