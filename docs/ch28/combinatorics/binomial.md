# Binomial Coefficients

The binomial coefficient $\binom{n}{k}$ counts the number of ways to choose $k$ items from a set of $n$ distinct items, without regard to order. This quantity appears throughout algorithm analysis (divide-and-conquer recurrences, probabilistic arguments) and combinatorics (counting subsets, lattice paths).

## Intuition

Imagine selecting a committee of $k$ people from a group of $n$. Each selection is an unordered subset of size $k$. The total number of such committees is $\binom{n}{k}$, read "$n$ choose $k$."

## Definition

For non-negative integers $n$ and $k$ with $0 \le k \le n$:

$$
\binom{n}{k} = \frac{n!}{k!\,(n-k)!}
$$

By convention, $\binom{n}{k} = 0$ when $k < 0$ or $k > n$.

## Key Properties

**Symmetry.** Choosing which $k$ items to include is equivalent to choosing which $n - k$ items to exclude:

$$
\binom{n}{k} = \binom{n}{n-k}
$$

**Absorption (extraction).** Pulling one factor out of the numerator:

$$
\binom{n}{k} = \frac{n}{k}\,\binom{n-1}{k-1} \quad (k \ge 1)
$$

**Pascal's identity.** The $k$-th element is either in the chosen subset or not:

$$
\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k} \quad (1 \le k \le n-1)
$$

??? note "Proof of Pascal's Identity"
    From the factorial definition:

    $$
    \binom{n-1}{k-1} + \binom{n-1}{k} = \frac{(n-1)!}{(k-1)!(n-k)!} + \frac{(n-1)!}{k!(n-k-1)!}
    $$

    Factor out $(n-1)!$ and find a common denominator $k!(n-k)!$:

    $$
    = \frac{(n-1)!\,k + (n-1)!\,(n-k)}{k!(n-k)!} = \frac{(n-1)!\,n}{k!(n-k)!} = \frac{n!}{k!(n-k)!} = \binom{n}{k}
    $$

**Vandermonde's identity.** For non-negative integers $m$, $n$, $r$:

$$
\binom{m+n}{r} = \sum_{k=0}^{r} \binom{m}{k}\binom{n}{r-k}
$$

**Binomial theorem.** For any real $x$, $y$ and non-negative integer $n$:

$$
(x + y)^n = \sum_{k=0}^{n} \binom{n}{k}\, x^k\, y^{n-k}
$$

Setting $x = y = 1$ gives $\sum_{k=0}^{n} \binom{n}{k} = 2^n$, confirming that a set of $n$ elements has $2^n$ subsets.

## Computing Binomial Coefficients

### Multiplicative Formula

The factorial definition causes overflow for moderate $n$. A better approach multiplies and divides incrementally:

$$
\binom{n}{k} = \frac{n \cdot (n-1) \cdots (n-k+1)}{k!} = \prod_{i=1}^{k} \frac{n - k + i}{i}
$$

Each partial product $\prod_{i=1}^{j} \frac{n-k+i}{i}$ is an integer, so the division is always exact when performed left to right.

```python
def binom(n: int, k: int) -> int:
    """Compute C(n, k) using the multiplicative formula.

    Runs in O(min(k, n-k)) time and O(1) space.
    """
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)  # exploit symmetry
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


if __name__ == "__main__":
    # === Example usage ===
    print(f"C(10, 3) = {binom(10, 3)}")   # 120
    print(f"C(20, 10) = {binom(20, 10)}") # 184756
```

### Pascal's Triangle (Dynamic Programming)

Build a table using Pascal's identity in $O(nk)$ time and $O(nk)$ space:

```python
def pascal_table(n: int) -> list[list[int]]:
    """Build Pascal's triangle up to row n.

    Returns a 2D list where C[i][j] = C(i, j).
    """
    C = [[0] * (n + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        C[i][0] = 1
        for j in range(1, i + 1):
            C[i][j] = C[i - 1][j - 1] + C[i - 1][j]
    return C


if __name__ == "__main__":
    # === Print first 6 rows ===
    table = pascal_table(5)
    for i in range(6):
        print([table[i][j] for j in range(i + 1)])
```

### Space-Optimized Pascal's Row

When only a single row $n$ is needed, use a 1D array updated right to left:

```python
def pascal_row(n: int) -> list[int]:
    """Compute row n of Pascal's triangle in O(n) space."""
    row = [0] * (n + 1)
    row[0] = 1
    for i in range(1, n + 1):
        for j in range(i, 0, -1):
            row[j] += row[j - 1]
    return row
```

## Bounds and Asymptotics

A useful bound for algorithm analysis:

$$
\left(\frac{n}{k}\right)^k \le \binom{n}{k} \le \left(\frac{en}{k}\right)^k
$$

For fixed $k$, $\binom{n}{k} = \Theta(n^k)$. For the central coefficient:

$$
\binom{2n}{n} \sim \frac{4^n}{\sqrt{\pi n}} \quad \text{(Stirling's approximation)}
$$

## Applications in Algorithms

| Application | How $\binom{n}{k}$ appears |
|---|---|
| Counting subsets | Number of $k$-element subsets of an $n$-set |
| Merge sort analysis | Number of inversions in a permutation |
| Probabilistic analysis | Expected value via indicator random variables |
| Hashing analysis | Birthday-problem style collision bounds |
| Divide and conquer | Recurrence solutions involving $\binom{n}{k}$ terms |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
- Graham, R. L., Knuth, D. E., & Patashnik, O. (1994). *Concrete Mathematics* (2nd ed.). Addison-Wesley. Chapter 5.
