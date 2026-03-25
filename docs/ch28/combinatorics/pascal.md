# Pascal's Triangle

Pascal's triangle is a triangular array of binomial coefficients where each entry is the sum of the two entries above it. Beyond its visual elegance, it provides an efficient way to compute binomial coefficients and reveals deep algebraic patterns used in algorithm analysis, probability, and number theory.

## Intuition

Build a triangle row by row. Row 0 is just $1$. Each subsequent entry is formed by adding the two numbers directly above it (treating missing entries as 0). The entry in row $n$, position $k$ equals $\binom{n}{k}$.

## Construction

```
Row 0:                1
Row 1:              1   1
Row 2:            1   2   1
Row 3:          1   3   3   1
Row 4:        1   4   6   4   1
Row 5:      1   5  10  10   5   1
Row 6:    1   6  15  20  15   6   1
```

## Pascal's Identity

The defining recurrence is:

$$
\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k} \quad \text{for } 1 \le k \le n-1
$$

with boundary conditions $\binom{n}{0} = \binom{n}{n} = 1$ for all $n \ge 0$.

??? note "Proof"
    Consider a set $S$ of $n$ elements and fix one element $x$. Every $k$-subset of $S$ either contains $x$ or does not:

    - **Contains $x$:** Choose the remaining $k-1$ elements from $S \setminus \{x\}$, giving $\binom{n-1}{k-1}$ subsets.
    - **Does not contain $x$:** Choose all $k$ elements from $S \setminus \{x\}$, giving $\binom{n-1}{k}$ subsets.

    These two cases are disjoint and exhaustive, so $\binom{n}{k} = \binom{n-1}{k-1} + \binom{n-1}{k}$.

## Properties Visible in Pascal's Triangle

**Row sums.** Summing all entries in row $n$ gives $2^n$:

$$
\sum_{k=0}^{n} \binom{n}{k} = 2^n
$$

This follows from the binomial theorem with $x = y = 1$.

**Alternating row sums.** For $n \ge 1$:

$$
\sum_{k=0}^{n} (-1)^k \binom{n}{k} = 0
$$

This follows from $(1 - 1)^n = 0$.

**Symmetry.** Each row is a palindrome: $\binom{n}{k} = \binom{n}{n-k}$.

**Diagonal sums (Fibonacci).** Summing entries along diagonals of slope $-1$ gives the Fibonacci numbers:

$$
F_{n+1} = \sum_{k=0}^{\lfloor n/2 \rfloor} \binom{n-k}{k}
$$

**Hockey stick identity.** Summing consecutive entries along a diagonal:

$$
\sum_{i=r}^{n} \binom{i}{r} = \binom{n+1}{r+1}
$$

??? note "Proof of Hockey Stick Identity"
    By induction on $n$. The base case $n = r$ gives $\binom{r}{r} = 1 = \binom{r+1}{r+1}$. For the inductive step:

    $$
    \sum_{i=r}^{n} \binom{i}{r} = \binom{n}{r} + \sum_{i=r}^{n-1} \binom{i}{r} = \binom{n}{r} + \binom{n}{r+1} = \binom{n+1}{r+1}
    $$

    where the last equality uses Pascal's identity.

## Divisibility Properties

**Row $p$ for prime $p$.** If $p$ is prime, then $\binom{p}{k} \equiv 0 \pmod{p}$ for $1 \le k \le p-1$. This means all interior entries of row $p$ are divisible by $p$.

**Lucas' theorem.** For a prime $p$, if $n = \sum n_i p^i$ and $k = \sum k_i p^i$ are the base-$p$ representations:

$$
\binom{n}{k} \equiv \prod_{i} \binom{n_i}{k_i} \pmod{p}
$$

## Building Pascal's Triangle

### Full Triangle

```python
def pascal_triangle(n: int) -> list[list[int]]:
    """Build Pascal's triangle with rows 0 through n.

    Time: O(n^2). Space: O(n^2).
    """
    triangle = [[1]]
    for i in range(1, n + 1):
        prev = triangle[-1]
        row = [1]
        for j in range(1, i):
            row.append(prev[j - 1] + prev[j])
        row.append(1)
        triangle.append(row)
    return triangle
```

### Single Row (Space-Optimized)

```python
def pascal_row(n: int) -> list[int]:
    """Compute row n of Pascal's triangle in O(n) space.

    Updates a single array right-to-left to avoid overwriting
    values needed for the current iteration.
    """
    row = [0] * (n + 1)
    row[0] = 1
    for i in range(1, n + 1):
        for j in range(i, 0, -1):
            row[j] += row[j - 1]
    return row


if __name__ == "__main__":
    # === Print first 8 rows ===
    tri = pascal_triangle(7)
    for i, row in enumerate(tri):
        padding = " " * (7 - i) * 2
        values = "  ".join(f"{v:3d}" for v in row)
        print(f"{padding}{values}")

    # === Verify row sums equal powers of 2 ===
    for i in range(8):
        assert sum(tri[i]) == 2 ** i
    print("\nAll row sums verified: sum(row n) = 2^n")
```

## Complexity

| Operation | Time | Space |
|---|---|---|
| Build full triangle (rows 0 to $n$) | $O(n^2)$ | $O(n^2)$ |
| Compute single row | $O(n^2)$ | $O(n)$ |
| Look up $\binom{n}{k}$ from prebuilt table | $O(1)$ | $O(n^2)$ precomputed |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
- Graham, R. L., Knuth, D. E., & Patashnik, O. (1994). *Concrete Mathematics* (2nd ed.). Addison-Wesley. Chapter 5.
