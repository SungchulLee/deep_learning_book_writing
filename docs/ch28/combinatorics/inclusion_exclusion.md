# Inclusion-Exclusion Principle

The inclusion-exclusion principle computes the size of a union of sets by alternately adding and subtracting the sizes of their intersections. It transforms a difficult "count elements in a union" problem into easier "count elements in intersections" sub-problems, making it indispensable for derangement counting, Euler's totient function, and sieve-based algorithms.

## Intuition

Adding $|A| + |B|$ double-counts elements in $A \cap B$, so we subtract $|A \cap B|$. With three sets, subtracting all pairwise intersections removes too much, so we add back the triple intersection. This alternating pattern generalizes to any number of sets.

## Two-Set Formula

For finite sets $A$ and $B$:

$$
|A \cup B| = |A| + |B| - |A \cap B|
$$

## Three-Set Formula

For finite sets $A$, $B$, $C$:

$$
|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |A \cap C| - |B \cap C| + |A \cap B \cap C|
$$

## General Formula

For finite sets $A_1, A_2, \ldots, A_n$:

$$
\left|\bigcup_{i=1}^{n} A_i\right| = \sum_{k=1}^{n} (-1)^{k+1} \sum_{1 \le i_1 < \cdots < i_k \le n} |A_{i_1} \cap \cdots \cap A_{i_k}|
$$

Equivalently, writing $S = \{1, 2, \ldots, n\}$:

$$
\left|\bigcup_{i=1}^{n} A_i\right| = \sum_{\emptyset \ne T \subseteq S} (-1)^{|T|+1} \left|\bigcap_{i \in T} A_i\right|
$$

## Proof by Induction

**Base case ($n = 1$).** $|A_1| = |A_1|$. Trivially true.

**Inductive step.** Assume the formula holds for $n-1$ sets. Write:

$$
\bigcup_{i=1}^{n} A_i = \left(\bigcup_{i=1}^{n-1} A_i\right) \cup A_n
$$

By the two-set formula:

$$
\left|\bigcup_{i=1}^{n} A_i\right| = \left|\bigcup_{i=1}^{n-1} A_i\right| + |A_n| - \left|\left(\bigcup_{i=1}^{n-1} A_i\right) \cap A_n\right|
$$

Since $\left(\bigcup_{i=1}^{n-1} A_i\right) \cap A_n = \bigcup_{i=1}^{n-1} (A_i \cap A_n)$, we can apply the inductive hypothesis to both unions of $n-1$ sets. Expanding and collecting terms with careful attention to signs yields the $n$-set formula.

??? note "Alternative Proof (Double Counting)"
    Fix any element $x$ in $\bigcup A_i$ and suppose $x$ belongs to exactly $m$ of the sets ($m \ge 1$). The right-hand side counts $x$ exactly:

    $$
    \binom{m}{1} - \binom{m}{2} + \binom{m}{3} - \cdots + (-1)^{m+1}\binom{m}{m}
    $$

    By the binomial theorem with $x = -1$:

    $$
    \sum_{k=0}^{m} \binom{m}{k}(-1)^k = (1 - 1)^m = 0
    $$

    So $\sum_{k=1}^{m} (-1)^{k+1}\binom{m}{k} = 1$. Every element in the union is counted exactly once.

## Example: Counting Integers Divisible by 2, 3, or 5

**Problem.** Among $\{1, 2, \ldots, 100\}$, how many are divisible by 2, 3, or 5?

Let $A_2$, $A_3$, $A_5$ be the sets of multiples of 2, 3, 5 respectively.

$$
|A_2| = 50, \quad |A_3| = 33, \quad |A_5| = 20
$$

$$
|A_2 \cap A_3| = |A_6| = 16, \quad |A_2 \cap A_5| = |A_{10}| = 10, \quad |A_3 \cap A_5| = |A_{15}| = 6
$$

$$
|A_2 \cap A_3 \cap A_5| = |A_{30}| = 3
$$

By inclusion-exclusion:

$$
|A_2 \cup A_3 \cup A_5| = 50 + 33 + 20 - 16 - 10 - 6 + 3 = 74
$$

## Application: Derangements

A **derangement** is a permutation with no fixed points. Let $A_i$ be the set of permutations of $[n]$ that fix element $i$. The number of permutations with at least one fixed point is $|\bigcup A_i|$.

Since $|A_{i_1} \cap \cdots \cap A_{i_k}| = (n-k)!$ (the remaining $n-k$ elements permute freely), and there are $\binom{n}{k}$ ways to choose $k$ indices:

$$
\left|\bigcup_{i=1}^{n} A_i\right| = \sum_{k=1}^{n} (-1)^{k+1} \binom{n}{k}(n-k)!
$$

The number of derangements $D_n = n! - |\bigcup A_i|$:

$$
D_n = n! \sum_{k=0}^{n} \frac{(-1)^k}{k!} \approx \frac{n!}{e}
$$

## Implementation

```python
from itertools import combinations


def inclusion_exclusion(
    universe_size: int,
    sets: list[set],
) -> int:
    """Count elements in the union of sets using inclusion-exclusion.

    Args:
        universe_size: Not used directly; sets contain elements.
        sets: List of sets whose union size we want.

    Returns:
        Size of the union.
    """
    n = len(sets)
    total = 0
    for k in range(1, n + 1):
        sign = (-1) ** (k + 1)
        for combo in combinations(range(n), k):
            intersection = sets[combo[0]]
            for idx in combo[1:]:
                intersection = intersection & sets[idx]
            total += sign * len(intersection)
    return total


def count_derangements(n: int) -> int:
    """Count derangements of [n] using the inclusion-exclusion formula."""
    result = 0
    factorial = 1
    for k in range(n + 1):
        if k > 0:
            factorial *= k
        # Not used: we compute directly from the formula
    # Direct computation
    from math import factorial as fact
    return sum((-1) ** k * fact(n) // fact(k) for k in range(n + 1))


if __name__ == "__main__":
    # === Example: multiples of 2, 3, or 5 in {1..100} ===
    A2 = {i for i in range(1, 101) if i % 2 == 0}
    A3 = {i for i in range(1, 101) if i % 3 == 0}
    A5 = {i for i in range(1, 101) if i % 5 == 0}
    result = inclusion_exclusion(100, [A2, A3, A5])
    print(f"|A2 ∪ A3 ∪ A5| = {result}")  # 74

    # === Derangements ===
    for n in range(1, 8):
        print(f"D_{n} = {count_derangements(n)}")
```

## Complexity

The general inclusion-exclusion formula requires summing over all $2^n - 1$ non-empty subsets of $n$ sets. This exponential cost is unavoidable in the worst case but often acceptable when $n$ is small (e.g., a fixed number of divisibility conditions).

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
- Graham, R. L., Knuth, D. E., & Patashnik, O. (1994). *Concrete Mathematics* (2nd ed.). Addison-Wesley. Chapter 4.
