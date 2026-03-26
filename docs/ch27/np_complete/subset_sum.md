# Subset Sum

Many computational problems ask whether a collection of numbers can combine to
hit a target value.  Subset Sum distills this question to its purest form and
serves as a gateway to understanding NP-completeness: it is easy to verify a
solution, yet no known polynomial-time algorithm can find one.

## Problem Definition

Given a finite set $S = \{a_1, a_2, \dots, a_n\}$ of non-negative integers and
a target integer $t$, the **Subset Sum** decision problem asks:

> Does there exist a subset $S' \subseteq S$ such that
> $\sum_{a \in S'} a = t$?

!!! example "Concrete Instance"
    Let $S = \{3, 7, 1, 8, 4\}$ and $t = 12$.  The subset $\{3, 1, 8\}$ sums
    to $12$, so the answer is **YES**.  Changing the target to $t = 2$ yields
    **NO**, because no subset reaches exactly $2$.

## Membership in NP

A certificate for a YES instance is simply the subset $S'$.  A polynomial-time
verifier checks two things:

1. $S' \subseteq S$.
2. $\sum_{a \in S'} a = t$.

Both checks run in $O(n)$ time, so Subset Sum $\in$ NP.

## NP-Completeness via Reduction from 3-SAT

The standard proof reduces **3-SAT** to Subset Sum.  The reduction constructs
numbers whose decimal (or large-base) digits encode clause satisfaction.

### Reduction Sketch

Given a 3-SAT formula $\phi$ with $n$ variables $x_1, \dots, x_n$ and $m$
clauses $C_1, \dots, C_m$:

1. **Create $n + m$ digit positions.**  The first $n$ positions correspond to
   variables; the last $m$ positions correspond to clauses.
2. **For each variable $x_i$,** create two numbers $v_i$ (for $x_i = \text{true}$)
   and $v_i'$ (for $x_i = \text{false}$).  In digit position $i$, both $v_i$
   and $v_i'$ have a $1$.  In each clause digit position $j$, $v_i$ has a $1$
   if $x_i$ appears positively in $C_j$, and $v_i'$ has a $1$ if $\lnot x_i$
   appears in $C_j$.
3. **For each clause $C_j$,** add slack numbers $s_j$ and $s_j'$ with a $1$
   only in position $j$.
4. **Set the target** $t$ to have a $1$ in each variable position and a $3$ in
   each clause position.

Use a base $b \ge 4$ to prevent carries across digit positions.

### Correctness Argument

- **If $\phi$ is satisfiable,** the truth assignment selects exactly one of
  $v_i, v_i'$ per variable, contributing $1$ to each variable digit.  Each
  satisfied clause receives $1$, $2$, or $3$ from the selected $v_i / v_i'$
  values; the slack numbers fill the remainder to reach $3$.
- **If the subset sums to $t$,** exactly one of $v_i, v_i'$ is chosen per
  variable (forced by the variable digits), and every clause digit reaches $3$,
  meaning every clause is satisfied.

The reduction runs in polynomial time because it creates $O(n + m)$ numbers,
each with $O(n + m)$ digits.

## Dynamic Programming Solution

Although the problem is NP-complete, a pseudo-polynomial algorithm exists.

```python
"""
Subset Sum via dynamic programming.

Time : O(n * t)
Space: O(t)
"""


# === Subset Sum DP ===
def subset_sum(nums: list[int], target: int) -> bool:
    """Return True if any subset of nums sums to target."""
    dp = [False] * (target + 1)
    dp[0] = True
    for num in nums:
        # Traverse right-to-left to avoid using num twice
        for j in range(target, num - 1, -1):
            if dp[j - num]:
                dp[j] = True
    return dp[target]


# === Example ===
if __name__ == "__main__":
    S = [3, 7, 1, 8, 4]
    t = 12
    print(f"S = {S}, t = {t}")
    print(f"Subset with sum {t} exists: {subset_sum(S, t)}")  # True

    t2 = 2
    print(f"Subset with sum {t2} exists: {subset_sum(S, t2)}")  # False
```

The running time is $O(n \cdot t)$.  Because $t$ can be exponential in the
input size (the number of bits needed to represent $t$ is $\log t$), this is
**pseudo-polynomial** rather than truly polynomial.

## Relationship to Other NP-Complete Problems

Subset Sum is a special case of the **0/1 Knapsack** problem (set all item
values equal to their weights).  It also connects to:

| Problem | Reduction Direction |
|---|---|
| 3-SAT | 3-SAT $\le_p$ Subset Sum |
| Partition | Subset Sum $\le_p$ Partition |
| 0/1 Knapsack | Subset Sum $\le_p$ 0/1 Knapsack |

!!! tip "Partition as a Variant"
    The **Partition** problem asks whether $S$ can be split into two subsets of
    equal sum.  This reduces to Subset Sum with $t = (\sum S) / 2$ and is
    itself NP-complete.

## Practical Considerations

- **Approximation.**  A fully polynomial-time approximation scheme (FPTAS)
  exists: for any $\epsilon > 0$, one can find a subset whose sum is within
  $(1 - \epsilon)t$ in time polynomial in $n$ and $1/\epsilon$.
- **Cryptographic relevance.**  Lattice-based cryptosystems build on the
  hardness of Subset Sum variants (e.g., the Merkle--Hellman knapsack system).
- **Meet-in-the-middle.**  Splitting $S$ into two halves and enumerating all
  $2^{n/2}$ sums per half yields an $O(2^{n/2})$ exact algorithm, improving
  over brute-force $O(2^n)$.

## Reference

- Sipser, M. *Introduction to the Theory of Computation*. Cengage Learning.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction
  to Algorithms* (CLRS), Chapter 34.
