# Recursion Tree Method

A recurrence like $T(n) = 2T(n/2) + n$ tells us the *relationship* between the running time at different problem sizes, but it does not immediately reveal the closed-form answer. The recursion tree method makes the answer visible by unfolding the recurrence into a tree, computing the work at each level, and summing across all levels. This visual approach builds the geometric intuition that underlies the [Master theorem](master.md) and is the standard technique for *guessing* an answer that can then be verified with the [substitution method](substitution.md).

## Building a Recursion Tree

Given a recurrence $T(n) = aT(n/b) + f(n)$, the recursion tree is constructed as follows:

1. **Root**: A single node representing the original problem of size $n$. The non-recursive work at the root is $f(n)$.
2. **Children**: The root has $a$ children, each representing a subproblem of size $n/b$. Each child does $f(n/b)$ non-recursive work.
3. **Recurse**: Each child spawns $a$ grandchildren of size $n/b^2$, and so on.
4. **Leaves**: The recursion bottoms out when the problem size reaches the base case, at depth $\log_b n$.

### Tree Structure

```
Level 0:                    f(n)                          → 1 node
                          /      \
Level 1:          f(n/b)    ...    f(n/b)                 → a nodes
                 / \              / \
Level 2:    f(n/b²) ...     f(n/b²) ...                  → a² nodes
                  ⋮                    ⋮
Level j:    a^j nodes, each doing f(n/b^j)               → a^j nodes
                  ⋮                    ⋮
Level log_b(n):   T(1) T(1) ... T(1)                     → a^(log_b n) = n^(log_b a) leaves
```

## Level-by-Level Analysis

At level $j$ (counting the root as level 0):

- **Number of nodes**: $a^j$
- **Subproblem size**: $n / b^j$
- **Work per node**: $f(n / b^j)$
- **Total work at level $j$**: $a^j \cdot f(n / b^j)$

The total running time is the sum over all levels, plus the leaf-level base-case work:

$$
T(n) = \sum_{j=0}^{\log_b n - 1} a^j \cdot f\!\left(\frac{n}{b^j}\right) + \Theta(n^{\log_b a})
$$

The last term $\Theta(n^{\log_b a})$ accounts for the $n^{\log_b a}$ leaves, each contributing $\Theta(1)$ work.

## Example 1: Merge Sort

Consider $T(n) = 2T(n/2) + cn$ where $c$ is a constant.

### Level-by-Level Costs

| Level | Nodes | Size | Work per node | Level total |
|-------|-------|------|---------------|-------------|
| 0 | 1 | $n$ | $cn$ | $cn$ |
| 1 | 2 | $n/2$ | $cn/2$ | $cn$ |
| 2 | 4 | $n/4$ | $cn/4$ | $cn$ |
| $j$ | $2^j$ | $n/2^j$ | $cn/2^j$ | $cn$ |
| $\log_2 n$ | $n$ | 1 | $\Theta(1)$ | $\Theta(n)$ |

Every level contributes exactly $cn$. There are $\log_2 n + 1$ levels (including the leaf level), giving:

$$
T(n) = cn \cdot \log_2 n + \Theta(n) = \Theta(n \log n)
$$

This is the **balanced case**: the work is evenly distributed across levels.

## Example 2: Root-Heavy Tree

Consider $T(n) = 3T(n/4) + cn^2$.

### Level-by-Level Costs

| Level | Nodes | Size | Work per node | Level total |
|-------|-------|------|---------------|-------------|
| 0 | 1 | $n$ | $cn^2$ | $cn^2$ |
| 1 | 3 | $n/4$ | $c(n/4)^2$ | $3cn^2/16$ |
| 2 | 9 | $n/16$ | $c(n/16)^2$ | $9cn^2/256$ |
| $j$ | $3^j$ | $n/4^j$ | $c(n/4^j)^2$ | $cn^2 (3/16)^j$ |

The level totals form a geometric series with ratio $r = 3/16 < 1$:

$$
T(n) = cn^2 \sum_{j=0}^{\log_4 n - 1} \left(\frac{3}{16}\right)^j + \Theta(n^{\log_4 3})
$$

Since the geometric series converges:

$$
\sum_{j=0}^{\infty} \left(\frac{3}{16}\right)^j = \frac{1}{1 - 3/16} = \frac{16}{13}
$$

So $T(n) = \Theta(n^2)$. The root dominates because the work *decreases* geometrically at each level. This corresponds to Case 3 of the Master theorem.

## Example 3: Leaf-Heavy Tree

Consider $T(n) = 4T(n/2) + cn$.

### Level-by-Level Costs

| Level | Nodes | Size | Work per node | Level total |
|-------|-------|------|---------------|-------------|
| 0 | 1 | $n$ | $cn$ | $cn$ |
| 1 | 4 | $n/2$ | $cn/2$ | $2cn$ |
| 2 | 16 | $n/4$ | $cn/4$ | $4cn$ |
| $j$ | $4^j$ | $n/2^j$ | $cn/2^j$ | $cn \cdot 2^j$ |
| $\log_2 n$ | $n^2$ | 1 | $\Theta(1)$ | $\Theta(n^2)$ |

The level totals form a geometric series with ratio $r = 2 > 1$, so the work *increases* geometrically. The last full level dominates:

$$
T(n) = cn \sum_{j=0}^{\log_2 n - 1} 2^j + \Theta(n^2) = cn(2^{\log_2 n} - 1) + \Theta(n^2) = \Theta(n^2)
$$

Here $n^{\log_b a} = n^{\log_2 4} = n^2$, confirming Case 1 of the Master theorem: the leaves dominate.

## The Three Geometric Patterns

The recursion tree reveals why the Master theorem has exactly three cases:

| Pattern | Ratio $r = a / b^c$ where $f(n) = \Theta(n^c)$ | Dominant level | Master case |
|---------|------------------------------------------------|----------------|-------------|
| Decreasing ($r < 1$) | Work shrinks geometrically | Root | Case 3 |
| Constant ($r = 1$) | Equal work at every level | All (summed) | Case 2 |
| Increasing ($r > 1$) | Work grows geometrically | Leaves | Case 1 |

The ratio $r$ determines whether the geometric series converges (root-heavy), is constant (balanced), or diverges (leaf-heavy).

## Using the Recursion Tree to Guess, Then Verify

The recursion tree method is typically used in two stages:

1. **Guess**: Draw the tree, sum the levels, and conjecture an asymptotic bound
2. **Verify**: Use the [substitution method](substitution.md) to prove the guess rigorously

The tree itself is not a formal proof because it relies on intuitive arguments about geometric series. The substitution method provides the inductive proof that the guess is correct.

??? example "Guess-and-Verify for $T(n) = 2T(n/2) + n$"
    **Guess** (from the tree): $T(n) = O(n \log n)$.

    **Verify** by induction: Assume $T(k) \leq ck \log k$ for all $k < n$. Then:

    $$
    T(n) = 2T(n/2) + n \leq 2c(n/2)\log(n/2) + n = cn(\log n - 1) + n = cn\log n - cn + n
    $$

    This is at most $cn \log n$ whenever $c \geq 1$. So $T(n) = O(n \log n)$.

## Handling Non-Standard Recurrences

### Unequal Subproblem Sizes

When subproblems have different sizes, the recursion tree is no longer perfectly balanced. Consider $T(n) = T(n/3) + T(2n/3) + n$.

The longest path goes from root to a leaf via the $2n/3$ branches, reaching the base case at depth $\log_{3/2} n$. The shortest path reaches it at depth $\log_3 n$. At every level, the total work is at most $n$ (because all subproblem sizes at any level sum to at most $n$). Therefore:

$$
T(n) = O(n \log_{3/2} n) = O(n \log n)
$$

A matching lower bound can be shown by noting that the total work at every level is at least $cn$ for sufficiently many levels, giving $T(n) = \Omega(n \log n)$.

### Additive Recurrences

For $T(n) = T(n-1) + f(n)$, the "tree" degenerates into a chain (each node has one child). The total work is simply:

$$
T(n) = \sum_{k=1}^{n} f(k) + T(0)
$$

This is a summation problem, not a geometric series, so the Master theorem does not apply. Direct summation or [generating functions](generating.md) are more appropriate.

## Connections to Other Topics

- **[Master Theorem](master.md)**: The theorem that formalizes the three geometric patterns
- **[Extended Master Theorem](extended_master.md)**: Handles the balanced case with logarithmic factors
- **[Substitution Method](substitution.md)**: Verifies guesses obtained from the recursion tree
- **[Recurrence from Divide and Conquer](divide_conquer.md)**: How to derive the recurrences that recursion trees visualize

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Section 4.4. MIT Press.
- Erickson, J. (2019). *Algorithms*, Chapter 1. Self-published.
