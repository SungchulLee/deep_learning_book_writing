# Partition Problem

The **Partition** problem asks a deceptively simple question: given a set of integers, can they be split into two subsets with equal sums? Despite its simplicity, Partition is NP-complete. It is the canonical example of a **weakly NP-hard** problem --- one that admits a pseudo-polynomial time algorithm and an FPTAS, distinguishing it from strongly NP-hard problems like 3-Partition.

## Problem Definition

!!! tip "Definition: Partition"
    Given a multiset $S = \{a_1, a_2, \ldots, a_n\}$ of positive integers with total sum $\Sigma = \sum_{i=1}^n a_i$, determine whether there exists a subset $A \subseteq S$ such that:

    $$
    \sum_{i \in A} a_i = \frac{\Sigma}{2}
    $$

If $\Sigma$ is odd, the answer is immediately "no."

## NP-Completeness

!!! tip "Theorem"
    Partition is NP-complete.

**Membership in NP.** The subset $A$ is a certificate. Verifying that its sum equals $\Sigma/2$ takes $O(n)$ time.

**NP-Hardness: Reduction from Subset Sum.** Given a Subset Sum instance $(S', t)$ asking whether a subset of $S'$ sums to $t$:

1. Let $\Sigma' = \sum_{a \in S'} a$.
2. If $t > \Sigma'$, return "no."
3. Add a new element $b = \Sigma' - 2t$ to create $S = S' \cup \{|b|\}$ (with appropriate sign handling):
    - If $\Sigma' - 2t \geq 0$: add element $b = \Sigma' - 2t$. New total = $2(\Sigma' - t)$.
    - If $\Sigma' - 2t < 0$: add element $b = 2t - \Sigma'$. New total = $2t$.

A subset of $S'$ summing to $t$ exists if and only if $S$ can be partitioned into two equal-sum halves. $\square$

## Pseudo-Polynomial Algorithm

Partition reduces to Subset Sum with target $\Sigma/2$, solvable by dynamic programming.

**DP formulation.** Let $\text{dp}[j]$ = true if some subset of $\{a_1, \ldots, a_i\}$ sums to $j$.

**Recurrence:**

$$
\text{dp}[j] = \text{dp}[j] \lor \text{dp}[j - a_i] \quad \text{for } i = 1, \ldots, n, \; j = \Sigma/2, \ldots, a_i
$$

**Base case:** $\text{dp}[0] = \text{true}$.

**Time:** $O(n \cdot \Sigma/2)$. **Space:** $O(\Sigma/2)$.

This is pseudo-polynomial: polynomial in the numeric value $\Sigma$ but exponential in the encoding size $\log \Sigma$.

## Weak vs Strong NP-Hardness

Partition is the textbook example distinguishing these categories:

| Property | Partition | 3-Partition |
|----------|-----------|-------------|
| NP-hard | Yes | Yes |
| Pseudo-polynomial algorithm | Yes ($O(n\Sigma)$) | No (unless P = NP) |
| FPTAS | Yes (via knapsack FPTAS) | No (unless P = NP) |
| NP-hard with poly-bounded numbers | No (becomes easy) | Yes (strongly NP-hard) |

**3-Partition** is a different, harder problem: given $3n$ integers with sum $nB$, can they be partitioned into $n$ triples each summing to $B$? This is strongly NP-complete.

## Approximation

### Differencing Heuristic (Karmarkar-Karp)

The **largest differencing method** repeatedly replaces the two largest numbers with their difference. This greedily reduces the discrepancy between the two partition halves.

1. Insert all numbers into a max-heap.
2. While the heap has more than one element:
    - Extract the two largest, $a$ and $b$.
    - Insert $|a - b|$ back into the heap.
3. The final value is the minimum achievable difference.

This heuristic runs in $O(n \log n)$ and often produces near-optimal partitions.

### FPTAS

Since Partition is equivalent to Subset Sum (with target $\Sigma/2$), the knapsack FPTAS applies. Scaling values by $K = \epsilon \cdot a_{\max} / n$ yields an algorithm running in $O(n^2 / \epsilon)$ that finds a partition with discrepancy at most $\epsilon \cdot \Sigma/2$.

## Variants

### Multi-Way Partition

Partition into $k$ subsets of equal sum. For $k \geq 3$, this is strongly NP-hard (reduces from 3-Partition), so no pseudo-polynomial algorithm exists unless P = NP.

### Balanced Partition

Require $|A| = |B| = n/2$ in addition to equal sums. This additional constraint preserves NP-completeness.

### Minimum Discrepancy

Minimize $|\sum_{i \in A} a_i - \sum_{i \notin A} a_i|$ rather than requiring exact equality. This optimization version is also NP-hard but admits good approximation.

## Connection to Other Problems

Partition occupies a central position in the NP-completeness reduction landscape:

$$
\text{3-SAT} \to \text{3DM} \to \text{Partition} \to \text{Subset Sum} \to \text{Knapsack}
$$

Each reduction preserves NP-hardness while moving to a more "numerical" problem structure.

??? example "Example: Partition with DP"
    **Instance:** $S = \{3, 1, 1, 2, 2, 1\}$, $\Sigma = 10$, target $= 5$.

    **DP table** (reachable sums after adding each element):

    | Element | Reachable sums |
    |---------|---------------|
    | Start | $\{0\}$ |
    | 3 | $\{0, 3\}$ |
    | 1 | $\{0, 1, 3, 4\}$ |
    | 1 | $\{0, 1, 2, 3, 4, 5\}$ |

    Sum 5 is reachable after processing the third element. **Answer: Yes.**

    **Partition:** $A = \{3, 1, 1\}$ (sum 5), $B = \{2, 2, 1\}$ (sum 5).

## Reference

- Garey, M. R., & Johnson, D. S. (1979). *Computers and Intractability*. W. H. Freeman.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
- Karmarkar, N., & Karp, R. M. (1982). The differencing method of set partitioning. Technical Report UCB/CSD-82-113, UC Berkeley.
