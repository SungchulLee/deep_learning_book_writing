# Exponential-Time Algorithms

When facing NP-hard problems where approximation is insufficient and an exact answer is required, we turn to **exponential-time algorithms**. While all brute-force approaches run in exponential time, clever techniques can significantly reduce the base of the exponential. An $O(1.2^n)$ algorithm is dramatically faster than $O(2^n)$ for moderate $n$: at $n = 100$, the ratio is about $10^{10}$. This page surveys the main techniques for designing faster exact algorithms.

## Brute Force Baseline

The naive approach to NP-hard problems enumerates all possible solutions:

| Problem | Brute Force | Search Space |
|---------|------------|-------------|
| SAT ($n$ variables) | $O(2^n \cdot m)$ | All truth assignments |
| TSP ($n$ cities) | $O(n! \cdot n)$ | All permutations |
| Vertex Cover | $O(2^n \cdot m)$ | All vertex subsets |
| Graph Coloring ($k$ colors) | $O(k^n \cdot m)$ | All colorings |

The goal is to beat these baselines, often by exploiting problem structure.

## Meet in the Middle

**Idea:** Split the problem into two halves of size $n/2$, solve each independently, then combine. This reduces $O(2^n)$ to $O(2^{n/2})$ at the cost of additional space.

### Application: Subset Sum

Given $n$ integers and a target $t$, determine if a subset sums to $t$.

1. Split the integers into sets $A$ (first $n/2$) and $B$ (last $n/2$).
2. Enumerate all $2^{n/2}$ subset sums of $A$; store in a hash table.
3. For each subset sum $s_B$ of $B$, check if $t - s_B$ exists in the hash table.

**Time:** $O(2^{n/2})$. **Space:** $O(2^{n/2})$.

This is a quadratic improvement in the exponent: $2^{n/2} = \sqrt{2^n}$.

## Inclusion-Exclusion

The **inclusion-exclusion principle** converts counting over combinatorial objects into an alternating sum, often yielding faster algorithms.

### Application: Hamiltonian Path

Count the number of Hamiltonian paths in a graph $G = (V, E)$ with $|V| = n$:

$$
\text{ham}(G) = \sum_{S \subseteq V} (-1)^{|V| - |S|} \cdot w(S)
$$

where $w(S)$ counts the number of walks of length $|V| - 1$ using only vertices in $S$. Each $w(S)$ is computed via matrix exponentiation on the adjacency matrix restricted to $S$.

**Time:** $O(2^n \cdot n^2)$. This matches the brute-force complexity in theory but the constant is much smaller in practice. More importantly, it uses only polynomial space (unlike the DP approach).

## Dynamic Programming over Subsets

The **Held-Karp algorithm** for TSP uses bitmask DP:

Let $\text{dp}[S][v]$ = minimum cost of a path starting at vertex 0, visiting all vertices in subset $S$, and ending at $v$.

**Recurrence:**

$$
\text{dp}[S][v] = \min_{u \in S \setminus \{v\}} \left(\text{dp}[S \setminus \{v\}][u] + w(u, v)\right)
$$

**Base case:** $\text{dp}[\{0\}][0] = 0$.

**Answer:** $\min_v (\text{dp}[V][v] + w(v, 0))$.

**Time:** $O(2^n \cdot n^2)$. **Space:** $O(2^n \cdot n)$.

This is much better than the $O(n!)$ brute force for TSP.

## Branch and Bound

**Branch and bound** systematically explores the solution space while pruning branches that provably cannot contain the optimum.

**Components:**

1. **Branching:** Split the problem into smaller subproblems (e.g., include/exclude a vertex).
2. **Bounding:** Compute a lower bound (for minimization) on each subproblem.
3. **Pruning:** Discard subproblems whose bound exceeds the best known solution.

The worst-case is still exponential, but pruning often eliminates most branches in practice.

## Faster Exact Algorithms

| Problem | Brute Force | Best Known | Technique |
|---------|------------|-----------|-----------|
| 3-SAT | $O(2^n)$ | $O(1.3070^n)$ | PPSZ algorithm |
| 3-Coloring | $O(3^n)$ | $O(1.3289^n)$ | Inclusion-exclusion |
| TSP | $O(n!)$ | $O(2^n \cdot n^2)$ | Held-Karp DP |
| Independent Set | $O(2^n)$ | $O(1.1996^n)$ | Measure and conquer |
| Subset Sum | $O(2^n)$ | $O(2^{n/2})$ | Meet in the middle |

## The Exponential Time Hypothesis

!!! warning "ETH (Impagliazzo-Paturi, 2001)"
    There exists a constant $\delta > 0$ such that 3-SAT cannot be solved in $O(2^{\delta n})$ time.

The **Strong ETH (SETH)** strengthens this: for every $\epsilon > 0$, there exists $k$ such that $k$-SAT cannot be solved in $O(2^{(1-\epsilon)n})$ time. SETH has implications for fine-grained complexity, ruling out certain polynomial improvements for problems like edit distance and longest common subsequence.

??? example "Example: Meet in the Middle for Subset Sum"
    **Instance:** $\{3, 7, 1, 8, 4, 2\}$, target $t = 14$.

    **Split:** $A = \{3, 7, 1\}$, $B = \{8, 4, 2\}$.

    **Subset sums of $A$:** $\{0, 3, 7, 1, 10, 4, 8, 11\}$.

    **Subset sums of $B$:** $\{0, 8, 4, 2, 12, 10, 6, 14\}$.

    **Lookup:** For $s_B = 4$, check $t - s_B = 10 \in A$-sums? Yes ($3 + 7 = 10$).

    **Solution:** $\{3, 7, 4\}$ sums to 14.

    **Savings:** Enumerated $2 \times 2^3 = 16$ subsets instead of $2^6 = 64$.

## Reference

- Fomin, F. V., & Kratsch, D. (2010). *Exact Exponential Algorithms*. Springer.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press.
