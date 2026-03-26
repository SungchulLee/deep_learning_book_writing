# DP Problem Complexities

Dynamic programming transforms exponential brute-force searches into polynomial-time
algorithms by storing solutions to overlapping subproblems. Understanding the time and
space complexity of each classic DP pattern helps you choose the right formulation and
predict whether a solution will fit within contest or interview time limits.

## One-Dimensional DP

These problems define a single state variable, typically an index $i$ into an array or
a remaining quantity.

| Problem | State | Recurrence | Time | Space |
|---|---|---|---|---|
| Fibonacci | $F(i)$ | $F(i) = F(i-1) + F(i-2)$ | $O(n)$ | $O(1)$ |
| Climbing Stairs | $dp[i]$ | $dp[i] = dp[i-1] + dp[i-2]$ | $O(n)$ | $O(1)$ |
| House Robber | $dp[i]$ | $dp[i] = \max(dp[i-1],\; dp[i-2] + a_i)$ | $O(n)$ | $O(1)$ |
| Coin Change | $dp[a]$ for amount $a$ | $dp[a] = \min_{c \in C}(dp[a - c]) + 1$ | $O(n \cdot W)$ | $O(W)$ |
| Word Break | $dp[i]$ | $dp[i] = \bigvee_{j < i} (dp[j] \wedge s[j:i] \in D)$ | $O(n^2 \cdot L)$ | $O(n)$ |
| Maximum Subarray | $dp[i]$ | $dp[i] = \max(a_i,\; dp[i-1] + a_i)$ | $O(n)$ | $O(1)$ |
| Longest Increasing Subseq. | $dp[i]$ | $dp[i] = \max_{j < i,\; a_j < a_i}(dp[j]) + 1$ | $O(n^2)$ | $O(n)$ |
| LIS (binary search) | tails array | patience sorting | $O(n \log n)$ | $O(n)$ |

Here $n$ is the input size, $W$ is the target amount, $C$ is the coin set, $L$ is
maximum word length, and $D$ is the dictionary.

## Two-Dimensional DP

Two state variables arise when the problem involves pairs of indices, a grid, or two
sequences.

| Problem | State | Time | Space | Space (optimized) |
|---|---|---|---|---|
| Longest Common Subsequence | $dp[i][j]$ | $O(mn)$ | $O(mn)$ | $O(\min(m,n))$ |
| Edit Distance | $dp[i][j]$ | $O(mn)$ | $O(mn)$ | $O(\min(m,n))$ |
| 0/1 Knapsack | $dp[i][w]$ | $O(nW)$ | $O(nW)$ | $O(W)$ |
| Grid Unique Paths | $dp[i][j]$ | $O(mn)$ | $O(mn)$ | $O(n)$ |
| Palindrome Subsequence | $dp[i][j]$ | $O(n^2)$ | $O(n^2)$ | $O(n)$ |
| Matrix Chain Multiplication | $dp[i][j]$ | $O(n^3)$ | $O(n^2)$ | $O(n^2)$ |
| Rod Cutting | $dp[i]$ | $O(n^2)$ | $O(n)$ | $O(n)$ |

For two-sequence problems, $m$ and $n$ are the respective sequence lengths. For the
knapsack, $W$ is the capacity. Matrix chain multiplication requires $O(n^3)$ because
each subproblem $(i, j)$ iterates over $O(n)$ split points.

## Why Space Optimization Works

Many 2D DP tables can be compressed to one or two rows because each cell depends only
on the current and previous rows. Consider the edit distance recurrence:

$$
dp[i][j] = \min\bigl(dp[i-1][j] + 1,\; dp[i][j-1] + 1,\; dp[i-1][j-1] + \delta\bigr)
$$

where $\delta = 0$ if $s_1[i] = s_2[j]$ and $\delta = 1$ otherwise. Since row $i$
depends only on row $i-1$, we need just two rows of length $\min(m, n) + 1$, reducing
space from $O(mn)$ to $O(\min(m, n))$.

## Advanced DP Patterns

| Pattern | Typical Time | Typical Space | Example Problems |
|---|---|---|---|
| Bitmask DP | $O(2^n \cdot n)$ | $O(2^n)$ | TSP, Hamiltonian path |
| Digit DP | $O(d \cdot s \cdot 2)$ | $O(d \cdot s)$ | Count numbers with property |
| Interval DP | $O(n^3)$ | $O(n^2)$ | Optimal BST, balloon burst |
| Tree DP | $O(n)$ | $O(n)$ | Max independent set on tree |
| DP on DAGs | $O(V + E)$ | $O(V)$ | Shortest path in DAG |
| Knuth optimization | $O(n^2)$ | $O(n^2)$ | Optimal BST (from $O(n^3)$) |

Here $d$ is the number of digits, $s$ is the number of states per digit position,
$V$ is the vertex count, and $E$ is the edge count.

!!! tip "Bitmask DP Feasibility"
    Bitmask DP is practical only when $n \le 20$ because $2^{20} \approx 10^6$.
    For $n = 25$, the state space exceeds $3 \times 10^7$, which is marginal under
    typical time limits.

## Pseudo-polynomial vs Polynomial

Some DP algorithms run in time proportional to the numeric value of the input rather
than the input size in bits. These are called **pseudo-polynomial**.

| Algorithm | Time | Polynomial? | Why |
|---|---|---|---|
| 0/1 Knapsack | $O(nW)$ | Pseudo-polynomial | $W$ is a value, not a count |
| Coin Change | $O(nW)$ | Pseudo-polynomial | Same reason |
| Subset Sum | $O(nS)$ | Pseudo-polynomial | $S$ is target sum |
| LCS | $O(mn)$ | Polynomial | $m, n$ are input lengths |
| Matrix Chain | $O(n^3)$ | Polynomial | $n$ is number of matrices |

The distinction matters for NP-hardness: the knapsack problem is NP-hard, yet the DP
solution is efficient when $W$ is small relative to $n$.

## Complexity Derivation Example

Consider the 0/1 Knapsack. Let $n$ items have weights $w_1, \ldots, w_n$ and values
$v_1, \ldots, v_n$. The recurrence is:

$$
dp[i][w] = \max\bigl(dp[i-1][w],\; dp[i-1][w - w_i] + v_i\bigr)
$$

- **States**: $n \times (W + 1)$ entries in the table.
- **Transition**: Each state requires $O(1)$ work (a single comparison).
- **Total time**: $O(nW)$.
- **Space**: $O(nW)$ for the full table, or $O(W)$ if only the previous row is kept.

!!! warning "Common Pitfall"
    Forgetting to handle the base case $dp[0][w] = 0$ for all $w$ leads to
    incorrect results. Always initialize the boundary of the DP table before
    filling interior cells.

## Reference

- [Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Kleinberg, J. and Tardos, E. *Algorithm Design*. Pearson, 2005.
