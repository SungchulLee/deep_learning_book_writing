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

## Exercises

**Exercise 1.**
Compare the time and space complexities of the 0/1 knapsack problem and the unbounded knapsack problem. Why does the unbounded version have a simpler recurrence?

??? success "Solution to Exercise 1"
    **0/1 knapsack**: each item can be used at most once. State: $dp[i][w]$ = max value using items $1 \ldots i$ with capacity $w$. Time: $O(nW)$ where $n$ is the number of items and $W$ is the capacity. Space: $O(nW)$, reducible to $O(W)$ with rolling array. **Unbounded knapsack**: each item can be used unlimited times. State: $dp[w]$ = max value with capacity $w$ (no item dimension needed). Time: $O(nW)$. Space: $O(W)$. The unbounded version is simpler because the decision at capacity $w$ does not depend on which items were previously used -- we can always consider every item again. This eliminates the item dimension from the state, reducing the recurrence to $dp[w] = \max_i(dp[w - w_i] + v_i)$. $\square$

---

**Exercise 2.**
The longest common subsequence (LCS) of two strings of lengths $m$ and $n$ can be computed in $O(mn)$ time and $O(mn)$ space. Describe how to reduce the space to $O(\min(m, n))$ while maintaining $O(mn)$ time.

??? success "Solution to Exercise 2"
    The standard DP table is $dp[i][j]$ where $dp[i][j]$ depends only on $dp[i-1][j]$, $dp[i][j-1]$, and $dp[i-1][j-1]$. Since row $i$ depends only on row $i-1$, we need only two rows at a time: the current row and the previous row. Keep two 1D arrays of size $\min(m, n) + 1$ (iterate over the longer string in the outer loop). At each step, compute the current row from the previous row, then swap. Space: $O(\min(m, n))$. This gives only the LCS length, not the actual subsequence. To reconstruct the LCS in reduced space, use Hirschberg's algorithm: divide the problem in half along the longer dimension, find the midpoint of the LCS using two forward/backward passes with $O(\min(m,n))$ space each, and recurse. Total time remains $O(mn)$; space is $O(\min(m,n))$. $\square$

---

**Exercise 3.**
Explain why the edit distance DP has $O(mn)$ time complexity and describe when this is too slow in practice.

??? success "Solution to Exercise 3"
    Edit distance between strings of length $m$ and $n$ fills an $(m+1) \times (n+1)$ table, with each cell computed in $O(1)$ from three neighbors. Total: $O(mn)$. This is too slow when both strings are long: for DNA sequences with $m = n = 10^6$, the DP requires $10^{12}$ operations ($\sim$hours). Alternatives for such cases: (1) banded DP -- if the edit distance is known to be small ($d \ll m$), only compute cells within distance $d$ of the diagonal, giving $O(md)$ time. (2) Approximate algorithms -- locality-sensitive hashing for approximate nearest neighbor in edit distance space. (3) For exact computation on very long strings, four-Russians speedup achieves $O(mn / \log^2 n)$. $\square$

---

**Exercise 4.**
The matrix chain multiplication problem has $O(n^3)$ time and $O(n^2)$ space. Derive these complexities from the recurrence $dp[i][j] = \min_{i \le k < j} (dp[i][k] + dp[k+1][j] + p_{i-1} p_k p_j)$.

??? success "Solution to Exercise 4"
    The state space has $O(n^2)$ entries: all pairs $(i, j)$ with $1 \le i \le j \le n$, giving $\binom{n}{2} + n = O(n^2)$ entries. Each entry $dp[i][j]$ requires trying $O(j - i)$ split points $k$, each costing $O(1)$. The total work is $\sum_{i=1}^{n} \sum_{j=i}^{n} (j - i) = \sum_{l=0}^{n-1} l(n - l) = O(n^3)$ (where $l = j - i$ is the chain length). Space: the DP table has $O(n^2)$ entries, each storing one value. Therefore, time is $O(n^3)$ and space is $O(n^2)$. Knuth's optimization (the optimal split point $k^*[i][j]$ is monotone) reduces the time to $O(n^2)$ for this specific problem. $\square$

---

**Exercise 5.**
Describe the tradeoff between top-down (memoized) and bottom-up (tabulation) DP. When does each approach outperform the other?

??? success "Solution to Exercise 5"
    **Top-down (memoization)**: recursion with caching. Only computes states that are actually needed. Overhead: function call stack and hash map lookups. **Bottom-up (tabulation)**: iterates through all states in dependency order. Computes all states, even those not needed for the answer. Overhead: none (simple loops and array access). Top-down wins when: (1) the state space is sparse -- many states are unreachable, so memoization avoids computing them (e.g., knapsack with large $W$ but few feasible weight combinations); (2) the dependency order is complex and hard to determine statically. Bottom-up wins when: (1) most states are visited -- the overhead of recursion and hash lookups is wasted; (2) cache locality matters -- sequential array access is faster than random memoization table access; (3) the state space is dense and small. In practice, bottom-up is preferred for standard DP problems (knapsack, LCS, edit distance); top-down is preferred for problems with irregular state spaces (game tree search, sparse DP). $\square$
