# Reconstructing Solutions

A dynamic programming algorithm typically computes the **value** of an optimal solution — the minimum cost, the maximum profit, or the length of a longest subsequence.  However, most applications need the actual solution itself: the specific cuts, the chosen items, or the aligned characters.  Reconstructing the solution from a completed DP table is a fundamental skill that complements the design of the recurrence.

## The Reconstruction Problem

After filling a DP table, every cell contains the optimal value for its corresponding subproblem.  The reconstruction task is to trace back through the table and recover the sequence of decisions that led to the optimal value.

Two standard approaches accomplish this:

1. **Backtracking through the DP table** — starting from the final answer cell, examine which choice led to the stored value and follow that path backward.
2. **Storing explicit decision pointers** — during the forward pass, record the choice made at each cell in a separate table, then follow these pointers to recover the solution.

Both approaches add only $O(1)$ work per cell, so reconstruction does not change the overall time complexity.

## Approach 1: Backtracking Through the Table

Consider the longest common subsequence (LCS) problem with strings $X = x_1 x_2 \cdots x_m$ and $Y = y_1 y_2 \cdots y_n$.  The recurrence is

$$
c[i][j] = \begin{cases} 0 & \text{if } i = 0 \text{ or } j = 0 \\ c[i-1][j-1] + 1 & \text{if } x_i = y_j \\ \max(c[i-1][j],\; c[i][j-1]) & \text{if } x_i \neq y_j \end{cases}
$$

After filling the table, start at $c[m][n]$ and trace backward:

- If $x_i = y_j$, the character $x_i$ is part of the LCS.  Move to $c[i-1][j-1]$.
- If $c[i-1][j] \ge c[i][j-1]$, move to $c[i-1][j]$.
- Otherwise, move to $c[i][j-1]$.

Continue until $i = 0$ or $j = 0$.  The characters collected (in reverse order) form the LCS.

```python
"""
Reconstruct the longest common subsequence by backtracking
through the DP table.
"""


# ===================================================================
# LCS with reconstruction
# ===================================================================
def lcs_with_reconstruction(x: str, y: str) -> str:
    """Compute LCS length and reconstruct the subsequence."""
    m, n = len(x), len(y)
    c = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
            else:
                c[i][j] = max(c[i - 1][j], c[i][j - 1])

    # Backtrack to reconstruct the LCS
    result = []
    i, j = m, n
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1]:
            result.append(x[i - 1])
            i -= 1
            j -= 1
        elif c[i - 1][j] >= c[i][j - 1]:
            i -= 1
        else:
            j -= 1

    return "".join(reversed(result))


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    x = "ABCBDAB"
    y = "BDCAB"
    lcs = lcs_with_reconstruction(x, y)
    print(f"X = {x}")
    print(f"Y = {y}")
    print(f"LCS = {lcs} (length {len(lcs)})")
```

**Output:**
```
X = ABCBDAB
Y = BDCAB
LCS = BCAB (length 4)
```

## Approach 2: Decision Pointers

Instead of re-deriving decisions during backtracking, store them during the forward pass.  For each cell, record which branch of the recurrence was taken.

For the rod cutting problem with recurrence

$$
r[n] = \max_{1 \le i \le n} \bigl(p_i + r[n - i]\bigr)
$$

maintain an auxiliary array $s[n]$ that records the first cut length $i$ that achieves the maximum:

```python
"""
Rod cutting with decision pointers for solution reconstruction.
"""


# ===================================================================
# Rod cutting with reconstruction
# ===================================================================
def rod_cutting(prices: list[int], n: int) -> tuple[int, list[int]]:
    """Return maximum revenue and the list of cut lengths."""
    r = [0] * (n + 1)
    s = [0] * (n + 1)

    for j in range(1, n + 1):
        best = -1
        for i in range(1, j + 1):
            if prices[i - 1] + r[j - i] > best:
                best = prices[i - 1] + r[j - i]
                s[j] = i
        r[j] = best

    # Reconstruct the cuts
    cuts = []
    remaining = n
    while remaining > 0:
        cuts.append(s[remaining])
        remaining -= s[remaining]

    return r[n], cuts


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    prices = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30]
    n = 7
    revenue, cuts = rod_cutting(prices, n)
    print(f"Rod length: {n}")
    print(f"Maximum revenue: {revenue}")
    print(f"Cuts: {cuts}")
```

**Output:**
```
Rod length: 7
Maximum revenue: 18
Cuts: [1, 6]
```

The pointer array $s$ stores the decision at each step, making reconstruction a simple loop that follows the recorded choices.

## General Reconstruction Pattern

Regardless of the specific problem, reconstruction follows a common pattern:

1. **Start** at the cell containing the final answer (e.g., $dp[n]$, $dp[m][n]$, or $dp[0][n-1]$).
2. **Determine** which choice produced the value in that cell — either by re-evaluating the recurrence or by consulting a stored decision.
3. **Record** the choice (the item taken, the cut made, the character matched).
4. **Move** to the subproblem cell implied by that choice.
5. **Repeat** until a base case is reached.

The result is the sequence of decisions in reverse order (from the final state back to the base case), which is then reversed to obtain the forward-order solution.

!!! tip "When to store pointers vs backtrack"
    Storing decision pointers uses $O(1)$ extra space per cell but makes reconstruction trivial.  Backtracking without pointers saves memory but requires re-evaluating the recurrence at each step.  For problems where the recurrence involves a $\min$ or $\max$ over many choices (like matrix chain multiplication), storing pointers is usually more convenient.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
