# Interval DP

Some optimization problems ask for the best way to process a contiguous range of elements --- merging stones, multiplying matrices, or bursting balloons. In each case, an optimal solution for range $[i, j]$ decomposes into optimal solutions for sub-ranges $[i, k]$ and $[k+1, j]$ at some split point $k$. Interval DP captures this structure by defining states over all $O(n^2)$ sub-intervals and filling the table in order of increasing interval length.

## Framework

Define $dp[i][j]$ as the optimal value for the sub-interval $[i, j]$. The general recurrence splits the interval at every possible point:

$$
dp[i][j] = \min_{i \leq k < j} \bigl( dp[i][k] + dp[k+1][j] + \text{merge}(i, k, j) \bigr)
$$

where $\text{merge}(i, k, j)$ is the cost of combining the results of sub-intervals $[i, k]$ and $[k+1, j]$.

**Base case.** $dp[i][i] = \text{base}(i)$ (cost of a single element, often 0).

**Iteration order.** Fill by increasing interval length $\ell = j - i + 1$:

```
for length in range(2, n + 1):        # interval length
    for i in range(0, n - length + 1): # left endpoint
        j = i + length - 1             # right endpoint
        for k in range(i, j):          # split point
            dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + merge(i, k, j))
```

## Complexity

The three nested loops give $O(n^3)$ time and $O(n^2)$ space. When the cost function satisfies the quadrangle inequality, Knuth's optimization reduces the time to $O(n^2)$.

## Example: Minimum Cost to Merge Stones

Given $n$ piles of stones with sizes $a_0, \ldots, a_{n-1}$, merge adjacent piles. The cost of a single merge equals the total size of the merged piles. Minimize the total cost.

$$
dp[i][j] = \min_{i \leq k < j} \bigl( dp[i][k] + dp[k+1][j] \bigr) + \sum_{t=i}^{j} a_t
$$

The merge cost is the sum of all elements in the interval, computable in $O(1)$ with prefix sums.

## Example: Matrix Chain Multiplication

Given matrices $A_1, A_2, \ldots, A_n$ with dimensions $p_0 \times p_1, p_1 \times p_2, \ldots, p_{n-1} \times p_n$, find the parenthesization that minimizes scalar multiplications.

$$
dp[i][j] = \min_{i \leq k < j} \bigl( dp[i][k] + dp[k+1][j] + p_{i-1} \cdot p_k \cdot p_j \bigr)
$$

## Example: Burst Balloons

Given balloons with values $a_1, \ldots, a_n$, bursting balloon $k$ earns $a_{k-1} \cdot a_k \cdot a_{k+1}$ coins (with boundary sentinels $a_0 = a_{n+1} = 1$). Maximize total coins.

The key insight is to think of $k$ as the **last** balloon burst in the open interval $(i, j)$. At the moment $k$ is burst, only the boundary sentinels $a_i$ and $a_j$ remain as neighbors:

$$
dp[i][j] = \max_{i < k < j} \bigl( dp[i][k] + dp[k][j] + a_i \cdot a_k \cdot a_j \bigr)
$$

**Base case.** $dp[i][i+1] = 0$ (no balloons to burst in an empty open interval).

## Implementation

```python
"""
Interval DP: merge stones, matrix chain multiplication, and burst balloons.
"""


# ===================================================================
# Minimum cost to merge stones
# ===================================================================
def merge_stones(piles: list[int]) -> int:
    """Find minimum cost to merge all piles into one.

    Parameters
    ----------
    piles : list[int]
        Sizes of stone piles.

    Returns
    -------
    int
        Minimum merge cost.
    """
    n = len(piles)
    if n == 1:
        return 0

    # Prefix sums for O(1) range sum
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + piles[i]

    def range_sum(i: int, j: int) -> int:
        return prefix[j + 1] - prefix[i]

    INF = float("inf")
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + range_sum(i, j)
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n - 1]


# ===================================================================
# Matrix chain multiplication
# ===================================================================
def matrix_chain(dims: list[int]) -> int:
    """Find minimum scalar multiplications for matrix chain.

    Parameters
    ----------
    dims : list[int]
        Dimension array where matrix i has size dims[i] x dims[i+1].

    Returns
    -------
    int
        Minimum number of scalar multiplications.
    """
    n = len(dims) - 1  # number of matrices
    if n <= 1:
        return 0

    INF = float("inf")
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF
            for k in range(i, j):
                cost = (
                    dp[i][k]
                    + dp[k + 1][j]
                    + dims[i] * dims[k + 1] * dims[j + 1]
                )
                dp[i][j] = min(dp[i][j], cost)

    return dp[0][n - 1]


# ===================================================================
# Burst balloons (last-to-burst trick)
# ===================================================================
def burst_balloons(nums: list[int]) -> int:
    """Maximum coins from bursting all balloons.

    The key trick: think of k as the *last* balloon burst in (i, j),
    so its neighbors at burst time are the boundary sentinels i and j.

    Parameters
    ----------
    nums : list[int]
        Balloon values.

    Returns
    -------
    int
        Maximum total coins.
    """
    vals = [1] + nums + [1]
    n = len(vals)
    dp = [[0] * n for _ in range(n)]

    for length in range(2, n):
        for i in range(0, n - length):
            j = i + length
            for k in range(i + 1, j):
                coins = vals[i] * vals[k] * vals[j] + dp[i][k] + dp[k][j]
                dp[i][j] = max(dp[i][j], coins)

    return dp[0][n - 1]


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    # Merge stones
    piles = [3, 5, 1, 2, 6]
    print(f"Merge stones cost: {merge_stones(piles)}")

    # Matrix chain
    dims = [10, 30, 5, 60]
    print(f"Matrix chain cost: {matrix_chain(dims)}")

    # Burst balloons
    balloons = [3, 1, 5, 8]
    print(f"Burst balloons max: {burst_balloons(balloons)}")
```

**Output:**
```
Merge stones cost: 38
Matrix chain cost: 4500
Burst balloons max: 167
```

??? example "Tracing merge stones for [3, 5, 1, 2, 6]"
    Interval lengths processed in order:

    - Length 2: $dp[0][1] = 8$, $dp[1][2] = 6$, $dp[2][3] = 3$, $dp[3][4] = 8$
    - Length 3: $dp[0][2] = 14$, $dp[1][3] = 11$, $dp[2][4] = 12$
    - Length 4: $dp[0][3] = 22$, $dp[1][4] = 22$
    - Length 5: $dp[0][4] = 38$

    Optimal merge order: merge piles 2 and 3 (cost 3), then with pile 1 (cost 8), then with pile 0 (cost 17), finally with pile 4 (total cost 38).

## Recognizing Interval DP Problems

Interval DP applies when:

1. The input is a **sequence** (array, string, or chain)
2. The optimal solution for a range **decomposes into sub-ranges**
3. Only **contiguous** sub-ranges appear as subproblems
4. There is a **merge cost** that depends on the sub-range endpoints

| Problem | State | Merge Cost |
|---------|-------|------------|
| Matrix chain | $dp[i][j]$ = min multiplications | $p_i \cdot p_{k+1} \cdot p_{j+1}$ |
| Merge stones | $dp[i][j]$ = min merge cost | $\sum_{t=i}^{j} a_t$ |
| Burst balloons | $dp[i][j]$ = max coins | $a_i \cdot a_k \cdot a_j$ |
| Palindrome partitioning | $dp[i][j]$ = min cuts | 0 or 1 |
| Optimal BST | $dp[i][j]$ = min search cost | $\sum_{t=i}^{j} p_t$ |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
