# Fibonacci

The Fibonacci sequence is the simplest and most widely used example for introducing dynamic programming.  Computing Fibonacci numbers with naive recursion leads to exponential time, while memoization and tabulation each reduce this to linear time.  This progression from exponential to linear illustrates the core benefit of dynamic programming and serves as a template for approaching more complex problems.

## Recurrence Definition

The Fibonacci sequence is defined by the recurrence

$$
F(n) = F(n-1) + F(n-2), \quad F(0) = 0, \quad F(1) = 1
$$

The first several values are $0, 1, 1, 2, 3, 5, 8, 13, 21, 34, \ldots$

## Approach 1: Naive Recursion

The most direct implementation translates the mathematical recurrence into code.  Each call spawns two recursive calls, producing an exponentially growing recursion tree.

```python
"""
Three approaches to computing Fibonacci numbers: naive recursion,
memoization (top-down DP), and tabulation (bottom-up DP).
"""


# ===================================================================
# Approach 1: Naive recursion
# ===================================================================
def fib_recursive(n: int) -> int:
    """Compute F(n) using naive recursion. Time: O(phi^n), Space: O(n)."""
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)
```

The recursion tree for $F(5)$ reveals the redundancy:

```
                    F(5)
                /         \
            F(4)           F(3)
           /    \         /    \
        F(3)   F(2)    F(2)   F(1)
       /  \    / \     / \
    F(2) F(1) F(1) F(0) F(1) F(0)
    / \
 F(1) F(0)
```

The number of calls $T(n)$ satisfies $T(n) = T(n-1) + T(n-2) + 1$, which grows as $O(\phi^n)$ where $\phi = (1+\sqrt{5})/2 \approx 1.618$.  The space complexity is $O(n)$ due to the maximum depth of the call stack.

## Approach 2: Memoization (Top-Down)

Memoization preserves the recursive structure but caches each result so that no subproblem is solved more than once.

```python
# ===================================================================
# Approach 2: Memoization (top-down DP)
# ===================================================================
def fib_memo(n: int, memo: dict[int, int] | None = None) -> int:
    """Compute F(n) using memoization. Time: O(n), Space: O(n)."""
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]
```

Each of the $n + 1$ distinct subproblems $F(0), F(1), \ldots, F(n)$ is computed exactly once and each computation takes $O(1)$ time, giving total time $O(n)$ and space $O(n)$ for the memoization table plus the call stack.

## Approach 3: Tabulation (Bottom-Up)

Tabulation eliminates recursion entirely by filling a table from the smallest subproblems upward.

```python
# ===================================================================
# Approach 3: Tabulation (bottom-up DP)
# ===================================================================
def fib_tabulation(n: int) -> int:
    """Compute F(n) using tabulation. Time: O(n), Space: O(n)."""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

The table is filled left to right in a single pass, so the time is $O(n)$ and the space is $O(n)$ for the array.

## Space Optimization

Since $F(n)$ depends only on the two preceding values, the table can be replaced by two variables, reducing space to $O(1)$.

```python
# ===================================================================
# Space-optimized tabulation
# ===================================================================
def fib_optimized(n: int) -> int:
    """Compute F(n) with O(1) space. Time: O(n), Space: O(1)."""
    if n <= 1:
        return n
    prev2, prev1 = 0, 1
    for _ in range(2, n + 1):
        prev2, prev1 = prev1, prev2 + prev1
    return prev1
```

## Complexity Comparison

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Naive recursion | $O(\phi^n)$ | $O(n)$ | Exponential due to overlapping subproblems |
| Memoization | $O(n)$ | $O(n)$ | Each subproblem solved once; call stack depth $n$ |
| Tabulation | $O(n)$ | $O(n)$ | No recursion overhead; simple loop |
| Space-optimized | $O(n)$ | $O(1)$ | Only two variables needed |

```python
# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    for n in [5, 10, 20]:
        r = fib_recursive(n)
        m = fib_memo(n)
        t = fib_tabulation(n)
        o = fib_optimized(n)
        assert r == m == t == o
        print(f"F({n}) = {o}")
```

**Output:**
```
F(5) = 5
F(10) = 55
F(20) = 6765
```

!!! tip "Pattern for DP problems"
    The Fibonacci example establishes a workflow that applies to nearly every DP problem: (1) write the recurrence, (2) identify overlapping subproblems, (3) add memoization or convert to tabulation, and (4) optimize space if only a few previous states are needed.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
