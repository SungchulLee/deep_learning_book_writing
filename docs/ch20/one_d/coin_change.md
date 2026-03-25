# Coin Change

The coin change problem asks for the minimum number of coins needed to make a given amount, drawn from a set of denominations.  Unlike greedy approaches that always pick the largest coin first (which can fail for arbitrary denominations), dynamic programming considers all possible choices and guarantees an optimal solution.  This problem illustrates how DP handles minimization with an unbounded number of choices per state.

## Problem Statement

Given an array $\text{coins}[0..k-1]$ of coin denominations and a target amount $n$, find the minimum number of coins that sum to $n$.  Each coin denomination can be used unlimited times.  If no combination sums to $n$, return $-1$.

**Example:** With $\text{coins} = [1, 3, 4]$ and $n = 6$, the answer is 2 (using two coins of value 3).

## Recurrence Derivation

Let $dp[i]$ denote the minimum number of coins needed to make amount $i$.  To form amount $i$, we can use any coin $c$ with $c \le i$.  Using coin $c$ reduces the problem to making amount $i - c$, which requires $dp[i-c]$ coins.  Taking the minimum over all valid coins gives

$$
dp[i] = \min_{c \in \text{coins},\; c \le i} \bigl(dp[i - c] + 1\bigr) \quad \text{for } i \ge 1
$$

with base case

$$
dp[0] = 0
$$

since zero coins are needed to make amount 0.  If no coin fits (i.e., every coin exceeds $i$ or every $dp[i-c]$ is infinity), then $dp[i] = \infty$, indicating the amount is unreachable.

## Optimal Substructure

Suppose an optimal solution uses coin $c$ as one of its coins, leaving amount $i - c$ to be covered.  The coins covering $i - c$ must themselves be an optimal solution for amount $i - c$.  Otherwise, substituting a better solution for $i - c$ would reduce the total coin count, contradicting optimality.

## Tabulation

```python
"""
Coin change: minimum number of coins to make a target amount.
"""


# ===================================================================
# Approach 1: Tabulation (bottom-up)
# ===================================================================
def coin_change(coins: list[int], amount: int) -> int:
    """Minimum coins for amount. Time: O(amount * k), Space: O(amount)."""
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for c in coins:
            if c <= i and dp[i - c] + 1 < dp[i]:
                dp[i] = dp[i - c] + 1

    return dp[amount] if dp[amount] != float("inf") else -1
```

The outer loop runs $n$ times and the inner loop runs $k$ times (number of denominations), giving time complexity $O(nk)$ and space complexity $O(n)$.

## Reconstructing the Solution

To find which coins are used, track the coin chosen at each amount:

```python
# ===================================================================
# Approach 2: With reconstruction
# ===================================================================
def coin_change_with_coins(coins: list[int], amount: int) -> tuple[int, list[int]]:
    """Return minimum count and the actual coins used."""
    dp = [float("inf")] * (amount + 1)
    choice = [-1] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for c in coins:
            if c <= i and dp[i - c] + 1 < dp[i]:
                dp[i] = dp[i - c] + 1
                choice[i] = c

    if dp[amount] == float("inf"):
        return -1, []

    # Backtrack
    result = []
    remaining = amount
    while remaining > 0:
        result.append(choice[remaining])
        remaining -= choice[remaining]
    return dp[amount], result
```

## Why Greedy Fails

For some denomination sets, a greedy algorithm (always choosing the largest coin that fits) does not produce the minimum number of coins.

**Example:** With $\text{coins} = [1, 3, 4]$ and $n = 6$:

- Greedy: pick 4, then 1, then 1 — total 3 coins.
- Optimal: pick 3, then 3 — total 2 coins.

The greedy approach fails because picking the largest coin first may preclude a better combination.  Dynamic programming avoids this by evaluating all choices.

## Complexity

| Aspect | Value |
|--------|-------|
| Time | $O(nk)$ where $k = |\text{coins}|$ |
| Space | $O(n)$ |
| Subproblems | $n + 1$ |
| Choices per subproblem | $k$ |

```python
# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    test_cases = [
        ([1, 3, 4], 6),
        ([1, 5, 10, 25], 30),
        ([2], 3),
    ]
    for coins, amount in test_cases:
        count, used = coin_change_with_coins(coins, amount)
        print(f"coins={coins}, amount={amount} -> {count} coins: {used}")
```

**Output:**
```
coins=[1, 3, 4], amount=6 -> 2 coins: [3, 3]
coins=[1, 5, 10, 25], amount=30 -> 2 coins: [5, 25]
coins=[2], amount=3 -> -1 coins: []
```

!!! tip "Counting combinations vs minimum coins"
    A related problem asks for the **number of ways** to make amount $n$ (not the minimum coins).  That variant uses a different recurrence — additive rather than min — and requires care to avoid counting permutations of the same combination.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 14. MIT Press.
