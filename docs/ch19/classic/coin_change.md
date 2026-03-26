# Coin Change (Greedy)

When making change with a set of coin denominations, the natural greedy strategy is to always choose the largest coin that does not exceed the remaining amount. This works correctly for standard currency systems (e.g., US coins: 1, 5, 10, 25 cents) but fails for arbitrary denomination sets. Understanding when greedy succeeds and when it fails motivates the use of dynamic programming for the general case.

## The Greedy Approach

Given denominations $d_1 > d_2 > \cdots > d_k$ and a target amount $A$, the greedy algorithm repeatedly selects the largest denomination that fits:

1. Set remaining $= A$, count $= 0$.
2. For each denomination $d_i$ (largest first): use $\lfloor \text{remaining} / d_i \rfloor$ coins, update remaining.
3. If remaining $= 0$, return count. Otherwise, no solution exists (with greedy).

## When Greedy Works

A coin system is called **canonical** if the greedy algorithm always produces the minimum number of coins. The US system $\{1, 5, 10, 25\}$ is canonical.

!!! warning "Greedy Fails for Non-Canonical Systems"
    For denominations $\{1, 3, 4\}$ and target $6$: greedy picks $4 + 1 + 1 = 3$ coins, but $3 + 3 = 2$ coins is optimal.

## Greedy Correctness for Standard Systems

For the US coin system, the greedy approach works because no optimal solution uses coins in a way that a larger coin could replace. For example, no optimal solution uses five pennies (they would be replaced by a nickel), and no optimal solution uses two dimes and a nickel (they would be replaced by a quarter). These dominance conditions guarantee greedy optimality.

## The General Solution: Dynamic Programming

For arbitrary denominations, the minimum number of coins to make amount $i$ satisfies:

$$
\text{dp}[i] = \min_{c \in \text{coins},\; c \le i} (\text{dp}[i - c] + 1)
$$

with base case $\text{dp}[0] = 0$ and $\text{dp}[i] = \infty$ for amounts that cannot be formed.

## Implementation

```python
"""
Coin change: greedy vs dynamic programming.

Shows when the greedy approach works (canonical systems) and
when DP is needed (arbitrary denominations).
"""

# === Greedy Coin Change ===

def coin_change_greedy(coins: list[int], amount: int) -> list[int]:
    """Make change using the greedy strategy (largest first).

    Args:
        coins: Available denominations (sorted descending).
        amount: Target amount.

    Returns:
        List of coins used (may not be optimal for non-canonical systems).
    """
    result = []
    remaining = amount
    for coin in sorted(coins, reverse=True):
        while remaining >= coin:
            result.append(coin)
            remaining -= coin
    return result if remaining == 0 else []


# === DP Coin Change ===

def coin_change_dp(coins: list[int], amount: int) -> int:
    """Find minimum number of coins to make the target amount.

    Args:
        coins: Available denominations.
        amount: Target amount.

    Returns:
        Minimum number of coins, or -1 if impossible.
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for i in range(1, amount + 1):
        for c in coins:
            if c <= i and dp[i - c] + 1 < dp[i]:
                dp[i] = dp[i - c] + 1

    return dp[amount] if dp[amount] != float('inf') else -1


# === Demonstration ===

if __name__ == "__main__":
    # Canonical system: greedy works
    us_coins = [25, 10, 5, 1]
    amount = 63
    greedy_result = coin_change_greedy(us_coins, amount)
    dp_result = coin_change_dp(us_coins, amount)
    print(f"US coins, amount={amount}:")
    print(f"  Greedy: {len(greedy_result)} coins {greedy_result}")
    print(f"  DP:     {dp_result} coins")

    # Non-canonical system: greedy fails
    coins = [1, 3, 4]
    amount = 6
    greedy_result = coin_change_greedy(coins, amount)
    dp_result = coin_change_dp(coins, amount)
    print(f"\nCoins {coins}, amount={amount}:")
    print(f"  Greedy: {len(greedy_result)} coins {greedy_result}")
    print(f"  DP:     {dp_result} coins")
```

**Output:**

```
US coins, amount=63:
  Greedy: 6 coins [25, 25, 10, 1, 1, 1]
  DP:     6 coins

Coins [1, 3, 4], amount=6:
  Greedy: 3 coins [4, 1, 1]
  DP:     2 coins
```

For the US system, greedy and DP agree. For $\{1, 3, 4\}$, greedy uses 3 coins while the optimal DP solution uses only 2 coins ($3 + 3$).

## Complexity

| Algorithm | Time | Space |
|-----------|:----:|:-----:|
| Greedy    | $O(k)$ where $k$ = number of denominations | $O(1)$ |
| DP        | $O(A \cdot k)$ where $A$ = amount | $O(A)$ |

The greedy approach is faster but only correct for canonical systems. The DP approach always finds the optimum but requires time proportional to the amount.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 16: Greedy Algorithms.
- Kozen, D., & Zaks, S. (1994). Optimal bounds for the change-making problem. *Theoretical Computer Science*, 123(2), 377--388.
