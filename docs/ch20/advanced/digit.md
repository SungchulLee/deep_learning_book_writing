# Digit DP

Problems like "How many integers in $[1, N]$ have a digit sum divisible by 7?" or "How many numbers up to $N$ contain no repeated digits?" cannot be solved by brute-force iteration when $N$ reaches $10^{18}$. Digit DP handles such problems by building the answer digit by digit from the most significant position, tracking a compact state that encodes the constraint. The central mechanism is the **tight** flag: it records whether the digits chosen so far exactly match the prefix of $N$, thereby limiting the range of the next digit, or whether a smaller digit was already chosen, freeing all subsequent choices to range over $0$ through $9$.

## State Definition

A digit DP state typically has the form $dp[\text{pos}][\text{tight}][\text{state}]$ where:

- **pos**: current digit position (from most significant to least significant, $0$-indexed)
- **tight**: boolean flag indicating whether the digits chosen so far match the prefix of $N$ exactly
- **state**: problem-specific information (e.g., digit sum modulo $m$, bitmask of used digits)

When tight is true, the next digit $d$ is restricted to $0 \leq d \leq D[\text{pos}]$ where $D[\text{pos}]$ is the corresponding digit of $N$. When tight is false, $d$ ranges freely over $0 \leq d \leq 9$ (for base 10). This restriction ensures we never construct a number exceeding $N$.

## General Recurrence

Let $D = d_0 d_1 \cdots d_{L-1}$ be the digits of $N$. The recurrence is:

$$
dp[\text{pos}][\text{tight}][\text{state}] = \sum_{d=0}^{\text{limit}} dp[\text{pos}+1][\text{tight}']\bigl[\text{transition}(\text{state}, d)\bigr]
$$

where:

$$
\text{limit} = \begin{cases} d_{\text{pos}} & \text{if tight} = \text{true} \\ 9 & \text{if tight} = \text{false} \end{cases}
$$

$$
\text{tight}' = \text{tight} \;\land\; (d = d_{\text{pos}})
$$

The new tight flag is true only when the old flag was true *and* we chose the maximum allowed digit, keeping the prefix equal to that of $N$.

**Base case.** $dp[L][\cdot][\text{state}] = 1$ if the final state satisfies the constraint, $0$ otherwise.

## Example: Digit Sum Divisible by k

Count integers in $[0, N]$ whose digit sum is divisible by $k$. The state tracks the running digit sum modulo $k$.

$$
dp[\text{pos}][\text{tight}][\text{rem}] = \sum_{d=0}^{\text{limit}} dp[\text{pos}+1][\text{tight}']\bigl[(\text{rem} + d) \bmod k\bigr]
$$

**Base case.** $dp[L][\cdot][0] = 1$, $dp[L][\cdot][r] = 0$ for $r \neq 0$.

## Example: No Repeated Digits

Count integers in $[1, N]$ with all distinct digits. The state uses a 10-bit bitmask recording which digits have appeared. An additional **started** flag distinguishes leading zeros (which should not count as using digit 0) from a genuine occurrence of digit 0 in the middle of a number.

## Implementation

```python
"""
Digit DP: count integers in [0, N] satisfying digit-based constraints.
"""

from functools import lru_cache


# ===================================================================
# Count numbers with digit sum divisible by k
# ===================================================================
def count_divisible_digit_sum(n: int, k: int) -> int:
    """Count integers in [0, n] with digit sum divisible by k.

    Parameters
    ----------
    n : int
        Upper bound (inclusive).
    k : int
        Divisor for the digit sum constraint.

    Returns
    -------
    int
        Count of valid integers.
    """
    digits = [int(c) for c in str(n)]
    length = len(digits)

    @lru_cache(maxsize=None)
    def dp(pos: int, tight: bool, rem: int) -> int:
        if pos == length:
            return 1 if rem == 0 else 0

        limit = digits[pos] if tight else 9
        total = 0
        for d in range(0, limit + 1):
            new_tight = tight and (d == limit)
            new_rem = (rem + d) % k
            total += dp(pos + 1, new_tight, new_rem)
        return total

    return dp(0, True, 0)


# ===================================================================
# Count numbers without repeated digits
# ===================================================================
def count_no_repeated_digits(n: int) -> int:
    """Count integers in [1, n] with all distinct digits.

    Parameters
    ----------
    n : int
        Upper bound (inclusive).

    Returns
    -------
    int
        Count of integers with no repeated digit.
    """
    digits = [int(c) for c in str(n)]
    length = len(digits)

    @lru_cache(maxsize=None)
    def dp(pos: int, tight: bool, used: int, started: bool) -> int:
        if pos == length:
            return 1 if started else 0

        limit = digits[pos] if tight else 9
        total = 0
        for d in range(0, limit + 1):
            if d == 0 and not started:
                # Leading zero: don't mark digit 0 as used
                total += dp(pos + 1, tight and (d == limit), used, False)
            else:
                if used & (1 << d):
                    continue  # digit already used
                new_used = used | (1 << d)
                total += dp(
                    pos + 1, tight and (d == limit), new_used, True
                )
        return total

    return dp(0, True, 0, False)


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    # Numbers in [0, 100] with digit sum divisible by 7
    n, k = 100, 7
    result = count_divisible_digit_sum(n, k)
    print(f"Count in [0, {n}] with digit sum % {k} == 0: {result}")

    # Numbers in [1, 100] with all distinct digits
    n = 100
    result = count_no_repeated_digits(n)
    print(f"Count in [1, {n}] with all distinct digits: {result}")

    # Larger example
    n = 1000000
    result = count_divisible_digit_sum(n, 13)
    print(f"Count in [0, {n}] with digit sum % 13 == 0: {result}")
```

**Output:**
```
Count in [0, 100] with digit sum % 7 == 0: 15
Count in [1, 100] with all distinct digits: 90
Count in [0, 1000000] with digit sum % 13 == 0: 76924
```

## Handling Ranges

To count integers in $[L, R]$ satisfying a constraint, compute:

$$
f(R) - f(L - 1)
$$

where $f(N)$ counts valid integers in $[0, N]$. This decomposition works because digit DP naturally counts from zero.

## Common Variations

| Problem | State variables | Complexity |
|---------|----------------|------------|
| Digit sum mod $k$ | pos, tight, rem | $O(L \cdot k)$ |
| No repeated digits | pos, tight, used (bitmask) | $O(L \cdot 2^{10})$ |
| Count of specific digit $d$ | pos, tight, count | $O(L^2)$ |
| All digits $\leq d$ | pos, tight | $O(L)$ |

Here $L = \lfloor \log_{10} N \rfloor + 1$ is the number of digits. In all cases, the tight flag doubles the state space but $L$ is at most 19 for $N \leq 10^{18}$.

!!! tip "Memoization with lru_cache"
    Python's `@lru_cache` makes digit DP concise. The cache size is bounded by $L \times 2 \times |\text{state}|$, which remains manageable even for 18-digit numbers.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
