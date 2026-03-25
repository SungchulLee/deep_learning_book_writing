# Memoization

Memoization is a technique that stores the results of expensive recursive calls and returns the cached result when the same inputs occur again. It transforms exponential-time recursive algorithms into polynomial-time ones by eliminating redundant computation — the key insight behind dynamic programming.

## The Problem: Redundant Computation

Naive recursive Fibonacci computes the same values repeatedly. For example, `fib(5)` calls `fib(3)` twice and `fib(2)` three times:

$$
T(n) = T(n-1) + T(n-2) + O(1) \implies T(n) = O(2^n)
$$

## The Solution: Cache Results

Store each computed result in a dictionary. Before recursing, check whether the answer is already cached:

```python
"""Memoization demonstrated with Fibonacci numbers."""


# === Fibonacci without Memoization ===

def fib_naive(n):
    """Compute Fibonacci number — exponential time."""
    if n <= 1:
        return n
    return fib_naive(n - 1) + fib_naive(n - 2)


# === Fibonacci with Memoization ===

def fib_memo(n, cache=None):
    """Compute Fibonacci number — O(n) time with memoization."""
    if cache is None:
        cache = {}
    if n in cache:
        return cache[n]
    if n <= 1:
        return n
    cache[n] = fib_memo(n - 1, cache) + fib_memo(n - 2, cache)
    return cache[n]


# === Main ===

if __name__ == "__main__":
    print("With memoization:")
    for i in [10, 20, 30, 40]:
        print(f"  fib({i}) = {fib_memo(i)}")
```

**Output:**
```
With memoization:
  fib(10) = 55
  fib(20) = 6765
  fib(30) = 832040
  fib(40) = 102334155
```

## Complexity Improvement

| Approach | Time | Space |
|---|---|---|
| Naive recursion | $O(2^n)$ | $O(n)$ stack |
| With memoization | $O(n)$ | $O(n)$ cache + stack |

Memoization reduces time from exponential to linear because each subproblem is solved only once. The space cost is the cache, which stores $O(n)$ entries.

## When to Use Memoization

Memoization is effective when a recursive function has **overlapping subproblems** — the same inputs are encountered multiple times. It is the top-down approach to dynamic programming.

## Reference

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
