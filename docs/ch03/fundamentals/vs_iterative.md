# Recursive vs Iterative

Any problem that can be solved recursively can also be solved iteratively, and vice versa. The choice between the two approaches involves trade-offs in clarity, performance, and space usage. Understanding when each approach is preferable is a fundamental skill in algorithm design.

## Side-by-Side Comparison

Consider computing the factorial $n! = n \times (n-1) \times \cdots \times 1$:

```python
"""Comparing recursive and iterative approaches to factorial."""


# === Recursive Factorial ===

def factorial_recursive(n):
    """Compute n! recursively."""
    if n <= 1:
        return 1
    return n * factorial_recursive(n - 1)


# === Iterative Factorial ===

def factorial_iterative(n):
    """Compute n! iteratively."""
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


# === Main ===

if __name__ == "__main__":
    for n in [0, 1, 5, 10]:
        r = factorial_recursive(n)
        i = factorial_iterative(n)
        print(f"{n}! = {r} (recursive) = {i} (iterative)")
```

**Output:**
```
0! = 1 (recursive) = 1 (iterative)
1! = 1 (recursive) = 1 (iterative)
5! = 120 (recursive) = 120 (iterative)
10! = 3628800 (recursive) = 3628800 (iterative)
```

## Trade-offs

| Aspect | Recursive | Iterative |
|---|---|---|
| Clarity | Natural for tree/divide-and-conquer problems | Natural for sequential processing |
| Space | $O(n)$ stack frames | $O(1)$ extra space (typically) |
| Overhead | Function call overhead per recursion | Loop overhead (minimal) |
| Stack overflow risk | Yes, for deep recursion | No |

## When to Prefer Each

**Recursion** is often clearer for problems with recursive structure: tree traversals, divide-and-conquer algorithms, and problems naturally defined by recurrences.

**Iteration** is often more efficient for simple sequential computations where recursion adds unnecessary overhead and stack usage.

## Reference

[Recursion의 개념과 기본 예제들](https://www.youtube.com/watch?v=tuzf1yLPgRI&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=2)
