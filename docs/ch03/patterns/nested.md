# Nested Recursion

Nested recursion is an unusual pattern where the result of one recursive call is used as an **argument** to another recursive call. Unlike linear or binary recursion, the recursion depth depends on the output of inner calls, making the behavior difficult to trace mentally and often producing surprising results.

## Structure

In nested recursion, a recursive call appears inside the argument list of another recursive call:

```
function f(n):
    if base_case:
        return simple_answer
    return f(f(n - 1))  # inner call feeds outer call
```

## Example: McCarthy's 91 Function

The most famous example of nested recursion is McCarthy's 91 function, which returns 91 for all inputs $n \leq 100$:

$$
M(n) = \begin{cases} n - 10 & \text{if } n > 100 \\ M(M(n + 11)) & \text{if } n \leq 100 \end{cases}
$$

```python
"""Nested recursion demonstrated with McCarthy's 91 function."""


# === McCarthy's 91 Function ===

def mccarthy91(n):
    """Compute McCarthy's 91 function using nested recursion."""
    if n > 100:
        return n - 10
    return mccarthy91(mccarthy91(n + 11))


# === Main ===

if __name__ == "__main__":
    for n in [85, 90, 95, 100, 101, 105, 111]:
        print(f"M({n}) = {mccarthy91(n)}")
```

**Output:**
```
M(85) = 91
M(90) = 91
M(95) = 91
M(100) = 91
M(101) = 91
M(105) = 95
M(111) = 101
```

## Why It Is Unusual

Nested recursion is rarely used in practice because:

1. **Hard to analyze**: the recursion depth depends on intermediate results, not just the input size
2. **Hard to trace**: mental execution requires tracking nested evaluations
3. **Often replaceable**: most practical problems have simpler formulations

However, it appears in certain mathematical functions and is important for understanding the full spectrum of recursive patterns.

## Reference

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
