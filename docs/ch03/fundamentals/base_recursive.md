# Base Case and Recursive Case

Every recursive function has two essential components: a **base case** that stops the recursion, and a **recursive case** that breaks the problem into a smaller instance. Without a base case, the function recurses indefinitely until the system runs out of stack space. Without a recursive case, there is no self-reference and the function is simply iterative.

## The Two Components

**Base case**: A condition under which the function returns directly without making a recursive call. This is the "ground floor" of the recursion — the simplest version of the problem whose answer is known.

**Recursive case**: The function calls itself with a modified argument that moves closer to the base case. Each recursive call must make progress toward the base case; otherwise, the recursion never terminates.

## Example

The following function prints integers from $n$ down to $1$. The base case is $n = 0$ (nothing to print), and the recursive case prints $n$ then recurses with $n - 1$:

```python
"""Base case and recursive case demonstration."""


# === Countdown Using Recursion ===

def countdown(n):
    """Print integers from n down to 1."""
    if n == 0:        # Base case
        return
    print(n)          # Recursive case: process current
    countdown(n - 1)  # Recurse on smaller problem


# === Main ===

if __name__ == "__main__":
    countdown(5)
```

**Output:**
```
5
4
3
2
1
```

## Why Both Are Necessary

| Missing component | Result |
|---|---|
| No base case | Infinite recursion → stack overflow |
| No recursive case | Not recursive — just a regular function |
| Base case never reached | Infinite recursion (e.g., `f(n)` calls `f(n+1)`) |

A well-designed recursive function guarantees that every chain of recursive calls eventually reaches the base case.

## Reference

[Recursion의 개념과 기본 예제들](https://www.youtube.com/watch?v=ln7AfppN7mY&list=PL52K_8WQO5oUuH06MLOrah4h05TZ4n38l&index=1)
