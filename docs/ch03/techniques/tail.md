# Tail Recursion

A recursive function is **tail recursive** if the recursive call is the very last operation — there is no computation left to do after the call returns. This distinction matters because tail recursive functions can be optimized by the compiler to reuse the current stack frame, eliminating the $O(n)$ stack overhead of standard recursion.

## Tail vs Non-Tail Recursion

In standard (non-tail) recursion, the function performs work after the recursive call:

```python
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)  # multiplication AFTER the recursive call returns
```

In tail recursion, the recursive call is the final operation:

```python
"""Tail recursion demonstrated with factorial."""


# === Tail Recursive Factorial ===

def factorial_tail(n, acc=1):
    """Compute n! using tail recursion with an accumulator."""
    if n <= 1:
        return acc
    return factorial_tail(n - 1, acc * n)  # nothing happens after this call


# === Main ===

if __name__ == "__main__":
    for n in [0, 1, 5, 10]:
        print(f"{n}! = {factorial_tail(n)}")
```

**Output:**
```
0! = 1
1! = 1
5! = 120
10! = 3628800
```

## Why Tail Recursion Matters

In non-tail recursion, each frame must remain on the stack because it has pending work (the multiplication). In tail recursion, the current frame has no more work to do, so the compiler can replace it with the next frame rather than stacking a new one.

| Property | Standard recursion | Tail recursion |
|---|---|---|
| Pending operations | Yes | No |
| Stack growth | $O(n)$ | $O(1)$ (with optimization) |
| Requires accumulator | No | Usually yes |

## Limitation in Python

Python does **not** perform tail call optimization. Even tail recursive Python functions use $O(n)$ stack space. However, understanding tail recursion remains valuable for languages that do optimize it (Scheme, Haskell, Scala) and for converting recursion to iteration.

## Reference

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
