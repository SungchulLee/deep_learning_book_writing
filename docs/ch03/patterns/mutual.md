# Mutual Recursion

Mutual recursion occurs when two or more functions call each other in a cycle: function A calls function B, which in turn calls function A. This pattern appears when a problem naturally decomposes into alternating phases — such as parsing nested expressions, or determining parity.

## Structure

In mutual recursion, no single function is self-recursive. Instead, the recursion emerges from the interaction between functions:

```
function A(n):
    if base_case: return ...
    return ... B(n - 1) ...

function B(n):
    if base_case: return ...
    return ... A(n - 1) ...
```

## Example: Even and Odd

The simplest mutual recursion determines whether a number is even or odd. A number $n$ is even if $n - 1$ is odd, and $n$ is odd if $n - 1$ is even:

```python
"""Mutual recursion demonstrated with even/odd determination."""


# === Mutually Recursive Even/Odd ===

def is_even(n):
    """Return True if n is even, using mutual recursion with is_odd."""
    if n == 0:
        return True
    return is_odd(n - 1)


def is_odd(n):
    """Return True if n is odd, using mutual recursion with is_even."""
    if n == 0:
        return False
    return is_even(n - 1)


# === Main ===

if __name__ == "__main__":
    for i in range(6):
        print(f"{i}: even={is_even(i)}, odd={is_odd(i)}")
```

**Output:**
```
0: even=True, odd=False
1: even=False, odd=True
2: even=True, odd=False
3: even=False, odd=True
4: even=True, odd=False
5: even=False, odd=True
```

## Complexity

Each call to `is_even` or `is_odd` reduces $n$ by 1, making a total of $n$ calls. Time and space are both $O(n)$. While this example is better solved with `n % 2`, mutual recursion becomes genuinely useful in recursive descent parsers and state-machine simulations.

## Reference

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
