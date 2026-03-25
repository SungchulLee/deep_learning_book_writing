# Tail Call Optimization

Tail call optimization (TCO) is a compiler technique that reuses the current stack frame for a tail-positioned function call instead of pushing a new frame. When applied to tail recursive functions, TCO converts $O(n)$ stack usage into $O(1)$, effectively turning recursion into iteration at the machine level.

## How TCO Works

Without TCO, each call pushes a new frame:

```
factorial_tail(5, 1)  → frame 1
  factorial_tail(4, 5)  → frame 2
    factorial_tail(3, 20) → frame 3
      ...                   → frame n
```

With TCO, the compiler recognizes that frame 1 has no more work to do after calling frame 2. So it replaces frame 1 with frame 2:

```
factorial_tail(5, 1)    → frame (reused)
factorial_tail(4, 5)    → frame (reused)
factorial_tail(3, 20)   → frame (reused)
...                      → frame (reused)
```

## Example: Simulating TCO in Python

Python does not support TCO natively. However, we can simulate it using a trampoline — a loop that repeatedly calls the returned function until a final value is produced:

```python
"""Simulating tail call optimization with a trampoline."""


# === Trampoline Framework ===

class TailCall:
    """Wrapper indicating a tail call to be trampolined."""
    def __init__(self, func, *args):
        self.func = func
        self.args = args


def trampoline(result):
    """Execute tail calls iteratively until a final value is reached."""
    while isinstance(result, TailCall):
        result = result.func(*result.args)
    return result


# === Factorial Using Trampoline ===

def factorial_tco(n, acc=1):
    """Tail recursive factorial returning TailCall for trampoline."""
    if n <= 1:
        return acc
    return TailCall(factorial_tco, n - 1, acc * n)


# === Main ===

if __name__ == "__main__":
    for n in [5, 10, 100]:
        result = trampoline(factorial_tco(n))
        print(f"{n}! = {result}")
```

**Output:**
```
5! = 120
10! = 3628800
100! = 93326215443944152681699238856266700490715968264381621468592963895217599993229915608941463976156518286253697920827223758251185210916864000000000000000000000000
```

## Language Support

| Language | TCO Support |
|---|---|
| Scheme | Guaranteed by specification |
| Haskell | Yes (lazy evaluation) |
| Scala | `@tailrec` annotation |
| Python | No |
| Java | No |
| C/C++ | Compiler-dependent |

## Reference

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
