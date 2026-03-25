# Stack Frames

Each active function call occupies a **stack frame** (also called an activation record) on the call stack. In recursive functions, multiple frames for the same function coexist simultaneously — one for each level of recursion. Understanding what each frame contains explains how recursion "remembers" the state of each call.

## Anatomy of a Stack Frame

A stack frame typically stores:

| Component | Purpose |
|---|---|
| Parameters | The arguments passed to this call |
| Local variables | Variables declared inside the function |
| Return address | Where to resume execution after this call returns |
| Return value | The value to pass back to the caller |

## Example: Factorial Stack Frames

Consider computing `factorial(4)`. Each recursive call creates a new frame, and each frame holds its own copy of `n`:

```python
"""Visualizing stack frames during recursive factorial computation."""


# === Factorial with Frame Tracing ===

def factorial(n, depth=0):
    """Compute n! while printing stack frame information."""
    indent = "  " * depth
    print(f"{indent}PUSH frame: factorial({n})")
    if n <= 1:
        print(f"{indent}  return 1")
        print(f"{indent}POP  frame: factorial({n})")
        return 1
    result = n * factorial(n - 1, depth + 1)
    print(f"{indent}  return {n} * ... = {result}")
    print(f"{indent}POP  frame: factorial({n})")
    return result


# === Main ===

if __name__ == "__main__":
    answer = factorial(4)
    print(f"\nfactorial(4) = {answer}")
```

**Output:**
```
PUSH frame: factorial(4)
  PUSH frame: factorial(3)
    PUSH frame: factorial(2)
      PUSH frame: factorial(1)
        return 1
      POP  frame: factorial(1)
      return 2 * ... = 2
    POP  frame: factorial(2)
    return 3 * ... = 6
  POP  frame: factorial(3)
  return 4 * ... = 24
POP  frame: factorial(4)

factorial(4) = 24
```

## Memory Implications

For a recursion of depth $d$, the call stack holds $d$ frames simultaneously. Each frame uses $O(1)$ space for its local variables, giving total stack space of $O(d)$. For `factorial(n)`, the depth is $n$, so the stack space is $O(n)$.

This is why deep recursion can cause stack overflow — each frame consumes memory, and system stacks have a fixed size limit.

## Reference

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
