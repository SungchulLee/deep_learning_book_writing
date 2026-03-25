# Converting to Iteration

Every recursive algorithm can be converted to an iterative one, and vice versa. The conversion is important in practice because iterative solutions avoid stack overflow risks and often run faster due to eliminated function call overhead. Three main techniques handle the conversion.

## Technique 1: Direct Loop Replacement

Tail recursive functions convert directly to loops. The accumulator becomes a loop variable:

```python
"""Converting tail recursion to iteration."""


# === Recursive Factorial (tail form) ===

def factorial_recursive(n, acc=1):
    """Tail recursive factorial."""
    if n <= 1:
        return acc
    return factorial_recursive(n - 1, acc * n)


# === Iterative Factorial ===

def factorial_iterative(n):
    """Same logic as above, expressed as a loop."""
    acc = 1
    while n > 1:
        acc *= n
        n -= 1
    return acc


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

## Technique 2: Explicit Stack

Non-tail recursive functions that require unwinding (like tree traversals) can use an explicit stack data structure to simulate the call stack:

```python
# Recursive DFS
def dfs_recursive(node):
    visit(node)
    for child in node.children:
        dfs_recursive(child)

# Iterative DFS with explicit stack
def dfs_iterative(root):
    stack = [root]
    while stack:
        node = stack.pop()
        visit(node)
        for child in reversed(node.children):
            stack.append(child)
```

## Technique 3: Trampoline

For mutually recursive or complex recursive functions, a trampoline loop can replace the recursion without restructuring the logic (see the Tail Call Optimization page).

## Reference

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
