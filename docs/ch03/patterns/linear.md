# Linear Recursion

Linear recursion is the simplest recursion pattern: the function makes exactly **one** recursive call per invocation, reducing the problem size by a constant amount each time. This creates a single chain of calls rather than a branching tree, making it straightforward to analyze and understand.

## Structure

A linearly recursive function follows this template:

```
function f(problem):
    if base_case:
        return simple_answer
    do_some_work()
    return f(smaller_problem)
```

Because there is only one recursive call, the call graph is a straight line — hence the name "linear recursion."

## Example: Sum of a List

Computing the sum of a list demonstrates linear recursion. Each call processes one element and recurses on the rest:

```python
"""Linear recursion demonstrated with list summation."""


# === Recursive Sum ===

def recursive_sum(arr):
    """Return the sum of elements in arr using linear recursion."""
    if len(arr) == 0:
        return 0
    return arr[0] + recursive_sum(arr[1:])


# === Main ===

if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9]
    print(f"List: {data}")
    print(f"Sum:  {recursive_sum(data)}")
```

**Output:**
```
List: [3, 1, 4, 1, 5, 9]
Sum:  23
```

## Complexity

With one recursive call per invocation and $O(1)$ work per call, the recurrence is:

$$
T(n) = T(n - 1) + O(1), \quad T(0) = O(1)
$$

This solves to $T(n) = O(n)$ time. Stack depth is also $O(n)$, which is the primary disadvantage compared to an equivalent iterative loop that uses $O(1)$ space.

## Common Examples

Linear recursion appears in: factorial, list reversal, string processing, linked list traversal, and computing powers ($x^n$).

## Reference

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
