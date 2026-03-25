# Multiple Recursion

Multiple recursion occurs when a function makes **three or more** recursive calls per invocation. While binary recursion splits a problem in two, multiple recursion handles cases where the problem naturally decomposes into many subproblems — as seen in backtracking over multiple choices or traversing trees with arbitrary branching factor.

## Structure

A multiply recursive function makes $k \geq 3$ recursive calls:

```
function f(problem):
    if base_case:
        return simple_answer
    for each sub in subproblems(problem):
        f(sub)
```

The call graph forms a tree with branching factor $k$, leading to exponential growth in the number of calls.

## Example: Generating All Subsets

Generating all subsets of a set is a natural multiple recursion problem. At each position, the function branches into "include" and "exclude" paths, but when generalized to $k$-ary choices, the branching factor exceeds two:

```python
"""Multiple recursion demonstrated with subset generation."""


# === Generate Subsets ===

def subsets(arr, index=0, current=None):
    """Generate all subsets of arr using recursion."""
    if current is None:
        current = []
    if index == len(arr):
        print(current)
        return
    # Exclude current element
    subsets(arr, index + 1, current)
    # Include current element
    subsets(arr, index + 1, current + [arr[index]])


# === Main ===

if __name__ == "__main__":
    subsets([1, 2, 3])
```

**Output:**
```
[]
[3]
[2]
[2, 3]
[1]
[1, 3]
[1, 2]
[1, 2, 3]
```

## Complexity

For a problem with branching factor $k$ and depth $d$, multiple recursion generates $O(k^d)$ calls. Subset generation with $n$ elements has $k = 2$ and $d = n$, giving $O(2^n)$ subsets — which is expected, since a set of size $n$ has exactly $2^n$ subsets.

## Reference

[Introduction to Algorithms (CLRS)](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
