# Linear Search

Linear search is the simplest searching algorithm: examine each element one at a time until the target is found or the list is exhausted. While iterative linear search is straightforward, implementing it recursively illustrates how any loop can be expressed as recursion — a key insight for understanding more complex recursive algorithms.

## Recursive Approach

The recursive formulation checks the current element and, if it does not match, recurses on the remainder of the list:

- **Base case 1**: the list is empty — return "not found"
- **Base case 2**: the first element matches the target — return its index
- **Recursive case**: search the rest of the list

```python
"""Recursive linear search with index tracking."""


# === Recursive Linear Search ===

def linear_search(arr, target, index=0):
    """Return the index of target in arr, or -1 if not found."""
    if index == len(arr):
        return -1
    if arr[index] == target:
        return index
    return linear_search(arr, target, index + 1)


# === Main ===

if __name__ == "__main__":
    data = [4, 2, 7, 1, 9, 3]
    print(f"Array: {data}")
    print(f"Search for 7: index {linear_search(data, 7)}")
    print(f"Search for 5: index {linear_search(data, 5)}")
```

**Output:**
```
Array: [4, 2, 7, 1, 9, 3]
Search for 7: index 2
Search for 5: index -1
```

## Complexity Analysis

Each recursive call processes one element and makes at most one recursive call, giving:

$$
T(n) = T(n - 1) + O(1), \quad T(0) = O(1)
$$

This solves to $T(n) = O(n)$ — the same as the iterative version. The space complexity is $O(n)$ due to the recursion stack, compared to $O(1)$ for the iterative approach.

## Reference

[Best, Worst and Average Case Analysis](https://www.youtube.com/watch?v=lj3E24nnPjI&list=PLDN4rrl48XKpZkf03iYFl-O29szjTrs_O&index=16)
