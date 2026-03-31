# Python's List sort Method

Python provides a built-in `sort()` method on lists that rearranges elements in place. Understanding its behavior -- including the in-place semantics, the default ordering, and the `reverse` parameter -- is essential before studying sorting algorithms in depth, because `sort()` serves as both a practical tool and a benchmark for comparison.

## Basic Usage

The `sort()` method arranges list elements in ascending order by default:

```python
"""Demonstration of Python's list.sort() method.

Shows ascending sort, descending sort, and the in-place return value.
"""


# === Ascending sort ===
a = [-9, 1, 8, 2, -7, 3, 6, 4, 5]
a.sort()
print("Ascending:", a)

# === Descending sort ===
b = [-9, 1, 8, 2, -7, 3, 6, 4, 5]
b.sort(reverse=True)
print("Descending:", b)
```

**Output:**
```
Ascending: [-9, -7, 1, 2, 3, 4, 5, 6, 8]
Descending: [8, 6, 5, 4, 3, 2, 1, -7, -9]
```

## In-Place Semantics

The `sort()` method modifies the list **in place** and returns `None` -- not the sorted list. This is a deliberate design choice in Python: methods that mutate their object return `None` to signal that the original was changed.

```python
a = [9, 1, 8, 2, 7, 3, 6, 4, 5]
result = a.sort()
print(result)  # None
print(a)       # [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

**Output:**
```
None
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

!!! warning "Common Mistake"
    Writing `a = a.sort()` replaces the sorted list with `None`. If you need a new sorted list while keeping the original unchanged, use the built-in function `sorted(a)` instead, which returns a new list.

## sort() vs sorted()

| Feature | `list.sort()` | `sorted(iterable)` |
|:---|:---|:---|
| Returns | `None` | New sorted list |
| Modifies original | Yes (in place) | No |
| Works on | Lists only | Any iterable |

## Implementation Details

Python's `sort()` uses **Timsort**, a hybrid algorithm combining merge sort and insertion sort. Timsort runs in $O(n \log n)$ worst-case time and $O(n)$ best-case time (when the input is already partially sorted). It is **stable**, meaning equal elements retain their original relative order.

## References

[Corey Schafer -- Sorting Lists, Tuples, and Objects](https://www.youtube.com/watch?v=D3JvDWO-BY4)
