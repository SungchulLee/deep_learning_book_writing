# Built-in sorted

Python provides two primary ways to sort data: the `list.sort()` method and the built-in `sorted()` function. While `list.sort()` modifies a list in place and returns `None` (see the [List Method sort](sort_method.md) page), `sorted()` takes any iterable and returns a **new sorted list**, leaving the original unchanged. This non-destructive behavior makes `sorted()` the preferred choice when the original data must be preserved or when sorting non-list iterables like tuples, sets, and dictionaries.

## Function Signature

The full signature of `sorted()` is:

```python
sorted(iterable, /, *, key=None, reverse=False)
```

- **`iterable`** — any iterable (list, tuple, set, dict, generator, string, etc.).
- **`key`** — a function applied to each element before comparison. Elements are compared by their key values rather than directly. See [Built-in sorted with key](sorted_with_key.md) for detailed usage.
- **`reverse`** — if `True`, sorts in descending order.

The return value is always a **new list**, even if the input is a different type.

## Sorting a List

The most common use case is sorting a list of numbers or strings.

```python
numbers = [9, 1, 8, 2, 7, 3, 6, 4, 5]
sorted(numbers)
```

**Output:**
```
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

The original list is unchanged:

```python
numbers  # still [9, 1, 8, 2, 7, 3, 6, 4, 5]
```

To sort in descending order, use `reverse=True`:

```python
sorted(numbers, reverse=True)
```

**Output:**
```
[9, 8, 7, 6, 5, 4, 3, 2, 1]
```

## Sorting a Tuple

Since tuples are immutable, `sorted()` is the only built-in way to obtain a sorted version. The result is a list, not a tuple.

```python
t = (9, 1, 8, 2, 7, 3, 6, 4, 5)
sorted(t)
```

**Output:**
```
[1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## Sorting a Set

Sets are unordered collections, so their iteration order is not predictable. Applying `sorted()` to a set produces a deterministic sorted list.

```python
s = {'name', 'job', 'age', 'os'}
sorted(s)
```

**Output:**
```
['age', 'job', 'name', 'os']
```

## Sorting a Dictionary

When `sorted()` receives a dictionary, it iterates over the **keys** by default.

```python
d = {'name': 'Jake', 'job': 'Programming', 'age': '29', 'os': 'Mac'}
sorted(d)
```

**Output:**
```
['age', 'job', 'name', 'os']
```

To sort by values instead, use the `key` parameter:

```python
sorted(d.items(), key=lambda item: item[1])
```

**Output:**
```
[('age', '29'), ('name', 'Jake'), ('os', 'Mac'), ('job', 'Programming')]
```

## Sorting a List of Tuples

When elements are tuples, Python compares them **lexicographically**: first by the first component, then by the second on ties, and so on.

```python
pairs = [(3, 34), (2, 35), (4, 30), (3, 33)]
sorted(pairs)
```

**Output:**
```
[(2, 35), (3, 33), (3, 34), (4, 30)]
```

The two tuples starting with `3` are ordered by their second component: $(3, 33)$ before $(3, 34)$.

## sorted vs list.sort

The table below summarizes the differences between the two approaches.

| Feature | `sorted()` | `list.sort()` |
|---------|-----------|--------------|
| Returns | New list | `None` (in-place) |
| Input types | Any iterable | Lists only |
| Original data | Preserved | Modified |
| Memory | Allocates new list | No extra allocation |
| Algorithm | Timsort | Timsort |
| Stable | Yes | Yes |

!!! tip "When to Use Which"
    Use `sorted()` when you need the original data intact or when sorting a non-list iterable. Use `list.sort()` when you want to sort a list in place and do not need the original order, saving memory by avoiding a copy.

## Implementation: Timsort

Both `sorted()` and `list.sort()` use **Timsort**, a hybrid sorting algorithm derived from merge sort and insertion sort. Timsort is:

- **Stable**: equal elements preserve their relative order.
- **Adaptive**: it exploits existing runs of sorted data, achieving $O(n)$ time on already-sorted input.
- **Worst-case** $O(n \log n)$: it never degrades beyond optimal comparison-based performance.
- **Space**: $O(n)$ auxiliary space for the merge buffer.

These properties make Timsort an excellent general-purpose sorting algorithm, which is why it was adopted as the default sort in Python (since 2002), Java (for objects, since Java SE 7), and several other languages and libraries.

## Reference

- Python Documentation. [Built-in Functions: sorted](https://docs.python.org/3/library/functions.html#sorted).
- Python Documentation. [Sorting HOW TO](https://docs.python.org/3/howto/sorting.html).
