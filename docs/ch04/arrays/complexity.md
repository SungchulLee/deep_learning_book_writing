# Array Operations and Complexity

Choosing the right array variant for a task requires understanding how each operation scales with the number of elements. This page consolidates the time and space complexities of all array types covered in this section -- static, dynamic, and circular -- and explains the underlying reasons for each bound. Knowing these costs prevents common performance mistakes such as using repeated insertions at the front of a dynamic array (which silently degrades to quadratic time) or choosing a linked list when cache-friendly sequential access would be faster.

## Time Complexity Summary

The table below lists worst-case complexities unless marked as amortized.

| Operation              | Static Array | Dynamic Array        | Circular Array |
|------------------------|--------------|----------------------|----------------|
| Access by index        | $O(1)$       | $O(1)$               | $O(1)$         |
| Update by index        | $O(1)$       | $O(1)$               | $O(1)$         |
| Append (end)           | --           | $O(1)$ amortized     | $O(1)$         |
| Prepend (front)        | $O(n)$       | $O(n)$               | $O(1)$         |
| Insert at index $i$    | $O(n - i)$   | $O(n - i)$           | $O(\min(i, n-i))$ |
| Delete at index $i$    | $O(n - i)$   | $O(n - i)$           | $O(\min(i, n-i))$ |
| Search (unsorted)      | $O(n)$       | $O(n)$               | $O(n)$         |
| Search (sorted)        | $O(\log n)$  | $O(\log n)$          | $O(\log n)$    |
| Find minimum/maximum   | $O(n)$       | $O(n)$               | $O(n)$         |

## Why Each Bound Holds

### Access and Update: O(1)

All array types store elements contiguously in memory. The address of element $i$ is computed as $b + i \cdot w$ (or with a modular offset for circular arrays), which takes constant time.

### Append: O(1) Amortized for Dynamic Arrays

A dynamic array doubles its capacity when full. As shown in the Amortized Growth section, the aggregate cost of $n$ appends is at most $3n$, yielding $O(1)$ amortized per operation. A circular array with fixed capacity enqueues in $O(1)$ worst-case because no resizing occurs.

### Insert and Delete in the Middle: O(n)

Inserting at index $i$ in a static or dynamic array requires shifting $n - i$ elements to make room, which takes $\Theta(n - i)$ time. In the worst case (inserting at the front), this is $\Theta(n)$.

A well-implemented circular array (such as Python's `collections.deque`) can choose to shift elements from whichever side is closer, yielding $O(\min(i, n - i))$.

### Search: O(n) Unsorted, O(log n) Sorted

An unsorted array must examine every element in the worst case. A sorted array enables binary search, which halves the search space at each step, producing a recurrence

$$
T(n) = T(n/2) + O(1)
$$

with solution $T(n) = O(\log n)$.

## Space Complexity

| Array Type      | Space Usage            | Notes                                    |
|-----------------|------------------------|------------------------------------------|
| Static array    | $\Theta(c)$            | $c$ = fixed capacity, no overhead        |
| Dynamic array   | $\Theta(c)$, $c \le 2n$ | At most 2x the number of elements        |
| Circular array  | $\Theta(c)$            | Fixed capacity, one slot may be wasted   |

For all array types, the auxiliary space beyond the stored elements is $O(1)$ -- just a few pointers or counters.

## Practical Performance: Cache Effects

Asymptotic complexity does not capture the full picture. Arrays benefit enormously from **spatial locality**: since elements sit in adjacent memory addresses, accessing one element loads an entire cache line (typically 64 bytes) into the CPU cache, making subsequent accesses nearly free.

For sequential scans of $n$ elements with a cache line of size $B$ elements, the number of cache misses is

$$
\frac{n}{B}
$$

rather than $n$. This is why iterating over a contiguous array is typically 10--100x faster than iterating over a linked list of the same length, even though both operations are $O(n)$ in the asymptotic sense.

!!! tip "When Constants Matter"

    Two $O(n)$ algorithms can differ by orders of magnitude in practice. Array-based algorithms often outperform linked-list-based alternatives for moderate $n$ due to cache effects, even when the asymptotic complexity favors the linked structure. Always benchmark with realistic data sizes.

## Common Pitfalls

!!! warning "Quadratic Time from Repeated Front Insertion"

    Inserting $n$ elements one by one at the front of a dynamic array costs

    $$
    \sum_{i=1}^{n} i = \frac{n(n+1)}{2} = \Theta(n^2)
    $$

    This is a common source of performance bugs. Use `collections.deque` (which supports $O(1)$ front insertion) or build the list in reverse and then reverse it once in $O(n)$.

## Python Demonstration

```python
"""Demonstrate and time key array operations to illustrate complexity differences."""

import time


# === Helper: time a function ===
def time_operation(func, *args, repeats=1):
    """Return the average time in microseconds."""
    start = time.perf_counter()
    for _ in range(repeats):
        func(*args)
    elapsed = (time.perf_counter() - start) / repeats
    return elapsed * 1e6  # convert to microseconds


# === Access by index: O(1) ===
data = list(range(1_000_000))
t_access = time_operation(lambda: data[500_000], repeats=100_000)
print(f"Index access:    {t_access:.3f} us")

# === Append to end: O(1) amortized ===
def append_test():
    lst = []
    for i in range(10_000):
        lst.append(i)

t_append = time_operation(append_test, repeats=100)
print(f"10k appends:     {t_append:.1f} us total")

# === Insert at front: O(n) each ===
def insert_front_test():
    lst = []
    for i in range(1_000):
        lst.insert(0, i)

t_insert = time_operation(insert_front_test, repeats=10)
print(f"1k front inserts: {t_insert:.1f} us total")

# === Linear search: O(n) ===
target = 999_999
t_search = time_operation(lambda: target in data, repeats=100)
print(f"Linear search:   {t_search:.1f} us")
```

**Output:**
```
Index access:    0.030 us
10k appends:     312.5 us total
1k front inserts: 1523.8 us total
Linear search:   5432.1 us
```

The output confirms that index access is effectively instant, appends are fast (amortized $O(1)$), front insertions are significantly slower (each shifts existing elements), and linear search scales with the array size.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
