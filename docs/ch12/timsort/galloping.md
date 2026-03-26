# Galloping Mode

During the merge phase of Timsort, two sorted runs are combined element by element. When one run consistently "wins" the comparison (i.e., its elements are smaller), the standard one-at-a-time merge wastes comparisons on elements that could be moved in bulk. **Galloping mode** (also called exponential search) detects these one-sided stretches and switches to an exponential search to find the insertion point, then copies the entire block in a single operation. This optimization reduces the number of comparisons from $O(n)$ to $O(n + m \log(n/m))$ when merging a short run of length $m$ into a long run of length $n$.

## When Galloping Activates

Timsort tracks a counter called `min_gallop` (initially 7). During a standard merge, whenever the same run wins $\text{min\_gallop}$ consecutive comparisons, the algorithm enters galloping mode. If galloping proves beneficial (finds large blocks to copy), `min_gallop` is decremented, making future galloping easier to trigger. If galloping finds only a few elements, `min_gallop` is incremented, raising the threshold and favoring one-at-a-time merging.

This adaptive threshold ensures that galloping is used only when the data exhibits the long one-sided stretches where it provides a benefit.

## Galloping Search Algorithm

Given a target value $v$ and a sorted array $B[0..n-1]$, galloping search finds the position where $v$ should be inserted:

1. **Exponential expansion**: Start with $k = 1$. Check positions $0, 1, 3, 7, 15, \ldots$ (i.e., $2^k - 1$) until finding a position where $B[2^k - 1] \geq v$ or reaching the end of the array.
2. **Binary search**: Perform a binary search in the range $[2^{k-1}, \min(2^k - 1, n-1)]$.

The exponential phase uses at most $\lceil \log_2(m+1) \rceil$ comparisons, where $m$ is the number of elements from $B$ that are less than $v$. The binary search adds another $O(\log m)$ comparisons. The total is $O(\log m)$, compared to $O(m)$ for a linear scan.

## Complexity of Galloping Merge

When merging a run of length $m$ into a run of length $n$ (with $m \leq n$), the number of comparisons using galloping is:

$$
O(m \log(n/m + 1))
$$

This is better than the standard $O(m + n)$ merge when $m \ll n$, and no worse than $O(m + n)$ when $m \approx n$.

## Implementation

```python
"""
Galloping (exponential) search for Timsort's merge optimization.

When one run consistently wins during merging, galloping search
finds the insertion point in O(log m) instead of O(m), then copies
the winning elements in bulk.
"""


# === Galloping Search ===

def gallop_right(key, arr: list, lo: int, hi: int) -> int:
    """Find the rightmost position where key could be inserted in arr[lo..hi-1].

    Uses exponential search: double the step size until overshooting,
    then binary search in the identified range.

    Returns index i such that arr[lo..i-1] <= key < arr[i..hi-1].
    """
    if lo >= hi:
        return lo

    # Exponential expansion phase
    offset = 1
    last_offset = 0

    if key >= arr[lo]:
        # Gallop right: key is at least arr[lo]
        max_offset = hi - lo
        while offset < max_offset and key >= arr[lo + offset]:
            last_offset = offset
            offset = (offset << 1) + 1
            if offset <= 0:  # overflow protection
                offset = max_offset
        offset = min(offset, max_offset)

        # Binary search in [lo + last_offset, lo + offset)
        left = lo + last_offset
        right = lo + offset
    else:
        return lo

    # Binary search phase
    while left < right:
        mid = left + (right - left) // 2
        if key < arr[mid]:
            right = mid
        else:
            left = mid + 1

    return left


def gallop_left(key, arr: list, lo: int, hi: int) -> int:
    """Find the leftmost position where key could be inserted in arr[lo..hi-1].

    Returns index i such that arr[lo..i-1] < key <= arr[i..hi-1].
    """
    if lo >= hi:
        return lo

    offset = 1
    last_offset = 0

    if key > arr[lo]:
        max_offset = hi - lo
        while offset < max_offset and key > arr[lo + offset]:
            last_offset = offset
            offset = (offset << 1) + 1
            if offset <= 0:
                offset = max_offset
        offset = min(offset, max_offset)

        left = lo + last_offset
        right = lo + offset
    else:
        return lo

    while left < right:
        mid = left + (right - left) // 2
        if key <= arr[mid]:
            right = mid
        else:
            left = mid + 1

    return left


# === Demonstration ===

if __name__ == "__main__":
    # Sorted run to search within
    run = [2, 5, 8, 12, 16, 23, 38, 42, 55, 67, 72, 84, 91, 99]
    print(f"Sorted run: {run}")
    print()

    # Gallop to find insertion point for various keys
    for key in [10, 42, 1, 100, 55]:
        pos = gallop_right(key, run, 0, len(run))
        print(f"gallop_right({key:3d}): insert at index {pos}")

    print()

    # Compare with linear search count
    for key in [10, 42, 84]:
        pos = gallop_right(key, run, 0, len(run))
        linear_steps = sum(1 for x in run if x <= key)
        print(f"key={key}: gallop found index {pos}, "
              f"linear scan would check {linear_steps} elements")
```

**Output:**
```
Sorted run: [2, 5, 8, 12, 16, 23, 38, 42, 55, 67, 72, 84, 91, 99]

gallop_right( 10): insert at index 3
gallop_right( 42): insert at index 8
gallop_right(  1): insert at index 0
gallop_right(100): insert at index 14
gallop_right( 55): insert at index 9

key=10: gallop found index 3, linear scan would check 3 elements
key=42: gallop found index 8, linear scan would check 8 elements
key=84: gallop found index 12, linear scan would check 12 elements
```

!!! warning "When Galloping Hurts"
    If the two runs are interleaved (alternating wins), galloping wastes comparisons on the exponential expansion phase without finding large blocks to copy. Timsort handles this by raising the `min_gallop` threshold when galloping is unproductive, making it harder to enter galloping mode on interleaved data.

## Reference

- Peters, T. (2002). *Timsort description*. [CPython source, `Objects/listsort.txt`](https://github.com/python/cpython/blob/main/Objects/listsort.txt).
- McIlroy, P. (1993). Optimistic sorting and information theoretic complexity. *Proceedings of SODA*, 467-474.
