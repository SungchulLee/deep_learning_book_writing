# Counting Sort

Comparison-based sorting algorithms such as merge sort and quicksort achieve at best
$O(n \log n)$ time because every comparison eliminates at most half of the remaining
orderings.  When the input consists of integers drawn from a known, bounded range
$\{0, 1, \dots, k\}$, we can bypass comparisons entirely.  **Counting sort** counts
how many times each value appears and uses those counts to place every element directly
into its correct output position, achieving linear time.

## Algorithm Overview

Counting sort operates in three phases:

1. **Count.**  Scan the input array $A[0 \dots n-1]$ and build a frequency array
   $C[0 \dots k]$ where $C[j]$ equals the number of elements equal to $j$.
2. **Cumulate.**  Transform $C$ into a prefix-sum array so that $C[j]$ now stores the
   number of elements that are *at most* $j$.  After this step, $C[j]$ tells us the
   last index (one-based) in the output that should hold a copy of value $j$.
3. **Place.**  Traverse the input array from right to left.  For each element
   $A[i]$, place it at output position $C[A[i]] - 1$ and decrement $C[A[i]]$.
   The right-to-left traversal guarantees **stability**: equal elements appear in
   the output in the same relative order as in the input.

## Complexity

Let $n$ denote the number of elements and $k$ the range of values.

$$
T(n, k) = \Theta(n + k), \qquad S(n, k) = \Theta(n + k)
$$

The time splits into $\Theta(n)$ for scanning the input and populating the output,
plus $\Theta(k)$ for initializing and cumulating the count array.  The space accounts
for the count array of size $k + 1$ and the output array of size $n$.

Counting sort is efficient when $k = O(n)$.  If $k$ is much larger than $n$, the
$\Theta(k)$ terms dominate and the algorithm becomes impractical.

## Worked Example

Consider the input array $A = [4, 2, 2, 8, 3, 3, 1]$ with $k = 8$.

| Phase | State |
|-------|-------|
| Input | $[4, 2, 2, 8, 3, 3, 1]$ |
| Count | $C = [0, 1, 2, 2, 1, 0, 0, 0, 1]$ |
| Cumulate | $C = [0, 1, 3, 5, 6, 6, 6, 6, 7]$ |
| Place (right-to-left) | Output: $[1, 2, 2, 3, 3, 4, 8]$ |

After cumulation, $C[3] = 5$ means there are five elements with value at most 3.
When we encounter $A[5] = 3$, we place it at index $C[3] - 1 = 4$, then decrement
$C[3]$ to 4 so the next occurrence of 3 lands at index 3.

## Stability

Counting sort is **stable**: elements with equal keys retain their original relative
order.  Stability is essential when counting sort serves as the inner sorting routine
for radix sort, where each pass sorts by one digit and must not disturb the order
established by previous passes.

The right-to-left traversal in the placement phase is what preserves stability.
Traversing left-to-right would place equal elements in reverse order.

## Implementation

```python
"""
Counting sort -- stable, linear-time integer sort.

Sorts an array of non-negative integers whose values lie in [0, k].
Time:  Theta(n + k)
Space: Theta(n + k)
"""

# === Counting sort (stable) ================================================

def counting_sort(arr: list[int], k: int) -> list[int]:
    """Return a new list containing the elements of *arr* in sorted order.

    Parameters
    ----------
    arr : list[int]
        Input array with values in the range [0, k].
    k : int
        Maximum value in *arr*.

    Returns
    -------
    list[int]
        Sorted array (stable).
    """
    n = len(arr)
    count = [0] * (k + 1)
    output = [0] * n

    # Phase 1 -- count occurrences
    for x in arr:
        count[x] += 1

    # Phase 2 -- cumulative counts (prefix sums)
    for j in range(1, k + 1):
        count[j] += count[j - 1]

    # Phase 3 -- place elements (right-to-left for stability)
    for i in range(n - 1, -1, -1):
        val = arr[i]
        count[val] -= 1
        output[count[val]] = val

    return output


# === Demo ===================================================================

if __name__ == "__main__":
    data = [4, 2, 2, 8, 3, 3, 1]
    k = 8
    sorted_data = counting_sort(data, k)
    print(f"Input:  {data}")
    print(f"Sorted: {sorted_data}")
```

**Output:**
```
Input:  [4, 2, 2, 8, 3, 3, 1]
Sorted: [1, 2, 2, 3, 3, 4, 8]
```

## When to Use Counting Sort

| Scenario | Recommendation |
|----------|---------------|
| $k = O(n)$ | Ideal -- true $\Theta(n)$ performance |
| $k \gg n$ | Avoid -- $\Theta(k)$ space and time waste |
| Stability required | Preferred -- inherently stable |
| Subroutine for radix sort | Standard choice |
| Floating-point or string keys | Not applicable -- integer keys only |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.), Chapter 8. MIT Press.
