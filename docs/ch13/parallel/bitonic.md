# Bitonic Sort

Most sorting algorithms are inherently sequential: each comparison depends on the
results of previous ones.  **Bitonic sort** is a parallel sorting algorithm whose
comparison pattern is fixed and data-independent, making it ideal for hardware
implementations (FPGA, GPU) and sorting networks.  It builds on the concept of
*bitonic sequences* -- sequences that first increase then decrease (or vice versa) --
and recursively merges them into sorted order.

## Bitonic Sequences

A sequence $a_0, a_1, \dots, a_{n-1}$ is **bitonic** if there exists an index $k$
such that:

$$
a_0 \le a_1 \le \cdots \le a_k \ge a_{k+1} \ge \cdots \ge a_{n-1}
$$

or the sequence is a cyclic rotation of such a sequence.  Examples:

- $[1, 3, 5, 7, 6, 4, 2]$ -- increases then decreases.
- $[7, 6, 4, 2, 1, 3, 5]$ -- cyclic rotation of a bitonic sequence.
- Any sorted sequence is trivially bitonic ($k = n-1$).
- Any reverse-sorted sequence is trivially bitonic ($k = 0$).

## Bitonic Merge

The key operation is the **bitonic merge**, which takes a bitonic sequence and produces
a sorted sequence.  Given a bitonic sequence of length $n$ (a power of 2):

1. Compare elements at distance $n/2$: pair $(a_i, a_{i + n/2})$ for $i = 0, \dots, n/2 - 1$.
2. For each pair, swap if the element at position $i$ is greater (for ascending order).
3. After this step, the two halves are each bitonic and every element in the first half
   is at most every element in the second half.
4. Recursively apply bitonic merge to each half.

The recursion has depth $\log_2 n$, and each level performs $n/2$ independent
comparisons.

## Bitonic Sort Algorithm

Bitonic sort builds up sorted sequences by alternately producing bitonic sequences and
merging them:

1. Start with $n$ individual elements (each trivially sorted).
2. For each block size $s = 2, 4, 8, \dots, n$:
    - Pair adjacent blocks of size $s/2$.
    - Sort one block ascending and the other descending, forming a bitonic sequence
      of size $s$.
    - Apply bitonic merge to produce a sorted sequence of size $s$.

## Complexity

$$
T_{\text{comparisons}} = O(n \log^2 n)
$$

$$
T_{\text{parallel}} = O(\log^2 n) \quad \text{with } n/2 \text{ processors}
$$

The algorithm performs $\log_2 n$ stages, and stage $k$ has $k$ comparison rounds,
giving a total of $\frac{1}{2}\log_2 n \cdot (\log_2 n + 1)$ parallel steps.

**Space:** $O(n)$ -- the sort is in-place (only swaps are needed).

## Worked Example

Sort $A = [3, 7, 4, 8, 6, 2, 1, 5]$ using bitonic sort (ascending).

**Stage 1 (blocks of 2):** Sort pairs alternately ascending/descending.

- $[3, 7] \to [3, 7]$ (ascending)
- $[4, 8] \to [8, 4]$ (descending)
- $[6, 2] \to [2, 6]$ (ascending)
- $[1, 5] \to [5, 1]$ (descending)

Result: $[3, 7, 8, 4, 2, 6, 5, 1]$

**Stage 2 (blocks of 4):** Bitonic merge groups of 4.

- $[3, 7, 8, 4]$ is bitonic -- merge ascending: $[3, 4, 7, 8]$
- $[2, 6, 5, 1]$ is bitonic -- merge descending: $[6, 5, 2, 1]$

Result: $[3, 4, 7, 8, 6, 5, 2, 1]$

**Stage 3 (block of 8):** The entire array is bitonic -- merge ascending.

- Compare at distance 4: $(3,6), (4,5), (7,2), (8,1) \to [3, 4, 2, 1, 6, 5, 7, 8]$
- Compare at distance 2: $(3,2), (4,1), (6,7), (5,8) \to [2, 1, 3, 4, 6, 5, 7, 8]$
- Compare at distance 1: $(2,1), (3,4), (6,5), (7,8) \to [1, 2, 3, 4, 5, 6, 7, 8]$

**Final result:** $[1, 2, 3, 4, 5, 6, 7, 8]$

## Implementation

```python
"""
Bitonic sort -- data-oblivious parallel sorting algorithm.

The comparison pattern is independent of input values, making it
suitable for hardware sorting networks and GPU implementations.
Time:  O(n log^2 n) comparisons (sequential)
       O(log^2 n) parallel steps with n/2 processors
Space: O(n) -- in-place via swaps
"""

# === Bitonic sort ===========================================================

def _compare_and_swap(arr: list[int], i: int, j: int, ascending: bool) -> None:
    """Swap arr[i] and arr[j] if they are in the wrong order."""
    if (arr[i] > arr[j]) == ascending:
        arr[i], arr[j] = arr[j], arr[i]


def _bitonic_merge(arr: list[int], lo: int, length: int, ascending: bool) -> None:
    """Merge a bitonic sequence arr[lo:lo+length] into sorted order."""
    if length <= 1:
        return

    half = length // 2
    for i in range(lo, lo + half):
        _compare_and_swap(arr, i, i + half, ascending)

    _bitonic_merge(arr, lo, half, ascending)
    _bitonic_merge(arr, lo + half, half, ascending)


def _bitonic_sort_rec(arr: list[int], lo: int, length: int, ascending: bool) -> None:
    """Recursively sort arr[lo:lo+length] using bitonic sort."""
    if length <= 1:
        return

    half = length // 2
    # Sort first half ascending, second half descending
    _bitonic_sort_rec(arr, lo, half, True)
    _bitonic_sort_rec(arr, lo + half, half, False)

    # Merge the resulting bitonic sequence
    _bitonic_merge(arr, lo, length, ascending)


def bitonic_sort(arr: list[int], ascending: bool = True) -> list[int]:
    """Sort *arr* using bitonic sort.

    Parameters
    ----------
    arr : list[int]
        Input array. Length must be a power of 2.
    ascending : bool
        Sort in ascending order if True.

    Returns
    -------
    list[int]
        Sorted array.
    """
    result = list(arr)
    n = len(result)
    assert n > 0 and (n & (n - 1)) == 0, "Length must be a power of 2"
    _bitonic_sort_rec(result, 0, n, ascending)
    return result


# === Demo ===================================================================

if __name__ == "__main__":
    data = [3, 7, 4, 8, 6, 2, 1, 5]
    sorted_data = bitonic_sort(data)
    print(f"Input:     {data}")
    print(f"Ascending: {sorted_data}")
    print(f"Descending: {bitonic_sort(data, ascending=False)}")

    # Larger example
    import random
    random.seed(42)
    large = [random.randint(0, 999) for _ in range(16)]
    print(f"\nInput (16): {large}")
    print(f"Sorted:     {bitonic_sort(large)}")
```

**Output:**
```
Input:     [3, 7, 4, 8, 6, 2, 1, 5]
Ascending: [1, 2, 3, 4, 5, 6, 7, 8]
Descending: [8, 7, 6, 5, 4, 3, 2, 1]

Input (16): [654, 114, 25, 837, 886, 544, 165, 572, 892, 400, 991, 985, 7, 426, 849, 156]
Sorted:     [7, 25, 114, 156, 165, 400, 426, 544, 572, 654, 837, 849, 886, 892, 985, 991]
```

## GPU and Hardware Applications

Bitonic sort is particularly attractive for GPUs because:

1. **Data-oblivious.** The comparison pattern does not depend on the data values,
   so all threads execute the same instructions (no branch divergence).
2. **Regular memory access.** Each stage accesses elements at fixed strides, which
   maps well to GPU memory coalescing.
3. **No synchronization within a warp.** Comparisons within a 32-thread warp can
   execute without explicit barriers.

For these reasons, bitonic sort is the sorting algorithm of choice for small to
moderate arrays on GPUs, despite its $O(n \log^2 n)$ comparison count being worse
than optimal $O(n \log n)$ merge sort.

## Reference

- Batcher, K. E. (1968). Sorting networks and their applications. *Proceedings
  of the AFIPS Spring Joint Computer Conference*, 307-314.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022).
  *Introduction to Algorithms* (4th ed.), Chapter 27. MIT Press.
