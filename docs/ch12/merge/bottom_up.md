# Bottom-Up Merge Sort

Top-down merge sort relies on recursion to break the array into singleton subarrays before merging upward.  Bottom-up merge sort reverses this perspective: it starts with individual elements (trivially sorted) and iteratively merges adjacent subarrays of increasing size -- 1, 2, 4, 8, and so on -- until the entire array is sorted.  This eliminates recursion entirely, which is advantageous in environments with limited stack space and simplifies the control flow.

## Algorithm Overview

Bottom-up merge sort proceeds in passes, where each pass doubles the size of the sorted subarrays:

1. **Pass 1** ($\text{width} = 1$): merge adjacent pairs of single elements into sorted pairs.
2. **Pass 2** ($\text{width} = 2$): merge adjacent sorted pairs into sorted quadruples.
3. **Pass 3** ($\text{width} = 4$): merge adjacent sorted quadruples into sorted octets.
4. Continue until $\text{width} \geq n$.

At each pass, every element participates in exactly one merge, so each pass costs $O(n)$.  There are $\lceil \log_2 n \rceil$ passes, giving $O(n \log n)$ total time.

## Pseudocode

```
BOTTOM-UP-MERGE-SORT(A, n):
    width = 1
    while width < n:
        for i = 0, 2*width, 4*width, ..., < n:
            left = i
            mid = min(i + width - 1, n - 1)
            right = min(i + 2*width - 1, n - 1)
            MERGE(A, left, mid, right)
        width = 2 * width
```

The `min` operations handle the boundary case where the array length is not a power of two, ensuring the last subarray is merged correctly even if it is shorter than `width`.

## Step-by-Step Example

Sort $[38, 27, 43, 3, 9, 82, 10]$ (length 7):

**Pass 1** (width = 1): merge pairs of single elements.

$$
[38, 27] \to [27, 38], \quad [43, 3] \to [3, 43], \quad [9, 82] \to [9, 82], \quad [10] \to [10]
$$

Array: $[27, 38, 3, 43, 9, 82, 10]$.

**Pass 2** (width = 2): merge sorted pairs into quadruples.

$$
[27, 38] + [3, 43] \to [3, 27, 38, 43], \quad [9, 82] + [10] \to [9, 10, 82]
$$

Array: $[3, 27, 38, 43, 9, 10, 82]$.

**Pass 3** (width = 4): merge into the final sorted array.

$$
[3, 27, 38, 43] + [9, 10, 82] \to [3, 9, 10, 27, 38, 43, 82]
$$

## Complexity Analysis

**Time complexity.** Each pass merges all $n$ elements: $O(n)$ per pass.  The number of passes is $\lceil \log_2 n \rceil$:

$$
T(n) = O(n) \cdot \lceil \log_2 n \rceil = O(n \log n)
$$

This holds for best, average, and worst cases -- the algorithm performs the same work regardless of input order.

**Space complexity.** The merge procedure requires $O(n)$ auxiliary space for temporary arrays.  Unlike top-down merge sort, there is no recursion stack:

$$
S(n) = O(n)
$$

The absence of the $O(\log n)$ stack overhead is the main advantage over the recursive version, though both are $O(n)$ overall.

## Comparison with Top-Down Merge Sort

| Property          | Top-down          | Bottom-up          |
|-------------------|-------------------|--------------------|
| Time              | $O(n \log n)$     | $O(n \log n)$      |
| Space             | $O(n) + O(\log n)$ stack | $O(n)$     |
| Recursion         | Yes               | No                 |
| Implementation    | Simpler logic     | Slightly more index arithmetic |
| Cache behavior    | Similar           | Similar            |
| Linked lists      | Natural           | Also natural       |

!!! tip "Bottom-up for linked lists"
    Bottom-up merge sort is particularly well-suited for **linked lists**, where the merge operation can be done in $O(1)$ extra space by relinking nodes.  This makes linked-list bottom-up merge sort both $O(n \log n)$ time and $O(1)$ auxiliary space -- a combination not achievable by array-based merge sort.

## Python Implementation

```python
"""
Bottom-up merge sort.

Sorts an array iteratively by merging subarrays of doubling width.
Avoids recursion entirely, making it suitable for environments with
limited stack space.
"""


# === Merge procedure ==========================================================

def merge(arr: list, left: int, mid: int, right: int) -> None:
    """Merge sorted subarrays arr[left..mid] and arr[mid+1..right]."""
    left_half = arr[left:mid + 1]
    right_half = arr[mid + 1:right + 1]
    i = j = 0
    k = left

    while i < len(left_half) and j < len(right_half):
        if left_half[i] <= right_half[j]:
            arr[k] = left_half[i]
            i += 1
        else:
            arr[k] = right_half[j]
            j += 1
        k += 1

    while i < len(left_half):
        arr[k] = left_half[i]
        i += 1
        k += 1
    while j < len(right_half):
        arr[k] = right_half[j]
        j += 1
        k += 1


# === Bottom-up merge sort =====================================================

def bottom_up_merge_sort(arr: list) -> None:
    """Sort arr in place using iterative bottom-up merge sort.

    Parameters
    ----------
    arr : list
        The array to sort (modified in place).
    """
    n = len(arr)
    width = 1

    while width < n:
        i = 0
        while i < n:
            left = i
            mid = min(i + width - 1, n - 1)
            right = min(i + 2 * width - 1, n - 1)
            if mid < right:
                merge(arr, left, mid, right)
            i += 2 * width
        width *= 2


# === Main =====================================================================

if __name__ == "__main__":
    data = [38, 27, 43, 3, 9, 82, 10]
    print(f"Before: {data}")
    bottom_up_merge_sort(data)
    print(f"After:  {data}")

    # Edge cases
    empty = []
    bottom_up_merge_sort(empty)
    print(f"Empty:  {empty}")

    single = [42]
    bottom_up_merge_sort(single)
    print(f"Single: {single}")

    already = [1, 2, 3, 4, 5]
    bottom_up_merge_sort(already)
    print(f"Sorted: {already}")
```

**Output:**
```
Before: [38, 27, 43, 3, 9, 82, 10]
After:  [3, 9, 10, 27, 38, 43, 82]
Empty:  []
Single: [42]
Sorted: [1, 2, 3, 4, 5]
```

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Problem 2-1.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley, Section 2.2.
- Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.). Addison-Wesley, Section 5.2.4.
