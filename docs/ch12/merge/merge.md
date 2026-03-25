# The Merge Procedure

Merge sort's power lies in a single, elegant subroutine: the **merge** procedure.  Given two sorted arrays, merge combines them into one sorted array in linear time by scanning both arrays from left to right and always choosing the smaller element.  This $O(n)$ merge step is the workhorse of the entire algorithm -- every recursive call ultimately relies on it, and its efficiency is what makes merge sort achieve $O(n \log n)$ overall.

## How Merge Works

Given two sorted subarrays $L[0..p-1]$ and $R[0..q-1]$, the merge procedure produces a single sorted array $A[0..p+q-1]$:

1. Maintain three indices: $i$ into $L$, $j$ into $R$, and $k$ into $A$.
2. Compare $L[i]$ with $R[j]$. Place the smaller value into $A[k]$.
3. Advance the index of whichever subarray contributed the element, and advance $k$.
4. When one subarray is exhausted, copy the remainder of the other subarray into $A$.

Because both subarrays are sorted, each comparison places exactly one element in its correct position.

## Pseudocode

```
MERGE(A, left, mid, right):
    L = A[left .. mid]          // copy left half
    R = A[mid+1 .. right]       // copy right half
    i = 0, j = 0, k = left

    while i < |L| and j < |R|:
        if L[i] <= R[j]:        // <= ensures stability
            A[k] = L[i]
            i = i + 1
        else:
            A[k] = R[j]
            j = j + 1
        k = k + 1

    copy remaining elements of L (if any) into A[k..]
    copy remaining elements of R (if any) into A[k..]
```

!!! tip "Stability through tie-breaking"
    The `<=` comparison (rather than `<`) ensures that when $L[i] = R[j]$, the element from the left subarray is chosen first.  This preserves the original relative order of equal elements, making merge sort **stable**.

## Complexity Analysis

**Time complexity.**  Each element is compared at most once and copied exactly once.  With $p + q = n$ total elements:

$$
T_{\text{merge}}(n) = O(n)
$$

Precisely, the merge performs at most $n - 1$ comparisons (when the last comparison exhausts one subarray) and exactly $n$ assignments.

**Space complexity.**  The procedure requires $O(n)$ auxiliary space for the temporary copies $L$ and $R$.  This is the primary cost of merge sort compared to in-place algorithms like heapsort.

## Step-by-Step Example

Merge $L = [3, 9, 27]$ and $R = [10, 38, 43]$:

| Step | $L[i]$ | $R[j]$ | Choice | $A$ so far |
|------|---------|---------|--------|------------|
| 1    | 3       | 10      | 3 (L)  | $[3]$ |
| 2    | 9       | 10      | 9 (L)  | $[3, 9]$ |
| 3    | 27      | 10      | 10 (R) | $[3, 9, 10]$ |
| 4    | 27      | 38      | 27 (L) | $[3, 9, 10, 27]$ |
| 5    | --      | 38      | copy R | $[3, 9, 10, 27, 38, 43]$ |

Total: 4 comparisons for 6 elements.

## Sentinel Technique

CLRS presents a variant using **sentinel values** ($\infty$) appended to $L$ and $R$.  This eliminates the need to check whether either subarray is exhausted during the main loop:

```
MERGE-WITH-SENTINELS(A, left, mid, right):
    L = A[left..mid] + [∞]
    R = A[mid+1..right] + [∞]
    i = 0, j = 0

    for k = left to right:
        if L[i] <= R[j]:
            A[k] = L[i]
            i = i + 1
        else:
            A[k] = R[j]
            j = j + 1
```

The sentinel values ensure that whenever one subarray is exhausted, every remaining comparison selects from the other subarray.  The asymptotic complexity is unchanged, but the inner loop has one fewer branch.

## Python Implementation

```python
"""
The merge procedure for merge sort.

Demonstrates both the standard two-pointer merge and the sentinel
variant.  The merge operation combines two sorted sequences into
one sorted sequence in O(n) time.
"""


# === Standard merge ===========================================================

def merge(left: list, right: list) -> list:
    """Merge two sorted lists into a single sorted list.

    Uses the two-pointer technique.  The <= comparison ensures stability.

    Parameters
    ----------
    left : list
        First sorted list.
    right : list
        Second sorted list.

    Returns
    -------
    list
        Merged sorted list of length len(left) + len(right).
    """
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result


# === Sentinel merge ===========================================================

def merge_sentinel(left: list, right: list) -> list:
    """Merge two sorted lists using sentinel values.

    Appends float('inf') to both lists to eliminate boundary checks.

    Parameters
    ----------
    left : list
        First sorted list.
    right : list
        Second sorted list.

    Returns
    -------
    list
        Merged sorted list.
    """
    left_s = left + [float("inf")]
    right_s = right + [float("inf")]
    result = []
    i = j = 0
    for _ in range(len(left) + len(right)):
        if left_s[i] <= right_s[j]:
            result.append(left_s[i])
            i += 1
        else:
            result.append(right_s[j])
            j += 1
    return result


# === Main =====================================================================

if __name__ == "__main__":
    L = [3, 9, 27]
    R = [10, 38, 43]

    print("Standard merge:")
    print(f"  merge({L}, {R}) = {merge(L, R)}")

    print("Sentinel merge:")
    print(f"  merge({L}, {R}) = {merge_sentinel(L, R)}")

    # Stability demonstration: merge records with equal keys
    records_l = [(1, "a"), (3, "b"), (5, "c")]
    records_r = [(1, "d"), (3, "e"), (4, "f")]
    merged = merge(records_l, records_r)
    print(f"\nStability test: {merged}")
    print("  (1,'a') appears before (1,'d') -- stable")
```

**Output:**
```
Standard merge:
  merge([3, 9, 27], [10, 38, 43]) = [3, 9, 10, 27, 38, 43]
Sentinel merge:
  merge([3, 9, 27], [10, 38, 43]) = [3, 9, 10, 27, 38, 43]

Stability test: [(1, 'a'), (1, 'd'), (3, 'b'), (3, 'e'), (4, 'f'), (5, 'c')]
  (1,'a') appears before (1,'d') -- stable
```

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Section 2.3.1.
- Knuth, D. E. (1998). *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.). Addison-Wesley, Section 5.2.4.
