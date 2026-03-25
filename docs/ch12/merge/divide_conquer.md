# Merge Sort as Divide and Conquer

Merge sort is the textbook example of the **divide-and-conquer** paradigm.  Many sorting algorithms process the input incrementally (insertion sort) or by selection (selection sort), but merge sort takes a fundamentally different approach: it splits the problem in half, solves each half recursively, and combines the solutions.  Understanding this structure is essential because the same paradigm underlies algorithms far beyond sorting -- from Strassen's matrix multiplication to the closest-pair problem.

## The Three Steps

Every divide-and-conquer algorithm follows three steps:

1. **Divide**: split the input into smaller subproblems of the same type.
2. **Conquer**: solve each subproblem recursively (or directly if small enough).
3. **Combine**: merge the subproblem solutions into a solution for the original problem.

For merge sort on an array $A[0..n-1]$:

| Step     | Merge sort action                           | Cost      |
|----------|---------------------------------------------|-----------|
| Divide   | Compute $\text{mid} = \lfloor n/2 \rfloor$ | $O(1)$    |
| Conquer  | Recursively sort $A[0..\text{mid}-1]$ and $A[\text{mid}..n-1]$ | $2T(n/2)$ |
| Combine  | Merge the two sorted halves                 | $O(n)$    |

The divide step is trivial (just an index computation), and the combine step is where the real work happens via the merge procedure.

## Recurrence Relation

The three-step structure leads directly to a recurrence for the running time $T(n)$:

$$
T(n) = \begin{cases} O(1) & \text{if } n \leq 1 \\ 2T(n/2) + O(n) & \text{if } n > 1 \end{cases}
$$

This recurrence captures the fact that we make two recursive calls on problems of half the size, plus linear work to merge.

### Solving by the Master Theorem

The recurrence has the form $T(n) = aT(n/b) + f(n)$ with $a = 2$, $b = 2$, and $f(n) = O(n)$.  The critical exponent is:

$$
\log_b a = \log_2 2 = 1
$$

Since $f(n) = \Theta(n^1)$, we are in **Case 2** of the Master Theorem ($f(n) = \Theta(n^{\log_b a})$), giving:

$$
T(n) = \Theta(n \log n)
$$

### Solving by Recursion Tree

An alternative derivation unfolds the recursion into a tree:

- **Level 0**: one problem of size $n$, total merge work $= cn$.
- **Level 1**: two problems of size $n/2$, total merge work $= 2 \cdot c(n/2) = cn$.
- **Level 2**: four problems of size $n/4$, total merge work $= 4 \cdot c(n/4) = cn$.
- **Level $k$**: $2^k$ problems of size $n/2^k$, total merge work $= cn$.

The tree has $\log_2 n$ levels, each contributing $cn$ work:

$$
T(n) = cn \cdot \log_2 n = \Theta(n \log n)
$$

??? note "Why each level costs exactly cn"
    At level $k$, the $2^k$ subproblems collectively contain all $n$ elements (partitioned among them).  The merge at each node touches every element in that subproblem exactly once.  Since the subproblems at any level are disjoint and cover the entire array, the total merge work per level is $cn$.

## Top-Down Merge Sort Algorithm

```
MERGE-SORT(A, left, right):
    if left < right:
        mid = (left + right) / 2
        MERGE-SORT(A, left, mid)       // conquer left half
        MERGE-SORT(A, mid + 1, right)  // conquer right half
        MERGE(A, left, mid, right)     // combine
```

The recursion bottoms out when a subarray has zero or one elements (already sorted by definition).

## Recursion Tree Visualization

Consider sorting $[38, 27, 43, 3, 9, 82, 10]$:

```
Level 0:  [38, 27, 43, 3, 9, 82, 10]
           /                       \
Level 1:  [38, 27, 43, 3]    [9, 82, 10]
           /         \          /       \
Level 2:  [38, 27]  [43, 3]  [9, 82]  [10]
           / \       / \       / \
Level 3:  [38][27] [43][3]  [9][82]

--- merge back up ---

Level 3:  [27,38]  [3,43]   [9,82]   [10]
Level 2:  [3, 27, 38, 43]   [9, 10, 82]
Level 1:  [3, 9, 10, 27, 38, 43, 82]
```

## Python Implementation

```python
"""
Merge sort as divide and conquer.

Implements top-down merge sort, illustrating the divide-conquer-combine
structure with clear separation of the three phases.
"""


# === Merge (combine step) =====================================================

def merge(arr: list, left: int, mid: int, right: int) -> None:
    """Merge two sorted subarrays arr[left..mid] and arr[mid+1..right] in place.

    Uses O(n) auxiliary space for temporary copies of both halves.
    The <= comparison ensures stability.
    """
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


# === Merge sort (divide and conquer) ==========================================

def merge_sort(arr: list, left: int = 0, right: int = None) -> None:
    """Sort arr[left..right] using top-down merge sort.

    Parameters
    ----------
    arr : list
        The array to sort (modified in place).
    left : int
        Start index of the subarray.
    right : int or None
        End index (inclusive).  Defaults to len(arr) - 1.
    """
    if right is None:
        right = len(arr) - 1

    if left < right:
        mid = (left + right) // 2       # Divide
        merge_sort(arr, left, mid)       # Conquer left
        merge_sort(arr, mid + 1, right)  # Conquer right
        merge(arr, left, mid, right)     # Combine


# === Main =====================================================================

if __name__ == "__main__":
    data = [38, 27, 43, 3, 9, 82, 10]
    print(f"Before: {data}")
    merge_sort(data)
    print(f"After:  {data}")

    # Verify stability: sort by first element of tuples
    records = [(3, "b"), (1, "a"), (3, "d"), (1, "c"), (2, "e")]
    records.sort(key=lambda x: x[0])  # Python's stable sort for reference
    print(f"\nPython stable sort: {records}")
```

**Output:**
```
Before: [38, 27, 43, 3, 9, 82, 10]
After:  [3, 9, 10, 27, 38, 43, 82]

Python stable sort: [(1, 'a'), (1, 'c'), (2, 'e'), (3, 'b'), (3, 'd')]
```

## Why Divide and Conquer Achieves n log n

The insight is that dividing the problem in half at each level creates $\log n$ levels of recursion, and each level performs $O(n)$ total merge work across all subproblems at that level.  The product $O(n) \times O(\log n) = O(n \log n)$ is the hallmark of efficient divide-and-conquer algorithms.

This is a strict improvement over the $O(n^2)$ algorithms (insertion sort, selection sort, bubble sort), which effectively reduce the problem size by only one element at each step, requiring $n$ levels of $O(n)$ work each.

## References

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press, Section 2.3.
- Sedgewick, R., & Wayne, K. (2011). *Algorithms* (4th ed.). Addison-Wesley, Section 2.2.
