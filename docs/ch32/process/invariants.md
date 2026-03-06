# Finding Invariants

An **invariant** is a property that remains true throughout the execution of an algorithm. Identifying invariants is a powerful technique for both designing correct algorithms and proving their correctness.

## What Is an Invariant?

A **loop invariant** holds at the start (and end) of every iteration of a loop. It serves three purposes:

1. **Initialization** -- it is true before the first iteration.
2. **Maintenance** -- if it is true before an iteration, it remains true after.
3. **Termination** -- when the loop ends, the invariant gives a useful property that helps show correctness.

## Example: Binary Search Invariant

In binary search on a sorted array, the invariant is:

$$\text{If the target exists in the array, it lies in } arr[lo \ldots hi]$$

```python
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    # Invariant: if target in arr, then arr[lo] <= target <= arr[hi]
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1  # Invariant maintained: target > arr[mid]
        else:
            hi = mid - 1  # Invariant maintained: target < arr[mid]
    return -1  # Invariant: lo > hi, so target not in arr
```

## Example: Partition Invariant (Quicksort)

In quicksort's Lomuto partition, the invariant is:

$$arr[low \ldots i-1] \le \text{pivot} \quad \text{and} \quad arr[i \ldots j-1] > \text{pivot}$$

```python
def partition(arr, low, high):
    pivot = arr[high]
    i = low
    # Invariant: arr[low..i-1] <= pivot, arr[i..j-1] > pivot
    for j in range(low, high):
        if arr[j] <= pivot:
            arr[i], arr[j] = arr[j], arr[i]
            i += 1
    arr[i], arr[high] = arr[high], arr[i]
    return i
```

## Example: Two-Pointer Invariant

For the two-sum problem on a sorted array, the invariant is:

$$\text{No valid pair exists entirely in } arr[0 \ldots l-1] \text{ or } arr[r+1 \ldots n-1]$$

```python
def two_sum_sorted(arr, target):
    l, r = 0, len(arr) - 1
    while l < r:
        s = arr[l] + arr[r]
        if s == target:
            return (l, r)
        elif s < target:
            l += 1  # arr[l] too small
        else:
            r -= 1  # arr[r] too large
    return None
```

## Using Invariants in Design

When designing an algorithm:

1. Decide what invariant you want to maintain.
2. Choose operations that preserve the invariant.
3. Show that the invariant implies correctness at termination.

# Reference

- Cormen, T. et al. *Introduction to Algorithms*, Chapter 2.1 (Loop Invariants), MIT Press, 2022.
- Gries, D. *The Science of Programming*, Springer, 1981.
