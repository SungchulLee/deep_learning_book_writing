# Deduplication

Given a collection of $n$ elements, **deduplication** removes all duplicate entries, retaining exactly one copy of each distinct value. This operation is fundamental in data processing --- from cleaning database records to removing redundant entries in log files. Hash-based sets provide the fastest general-purpose solution, achieving $O(n)$ expected time compared to $O(n^2)$ for brute force and $O(n \log n)$ for sorting-based approaches.

## Problem Statement

Given an array $A[0 \ldots n-1]$, produce an array $B$ containing exactly one copy of each distinct element in $A$, preserving the first occurrence order. The number of distinct elements is $d \le n$.

## Approaches and Complexity

### Brute Force

For each element, scan all previous elements to check for duplicates:

$$
T(n) = \sum_{i=0}^{n-1} O(i) = O(n^2)
$$

This requires no extra space beyond the output array.

### Sorting-Based

Sort the array, then scan linearly to remove adjacent duplicates:

$$
T(n) = O(n \log n) + O(n) = O(n \log n)
$$

This approach is efficient but destroys the original insertion order. A stable sort preserves relative order, but the overall cost remains $O(n \log n)$.

### Hash-Set-Based

Insert each element into a hash set. If the element is already present, skip it; otherwise, add it to the output:

$$
T(n) = O(n) \quad \text{expected}
$$

Each insertion and membership test in the hash set takes $O(1)$ expected time under the simple uniform hashing assumption. The space cost is $O(d)$ for the hash set.

## Complexity Comparison

| Method | Time | Space | Order preserved |
|---|---|---|---|
| Brute force | $O(n^2)$ | $O(1)$ | Yes |
| Sorting | $O(n \log n)$ | $O(1)$ or $O(n)$ | No (unless stable) |
| Hash set | $O(n)$ expected | $O(d)$ | Yes |

The hash-set approach dominates when order preservation is required and extra space is available.

## Correctness Argument

The hash-set method is correct because:

1. **No duplicates in output**: an element is appended to the output only when it is absent from the set. After appending, it is added to the set, preventing future duplicates.
2. **No elements lost**: every element in $A$ is processed. If it is new, it is added to the output; if it is a duplicate, its first occurrence is already in the output.
3. **Order preserved**: elements are appended to the output in the order they first appear in $A$.

## Streaming Deduplication

When the input arrives as a stream and cannot be stored entirely in memory, exact deduplication requires $O(d)$ space for the hash set, which may be prohibitive for large $d$. In such cases, approximate deduplication using a **Bloom filter** trades a small false positive rate for dramatically reduced memory:

- A Bloom filter uses $O(d)$ bits (not entries) to represent the seen set.
- False positives cause some unique elements to be incorrectly classified as duplicates.
- False negatives never occur: no duplicate passes through as unique.

## Python Implementation

```python
"""
Deduplication using hash sets.

Compares brute-force, sorting-based, and hash-set-based
approaches to removing duplicate elements.
"""


# === Brute Force Deduplication ===

def dedup_brute(arr):
    """Remove duplicates in O(n^2) time, preserving order."""
    result = []
    for item in arr:
        if item not in result:
            result.append(item)
    return result


# === Sorting-Based Deduplication ===

def dedup_sort(arr):
    """Remove duplicates in O(n log n) time. Does NOT preserve order."""
    if not arr:
        return []
    sorted_arr = sorted(arr)
    result = [sorted_arr[0]]
    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] != sorted_arr[i - 1]:
            result.append(sorted_arr[i])
    return result


# === Hash-Set Deduplication ===

def dedup_hash(arr):
    """Remove duplicates in O(n) expected time, preserving order."""
    seen = set()
    result = []
    for item in arr:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# === Demonstration ===

if __name__ == "__main__":
    data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

    print(f"Input:      {data}")
    print(f"Brute:      {dedup_brute(data)}")
    print(f"Sort-based: {dedup_sort(data)}")
    print(f"Hash-based: {dedup_hash(data)}")

    # String deduplication
    words = ["apple", "banana", "apple", "cherry", "banana", "date"]
    print(f"\nWords input: {words}")
    print(f"Hash-based:  {dedup_hash(words)}")
```

**Output:**
```
Input:      [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
Brute:      [3, 1, 4, 5, 9, 2, 6]
Sort-based: [1, 2, 3, 4, 5, 6, 9]
Hash-based: [3, 1, 4, 5, 9, 2, 6]

Words input: ['apple', 'banana', 'apple', 'cherry', 'banana', 'date']
Hash-based:  ['apple', 'banana', 'cherry', 'date']
```

## Reference

- [Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
