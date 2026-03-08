# Prefix Sums

A **prefix sum** array allows answering range sum queries in $O(1)$ after $O(n)$ preprocessing. It is one of the most fundamental optimization techniques.

## Definition

Given an array $a[0 \ldots n-1]$, the prefix sum array $P$ is defined as:

$$P[0] = 0, \quad P[i] = \sum_{j=0}^{i-1} a[j] = P[i-1] + a[i-1]$$

The sum of elements in range $[l, r)$ is:

$$\sum_{j=l}^{r-1} a[j] = P[r] - P[l]$$

## 1D Prefix Sum

```python
def build_prefix_sum(arr):
    prefix = [0] * (len(arr) + 1)
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]
    return prefix

def range_sum(prefix, l, r):
    return prefix[r + 1] - prefix[l]

arr = [3, 1, 4, 1, 5, 9, 2, 6]
prefix = build_prefix_sum(arr)
print(range_sum(prefix, 2, 5))  # Output: 19  (4+1+5+9)
```

## 2D Prefix Sum

For a matrix, the prefix sum allows $O(1)$ submatrix sum queries:

$$\text{Sum}(r_1, c_1, r_2, c_2) = P[r_2+1][c_2+1] - P[r_1][c_2+1] - P[r_2+1][c_1] + P[r_1][c_1]$$

```python
def build_2d_prefix(matrix):
    rows, cols = len(matrix), len(matrix[0])
    P = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(rows):
        for j in range(cols):
            P[i+1][j+1] = matrix[i][j] + P[i][j+1] + P[i+1][j] - P[i][j]
    return P

def submatrix_sum(P, r1, c1, r2, c2):
    return P[r2+1][c2+1] - P[r1][c2+1] - P[r2+1][c1] + P[r1][c1]

matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
P = build_2d_prefix(matrix)
print(submatrix_sum(P, 1, 1, 2, 2))  # Output: 28  (5+6+8+9)
```

## Application: Count Subarrays with Sum Equal to k

```python
from collections import defaultdict

def count_subarrays_sum_k(arr, k):
    count = 0
    prefix = 0
    freq = defaultdict(int)
    freq[0] = 1
    for x in arr:
        prefix += x
        count += freq[prefix - k]
        freq[prefix] += 1
    return count

print(count_subarrays_sum_k([1, 1, 1], 2))  # Output: 2
```

## Complexity

| Operation | Time | Space |
|---|---|---|
| Build 1D prefix sum | $O(n)$ | $O(n)$ |
| Query 1D range sum | $O(1)$ | -- |
| Build 2D prefix sum | $O(nm)$ | $O(nm)$ |
| Query 2D submatrix sum | $O(1)$ | -- |

# Reference

- Cormen, T. et al. *Introduction to Algorithms*, MIT Press, 2022.
- [Prefix Sum Array -- GeeksforGeeks](https://www.geeksforgeeks.org/prefix-sum-array-implementation-applications-competitive-programming/)
