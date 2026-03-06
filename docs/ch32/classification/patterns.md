# Recognizing Patterns

The first and most critical step in solving any algorithmic problem is **pattern recognition** -- identifying which category of known problems your current task resembles. Experienced problem solvers maintain a mental catalog of problem archetypes and match new problems against them.

## Common Problem Archetypes

| Pattern | Signature | Example |
|---|---|---|
| Sorting + Greedy | Optimal ordering, scheduling | Activity selection, job scheduling |
| Two Pointers | Sorted array, pair conditions | Two-sum on sorted array |
| Sliding Window | Contiguous subarray/substring | Maximum sum subarray of size $k$ |
| BFS/DFS | Graph traversal, connected components | Shortest path in unweighted graph |
| Dynamic Programming | Overlapping subproblems, optimal substructure | Longest common subsequence |
| Binary Search | Monotonic predicate | Minimum capacity to ship packages |
| Union-Find | Connectivity, grouping | Number of connected components |
| Topological Sort | Ordering with dependencies | Course schedule |
| Divide and Conquer | Independent subproblems | Merge sort, closest pair of points |

## Signal Words in Problem Statements

Certain keywords in problem descriptions hint at the correct approach:

- **"Minimum number of operations"** -- BFS or DP
- **"All possible combinations/permutations"** -- Backtracking
- **"Maximum/minimum with constraints"** -- DP or Greedy
- **"Can you reach / is it possible"** -- BFS/DFS or DP
- **"Contiguous subarray"** -- Sliding window or prefix sums
- **"Sorted input"** -- Binary search or two pointers

## Example: Classifying a New Problem

**Problem:** Given an array of integers, find the length of the longest subarray whose sum is at most $k$.

**Analysis:**
- "Subarray" means contiguous, suggesting a sliding window candidate.
- "Longest" means maximize window size.
- "Sum at most $k$" is a monotonic condition (adding elements increases sum).

This matches the **sliding window** pattern perfectly.

```python
def longest_subarray_at_most_k(arr, k):
    left = 0
    current_sum = 0
    max_length = 0
    for right in range(len(arr)):
        current_sum += arr[right]
        while current_sum > k and left <= right:
            current_sum -= arr[left]
            left += 1
        max_length = max(max_length, right - left + 1)
    return max_length

# Example
print(longest_subarray_at_most_k([1, 2, 3, 1, 1], 5))  # Output: 3
```

## Complexity of Pattern Recognition

The more problems you solve, the faster you recognize patterns. A structured approach:

1. Read the problem constraints ($n \le 10^5$ suggests $O(n \log n)$).
2. Identify the input/output structure.
3. Match against known archetypes.
4. Verify by checking edge cases.

# Reference

- Skiena, S. *The Algorithm Design Manual*, Springer, 2020.
- LeetCode Patterns: [Blind 75](https://leetcode.com/discuss/general-discussion/460599/blind-75-leetcode-questions)
