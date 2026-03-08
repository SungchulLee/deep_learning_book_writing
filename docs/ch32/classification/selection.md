# When to Use What

Choosing the right algorithmic technique is the most important decision in problem solving. This page provides a systematic guide for selecting the appropriate approach based on problem characteristics.

## Decision Matrix

| Problem Characteristic | Recommended Technique | Time Complexity |
|---|---|---|
| Find shortest path (unweighted) | BFS | $O(V + E)$ |
| Find shortest path (weighted, non-negative) | Dijkstra | $O((V+E) \log V)$ |
| Find shortest path (negative weights) | Bellman-Ford | $O(VE)$ |
| Optimize over subsets with constraints | DP | Varies |
| Schedule / select maximum non-overlapping | Greedy (sort by end) | $O(n \log n)$ |
| Find connected components | BFS/DFS or Union-Find | $O(V + E)$ |
| Dependency ordering | Topological Sort | $O(V + E)$ |
| Range queries (static) | Sparse Table / Prefix Sums | $O(1)$ query |
| Range queries (dynamic) | Segment Tree / BIT | $O(\log n)$ query |
| String matching | KMP / Rabin-Karp | $O(n + m)$ |
| Contiguous subarray optimization | Sliding Window | $O(n)$ |
| Pair/triplet in sorted array | Two Pointers | $O(n)$ |
| Search in sorted space | Binary Search | $O(\log n)$ |

## Constraint-Based Selection

The input size $n$ often dictates which approach is feasible within typical time limits (1-2 seconds):

| Constraint | Feasible Complexity | Suggested Approach |
|---|---|---|
| $n \le 20$ | $O(2^n)$ or $O(n!)$ | Brute force, bitmask DP |
| $n \le 100$ | $O(n^3)$ | Floyd-Warshall, cubic DP |
| $n \le 5000$ | $O(n^2)$ | Quadratic DP, nested loops |
| $n \le 10^5$ | $O(n \log n)$ | Sorting, segment trees, binary search |
| $n \le 10^7$ | $O(n)$ | Linear scan, prefix sums, two pointers |
| $n \le 10^{18}$ | $O(\log n)$ | Binary search, matrix exponentiation |

## Example: Choosing Between Approaches

**Problem:** Given $n$ intervals, find the minimum number of points such that every interval contains at least one point.

**Analysis:**
- This is an interval covering / scheduling problem.
- Sort intervals by right endpoint.
- Greedily place a point at the rightmost position of the first uncovered interval.
- This has the greedy-choice property, so **Greedy** is correct.

```python
def min_points_to_cover(intervals):
    intervals.sort(key=lambda x: x[1])
    points = []
    for start, end in intervals:
        if not points or points[-1] < start:
            points.append(end)
    return points

intervals = [(1, 4), (2, 6), (5, 7), (8, 10), (9, 12)]
print(min_points_to_cover(intervals))  # Output: [4, 7, 10]
```

# Reference

- Skiena, S. *The Algorithm Design Manual*, Springer, 2020.
- Halim, S. & Halim, F. *Competitive Programming 4*, 2020.
