# Incremental Design

**Incremental design** (also called **online** or **streaming** algorithms) builds solutions by processing one element at a time and maintaining a valid solution at every step. This is the foundation of many efficient algorithms.

## The Incremental Paradigm

$$\text{Solution}(n) = \text{Update}(\text{Solution}(n-1), \text{element}_n)$$

At each step, we incorporate the next element and update our answer in $O(1)$ or $O(\log n)$ time, achieving an overall efficient solution.

## Example: Kadane's Algorithm (Incremental Max Subarray)

```python
def kadane(arr):
    max_ending_here = arr[0]
    max_so_far = arr[0]
    for i in range(1, len(arr)):
        # Incremental decision: extend current subarray or start new
        max_ending_here = max(arr[i], max_ending_here + arr[i])
        max_so_far = max(max_so_far, max_ending_here)
    return max_so_far

print(kadane([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # Output: 6
```

## Example: Incremental Convex Hull

Andrew's monotone chain algorithm adds points one at a time and maintains the hull:

```python
def cross(O, A, B):
    return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    # Build lower hull incrementally
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull incrementally
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

points = [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0), (1, 0)]
print(convex_hull(points))
```

## Example: Running Median (Incremental)

Maintain two heaps to get the median as each element arrives:

```python
import heapq

class RunningMedian:
    def __init__(self):
        self.lo = []  # max-heap (negated)
        self.hi = []  # min-heap

    def add(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2

rm = RunningMedian()
for x in [2, 1, 5, 7, 2, 0, 5]:
    rm.add(x)
    print(f"Added {x}, median = {rm.median()}")
```

# Reference

- Cormen, T. et al. *Introduction to Algorithms*, MIT Press, 2022.
- Skiena, S. *The Algorithm Design Manual*, Springer, 2020.
