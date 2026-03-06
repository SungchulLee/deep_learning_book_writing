# Monotonic Stack/Queue

A **monotonic stack** (or **monotonic queue**) maintains elements in sorted order as they are pushed, enabling efficient solutions to problems involving the **next greater/smaller element** or **sliding window min/max**.

## Monotonic Stack

A monotonic stack pops elements that violate the monotonic order before pushing a new element.

### Next Greater Element

For each element, find the first element to its right that is greater.

```python
def next_greater_element(arr):
    n = len(arr)
    result = [-1] * n
    stack = []  # indices, arr values are decreasing
    for i in range(n):
        while stack and arr[stack[-1]] < arr[i]:
            result[stack.pop()] = arr[i]
        stack.append(i)
    return result

print(next_greater_element([4, 5, 2, 10, 8]))
# Output: [5, 10, 10, -1, -1]
```

### Largest Rectangle in Histogram

```python
def largest_rectangle_histogram(heights):
    stack = []
    max_area = 0
    heights.append(0)  # sentinel
    for i, h in enumerate(heights):
        while stack and heights[stack[-1]] > h:
            height = heights[stack.pop()]
            width = i if not stack else i - stack[-1] - 1
            max_area = max(max_area, height * width)
        stack.append(i)
    heights.pop()  # remove sentinel
    return max_area

print(largest_rectangle_histogram([2, 1, 5, 6, 2, 3]))  # Output: 10
```

## Monotonic Queue (Deque)

A monotonic deque efficiently answers **sliding window minimum/maximum** queries in $O(1)$ amortized per element.

### Sliding Window Maximum

```python
from collections import deque

def sliding_window_max(arr, k):
    dq = deque()  # stores indices; arr values are decreasing
    result = []
    for i in range(len(arr)):
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        while dq and arr[dq[-1]] <= arr[i]:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            result.append(arr[dq[0]])
    return result

print(sliding_window_max([1, 3, -1, -3, 5, 3, 6, 7], 3))
# Output: [3, 3, 5, 5, 6, 7]
```

## Complexity

| Operation | Time | Space |
|---|---|---|
| Next greater element | $O(n)$ | $O(n)$ |
| Largest rectangle in histogram | $O(n)$ | $O(n)$ |
| Sliding window max/min | $O(n)$ total | $O(k)$ |

Each element is pushed and popped at most once, giving amortized $O(1)$ per operation.

# Reference

- LeetCode: [84. Largest Rectangle in Histogram](https://leetcode.com/problems/largest-rectangle-in-histogram/)
- LeetCode: [239. Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/)
