# Two Pointers

The **two pointers** technique uses two indices that move through a data structure (usually a sorted array) to solve problems in $O(n)$ time that would naively require $O(n^2)$.

## When to Use Two Pointers

- The input is **sorted** (or can be sorted).
- You are looking for **pairs** or **subarrays** satisfying a condition.
- The condition is **monotonic**: moving one pointer in a direction either improves or worsens the condition.

## Pattern 1: Opposite Ends

Two pointers start at opposite ends and move inward.

**Problem:** Find two numbers in a sorted array that sum to a target.

```python
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        s = arr[left] + arr[right]
        if s == target:
            return (left, right)
        elif s < target:
            left += 1
        else:
            right -= 1
    return None

print(two_sum_sorted([1, 2, 4, 6, 8, 10], 10))  # Output: (1, 4)
```

## Pattern 2: Same Direction (Fast/Slow)

Both pointers start at the beginning; one moves faster.

**Problem:** Remove duplicates from a sorted array in-place.

```python
def remove_duplicates(arr):
    if not arr:
        return 0
    slow = 0
    for fast in range(1, len(arr)):
        if arr[fast] != arr[slow]:
            slow += 1
            arr[slow] = arr[fast]
    return slow + 1

arr = [1, 1, 2, 2, 3, 4, 4, 5]
length = remove_duplicates(arr)
print(arr[:length])  # Output: [1, 2, 3, 4, 5]
```

## Pattern 3: Three Pointers (3Sum)

Reduce 3-sum to 2-sum by fixing one element and using two pointers on the rest.

```python
def three_sum(nums):
    nums.sort()
    result = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            s = nums[i] + nums[left] + nums[right]
            if s == 0:
                result.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]:
                    left += 1
                while left < right and nums[right] == nums[right-1]:
                    right -= 1
                left += 1
                right -= 1
            elif s < 0:
                left += 1
            else:
                right -= 1
    return result

print(three_sum([-1, 0, 1, 2, -1, -4]))
# Output: [[-1, -1, 2], [-1, 0, 1]]
```

## Complexity

| Pattern | Time | Space |
|---|---|---|
| Opposite ends | $O(n)$ | $O(1)$ |
| Fast/slow | $O(n)$ | $O(1)$ |
| 3Sum (fix + two pointers) | $O(n^2)$ | $O(1)$ extra |

# Reference

- LeetCode Two Pointers tag: [https://leetcode.com/tag/two-pointers/](https://leetcode.com/tag/two-pointers/)
- Skiena, S. *The Algorithm Design Manual*, Springer, 2020.
