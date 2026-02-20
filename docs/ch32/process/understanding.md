# Understanding the Problem

Before writing a single line of code, you must **fully understand the problem**. Misunderstanding even one constraint wastes enormous effort. This page describes a disciplined approach to problem comprehension.

## The Five-Step Understanding Protocol

1. **Read the entire problem** -- do not start coding after reading half the statement.
2. **Identify inputs and outputs** -- what exactly is given, and what exactly must be returned?
3. **Note constraints** -- input sizes, value ranges, time/memory limits.
4. **Work through examples** -- manually trace the sample inputs and outputs.
5. **Identify edge cases** -- empty input, single element, maximum size, negative values.

## Constraint Analysis

Constraints are the most valuable part of a problem statement. They tell you which algorithms are feasible:

$$n \le 10^5 \implies O(n \log n) \text{ is likely expected}$$
$$n \le 20 \implies O(2^n) \text{ or } O(n \cdot 2^n) \text{ bitmask approach}$$

## Example: Parsing a Problem

**Problem statement:** "Given an array of $n$ integers ($1 \le n \le 10^5$, $-10^9 \le a_i \le 10^9$), find the maximum sum of any contiguous subarray."

**Understanding checklist:**

- **Input:** Array of integers (can be negative).
- **Output:** A single integer (maximum subarray sum).
- **Constraints:** $n$ up to $10^5$, need $O(n)$ or $O(n \log n)$.
- **Edge cases:** All negative numbers (answer is the largest single element), single element array.
- **Known algorithm:** Kadane's algorithm, $O(n)$.

```python
def max_subarray_sum(arr):
    max_sum = arr[0]
    current = arr[0]
    for i in range(1, len(arr)):
        current = max(arr[i], current + arr[i])
        max_sum = max(max_sum, current)
    return max_sum

print(max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]))  # Output: 6
print(max_subarray_sum([-3, -2, -1]))  # Output: -1 (edge case)
```

## Common Misunderstandings

| Trap | Consequence |
|---|---|
| Ignoring that array can be all negative | Returning 0 instead of max negative |
| Missing 1-indexed vs 0-indexed | Off-by-one errors |
| Confusing "subsequence" with "subarray" | Completely wrong approach |
| Overlooking modular arithmetic requirement | Integer overflow or wrong answer |

# Reference

- Polya, G. *How to Solve It*, Princeton University Press, 1945.
- Skiena, S. *The Algorithm Design Manual*, Chapter 1, Springer, 2020.
