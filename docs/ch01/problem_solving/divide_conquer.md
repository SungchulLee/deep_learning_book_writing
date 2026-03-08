# Divide and Conquer Preview

**Divide and Conquer** breaks a problem into smaller subproblems, solves them recursively, and combines the results.

$$

T(n) = aT\left(\frac{n}{b}\right) + f(n)

$$

where $a$ = number of subproblems, $n/b$ = subproblem size, $f(n)$ = combine cost.

## Three Steps

1. **Divide** the problem into subproblems
2. **Conquer** subproblems recursively
3. **Combine** solutions

```python
def merge_sort(arr):
    """Classic divide and conquer: O(n log n)."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def main():
    arr = [38, 27, 43, 3, 9, 82, 10]
    print(f"Input:  {arr}")
    print(f"Sorted: {merge_sort(arr)}")

if __name__ == "__main__":
    main()
```

**Output:**
```
Input:  [38, 27, 43, 3, 9, 82, 10]
Sorted: [3, 9, 10, 27, 38, 43, 82]
```

# Reference

[Introduction to Algorithms (CLRS), Chapter 4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
