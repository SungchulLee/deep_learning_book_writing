# Brute Force

**Brute force** (exhaustive search) tries all possible solutions and picks the best one.

$$

T(n) = |\text{Solution Space}| \times \text{Cost per check}

$$

## Characteristics

- **Simple** to implement and understand
- **Correct** by construction (checks everything)
- **Slow** — often exponential time
- **Baseline** — useful for testing optimized solutions

```python
def two_sum_brute_force(arr, target):
    """Find two elements that sum to target. O(n^2)."""
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if arr[i] + arr[j] == target:
                return (i, j)
    return None

def two_sum_optimized(arr, target):
    """Find two elements that sum to target. O(n)."""
    seen = {}
    for i, val in enumerate(arr):
        complement = target - val
        if complement in seen:
            return (seen[complement], i)
        seen[val] = i
    return None

def main():
    arr = [2, 7, 11, 15]
    target = 9
    print(f"Brute force: {two_sum_brute_force(arr, target)}")
    print(f"Optimized:   {two_sum_optimized(arr, target)}")

if __name__ == "__main__":
    main()
```

**Output:**
```
Brute force: (0, 1)
Optimized:   (0, 1)
```

# Reference

[Introduction to Algorithms (CLRS), Chapter 2](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
