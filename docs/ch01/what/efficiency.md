# Efficiency

Algorithm efficiency measures how resources (time and space) grow with input size $n$.

$$

\text{Efficiency} = f(n) \text{ where } n = |\text{input}|

$$

## Why Efficiency Matters

| Input Size $n$ | $O(n)$ | $O(n \log n)$ | $O(n^2)$ | $O(2^n)$ |
|---|---|---|---|---|
| 10 | 10 | 33 | 100 | 1,024 |
| 100 | 100 | 664 | 10,000 | $\approx 10^{30}$ |
| 1,000 | 1,000 | 9,966 | 1,000,000 | $\approx 10^{301}$ |

```python
import time

def linear_search(arr, target):
    """O(n) — checks each element."""
    for i, val in enumerate(arr):
        if val == target:
            return i
    return -1

def binary_search(arr, target):
    """O(log n) — halves the search space."""
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

def main():
    arr = list(range(1_000_000))
    target = 999_999

    start = time.time()
    linear_search(arr, target)
    t1 = time.time() - start

    start = time.time()
    binary_search(arr, target)
    t2 = time.time() - start

    print(f"Linear search: {t1:.6f} sec")
    print(f"Binary search: {t2:.6f} sec")
    print(f"Speedup: {t1/t2:.0f}x")

if __name__ == "__main__":
    main()
```

# Reference

[Introduction to Algorithms (CLRS), Chapter 1](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
