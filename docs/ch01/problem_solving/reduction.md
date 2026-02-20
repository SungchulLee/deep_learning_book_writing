# Reduction

**Reduction** transforms one problem into another that we already know how to solve.

$$
\text{Problem A} \leq_p \text{Problem B}
$$

If we can reduce $A$ to $B$ in polynomial time, then solving $B$ also solves $A$.

## Example: Median via Sorting

Finding the median can be **reduced** to sorting:

```python
def find_median_via_sort(arr):
    """Reduce median-finding to sorting."""
    sorted_arr = sorted(arr)  # O(n log n)
    n = len(sorted_arr)
    if n % 2 == 1:
        return sorted_arr[n // 2]
    else:
        return (sorted_arr[n // 2 - 1] + sorted_arr[n // 2]) / 2


def main():
    tests = [
        [3, 1, 4, 1, 5],
        [2, 4, 6, 8],
        [7],
    ]
    for arr in tests:
        print(f"Array: {arr}, Median: {find_median_via_sort(arr)}")


if __name__ == "__main__":
    main()
```

**Output:**
```
Array: [3, 1, 4, 1, 5], Median: 3
Array: [2, 4, 6, 8], Median: 5.0
Array: [7], Median: 7
```


# Reference

[Algorithm Design (Kleinberg & Tardos), Chapter 8](https://www.cs.princeton.edu/~wayne/kleinberg-tardos/)
