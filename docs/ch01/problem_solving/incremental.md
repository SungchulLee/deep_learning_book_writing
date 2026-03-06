# Incremental Improvement

The **incremental** approach builds a solution one element at a time, maintaining correctness at each step.

$$

\text{Solution}_i = \text{Extend}(\text{Solution}_{i-1}, \text{element}_i)

$$

## Insertion Sort as Incremental Algorithm

At each step $i$, we insert $A[i]$ into the already-sorted subarray $A[0..i-1]$.

```python
def insertion_sort(arr):
    """Incremental approach: insert each element into sorted prefix."""
    arr = arr.copy()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        # Shift elements greater than key to the right
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
        print(f"  Step {i}: {arr}")
    return arr

def main():
    arr = [5, 2, 4, 6, 1, 3]
    print(f"Input: {arr}")
    result = insertion_sort(arr)
    print(f"Sorted: {result}")

if __name__ == "__main__":
    main()
```

**Output:**
```
Input: [5, 2, 4, 6, 1, 3]
  Step 1: [2, 5, 4, 6, 1, 3]
  Step 2: [2, 4, 5, 6, 1, 3]
  Step 3: [2, 4, 5, 6, 1, 3]
  Step 4: [1, 2, 4, 5, 6, 3]
  Step 5: [1, 2, 3, 4, 5, 6]
Sorted: [1, 2, 3, 4, 5, 6]
```

# Reference

[Introduction to Algorithms (CLRS), Section 2.1](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
