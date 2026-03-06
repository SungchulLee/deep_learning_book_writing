# Correctness

An algorithm is **correct** if, for every input instance, it halts with the correct output.

$$

\forall \text{ valid input } x: \text{Algorithm}(x) = \text{Expected}(x)

$$

## Loop Invariants

A common technique for proving correctness is using **loop invariants** — properties that hold:

1. **Initialization**: True before the first iteration
2. **Maintenance**: If true before an iteration, remains true after
3. **Termination**: When the loop ends, the invariant gives a useful property

```python
def insertion_sort(arr):
    """
    Correctness proved via loop invariant:
    At the start of each iteration of the outer loop,
    arr[0..i-1] contains the same elements as the original
    arr[0..i-1] but in sorted order.
    """
    arr = arr.copy()
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def main():
    arr = [5, 2, 4, 6, 1, 3]
    print(f"Input:  {arr}")
    print(f"Sorted: {insertion_sort(arr)}")

if __name__ == "__main__":
    main()
```

**Output:**
```
Input:  [5, 2, 4, 6, 1, 3]
Sorted: [1, 2, 3, 4, 5, 6]
```

# Reference

[Introduction to Algorithms (CLRS), Section 2.1](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
