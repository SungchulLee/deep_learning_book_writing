# Randomized Quicksort

**Randomized Quicksort** is an important concept in algorithm design and analysis.

$$T(n) = O(n \log n) \text{ average}, \quad O(n^2) \text{ worst case}$$

```python
def quicksort(arr):
    if len(arr) <= 1: return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)

print(quicksort([3,6,8,10,1,2,1]))
```

**Output:**
```
[1, 1, 2, 3, 6, 8, 10]
```


# Reference

[Introduction to Algorithms (CLRS), Chapters 2, 7](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
