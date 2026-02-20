# Merge Procedure

The merge procedure combines two sorted arrays into one sorted array in $O(n)$ time.

```python
def merge_sort(arr):
    if len(arr) <= 1: return arr
    mid = len(arr)//2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]: result.append(left[i]); i += 1
        else: result.append(right[j]); j += 1
    return result + left[i:] + right[j:]

print(merge_sort([38,27,43,3,9,82,10]))
```

**Output:**
```
[3, 9, 10, 27, 38, 43, 82]
```


# Reference

[Introduction to Algorithms (CLRS), Chapters 2, 7](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
