# Counting Operations

**Counting Operations** is an important concept in algorithm design and analysis.

```python
def counting_sort(arr, max_val):
    count = [0]*(max_val+1)
    for x in arr: count[x] += 1
    result = []
    for i, c in enumerate(count):
        result.extend([i]*c)
    return result

print(counting_sort([4,2,2,8,3,3,1], 8))
```

**Output:**
```
[1, 2, 2, 3, 3, 4, 8]
```


# Reference

[Introduction to Algorithms (CLRS), Chapters 2-4](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
