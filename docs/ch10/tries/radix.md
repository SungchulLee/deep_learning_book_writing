# Radix Trees

Radix trees (compact tries) store edges as strings rather than single characters.

```python
def counting_sort_by_digit(arr, exp):
    n = len(arr); output = [0]*n; count = [0]*10
    for x in arr: count[(x//exp)%10] += 1
    for i in range(1,10): count[i] += count[i-1]
    for x in reversed(arr):
        idx = (x//exp)%10; output[count[idx]-1] = x; count[idx] -= 1
    return output

def radix_sort(arr):
    if not arr: return arr
    mx = max(arr); exp = 1
    while mx // exp > 0:
        arr = counting_sort_by_digit(arr, exp); exp *= 10
    return arr

print(radix_sort([170,45,75,90,802,24,2,66]))
```

**Output:**
```
[2, 24, 45, 66, 75, 90, 170, 802]
```


# Reference

[Introduction to Algorithms (CLRS), Chapter 14](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
