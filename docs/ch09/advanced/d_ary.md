# d-ary Heaps

$d$-ary heaps generalize binary heaps; each node has $d$ children. Decreasing $d$ speeds up decrease-key.

```python
import heapq

arr = [5,3,8,1,9,2]
heapq.heapify(arr)
result = [heapq.heappop(arr) for _ in range(len(arr))]
print(f"Heap sort: {result}")
```

**Output:**
```
Heap sort: [1, 2, 3, 5, 8, 9]
```


# Reference

[Introduction to Algorithms (CLRS), Chapter 6](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
