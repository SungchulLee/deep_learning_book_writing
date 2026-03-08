# Leftist Heaps

Leftist heaps maintain the property that the left subtree is always at least as heavy as the right.

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
