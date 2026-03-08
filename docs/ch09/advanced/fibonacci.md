# Fibonacci Heaps

Fibonacci heaps provide $O(1)$ amortized insert and decrease-key, optimal for Dijkstra and Prim.

$$F(n) = F(n-1) + F(n-2), \quad F(0)=0, F(1)=1$$

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
