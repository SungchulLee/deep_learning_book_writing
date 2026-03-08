# BFS Preview

BFS uses a queue to explore nodes level by level, useful for shortest paths in unweighted graphs.

$$T(V, E) = O(V + E)$$

```python
from collections import deque

q = deque()
for x in [1,2,3,4]: q.append(x)
while q:
    print(q.popleft(), end=" ")
print()
```

**Output:**
```
1 2 3 4
```

# Reference

[Introduction to Algorithms (CLRS), Chapter 10](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
