# 0-1 BFS

0-1 BFS uses a deque for graphs with edge weights 0 and 1, running in $O(V+E)$.

$$T(V, E) = O(V + E)$$

```python
from collections import deque
def bfs(graph, start):
    visited, queue = {start}, deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for nb in graph[node]:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
    return order

graph = {0:[1,2], 1:[3], 2:[3,4], 3:[], 4:[]}
print(f"BFS from 0: {bfs(graph, 0)}")
```

**Output:**
```
BFS from 0: [0, 1, 2, 3, 4]
```

# Reference

[Introduction to Algorithms (CLRS), Chapters 24-25](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
