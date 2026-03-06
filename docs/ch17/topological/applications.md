# Applications

**Applications** is an important concept in algorithm design and analysis.

```python
from collections import deque
def topo_sort(graph, n):
    indeg = [0]*n
    for u in graph:
        for v in graph[u]: indeg[v] += 1
    q = deque(v for v in range(n) if indeg[v] == 0)
    order = []
    while q:
        u = q.popleft(); order.append(u)
        for v in graph[u]:
            indeg[v] -= 1
            if indeg[v] == 0: q.append(v)
    return order

G = {0:[1,2], 1:[3], 2:[3], 3:[4], 4:[]}
print(f"Topological order: {topo_sort(G, 5)}")
```

**Output:**
```
Topological order: [0, 1, 2, 3, 4]
```

# Reference

[Introduction to Algorithms (CLRS), Chapters 22, 26](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
