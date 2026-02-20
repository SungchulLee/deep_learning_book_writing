# Polygon Area

Polygon area via the shoelace formula: $A = \frac{1}{2}|\sum(x_i y_{i+1} - x_{i+1} y_i)|$

$$T(V, E) = O((V + E) \log V) \text{ with binary heap}$$

```python
import heapq
def prim(graph, start=0):
    visited, mst, pq = set(), [], [(0, start, -1)]
    while pq:
        w, u, parent = heapq.heappop(pq)
        if u in visited: continue
        visited.add(u)
        if parent >= 0: mst.append((parent, u, w))
        for v, wt in graph[u]:
            if v not in visited: heapq.heappush(pq, (wt, v, u))
    return mst

G = {0:[(1,4),(2,1)], 1:[(0,4),(2,3),(3,2)], 2:[(0,1),(1,3),(3,5)], 3:[(1,2),(2,5)]}
print(prim(G))
```

**Output:**
```
[(0, 2, 1), (2, 1, 3), (1, 3, 2)]
```


# Reference

[Computational Geometry (de Berg et al.)](https://www.springer.com/gp/book/9783540779735)
