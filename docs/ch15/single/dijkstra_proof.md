# Dijkstra Correctness

**Dijkstra Correctness** is an important concept in algorithm design and analysis.

$$T(V, E) = O((V + E) \log V) \text{ with binary heap}$$

```python
import heapq
def dijkstra(graph, src):
    dist = {v: float("inf") for v in graph}
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(pq, (dist[v], v))
    return dist

G = {0:[(1,4),(2,1)], 1:[(3,1)], 2:[(1,2),(3,5)], 3:[]}
print(dijkstra(G, 0))
```

**Output:**
```
{0: 0, 1: 3, 2: 1, 3: 4}
```


# Reference

[Introduction to Algorithms (CLRS), Chapters 24-25](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
