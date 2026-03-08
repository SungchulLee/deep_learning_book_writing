# Bellman-Ford in RIP

RIP uses Bellman-Ford (distance vector) for routing with hop-count metric.

```python
def bellman_ford(n, edges, src):
    dist = [float("inf")]*n
    dist[src] = 0
    for _ in range(n-1):
        for u,v,w in edges:
            if dist[u]+w < dist[v]: dist[v] = dist[u]+w
    return dist

edges = [(0,1,4),(0,2,1),(2,1,2),(1,3,1),(2,3,5)]
print(bellman_ford(4, edges, 0))
```

**Output:**
```
[0, 3, 1, 4]
```

# Reference

[Designing Data-Intensive Applications (Kleppmann)](https://dataintensive.net/)
