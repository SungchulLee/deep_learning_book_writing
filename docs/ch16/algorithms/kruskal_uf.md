# Kruskal with Union-Find

**Kruskal with Union-Find** is an important concept in algorithm design and analysis.

$$T(E, V) = O(E \log E)$$

```python
class UF:
    def __init__(self, n): self.p = list(range(n)); self.r = [0]*n
    def find(self, x):
        while self.p[x] != x: self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b: return False
        if self.r[a] < self.r[b]: a, b = b, a
        self.p[b] = a
        if self.r[a] == self.r[b]: self.r[a] += 1
        return True

def kruskal(n, edges):
    edges.sort(key=lambda e: e[2])
    uf, mst = UF(n), []
    for u,v,w in edges:
        if uf.union(u,v): mst.append((u,v,w))
    return mst

edges = [(0,1,4),(0,2,1),(1,2,3),(1,3,2),(2,3,5)]
print(kruskal(4, edges))
```

**Output:**
```
[(0, 2, 1), (1, 3, 2), (1, 2, 3)]
```


# Reference

[Introduction to Algorithms (CLRS), Chapter 23](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
