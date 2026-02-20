# Applications

**Applications** is an important concept in algorithm design and analysis.

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0]*n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b: return False
        if self.rank[a] < self.rank[b]: a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]: self.rank[a] += 1
        return True

uf = UnionFind(5)
print(uf.union(0,1), uf.union(2,3), uf.union(1,3))
print(f"0 and 3 connected: {uf.find(0)==uf.find(3)}")
```

**Output:**
```
True True True
0 and 3 connected: True
```


# Reference

[Introduction to Algorithms (CLRS), Chapter 23](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
