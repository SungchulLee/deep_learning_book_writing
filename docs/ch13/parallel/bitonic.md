# Bitonic Sort

Bitonic sort is a parallel comparison-based sorting algorithm suitable for hardware implementation.

```python
class FenwickTree:
    def __init__(self, n): self.n = n; self.tree = [0]*(n+1)
    def update(self, i, delta):
        while i <= self.n: self.tree[i] += delta; i += i & (-i)
    def query(self, i):
        s = 0
        while i > 0: s += self.tree[i]; i -= i & (-i)
        return s
    def range_query(self, l, r): return self.query(r) - self.query(l-1)

ft = FenwickTree(5)
for i,v in enumerate([1,3,5,7,9],1): ft.update(i,v)
print(f"Sum [1,3]: {ft.range_query(1,3)}")
print(f"Sum [2,5]: {ft.range_query(2,5)}")
```

**Output:**
```
Sum [1,3]: 9
Sum [2,5]: 24
```

# Reference

[Introduction to Algorithms (CLRS), Chapters 8-9](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
