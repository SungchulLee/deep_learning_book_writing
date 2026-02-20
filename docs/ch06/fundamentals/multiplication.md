# Multiplication Method

**Multiplication Method** is an important concept in algorithm design and analysis.

```python
class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]
    def _hash(self, key): return hash(key) % self.size
    def put(self, key, val):
        h = self._hash(key)
        for i, (k,v) in enumerate(self.table[h]):
            if k == key: self.table[h][i] = (key, val); return
        self.table[h].append((key, val))
    def get(self, key):
        h = self._hash(key)
        for k,v in self.table[h]:
            if k == key: return v
        return None

ht = HashTable()
ht.put("a", 1); ht.put("b", 2)
print(ht.get("a"), ht.get("b"), ht.get("c"))
```

**Output:**
```
1 2 None
```


# Reference

[Introduction to Algorithms (CLRS), Chapter 11](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
