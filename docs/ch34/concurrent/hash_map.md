# Concurrent Hash Map

Concurrent hash maps use fine-grained locking or lock-free techniques for thread safety.

$$h(k) = k \bmod m$$

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

[Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)
