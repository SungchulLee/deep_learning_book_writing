# LRU Cache

Cache performance depends on spatial and temporal locality; arrays have better cache behavior than linked lists.

```python
from collections import OrderedDict
class LRUCache:
    def __init__(self, cap):
        self.cap = cap; self.cache = OrderedDict()
    def get(self, key):
        if key not in self.cache: return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, val):
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key] = val
        if len(self.cache) > self.cap: self.cache.popitem(last=False)

c = LRUCache(2)
c.put(1,1); c.put(2,2)
print(c.get(1))
c.put(3,3)
print(c.get(2))
print(c.get(3))
```

**Output:**
```
1
-1
3
```


# Reference

[Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)
