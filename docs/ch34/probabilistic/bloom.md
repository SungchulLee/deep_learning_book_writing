# Bloom Filters

Longest increasing subsequence in $O(n \log n)$ using patience sorting.

$$P(\text{false positive}) \approx \left(1 - e^{-kn/m}\right)^k$$

```python
import hashlib
class BloomFilter:
    def __init__(self, size=100, hashes=3):
        self.bits = [False]*size; self.size = size; self.k = hashes
    def _hashes(self, item):
        for i in range(self.k):
            h = int(hashlib.md5(f"{item}{i}".encode()).hexdigest(), 16)
            yield h % self.size
    def add(self, item):
        for h in self._hashes(item): self.bits[h] = True
    def check(self, item):
        return all(self.bits[h] for h in self._hashes(item))

bf = BloomFilter()
for w in ["apple","banana","cherry"]: bf.add(w)
for w in ["apple","banana","grape","melon"]:
    print(f"{w}: {bf.check(w)}")
```

**Output:**
```
apple: True
banana: True
grape: False
melon: False
```

# Reference

[Advanced Data Structures (Brass)](https://www.cambridge.org/core/books/advanced-data-structures/D56E2269D7CEE969A3B8105D3541F601)
