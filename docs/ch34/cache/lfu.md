# LFU Cache

While LRU evicts the item that has not been accessed for the longest time, some workloads require a policy that favors items accessed many times, even if not recently. **Least Frequently Used (LFU)** evicts the item with the lowest access count. When multiple items share the same minimum frequency, LFU breaks ties by evicting the least recently used among them. With careful data structure design, all LFU operations run in $O(1)$ time.

## Design

An $O(1)$ LFU cache uses three interconnected data structures:

1. **Key-value map**: Maps each key to its value, frequency, and position.
2. **Frequency map**: Maps each frequency count $f$ to an ordered set (doubly-linked list) of all keys with frequency $f$.
3. **Minimum frequency tracker**: A single integer $f_{\min}$ recording the current smallest frequency among cached items.

### Operations

**Get(key)**:

1. Look up the key in the key-value map. If absent, return miss.
2. Increment the key's frequency from $f$ to $f + 1$.
3. Move the key from frequency bucket $f$ to bucket $f + 1$.
4. If bucket $f$ is now empty and $f = f_{\min}$, increment $f_{\min}$.
5. Return the value.

**Put(key, value)**:

1. If the key exists, update its value and perform the same frequency increment as Get.
2. If the cache is full, evict the LRU item from the $f_{\min}$ bucket.
3. Insert the new key with frequency 1 into bucket 1. Set $f_{\min} = 1$.

## Implementation

```python
"""
O(1) LFU (Least Frequently Used) Cache.

Uses a hash map for key lookup and a frequency-to-keys map
(using OrderedDict for LRU ordering within each frequency)
to achieve O(1) get and put operations.
"""

from collections import OrderedDict, defaultdict

# ===================================================================
# LFU Cache
# ===================================================================

class LFUCache:
    """Least Frequently Used cache with O(1) operations.

    Args:
        capacity: maximum number of items
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.key_to_val = {}
        self.key_to_freq = {}
        self.freq_to_keys = defaultdict(OrderedDict)
        self.min_freq = 0

    def get(self, key):
        """Get value by key. Returns -1 on miss."""
        if key not in self.key_to_val:
            return -1
        self._increment_freq(key)
        return self.key_to_val[key]

    def put(self, key, value):
        """Insert or update key-value pair."""
        if self.capacity <= 0:
            return

        if key in self.key_to_val:
            self.key_to_val[key] = value
            self._increment_freq(key)
            return

        # Evict if at capacity
        if len(self.key_to_val) >= self.capacity:
            # Evict LRU item from min_freq bucket
            evict_key, _ = self.freq_to_keys[self.min_freq].popitem(
                last=False)
            del self.key_to_val[evict_key]
            del self.key_to_freq[evict_key]

        # Insert new key
        self.key_to_val[key] = value
        self.key_to_freq[key] = 1
        self.freq_to_keys[1][key] = None
        self.min_freq = 1

    def _increment_freq(self, key):
        """Move key from frequency f to f+1."""
        freq = self.key_to_freq[key]
        self.key_to_freq[key] = freq + 1

        # Remove from current frequency bucket
        del self.freq_to_keys[freq][key]
        if not self.freq_to_keys[freq]:
            del self.freq_to_keys[freq]
            if self.min_freq == freq:
                self.min_freq += 1

        # Add to next frequency bucket
        self.freq_to_keys[freq + 1][key] = None

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    cache = LFUCache(3)

    # Insert items
    cache.put("A", 1)
    cache.put("B", 2)
    cache.put("C", 3)

    # Access A and B to increase their frequency
    cache.get("A")  # freq(A)=2
    cache.get("A")  # freq(A)=3
    cache.get("B")  # freq(B)=2

    # Insert D -- evicts C (freq=1, the least frequent)
    cache.put("D", 4)

    print("LFU Cache (capacity=3):")
    print(f"  get(A) = {cache.get('A')}")  # 1 (freq=4)
    print(f"  get(B) = {cache.get('B')}")  # 2 (freq=3)
    print(f"  get(C) = {cache.get('C')}")  # -1 (evicted)
    print(f"  get(D) = {cache.get('D')}")  # 4 (freq=2)

    # Insert E -- evicts D (freq=2, least frequent now)
    cache.put("E", 5)
    print(f"  get(D) = {cache.get('D')}")  # -1 (evicted)
    print(f"  get(E) = {cache.get('E')}")  # 5
```

**Output:**
```
LFU Cache (capacity=3):
  get(A) = 1
  get(B) = 2
  get(C) = -1
  get(D) = 4
  get(D) = -1
  get(E) = 5
```

## Complexity

| Operation | Time | Space |
|---|---|---|
| `get` | $O(1)$ | -- |
| `put` | $O(1)$ amortized | -- |
| Total space | -- | $O(c)$ |

The $O(1)$ bound relies on hash map lookups and doubly-linked list operations within each frequency bucket.

## Comparison with LRU

| Property | LRU | LFU |
|---|---|---|
| Eviction criterion | Least recently used | Least frequently used |
| Favors | Temporal locality | Frequency of access |
| Weakness | Scan pollution | Stale high-frequency items |
| Complexity | $O(1)$ | $O(1)$ with careful design |

!!! warning "Cache pollution in LFU"
    Items that were frequently accessed in the past but are no longer needed accumulate high frequency counts and resist eviction. This "cache pollution" problem motivates aging mechanisms or hybrid policies like [ARC](arc.md).

## Reference

- Shah, K., Mitra, A., and Matani, D. (2010). "An O(1) algorithm for implementing the LFU cache eviction scheme." *Technical Report*.
