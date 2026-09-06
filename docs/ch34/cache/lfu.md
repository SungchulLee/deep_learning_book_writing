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

## Exercises

**Exercise 1.**
Describe the data structures needed to implement LFU with $O(1)$ get and put operations. What role does each structure play?

??? success "Solution to Exercise 1"
    Three structures are required: (1) A **hash map** from key to node, enabling $O(1)$ lookup of any cached item. (2) A **hash map** from frequency to a doubly-linked list (frequency bucket), where each bucket holds all items with that access count, in insertion order. (3) A **min-frequency tracker** storing the current minimum frequency in the cache. On `get(key)`: look up the node, remove it from its current frequency bucket, increment its frequency, insert into the next frequency bucket, update min-frequency if the old bucket is now empty. On `put(key, value)` when full: evict the LRU item from the min-frequency bucket (the tail of that doubly-linked list), reset min-frequency to 1 for the new item. All operations are pointer manipulations and hash lookups, each $O(1)$. $\square$

---

**Exercise 2.**
Trace through an LFU cache of capacity 3 on the access sequence: put(A,1), put(B,2), put(C,3), get(B), put(D,4). Which item is evicted and why?

??? success "Solution to Exercise 2"
    After put(A,1): cache = {A:1}, frequencies: A=1, min_freq=1. After put(B,2): cache = {A:1, B:2}, frequencies: A=1, B=1, min_freq=1. After put(C,3): cache = {A:1, B:2, C:3}, frequencies: A=1, B=1, C=1, min_freq=1. After get(B): B's frequency increases to 2. Cache = {A:1, B:2, C:3}, frequencies: A=1, C=1, B=2, min_freq=1. After put(D,4): cache is full, evict the LRU item at min_freq=1. The frequency-1 bucket contains [A, C] in insertion order. The LRU (oldest) is A. Evict A. Insert D with frequency 1. Final cache = {B:2, C:3, D:4}, frequencies: C=1, D=1, B=2, min_freq=1. $\square$

---

**Exercise 3.**
Explain the "frequency pollution" problem in LFU caches and propose a mitigation strategy.

??? success "Solution to Exercise 3"
    Frequency pollution occurs when items that were heavily accessed in the past but are no longer relevant accumulate high frequency counts. These "stale popular" items are nearly impossible to evict because their counts exceed those of newly inserted items (which start at frequency 1). This blocks fresh, currently relevant items from remaining in the cache. Mitigation strategies: (1) **Aging/decay**: periodically halve all frequency counts (e.g., every $T$ accesses), allowing stale items to lose their advantage. (2) **Windowed LFU**: track frequency only within a sliding window of the last $W$ accesses rather than over the entire lifetime. (3) **Hybrid policies**: combine LFU with LRU (as in LRFU or ARC) to balance frequency and recency, preventing purely frequency-based decisions from dominating. $\square$

---

**Exercise 4.**
Prove that maintaining a min-frequency variable can be updated in $O(1)$ during both get and put operations without scanning all frequency buckets.

??? success "Solution to Exercise 4"
    The min-frequency variable `min_freq` changes only in two situations: (1) **put (new item)**: a new item always enters with frequency 1, so `min_freq = 1`. This is $O(1)$. (2) **get (existing item)**: the item moves from frequency $f$ to $f+1$. If $f =$ `min_freq` and the frequency-$f$ bucket becomes empty after removal, then `min_freq` must increase. Crucially, it increases to exactly $f + 1$ (not higher), because the item we just moved is now at frequency $f + 1$, guaranteeing that the $f+1$ bucket is non-empty. No scanning is needed: we simply check whether the old bucket is empty and, if so, increment `min_freq` by 1. If $f >$ `min_freq` or the old bucket is non-empty, `min_freq` is unchanged. Both cases are $O(1)$. $\square$

---

**Exercise 5.**
Compare LFU and LRU on a workload consisting of $n$ items with Zipfian frequency distribution (item $i$ accessed with probability proportional to $1/i$). For a cache of size $k \ll n$, which policy achieves a higher hit rate and why?

??? success "Solution to Exercise 5"
    Under a Zipfian distribution, a small number of items account for most accesses (item 1 is accessed most, item 2 roughly half as often, etc.). LFU is better suited here because it directly identifies and retains the most frequently accessed items. After a warm-up period, LFU's cache contains items 1 through $k$ (approximately), achieving the optimal hit rate of $\sum_{i=1}^{k} 1/i \,/\, \sum_{i=1}^{n} 1/i$. LRU also performs well on Zipfian workloads because popular items are accessed recently with high probability, but it is susceptible to occasional accesses of rare items displacing popular ones. For highly skewed Zipf parameters ($\alpha > 1$), LFU's advantage grows because the gap between popular and unpopular items widens. For $\alpha$ near 0 (near-uniform), both policies perform similarly. $\square$
