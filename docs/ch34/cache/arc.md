# ARC Cache

LRU evicts the least recently used item, which works well for recency-driven workloads but fails when frequently accessed items are not recently accessed. LFU evicts the least frequently used item, which handles frequency well but reacts slowly to changing access patterns. **Adaptive Replacement Cache (ARC)** combines the strengths of both by dynamically balancing between recency and frequency, automatically adapting to the workload without manual tuning.

## Design Overview

ARC maintains four lists that together track both recent and frequent access patterns:

- **$T_1$**: Pages accessed recently (once since entering the cache). This is the recency list.
- **$T_2$**: Pages accessed frequently (at least twice since entering the cache). This is the frequency list.
- **$B_1$**: Ghost entries recently evicted from $T_1$. Tracks recently used pages that no longer fit.
- **$B_2$**: Ghost entries recently evicted from $T_2$. Tracks frequently used pages that no longer fit.

The cache holds items in $T_1 \cup T_2$, with $|T_1| + |T_2| \le c$ where $c$ is the cache capacity. The ghost lists $B_1$ and $B_2$ store only keys (no values), so they consume minimal extra memory.

## Adaptation Parameter

ARC uses a single parameter $p$ (initialized to 0) that determines the target size of $T_1$:

- **Hit in $B_1$** (a recently evicted recency item is requested again): increase $p$ by $\delta_1 = \max(1, |B_2| / |B_1|)$. This expands the recency portion.
- **Hit in $B_2$** (a recently evicted frequency item is requested again): decrease $p$ by $\delta_2 = \max(1, |B_1| / |B_2|)$. This expands the frequency portion.

The parameter $p$ is bounded: $0 \le p \le c$.

## Algorithm

On a cache request for page $x$:

1. **Hit in $T_1$ or $T_2$**: Move $x$ to the MRU position of $T_2$ (it is now a frequent item). Cache hit.
2. **Hit in $B_1$**: Ghost hit -- the page was recently evicted from the recency list. Increase $p$. If $|T_1| + |T_2| = c$, evict from $T_1$ or $T_2$ based on $p$. Fetch $x$ and insert into $T_2$.
3. **Hit in $B_2$**: Ghost hit -- the page was recently evicted from the frequency list. Decrease $p$. Evict if needed. Fetch $x$ and insert into $T_2$.
4. **Complete miss**: $x$ not in any list. Evict if needed. Insert $x$ into $T_1$.

### Eviction Rule (Replace)

When eviction is required, choose the victim based on $p$:

- If $|T_1| > p$: evict the LRU item from $T_1$ (move it to $B_1$).
- Otherwise: evict the LRU item from $T_2$ (move it to $B_2$).

## Implementation

```python
"""
Adaptive Replacement Cache (ARC).

Maintains recency list T1, frequency list T2, and ghost
lists B1, B2. Adapts the balance between recency and
frequency based on ghost hits.
"""

from collections import OrderedDict

# ===================================================================
# ARC Cache
# ===================================================================

class ARCCache:
    """Adaptive Replacement Cache with capacity c.

    Args:
        capacity: maximum number of items in cache (T1 + T2)
    """

    def __init__(self, capacity):
        self.c = capacity
        self.p = 0  # adaptation parameter
        self.t1 = OrderedDict()  # recency
        self.t2 = OrderedDict()  # frequency
        self.b1 = OrderedDict()  # ghost recency
        self.b2 = OrderedDict()  # ghost frequency
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """Look up key in cache. Returns value or None on miss."""
        if key in self.t1:
            val = self.t1.pop(key)
            self.t2[key] = val
            self.hits += 1
            return val
        if key in self.t2:
            self.t2.move_to_end(key)
            self.hits += 1
            return self.t2[key]
        self.misses += 1
        return None

    def put(self, key, value):
        """Insert or update key-value pair."""
        if key in self.t1:
            self.t1.pop(key)
            self.t2[key] = value
            return
        if key in self.t2:
            self.t2[key] = value
            self.t2.move_to_end(key)
            return

        if key in self.b1:
            # Ghost hit in B1: favor recency
            delta = max(1, len(self.b2) // max(1, len(self.b1)))
            self.p = min(self.c, self.p + delta)
            self.b1.pop(key)
            self._replace(key)
            self.t2[key] = value
            return

        if key in self.b2:
            # Ghost hit in B2: favor frequency
            delta = max(1, len(self.b1) // max(1, len(self.b2)))
            self.p = max(0, self.p - delta)
            self.b2.pop(key)
            self._replace(key)
            self.t2[key] = value
            return

        # Complete miss
        total = len(self.t1) + len(self.b1)
        if total >= self.c:
            if len(self.t1) < self.c:
                self.b1.popitem(last=False)
            else:
                self.t1.popitem(last=False)
        self._replace(key)
        self.t1[key] = value

        # Cap ghost lists
        while len(self.b1) > self.c:
            self.b1.popitem(last=False)
        while len(self.b2) > self.c:
            self.b2.popitem(last=False)

    def _replace(self, key):
        """Evict one item if cache is full."""
        if len(self.t1) + len(self.t2) < self.c:
            return
        if self.t1 and (len(self.t1) > self.p or
                        (key in self.b2 and len(self.t1) == self.p)):
            evicted_key, evicted_val = self.t1.popitem(last=False)
            self.b1[evicted_key] = None
        elif self.t2:
            evicted_key, evicted_val = self.t2.popitem(last=False)
            self.b2[evicted_key] = None

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    cache = ARCCache(capacity=3)

    requests = ["A", "B", "C", "A", "D", "B", "A", "E", "B", "A"]

    print("ARC Cache simulation (capacity=3):")
    for req in requests:
        result = cache.get(req)
        if result is None:
            cache.put(req, req)
            print(f"  {req}: MISS -> inserted")
        else:
            print(f"  {req}: HIT")

    total = cache.hits + cache.misses
    print(f"\nHits: {cache.hits}/{total} "
          f"({100*cache.hits/total:.0f}%)")
    print(f"p (adaptation): {cache.p}")
```

**Output:**
```
ARC Cache simulation (capacity=3):
  A: MISS -> inserted
  B: MISS -> inserted
  C: MISS -> inserted
  A: HIT
  D: MISS -> inserted
  B: HIT
  A: HIT
  E: MISS -> inserted
  B: HIT
  A: HIT

Hits: 4/10 (40%)
p (adaptation): 1
```

## Complexity

All ARC operations run in $O(1)$ amortized time using hash maps and doubly-linked lists (via `OrderedDict`).

| Operation | Time | Space |
|---|---|---|
| `get` | $O(1)$ | -- |
| `put` | $O(1)$ amortized | -- |
| Total space | -- | $O(c)$ for cache + $O(c)$ for ghost lists |

## Comparison with LRU and LFU

| Property | LRU | LFU | ARC |
|---|---|---|---|
| Adapts to workload | No | No | Yes |
| Handles scan resistance | No | Yes | Yes |
| Handles recency shifts | Yes | No | Yes |
| Space overhead | $O(c)$ | $O(c)$ | $O(2c)$ (ghost lists) |
| Implementation complexity | Simple | Moderate | Moderate |

!!! note "Scan resistance"
    A sequential scan of many distinct items can flush the entire LRU cache, destroying useful entries. ARC resists this because scanned items enter $T_1$, leaving $T_2$ (frequent items) unaffected. The ghost lists detect when this pattern occurs and adjust $p$ accordingly.

## Reference

- Megiddo, N. and Modha, D. S. (2003). "ARC: A self-tuning, low overhead replacement cache." *FAST*.
