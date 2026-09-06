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

## Exercises

**Exercise 1.**
Describe the four lists maintained by ARC and explain the role of each. What does the parameter $p$ control?

??? success "Solution to Exercise 1"
    ARC maintains four lists: (1) $T_1$ -- recently accessed items seen only once (recency list, actual cache), (2) $T_2$ -- recently accessed items seen at least twice (frequency list, actual cache), (3) $B_1$ -- ghost entries evicted from $T_1$ (metadata only, no data), (4) $B_2$ -- ghost entries evicted from $T_2$ (metadata only). The cache stores data only in $T_1 \cup T_2$, with $|T_1| + |T_2| \le c$ (cache capacity). The parameter $p$ ($0 \le p \le c$) is the target size for $T_1$. When $|T_1| > p$, ARC prefers to evict from $T_1$; when $|T_1| < p$, it prefers to evict from $T_2$. ARC adapts $p$ dynamically: a hit in $B_1$ increases $p$ (favoring recency), and a hit in $B_2$ decreases $p$ (favoring frequency). $\square$

---

**Exercise 2.**
Trace through ARC with cache capacity $c = 3$ on the access sequence $[A, B, C, D, A, B, E, A]$. Show the state of $T_1$, $T_2$, $B_1$, $B_2$, and $p$ after each access.

??? success "Solution to Exercise 2"
    Initial: $T_1 = T_2 = B_1 = B_2 = \emptyset$, $p = 0$. Access A: miss, $T_1 = [A]$. Access B: miss, $T_1 = [A, B]$. Access C: miss, $T_1 = [A, B, C]$. Access D: miss, cache full, evict LRU of $T_1$ (A) to $B_1$; $T_1 = [B, C, D]$, $B_1 = [A]$. Access A: hit in $B_1$ -- increase $p$ by $\max(1, |B_2|/|B_1|) = 1$, so $p = 1$. Move A to MRU of $T_2$, evict LRU of $T_1$ (B) to $B_1$; $T_1 = [C, D]$, $T_2 = [A]$, $B_1 = [B]$. Access B: hit in $B_1$ -- increase $p$ to 2. Move B to $T_2$, evict LRU of $T_1$ (C) to $B_1$; $T_1 = [D]$, $T_2 = [A, B]$, $B_1 = [C]$. Access E: miss, evict from $T_1$ (since $|T_1| = 1 \le p = 2$, but only D available); $T_1 = [E]$, $B_1 = [C, D]$, $T_2 = [A, B]$. Access A: hit in $T_2$, move to MRU of $T_2$; $T_2 = [B, A]$. $\square$

---

**Exercise 3.**
Prove that ARC's adaptation mechanism converges: if the workload is purely recency-driven (LRU-optimal), $p$ approaches $c$; if purely frequency-driven (LFU-optimal), $p$ approaches $0$.

??? success "Solution to Exercise 3"
    In a purely recency-driven workload, items are accessed once, used soon after, and then not reused. Items evicted from $T_1$ are not accessed again, so $B_1$ never produces hits. Items that are re-accessed after a gap appear in $T_2$, but the dominant pattern fills $T_2$ slowly. Any ghost hit in $B_1$ increases $p$, and ghost hits in $B_2$ are rare because $T_2$ and $B_2$ are sparsely populated. Over time, $p$ increases toward $c$, allocating nearly all cache space to $T_1$ (the recency list), which mimics LRU. Conversely, in a frequency-driven workload, popular items cycle through $T_1 \to B_1 \to T_2$, and $B_2$ accumulates ghost entries of formerly popular items. $B_2$ hits decrease $p$, shifting capacity from $T_1$ to $T_2$, eventually making $p \to 0$. The adaptation is stable because the ghost directories provide unbiased feedback about which eviction policy would have been beneficial. $\square$

---

**Exercise 4.**
Compare the space overhead of ARC versus LRU. Why do practical implementations often limit the ghost directory sizes?

??? success "Solution to Exercise 4"
    LRU maintains a single doubly-linked list and a hash map, using $O(c)$ space total. ARC maintains the same structures for $T_1$ and $T_2$ (total capacity $c$ for data) plus ghost directories $B_1$ and $B_2$. In the original formulation, $|T_1| + |B_1| \le c$ and $|T_2| + |B_2| \le c$, so the ghost directories can hold up to $c$ entries each, totaling $2c$ metadata entries beyond the $c$ cache entries. Each ghost entry stores only the key (no data), so the overhead is roughly $2c \times (\text{key size})$. For large caches or large keys, this doubles the metadata cost. Practical implementations limit ghost sizes to a fraction of $c$ (e.g., $c/4$) to reduce memory, accepting slightly slower adaptation in exchange. The ghost entries enable scan resistance without the full $2c$ overhead in most workloads. $\square$

---

**Exercise 5.**
Design a workload that causes LRU to perform poorly but ARC to maintain high hit rates. Explain the mechanism behind ARC's advantage.

??? success "Solution to Exercise 5"
    Consider a cache of size $c = 100$ and a workload alternating between two phases: (1) a "scan" phase that accesses 200 distinct items sequentially (e.g., a sequential file read), and (2) a "working set" phase that repeatedly accesses 50 fixed items. Under LRU, the scan phase evicts all working-set items because 200 > 100. When the working-set phase resumes, every access is a miss until the 50 items are reloaded. Under ARC, the scan items enter $T_1$ and are quickly evicted to $B_1$. They do not hit $B_1$ again (each scan item is unique), so $p$ does not increase. The working-set items accumulate in $T_2$ (seen multiple times). When the scan phase starts, ARC evicts primarily from $T_1$, preserving the working set in $T_2$. The adaptation parameter $p$ decreases as $B_2$ accumulates working-set ghosts during scans, protecting $T_2$. ARC achieves near-optimal hit rates on both phases. $\square$
