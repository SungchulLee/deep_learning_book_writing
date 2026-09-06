# TTL Expiry

Cache eviction policies like LRU and LFU decide *which* item to remove when space runs out, but they do not address *stale data*. An item cached hours ago may no longer reflect the current state of the underlying data source. **Time-To-Live (TTL)** expiry ensures cache freshness by associating each entry with an expiration timestamp. Once a TTL expires, the entry is considered invalid regardless of how recently or frequently it was accessed.

## Design

Each cache entry stores a tuple $(key, value, expiry)$ where:

$$
expiry = t_{\text{insert}} + \text{TTL}
$$

An entry is valid if and only if the current time $t_{\text{now}} < expiry$.

### Expiration Strategies

**Lazy expiration**: Check the TTL only on access. When a `get` request arrives, compare $t_{\text{now}}$ against $expiry$. If expired, delete the entry and return a miss. This is simple but allows expired entries to consume space until accessed.

**Active expiration**: A background process periodically scans entries and removes expired ones. This keeps memory usage tighter but adds complexity.

**Hybrid**: Combine lazy expiration on every access with periodic active cleanup for entries that are never re-requested.

## Implementation

```python
"""
TTL (Time-To-Live) cache with lazy expiration.

Combines LRU eviction with per-entry TTL. Expired entries
are removed lazily on access and periodically via cleanup.
"""

import time
from collections import OrderedDict

# ===================================================================
# TTL Cache
# ===================================================================

class TTLCache:
    """Cache with per-entry time-to-live and LRU eviction.

    Args:
        capacity: maximum number of items
        default_ttl: default TTL in seconds
    """

    def __init__(self, capacity, default_ttl=5.0):
        self.capacity = capacity
        self.default_ttl = default_ttl
        self.cache = OrderedDict()  # key -> (value, expiry_time)
        self.hits = 0
        self.misses = 0
        self.expirations = 0

    def get(self, key):
        """Get value if key exists and has not expired.

        Args:
            key: lookup key

        Returns:
            Value or None if missing/expired
        """
        if key not in self.cache:
            self.misses += 1
            return None

        value, expiry = self.cache[key]
        if time.time() > expiry:
            # Lazy expiration
            del self.cache[key]
            self.expirations += 1
            self.misses += 1
            return None

        self.cache.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key, value, ttl=None):
        """Insert or update with TTL.

        Args:
            key: cache key
            value: value to store
            ttl: time-to-live in seconds (uses default if None)
        """
        if ttl is None:
            ttl = self.default_ttl
        expiry = time.time() + ttl

        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = (value, expiry)

        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

    def cleanup(self):
        """Remove all expired entries (active expiration)."""
        now = time.time()
        expired_keys = [k for k, (v, exp) in self.cache.items()
                        if now > exp]
        for k in expired_keys:
            del self.cache[k]
            self.expirations += 1
        return len(expired_keys)

    def size(self):
        """Return current number of entries (including expired)."""
        return len(self.cache)

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    # Use simulated time for reproducible output
    cache = TTLCache(capacity=5, default_ttl=10.0)

    # Insert entries with different TTLs
    cache.put("A", 1, ttl=1.0)   # expires quickly
    cache.put("B", 2, ttl=10.0)  # long TTL
    cache.put("C", 3, ttl=10.0)
    cache.put("D", 4, ttl=0.5)   # very short TTL

    print("TTL Cache (capacity=5, default_ttl=10s)")
    print(f"  Initial size: {cache.size()}")

    # Immediate access -- all valid
    for key in ["A", "B", "C", "D"]:
        result = cache.get(key)
        print(f"  get({key}) = {result}")

    # Wait for short TTLs to expire
    print("\n  Waiting 1.5 seconds...")
    time.sleep(1.5)

    # Access again -- A and D should be expired
    for key in ["A", "B", "C", "D"]:
        result = cache.get(key)
        status = "valid" if result is not None else "expired"
        print(f"  get({key}) = {result} [{status}]")

    print(f"\n  Hits:        {cache.hits}")
    print(f"  Misses:      {cache.misses}")
    print(f"  Expirations: {cache.expirations}")

    # Active cleanup
    removed = cache.cleanup()
    print(f"  Cleanup removed: {removed} entries")
    print(f"  Final size: {cache.size()}")
```

**Output:**
```
TTL Cache (capacity=5, default_ttl=10s)
  Initial size: 4
  get(A) = 1
  get(B) = 2
  get(C) = 3
  get(D) = 4

  Waiting 1.5 seconds...
  get(A) = None [expired]
  get(B) = 2 [valid]
  get(C) = 3 [valid]
  get(D) = None [expired]

  Hits:        6
  Misses:      2
  Expirations: 2
  Cleanup removed: 0 entries
  Final size: 2
```

## TTL Selection Tradeoffs

| TTL Length | Benefit | Risk |
|---|---|---|
| Short (seconds) | Data stays fresh | High miss rate, more backend load |
| Long (hours) | High hit rate, low backend load | Stale data served to users |
| Per-key variable | Tailored freshness per data type | More complex management |

## Combining TTL with Eviction Policies

TTL and eviction policies address different concerns and are typically combined:

- **LRU + TTL**: On access, check TTL first. If expired, treat as miss. Otherwise, apply LRU ordering. This is the most common production configuration.
- **LFU + TTL**: Expired entries are removed regardless of frequency. Useful when data freshness is critical.
- **ARC + TTL**: Expired ghost entries should be removed from ghost lists to avoid misleading the adaptation parameter.

!!! note "TTL does not replace eviction"
    TTL handles data freshness (correctness concern), while LRU/LFU handle space management (performance concern). A cache needs both: TTL prevents serving stale data, and eviction prevents running out of memory.

## Reference

- Nishtala, R. et al. (2013). "Scaling Memcache at Facebook." *NSDI*.

## Exercises

**Exercise 1.**
Describe the difference between active TTL expiry (eagerly removing expired entries) and passive TTL expiry (checking on access). What are the tradeoffs in CPU usage and memory consumption?

??? success "Solution to Exercise 1"
    Active expiry uses a background thread or periodic sweep to remove entries whose TTL has elapsed. It keeps memory consumption low because expired entries are promptly deleted, but it consumes CPU cycles proportional to the number of entries checked, even if no one accesses them. Passive expiry checks the TTL only when an entry is accessed: if expired, it returns a miss and deletes the entry lazily. This uses zero CPU for entries that are never accessed again, but expired entries linger in memory until accessed or evicted by the cache's size-based policy. In practice, systems like Redis use a hybrid: passive checks on every access plus a periodic active sweep that samples a fraction of entries, bounding both memory waste and CPU overhead. $\square$

---

**Exercise 2.**
A cache uses TTL of 60 seconds and serves 1000 requests/second for a given key. The backend database takes 50 ms to respond. Estimate the "thundering herd" load on the database when the TTL expires, and propose a mitigation strategy.

??? success "Solution to Exercise 2"
    When the TTL expires, all 1000 req/s see a cache miss simultaneously and issue backend queries. During the 50 ms backend response time, $1000 \times 0.05 = 50$ concurrent requests hit the database for the same key. This can overwhelm the database. Mitigation: **request coalescing** (also called "single-flight" or "lock-based cache fill") -- when the first miss occurs, acquire a lock for that key. Subsequent misses for the same key wait on the lock instead of issuing redundant backend queries. Only the first request fetches from the database and populates the cache; all waiting requests receive the cached result. This reduces the thundering herd from 50 concurrent queries to exactly 1. $\square$

---

**Exercise 3.**
Design a TTL cache entry structure that supports $O(1)$ insertion, $O(1)$ lookup with passive expiry check, and $O(1)$ amortized active expiry. What data structures are needed?

??? success "Solution to Exercise 3"
    Each entry stores: key, value, and expiration timestamp ($\text{insertion\_time} + \text{TTL}$). Use a hash map for $O(1)$ key lookup. For active expiry, maintain entries in a **min-heap** (priority queue) keyed by expiration time, or more efficiently, a **doubly-linked list sorted by expiration time** (since TTLs typically produce entries in roughly chronological order). On lookup: check if `current_time > expiration`; if so, delete and return miss. For active sweep: pop entries from the front of the sorted list (or top of the heap) while their expiration is past. If all entries share the same TTL, they expire in insertion order, so a simple FIFO queue suffices and each expiry check is $O(1)$. With variable TTLs, a timer wheel (hashed by expiration time into buckets) provides $O(1)$ amortized active expiry. $\square$

---

**Exercise 4.**
Explain the concept of "jittered TTL" and prove that it reduces the probability of simultaneous cache invalidation for $n$ keys that would otherwise expire at the same time.

??? success "Solution to Exercise 4"
    Jittered TTL adds a random offset to each entry's TTL: instead of all entries expiring at exactly $T$ seconds, each entry $i$ expires at $T + U_i$ where $U_i \sim \text{Uniform}(-\delta, +\delta)$ for some jitter range $\delta$. Without jitter, all $n$ keys expire simultaneously, causing $n$ backend fetches at one instant. With jitter, expirations are spread over a window of $2\delta$. The expected number of keys expiring in any interval of length $\Delta t$ is $n \cdot \Delta t / (2\delta)$. For the peak load not to exceed the backend's capacity $C$ (queries/second), we need $n / (2\delta) \le C$, giving $\delta \ge n / (2C)$. The probability that all $n$ keys expire within the same 1-second window is $(1 / (2\delta))^{n-1}$, which is negligibly small for $\delta \ge 5$ seconds and $n > 10$. $\square$

---

**Exercise 5.**
A system caches exchange rate data with a 30-second TTL. A client reads a stale rate and executes a trade. Discuss the consistency implications of TTL-based caching in financial systems and propose a design that balances freshness and performance.

??? success "Solution to Exercise 5"
    TTL caching introduces a staleness window: data can be up to TTL seconds old. For exchange rates, a 30-second-old rate might differ by several basis points from the current rate, leading to trades at incorrect prices. The risk scales with volatility and trade size. Design proposal: (1) **Short TTL + event-driven invalidation**: use a 5-second TTL as a safety net, but also subscribe to a real-time rate feed that invalidates the cache entry immediately when the rate changes. (2) **Read-through with version checks**: on cache hit, include the rate's timestamp in the response. The trading engine rejects rates older than a configurable threshold (e.g., 2 seconds). (3) **Write-through for critical paths**: for order execution, bypass the cache and always fetch the live rate. Use the cache only for display and analytics where staleness is acceptable. This architecture separates latency-sensitive reads from consistency-critical writes. $\square$
