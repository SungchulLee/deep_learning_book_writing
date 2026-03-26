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
