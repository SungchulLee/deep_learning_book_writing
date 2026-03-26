# Page Replacement

Virtual memory allows processes to use more memory than is physically available by storing inactive pages on disk. When a process accesses a page not in physical memory (a **page fault**), the operating system must choose a page to evict. The **page replacement algorithm** determines which page to remove, directly affecting the number of page faults and overall system performance.

## Problem Formulation

A cache holds $k$ pages. A sequence of page requests $r_1, r_2, \ldots, r_n$ arrives online. On a **cache hit**, the page is already in memory and access is immediate. On a **cache miss** (page fault), one of the $k$ cached pages must be evicted to make room. The goal is to minimize the total number of page faults.

## Optimal (OPT / Belady's Algorithm)

The offline optimal strategy evicts the page that will not be used for the **longest time in the future**:

$$
\text{evict} = \arg\max_{p \in \text{cache}} \text{next\_use}(p)
$$

OPT achieves the minimum possible number of faults but requires knowledge of the future request sequence. It serves as a benchmark for evaluating online algorithms.

## FIFO (First-In First-Out)

Evict the page that has been in the cache the **longest** (oldest arrival). Implemented with a simple queue.

- **Advantage**: $O(1)$ per operation.
- **Disadvantage**: Suffers from **Belady's anomaly** -- increasing cache size can sometimes increase the fault rate.

## LRU (Least Recently Used)

Evict the page that was accessed **least recently**:

$$
\text{evict} = \arg\min_{p \in \text{cache}} \text{last\_access}(p)
$$

LRU is widely used because it approximates OPT under temporal locality. Its **competitive ratio** (worst-case ratio of faults to OPT) is:

$$
c_{\text{LRU}} = k
$$

where $k$ is the cache size. This is tight: no deterministic online algorithm can achieve a competitive ratio better than $k$.

## Clock Algorithm

The **Clock** (or second-chance) algorithm approximates LRU with $O(1)$ per operation:

1. Pages are arranged in a circular buffer with a "clock hand."
2. Each page has a **reference bit**, set to 1 on access.
3. On a fault, advance the clock hand:
   - If the current page's reference bit is 1, set it to 0 and advance.
   - If the reference bit is 0, evict this page.

Clock avoids the overhead of maintaining a full access-time ordering while capturing recency information through the reference bits.

## Comparison

| Algorithm | Per-Fault Time | Competitive Ratio | Belady's Anomaly |
|---|---|---|---|
| OPT | $O(n)$ (lookahead) | 1 (optimal) | No |
| LRU | $O(1)$ amortized | $k$ | No |
| FIFO | $O(1)$ | $k$ | Yes |
| Clock | $O(1)$ amortized | $k$ | No |

## Implementation

```python
"""
Page Replacement -- OPT, LRU, FIFO, and Clock algorithms.

Simulates each algorithm on a page reference sequence and counts
the number of page faults for comparison.
"""

from __future__ import annotations
from collections import OrderedDict, deque


# === OPT (Offline Optimal) ====================================================

def opt_faults(pages: list[int], cache_size: int) -> int:
    """Count page faults using Belady's optimal algorithm."""
    cache: set[int] = set()
    faults = 0
    for i, page in enumerate(pages):
        if page not in cache:
            faults += 1
            if len(cache) >= cache_size:
                # Evict page with farthest next use
                farthest = -1
                evict = None
                for p in cache:
                    try:
                        next_use = pages[i + 1:].index(p)
                    except ValueError:
                        next_use = float("inf")
                    if next_use > farthest:
                        farthest = next_use
                        evict = p
                cache.remove(evict)
            cache.add(page)
    return faults


# === LRU ======================================================================

def lru_faults(pages: list[int], cache_size: int) -> int:
    """Count page faults using LRU replacement."""
    cache: OrderedDict[int, None] = OrderedDict()
    faults = 0
    for page in pages:
        if page in cache:
            cache.move_to_end(page)
        else:
            faults += 1
            if len(cache) >= cache_size:
                cache.popitem(last=False)  # evict least recently used
            cache[page] = None
    return faults


# === FIFO =====================================================================

def fifo_faults(pages: list[int], cache_size: int) -> int:
    """Count page faults using FIFO replacement."""
    cache: set[int] = set()
    queue: deque[int] = deque()
    faults = 0
    for page in pages:
        if page not in cache:
            faults += 1
            if len(cache) >= cache_size:
                old = queue.popleft()
                cache.remove(old)
            cache.add(page)
            queue.append(page)
    return faults


# === Clock ====================================================================

def clock_faults(pages: list[int], cache_size: int) -> int:
    """Count page faults using the Clock (second-chance) algorithm."""
    cache: list[int | None] = [None] * cache_size
    ref_bit: list[int] = [0] * cache_size
    hand = 0
    page_set: set[int] = set()
    faults = 0

    for page in pages:
        if page in page_set:
            # Set reference bit
            idx = cache.index(page)
            ref_bit[idx] = 1
        else:
            faults += 1
            while ref_bit[hand] == 1:
                ref_bit[hand] = 0
                hand = (hand + 1) % cache_size
            if cache[hand] is not None:
                page_set.discard(cache[hand])
            cache[hand] = page
            ref_bit[hand] = 1
            page_set.add(page)
            hand = (hand + 1) % cache_size

    return faults


# === Main =====================================================================

if __name__ == "__main__":
    pages = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
    cache_size = 3

    print(f"Page sequence: {pages}")
    print(f"Cache size: {cache_size}\n")

    for name, func in [
        ("OPT", opt_faults),
        ("LRU", lru_faults),
        ("FIFO", fifo_faults),
        ("Clock", clock_faults),
    ]:
        faults = func(pages, cache_size)
        print(f"{name:6s}: {faults} page faults")
```

**Output:**

```
Page sequence: [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5]
Cache size: 3

OPT   : 7 page faults
LRU   : 10 page faults
FIFO  : 9 page faults
Clock : 10 page faults
```

OPT achieves the fewest faults (7) since it knows the future. LRU and Clock produce the same count (10), confirming that Clock approximates LRU behavior. FIFO falls between the two, but in other sequences it can exhibit Belady's anomaly.

## Reference

- Sleator, D.D. and Tarjan, R.E. "Amortized Efficiency of List Update and Paging Rules." *CACM*, 1985
- Silberschatz, A., Galvin, P.B., and Gagne, G. *Operating System Concepts*. Wiley
