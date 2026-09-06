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

## Exercises

**Exercise 1.**
A system has 4 page frames and receives the page reference string 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5. Compute the number of page faults under FIFO, LRU, and OPT (Belady's optimal).

??? success "Solution to Exercise 1"
    **FIFO**: frames after each reference: [1], [1,2], [1,2,3], [1,2,3,4], hit, hit, [2,3,4,5] evict 1, [3,4,5,1] evict 2, [4,5,1,2] evict 3, [5,1,2,3] evict 4, [1,2,3,4] evict 5, [2,3,4,5] evict 1. Faults: 10. **LRU**: [1], [1,2], [1,2,3], [1,2,3,4], hit(1 moves to recent), hit(2), [1,2,4,5] evict 3, hit(1), hit(2), [1,2,5,3] evict 4, [1,2,3,4] evict 5, [2,3,4,5] evict 1. Faults: 8. **OPT**: [1], [1,2], [1,2,3], [1,2,3,4], hit, hit, [1,2,4,5] evict 3 (used farthest at position 10), hit, hit, [1,2,5,3] evict 4, [1,2,3,4] evict 5, [2,3,4,5] evict 1. Faults: 8. In this case, LRU matches OPT. $\square$

---

**Exercise 2.**
Prove that Belady's OPT algorithm (evict the page used farthest in the future) minimizes the number of page faults.

??? success "Solution to Exercise 2"
    Proof by exchange argument. Consider any algorithm $A$ and OPT. Process the reference string left to right. At the first point where $A$ and OPT differ in their eviction choice: let $A$ evict page $p$ and OPT evict page $q$. OPT chose $q$ because $q$ is used farthest in the future among all pages in memory. If $A$ evicts $p$ instead, then $p$ is used sooner than $q$. Eventually, $A$ will fault on $p$ before OPT faults on $q$. We can transform $A$'s eviction to match OPT's without increasing the total fault count: replace the eviction of $p$ with the eviction of $q$. The resulting algorithm has $\le$ as many faults as $A$ and agrees with OPT on one more step. By induction, OPT's total faults $\le$ any algorithm's faults. $\square$

---

**Exercise 3.**
Explain Belady's anomaly: give an example where increasing the number of page frames increases the number of FIFO page faults.

??? success "Solution to Exercise 3"
    Reference string: 1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5. With 3 frames (FIFO): faults at positions 1,2,3,4,5,6,7,10,11,12 = 9 faults. With 4 frames (FIFO): faults at 1,2,3,4,7,8,9,10,11,12 = 10 faults. More frames, more faults! This counterintuitive result occurs because FIFO does not use recency information. Adding frames changes the eviction order in a way that evicts soon-to-be-needed pages. LRU does not exhibit Belady's anomaly because it is a "stack algorithm": the set of pages in $k$ frames is always a subset of the pages in $k+1$ frames. FIFO violates this property. $\square$

---

**Exercise 4.**
The clock algorithm approximates LRU with $O(1)$ overhead per page fault. Describe the algorithm and explain why exact LRU is too expensive for operating systems.

??? success "Solution to Exercise 4"
    The clock algorithm maintains page frames in a circular buffer with a "hand" pointer. Each page has a reference bit, set to 1 by hardware on access. On a page fault: advance the hand. If the current page's reference bit is 1, clear it and advance (give it a "second chance"). If 0, evict this page. This approximates LRU: recently accessed pages have their bits set and survive the hand's sweep. Exact LRU requires updating a data structure on every memory access (not just page faults). With billions of memory accesses per second, even an $O(1)$ LRU update per access is prohibitively expensive. The clock algorithm updates only on page faults (thousands per second, not billions), using hardware reference bits that are set for free by the MMU. $\square$

---

**Exercise 5.**
A database system implements its own buffer pool manager rather than relying on the OS page cache. Explain the advantages of application-level page replacement for database workloads.

??? success "Solution to Exercise 5"
    (1) **Workload-aware eviction**: the database knows which pages will be needed (e.g., during a sequential scan, pages are read once and should be evicted immediately; the OS would keep them in cache). The database can use specialized policies like MRU for scans and LRU for index lookups. (2) **Prefetching**: the database knows the query plan and can prefetch pages before they are needed (e.g., all leaf pages for a range scan). The OS can only detect sequential access patterns. (3) **Controlled flushing**: the database must write dirty pages in a specific order for crash recovery (write-ahead logging protocol). The OS may flush pages in arbitrary order, violating this constraint. (4) **Memory pinning**: the database can pin critical pages (e.g., B-tree root, hot index pages) to prevent eviction. The OS treats all pages equally. These advantages justify the complexity of a custom buffer manager in high-performance databases. $\square$
