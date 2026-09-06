# Memory Allocation

An operating system must manage a pool of physical memory, allocating and freeing blocks of varying sizes as processes request and release memory. The **memory allocation** problem is to satisfy allocation requests quickly while minimizing wasted space (fragmentation). Different algorithms trade off speed, fragmentation, and implementation complexity.

## The Allocation Problem

Given a contiguous memory region of $N$ bytes and a sequence of allocation requests (of size $s_i$) and free operations, find a free block of at least $s_i$ bytes for each request.

Two types of fragmentation arise:

- **External fragmentation**: Free memory is scattered in small blocks, so a large request cannot be satisfied even though total free memory is sufficient.
- **Internal fragmentation**: An allocated block is larger than requested; the excess is wasted.

## First-Fit

Scan the free list from the beginning and allocate the **first** block large enough:

$$
T_{\text{alloc}} = O(n), \quad T_{\text{free}} = O(n)
$$

where $n$ is the number of free blocks. First-fit tends to fragment the beginning of memory, but is fast in practice because it stops at the first match.

## Best-Fit

Scan the entire free list and choose the **smallest** block that satisfies the request:

$$
T_{\text{alloc}} = O(n), \quad T_{\text{free}} = O(n)
$$

Best-fit minimizes immediate waste but tends to create many tiny leftover fragments that are too small to be useful.

## Worst-Fit

Choose the **largest** free block, leaving a large remainder that may still be usable:

$$
T_{\text{alloc}} = O(n), \quad T_{\text{free}} = O(n)
$$

In practice, worst-fit performs poorly because it rapidly exhausts large blocks.

## Buddy System

The **buddy system** constrains all block sizes to powers of two, enabling $O(\log N)$ allocation and coalescing.

To allocate $s$ bytes:

1. Round $s$ up to the next power of two: $2^k$ where $k = \lceil \log_2 s \rceil$.
2. If a free block of size $2^k$ exists, allocate it.
3. Otherwise, find the smallest free block of size $2^j > 2^k$, and recursively split it in half until a block of size $2^k$ is obtained.

To free a block of size $2^k$ at address $a$:

1. Compute the buddy address: $a \oplus 2^k$ (XOR flips the $k$-th bit).
2. If the buddy is free, merge them into a block of size $2^{k+1}$ and repeat.

$$
T_{\text{alloc}} = O(\log N), \quad T_{\text{free}} = O(\log N)
$$

!!! warning "Internal fragmentation in buddy systems"
    A request for $2^k + 1$ bytes wastes nearly half the allocated block ($2^{k+1}$ bytes). The worst-case internal fragmentation is approximately 50%.

## Comparison

| Algorithm | Alloc Time | Free Time | External Frag. | Internal Frag. |
|---|---|---|---|---|
| First-fit | $O(n)$ | $O(n)$ | Moderate | Low |
| Best-fit | $O(n)$ | $O(n)$ | High (tiny fragments) | Minimal |
| Worst-fit | $O(n)$ | $O(n)$ | High | Moderate |
| Buddy system | $O(\log N)$ | $O(\log N)$ | Low | Up to 50% |

## Implementation

```python
"""
Memory Allocation -- first-fit, best-fit, and buddy system.

Simulates three allocation strategies on a contiguous memory pool
and reports fragmentation after a sequence of allocations and frees.
"""

from __future__ import annotations


# === First-Fit Allocator ======================================================

class FirstFitAllocator:
    """Allocate using the first sufficiently large free block."""

    def __init__(self, size: int):
        self.size = size
        self.free_blocks: list[tuple[int, int]] = [(0, size)]  # (start, size)
        self.allocated: dict[int, int] = {}  # start -> size

    def alloc(self, request: int) -> int | None:
        """Allocate *request* bytes. Returns start address or None."""
        for i, (start, bsize) in enumerate(self.free_blocks):
            if bsize >= request:
                self.allocated[start] = request
                if bsize > request:
                    self.free_blocks[i] = (start + request, bsize - request)
                else:
                    self.free_blocks.pop(i)
                return start
        return None

    def free(self, addr: int) -> None:
        """Free the block at *addr*."""
        size = self.allocated.pop(addr)
        self.free_blocks.append((addr, size))
        self.free_blocks.sort()
        self._coalesce()

    def _coalesce(self) -> None:
        merged = []
        for start, size in self.free_blocks:
            if merged and merged[-1][0] + merged[-1][1] == start:
                merged[-1] = (merged[-1][0], merged[-1][1] + size)
            else:
                merged.append((start, size))
        self.free_blocks = merged

    def fragmentation(self) -> int:
        """Number of free fragments."""
        return len(self.free_blocks)


# === Buddy System =============================================================

class BuddyAllocator:
    """Power-of-two buddy system allocator."""

    def __init__(self, total_size: int):
        self.total_order = total_size.bit_length() - 1
        if (1 << self.total_order) < total_size:
            self.total_order += 1
        self.total_size = 1 << self.total_order
        # free_lists[k] = set of free block start addresses of size 2^k
        self.free_lists: list[set[int]] = [set() for _ in range(self.total_order + 1)]
        self.free_lists[self.total_order].add(0)
        self.allocated: dict[int, int] = {}  # start -> order

    def alloc(self, request: int) -> int | None:
        """Allocate at least *request* bytes. Returns start address."""
        order = max(0, (request - 1).bit_length())
        # Find smallest available block >= 2^order
        for k in range(order, self.total_order + 1):
            if self.free_lists[k]:
                addr = self.free_lists[k].pop()
                # Split down to target order
                while k > order:
                    k -= 1
                    buddy = addr + (1 << k)
                    self.free_lists[k].add(buddy)
                self.allocated[addr] = order
                return addr
        return None

    def free(self, addr: int) -> None:
        """Free the block at *addr* and coalesce with buddy if possible."""
        order = self.allocated.pop(addr)
        while order < self.total_order:
            buddy = addr ^ (1 << order)
            if buddy in self.free_lists[order]:
                self.free_lists[order].remove(buddy)
                addr = min(addr, buddy)
                order += 1
            else:
                break
        self.free_lists[order].add(addr)


# === Main =====================================================================

if __name__ == "__main__":
    print("First-Fit Allocator (1024 bytes):")
    ff = FirstFitAllocator(1024)
    a1 = ff.alloc(200)
    a2 = ff.alloc(300)
    a3 = ff.alloc(100)
    print(f"  Allocated: {a1}, {a2}, {a3}")
    ff.free(a2)  # free middle block
    print(f"  After freeing {a2}: {ff.fragmentation()} free fragments")
    a4 = ff.alloc(250)
    print(f"  Alloc 250 -> {a4} (reuses freed block)")

    print("\nBuddy Allocator (1024 bytes):")
    buddy = BuddyAllocator(1024)
    b1 = buddy.alloc(100)  # gets 128
    b2 = buddy.alloc(200)  # gets 256
    b3 = buddy.alloc(50)   # gets 64
    print(f"  Allocated: {b1} (128B), {b2} (256B), {b3} (64B)")
    buddy.free(b1)
    buddy.free(b3)
    print(f"  Freed {b1} and {b3}")
    b4 = buddy.alloc(60)   # gets 64
    print(f"  Alloc 60 -> {b4} (gets 64B block)")
```

**Output:**

```
First-Fit Allocator (1024 bytes):
  Allocated: 0, 200, 500
  After freeing 200: 2 free fragments
  Alloc 250 -> 200 (reuses freed block)

Buddy Allocator (1024 bytes):
  Allocated: 0 (128B), 256 (256B), 128 (64B)
  Freed 0 and 128
  Alloc 60 -> 0 (gets 64B block)
```

First-fit reuses the freed middle block for the next allocation. The buddy system demonstrates power-of-two splitting and coalescing, with the XOR-based buddy computation enabling efficient merge operations.

## Reference

- Knuth, D.E. *The Art of Computer Programming, Vol. 1: Fundamental Algorithms*. Addison-Wesley
- Silberschatz, A., Galvin, P.B., and Gagne, G. *Operating System Concepts*. Wiley

## Exercises

**Exercise 1.**
Compare first-fit, best-fit, and worst-fit allocation strategies. Which minimizes external fragmentation in practice?

??? success "Solution to Exercise 1"
    **First-fit**: scan the free list from the start, allocate the first block that is large enough. Fast ($O(n)$ worst case, often $O(1)$ with a sorted list). Tends to fragment the beginning of memory. **Best-fit**: search for the smallest block that fits. Minimizes wasted space per allocation but creates many tiny unusable fragments. Slow ($O(n)$ without optimization). **Worst-fit**: allocate the largest available block. The leftover fragment is larger and potentially reusable. But it quickly consumes large blocks needed for big allocations. In practice, first-fit minimizes fragmentation best: it is fast, produces moderate-sized fragments, and performs well across diverse workloads. Best-fit creates too many tiny fragments; worst-fit wastes large blocks. Knuth's simulations confirmed first-fit's superiority for general workloads. $\square$

---

**Exercise 2.**
Explain the buddy system for memory allocation. What are its time complexity and fragmentation characteristics?

??? success "Solution to Exercise 2"
    The buddy system divides memory into blocks of sizes $2^k$. To allocate a request of size $s$: find the smallest $k$ such that $2^k \ge s$. If no free block of size $2^k$ exists, split a larger block ($2^{k+1}$) into two "buddies" of size $2^k$. On deallocation, if a block's buddy is also free, merge them into a $2^{k+1}$ block (recursively). Time complexity: allocation and deallocation are $O(\log n)$ where $n$ is the total memory size (at most $\log_2 n$ levels of splitting/merging). Fragmentation: internal fragmentation can be up to 50% (a request for $2^k + 1$ bytes wastes nearly half of a $2^{k+1}$ block). External fragmentation is limited because merging consolidates adjacent free blocks. The buddy system is used in the Linux kernel's page allocator. $\square$

---

**Exercise 3.**
A process requests blocks of sizes 100, 250, 50, 300, 200, 150 from a 1000-byte heap. Show the state of the free list after each allocation using first-fit, and identify when fragmentation prevents allocation despite sufficient total free space.

??? success "Solution to Exercise 3"
    Initial free list: [0-999] (1000 bytes). Alloc 100: [100-999] free. Alloc 250: [350-999] free. Alloc 50: [400-999] free. Alloc 300: [700-999] free (300 bytes left). Alloc 200: [900-999] free (100 bytes left). Alloc 150: fails! Only 100 bytes free, need 150. Total allocated: 100+250+50+300+200 = 900 bytes. Total free: 100 bytes. No fragmentation issue here -- simply not enough total free space. Now consider: free the 250-byte block (positions 100-349). Free list: [100-349] (250), [900-999] (100). Total free: 350 bytes. Alloc 300: 300 fits in [100-349]. Success. But if we had freed the 100-byte block instead: free list: [0-99] (100), [900-999] (100). Total free: 200 bytes. Alloc 150: fails despite 200 bytes free -- external fragmentation. $\square$

---

**Exercise 4.**
Describe slab allocation (used in the Linux kernel). Why is it more efficient than general-purpose allocation for kernel objects?

??? success "Solution to Exercise 4"
    The kernel frequently allocates and frees objects of fixed sizes (inodes, task structs, socket buffers). A slab allocator pre-allocates pages ("slabs") divided into fixed-size slots matching specific object types. Each object type has a cache of pre-initialized slabs. Allocation: take a free slot from the cache ($O(1)$). Deallocation: return the slot ($O(1)$). Advantages over general-purpose allocators: (1) **Zero fragmentation** for the target object size (all slots are exactly the right size). (2) **Constructor reuse**: freed objects retain partial initialization, so re-allocation skips expensive setup. (3) **Cache locality**: objects of the same type are packed into contiguous pages, improving CPU cache hit rates. (4) **No search overhead**: free slots are tracked per-cache, not in a global free list. The tradeoff: memory used by inactive caches is not available for other purposes until the cache is shrunk. $\square$

---

**Exercise 5.**
Explain how `malloc` in glibc manages memory using arenas, bins, and mmap. Why does it use different strategies for small and large allocations?

??? success "Solution to Exercise 5"
    glibc `malloc` (ptmalloc2) uses: (1) **Arenas**: each thread has a preferred arena (a separate heap region) to reduce lock contention in multithreaded programs. (2) **Bins**: free chunks are organized by size into bins. Small bins (exact sizes up to 512 bytes) use doubly-linked lists. Large bins (512+ bytes) use sorted lists with best-fit search. Fast bins cache recently freed small chunks for immediate reuse without coalescing. (3) **mmap**: allocations above a threshold (default 128 KB) use `mmap` to obtain memory directly from the OS, bypassing the heap entirely. Freed mmap'd regions are returned to the OS immediately. Small allocations use bins because they are frequent and must be fast; pooling avoids system call overhead. Large allocations use mmap because they are rare, and returning memory to the OS prevents long-lived large blocks from fragmenting the heap. The dual strategy optimizes both throughput (small allocations) and memory efficiency (large allocations). $\square$
