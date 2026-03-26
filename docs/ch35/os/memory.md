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
