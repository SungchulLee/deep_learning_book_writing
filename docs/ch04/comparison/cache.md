# Cache Performance

Asymptotic complexity alone does not fully predict the real-world speed of
data structure operations. A linked list traversal and an array traversal
are both $O(n)$, yet the array version often runs several times faster in
practice. The reason lies in how modern processors access memory: through a
hierarchy of caches that reward **spatial locality** and **temporal
locality**. This page explains why arrays are cache-friendly, why linked
lists are cache-hostile, and how to reason about cache effects when choosing
a data structure.

## The Memory Hierarchy

Modern computers access memory through multiple levels of cache, each faster
but smaller than the next:

| Level | Typical size | Access latency |
|---|---|---|
| L1 cache | 32--64 KB | ~1 ns |
| L2 cache | 256 KB--1 MB | ~3--10 ns |
| L3 cache | 4--32 MB | ~10--40 ns |
| Main memory (DRAM) | 8--64 GB | ~50--100 ns |

When the CPU reads a single byte, it does not fetch just that byte. Instead,
it loads an entire **cache line** (typically 64 bytes) into the cache. If the
next memory access falls within the same cache line, it is served from the
cache -- a **cache hit**. If it falls outside all cached lines, the CPU must
wait for main memory -- a **cache miss**.

## Spatial Locality

**Spatial locality** means that accesses to nearby memory addresses tend to
occur close together in time. Arrays exploit spatial locality perfectly:
elements are stored contiguously, so a single cache line prefetch loads
multiple adjacent elements.

Consider iterating over an array of 4-byte integers. A 64-byte cache line
holds 16 integers. After the first access triggers a cache miss, the next
15 accesses are cache hits -- a miss rate of only $1/16 \approx 6\%$.

In contrast, linked list nodes are allocated independently by the memory
allocator. Successive nodes may reside on different cache lines or even
different memory pages. Each `node = node.next` dereference potentially
triggers a cache miss, approaching a miss rate of $100\%$ in the worst
case.

## Temporal Locality

**Temporal locality** means that recently accessed memory is likely to be
accessed again soon. Both arrays and linked lists benefit equally from
temporal locality in repeated traversals -- the advantage of arrays is
primarily spatial.

However, arrays gain an indirect temporal benefit: because fewer cache
lines are loaded (elements are packed densely), the working set stays
smaller, and previously loaded data is less likely to be evicted before
reuse.

## Quantifying the Impact

The practical speed gap between array and linked list traversal is
significant. A rough model illustrates why.

Suppose each cache miss costs an extra 50 ns compared to a cache hit, and
we traverse $n$ elements:

- **Array**: approximately $n / 16$ cache misses (one per cache line of
  4-byte integers).
- **Linked list**: up to $n$ cache misses in the worst case.

For $n = 10{,}000$ elements:

- Array: ~625 misses $\times$ 50 ns = ~31 microseconds of miss penalty.
- Linked list: ~10,000 misses $\times$ 50 ns = ~500 microseconds of
  miss penalty.

This is a $16\times$ difference in miss penalty alone, on top of identical
$O(n)$ computation. In practice, hardware prefetchers further reduce array
miss rates because they detect sequential access patterns automatically.

## Prefetching

Modern CPUs include hardware prefetchers that detect sequential memory
access patterns and load cache lines ahead of the program's demand. Arrays
benefit enormously from prefetching because their access pattern is
perfectly sequential.

Linked lists defeat prefetchers because the address of the next node is
not known until the current node's pointer is dereferenced. This creates
a **pointer-chasing** pattern that is inherently serial and
unpredictable from the hardware's perspective.

## Practical Implications

| Factor | Array | Linked list |
|---|---|---|
| Spatial locality | Excellent | Poor |
| Hardware prefetching | Effective | Defeated |
| Cache line utilization | High (data packed) | Low (pointers + fragmentation) |
| TLB misses | Rare (contiguous pages) | Frequent (scattered pages) |
| Branch prediction | N/A | Indirect jumps less predictable |

!!! tip "Rule of thumb"
    If the workload is traversal-heavy, prefer arrays even when asymptotic
    complexity suggests linked lists should be equivalent. The constant
    factor hidden in $O(n)$ can differ by an order of magnitude due to
    cache effects.

## Mitigations for Linked Lists

When a linked list is structurally necessary, several techniques reduce
cache misses:

- **Pool allocation**: Allocate nodes from a pre-allocated contiguous
  buffer so that nodes are physically close in memory.
- **Unrolled linked lists**: Store multiple elements per node, combining
  the structural flexibility of a linked list with the spatial locality
  of an array within each node.
- **Cache-oblivious layouts**: Rearrange nodes in memory order to match
  traversal order (requires periodic reorganization).

These techniques narrow the performance gap but do not fully close it.

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.
  *Introduction to Algorithms* (4th ed.), Chapter 10. MIT Press.
- Drepper, U. *What Every Programmer Should Know About Memory*, 2007.
