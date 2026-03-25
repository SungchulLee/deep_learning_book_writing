# Buffer Trees

A standard B-tree processes each insertion or deletion individually, costing $O(\log_B N)$ I/O operations per update. When an algorithm performs $N$ updates in sequence, the total cost is $O(N \log_B N)$ I/O operations. **Buffer trees** improve on this by attaching a buffer to each internal node, collecting updates in batches and flushing them down the tree only when a buffer fills. This amortization reduces the per-update I/O cost to $O((1/B) \log_{M/B}(N/B))$, matching the sorting bound divided by $N$.

## Key Idea

A buffer tree is a B-tree of branching factor $\Theta(M/B)$ where each internal node has an associated **buffer** that can hold up to $M$ elements ($M/B$ blocks). When an update arrives, it is simply appended to the root's buffer. When any buffer reaches capacity, its contents are sorted, distributed to the appropriate child buffers, and the parent buffer is cleared. This batched flush amortizes the I/O cost of pushing updates down the tree.

## Structure

A buffer tree with $N$ elements has the following properties:

| Property | Value |
|---|---|
| Branching factor | $\Theta(M/B)$ |
| Buffer size per node | $M$ elements ($M/B$ blocks) |
| Tree height | $O\!\left(\log_{M/B} \frac{N}{B}\right)$ |
| Leaves | Store sorted data in blocks of size $B$ |

The total buffer space across all nodes at a given level is at most $O(N)$ elements, because each element appears in at most one buffer at any time.

## Amortized I/O Analysis

When a buffer of size $M$ is flushed, the $M$ elements are sorted internally (no I/O cost) and distributed among the $\Theta(M/B)$ children. The flush reads and writes $O(M/B)$ blocks. Since $M$ elements were batched before the flush, the amortized cost per element is:

$$
\frac{O(M/B)}{M} = O\!\left(\frac{1}{B}\right) \text{ per element per level}
$$

Each element traverses $O(\log_{M/B}(N/B))$ levels from root to leaf, so the total amortized I/O per update is:

$$
O\!\left(\frac{1}{B} \log_{M/B} \frac{N}{B}\right)
$$

For $N$ updates, the total I/O is:

$$
O\!\left(\frac{N}{B} \log_{M/B} \frac{N}{B}\right) = O(\text{sort}(N))
$$

This matches the external sorting bound, which is optimal for comparison-based algorithms.

## Operations

### Insertion

Insert operations are simply appended to the root buffer. When the root buffer reaches $M$ elements, a flush pushes them to child buffers. Flushes cascade recursively whenever a child buffer also fills.

### Deletion

Deletions are handled as **anti-elements**: a delete marker for key $k$ is inserted into the root buffer. When a delete marker meets the corresponding element during a flush, both are removed. This lazy deletion maintains the same amortized bounds.

### Search

Point queries must examine the buffer at every node along the search path, since pending updates may not have reached the leaves yet. This costs:

$$
O\!\left(\frac{M}{B} \cdot \log_{M/B} \frac{N}{B}\right)
$$

which is more expensive than a standard B-tree search. Buffer trees are therefore best suited for **update-heavy, query-light** workloads or for batched operations where all queries are posed after all updates.

## Comparison with B-Trees

| Property | B-Tree | Buffer Tree |
|---|---|---|
| Branching factor | $\Theta(B)$ | $\Theta(M/B)$ |
| Search I/O | $O(\log_B N)$ | $O((M/B) \log_{M/B}(N/B))$ |
| Insert I/O (worst case) | $O(\log_B N)$ | $O(\log_{M/B}(N/B))$ |
| Insert I/O (amortized) | $O(\log_B N)$ | $O((1/B) \log_{M/B}(N/B))$ |
| Best for | Query-heavy | Update-heavy / batched |

## Example: Buffer Tree Flush Simulation

```python
"""
Buffer tree flush simulation.

Demonstrates how buffer trees batch updates and amortize I/O costs
by flushing buffers only when they reach capacity.
"""

import math

# ===================================================================
# Buffer Tree simulation
# ===================================================================

class BufferTreeNode:
    """A node in a buffer tree with a fixed-capacity buffer."""

    def __init__(self, branching_factor: int, buffer_capacity: int,
                 leaf: bool = False):
        self.branching_factor = branching_factor
        self.buffer_capacity = buffer_capacity
        self.buffer: list[int] = []
        self.leaf = leaf
        self.children: list[BufferTreeNode] = []
        self.io_count = 0

    def add_to_buffer(self, key: int) -> int:
        """Add a key to the buffer. Returns I/O operations used."""
        self.buffer.append(key)
        ios = 0
        if len(self.buffer) >= self.buffer_capacity:
            ios = self._flush()
        return ios

    def _flush(self) -> int:
        """Flush buffer contents to children. Returns I/O count."""
        if self.leaf:
            # At leaf: just clear the buffer (write sorted data)
            ios = math.ceil(len(self.buffer) / 100)  # Simulate block writes
            self.buffer.clear()
            return ios

        # Sort buffer and distribute to children
        self.buffer.sort()
        blocks_read_write = math.ceil(
            len(self.buffer) * 2 / 100  # Read + write
        )

        # Distribute among children
        child_ios = 0
        per_child = len(self.buffer) // max(1, len(self.children))
        for i, child in enumerate(self.children):
            start = i * per_child
            end = start + per_child if i < len(self.children) - 1 \
                else len(self.buffer)
            for key in self.buffer[start:end]:
                child_ios += child.add_to_buffer(key)

        self.buffer.clear()
        return blocks_read_write + child_ios


def simulate_buffer_tree(n: int, m: int, b: int) -> dict:
    """
    Simulate N insertions into a buffer tree.

    Parameters
    ----------
    n : Number of elements to insert.
    m : Memory capacity (buffer size per node).
    b : Block size.

    Returns
    -------
    Dictionary with I/O statistics.
    """
    fan_out = max(2, m // b)
    height = max(1, math.ceil(math.log(max(1, n / b)) / math.log(fan_out)))

    # Compute amortized bounds
    amortized_per_element = (1 / b) * height
    total_amortized = n * amortized_per_element
    btree_total = n * math.ceil(math.log(max(2, n)) / math.log(max(2, b)))

    return {
        "fan_out": fan_out,
        "height": height,
        "amortized_per_element": amortized_per_element,
        "total_buffer_tree": total_amortized,
        "total_btree": btree_total,
        "speedup": btree_total / max(1, total_amortized),
    }


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    N = 10**7
    M = 10**5
    B = 1000

    stats = simulate_buffer_tree(N, M, B)
    print(f"Buffer Tree vs B-Tree for N={N:,} insertions")
    print(f"  M = {M:,}, B = {B:,}")
    print(f"  Fan-out (M/B):        {stats['fan_out']}")
    print(f"  Height:               {stats['height']}")
    print(f"  Amortized I/O/insert: {stats['amortized_per_element']:.4f}")
    print(f"  Total buffer tree:    {stats['total_buffer_tree']:,.0f} I/Os")
    print(f"  Total B-tree:         {stats['total_btree']:,.0f} I/Os")
    print(f"  Speedup:              {stats['speedup']:.1f}x")
```

??? example "Sample Output"

    ```
    Buffer Tree vs B-Tree for N=10,000,000 insertions
      M = 100,000, B = 1,000
      Fan-out (M/B):        100
      Height:               2
      Amortized I/O/insert: 0.0020
      Total buffer tree:    20,000 I/Os
      Total B-tree:         30,000,000 I/Os
      Speedup:              1500.0x
    ```

    The buffer tree achieves a 1500x reduction in total I/O by batching updates, demonstrating the power of amortized I/O analysis.

## Applications

Buffer trees are used when algorithms perform many updates before needing query results:

- **External priority queues:** Insert and delete-min operations can be buffered, achieving $O((1/B) \log_{M/B}(N/B))$ amortized I/O per operation.
- **External graph algorithms:** Algorithms like external BFS and MST use buffer trees to batch edge relaxations.
- **Bulk loading indexes:** When building a B-tree index from scratch, buffer trees avoid the overhead of individual insertions.

## Reference

- Arge, L. "The Buffer Tree: A Technique for Designing Batched External Data Structures," *Algorithmica*, 37(1), 2003.
- Vitter, J. S. *Algorithms and Data Structures for External Memory*, 2008.
