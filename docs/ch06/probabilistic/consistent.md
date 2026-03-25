# Consistent Hashing

When a distributed system stores data across $m$ servers using simple modular hashing $h(k) = k \bmod m$, adding or removing a single server changes $m$ and forces nearly every key to remap to a different server.  For a cache cluster serving millions of requests, this mass remapping triggers a **cache avalanche** -- almost every request becomes a cache miss simultaneously.  Consistent hashing solves this problem by ensuring that adding or removing a node remaps only $O(n/m)$ keys on average, where $n$ is the total number of keys and $m$ is the number of nodes.

## The Problem with Modular Hashing

Consider $n$ keys distributed across $m$ servers using $h(k) = k \bmod m$.  When the server count changes from $m$ to $m+1$, a key $k$ stays on the same server only if:

$$
k \bmod m = k \bmod (m+1)
$$

For most keys, this equality does not hold.  The fraction of keys that must move approaches:

$$
\frac{m}{m+1} \approx 1 - \frac{1}{m}
$$

With $m = 100$ servers, approximately 99% of keys must be remapped when a single server is added.  This is unacceptable for large-scale systems.

## The Hash Ring

Consistent hashing maps both keys and nodes onto a circular hash space (a "ring") of size $2^{b}$, where $b$ is the number of output bits of the hash function.

**Setup.** Choose a hash function $h: U \to [0, 2^b)$ (e.g., SHA-1 with $b = 160$).  Each node $s_i$ is hashed to a point on the ring: $h(s_i)$.  Each key $k$ is also hashed to a point: $h(k)$.

**Assignment rule.** A key $k$ is assigned to the first node encountered when walking clockwise from $h(k)$ on the ring.  Formally, key $k$ is stored on node:

$$
\text{node}(k) = \arg\min_{s_i} \bigl( h(s_i) - h(k) \bigr) \bmod 2^b
$$

where the minimum is taken over all active nodes.

??? example "Walking the ring"
    Suppose we have three nodes $A$, $B$, $C$ hashed to positions 10, 80, 200 on a ring of size 256.  A key $k$ with $h(k) = 50$ lies between $A$ (10) and $B$ (80), so it is assigned to $B$.  A key with $h(k) = 210$ lies between $C$ (200) and $A$ (10, wrapping around), so it is assigned to $A$.

## Adding and Removing Nodes

The key advantage of consistent hashing is the minimal disruption when the node set changes.

**Adding a node.** When a new node $s_{\text{new}}$ joins, it takes over only the keys in the arc between its predecessor on the ring and itself.  All other keys remain on their current nodes.

**Removing a node.** When node $s_i$ is removed, its keys transfer to the next node clockwise.  No other keys move.

**Theorem.** When a node is added to or removed from a system of $m$ nodes storing $n$ keys, the expected number of keys that must be remapped is:

$$
\frac{n}{m+1} \quad \text{(addition)} \qquad \text{or} \qquad \frac{n}{m} \quad \text{(removal)}
$$

assuming keys are uniformly distributed on the ring.

This is optimal: any scheme that balances load across $m$ nodes must reassign at least $n/(m+1)$ keys when a new node joins, since the new node must receive its fair share of keys.

## Virtual Nodes

With only $m$ physical nodes on the ring, the arc lengths between adjacent nodes can be highly uneven, leading to load imbalance.  The standard deviation of the load per node with $m$ nodes is $O(n / \sqrt{m})$, which is large.

**Virtual nodes** solve this by mapping each physical node to $v$ positions on the ring.  Physical node $s_i$ is represented by $v$ virtual nodes $s_i^{(1)}, s_i^{(2)}, \ldots, s_i^{(v)}$, each hashed independently:

$$
h(s_i^{(j)}) = h(\text{concat}(s_i, j)) \quad \text{for } j = 1, 2, \ldots, v
$$

With $v$ virtual nodes per physical node, the ring has $vm$ points, and the arc lengths become more uniform.  The load imbalance decreases as $v$ increases:

$$
\text{max load} \leq \frac{n}{m} \left(1 + O\!\left(\sqrt{\frac{\log(vm)}{v}}\right)\right)
$$

In practice, $v = 100$ to $200$ virtual nodes per physical node provides good balance.

!!! tip "Choosing the number of virtual nodes"
    More virtual nodes give better balance but increase the size of the ring data structure (a sorted array or balanced BST of $vm$ entries).  Lookup time grows from $O(\log m)$ to $O(\log(vm))$, which is a modest increase.  Systems like Amazon DynamoDB and Apache Cassandra typically use 256 virtual nodes per physical node.

## Implementation

The ring is stored as a sorted array of (hash value, node ID) pairs.  Key lookup uses binary search to find the first node clockwise from the key's hash.

```python
"""Consistent hashing with virtual nodes."""

import hashlib
from bisect import bisect_right


# === Consistent Hash Ring ===

class ConsistentHashRing:
    """A consistent hash ring with configurable virtual nodes."""

    def __init__(self, num_virtual: int = 150):
        self.num_virtual = num_virtual
        self.ring: list[tuple[int, str]] = []
        self.nodes: set[str] = set()

    def _hash(self, key: str) -> int:
        """Hash a key to a position on the ring."""
        digest = hashlib.sha256(key.encode()).hexdigest()
        return int(digest, 16) % (2**32)

    def add_node(self, node: str) -> None:
        """Add a physical node with its virtual nodes."""
        self.nodes.add(node)
        for i in range(self.num_virtual):
            h = self._hash(f"{node}:{i}")
            self.ring.append((h, node))
        self.ring.sort()

    def remove_node(self, node: str) -> None:
        """Remove a physical node and all its virtual nodes."""
        self.nodes.discard(node)
        self.ring = [(h, n) for h, n in self.ring if n != node]

    def get_node(self, key: str) -> str:
        """Find the node responsible for a given key."""
        if not self.ring:
            raise ValueError("Empty ring")
        h = self._hash(key)
        idx = bisect_right(self.ring, (h,))
        if idx == len(self.ring):
            idx = 0  # wrap around
        return self.ring[idx][1]


# === Demonstration ===

if __name__ == "__main__":
    ring = ConsistentHashRing(num_virtual=150)
    for node in ["server-A", "server-B", "server-C"]:
        ring.add_node(node)

    # Assign 1000 keys and count distribution
    from collections import Counter
    counts = Counter(ring.get_node(f"key-{i}") for i in range(1000))
    print("Distribution with 3 nodes:")
    for node, count in sorted(counts.items()):
        print(f"  {node}: {count} keys")

    # Add a node and measure how many keys moved
    old_assignments = {f"key-{i}": ring.get_node(f"key-{i}") for i in range(1000)}
    ring.add_node("server-D")
    moved = sum(
        1 for k in old_assignments if ring.get_node(k) != old_assignments[k]
    )
    print(f"\nAfter adding server-D: {moved}/1000 keys moved")
    print(f"Expected: ~{1000 // 4} keys (n/(m+1) = 1000/4)")
```

## Complexity

| Operation | Time | Space |
|---|---|---|
| Add node | $O(v \log(vm))$ | $O(vm)$ total |
| Remove node | $O(vm)$ | $O(vm)$ total |
| Lookup key | $O(\log(vm))$ | -- |

where $v$ is the number of virtual nodes per physical node and $m$ is the number of physical nodes.

## Bounded-Load Consistent Hashing

Standard consistent hashing does not provide hard load guarantees.  **Bounded-load consistent hashing** (Mirrokni et al., 2018) adds a capacity constraint: each node may hold at most $(1 + \varepsilon) \cdot n/m$ keys for a tunable parameter $\varepsilon > 0$.

When a key's clockwise successor node is full, the key continues clockwise until it finds a node with available capacity.  This provides a worst-case load bound:

$$
\text{max load per node} \leq \left\lceil (1 + \varepsilon) \cdot \frac{n}{m} \right\rceil
$$

Google's load balancer uses this variant to distribute traffic across backend servers.

## Applications

Consistent hashing is fundamental to distributed systems:

- **Distributed caches** (Memcached, Redis Cluster): Partition cache keys across servers with minimal remapping during scaling events.
- **Distributed databases** (Amazon DynamoDB, Apache Cassandra): Partition data across storage nodes with automatic rebalancing.
- **Content delivery networks** (Akamai): Route requests to the nearest cache that holds the requested content.
- **Load balancers**: Distribute requests across backend servers while preserving session affinity.

## Reference

- Karger, D., Lehman, E., Leighton, T., Panigrahy, R., Levine, M., & Lewin, D. (1997). Consistent hashing and random trees. *Proceedings of the 29th ACM Symposium on Theory of Computing (STOC)*, 654--663.
- Mirrokni, V., Thorup, M., & Zadimoghaddam, M. (2018). Consistent hashing with bounded loads. *Proceedings of the 29th ACM-SIAM Symposium on Discrete Algorithms (SODA)*.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 11. MIT Press.
