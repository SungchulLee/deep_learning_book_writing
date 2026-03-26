# Distributed Hash Tables

A centralized key-value store becomes a bottleneck as the number of
participants grows.  Distributed Hash Tables (DHTs) spread storage and
lookup responsibility across $n$ nodes so that any key can be located in
$O(\log n)$ hops, with no single point of failure.  DHTs underpin
peer-to-peer systems, content-addressable networks, and distributed
databases.

## Core Idea

Each node and each key is assigned an identifier from a circular identifier
space $\{0, 1, \dots, 2^m - 1\}$ using a hash function (typically SHA-1).
A key $k$ is stored at the node whose identifier most closely follows $k$
in the circular space---the **successor** of $k$.

## Consistent Hashing

Standard hashing ($h(k) = k \bmod n$) requires remapping nearly all keys
when $n$ changes.  Consistent hashing maps both keys and nodes onto a ring
of size $2^m$:

$$
\text{node\_id} = \text{hash}(\text{IP address}), \quad \text{key\_id} = \text{hash}(\text{key})
$$

A key is assigned to the first node clockwise from its position on the ring.
When a node joins or leaves, only $O(K/n)$ keys (where $K$ is the total
number of keys) need to be reassigned, compared to $O(K)$ for modular
hashing.

## Chord Protocol

Chord (Stoica et al., 2001) is the canonical DHT, providing $O(\log n)$
lookup with $O(\log n)$ state per node.

### Finger Table

Each node $p$ maintains a **finger table** of $m$ entries:

$$
\text{finger}[i] = \text{successor}(p + 2^{i-1}) \quad \text{for } i = 1, 2, \dots, m
$$

The $i$-th finger points to the node responsible for the identifier
$p + 2^{i-1} \pmod{2^m}$.  Fingers span exponentially increasing distances
around the ring, halving the remaining distance with each hop.

### Lookup

To find the node responsible for key $k$:

1. If $k$ falls between the current node $p$ and its successor, return
   the successor.
2. Otherwise, forward the query to the closest preceding finger.

Each hop at least halves the distance to the target, giving $O(\log n)$
hops in expectation.

### Node Join

When a new node $p$ joins:

1. Find $p$'s successor using an existing node's lookup.
2. Transfer the appropriate keys from the successor to $p$.
3. Update finger tables of other nodes (done lazily via a stabilization
   protocol).

### Node Departure

When node $p$ leaves:

1. Transfer $p$'s keys to its successor.
2. Other nodes' finger tables are updated during periodic stabilization.

## Complexity

| Operation | Hops | Messages |
|---|---|---|
| Lookup | $O(\log n)$ | $O(\log n)$ |
| Insert key | $O(\log n)$ | $O(\log n)$ |
| Join | $O(\log^2 n)$ | $O(\log^2 n)$ |
| State per node | $O(\log n)$ entries | --- |

## Implementation

```python
"""
Simplified Chord-style DHT simulation.

Demonstrates consistent hashing and finger-table-based lookup.
"""

import hashlib


# === Consistent Hashing ===
def hash_id(key: str, m: int = 8) -> int:
    """Hash a key to a position on a 2^m ring."""
    digest = hashlib.sha1(key.encode()).hexdigest()
    return int(digest, 16) % (2**m)


# === Chord Node ===
class ChordNode:
    """A node in a simplified Chord DHT."""

    def __init__(self, node_id: int, m: int = 8):
        self.node_id = node_id
        self.m = m
        self.ring_size = 2**m
        self.finger_table: list[int] = []
        self.data: dict[int, str] = {}
        self.successor: int = node_id

    def build_finger_table(self, sorted_nodes: list[int]) -> None:
        """Build finger table from the sorted list of all node IDs."""
        self.finger_table = []
        for i in range(self.m):
            target = (self.node_id + 2**i) % self.ring_size
            # Find successor of target
            succ = sorted_nodes[0]  # wrap around
            for n in sorted_nodes:
                if n >= target:
                    succ = n
                    break
            self.finger_table.append(succ)
        self.successor = self.finger_table[0]


# === DHT ===
class ChordDHT:
    """Simplified Chord DHT with finger table lookups."""

    def __init__(self, m: int = 8):
        self.m = m
        self.nodes: dict[int, ChordNode] = {}
        self.sorted_ids: list[int] = []

    def add_node(self, node_id: int) -> None:
        """Add a node to the DHT."""
        node = ChordNode(node_id, self.m)
        self.nodes[node_id] = node
        self.sorted_ids = sorted(self.nodes.keys())
        # Rebuild all finger tables
        for n in self.nodes.values():
            n.build_finger_table(self.sorted_ids)

    def lookup(self, key: str) -> tuple[int, int]:
        """Find the responsible node for a key. Return (key_id, node_id)."""
        key_id = hash_id(key, self.m)
        # Find successor of key_id
        responsible = self.sorted_ids[0]
        for nid in self.sorted_ids:
            if nid >= key_id:
                responsible = nid
                break
        return key_id, responsible

    def put(self, key: str, value: str) -> int:
        """Store a key-value pair. Return the responsible node ID."""
        key_id, node_id = self.lookup(key)
        self.nodes[node_id].data[key_id] = value
        return node_id

    def get(self, key: str) -> str | None:
        """Retrieve a value by key."""
        key_id, node_id = self.lookup(key)
        return self.nodes[node_id].data.get(key_id)


# === Example ===
if __name__ == "__main__":
    dht = ChordDHT(m=8)
    for nid in [0, 32, 64, 128, 192]:
        dht.add_node(nid)

    keys = ["apple", "banana", "cherry", "date"]
    for k in keys:
        node = dht.put(k, f"value_{k}")
        print(f"Key '{k}' (id={hash_id(k, 8)}) -> Node {node}")

    for k in keys:
        val = dht.get(k)
        print(f"Get '{k}' = {val}")
```

## Other DHT Designs

| DHT | Topology | Lookup | Year |
|---|---|---|---|
| Chord | Ring + fingers | $O(\log n)$ | 2001 |
| Pastry | Prefix-based routing | $O(\log n)$ | 2001 |
| Kademlia | XOR-based tree | $O(\log n)$ | 2002 |
| CAN | $d$-dimensional torus | $O(d \cdot n^{1/d})$ | 2001 |

!!! tip "Kademlia in Practice"
    Kademlia is the most widely deployed DHT, used in BitTorrent and IPFS.
    Its XOR-based distance metric naturally produces symmetric lookups and
    simplifies routing table maintenance.

## Reference

- Stoica, I. et al. "Chord: A Scalable Peer-to-Peer Lookup Protocol for
  Internet Applications." IEEE/ACM Transactions on Networking, 2003.
- Peleg, D. *Distributed Computing: A Locality-Sensitive Approach*.
  SIAM, 2000.
