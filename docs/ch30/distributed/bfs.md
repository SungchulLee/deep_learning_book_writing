# Distributed BFS

In a distributed network, each node is a processor that communicates only
with its direct neighbors via message passing.  No single processor has a
global view of the graph.  Distributed BFS constructs a breadth-first
spanning tree from a designated source, enabling shortest-path routing and
forming the backbone for many other distributed algorithms.

## Model

We assume the **synchronous message-passing model**:

- The network is an undirected connected graph $G = (V, E)$ with $n = |V|$
  nodes and $m = |E|$ edges.
- Each node has a unique ID and knows only its neighbors.
- Computation proceeds in synchronous **rounds**: in each round, every node
  can send messages to its neighbors, receive messages, and perform local
  computation.
- A designated **source** node $s$ initiates the BFS.

## Algorithm (Synchronous)

The algorithm mirrors sequential BFS but executes one level per round.

**Initialization.**  The source $s$ sets its distance $d_s = 0$ and its
parent $\text{parent}_s = \text{nil}$.

**Round $i$ (for $i = 0, 1, 2, \dots$):**

1. Every node $v$ that set $d_v = i$ in the previous round sends a
   `SEARCH` message to all its neighbors.
2. Every node $u$ that has not yet been visited and receives a `SEARCH`
   message sets $d_u = i + 1$ and $\text{parent}_u = v$ (for the first
   `SEARCH` received).  It then marks itself as visited.

The algorithm terminates when a round produces no new visited nodes.

## Complexity

| Metric | Bound |
|---|---|
| Rounds (time) | $O(D)$ where $D$ is the diameter of $G$ |
| Messages | $O(m)$ total across all rounds |
| Space per node | $O(\deg(v))$ |

Each edge carries at most two messages (one from each endpoint), giving
the $O(m)$ message bound.

## Asynchronous Variant

In an asynchronous network, messages may have arbitrary (but finite) delays.
The synchronous algorithm does not directly apply because nodes cannot
determine when a round ends.

The **asynchronous BFS** algorithm by Awerbuch uses an acknowledgment-based
approach:

1. The source sends `SEARCH` messages.
2. A node $u$ receiving its first `SEARCH` sets its parent and forwards
   `SEARCH` to all neighbors, then waits for acknowledgments.
3. Acknowledgments propagate back to the source, which initiates the next
   level.

This achieves $O(D)$ time complexity (measured in maximum message delay
units) and $O(m + n D)$ messages in the worst case.

!!! note "Layered vs. Flooding"
    A simple flooding approach broadcasts from the source; each node
    forwards the first message it receives.  This finds a BFS tree in
    $O(D)$ time with $O(m)$ messages but only in the synchronous model.
    Asynchronous flooding may not produce a correct BFS tree (it yields a
    shortest-path tree only if all link delays are equal).

## Simulation

```python
"""
Simulation of synchronous distributed BFS.

Each node is modeled as a processor with a local state.
Messages are exchanged in synchronous rounds.
"""


# === Node Processor ===
class Node:
    """Represents a processor in the distributed network."""

    def __init__(self, node_id: int, neighbors: list[int]):
        self.node_id = node_id
        self.neighbors = neighbors
        self.distance = -1
        self.parent = -1
        self.visited = False
        self.inbox: list[tuple[int, int]] = []  # (sender, distance)

    def receive(self, sender: int, dist: int) -> None:
        """Receive a SEARCH message."""
        self.inbox.append((sender, dist))

    def process_round(self) -> list[tuple[int, int]]:
        """Process messages and return outgoing (neighbor, distance) pairs."""
        outgoing = []
        if not self.visited and self.inbox:
            sender, dist = self.inbox[0]
            self.distance = dist + 1
            self.parent = sender
            self.visited = True
            for nb in self.neighbors:
                outgoing.append((nb, self.distance))
        self.inbox.clear()
        return outgoing


# === Distributed BFS Simulator ===
def distributed_bfs(adj: dict[int, list[int]], source: int) -> dict[int, int]:
    """Simulate synchronous distributed BFS. Return {node: distance}."""
    nodes = {v: Node(v, neighbors) for v, neighbors in adj.items()}

    # Initialize source
    nodes[source].distance = 0
    nodes[source].visited = True

    # Round 0: source sends to neighbors
    messages = [(nb, source, 0) for nb in adj[source]]

    round_num = 0
    while messages:
        round_num += 1
        # Deliver messages
        for dest, sender, dist in messages:
            nodes[dest].receive(sender, dist)

        # Process round and collect new messages
        new_messages = []
        for node in nodes.values():
            outgoing = node.process_round()
            for nb, dist in outgoing:
                new_messages.append((nb, node.node_id, dist))
        messages = new_messages

    return {v: node.distance for v, node in nodes.items()}


# === Example ===
if __name__ == "__main__":
    graph = {
        0: [1, 2],
        1: [0, 3],
        2: [0, 3, 4],
        3: [1, 2],
        4: [2],
    }
    distances = distributed_bfs(graph, source=0)
    print("Distributed BFS distances from node 0:")
    for node, dist in sorted(distances.items()):
        print(f"  Node {node}: distance {dist}")
```

## Applications

- **Routing tables.**  BFS trees provide shortest-path routing in
  unweighted networks.
- **Topology discovery.**  Distributed BFS reveals the network structure
  without centralized knowledge.
- **Building block.**  Many distributed algorithms (leader election,
  spanning tree construction) use BFS as a subroutine.

## Reference

- Lynch, N. *Distributed Algorithms*. Morgan Kaufmann, 1996.
- Peleg, D. *Distributed Computing: A Locality-Sensitive Approach*.
  SIAM, 2000.
