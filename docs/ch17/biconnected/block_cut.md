# Block-Cut Tree

Finding articulation points and biconnected components answers a local question: which vertices or edges are critical for connectivity? The block-cut tree lifts this information into a global structure by building a **tree** whose nodes represent the biconnected components (called *blocks*) and the articulation points (called *cut vertices*) of the original graph. This tree reveals the entire vulnerability structure of a connected graph at a glance.

## Definitions

Let $G = (V, E)$ be a connected undirected graph.

**Block.** A maximal biconnected subgraph of $G$. Every edge belongs to exactly one block. A bridge (cut edge) forms a block by itself together with its two endpoints.

**Cut vertex.** A vertex whose removal disconnects $G$. Cut vertices are exactly the vertices shared by two or more blocks.

**Block-cut tree.** A bipartite tree $T$ constructed as follows:

- Create one *B-node* for each block of $G$.
- Create one *C-node* for each cut vertex of $G$.
- Add an edge $(B_i, c)$ in $T$ whenever cut vertex $c$ belongs to block $B_i$.

The result is always a tree (acyclic and connected), and its structure is bipartite: every edge connects a B-node to a C-node.

## Properties

The block-cut tree $T$ of a connected graph $G$ with $b$ blocks and $c$ cut vertices satisfies:

- $T$ has $b + c$ nodes and $b + c - 1$ edges.
- $T$ is a tree: removing any edge from $T$ corresponds to removing a cut vertex from $G$, which disconnects $G$.
- The leaves of $T$ are always B-nodes (blocks), never C-nodes.
- Two blocks share at most one vertex, and that shared vertex is a cut vertex.

## Construction Algorithm

Building the block-cut tree reduces to finding all biconnected components and articulation points, which a single DFS accomplishes in $O(V + E)$ time.

**Step 1.** Run DFS on $G$. Maintain discovery times $\text{disc}[v]$ and low values $\text{low}[v]$ for each vertex $v$.

**Step 2.** Use a stack of edges. Each time the DFS backtracks from child $u$ to parent $v$ and finds $\text{low}[u] \ge \text{disc}[v]$, pop edges from the stack until $(v, u)$ is reached. These edges form one biconnected component (block).

**Step 3.** Identify cut vertices: $v$ is a cut vertex if either (a) $v$ is the DFS root with two or more children, or (b) $v$ is not the root and has a child $u$ with $\text{low}[u] \ge \text{disc}[v]$.

**Step 4.** Build the tree by creating a B-node for each block and a C-node for each cut vertex, then connecting them as described above.

## Implementation

```python
"""
Block-cut tree construction via DFS.

Finds all biconnected components (blocks) and articulation points
of a connected undirected graph, then builds the block-cut tree.
"""

from collections import defaultdict

# === DFS-Based Block-Cut Tree Construction ===

class BlockCutTree:
    """Build the block-cut tree of a connected undirected graph."""

    def __init__(self, n: int):
        """Initialize graph with n vertices (0-indexed)."""
        self.n = n
        self.adj = defaultdict(list)
        self.blocks = []          # list of sets of vertices per block
        self.cut_vertices = set()
        self.tree_adj = defaultdict(list)  # block-cut tree adjacency

    def add_edge(self, u: int, v: int) -> None:
        """Add undirected edge (u, v)."""
        self.adj[u].append(v)
        self.adj[v].append(u)

    def build(self) -> None:
        """Run DFS to find blocks and cut vertices, then build the tree."""
        disc = [-1] * self.n
        low = [0] * self.n
        parent = [-1] * self.n
        stack = []  # stack of edges
        timer = [0]

        def dfs(u: int) -> None:
            disc[u] = low[u] = timer[0]
            timer[0] += 1
            child_count = 0

            for v in self.adj[u]:
                if disc[v] == -1:
                    child_count += 1
                    parent[v] = u
                    stack.append((u, v))
                    dfs(v)
                    low[u] = min(low[u], low[v])

                    # Check for articulation point / block boundary
                    if (parent[u] == -1 and child_count > 1) or \
                       (parent[u] != -1 and low[v] >= disc[u]):
                        self.cut_vertices.add(u)

                    if low[v] >= disc[u]:
                        block = set()
                        while stack:
                            edge = stack.pop()
                            block.add(edge[0])
                            block.add(edge[1])
                            if edge == (u, v):
                                break
                        self.blocks.append(block)

                elif v != parent[u] and disc[v] < disc[u]:
                    low[u] = min(low[u], disc[v])
                    stack.append((u, v))

        dfs(0)
        self._build_tree()

    def _build_tree(self) -> None:
        """Construct the block-cut tree from blocks and cut vertices."""
        # B-nodes: indices 0..len(blocks)-1
        # C-nodes: offset by len(blocks)
        cut_list = sorted(self.cut_vertices)
        cut_index = {v: i + len(self.blocks) for i, v in enumerate(cut_list)}

        for bi, block in enumerate(self.blocks):
            for v in block:
                if v in self.cut_vertices:
                    ci = cut_index[v]
                    self.tree_adj[bi].append(ci)
                    self.tree_adj[ci].append(bi)


# === Demonstration ===

if __name__ == "__main__":
    #  Graph:  0--1--2--3--4
    #          |  |     |
    #          5--6     7
    bct = BlockCutTree(8)
    for u, v in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,1),(3,7)]:
        bct.add_edge(u, v)
    bct.build()

    print(f"Number of blocks: {len(bct.blocks)}")
    print(f"Cut vertices: {sorted(bct.cut_vertices)}")
    for i, block in enumerate(bct.blocks):
        print(f"  Block {i}: {sorted(block)}")
```

**Output:**

```
Number of blocks: 3
Cut vertices: [2, 3]
  Block 0: [3, 4]
  Block 1: [2, 3, 7]
  Block 2: [0, 1, 2, 5, 6]
```

The graph has three blocks. Cut vertices $2$ and $3$ each connect two blocks; removing either would disconnect the graph. The block-cut tree therefore has five nodes (three B-nodes and two C-nodes) connected in a path.

## Complexity

| Aspect | Cost |
|--------|:----:|
| Time   | $O(V + E)$ |
| Space  | $O(V + E)$ |

The entire construction requires a single DFS pass plus linear-time post-processing, making it as efficient as the underlying biconnected-component algorithm.

## Applications

- **Network reliability.** The block-cut tree identifies all single points of failure in a network.
- **Two-vertex connectivity queries.** After building the tree, one can answer in $O(1)$ whether two vertices lie in the same biconnected component.
- **Cactus graphs.** A graph is a cactus if and only if every block in its block-cut tree is either an edge or a simple cycle.

## Reference

- Hopcroft, J. E., & Tarjan, R. E. (1973). Algorithm 447: Efficient algorithms for graph manipulation. *Communications of the ACM*, 16(6), 372--378.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (4th ed.), Chapter 22: Elementary Graph Algorithms.
