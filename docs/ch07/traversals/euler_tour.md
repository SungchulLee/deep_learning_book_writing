# Euler Tour

The standard tree traversals -- preorder, inorder, postorder -- each visit every node exactly once.  The **Euler tour** takes a different approach: it traces the outline of the tree, visiting each node multiple times as the traversal enters and leaves subtrees.  This produces a linear sequence that encodes the entire tree structure, enabling powerful reductions of tree queries (such as lowest common ancestor and subtree sums) to array range queries, which can be answered in $O(1)$ time after $O(n)$ preprocessing.

## Definition

An Euler tour of a rooted tree with $n$ nodes produces a sequence of length $2n$ by performing a DFS and recording each node both when the traversal **enters** it and when it **exits** (returns from all its children).

Formally, define $\text{ET}[0 \ldots 2n-1]$ by the following recursive procedure applied to the root:

$$
\text{EulerTour}(v): \quad \text{record } v \;\text{(enter)}, \quad \text{recurse on each child of } v, \quad \text{record } v \;\text{(exit)}
$$

The resulting sequence contains each node exactly twice: once at its **entry time** $\text{tin}(v)$ and once at its **exit time** $\text{tout}(v)$.

??? example "Euler tour of a sample tree"
    Consider the tree:

    ```
           A
          / \
         B   C
        / \   \
       D   E   F
    ```

    The Euler tour visits: **A** B D D E E B **C** F F C **A**

    | Step | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
    |------|---|---|---|---|---|---|---|---|---|---|----|----|
    | Node | A | B | D | D | E | E | B | C | F | F | C  | A  |
    | Type | in| in| in|out| in|out|out| in| in|out|out |out |

    Entry/exit times: $\text{tin}(A)=0, \text{tout}(A)=11$; $\text{tin}(B)=1, \text{tout}(B)=6$; $\text{tin}(D)=2, \text{tout}(D)=3$.

## Key Properties

The entry and exit times encode the tree structure in useful ways:

**Ancestor test.** Node $u$ is an ancestor of node $v$ if and only if:

$$
\text{tin}(u) \leq \text{tin}(v) \leq \text{tout}(v) \leq \text{tout}(u)
$$

This is an $O(1)$ check after building the tour.

**Subtree range.** The subtree rooted at $v$ corresponds to the contiguous interval $[\text{tin}(v), \text{tout}(v)]$ in the Euler tour.  Any subtree query (sum, min, max, count) reduces to a range query on this interval.

**LCA reduction.** The lowest common ancestor of nodes $u$ and $v$ is the node with the minimum depth in the Euler tour between positions $\text{tin}(u)$ and $\text{tin}(v)$.  This reduces LCA to a range minimum query (RMQ), solvable in $O(1)$ per query after $O(n)$ preprocessing using a sparse table.

## Variants

### Full Euler Tour (2n entries)

Each node appears exactly twice.  Used for subtree queries by mapping the subtree of $v$ to the range $[\text{tin}(v), \text{tout}(v)]$.

### LCA Euler Tour (2n - 1 entries)

Record the node at every edge traversal (both downward and upward), including recording the parent when returning from a child.  This produces $2n - 1$ entries, and the LCA of $u$ and $v$ is the shallowest node between their first occurrences.

### Flat Euler Tour (n entries, entry only)

Record each node only on entry.  The resulting sequence is the preorder traversal.  Combined with subtree sizes, it can still answer subtree queries by mapping the subtree of $v$ to the range $[\text{tin}(v), \text{tin}(v) + \text{size}(v) - 1]$.

## Implementation

```python
"""Euler tour of a rooted tree with entry/exit times."""


# === Tree Node ===

class TreeNode:
    """A node in a rooted tree."""

    def __init__(self, val: int):
        self.val = val
        self.children: list["TreeNode"] = []


# === Euler Tour ===

def euler_tour(root: TreeNode) -> tuple[list[int], dict[int, int], dict[int, int]]:
    """
    Compute the Euler tour of a rooted tree.

    Returns:
        tour: list of node values in Euler tour order (length 2n)
        tin:  dict mapping node value to entry time
        tout: dict mapping node value to exit time
    """
    tour: list[int] = []
    tin: dict[int, int] = {}
    tout: dict[int, int] = {}
    timer = [0]

    def dfs(node: TreeNode) -> None:
        tin[node.val] = timer[0]
        tour.append(node.val)
        timer[0] += 1
        for child in node.children:
            dfs(child)
        tout[node.val] = timer[0]
        tour.append(node.val)
        timer[0] += 1

    dfs(root)
    return tour, tin, tout


def is_ancestor(u: int, v: int, tin: dict[int, int], tout: dict[int, int]) -> bool:
    """Check if u is an ancestor of v in O(1) using entry/exit times."""
    return tin[u] <= tin[v] and tout[v] <= tout[u]


# === Demonstration ===

if __name__ == "__main__":
    # Build the example tree:  A(0) -> B(1), C(2);  B -> D(3), E(4);  C -> F(5)
    nodes = [TreeNode(i) for i in range(6)]
    nodes[0].children = [nodes[1], nodes[2]]  # A -> B, C
    nodes[1].children = [nodes[3], nodes[4]]  # B -> D, E
    nodes[2].children = [nodes[5]]            # C -> F

    tour, tin, tout = euler_tour(nodes[0])
    labels = "ABCDEF"

    print("Euler tour:", " ".join(labels[v] for v in tour))
    print()
    for i, label in enumerate(labels):
        print(f"  {label}: tin={tin[i]}, tout={tout[i]}")

    print()
    print(f"Is A ancestor of D? {is_ancestor(0, 3, tin, tout)}")  # True
    print(f"Is B ancestor of F? {is_ancestor(1, 5, tin, tout)}")  # False
    print(f"Is C ancestor of F? {is_ancestor(2, 5, tin, tout)}")  # True
```

## Applications

| Application | Reduction | Query time |
|---|---|---|
| Subtree sum/min/max | Range query on $[\text{tin}(v), \text{tout}(v)]$ | $O(1)$ with sparse table |
| Subtree update | Range update on $[\text{tin}(v), \text{tout}(v)]$ | $O(\log n)$ with Fenwick tree |
| Lowest common ancestor | Range minimum query on depths | $O(1)$ after $O(n)$ preprocessing |
| Ancestor test | Compare entry/exit times | $O(1)$ |
| Subtree size | $\text{tout}(v) - \text{tin}(v) + 1)/2$ or direct count | $O(1)$ |

## Complexity

Building the Euler tour requires a single DFS traversal:

- **Time:** $O(n)$
- **Space:** $O(n)$ for the tour array and time stamps

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 12. MIT Press.
- Bender, M. A., & Farach-Colton, M. (2000). The LCA problem revisited. *Proceedings of LATIN 2000*, 88--94.
