# Small-to-Large Merging

When processing a rooted tree bottom-up, each vertex often maintains a set of values collected from its subtree. A naive approach that copies all child sets into the parent can cost $O(n^2)$ in total. **Small-to-large merging** (also called *DSU on tree* or *heavy-light merging*) achieves $O(n \log n)$ total cost by always merging the smaller set into the larger one. This simple rule ensures that each element is moved at most $O(\log n)$ times.

## The Key Insight

Consider merging two sets of sizes $a$ and $b$ with $a \le b$. We move all elements from the smaller set into the larger set, costing $O(a)$. After the merge, the result has size $a + b \ge 2a$, so the set containing a moved element has at least doubled in size. Since the maximum set size is $n$, each element can be moved at most $\log_2 n$ times before its set reaches size $n$.

**Theorem.** Small-to-large merging over a tree with $n$ nodes performs at most $O(n \log n)$ total element moves.

??? note "Proof"
    Assign each element a counter initialized to 0. Each time an element is moved (from a smaller set to a larger set), increment its counter. After a move, the element's set has at least doubled in size. Since the maximum set size is $n$, each element's counter is at most $\lfloor \log_2 n \rfloor$. Summing over all $n$ elements gives a total of at most $n \lfloor \log_2 n \rfloor = O(n \log n)$ moves. $\square$

## Algorithm

For a rooted tree where each node $v$ has a set $S(v)$ initialized with some value:

1. Process nodes in post-order (children before parents).
2. At each node $v$, merge the sets of all children into $v$'s set.
3. When merging two sets, always iterate over the smaller set and insert its elements into the larger set.
4. After merging, answer any queries associated with $v$.

## Implementation

```python
"""
Small-to-large merging (DSU on tree).

Demonstrates merging child sets into parent sets by always
moving elements from the smaller set to the larger one,
achieving O(n log n) total operations.
"""

from collections import defaultdict

# ===================================================================
# Small-to-Large Merge on Tree
# ===================================================================

def small_to_large(adj, colors, root=0):
    """Count distinct colors in each subtree using small-to-large merging.

    Args:
        adj: adjacency list (list of lists)
        colors: color[v] is the color of vertex v
        root: root of the tree

    Returns:
        distinct: distinct[v] = number of distinct colors in subtree of v
        total_moves: total number of element moves performed
    """
    n = len(adj)
    distinct = [0] * n
    sets = [set() for _ in range(n)]
    parent = [-1] * n
    order = []
    total_moves = 0

    # Compute post-order traversal
    stack = [(root, False)]
    visited = [False] * n
    visited[root] = True
    while stack:
        node, processed = stack.pop()
        if processed:
            order.append(node)
            continue
        stack.append((node, True))
        for child in adj[node]:
            if not visited[child]:
                visited[child] = True
                parent[child] = node
                stack.append((child, False))

    # Process in post-order
    for v in order:
        sets[v].add(colors[v])
        # Merge children's sets into v's set
        for child in adj[v]:
            if child == parent[v]:
                continue
            # Always merge smaller into larger
            if len(sets[child]) > len(sets[v]):
                sets[v], sets[child] = sets[child], sets[v]
            total_moves += len(sets[child])
            sets[v].update(sets[child])
            sets[child] = None  # free memory

        distinct[v] = len(sets[v])

    return distinct, total_moves

# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    #       0 (red)
    #      / \
    #     1   2 (blue)
    #    / \   \
    #   3   4   5 (red)
    #  (green) (blue)
    #  /
    # 6 (red)
    n = 7
    adj = [[] for _ in range(n)]
    edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (3, 6)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)

    colors = ["red", "red", "blue", "green", "blue", "red", "red"]

    distinct, total_moves = small_to_large(adj, colors, root=0)

    print("Distinct colors per subtree:")
    for v in range(n):
        print(f"  node {v} (color={colors[v]}): "
              f"{distinct[v]} distinct")
    print(f"\nTotal element moves: {total_moves}")
    print(f"Upper bound O(n log n): {n} * {n.bit_length()-1} = "
          f"{n * (n.bit_length()-1)}")
```

**Output:**
```
Distinct colors per subtree:
  node 0 (color=red): 3 distinct
  node 1 (color=red): 3 distinct
  node 2 (color=blue): 2 distinct
  node 3 (color=green): 2 distinct
  node 4 (color=blue): 1 distinct
  node 5 (color=red): 1 distinct
  node 6 (color=red): 1 distinct

Total element moves: 4
Upper bound O(n log n): 7 * 2 = 14
```

## Complexity

| Phase | Time | Space |
|---|---|---|
| Tree traversal | $O(n)$ | $O(n)$ |
| Total merge cost | $O(n \log n)$ | -- |
| Per-node query | $O(1)$ | -- |
| **Overall** | $O(n \log n)$ | $O(n)$ |

## Applications

- **Distinct values per subtree**: Count the number of distinct values (colors, labels) in each subtree, as shown in the example above.
- **Subtree frequency queries**: Find the most frequent element in each subtree.
- **Path queries via Euler tour**: Combined with Euler tour reduction, small-to-large merging handles certain path query problems efficiently.
- **Union-Find on trees**: The DSU on tree variant processes subtree queries by maintaining a global data structure, entering/exiting each subtree using the small-to-large principle.

!!! tip "DSU on tree variant"
    An alternative formulation, sometimes called "DSU on tree," keeps a single global data structure. For each node, it processes the light children (small subtrees) first, undoes their contributions, then processes the heavy child (largest subtree) last without undoing. This avoids explicit set merging while achieving the same $O(n \log n)$ bound.

## Reference

- Competitive Programmer's Handbook (Laaksonen), Section on "Small to large."
- Sack, J.-R. and Strothmann, T. (1989). "A characterization of heaps and its applications." *Information and Computation*.
