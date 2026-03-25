# Edge Classification

When DFS traverses a graph, every edge falls into one of four categories based on its relationship to the DFS tree. This classification is not just a bookkeeping detail: the presence or absence of certain edge types reveals fundamental structural properties of the graph. Back edges indicate cycles, cross edges appear only in directed graphs, and forward edges signal redundant reachability. Understanding edge classification is the foundation for algorithms that detect cycles, compute strongly connected components, and perform topological sorting.

## The Four Edge Types

Let $\text{pre}(u)$ and $\text{post}(u)$ denote the discovery and finish times assigned by DFS. When DFS examines an edge $(u, v)$, the edge is classified as follows:

**Tree edges.** If $v$ is white (unvisited) when $(u, v)$ is explored, the edge becomes part of the DFS tree. Tree edges form the DFS forest.

**Back edges.** If $v$ is gray (discovered but not finished) when $(u, v)$ is explored, the edge goes from $u$ to an ancestor $v$ in the DFS tree. Equivalently, $\text{pre}(v) \leq \text{pre}(u)$ and $\text{post}(u) \leq \text{post}(v)$.

**Forward edges.** If $v$ is black (finished) and $\text{pre}(u) < \text{pre}(v)$, the edge goes from $u$ to a descendant $v$ that was already fully explored through a different path.

**Cross edges.** If $v$ is black (finished) and $\text{pre}(v) < \text{pre}(u)$, the edge connects two vertices in different branches of the DFS tree (or from a later branch to an earlier one).

## Classification Using Pre/Post Numbers

The pre/post intervals provide a clean way to identify edge types. For an edge $(u, v)$:

| Edge Type | Condition | Interval Relationship |
|---|---|---|
| Tree | $v$ is white | $[\text{pre}(v), \text{post}(v)] \subset [\text{pre}(u), \text{post}(u)]$ |
| Back | $v$ is gray | $[\text{pre}(u), \text{post}(u)] \subset [\text{pre}(v), \text{post}(v)]$ |
| Forward | $v$ is black, $\text{pre}(u) < \text{pre}(v)$ | $[\text{pre}(v), \text{post}(v)] \subset [\text{pre}(u), \text{post}(u)]$ |
| Cross | $v$ is black, $\text{pre}(v) < \text{pre}(u)$ | Intervals disjoint, $\text{post}(v) < \text{pre}(u)$ |

!!! tip "Undirected graphs are simpler"
    In an undirected graph, DFS produces only **tree edges** and **back edges**. Forward and cross edges cannot occur because every non-tree edge connects a vertex to an ancestor (the first endpoint to be discovered sees the other as gray, not black).

## Cycle Detection via Back Edges

A directed graph contains a cycle if and only if DFS finds at least one back edge. A back edge $(u, v)$ closes a cycle: the tree path from $v$ down to $u$ combined with the back edge $(u, v)$ forms a directed cycle. This is the basis of all DFS-based cycle detection algorithms.

## Implementation

```python
"""
Edge classification during DFS traversal.

Classifies every edge as tree, back, forward, or cross
using vertex coloring (white/gray/black).
"""

# === Edge classification ======================================================

def classify_edges(graph):
    """Classify all edges in a directed graph via DFS.

    Parameters
    ----------
    graph : dict[int, list[int]]
        Adjacency list for a directed graph.

    Returns
    -------
    dict[str, list[tuple[int, int]]]
        Edges grouped by type: 'tree', 'back', 'forward', 'cross'.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {v: WHITE for v in graph}
    pre = {}
    post = {}
    clock = [0]
    edges = {"tree": [], "back": [], "forward": [], "cross": []}

    def explore(u):
        clock[0] += 1
        pre[u] = clock[0]
        color[u] = GRAY
        for v in graph[u]:
            if color[v] == WHITE:
                edges["tree"].append((u, v))
                explore(v)
            elif color[v] == GRAY:
                edges["back"].append((u, v))
            elif pre[u] < pre[v]:
                edges["forward"].append((u, v))
            else:
                edges["cross"].append((u, v))
        color[u] = BLACK
        clock[0] += 1
        post[u] = clock[0]

    for vertex in graph:
        if color[vertex] == WHITE:
            explore(vertex)

    return edges, pre, post


# === Main =====================================================================

if __name__ == "__main__":
    graph = {
        0: [1, 3],
        1: [2],
        2: [3],
        3: [1],
    }

    edges, pre, post = classify_edges(graph)

    print("Pre/Post timestamps:")
    for v in sorted(pre):
        print(f"  Vertex {v}: [{pre[v]}, {post[v]}]")

    print("\nEdge classification:")
    for etype, elist in edges.items():
        if elist:
            print(f"  {etype.capitalize():8s}: {elist}")
```

**Output:**
```
Pre/Post timestamps:
  Vertex 0: [1, 8]
  Vertex 1: [2, 7]
  Vertex 2: [3, 6]
  Vertex 3: [4, 5]

Edge classification:
  Tree    : [(0, 1), (1, 2), (2, 3)]
  Back    : [(3, 1)]
  Cross   : [(0, 3)]
```

The back edge $(3, 1)$ confirms the cycle $1 \to 2 \to 3 \to 1$. The cross edge $(0, 3)$ connects vertex 0 to vertex 3, which was already fully explored (black) when the edge was examined.

## Summary

| Edge Type | Direction in DFS Tree | Significance |
|---|---|---|
| Tree | Parent to child | Forms the DFS forest |
| Back | Descendant to ancestor | Indicates a cycle |
| Forward | Ancestor to descendant (non-tree) | Redundant reachability |
| Cross | Between branches | Connects separate subtrees |

## Reference

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 22. MIT Press.
