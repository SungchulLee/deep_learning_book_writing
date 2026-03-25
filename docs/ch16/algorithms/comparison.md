# MST Algorithm Comparison

Kruskal's, Prim's, and Boruvka's algorithms all solve the same problem -- finding a minimum spanning tree -- but they differ in strategy, data structure requirements, and performance characteristics. Choosing the right algorithm depends on the graph's density, representation, and whether parallelism is available. This page consolidates the trade-offs to guide algorithm selection.

## Complexity Summary

| Algorithm | Time | Space | Data Structure |
|-----------|------|-------|----------------|
| Kruskal's | $O(E \log E)$ | $O(V + E)$ | Union-Find |
| Prim's (array) | $O(V^2)$ | $O(V)$ | Array |
| Prim's (binary heap) | $O(E \log V)$ | $O(V + E)$ | Binary heap |
| Prim's (Fibonacci heap) | $O(E + V \log V)$ | $O(V + E)$ | Fibonacci heap |
| Boruvka's | $O(E \log V)$ | $O(V + E)$ | Union-Find |

Since $\log E = O(\log V)$ for simple graphs (because $E \le V^2$), Kruskal's $O(E \log E)$ and $O(E \log V)$ are asymptotically equivalent.

## Strategy Comparison

| Aspect | Kruskal's | Prim's | Boruvka's |
|--------|-----------|--------|-----------|
| Growth model | Edge-centric: merge forest | Vertex-centric: grow one tree | Component-centric: all grow simultaneously |
| Edge processing | Global sorted order | Local neighbors of tree | All edges per round |
| Number of passes | 1 pass over sorted edges | $V$ extractions | $O(\log V)$ rounds |
| Greedy choice | Lightest edge not forming a cycle | Lightest edge leaving the tree | Lightest edge leaving each component |
| Theoretical basis | Cut property (forest components) | Cut property (tree vs. non-tree) | Cut property (all components) |

## Performance by Graph Density

The best algorithm depends on the relationship between $|E|$ and $|V|$:

**Sparse graphs** ($E = O(V)$ or $E = O(V \log V)$):

- Kruskal's: $O(V \log V)$ -- fast because sorting few edges is cheap.
- Prim's (binary heap): $O(V \log^2 V)$ or $O(V \log V)$ -- competitive.
- **Recommendation**: Kruskal's or Prim's with binary heap.

**Medium-density graphs** ($E = \Theta(V^{1.5})$):

- Kruskal's: $O(V^{1.5} \log V)$.
- Prim's (Fibonacci heap): $O(V^{1.5} + V \log V) = O(V^{1.5})$.
- **Recommendation**: Prim's with Fibonacci heap has the best asymptotic bound.

**Dense graphs** ($E = \Theta(V^2)$):

- Kruskal's: $O(V^2 \log V)$ -- dominated by sorting.
- Prim's (array): $O(V^2)$ -- optimal for this density.
- Prim's (Fibonacci heap): $O(V^2 + V \log V) = O(V^2)$ -- same as array but higher constants.
- **Recommendation**: Prim's with simple array implementation.

## Practical Considerations

Beyond asymptotic complexity, several practical factors influence the choice:

### Input format
- **Edge list**: Kruskal's is natural (sort and iterate). Prim's requires building an adjacency list first.
- **Adjacency list/matrix**: Prim's works directly. Kruskal's requires extracting all edges.

### Implementation simplicity
- Kruskal's with Union-Find is straightforward to implement correctly.
- Prim's with a binary heap requires careful DECREASE-KEY handling (or lazy deletion).
- Fibonacci heaps are complex to implement and rarely justify their theoretical advantage.
- Boruvka's requires careful deduplication of edges selected by multiple components.

### Parallelism
- **Boruvka's** is the most parallelizable: each round processes all edges independently.
- **Kruskal's** is inherently sequential (edges must be processed in sorted order).
- **Prim's** is inherently sequential (each step depends on the previous extraction).

### Pre-sorted edges
- If edges arrive pre-sorted (e.g., from a database index), Kruskal's runs in $O(E \cdot \alpha(V))$, which is nearly linear.

## Decision Guide

The following flowchart summarizes the selection process:

1. **Is the graph dense** ($E = \Theta(V^2)$)? Use **Prim's with array** -- $O(V^2)$.
2. **Are edges pre-sorted?** Use **Kruskal's** -- nearly $O(E)$.
3. **Is parallelism available?** Use **Boruvka's** -- $O((E \log V) / p)$ with $p$ processors.
4. **Is implementation simplicity paramount?** Use **Kruskal's with Union-Find**.
5. **Otherwise**: use **Prim's with binary heap** -- $O(E \log V)$ with good cache behavior.

## Lower Bound

The MST problem has a known lower bound of $\Omega(E)$ in the comparison model (every edge must be examined). The best known deterministic algorithm achieves $O(E \cdot \alpha(V))$ using a combination of Boruvka phases and edge contraction (Chazelle, 2000). A randomized algorithm by Karger, Klein, and Tarjan (1995) achieves expected $O(E)$ time.

## Reference

- [Introduction to Algorithms (CLRS), Chapter 23](https://mitpress.mit.edu/books/introduction-algorithms-fourth-edition)
- Chazelle, B. (2000). A minimum spanning tree algorithm with inverse-Ackermann type complexity. *JACM*, 47(6), 1028--1047.
- Karger, D. R., Klein, P. N., & Tarjan, R. E. (1995). A randomized linear-time algorithm to find minimum spanning trees. *JACM*, 42(2), 321--328.
