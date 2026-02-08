# 29.1.2 Graph Representations

## Introduction

The choice of graph representation profoundly affects both the efficiency and expressiveness of graph algorithms and neural network models. This section covers the principal data structures for representing graphs, their mathematical properties, and computational trade-offs.

## Adjacency List

The **adjacency list** representation stores, for each node, a list of its neighbors. For weighted graphs, each neighbor entry also includes the edge weight.

$$\text{adj}[u] = \{(v, w_{uv}) : (u, v) \in E\}$$

**Complexity**:
- Space: $O(|V| + |E|)$
- Check if edge exists: $O(d(u))$ where $d(u)$ is the degree of $u$
- Iterate over neighbors: $O(d(u))$

Adjacency lists are the preferred representation for **sparse graphs** where $|E| \ll |V|^2$.

## Adjacency Matrix

The **adjacency matrix** $A \in \mathbb{R}^{|V| \times |V|}$ is defined as:

$$A_{ij} = \begin{cases} w_{ij} & \text{if } (i, j) \in E \\ 0 & \text{otherwise} \end{cases}$$

For unweighted graphs, $A_{ij} \in \{0, 1\}$. For undirected graphs, $A$ is symmetric: $A = A^T$.

**Properties of the adjacency matrix**:
- $A^k_{ij}$ counts the number of walks of length $k$ from node $i$ to node $j$
- The eigenvalues of $A$ form the **spectrum** of the graph
- The trace $\text{tr}(A^k)$ counts closed walks of length $k$

**Complexity**:
- Space: $O(|V|^2)$
- Check if edge exists: $O(1)$
- Iterate over neighbors: $O(|V|)$

## Sparse Matrix Representations

For large sparse graphs, sparse matrix formats save memory and computation.

### COO (Coordinate) Format
Stores edges as three arrays: row indices, column indices, and values.

$$\text{COO} = (\text{row}[], \text{col}[], \text{val}[])$$

This is the format used by **PyTorch Geometric** for `edge_index`.

### CSR (Compressed Sparse Row) Format
Stores the adjacency in a compressed row format using three arrays:
- `indptr`: Row pointers (length $|V| + 1$)
- `indices`: Column indices of non-zero entries
- `data`: Values of non-zero entries

**Complexity**: Space $O(|V| + |E|)$, row slicing $O(d(u))$.

### CSC (Compressed Sparse Column) Format
Similar to CSR but compressed along columns. Efficient for column-based operations.

## Edge List

The **edge list** representation stores edges as a list of tuples:

$$\text{edges} = [(u_1, v_1, w_1), (u_2, v_2, w_2), \ldots, (u_m, v_m, w_m)]$$

**Complexity**: Space $O(|E|)$. Efficient for iterating over all edges.

## Incidence Matrix

The **incidence matrix** $B \in \mathbb{R}^{|V| \times |E|}$ relates nodes to edges:

For undirected graphs:
$$B_{ve} = \begin{cases} 1 & \text{if node } v \text{ is an endpoint of edge } e \\ 0 & \text{otherwise} \end{cases}$$

For directed graphs:
$$B_{ve} = \begin{cases} +1 & \text{if edge } e \text{ leaves node } v \\ -1 & \text{if edge } e \text{ enters node } v \\ 0 & \text{otherwise} \end{cases}$$

The **graph Laplacian** can be expressed as $L = B B^T$ for undirected graphs.

## PyTorch Geometric Representation

PyTorch Geometric (PyG) uses a specific representation optimized for GPU computation:

- **`edge_index`**: A `[2, num_edges]` tensor in COO format. `edge_index[0]` contains source nodes and `edge_index[1]` contains target nodes.
- **`x`**: Node feature matrix of shape `[num_nodes, num_node_features]`
- **`edge_attr`**: Edge feature matrix of shape `[num_edges, num_edge_features]`

This representation is batching-friendly and supports efficient message passing on GPUs.

## Comparison of Representations

| Representation | Space | Edge Lookup | Neighbor Iteration | Best For |
|----------------|-------|-------------|-------------------|----------|
| Adjacency List | $O(V + E)$ | $O(d)$ | $O(d)$ | Sparse graphs |
| Adjacency Matrix | $O(V^2)$ | $O(1)$ | $O(V)$ | Dense, spectral methods |
| Sparse COO | $O(E)$ | $O(E)$ | $O(E)$ | GPU computation, PyG |
| Sparse CSR | $O(V + E)$ | $O(\log d)$ | $O(d)$ | Row-based operations |
| Edge List | $O(E)$ | $O(E)$ | $O(E)$ | Edge-centric algorithms |
| Incidence Matrix | $O(V \cdot E)$ | $O(1)$ | $O(E)$ | Theoretical analysis |

## Conversion Between Representations

Converting between representations is a common operation:

- **Adjacency List → Matrix**: Fill matrix entries from neighbor lists, $O(|V| + |E|)$
- **Matrix → Edge List**: Iterate non-zero entries, $O(|V|^2)$ for dense, $O(|E|)$ for sparse
- **Edge List → COO**: Direct mapping of source and target arrays

## Quantitative Finance Considerations

In financial applications, the choice of representation matters:

- **Correlation matrices** are naturally dense → adjacency matrix or numpy/scipy operations
- **Transaction networks** are typically sparse → adjacency list or COO format
- **Order book graphs** change dynamically → efficient update operations favor adjacency lists
- **Large-scale interbank networks** → sparse formats for scalability

## Summary

Graph representation is not merely a data structure choice but a computational design decision that impacts model performance, memory usage, and algorithmic capabilities. Modern GNN frameworks like PyTorch Geometric use COO-based sparse representations to balance flexibility and GPU efficiency.
