"""
Chapter 29.1.2: Graph Representations
Different ways to represent graphs with conversions between them.
"""

import numpy as np
import scipy.sparse as sp
import torch
from typing import List, Tuple, Dict, Optional


# ============================================================
# 1. Adjacency List Representation
# ============================================================

class AdjacencyList:
    """Graph represented as adjacency list."""

    def __init__(self, num_nodes: int, directed: bool = False):
        self.num_nodes = num_nodes
        self.directed = directed
        self.adj: Dict[int, List[Tuple[int, float]]] = {
            i: [] for i in range(num_nodes)
        }

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))

    def neighbors(self, node: int) -> List[Tuple[int, float]]:
        return self.adj[node]

    def to_adjacency_matrix(self) -> np.ndarray:
        """Convert to dense adjacency matrix."""
        A = np.zeros((self.num_nodes, self.num_nodes))
        for u in range(self.num_nodes):
            for v, w in self.adj[u]:
                A[u, v] = w
        return A

    def to_edge_list(self) -> List[Tuple[int, int, float]]:
        """Convert to edge list."""
        edges = []
        seen = set()
        for u in range(self.num_nodes):
            for v, w in self.adj[u]:
                edge_key = (u, v) if self.directed else (min(u, v), max(u, v))
                if edge_key not in seen:
                    edges.append((u, v, w))
                    seen.add(edge_key)
        return edges

    def to_coo(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to COO format (row, col, val)."""
        rows, cols, vals = [], [], []
        for u in range(self.num_nodes):
            for v, w in self.adj[u]:
                rows.append(u)
                cols.append(v)
                vals.append(w)
        return np.array(rows), np.array(cols), np.array(vals)

    def __repr__(self):
        num_edges = sum(len(v) for v in self.adj.values())
        if not self.directed:
            num_edges //= 2
        return f"AdjacencyList(nodes={self.num_nodes}, edges={num_edges})"


# ============================================================
# 2. Adjacency Matrix Representation
# ============================================================

class AdjacencyMatrix:
    """Graph represented as dense adjacency matrix."""

    def __init__(self, matrix: np.ndarray, directed: bool = False):
        assert matrix.shape[0] == matrix.shape[1], "Matrix must be square"
        self.matrix = matrix.copy()
        self.directed = directed
        self.num_nodes = matrix.shape[0]

    @classmethod
    def from_edges(cls, num_nodes: int, edges: List[Tuple[int, int]],
                   weights: Optional[List[float]] = None,
                   directed: bool = False) -> 'AdjacencyMatrix':
        A = np.zeros((num_nodes, num_nodes))
        for idx, (u, v) in enumerate(edges):
            w = weights[idx] if weights else 1.0
            A[u, v] = w
            if not directed:
                A[v, u] = w
        return cls(A, directed)

    def neighbors(self, node: int) -> List[int]:
        return list(np.nonzero(self.matrix[node])[0])

    def degree(self, node: int) -> int:
        return int(np.sum(self.matrix[node] != 0))

    def degree_matrix(self) -> np.ndarray:
        """Return the degree matrix D."""
        degrees = np.sum(self.matrix != 0, axis=1)
        return np.diag(degrees)

    def walks_of_length_k(self, k: int) -> np.ndarray:
        """Compute A^k: matrix where (i,j) counts walks of length k."""
        return np.linalg.matrix_power(self.matrix, k)

    def spectrum(self) -> np.ndarray:
        """Return eigenvalues of the adjacency matrix."""
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        return np.sort(eigenvalues)[::-1]

    def to_sparse_coo(self) -> sp.coo_matrix:
        return sp.coo_matrix(self.matrix)

    def to_sparse_csr(self) -> sp.csr_matrix:
        return sp.csr_matrix(self.matrix)

    def to_adjacency_list(self) -> AdjacencyList:
        adj_list = AdjacencyList(self.num_nodes, self.directed)
        seen = set()
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.matrix[i, j] != 0:
                    edge = (i, j) if self.directed else (min(i, j), max(i, j))
                    if edge not in seen:
                        adj_list.add_edge(i, j, self.matrix[i, j])
                        seen.add(edge)
        return adj_list

    def __repr__(self):
        nnz = np.count_nonzero(self.matrix)
        if not self.directed:
            nnz //= 2
        return f"AdjacencyMatrix(nodes={self.num_nodes}, edges={nnz})"


# ============================================================
# 3. Sparse Matrix Representations
# ============================================================

def demo_sparse_representations():
    """Demonstrate sparse matrix formats for graphs."""
    print("=" * 60)
    print("Sparse Matrix Representations")
    print("=" * 60)

    # Create a sample graph
    num_nodes = 6
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

    # Dense adjacency matrix
    A_dense = np.zeros((num_nodes, num_nodes))
    for u, v in edges:
        A_dense[u, v] = 1
        A_dense[v, u] = 1

    print(f"Dense matrix shape: {A_dense.shape}")
    print(f"Dense matrix memory: {A_dense.nbytes} bytes")
    print(f"Non-zero entries: {np.count_nonzero(A_dense)}")

    # COO format
    A_coo = sp.coo_matrix(A_dense)
    print(f"\nCOO format:")
    print(f"  Row indices: {A_coo.row}")
    print(f"  Col indices: {A_coo.col}")
    print(f"  Values: {A_coo.data}")
    print(f"  Memory: ~{A_coo.data.nbytes + A_coo.row.nbytes + A_coo.col.nbytes} bytes")

    # CSR format
    A_csr = sp.csr_matrix(A_dense)
    print(f"\nCSR format:")
    print(f"  Indptr: {A_csr.indptr}")
    print(f"  Indices: {A_csr.indices}")
    print(f"  Data: {A_csr.data}")
    print(f"  Memory: ~{A_csr.data.nbytes + A_csr.indptr.nbytes + A_csr.indices.nbytes} bytes")

    # CSC format
    A_csc = sp.csc_matrix(A_dense)
    print(f"\nCSC format:")
    print(f"  Indptr: {A_csc.indptr}")
    print(f"  Indices: {A_csc.indices}")
    print(f"  Data: {A_csc.data}")

    # Verify all formats produce the same dense matrix
    assert np.allclose(A_coo.toarray(), A_dense)
    assert np.allclose(A_csr.toarray(), A_dense)
    assert np.allclose(A_csc.toarray(), A_dense)
    print("\nAll sparse formats verified to produce identical dense matrices.")

    return A_coo, A_csr, A_csc


# ============================================================
# 4. PyTorch Geometric COO Format
# ============================================================

def demo_pyg_format():
    """Demonstrate PyTorch Geometric edge_index format."""
    print("\n" + "=" * 60)
    print("PyTorch Geometric COO Format")
    print("=" * 60)

    # Create edge_index in COO format [2, num_edges]
    # For undirected graphs, each edge is stored twice
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4)]

    # Build edge_index (include both directions for undirected)
    source_nodes = []
    target_nodes = []
    for u, v in edges:
        source_nodes.extend([u, v])
        target_nodes.extend([v, u])

    edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)

    print(f"edge_index shape: {edge_index.shape}")
    print(f"edge_index:\n{edge_index}")
    print(f"Number of directed edges: {edge_index.shape[1]}")
    print(f"Number of undirected edges: {edge_index.shape[1] // 2}")

    # Node features
    num_nodes = 5
    num_features = 3
    x = torch.randn(num_nodes, num_features)
    print(f"\nNode feature matrix shape: {x.shape}")
    print(f"Node features:\n{x}")

    # Edge attributes
    num_edge_features = 2
    edge_attr = torch.randn(edge_index.shape[1], num_edge_features)
    print(f"\nEdge attribute matrix shape: {edge_attr.shape}")

    return edge_index, x, edge_attr


# ============================================================
# 5. Conversion Utilities
# ============================================================

def adjacency_to_edge_index(A: np.ndarray) -> torch.Tensor:
    """Convert adjacency matrix to PyG edge_index format."""
    rows, cols = np.nonzero(A)
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    return edge_index


def edge_index_to_adjacency(edge_index: torch.Tensor,
                             num_nodes: int) -> np.ndarray:
    """Convert PyG edge_index to adjacency matrix."""
    A = np.zeros((num_nodes, num_nodes))
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    A[src, dst] = 1
    return A


def edge_list_to_csr(edges: List[Tuple[int, int]],
                      num_nodes: int,
                      weights: Optional[List[float]] = None) -> sp.csr_matrix:
    """Convert edge list to CSR sparse matrix."""
    rows = [e[0] for e in edges]
    cols = [e[1] for e in edges]
    vals = weights if weights else [1.0] * len(edges)
    return sp.csr_matrix((vals, (rows, cols)), shape=(num_nodes, num_nodes))


def demo_conversions():
    """Demonstrate conversions between representations."""
    print("\n" + "=" * 60)
    print("Conversion Between Representations")
    print("=" * 60)

    # Start with adjacency matrix
    A = np.array([
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1],
        [0, 0, 1, 1, 0]
    ], dtype=float)

    print("Original Adjacency Matrix:")
    print(A)

    # To edge_index
    edge_index = adjacency_to_edge_index(A)
    print(f"\nEdge index:\n{edge_index}")

    # Back to adjacency matrix
    A_recovered = edge_index_to_adjacency(edge_index, num_nodes=5)
    print(f"\nRecovered Adjacency Matrix:")
    print(A_recovered)
    assert np.allclose(A, A_recovered)
    print("Conversion verified: A == recovered A")

    # To adjacency list
    adj_mat = AdjacencyMatrix(A)
    adj_list = adj_mat.to_adjacency_list()
    print(f"\nAdjacency List: {adj_list}")
    for node in range(5):
        neighbors = adj_list.neighbors(node)
        print(f"  Node {node}: {neighbors}")

    # To sparse CSR
    csr = sp.csr_matrix(A)
    print(f"\nCSR indptr: {csr.indptr}")
    print(f"CSR indices: {csr.indices}")

    # Edge list
    edge_list = adj_list.to_edge_list()
    print(f"\nEdge list: {edge_list}")


# ============================================================
# 6. Incidence Matrix
# ============================================================

def demo_incidence_matrix():
    """Demonstrate incidence matrix representation."""
    print("\n" + "=" * 60)
    print("Incidence Matrix")
    print("=" * 60)

    num_nodes = 4
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]

    # Build incidence matrix for undirected graph
    B = np.zeros((num_nodes, len(edges)))
    for e_idx, (u, v) in enumerate(edges):
        B[u, e_idx] = 1
        B[v, e_idx] = 1

    print("Incidence Matrix B (undirected):")
    print(B)

    # Verify: L = B @ B^T - D gives the graph Laplacian structure
    BBT = B @ B.T
    print(f"\nB @ B^T:")
    print(BBT)

    # This equals D + A for undirected graphs
    # Actually, for undirected: B @ B^T = D + A (not exactly Laplacian)
    # For directed incidence: B @ B^T = D - A = L

    # Build directed incidence matrix
    B_dir = np.zeros((num_nodes, len(edges)))
    for e_idx, (u, v) in enumerate(edges):
        B_dir[u, e_idx] = 1   # Edge leaves u
        B_dir[v, e_idx] = -1  # Edge enters v

    print("\nIncidence Matrix B (directed):")
    print(B_dir)

    L = B_dir @ B_dir.T
    print(f"\nGraph Laplacian (B_dir @ B_dir^T):")
    print(L)

    # Verify Laplacian: L = D - A
    A = np.zeros((num_nodes, num_nodes))
    for u, v in edges:
        A[u, v] = 1
        A[v, u] = 1
    D = np.diag(A.sum(axis=1))
    L_check = D - A
    print(f"\nL = D - A:")
    print(L_check)
    assert np.allclose(L, L_check)
    print("Verified: B_dir @ B_dir^T == D - A")


# ============================================================
# 7. Memory and Performance Comparison
# ============================================================

def benchmark_representations():
    """Compare memory usage of different representations."""
    print("\n" + "=" * 60)
    print("Memory Comparison of Graph Representations")
    print("=" * 60)

    sizes = [100, 1000, 5000]
    sparsity_levels = [0.01, 0.05, 0.1]

    for n in sizes:
        for sparsity in sparsity_levels:
            num_edges = int(n * n * sparsity)

            # Dense matrix memory
            dense_bytes = n * n * 8  # float64

            # Sparse COO memory
            coo_bytes = num_edges * (8 + 4 + 4)  # val + row + col

            # Sparse CSR memory
            csr_bytes = num_edges * (8 + 4) + (n + 1) * 4  # val + indices + indptr

            # Edge list memory
            edge_bytes = num_edges * (4 + 4 + 8)  # src + dst + weight

            # PyG edge_index memory (int64)
            pyg_bytes = num_edges * 2 * 8  # two rows of int64

            print(f"\nn={n:5d}, sparsity={sparsity:.2f}, edges~{num_edges}")
            print(f"  Dense:     {dense_bytes / 1024:8.1f} KB")
            print(f"  COO:       {coo_bytes / 1024:8.1f} KB "
                  f"({dense_bytes / max(coo_bytes, 1):.1f}x saving)")
            print(f"  CSR:       {csr_bytes / 1024:8.1f} KB "
                  f"({dense_bytes / max(csr_bytes, 1):.1f}x saving)")
            print(f"  Edge list: {edge_bytes / 1024:8.1f} KB")
            print(f"  PyG:       {pyg_bytes / 1024:8.1f} KB")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    demo_sparse_representations()
    demo_pyg_format()
    demo_conversions()
    demo_incidence_matrix()
    benchmark_representations()
