"""
Chapter 29.1.3: Adjacency Matrix
Deep dive into adjacency matrices, Laplacians, and spectral properties.
"""

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import matplotlib.pyplot as plt
import networkx as nx
from typing import Tuple, Optional


# ============================================================
# 1. Adjacency Matrix Construction and Properties
# ============================================================

def construct_adjacency_matrix(edges: list, num_nodes: int,
                                weights: Optional[list] = None,
                                directed: bool = False) -> np.ndarray:
    """Construct adjacency matrix from edge list."""
    A = np.zeros((num_nodes, num_nodes))
    for idx, (u, v) in enumerate(edges):
        w = weights[idx] if weights else 1.0
        A[u, v] = w
        if not directed:
            A[v, u] = w
    return A


def demo_adjacency_properties():
    """Demonstrate key properties of adjacency matrices."""
    print("=" * 60)
    print("Adjacency Matrix Properties")
    print("=" * 60)

    # Create a sample graph
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    n = 5
    A = construct_adjacency_matrix(edges, n)

    print("Adjacency Matrix A:")
    print(A)

    # Symmetry check
    print(f"\nSymmetric: {np.allclose(A, A.T)}")

    # Degree matrix
    D = np.diag(A.sum(axis=1))
    print(f"\nDegree Matrix D:\n{D}")
    print(f"Degrees: {np.diag(D).astype(int)}")

    # A^2: common neighbors
    A2 = A @ A
    print(f"\nA^2 (walks of length 2):\n{A2.astype(int)}")
    print(f"Common neighbors of (0,3): {int(A2[0, 3])}")

    # A^3: triangles
    A3 = A @ A @ A
    num_triangles = int(np.trace(A3)) // 6
    print(f"\nNumber of triangles: {num_triangles}")

    # Spectrum
    eigenvalues = np.sort(np.linalg.eigvalsh(A))[::-1]
    print(f"\nSpectrum of A: {np.round(eigenvalues, 4)}")
    print(f"Largest eigenvalue: {eigenvalues[0]:.4f}")
    print(f"Average degree: {np.mean(np.diag(D)):.4f}")
    print(f"Max degree: {np.max(np.diag(D)):.0f}")

    return A, D


# ============================================================
# 2. Graph Laplacian
# ============================================================

def compute_laplacians(A: np.ndarray) -> dict:
    """Compute various Laplacian matrices from adjacency matrix."""
    n = A.shape[0]
    D = np.diag(A.sum(axis=1))
    d = A.sum(axis=1)

    # Unnormalized Laplacian
    L = D - A

    # Symmetric normalized Laplacian
    d_inv_sqrt = np.zeros_like(d)
    nonzero = d > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # Random walk Laplacian
    d_inv = np.zeros_like(d)
    d_inv[nonzero] = 1.0 / d[nonzero]
    D_inv = np.diag(d_inv)
    L_rw = np.eye(n) - D_inv @ A

    return {
        'L': L,
        'L_sym': L_sym,
        'L_rw': L_rw,
        'D': D,
        'D_inv_sqrt': D_inv_sqrt,
        'D_inv': D_inv
    }


def demo_laplacians():
    """Demonstrate Laplacian properties."""
    print("\n" + "=" * 60)
    print("Graph Laplacian Properties")
    print("=" * 60)

    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    n = 5
    A = construct_adjacency_matrix(edges, n)
    laplacians = compute_laplacians(A)

    L = laplacians['L']
    print("Unnormalized Laplacian L = D - A:")
    print(L)

    # Eigenvalues of L
    eigvals_L = np.sort(np.linalg.eigvalsh(L))
    print(f"\nEigenvalues of L: {np.round(eigvals_L, 4)}")
    print(f"Smallest eigenvalue (should be ~0): {eigvals_L[0]:.6f}")

    # Positive semi-definite check
    print(f"All eigenvalues >= 0: {all(eigvals_L >= -1e-10)}")

    # Quadratic form: smoothness measure
    # x^T L x = 0.5 * sum_{(i,j) in E} (x_i - x_j)^2
    x = np.array([1.0, 0.8, 0.6, 0.4, 0.2])  # Smooth signal
    smoothness = x @ L @ x
    manual = 0.5 * sum(
        A[i, j] * (x[i] - x[j]) ** 2
        for i in range(n) for j in range(n)
    )
    print(f"\nSmooth signal x = {x}")
    print(f"x^T L x = {smoothness:.4f}")
    print(f"Manual computation = {manual:.4f}")
    print(f"Match: {np.isclose(smoothness, manual)}")

    # Non-smooth signal
    x_rough = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    smoothness_rough = x_rough @ L @ x_rough
    print(f"\nRough signal x = {x_rough}")
    print(f"x^T L x = {smoothness_rough:.4f}")

    # Symmetric normalized Laplacian
    L_sym = laplacians['L_sym']
    eigvals_sym = np.sort(np.linalg.eigvalsh(L_sym))
    print(f"\nSymmetric Normalized Laplacian eigenvalues: {np.round(eigvals_sym, 4)}")
    print(f"Range: [{eigvals_sym[0]:.4f}, {eigvals_sym[-1]:.4f}] (should be in [0, 2])")

    return laplacians


# ============================================================
# 3. Normalized Adjacency Matrix (GCN-style)
# ============================================================

def gcn_norm(A: np.ndarray, add_self_loops: bool = True) -> np.ndarray:
    """
    Compute the GCN-normalized adjacency matrix:
    A_hat = D_tilde^{-1/2} A_tilde D_tilde^{-1/2}
    where A_tilde = A + I (with self-loops).
    """
    n = A.shape[0]

    if add_self_loops:
        A_tilde = A + np.eye(n)
    else:
        A_tilde = A.copy()

    # Degree of modified adjacency
    d = A_tilde.sum(axis=1)
    d_inv_sqrt = np.zeros_like(d)
    nonzero = d > 0
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(d[nonzero])
    D_inv_sqrt = np.diag(d_inv_sqrt)

    A_hat = D_inv_sqrt @ A_tilde @ D_inv_sqrt
    return A_hat


def demo_gcn_normalization():
    """Demonstrate GCN normalization (renormalization trick)."""
    print("\n" + "=" * 60)
    print("GCN Normalization (Renormalization Trick)")
    print("=" * 60)

    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
    n = 4
    A = construct_adjacency_matrix(edges, n)

    print("Original A:")
    print(A)

    # Without self-loops
    A_norm = gcn_norm(A, add_self_loops=False)
    print(f"\nNormalized (no self-loops):")
    print(np.round(A_norm, 4))

    # With self-loops
    A_hat = gcn_norm(A, add_self_loops=True)
    print(f"\nNormalized (with self-loops - GCN style):")
    print(np.round(A_hat, 4))

    # Row sums after normalization
    print(f"\nRow sums of A_hat: {np.round(A_hat.sum(axis=1), 4)}")

    # Eigenvalues comparison
    eigvals_A = np.sort(np.linalg.eigvalsh(A))[::-1]
    eigvals_hat = np.sort(np.linalg.eigvalsh(A_hat))[::-1]
    print(f"\nEigenvalues of A: {np.round(eigvals_A, 4)}")
    print(f"Eigenvalues of A_hat: {np.round(eigvals_hat, 4)}")

    return A_hat


# ============================================================
# 4. Transition Matrix (Random Walk)
# ============================================================

def demo_transition_matrix():
    """Demonstrate random walk transition matrix."""
    print("\n" + "=" * 60)
    print("Random Walk Transition Matrix")
    print("=" * 60)

    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
    n = 5
    A = construct_adjacency_matrix(edges, n)

    # Transition matrix P = D^{-1} A
    D_inv = np.diag(1.0 / A.sum(axis=1))
    P = D_inv @ A

    print("Transition Matrix P = D^{-1} A:")
    print(np.round(P, 4))
    print(f"\nRow sums (should be 1): {np.round(P.sum(axis=1), 4)}")

    # Random walk simulation
    np.random.seed(42)
    current_node = 0
    walk = [current_node]
    walk_length = 10

    for _ in range(walk_length):
        probs = P[current_node]
        next_node = np.random.choice(n, p=probs)
        walk.append(next_node)
        current_node = next_node

    print(f"\nRandom walk starting from node 0: {walk}")

    # Stationary distribution (for connected graphs)
    eigenvalues, eigenvectors = np.linalg.eig(P.T)
    # Find eigenvector for eigenvalue 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    stationary = np.real(eigenvectors[:, idx])
    stationary = stationary / stationary.sum()
    print(f"\nStationary distribution: {np.round(stationary, 4)}")

    # Verify: proportional to degree
    degrees = A.sum(axis=1)
    expected = degrees / degrees.sum()
    print(f"Expected (proportional to degree): {np.round(expected, 4)}")
    print(f"Match: {np.allclose(stationary, expected, atol=1e-6)}")

    return P


# ============================================================
# 5. Spectral Analysis
# ============================================================

def spectral_analysis(A: np.ndarray, graph_name: str = "Graph"):
    """Perform spectral analysis of the adjacency matrix."""
    print(f"\n{'=' * 60}")
    print(f"Spectral Analysis: {graph_name}")
    print(f"{'=' * 60}")

    n = A.shape[0]

    # Adjacency spectrum
    eigvals_A, eigvecs_A = np.linalg.eigh(A)
    idx = np.argsort(eigvals_A)[::-1]
    eigvals_A = eigvals_A[idx]
    eigvecs_A = eigvecs_A[:, idx]

    print(f"Adjacency eigenvalues: {np.round(eigvals_A[:5], 4)}...")
    print(f"Spectral radius: {eigvals_A[0]:.4f}")
    print(f"Spectral gap: {eigvals_A[0] - eigvals_A[1]:.4f}")

    # Laplacian spectrum
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals_L = np.sort(np.linalg.eigvalsh(L))

    print(f"\nLaplacian eigenvalues: {np.round(eigvals_L[:5], 4)}...")
    print(f"Algebraic connectivity (lambda_2): {eigvals_L[1]:.4f}")
    print(f"Number of connected components: "
          f"{np.sum(np.abs(eigvals_L) < 1e-10)}")

    return eigvals_A, eigvals_L


def demo_spectral_comparison():
    """Compare spectra of different graph types."""
    graphs = {
        'Path (n=20)': nx.path_graph(20),
        'Cycle (n=20)': nx.cycle_graph(20),
        'Complete (n=20)': nx.complete_graph(20),
        'Star (n=20)': nx.star_graph(19),
        'Barbell (5,5)': nx.barbell_graph(5, 1),
    }

    for name, G in graphs.items():
        A = nx.adjacency_matrix(G).toarray().astype(float)
        spectral_analysis(A, name)


# ============================================================
# 6. Financial Correlation to Adjacency
# ============================================================

def correlation_to_adjacency(corr_matrix: np.ndarray,
                              method: str = 'threshold',
                              threshold: float = 0.5,
                              power: float = 1.0) -> np.ndarray:
    """
    Convert correlation matrix to adjacency matrix.

    Methods:
        'threshold': Binary adjacency with |corr| > threshold
        'soft': Weighted adjacency using |corr|^power
        'mst': Minimum spanning tree from distance matrix
    """
    n = corr_matrix.shape[0]

    if method == 'threshold':
        A = (np.abs(corr_matrix) > threshold).astype(float)
        np.fill_diagonal(A, 0)

    elif method == 'soft':
        A = np.abs(corr_matrix) ** power
        np.fill_diagonal(A, 0)

    elif method == 'mst':
        # Distance: d_ij = sqrt(2(1 - rho_ij))
        distance = np.sqrt(2 * (1 - corr_matrix))
        np.fill_diagonal(distance, 0)
        G = nx.from_numpy_array(distance)
        mst = nx.minimum_spanning_tree(G)
        A = nx.adjacency_matrix(mst).toarray().astype(float)

    else:
        raise ValueError(f"Unknown method: {method}")

    return A


def demo_financial_adjacency():
    """Demonstrate converting correlation matrix to graph adjacency."""
    print("\n" + "=" * 60)
    print("Financial Correlation → Adjacency Matrix")
    print("=" * 60)

    np.random.seed(42)
    n_assets = 6
    n_days = 252
    asset_names = ["AAPL", "MSFT", "GOOGL", "JPM", "GS", "BAC"]

    # Simulate returns with sector structure
    # Tech sector: AAPL, MSFT, GOOGL
    # Finance sector: JPM, GS, BAC
    market = np.random.randn(n_days) * 0.01
    tech_factor = np.random.randn(n_days) * 0.005
    fin_factor = np.random.randn(n_days) * 0.005

    returns = np.column_stack([
        market + tech_factor + np.random.randn(n_days) * 0.003,
        market + tech_factor + np.random.randn(n_days) * 0.003,
        market + tech_factor + np.random.randn(n_days) * 0.003,
        market + fin_factor + np.random.randn(n_days) * 0.004,
        market + fin_factor + np.random.randn(n_days) * 0.004,
        market + fin_factor + np.random.randn(n_days) * 0.004,
    ])

    corr = np.corrcoef(returns.T)
    print("Correlation Matrix:")
    header = "      " + "  ".join(f"{name:>6}" for name in asset_names)
    print(header)
    for i, name in enumerate(asset_names):
        row = f"{name:>5} " + "  ".join(f"{corr[i, j]:6.3f}" for j in range(n_assets))
        print(row)

    # Method 1: Thresholding
    A_thresh = correlation_to_adjacency(corr, method='threshold', threshold=0.4)
    print(f"\nThreshold adjacency (tau=0.4):")
    print(A_thresh.astype(int))
    print(f"Number of edges: {int(A_thresh.sum()) // 2}")

    # Method 2: Soft weighting
    A_soft = correlation_to_adjacency(corr, method='soft', power=2)
    print(f"\nSoft adjacency (power=2) - first 3 rows:")
    print(np.round(A_soft[:3, :], 4))

    # Method 3: MST
    A_mst = correlation_to_adjacency(corr, method='mst')
    print(f"\nMST adjacency:")
    print(A_mst.astype(int))
    print(f"Number of edges: {int(A_mst.sum()) // 2} (should be {n_assets - 1})")

    # Spectral analysis of threshold graph
    if A_thresh.sum() > 0:
        spectral_analysis(A_thresh, "Financial Threshold Graph")


# ============================================================
# 7. Visualization
# ============================================================

def visualize_spectra():
    """Visualize adjacency and Laplacian spectra."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    graphs = {
        'Karate Club': nx.karate_club_graph(),
        'Barabasi-Albert (n=50)': nx.barabasi_albert_graph(50, 2, seed=42),
        'Watts-Strogatz (n=50)': nx.watts_strogatz_graph(50, 4, 0.3, seed=42),
        'Erdos-Renyi (n=50)': nx.erdos_renyi_graph(50, 0.1, seed=42),
    }

    for idx, (name, G) in enumerate(graphs.items()):
        ax = axes[idx // 2, idx % 2]
        A = nx.adjacency_matrix(G).toarray().astype(float)
        D = np.diag(A.sum(axis=1))
        L = D - A

        eigvals_L = np.sort(np.linalg.eigvalsh(L))

        ax.bar(range(len(eigvals_L)), eigvals_L, color='steelblue', alpha=0.7)
        ax.set_title(f"{name}\n(n={G.number_of_nodes()}, m={G.number_of_edges()})")
        ax.set_xlabel("Index")
        ax.set_ylabel("Laplacian Eigenvalue")
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig("laplacian_spectra.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSpectra visualization saved to laplacian_spectra.png")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    A, D = demo_adjacency_properties()
    laplacians = demo_laplacians()
    A_hat = demo_gcn_normalization()
    P = demo_transition_matrix()
    demo_spectral_comparison()
    demo_financial_adjacency()
    visualize_spectra()
