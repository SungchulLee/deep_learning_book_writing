"""
Chapter 29.3.1: Spectral Graph Theory
Eigenanalysis, graph signals, and spectral filtering.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def laplacian_eigendecomposition(A):
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], eigenvectors[:, idx], L


def demo_spectral_decomposition():
    print("=" * 60)
    print("Spectral Graph Theory: Eigendecomposition")
    print("=" * 60)
    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, L = laplacian_eigendecomposition(A)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"Eigenvalues (first 8): {np.round(eigenvalues[:8], 4)}")
    print(f"Algebraic connectivity: {eigenvalues[1]:.6f}")
    L_rec = U @ np.diag(eigenvalues) @ U.T
    print(f"Reconstruction error: {np.max(np.abs(L - L_rec)):.2e}")
    return eigenvalues, U, L


def demo_graph_signals():
    print("\n" + "=" * 60)
    print("Graph Signals and Smoothness")
    print("=" * 60)
    G = nx.path_graph(20)
    A = nx.adjacency_matrix(G).toarray().astype(float)
    _, _, L = laplacian_eigendecomposition(A)
    n = 20
    signals = {
        'Constant': np.ones(n),
        'Linear': np.linspace(0, 1, n),
        'Sin': np.sin(np.linspace(0, 2 * np.pi, n)),
        'Alternating': np.array([(-1)**i for i in range(n)], dtype=float),
    }
    for name, f in signals.items():
        fn = f / (np.linalg.norm(f) + 1e-10)
        print(f"  {name:15s}: smoothness = {fn @ L @ fn:.6f}")


def demo_gft():
    print("\n" + "=" * 60)
    print("Graph Fourier Transform")
    print("=" * 60)
    G = nx.path_graph(30)
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, _ = laplacian_eigendecomposition(A)
    n = 30
    f = np.sin(np.linspace(0, 2 * np.pi, n))
    f_hat = U.T @ f
    print(f"Smooth signal - energy in low freq: {np.sum(f_hat[:5]**2)/np.sum(f_hat**2):.4f}")
    f_rough = np.array([(-1)**i for i in range(n)], dtype=float)
    f_hat_r = U.T @ f_rough
    print(f"Rough signal - energy in high freq: {np.sum(f_hat_r[-5:]**2)/np.sum(f_hat_r**2):.4f}")
    print(f"Reconstruction error: {np.max(np.abs(f - U @ f_hat)):.2e}")


def spectral_filter(signal, U, eigenvalues, filter_fn):
    f_hat = U.T @ signal
    return U @ (filter_fn(eigenvalues) * f_hat)


def demo_spectral_filtering():
    print("\n" + "=" * 60)
    print("Spectral Graph Filtering")
    print("=" * 60)
    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, L = laplacian_eigendecomposition(A)
    n = A.shape[0]
    np.random.seed(42)
    labels = np.array([0 if G.nodes[i].get('club', '') == 'Mr. Hi' else 1
                        for i in range(n)], dtype=float)
    noisy = labels + np.random.randn(n) * 0.5
    low_pass = lambda lam: np.exp(-5 * lam / eigenvalues[-1])
    filtered = spectral_filter(noisy, U, eigenvalues, low_pass)
    print(f"Noisy MSE: {np.mean((noisy - labels)**2):.4f}")
    print(f"Filtered MSE: {np.mean((filtered - labels)**2):.4f}")

    filters = {
        'Low-pass': lambda lam: np.exp(-2 * lam / eigenvalues[-1]),
        'Band-pass': lambda lam: np.exp(-((lam - eigenvalues[-1]/3)**2) / 0.5),
        'High-pass': lambda lam: 1 - np.exp(-5 * lam / eigenvalues[-1]),
    }
    for name, filt in filters.items():
        out = spectral_filter(noisy, U, eigenvalues, filt)
        print(f"  {name:12s}: MSE={np.mean((out - labels)**2):.4f}")


def demo_fiedler_partitioning():
    print("\n" + "=" * 60)
    print("Fiedler Vector Partitioning")
    print("=" * 60)
    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, _ = laplacian_eigendecomposition(A)
    fiedler = U[:, 1]
    partition_a = set(np.where(fiedler >= 0)[0])
    partition_b = set(np.where(fiedler < 0)[0])
    true_a = {i for i in range(34) if G.nodes[i].get('club', '') == 'Mr. Hi'}
    true_b = set(range(34)) - true_a
    overlap_1 = len(partition_a & true_a) + len(partition_b & true_b)
    overlap_2 = len(partition_a & true_b) + len(partition_b & true_a)
    accuracy = max(overlap_1, overlap_2) / 34
    print(f"Partition sizes: {len(partition_a)}, {len(partition_b)}")
    print(f"Spectral partitioning accuracy: {accuracy:.4f}")
    cut_edges = sum(1 for u, v in G.edges() if (u in partition_a) != (v in partition_a))
    print(f"Cut edges: {cut_edges}")


def visualize_spectra():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, _ = laplacian_eigendecomposition(A)

    axes[0, 0].bar(range(len(eigenvalues)), eigenvalues, color='steelblue')
    axes[0, 0].set_title("Laplacian Spectrum")
    axes[0, 0].set_xlabel("Index")
    axes[0, 0].set_ylabel("Eigenvalue")

    for k in [0, 1, 2, 3]:
        axes[0, 1].plot(U[:, k], label=f"u_{k}")
    axes[0, 1].set_title("First 4 Eigenvectors")
    axes[0, 1].legend()

    lam_range = np.linspace(0, eigenvalues[-1], 100)
    axes[1, 0].plot(lam_range, np.exp(-2 * lam_range / eigenvalues[-1]), label='Low-pass')
    axes[1, 0].plot(lam_range, 1 - np.exp(-5 * lam_range / eigenvalues[-1]), label='High-pass')
    axes[1, 0].set_title("Spectral Filters")
    axes[1, 0].legend()

    labels = np.array([0 if G.nodes[i].get('club', '') == 'Mr. Hi' else 1
                        for i in range(34)], dtype=float)
    f_hat = U.T @ labels
    axes[1, 1].bar(range(len(f_hat)), np.abs(f_hat), color='coral')
    axes[1, 1].set_title("GFT of Community Labels")
    axes[1, 1].set_xlabel("Frequency index")

    plt.tight_layout()
    plt.savefig("spectral_theory.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nVisualization saved to spectral_theory.png")


if __name__ == "__main__":
    demo_spectral_decomposition()
    demo_graph_signals()
    demo_gft()
    demo_spectral_filtering()
    demo_fiedler_partitioning()
    visualize_spectra()
