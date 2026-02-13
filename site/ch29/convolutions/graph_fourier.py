"""
Chapter 29.3.2: Graph Fourier Transform
GFT implementation, spectral convolution, and Parseval's theorem.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def compute_gft_basis(A):
    """Compute the GFT basis (Laplacian eigenvectors)."""
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigenvalues, U = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    return eigenvalues[idx], U[:, idx], L


def gft(signal, U):
    """Forward Graph Fourier Transform."""
    return U.T @ signal


def igft(spectral_coeffs, U):
    """Inverse Graph Fourier Transform."""
    return U @ spectral_coeffs


def spectral_convolution(f, g, U):
    """Convolution on graph via spectral domain."""
    f_hat = gft(f, U)
    g_hat = gft(g, U)
    return igft(f_hat * g_hat, U)


def demo_gft():
    """Demonstrate the Graph Fourier Transform."""
    print("=" * 60)
    print("Graph Fourier Transform")
    print("=" * 60)

    G = nx.path_graph(50)
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, L = compute_gft_basis(A)
    n = 50

    # Different signals
    signals = {
        'Low-freq (sin 1 cycle)': np.sin(np.linspace(0, 2*np.pi, n)),
        'High-freq (sin 10 cycles)': np.sin(np.linspace(0, 20*np.pi, n)),
        'Step function': np.concatenate([np.ones(25), -np.ones(25)]),
        'Random': np.random.randn(n),
    }

    for name, f in signals.items():
        f_hat = gft(f, U)
        # Energy distribution
        total_energy = np.sum(f_hat**2)
        low_energy = np.sum(f_hat[:n//4]**2) / total_energy
        high_energy = np.sum(f_hat[3*n//4:]**2) / total_energy
        print(f"  {name:30s}: low={low_energy:.3f}, high={high_energy:.3f}")


def demo_parseval():
    """Verify Parseval's theorem on graphs."""
    print("\n" + "=" * 60)
    print("Parseval's Theorem")
    print("=" * 60)

    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    _, U, _ = compute_gft_basis(A)

    np.random.seed(42)
    for trial in range(5):
        f = np.random.randn(A.shape[0])
        f_hat = gft(f, U)
        spatial_energy = np.sum(f**2)
        spectral_energy = np.sum(f_hat**2)
        print(f"  Trial {trial+1}: spatial={spatial_energy:.4f}, "
              f"spectral={spectral_energy:.4f}, "
              f"match={np.isclose(spatial_energy, spectral_energy)}")


def demo_spectral_convolution():
    """Demonstrate graph convolution in the spectral domain."""
    print("\n" + "=" * 60)
    print("Spectral Convolution")
    print("=" * 60)

    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, L = compute_gft_basis(A)
    n = A.shape[0]

    np.random.seed(42)
    signal = np.random.randn(n) + np.array(
        [0 if G.nodes[i].get('club', '') == 'Mr. Hi' else 2 for i in range(n)],
        dtype=float)

    # Learnable spectral filter (simulated)
    # Low-pass filter
    theta_lowpass = np.exp(-eigenvalues / eigenvalues[-1])
    filtered = U @ (theta_lowpass * (U.T @ signal))

    # Equivalent to: g_theta(L) @ signal
    g_L = U @ np.diag(theta_lowpass) @ U.T
    filtered_matrix = g_L @ signal

    print(f"Spectral filtering match: {np.allclose(filtered, filtered_matrix)}")
    print(f"Original signal range: [{signal.min():.2f}, {signal.max():.2f}]")
    print(f"Filtered signal range: [{filtered.min():.2f}, {filtered.max():.2f}]")


def demo_learnable_spectral_filter():
    """Simulate learning spectral filter parameters."""
    print("\n" + "=" * 60)
    print("Learnable Spectral Filter")
    print("=" * 60)

    G = nx.karate_club_graph()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, L = compute_gft_basis(A)
    n = A.shape[0]

    # Target: community labels
    labels = np.array([0 if G.nodes[i].get('club', '') == 'Mr. Hi' else 1
                        for i in range(n)], dtype=float)

    # Input: noisy signal
    np.random.seed(42)
    signal = labels + np.random.randn(n) * 0.5

    # Learn theta via gradient descent
    theta = np.random.randn(n) * 0.1
    lr = 0.01

    for epoch in range(200):
        filtered = U @ (theta * (U.T @ signal))
        error = filtered - labels
        loss = np.mean(error**2)

        # Gradient: d_loss/d_theta
        f_hat = U.T @ signal
        grad = 2 * (U.T @ error) * f_hat / n
        theta -= lr * grad

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: Loss={loss:.4f}")

    # Final filter shape
    print(f"\nLearned filter (first 5): {np.round(theta[:5], 3)}")
    print(f"Ideal low-pass (first 5): {np.round(np.exp(-eigenvalues[:5]), 3)}")


def visualize_gft():
    """Visualize GFT components."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    G = nx.path_graph(50)
    A = nx.adjacency_matrix(G).toarray().astype(float)
    eigenvalues, U, _ = compute_gft_basis(A)

    # Eigenvectors (basis functions)
    for k in range(6):
        ax = axes[k // 3, k % 3]
        ax.plot(U[:, k], 'b-', linewidth=1.5)
        ax.set_title(f"Eigenvector u_{k} (λ={eigenvalues[k]:.3f})")
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle("Graph Fourier Basis (Path Graph)", fontsize=14)
    plt.tight_layout()
    plt.savefig("graph_fourier.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nVisualization saved to graph_fourier.png")


if __name__ == "__main__":
    demo_gft()
    demo_parseval()
    demo_spectral_convolution()
    demo_learnable_spectral_filter()
    visualize_gft()
