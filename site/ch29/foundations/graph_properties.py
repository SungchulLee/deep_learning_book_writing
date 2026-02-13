"""
Chapter 29.1.5: Graph Properties
Comprehensive graph property computation and analysis.
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, List, Tuple


# ============================================================
# 1. Connectivity Properties
# ============================================================

def analyze_connectivity(G: nx.Graph, name: str = "Graph"):
    """Analyze connectivity properties of a graph."""
    print(f"\n--- Connectivity Analysis: {name} ---")

    if G.is_directed():
        # Weakly connected components
        wcc = list(nx.weakly_connected_components(G))
        print(f"Weakly connected components: {len(wcc)}")
        print(f"Largest WCC size: {len(max(wcc, key=len))}")

        # Strongly connected components
        scc = list(nx.strongly_connected_components(G))
        print(f"Strongly connected components: {len(scc)}")
        print(f"Largest SCC size: {len(max(scc, key=len))}")
    else:
        components = list(nx.connected_components(G))
        print(f"Connected components: {len(components)}")
        print(f"Component sizes: {sorted([len(c) for c in components], reverse=True)}")

        if nx.is_connected(G):
            # Algebraic connectivity
            A = nx.adjacency_matrix(G).toarray().astype(float)
            D = np.diag(A.sum(axis=1))
            L = D - A
            eigvals = np.sort(np.linalg.eigvalsh(L))
            lambda_2 = eigvals[1]
            print(f"Algebraic connectivity (lambda_2): {lambda_2:.6f}")

            # Vertex connectivity
            print(f"Node connectivity: {nx.node_connectivity(G)}")
            print(f"Edge connectivity: {nx.edge_connectivity(G)}")


# ============================================================
# 2. Distance Properties
# ============================================================

def analyze_distances(G: nx.Graph, name: str = "Graph"):
    """Analyze distance-related properties."""
    print(f"\n--- Distance Properties: {name} ---")

    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
        print(f"Using largest connected component (n={G.number_of_nodes()})")

    # Diameter and radius
    eccentricities = nx.eccentricity(G)
    diameter = max(eccentricities.values())
    radius = min(eccentricities.values())
    center = nx.center(G)
    periphery = nx.periphery(G)

    print(f"Diameter: {diameter}")
    print(f"Radius: {radius}")
    print(f"Center nodes: {center[:5]}{'...' if len(center) > 5 else ''}")
    print(f"Periphery nodes: {periphery[:5]}{'...' if len(periphery) > 5 else ''}")

    # Average shortest path
    avg_path = nx.average_shortest_path_length(G)
    print(f"Average shortest path length: {avg_path:.4f}")
    print(f"log(n) = {np.log(G.number_of_nodes()):.4f}")
    print(f"Small-world (avg_path ~ log(n)): "
          f"ratio = {avg_path / np.log(G.number_of_nodes()):.4f}")

    return eccentricities


# ============================================================
# 3. Clustering and Community
# ============================================================

def analyze_clustering(G: nx.Graph, name: str = "Graph"):
    """Analyze clustering and community structure."""
    print(f"\n--- Clustering Analysis: {name} ---")

    # Local clustering coefficients
    cc = nx.clustering(G)
    cc_values = list(cc.values())

    print(f"Average clustering coefficient: {np.mean(cc_values):.4f}")
    print(f"Min clustering: {min(cc_values):.4f}")
    print(f"Max clustering: {max(cc_values):.4f}")

    # Global clustering (transitivity)
    transitivity = nx.transitivity(G)
    print(f"Transitivity (global clustering): {transitivity:.4f}")

    # Triangle count
    triangles = nx.triangles(G)
    total_triangles = sum(triangles.values()) // 3
    print(f"Number of triangles: {total_triangles}")

    # Compare with random graph
    n, m = G.number_of_nodes(), G.number_of_edges()
    p = 2 * m / (n * (n - 1))
    expected_cc_random = p
    print(f"Expected clustering (random): {expected_cc_random:.4f}")
    print(f"Ratio (actual/random): {np.mean(cc_values) / max(expected_cc_random, 1e-10):.2f}")

    # Community detection (greedy modularity)
    communities = list(nx.community.greedy_modularity_communities(G))
    modularity = nx.community.modularity(G, communities)
    print(f"\nGreedy modularity communities: {len(communities)}")
    print(f"Community sizes: {sorted([len(c) for c in communities], reverse=True)}")
    print(f"Modularity: {modularity:.4f}")

    return communities


# ============================================================
# 4. Degree Distribution Analysis
# ============================================================

def analyze_degree_distribution(G: nx.Graph, name: str = "Graph"):
    """Analyze the degree distribution."""
    print(f"\n--- Degree Distribution: {name} ---")

    degrees = [d for _, d in G.degree()]

    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Min degree: {min(degrees)}")
    print(f"Max degree: {max(degrees)}")
    print(f"Mean degree: {np.mean(degrees):.2f}")
    print(f"Median degree: {np.median(degrees):.2f}")
    print(f"Std degree: {np.std(degrees):.2f}")

    # Degree distribution
    degree_count = Counter(degrees)
    sorted_degrees = sorted(degree_count.keys())

    # Check for power-law (log-log linearity)
    if max(degrees) > 1:
        log_d = np.log(sorted_degrees)
        log_p = np.log([degree_count[d] / len(degrees) for d in sorted_degrees])

        # Simple linear regression on log-log scale
        valid = np.isfinite(log_d) & np.isfinite(log_p)
        if sum(valid) > 2:
            coeffs = np.polyfit(log_d[valid], log_p[valid], 1)
            gamma = -coeffs[0]
            print(f"Estimated power-law exponent gamma: {gamma:.2f}")
            print(f"(Scale-free if gamma in [2, 3])")

    # Degree assortativity
    assort = nx.degree_assortativity_coefficient(G)
    print(f"Degree assortativity: {assort:.4f}")
    print(f"  (positive = assortative, negative = disassortative)")

    return degrees


# ============================================================
# 5. Spectral Properties
# ============================================================

def analyze_spectral(G: nx.Graph, name: str = "Graph"):
    """Analyze spectral properties."""
    print(f"\n--- Spectral Properties: {name} ---")

    A = nx.adjacency_matrix(G).toarray().astype(float)
    n = A.shape[0]
    D = np.diag(A.sum(axis=1))
    L = D - A

    # Adjacency spectrum
    eigvals_A = np.sort(np.linalg.eigvalsh(A))[::-1]
    print(f"Adjacency eigenvalues (top 5): {np.round(eigvals_A[:5], 4)}")
    print(f"Spectral radius: {eigvals_A[0]:.4f}")

    if len(eigvals_A) > 1:
        spectral_gap = eigvals_A[0] - eigvals_A[1]
        print(f"Spectral gap (lambda_1 - lambda_2): {spectral_gap:.4f}")

    # Laplacian spectrum
    eigvals_L = np.sort(np.linalg.eigvalsh(L))
    print(f"\nLaplacian eigenvalues (first 5): {np.round(eigvals_L[:5], 4)}")
    print(f"Algebraic connectivity: {eigvals_L[1]:.6f}")
    print(f"Largest Laplacian eigenvalue: {eigvals_L[-1]:.4f}")

    # Spectral energy
    energy = np.sum(np.abs(eigvals_A))
    print(f"Graph energy: {energy:.4f}")

    return eigvals_A, eigvals_L


# ============================================================
# 6. Small-World and Scale-Free Tests
# ============================================================

def test_small_world(G: nx.Graph, name: str = "Graph"):
    """Test if graph exhibits small-world property."""
    print(f"\n--- Small-World Test: {name} ---")

    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = 2 * m / (n * (n - 1))

    # Average clustering
    C = nx.average_clustering(G)

    # Average path length
    if nx.is_connected(G):
        L = nx.average_shortest_path_length(G)
    else:
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc)
        L = nx.average_shortest_path_length(G_cc)

    # Random graph comparison
    C_random = p
    L_random = np.log(n) / np.log(max(np.mean([d for _, d in G.degree()]), 1.1))

    # Small-world coefficient
    sigma = (C / max(C_random, 1e-10)) / (L / max(L_random, 1e-10))

    print(f"Clustering: {C:.4f} (random: {C_random:.4f})")
    print(f"Avg path length: {L:.4f} (random estimate: {L_random:.4f})")
    print(f"Small-world sigma: {sigma:.4f}")
    print(f"Small-world: {'Yes' if sigma > 1 else 'No'} (sigma > 1)")

    return sigma


# ============================================================
# 7. Comprehensive Graph Analysis
# ============================================================

def comprehensive_analysis(G: nx.Graph, name: str = "Graph"):
    """Run all analyses on a graph."""
    print("\n" + "=" * 60)
    print(f"Comprehensive Analysis: {name}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")
    print("=" * 60)

    analyze_connectivity(G, name)
    if nx.is_connected(G):
        analyze_distances(G, name)
    analyze_clustering(G, name)
    analyze_degree_distribution(G, name)
    if G.number_of_nodes() <= 500:
        analyze_spectral(G, name)
    test_small_world(G, name)


# ============================================================
# 8. Financial Network Properties
# ============================================================

def analyze_financial_network():
    """Analyze properties specific to financial networks."""
    print("\n" + "=" * 60)
    print("Financial Network Analysis")
    print("=" * 60)

    np.random.seed(42)
    n_assets = 30
    n_days = 504  # 2 years

    # Simulate with sector structure
    n_sectors = 5
    sector_labels = np.repeat(range(n_sectors), n_assets // n_sectors)

    market = np.random.randn(n_days) * 0.01
    returns = np.zeros((n_days, n_assets))
    for i in range(n_assets):
        sector_factor = np.random.randn(n_days) * 0.005
        returns[:, i] = (market + sector_factor +
                         np.random.randn(n_days) * 0.003)

    # Build correlation graph
    corr = np.corrcoef(returns.T)
    threshold = 0.3
    A = (np.abs(corr) > threshold).astype(float)
    np.fill_diagonal(A, 0)

    G = nx.from_numpy_array(A)
    comprehensive_analysis(G, "Correlation Network (threshold=0.3)")

    # Analyze sector structure
    communities = list(nx.community.greedy_modularity_communities(G))
    print(f"\nSector alignment analysis:")
    for i, comm in enumerate(communities):
        sector_counts = Counter(sector_labels[list(comm)])
        dominant_sector = sector_counts.most_common(1)[0]
        purity = dominant_sector[1] / len(comm)
        print(f"  Community {i}: size={len(comm)}, "
              f"dominant_sector={dominant_sector[0]}, purity={purity:.2f}")

    # Density evolution during market stress
    print(f"\nDensity evolution (varying threshold):")
    for tau in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        A_tau = (np.abs(corr) > tau).astype(float)
        np.fill_diagonal(A_tau, 0)
        density = A_tau.sum() / (n_assets * (n_assets - 1))
        G_tau = nx.from_numpy_array(A_tau)
        n_comp = nx.number_connected_components(G_tau)
        print(f"  tau={tau:.1f}: density={density:.3f}, "
              f"components={n_comp}, edges={int(A_tau.sum()) // 2}")


# ============================================================
# 9. Visualization
# ============================================================

def visualize_properties():
    """Visualize various graph properties."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    G = nx.barabasi_albert_graph(200, 2, seed=42)

    # 1. Degree distribution (log-log)
    degrees = [d for _, d in G.degree()]
    degree_count = Counter(degrees)
    x = sorted(degree_count.keys())
    y = [degree_count[d] for d in x]
    axes[0, 0].loglog(x, y, 'bo', markersize=5)
    axes[0, 0].set_xlabel("Degree (log)")
    axes[0, 0].set_ylabel("Count (log)")
    axes[0, 0].set_title("Degree Distribution (Barabasi-Albert)")

    # 2. Clustering coefficient distribution
    cc = list(nx.clustering(G).values())
    axes[0, 1].hist(cc, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0, 1].set_xlabel("Local Clustering Coefficient")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Clustering Coefficient Distribution")

    # 3. Shortest path length distribution
    if nx.is_connected(G):
        all_paths = dict(nx.all_pairs_shortest_path_length(G))
        path_lengths = []
        for u in all_paths:
            for v, d in all_paths[u].items():
                if u < v:
                    path_lengths.append(d)
        axes[1, 0].hist(path_lengths, bins=range(max(path_lengths) + 2),
                        color='green', alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel("Shortest Path Length")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].set_title("Shortest Path Distribution")

    # 4. Laplacian spectrum
    A = nx.adjacency_matrix(G).toarray().astype(float)
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals = np.sort(np.linalg.eigvalsh(L))
    axes[1, 1].plot(eigvals, 'r-', linewidth=0.5)
    axes[1, 1].set_xlabel("Index")
    axes[1, 1].set_ylabel("Eigenvalue")
    axes[1, 1].set_title("Laplacian Spectrum")

    plt.tight_layout()
    plt.savefig("graph_properties.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("\nVisualization saved to graph_properties.png")


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    # Analyze standard graphs
    graphs = {
        'Karate Club': nx.karate_club_graph(),
        'Watts-Strogatz': nx.watts_strogatz_graph(100, 4, 0.1, seed=42),
        'Barabasi-Albert': nx.barabasi_albert_graph(100, 2, seed=42),
    }

    for name, G in graphs.items():
        comprehensive_analysis(G, name)

    analyze_financial_network()
    visualize_properties()
