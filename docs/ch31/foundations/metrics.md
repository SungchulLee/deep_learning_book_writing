# Evaluation Metrics for Graph Generation

## Overview

Evaluating graph generation models requires comparing the distribution of generated graphs $\{G_i^{\text{gen}}\}$ against a reference distribution $\{G_i^{\text{ref}}\}$. Unlike image generation where Fréchet Inception Distance (FID) provides a standardized metric, graph evaluation demands multiple complementary measures capturing structural, statistical, and domain-specific properties. No single metric suffices — a generator may produce graphs with correct degree distributions but wrong clustering patterns, or vice versa.

## Graph Statistics

The primary evaluation approach computes distributional statistics of graph-level properties and compares them between generated and reference sets.

### Degree Distribution

The degree of node $v$ is $d_v = \sum_{u} A_{vu}$. The degree distribution $P(d)$ captures the fundamental connectivity pattern. For a set of graphs, we compute the empirical degree distribution by aggregating across all graphs:

$$
\hat{P}(d) = \frac{1}{\sum_i |\mathcal{V}_i|} \sum_{i} \sum_{v \in \mathcal{V}_i} \mathbf{1}[d_v = d]
$$

### Clustering Coefficient Distribution

The local clustering coefficient of node $v$ measures the fraction of possible triangles through $v$ that exist:

$$
C_v = \frac{2 |\{(u,w) : u,w \in \mathcal{N}(v), (u,w) \in \mathcal{E}\}|}{d_v(d_v - 1)}
$$

The distribution of $C_v$ across nodes captures the local structural density and transitivity of the graph.

### Orbit Counts

Graph orbits (from the Lov\'asz framework) count occurrences of small subgraph patterns (graphlets). For a node $v$, the orbit count $o_k(v)$ tallies how many times $v$ participates in the $k$-th orbit of graphlets with up to 5 nodes. The 73-dimensional orbit count distribution provides a fine-grained structural fingerprint.

### Spectral Statistics

The eigenvalues $\lambda_1 \leq \lambda_2 \leq \cdots \leq \lambda_n$ of the normalized Laplacian capture global graph structure:
- **Spectral gap** $\lambda_2$: measures connectivity and expansion
- **Spectral distribution**: the histogram of eigenvalues characterizes graph families

## Distribution Comparison Metrics

### Maximum Mean Discrepancy (MMD)

MMD measures the distance between two distributions in a reproducing kernel Hilbert space (RKHS). Given graph statistic samples $\{s_i^{\text{ref}}\}$ and $\{s_j^{\text{gen}}\}$:

$$
\text{MMD}^2 = \frac{1}{m^2}\sum_{i,j} k(s_i^{\text{ref}}, s_j^{\text{ref}}) + \frac{1}{m'^2}\sum_{i,j} k(s_i^{\text{gen}}, s_j^{\text{gen}}) - \frac{2}{mm'}\sum_{i,j} k(s_i^{\text{ref}}, s_j^{\text{gen}})
$$

where $k$ is a kernel function (typically Gaussian RBF or total variation). Lower MMD indicates better distributional match.

The standard graph generation benchmark computes MMD for degree, clustering, and orbit distributions:

$$
\text{MMD}_{\text{total}} = \text{MMD}_{\text{degree}} + \text{MMD}_{\text{clustering}} + \text{MMD}_{\text{orbit}}
$$

### Earth Mover's Distance (Wasserstein)

The 1-Wasserstein distance between graph statistic distributions provides a metric-aware comparison:

$$
W_1(P, Q) = \inf_{\gamma \in \Gamma(P,Q)} \int \|x - y\| \, d\gamma(x,y)
$$

For discrete distributions represented as histograms, this reduces to solving a linear program or equivalently computing the $L^1$ distance between cumulative distribution functions.

### Fréchet Graph Distance (FGD)

Analogous to FID for images, FGD uses a pretrained GNN encoder $f$ to embed graphs into a feature space, then computes the Fréchet distance between Gaussian approximations:

$$
\text{FGD} = \|\boldsymbol{\mu}_{\text{ref}} - \boldsymbol{\mu}_{\text{gen}}\|^2 + \text{Tr}\left(\boldsymbol{\Sigma}_{\text{ref}} + \boldsymbol{\Sigma}_{\text{gen}} - 2(\boldsymbol{\Sigma}_{\text{ref}}\boldsymbol{\Sigma}_{\text{gen}})^{1/2}\right)
$$

where $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ are the mean and covariance of $f(\mathcal{G})$ computed over each set.

## Validity and Uniqueness

### Validity Rate

The fraction of generated graphs satisfying domain constraints:

$$
\text{Validity} = \frac{|\{G_i^{\text{gen}} : \text{valid}(G_i^{\text{gen}})\}|}{|\{G_i^{\text{gen}}\}|}
$$

Domain-specific validity includes:
- **Molecular**: correct valency, no charged fragments, synthesizability
- **Financial**: connected components, degree bounds, weight constraints

### Uniqueness and Novelty

$$
\text{Uniqueness} = \frac{|\text{unique}(\{G_i^{\text{gen}}\})|}{|\{G_i^{\text{gen}}\}|}
$$

$$
\text{Novelty} = \frac{|\{G_i^{\text{gen}} : G_i^{\text{gen}} \notin \{G_j^{\text{ref}}\}\}|}{|\{G_i^{\text{gen}}\}|}
$$

Novelty requires graph isomorphism testing, which is computationally expensive. In practice, hash-based approximations (e.g., Weisfeiler-Lehman hash) are used.

## Implementation: Evaluation Suite

```python
"""
Evaluation metrics for graph generation.
"""
import torch
import numpy as np
from typing import Optional
from scipy.stats import wasserstein_distance
from collections import Counter


def degree_distribution(adj: torch.Tensor) -> np.ndarray:
    """Compute degree distribution as normalized histogram."""
    degrees = adj.sum(dim=1).long().cpu().numpy()
    max_deg = max(degrees.max(), 1)
    hist = np.zeros(max_deg + 1)
    for d in degrees:
        hist[d] += 1
    return hist / hist.sum()


def clustering_coefficients(adj: torch.Tensor) -> np.ndarray:
    """Compute local clustering coefficients for all nodes."""
    adj_np = adj.cpu().numpy()
    n = adj_np.shape[0]
    coeffs = np.zeros(n)

    for v in range(n):
        neighbors = np.where(adj_np[v] > 0)[0]
        k = len(neighbors)
        if k < 2:
            coeffs[v] = 0.0
            continue
        # Count edges among neighbors
        subgraph = adj_np[np.ix_(neighbors, neighbors)]
        triangles = subgraph.sum() / 2  # Each edge counted twice
        coeffs[v] = 2 * triangles / (k * (k - 1))

    return coeffs


def spectral_distribution(
    adj: torch.Tensor, num_bins: int = 50
) -> np.ndarray:
    """Compute normalized Laplacian eigenvalue distribution."""
    degree = adj.sum(dim=1)
    # Handle isolated nodes
    degree_inv_sqrt = torch.zeros_like(degree)
    mask = degree > 0
    degree_inv_sqrt[mask] = 1.0 / torch.sqrt(degree[mask])

    D_inv_sqrt = torch.diag(degree_inv_sqrt)
    L_norm = torch.eye(adj.size(0)) - D_inv_sqrt @ adj @ D_inv_sqrt

    eigenvalues = torch.linalg.eigvalsh(L_norm).cpu().numpy()
    hist, _ = np.histogram(eigenvalues, bins=num_bins, range=(0, 2), density=True)
    return hist / (hist.sum() + 1e-10)


def gaussian_rbf_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian RBF kernel between two histogram vectors."""
    return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))


def compute_mmd(
    samples_ref: list[np.ndarray],
    samples_gen: list[np.ndarray],
    kernel_fn=None,
    sigma: float = 1.0,
) -> float:
    """
    Compute Maximum Mean Discrepancy between two sets of distributions.
    
    Args:
        samples_ref: list of histogram vectors from reference graphs
        samples_gen: list of histogram vectors from generated graphs
        kernel_fn: kernel function (default: Gaussian RBF)
        sigma: kernel bandwidth
        
    Returns:
        MMD^2 value
    """
    if kernel_fn is None:
        kernel_fn = lambda x, y: gaussian_rbf_kernel(x, y, sigma)

    # Pad histograms to same length
    max_len = max(
        max(len(s) for s in samples_ref),
        max(len(s) for s in samples_gen),
    )
    ref = [np.pad(s, (0, max_len - len(s))) for s in samples_ref]
    gen = [np.pad(s, (0, max_len - len(s))) for s in samples_gen]

    m, mp = len(ref), len(gen)

    # K(ref, ref)
    k_rr = sum(kernel_fn(ref[i], ref[j]) for i in range(m) for j in range(m)) / (m * m)
    # K(gen, gen)
    k_gg = sum(kernel_fn(gen[i], gen[j]) for i in range(mp) for j in range(mp)) / (mp * mp)
    # K(ref, gen)
    k_rg = sum(kernel_fn(ref[i], gen[j]) for i in range(m) for j in range(mp)) / (m * mp)

    return float(k_rr + k_gg - 2 * k_rg)


def evaluate_generation(
    adj_ref: list[torch.Tensor],
    adj_gen: list[torch.Tensor],
) -> dict[str, float]:
    """
    Comprehensive evaluation of generated graphs against reference.
    
    Returns:
        Dictionary of metric name -> value
    """
    results = {}

    # Degree MMD
    deg_ref = [degree_distribution(a) for a in adj_ref]
    deg_gen = [degree_distribution(a) for a in adj_gen]
    results["mmd_degree"] = compute_mmd(deg_ref, deg_gen)

    # Clustering MMD
    clust_ref = [np.histogram(clustering_coefficients(a), bins=20, range=(0, 1), density=True)[0]
                 for a in adj_ref]
    clust_gen = [np.histogram(clustering_coefficients(a), bins=20, range=(0, 1), density=True)[0]
                 for a in adj_gen]
    results["mmd_clustering"] = compute_mmd(clust_ref, clust_gen)

    # Spectral MMD
    spec_ref = [spectral_distribution(a) for a in adj_ref]
    spec_gen = [spectral_distribution(a) for a in adj_gen]
    results["mmd_spectral"] = compute_mmd(spec_ref, spec_gen)

    # Basic graph statistics
    def graph_stats(adj_list):
        nodes = [a.size(0) for a in adj_list]
        edges = [a.sum().item() / 2 for a in adj_list]
        densities = [e / (n * (n - 1) / 2) if n > 1 else 0
                     for n, e in zip(nodes, edges)]
        return {
            "avg_nodes": np.mean(nodes),
            "avg_edges": np.mean(edges),
            "avg_density": np.mean(densities),
        }

    ref_stats = graph_stats(adj_ref)
    gen_stats = graph_stats(adj_gen)

    results["ref_avg_nodes"] = ref_stats["avg_nodes"]
    results["gen_avg_nodes"] = gen_stats["avg_nodes"]
    results["ref_avg_density"] = ref_stats["avg_density"]
    results["gen_avg_density"] = gen_stats["avg_density"]

    # Wasserstein distance on degree distributions
    all_deg_ref = np.concatenate([a.sum(1).cpu().numpy() for a in adj_ref])
    all_deg_gen = np.concatenate([a.sum(1).cpu().numpy() for a in adj_gen])
    results["wasserstein_degree"] = wasserstein_distance(all_deg_ref, all_deg_gen)

    return results


if __name__ == "__main__":
    # Generate reference graphs (Erdős–Rényi)
    n, p = 20, 0.15
    ref_graphs = []
    for _ in range(50):
        adj = (torch.rand(n, n) < p).float()
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()
        ref_graphs.append(adj)

    # "Generated" graphs — similar distribution
    gen_good = []
    for _ in range(50):
        adj = (torch.rand(n, n) < p).float()
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()
        gen_good.append(adj)

    # "Generated" graphs — different distribution (denser)
    gen_bad = []
    for _ in range(50):
        adj = (torch.rand(n, n) < 0.5).float()
        adj = torch.triu(adj, diagonal=1)
        adj = adj + adj.t()
        gen_bad.append(adj)

    print("=== Good Generator (same distribution) ===")
    results_good = evaluate_generation(ref_graphs, gen_good)
    for k, v in results_good.items():
        print(f"  {k}: {v:.6f}")

    print("\n=== Bad Generator (different distribution) ===")
    results_bad = evaluate_generation(ref_graphs, gen_bad)
    for k, v in results_bad.items():
        print(f"  {k}: {v:.6f}")
```
