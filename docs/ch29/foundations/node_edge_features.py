"""
Chapter 29.1.4: Node and Edge Features
Feature engineering for graph neural networks.
"""

import numpy as np
import networkx as nx
import torch
from typing import Dict, List, Optional, Tuple


# ============================================================
# 1. Node Feature Engineering
# ============================================================

class NodeFeatureExtractor:
    """Extract structural and positional features for graph nodes."""

    def __init__(self, G: nx.Graph):
        self.G = G
        self.n = G.number_of_nodes()

    def degree_features(self) -> np.ndarray:
        degrees = np.array([self.G.degree(n) for n in range(self.n)])
        return degrees.reshape(-1, 1)

    def normalized_degree(self) -> np.ndarray:
        degrees = self.degree_features()
        max_deg = degrees.max()
        return degrees / max_deg if max_deg > 0 else degrees

    def clustering_coefficient(self) -> np.ndarray:
        cc = nx.clustering(self.G)
        return np.array([cc[n] for n in range(self.n)]).reshape(-1, 1)

    def pagerank(self, alpha: float = 0.85) -> np.ndarray:
        pr = nx.pagerank(self.G, alpha=alpha)
        return np.array([pr[n] for n in range(self.n)]).reshape(-1, 1)

    def betweenness_centrality(self) -> np.ndarray:
        bc = nx.betweenness_centrality(self.G)
        return np.array([bc[n] for n in range(self.n)]).reshape(-1, 1)

    def eigenvector_centrality(self, max_iter: int = 1000) -> np.ndarray:
        try:
            ec = nx.eigenvector_centrality(self.G, max_iter=max_iter)
            return np.array([ec[n] for n in range(self.n)]).reshape(-1, 1)
        except nx.PowerIterationFailedConvergence:
            return np.zeros((self.n, 1))

    def closeness_centrality(self) -> np.ndarray:
        cc = nx.closeness_centrality(self.G)
        return np.array([cc[n] for n in range(self.n)]).reshape(-1, 1)

    def triangle_count(self) -> np.ndarray:
        triangles = nx.triangles(self.G)
        return np.array([triangles[n] for n in range(self.n)]).reshape(-1, 1)

    def laplacian_positional_encoding(self, k: int = 4) -> np.ndarray:
        """Positional encodings from k smallest non-trivial Laplacian eigenvectors."""
        A = nx.adjacency_matrix(self.G).toarray().astype(float)
        D = np.diag(A.sum(axis=1))
        L = D - A
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        k = min(k, self.n - 1)
        return eigenvectors[:, 1:k + 1]

    def random_walk_encoding(self, walk_length: int = 4) -> np.ndarray:
        """Random walk structural encoding: diag(M^k) for k = 1..walk_length."""
        A = nx.adjacency_matrix(self.G).toarray().astype(float)
        D_inv = np.diag(1.0 / np.maximum(A.sum(axis=1), 1e-10))
        M = D_inv @ A
        features = []
        Mk = np.eye(self.n)
        for _ in range(walk_length):
            Mk = Mk @ M
            features.append(np.diag(Mk).reshape(-1, 1))
        return np.hstack(features)

    def one_hot_degree(self, max_degree: int = 10) -> np.ndarray:
        """One-hot encoding of node degree."""
        degrees = np.array([min(self.G.degree(n), max_degree)
                            for n in range(self.n)])
        one_hot = np.zeros((self.n, max_degree + 1))
        one_hot[np.arange(self.n), degrees] = 1
        return one_hot

    def get_all_structural_features(self) -> np.ndarray:
        """Concatenate all structural features."""
        features = [
            self.normalized_degree(),
            self.clustering_coefficient(),
            self.pagerank(),
            self.betweenness_centrality(),
            self.closeness_centrality(),
            self.triangle_count() / max(1, self.G.number_of_edges()),
        ]
        return np.hstack(features)


def demo_node_features():
    """Demonstrate node feature extraction."""
    print("=" * 60)
    print("Node Feature Extraction")
    print("=" * 60)

    G = nx.karate_club_graph()
    extractor = NodeFeatureExtractor(G)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    features = extractor.get_all_structural_features()
    print(f"\nStructural features shape: {features.shape}")

    feature_names = ['Degree', 'Clustering', 'PageRank',
                     'Betweenness', 'Closeness', 'TriangleRatio']
    for i, name in enumerate(feature_names):
        top5 = np.argsort(features[:, i])[::-1][:5]
        vals = features[top5, i]
        print(f"  Top 5 by {name}: "
              f"{list(zip(top5.tolist(), np.round(vals, 4).tolist()))}")

    pos_enc = extractor.laplacian_positional_encoding(k=4)
    print(f"\nLaplacian PE shape: {pos_enc.shape}")
    print(f"First 3 nodes:\n{np.round(pos_enc[:3], 4)}")

    rw_enc = extractor.random_walk_encoding(walk_length=4)
    print(f"\nRandom walk encoding shape: {rw_enc.shape}")
    print(f"First 3 nodes:\n{np.round(rw_enc[:3], 4)}")

    return features, pos_enc


# ============================================================
# 2. Edge Feature Engineering
# ============================================================

class EdgeFeatureExtractor:
    """Extract features for graph edges."""

    def __init__(self, G: nx.Graph):
        self.G = G
        self.edges = list(G.edges())

    def common_neighbors(self) -> np.ndarray:
        cn = []
        for u, v in self.edges:
            cn.append(len(list(nx.common_neighbors(self.G, u, v))))
        return np.array(cn).reshape(-1, 1)

    def jaccard_coefficient(self) -> np.ndarray:
        jc = []
        preds = nx.jaccard_coefficient(self.G, self.edges)
        for u, v, p in preds:
            jc.append(p)
        return np.array(jc).reshape(-1, 1)

    def adamic_adar(self) -> np.ndarray:
        aa = []
        preds = nx.adamic_adar_index(self.G, self.edges)
        for u, v, p in preds:
            aa.append(p)
        return np.array(aa).reshape(-1, 1)

    def edge_betweenness(self) -> np.ndarray:
        eb = nx.edge_betweenness_centrality(self.G)
        values = [eb.get((u, v), eb.get((v, u), 0)) for u, v in self.edges]
        return np.array(values).reshape(-1, 1)

    def get_all_features(self) -> np.ndarray:
        return np.hstack([
            self.common_neighbors(),
            self.jaccard_coefficient(),
            self.adamic_adar(),
            self.edge_betweenness(),
        ])


def demo_edge_features():
    """Demonstrate edge feature extraction."""
    print("\n" + "=" * 60)
    print("Edge Feature Extraction")
    print("=" * 60)

    G = nx.karate_club_graph()
    extractor = EdgeFeatureExtractor(G)
    features = extractor.get_all_features()

    print(f"Edge feature shape: {features.shape}")
    print(f"Feature names: common_neighbors, jaccard, adamic_adar, betweenness")
    print(f"\nFirst 5 edges with features:")
    edges = list(G.edges())
    for i in range(5):
        print(f"  Edge {edges[i]}: {np.round(features[i], 4)}")


# ============================================================
# 3. Feature Normalization
# ============================================================

class FeatureNormalizer:
    """Normalize graph features with various strategies."""

    @staticmethod
    def standard_normalize(X: np.ndarray) -> np.ndarray:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        return (X - mean) / std

    @staticmethod
    def minmax_normalize(X: np.ndarray) -> np.ndarray:
        xmin = X.min(axis=0)
        xmax = X.max(axis=0)
        denom = xmax - xmin
        denom[denom == 0] = 1.0
        return (X - xmin) / denom

    @staticmethod
    def l2_normalize(X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms


def demo_normalization():
    """Demonstrate feature normalization."""
    print("\n" + "=" * 60)
    print("Feature Normalization")
    print("=" * 60)

    G = nx.karate_club_graph()
    extractor = NodeFeatureExtractor(G)
    X = extractor.get_all_structural_features()

    print(f"Raw features - mean: {np.round(X.mean(axis=0), 4)}")
    print(f"Raw features - std: {np.round(X.std(axis=0), 4)}")

    X_std = FeatureNormalizer.standard_normalize(X)
    print(f"\nStandard normalized - mean: {np.round(X_std.mean(axis=0), 4)}")
    print(f"Standard normalized - std: {np.round(X_std.std(axis=0), 4)}")

    X_mm = FeatureNormalizer.minmax_normalize(X)
    print(f"\nMinMax normalized - min: {np.round(X_mm.min(axis=0), 4)}")
    print(f"MinMax normalized - max: {np.round(X_mm.max(axis=0), 4)}")


# ============================================================
# 4. Financial Graph Features
# ============================================================

def create_financial_graph_features():
    """Create node and edge features for a financial network."""
    print("\n" + "=" * 60)
    print("Financial Graph Features")
    print("=" * 60)

    np.random.seed(42)
    n_assets = 8
    asset_names = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "BAC", "WFC"]
    sectors = ["Tech", "Tech", "Tech", "Tech", "Finance", "Finance", "Finance", "Finance"]

    # Simulate returns
    n_days = 252
    market = np.random.randn(n_days) * 0.01
    tech_factor = np.random.randn(n_days) * 0.005
    fin_factor = np.random.randn(n_days) * 0.005

    returns = np.column_stack([
        market + tech_factor + np.random.randn(n_days) * 0.003,
        market + tech_factor + np.random.randn(n_days) * 0.004,
        market + tech_factor + np.random.randn(n_days) * 0.003,
        market + tech_factor + np.random.randn(n_days) * 0.005,
        market + fin_factor + np.random.randn(n_days) * 0.004,
        market + fin_factor + np.random.randn(n_days) * 0.003,
        market + fin_factor + np.random.randn(n_days) * 0.004,
        market + fin_factor + np.random.randn(n_days) * 0.003,
    ])

    # Node features
    node_features = {
        'mean_return': returns.mean(axis=0),
        'volatility': returns.std(axis=0),
        'sharpe': returns.mean(axis=0) / returns.std(axis=0),
        'skewness': np.array([
            np.mean(((r - r.mean()) / r.std()) ** 3) for r in returns.T
        ]),
        'kurtosis': np.array([
            np.mean(((r - r.mean()) / r.std()) ** 4) - 3 for r in returns.T
        ]),
        'max_drawdown': np.array([
            np.min(np.cumsum(r) - np.maximum.accumulate(np.cumsum(r)))
            for r in returns.T
        ]),
    }

    # Sector one-hot encoding
    unique_sectors = list(set(sectors))
    sector_onehot = np.zeros((n_assets, len(unique_sectors)))
    for i, s in enumerate(sectors):
        sector_onehot[i, unique_sectors.index(s)] = 1

    # Combine all node features
    X = np.column_stack([node_features[k] for k in node_features])
    X = np.hstack([X, sector_onehot])

    print(f"Node feature matrix shape: {X.shape}")
    print(f"Features: {list(node_features.keys())} + sector_onehot")
    for i, name in enumerate(asset_names):
        print(f"  {name}: {np.round(X[i, :6], 4)}")

    # Build correlation graph
    corr = np.corrcoef(returns.T)
    threshold = 0.3
    A = (np.abs(corr) > threshold).astype(float)
    np.fill_diagonal(A, 0)

    # Edge features: correlation, rolling correlation std
    G = nx.from_numpy_array(A)
    edges = list(G.edges())

    edge_features = []
    for u, v in edges:
        # Correlation value
        corr_val = corr[u, v]
        # Rolling correlation std (measure of stability)
        window = 60
        rolling_corrs = []
        for t in range(window, n_days):
            rc = np.corrcoef(returns[t-window:t, u], returns[t-window:t, v])[0, 1]
            rolling_corrs.append(rc)
        corr_std = np.std(rolling_corrs)
        # Same sector indicator
        same_sector = float(sectors[u] == sectors[v])
        edge_features.append([corr_val, corr_std, same_sector])

    E = np.array(edge_features)
    print(f"\nEdge feature matrix shape: {E.shape}")
    print(f"Edge features: [correlation, corr_volatility, same_sector]")
    for i, (u, v) in enumerate(edges[:5]):
        print(f"  {asset_names[u]}-{asset_names[v]}: {np.round(E[i], 4)}")

    # Convert to PyTorch tensors
    x_tensor = torch.tensor(X, dtype=torch.float32)
    edge_index = torch.tensor([[u, v] for u, v in edges] +
                               [[v, u] for u, v in edges], dtype=torch.long).T
    edge_attr = torch.tensor(
        np.vstack([E, E]),  # Duplicate for both directions
        dtype=torch.float32
    )

    print(f"\nPyTorch tensors:")
    print(f"  x: {x_tensor.shape}")
    print(f"  edge_index: {edge_index.shape}")
    print(f"  edge_attr: {edge_attr.shape}")

    return x_tensor, edge_index, edge_attr


# ============================================================
# 5. PyTorch Geometric Data Object
# ============================================================

def create_pyg_data_object():
    """Create a complete PyG Data object with features."""
    print("\n" + "=" * 60)
    print("PyTorch Geometric Data Object")
    print("=" * 60)

    # Simple example graph
    num_nodes = 6
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (4, 5)]

    # Build edge_index (bidirectional)
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Node features: 4-dimensional
    x = torch.randn(num_nodes, 4)

    # Edge attributes: 2-dimensional
    edge_attr = torch.randn(edge_index.shape[1], 2)

    # Node labels (for classification)
    y = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    # Create Data object (simulating PyG's Data)
    data = {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'y': y,
        'num_nodes': num_nodes,
    }

    print(f"Data object:")
    print(f"  x (node features): {data['x'].shape}")
    print(f"  edge_index: {data['edge_index'].shape}")
    print(f"  edge_attr: {data['edge_attr'].shape}")
    print(f"  y (labels): {data['y'].shape}")
    print(f"  num_nodes: {data['num_nodes']}")

    # Verify edge_index
    print(f"\n  Unique source nodes: {torch.unique(edge_index[0]).tolist()}")
    print(f"  Unique target nodes: {torch.unique(edge_index[1]).tolist()}")
    print(f"  Number of directed edges: {edge_index.shape[1]}")

    return data


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    features, pos_enc = demo_node_features()
    demo_edge_features()
    demo_normalization()
    create_financial_graph_features()
    create_pyg_data_object()
