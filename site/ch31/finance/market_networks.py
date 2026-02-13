"""
31.6.3 Synthetic Market Networks — Implementation

Correlation network construction, RMT filtering, factor-based
generation, regime-switching networks, and portfolio analytics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============================================================
# Correlation Network Construction
# ============================================================

class CorrelationNetwork:
    """
    Market correlation network with various construction methods.

    Args:
        returns: [T, N] return matrix (T periods, N assets).
        asset_names: Optional list of asset names.
        sectors: Optional sector labels per asset.
    """

    def __init__(
        self,
        returns: np.ndarray,
        asset_names: Optional[List[str]] = None,
        sectors: Optional[List[str]] = None,
    ):
        self.returns = returns
        self.T, self.N = returns.shape
        self.asset_names = asset_names or [f"Asset_{i}" for i in range(self.N)]
        self.sectors = sectors or ["Unknown"] * self.N

        # Compute correlation matrix
        self.corr_matrix = np.corrcoef(returns.T)
        # Distance matrix
        self.dist_matrix = np.sqrt(2.0 * (1.0 - np.clip(self.corr_matrix, -1, 1)))

    def threshold_network(self, theta: float = 0.5) -> np.ndarray:
        """
        Build adjacency from correlation threshold.

        Args:
            theta: Correlation threshold.

        Returns:
            [N, N] binary adjacency matrix.
        """
        adj = (np.abs(self.corr_matrix) > theta).astype(float)
        np.fill_diagonal(adj, 0)
        return adj

    def minimum_spanning_tree(self) -> np.ndarray:
        """
        Compute the MST of the distance matrix using Prim's algorithm.

        Returns:
            [N, N] adjacency of the MST (weighted by correlation).
        """
        N = self.N
        in_tree = np.zeros(N, dtype=bool)
        mst_adj = np.zeros((N, N))
        min_edge = np.full(N, np.inf)
        parent = np.full(N, -1, dtype=int)

        # Start from node 0
        min_edge[0] = 0
        for _ in range(N):
            # Find minimum edge not in tree
            candidates = np.where(~in_tree)[0]
            u = candidates[np.argmin(min_edge[candidates])]
            in_tree[u] = True

            if parent[u] >= 0:
                w = self.corr_matrix[u, parent[u]]
                mst_adj[u, parent[u]] = w
                mst_adj[parent[u], u] = w

            # Update edges
            for v in range(N):
                if not in_tree[v] and self.dist_matrix[u, v] < min_edge[v]:
                    min_edge[v] = self.dist_matrix[u, v]
                    parent[v] = u

        return mst_adj

    def network_statistics(self, adj: np.ndarray) -> Dict[str, float]:
        """Compute network statistics for an adjacency matrix."""
        binary = (np.abs(adj) > 1e-10).astype(float)
        np.fill_diagonal(binary, 0)
        degrees = binary.sum(axis=1)
        N = self.N

        stats = {
            "num_edges": int(binary.sum()) // 2,
            "density": float(binary.sum() / (N * (N - 1))),
            "mean_degree": float(degrees.mean()),
            "max_degree": float(degrees.max()),
            "mean_correlation": float(self.corr_matrix[
                np.triu_indices(N, k=1)
            ].mean()),
        }

        # Clustering coefficient (average local)
        cc = 0.0
        count = 0
        for i in range(N):
            neighbors = np.where(binary[i] > 0)[0]
            k = len(neighbors)
            if k < 2:
                continue
            # Count edges among neighbors
            sub = binary[np.ix_(neighbors, neighbors)]
            triangles = sub.sum() / 2
            possible = k * (k - 1) / 2
            cc += triangles / possible
            count += 1
        stats["avg_clustering"] = cc / count if count > 0 else 0.0

        return stats


# ============================================================
# Random Matrix Theory Filtering
# ============================================================

class RMTFilter:
    """
    Random Matrix Theory based correlation matrix denoising.

    Separates signal eigenvalues from noise eigenvalues using
    the Marchenko-Pastur distribution.

    Args:
        returns: [T, N] return matrix.
    """

    def __init__(self, returns: np.ndarray):
        self.T, self.N = returns.shape
        self.q = self.N / self.T  # ratio

        # Marchenko-Pastur bounds
        self.lambda_plus = (1 + np.sqrt(self.q)) ** 2
        self.lambda_minus = (1 - np.sqrt(self.q)) ** 2

    def filter_correlation(self, corr: np.ndarray) -> np.ndarray:
        """
        Filter the correlation matrix by removing noise eigenvalues.

        Eigenvalues within the Marchenko-Pastur band are replaced
        with their average, preserving the trace.

        Args:
            corr: [N, N] raw correlation matrix.

        Returns:
            [N, N] filtered correlation matrix.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(corr)

        # Identify signal vs noise
        signal_mask = eigenvalues > self.lambda_plus
        noise_mask = ~signal_mask

        # Replace noise eigenvalues with average
        if noise_mask.sum() > 0:
            noise_avg = eigenvalues[noise_mask].mean()
            filtered_eigenvalues = eigenvalues.copy()
            filtered_eigenvalues[noise_mask] = noise_avg

            # Reconstruct
            filtered = eigenvectors @ np.diag(filtered_eigenvalues) @ eigenvectors.T

            # Force unit diagonal
            d = np.sqrt(np.diag(filtered))
            d[d < 1e-10] = 1.0
            filtered = filtered / np.outer(d, d)
            np.fill_diagonal(filtered, 1.0)
        else:
            filtered = corr.copy()

        return filtered

    def signal_eigenvalues(self, corr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return eigenvalues and eigenvectors above the MP bound."""
        eigenvalues, eigenvectors = np.linalg.eigh(corr)
        mask = eigenvalues > self.lambda_plus
        return eigenvalues[mask], eigenvectors[:, mask]


# ============================================================
# Factor-Based Market Network Generator
# ============================================================

class FactorBasedMarketGenerator:
    """
    Generate synthetic market networks from a factor model.

    r_i(t) = alpha_i + sum_k beta_ik * f_k(t) + eps_i(t)

    By varying factor structure, diverse market networks emerge.

    Args:
        num_assets: Number of assets.
        num_factors: Number of systematic factors.
        num_periods: Number of time periods.
        num_sectors: Number of sectors (determines loading structure).
    """

    def __init__(
        self,
        num_assets: int = 50,
        num_factors: int = 5,
        num_periods: int = 252,
        num_sectors: int = 5,
    ):
        self.num_assets = num_assets
        self.num_factors = num_factors
        self.num_periods = num_periods
        self.num_sectors = num_sectors

    def generate(
        self,
        market_factor_vol: float = 0.15,
        sector_factor_vol: float = 0.10,
        idiosyncratic_vol: float = 0.20,
        regime: str = "normal",
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate synthetic returns and correlation network.

        Args:
            market_factor_vol: Volatility of the market factor.
            sector_factor_vol: Volatility of sector factors.
            idiosyncratic_vol: Idiosyncratic volatility.
            regime: 'normal', 'crisis', or 'low_vol'.

        Returns:
            (returns [T, N], correlation_matrix [N, N], sector_labels)
        """
        N = self.num_assets
        K = self.num_factors
        T = self.num_periods

        # Regime adjustments
        if regime == "crisis":
            market_factor_vol *= 2.5
            sector_factor_vol *= 1.5
            idiosyncratic_vol *= 1.5
        elif regime == "low_vol":
            market_factor_vol *= 0.5
            sector_factor_vol *= 0.5
            idiosyncratic_vol *= 0.6

        # Assign assets to sectors
        sector_labels = []
        sector_names = [f"Sector_{i}" for i in range(self.num_sectors)]
        assets_per_sector = N // self.num_sectors
        for s in range(self.num_sectors):
            count = assets_per_sector if s < self.num_sectors - 1 else N - len(sector_labels)
            sector_labels.extend([sector_names[s]] * count)

        # Factor loadings
        betas = np.zeros((N, K))

        # Market factor (factor 0): all assets have positive loading
        betas[:, 0] = np.random.uniform(0.5, 1.5, N)

        # Sector factors (factors 1..num_sectors-1)
        for s in range(min(self.num_sectors, K - 1)):
            start = s * assets_per_sector
            end = start + assets_per_sector if s < self.num_sectors - 1 else N
            betas[start:end, s + 1] = np.random.uniform(0.3, 1.0, end - start)

        # Generate factor returns
        factor_vols = np.array(
            [market_factor_vol]
            + [sector_factor_vol] * min(self.num_sectors, K - 1)
            + [sector_factor_vol * 0.5] * max(0, K - 1 - self.num_sectors)
        )[:K]

        # Factor correlation (market factor slightly correlated with sectors)
        factor_corr = np.eye(K)
        for k in range(1, K):
            factor_corr[0, k] = 0.2
            factor_corr[k, 0] = 0.2

        factor_cov = np.outer(factor_vols, factor_vols) * factor_corr
        # Ensure PSD
        eigvals = np.linalg.eigvalsh(factor_cov)
        if eigvals.min() < 0:
            factor_cov += (abs(eigvals.min()) + 1e-6) * np.eye(K)

        factor_returns = np.random.multivariate_normal(
            np.zeros(K), factor_cov / T, size=T,
        )

        # Idiosyncratic returns
        idio_vols = np.random.uniform(
            idiosyncratic_vol * 0.5, idiosyncratic_vol * 1.5, N,
        )
        idio_returns = np.random.randn(T, N) * idio_vols / np.sqrt(T)

        # Total returns
        returns = factor_returns @ betas.T + idio_returns

        # Correlation matrix
        corr = np.corrcoef(returns.T)

        return returns, corr, sector_labels


# ============================================================
# Regime-Switching Network Generator
# ============================================================

class RegimeSwitchingNetworkGenerator:
    """
    Generate sequences of market networks with regime switching.

    Uses a hidden Markov model to transition between market regimes,
    each with distinct correlation structures.

    Args:
        num_assets: Number of assets.
        transition_matrix: [num_regimes, num_regimes] Markov transition probs.
    """

    def __init__(
        self,
        num_assets: int = 30,
        transition_matrix: Optional[np.ndarray] = None,
    ):
        self.num_assets = num_assets
        self.factor_gen = FactorBasedMarketGenerator(
            num_assets=num_assets, num_factors=4, num_periods=63,
        )

        if transition_matrix is None:
            # 3 regimes: normal, crisis, low_vol
            self.transition_matrix = np.array([
                [0.90, 0.05, 0.05],  # normal
                [0.15, 0.75, 0.10],  # crisis
                [0.10, 0.05, 0.85],  # low_vol
            ])
        else:
            self.transition_matrix = transition_matrix

        self.regime_names = ["normal", "crisis", "low_vol"]

    def generate_sequence(
        self, num_windows: int = 12, initial_regime: int = 0,
    ) -> List[Dict]:
        """
        Generate a sequence of market network snapshots.

        Args:
            num_windows: Number of time windows (e.g., months).
            initial_regime: Starting regime index.

        Returns:
            List of dicts with 'regime', 'returns', 'corr', 'mst_stats'.
        """
        regime = initial_regime
        snapshots = []

        for w in range(num_windows):
            regime_name = self.regime_names[regime]

            returns, corr, sectors = self.factor_gen.generate(regime=regime_name)
            net = CorrelationNetwork(returns, sectors=sectors)
            mst = net.minimum_spanning_tree()
            stats = net.network_statistics(mst)
            stats["mean_abs_corr"] = float(np.abs(
                corr[np.triu_indices(self.num_assets, k=1)]
            ).mean())

            snapshots.append({
                "window": w,
                "regime": regime_name,
                "returns": returns,
                "corr": corr,
                "mst_stats": stats,
            })

            # Transition to next regime
            regime = np.random.choice(
                len(self.regime_names),
                p=self.transition_matrix[regime],
            )

        return snapshots


# ============================================================
# Portfolio Analytics on Market Networks
# ============================================================

class NetworkPortfolioAnalyzer:
    """
    Portfolio analytics using market network structure.

    Uses eigenvector centrality, community detection, and
    network-based diversification measures.

    Args:
        corr_network: CorrelationNetwork instance.
    """

    def __init__(self, corr_network: CorrelationNetwork):
        self.net = corr_network

    def eigenvector_centrality(self, adj: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute eigenvector centrality (leading eigenvector of adj).

        Highly central assets are more exposed to systemic risk.
        """
        if adj is None:
            adj = np.abs(self.net.corr_matrix)
            np.fill_diagonal(adj, 0)

        eigenvalues, eigenvectors = np.linalg.eigh(adj)
        centrality = np.abs(eigenvectors[:, -1])
        return centrality / centrality.sum()

    def network_diversification_score(self, weights: np.ndarray) -> float:
        """
        Compute network-based diversification score for a portfolio.

        Score is higher when portfolio weight is distributed across
        weakly connected network communities.

        Args:
            weights: [N] portfolio weight vector.

        Returns:
            Diversification score in [0, 1].
        """
        N = self.net.N
        corr = np.abs(self.net.corr_matrix)
        np.fill_diagonal(corr, 0)

        # Weighted average correlation of the portfolio
        w = np.abs(weights)
        w = w / (w.sum() + 1e-10)
        portfolio_corr = w @ corr @ w

        # Maximum possible (concentrated in one asset)
        max_corr = corr.max()

        # Score: 1 - normalized portfolio correlation
        score = 1.0 - portfolio_corr / (max_corr + 1e-10)
        return float(np.clip(score, 0, 1))

    def risk_contribution(self, weights: np.ndarray) -> np.ndarray:
        """
        Compute marginal risk contribution of each asset
        using the correlation network.
        """
        cov = self.net.corr_matrix * np.outer(
            self.net.returns.std(axis=0),
            self.net.returns.std(axis=0),
        )
        port_var = weights @ cov @ weights
        if port_var < 1e-12:
            return np.zeros(self.net.N)

        marginal = cov @ weights
        risk_contrib = weights * marginal / np.sqrt(port_var)
        return risk_contrib

    def suggest_diversifying_assets(
        self, current_assets: List[int], top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Suggest assets that would most improve portfolio diversification.

        Finds assets with lowest average correlation to current holdings.

        Args:
            current_assets: Indices of currently held assets.
            top_k: Number of suggestions.

        Returns:
            List of (asset_index, avg_correlation) tuples.
        """
        if not current_assets:
            return []

        candidates = [
            i for i in range(self.net.N) if i not in current_assets
        ]

        scores = []
        for c in candidates:
            avg_corr = np.mean([
                abs(self.net.corr_matrix[c, a]) for a in current_assets
            ])
            scores.append((c, avg_corr))

        # Sort by lowest correlation (best diversifiers)
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # ---- Factor-based generation ----
    print("=== Factor-Based Market Network ===")
    gen = FactorBasedMarketGenerator(num_assets=30, num_factors=4, num_periods=252)

    for regime in ["normal", "crisis", "low_vol"]:
        returns, corr, sectors = gen.generate(regime=regime)
        net = CorrelationNetwork(returns, sectors=sectors)

        # MST
        mst = net.minimum_spanning_tree()
        stats = net.network_statistics(mst)
        mean_corr = np.abs(corr[np.triu_indices(30, k=1)]).mean()
        print(f"\n{regime.upper()} regime:")
        print(f"  Mean |correlation|: {mean_corr:.4f}")
        print(f"  MST stats: {stats}")

    # ---- RMT Filtering ----
    print("\n=== RMT Filtering ===")
    returns_normal, corr_raw, _ = gen.generate(regime="normal")
    rmt = RMTFilter(returns_normal)
    print(f"MP bounds: [{rmt.lambda_minus:.4f}, {rmt.lambda_plus:.4f}]")

    eigenvalues = np.linalg.eigvalsh(corr_raw)
    signal_count = (eigenvalues > rmt.lambda_plus).sum()
    print(f"Signal eigenvalues: {signal_count} / {len(eigenvalues)}")

    corr_filtered = rmt.filter_correlation(corr_raw)
    print(f"Raw correlation range: [{corr_raw.min():.4f}, {corr_raw.max():.4f}]")
    print(f"Filtered correlation range: [{corr_filtered.min():.4f}, {corr_filtered.max():.4f}]")

    # ---- Threshold network ----
    print("\n=== Threshold Networks ===")
    net = CorrelationNetwork(returns_normal)
    for theta in [0.3, 0.5, 0.7]:
        adj = net.threshold_network(theta)
        stats = net.network_statistics(adj)
        print(f"  theta={theta}: edges={stats['num_edges']}, "
              f"density={stats['density']:.4f}, "
              f"clustering={stats['avg_clustering']:.4f}")

    # ---- Regime switching ----
    print("\n=== Regime-Switching Sequence ===")
    rs_gen = RegimeSwitchingNetworkGenerator(num_assets=20)
    snapshots = rs_gen.generate_sequence(num_windows=8)
    for snap in snapshots:
        s = snap["mst_stats"]
        print(f"  Window {snap['window']}: regime={snap['regime']:8s}, "
              f"mean_|corr|={s['mean_abs_corr']:.4f}, "
              f"density={s['density']:.4f}")

    # ---- Portfolio analytics ----
    print("\n=== Portfolio Analytics ===")
    net = CorrelationNetwork(returns_normal)
    analyzer = NetworkPortfolioAnalyzer(net)

    centrality = analyzer.eigenvector_centrality()
    top_central = np.argsort(centrality)[-5:][::-1]
    print(f"Most central assets: {top_central}")
    print(f"Centrality values: {centrality[top_central]}")

    # Equal-weight portfolio
    eq_weights = np.ones(30) / 30
    div_score = analyzer.network_diversification_score(eq_weights)
    print(f"\nEqual-weight diversification score: {div_score:.4f}")

    # Concentrated portfolio (first 5 assets)
    conc_weights = np.zeros(30)
    conc_weights[:5] = 0.2
    div_score_conc = analyzer.network_diversification_score(conc_weights)
    print(f"Concentrated diversification score: {div_score_conc:.4f}")

    # Suggest diversifiers
    current = [0, 1, 2, 3, 4]
    suggestions = analyzer.suggest_diversifying_assets(current, top_k=5)
    print(f"\nBest diversifiers for assets {current}:")
    for idx, avg_corr in suggestions:
        print(f"  Asset {idx}: avg |corr| = {avg_corr:.4f}")
