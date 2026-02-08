"""
31.6.1 Financial Network Generation — Implementation

Classical and deep generative models for financial networks,
systemic risk simulation (Eisenberg-Noe, DebtRank), and
conditional generation for stress testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


# ============================================================
# Financial Network Data Structure
# ============================================================

@dataclass
class FinancialNetwork:
    """
    Represents a financial network (e.g., interbank lending).

    Attributes:
        num_nodes: Number of institutions.
        adjacency: [N, N] weighted adjacency matrix.
        node_attributes: Dict of node attribute arrays.
        is_directed: Whether the network is directed.
        metadata: Additional information.
    """
    num_nodes: int
    adjacency: np.ndarray
    node_attributes: Dict[str, np.ndarray] = field(default_factory=dict)
    is_directed: bool = True
    metadata: Dict = field(default_factory=dict)

    @property
    def density(self) -> float:
        """Edge density (fraction of possible edges present)."""
        binary = (np.abs(self.adjacency) > 1e-10).astype(float)
        np.fill_diagonal(binary, 0)
        possible = self.num_nodes * (self.num_nodes - 1)
        if not self.is_directed:
            possible //= 2
            binary = np.triu(binary)
        return binary.sum() / max(possible, 1)

    @property
    def degree_sequence(self) -> np.ndarray:
        """Out-degree sequence for directed, total degree for undirected."""
        binary = (np.abs(self.adjacency) > 1e-10).astype(float)
        np.fill_diagonal(binary, 0)
        if self.is_directed:
            return binary.sum(axis=1)
        else:
            return binary.sum(axis=1) + binary.sum(axis=0)

    @property
    def total_weight(self) -> float:
        """Sum of all edge weights."""
        adj = self.adjacency.copy()
        np.fill_diagonal(adj, 0)
        return float(np.abs(adj).sum())

    def summary(self) -> Dict[str, float]:
        """Compute summary statistics."""
        degrees = self.degree_sequence
        weights = self.adjacency[np.abs(self.adjacency) > 1e-10]
        return {
            "num_nodes": self.num_nodes,
            "density": self.density,
            "total_weight": self.total_weight,
            "mean_degree": float(degrees.mean()),
            "max_degree": float(degrees.max()),
            "degree_std": float(degrees.std()),
            "mean_weight": float(np.abs(weights).mean()) if len(weights) > 0 else 0.0,
        }


# ============================================================
# Classical Financial Network Generators
# ============================================================

class ErdosRenyiFinancialGenerator:
    """
    Erdős-Rényi generator adapted for financial networks.

    Generates a random directed weighted network where each
    edge exists independently with probability p, and weights
    are drawn from a specified distribution.

    Args:
        num_nodes: Number of financial institutions.
        edge_prob: Edge probability.
        weight_distribution: 'lognormal', 'exponential', or 'pareto'.
        weight_params: Parameters for the weight distribution.
    """

    def __init__(
        self,
        num_nodes: int = 50,
        edge_prob: float = 0.1,
        weight_distribution: str = "lognormal",
        weight_params: Optional[Dict] = None,
    ):
        self.num_nodes = num_nodes
        self.edge_prob = edge_prob
        self.weight_distribution = weight_distribution
        self.weight_params = weight_params or {"mean": 0.0, "sigma": 1.0}

    def _sample_weight(self, size: int) -> np.ndarray:
        """Sample edge weights from the specified distribution."""
        if self.weight_distribution == "lognormal":
            return np.random.lognormal(
                mean=self.weight_params.get("mean", 0.0),
                sigma=self.weight_params.get("sigma", 1.0),
                size=size,
            )
        elif self.weight_distribution == "exponential":
            return np.random.exponential(
                scale=self.weight_params.get("scale", 1.0),
                size=size,
            )
        elif self.weight_distribution == "pareto":
            return np.random.pareto(
                a=self.weight_params.get("alpha", 2.0),
                size=size,
            ) + 1.0
        else:
            return np.random.uniform(0.1, 10.0, size=size)

    def generate(self) -> FinancialNetwork:
        """Generate a random financial network."""
        # Generate binary adjacency
        adj = np.random.binomial(1, self.edge_prob, (self.num_nodes, self.num_nodes)).astype(float)
        np.fill_diagonal(adj, 0)

        # Assign weights to existing edges
        num_edges = int(adj.sum())
        if num_edges > 0:
            weights = self._sample_weight(num_edges)
            adj[adj > 0] = weights

        # Node attributes: bank size (log-normal)
        bank_sizes = np.random.lognormal(mean=5.0, sigma=1.5, size=self.num_nodes)

        return FinancialNetwork(
            num_nodes=self.num_nodes,
            adjacency=adj,
            node_attributes={"bank_size": bank_sizes},
            is_directed=True,
            metadata={"generator": "erdos_renyi"},
        )


class FitnessModelGenerator:
    """
    Fitness-based financial network generator.

    Edge probabilities depend on node fitnesses (e.g., bank size),
    naturally producing heavy-tailed degree distributions.

    P(edge i→j) = sigmoid(α + β·log(x_i) + γ·log(x_j))

    Args:
        num_nodes: Number of institutions.
        alpha: Baseline log-odds.
        beta: Sender fitness coefficient.
        gamma: Receiver fitness coefficient.
        fitness_distribution: Distribution for node fitnesses.
    """

    def __init__(
        self,
        num_nodes: int = 50,
        alpha: float = -3.0,
        beta: float = 0.5,
        gamma: float = 0.5,
        fitness_distribution: str = "lognormal",
    ):
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.fitness_distribution = fitness_distribution

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    def generate(self) -> FinancialNetwork:
        """Generate a fitness-based financial network."""
        # Sample fitnesses (bank sizes)
        if self.fitness_distribution == "lognormal":
            fitnesses = np.random.lognormal(mean=5.0, sigma=1.5, size=self.num_nodes)
        elif self.fitness_distribution == "pareto":
            fitnesses = np.random.pareto(a=1.5, size=self.num_nodes) + 1.0
        else:
            fitnesses = np.random.exponential(scale=10.0, size=self.num_nodes)

        log_fitness = np.log(fitnesses + 1e-8)

        # Compute edge probabilities
        log_odds = (
            self.alpha
            + self.beta * log_fitness[:, None]
            + self.gamma * log_fitness[None, :]
        )
        probs = self._sigmoid(log_odds)
        np.fill_diagonal(probs, 0)

        # Sample edges
        adj = np.random.binomial(1, probs).astype(float)

        # Edge weights proportional to product of fitnesses
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if adj[i, j] > 0:
                    adj[i, j] = np.random.lognormal(
                        mean=np.log(np.sqrt(fitnesses[i] * fitnesses[j])),
                        sigma=0.5,
                    )

        return FinancialNetwork(
            num_nodes=self.num_nodes,
            adjacency=adj,
            node_attributes={
                "bank_size": fitnesses,
                "log_fitness": log_fitness,
            },
            is_directed=True,
            metadata={"generator": "fitness_model"},
        )


class CorePeripheryGenerator:
    """
    Core-periphery network generator for interbank markets.

    Explicitly models a dense core of large banks and a sparse
    periphery connected primarily to the core.

    Args:
        num_core: Number of core institutions.
        num_periphery: Number of periphery institutions.
        core_core_prob: Edge probability within the core.
        core_periphery_prob: Edge probability between core and periphery.
        periphery_periphery_prob: Edge probability within the periphery.
    """

    def __init__(
        self,
        num_core: int = 10,
        num_periphery: int = 40,
        core_core_prob: float = 0.6,
        core_periphery_prob: float = 0.15,
        periphery_periphery_prob: float = 0.02,
    ):
        self.num_core = num_core
        self.num_periphery = num_periphery
        self.num_nodes = num_core + num_periphery
        self.cc_prob = core_core_prob
        self.cp_prob = core_periphery_prob
        self.pp_prob = periphery_periphery_prob

    def generate(self) -> FinancialNetwork:
        """Generate a core-periphery financial network."""
        n = self.num_nodes
        nc = self.num_core

        # Block probability matrix
        adj = np.zeros((n, n))

        # Core-core
        adj[:nc, :nc] = np.random.binomial(1, self.cc_prob, (nc, nc))
        # Core-periphery (both directions)
        adj[:nc, nc:] = np.random.binomial(1, self.cp_prob, (nc, n - nc))
        adj[nc:, :nc] = np.random.binomial(1, self.cp_prob, (n - nc, nc))
        # Periphery-periphery
        adj[nc:, nc:] = np.random.binomial(1, self.pp_prob, (n - nc, n - nc))

        np.fill_diagonal(adj, 0)
        adj = adj.astype(float)

        # Assign weights: core transactions are larger
        core_size = np.random.lognormal(7.0, 0.5, nc)
        periph_size = np.random.lognormal(4.0, 1.0, n - nc)
        sizes = np.concatenate([core_size, periph_size])

        for i in range(n):
            for j in range(n):
                if adj[i, j] > 0:
                    adj[i, j] = np.random.lognormal(
                        mean=np.log(np.sqrt(sizes[i] * sizes[j]) * 0.01),
                        sigma=0.5,
                    )

        # Node attributes
        is_core = np.array([1] * nc + [0] * (n - nc))

        return FinancialNetwork(
            num_nodes=n,
            adjacency=adj,
            node_attributes={
                "bank_size": sizes,
                "is_core": is_core,
            },
            is_directed=True,
            metadata={"generator": "core_periphery", "num_core": nc},
        )


# ============================================================
# Systemic Risk Models
# ============================================================

def eisenberg_noe_clearing(
    obligations: np.ndarray,
    external_assets: np.ndarray,
    max_iterations: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eisenberg-Noe clearing payment vector.

    Finds the fixed-point payment vector when banks may default
    on their interbank obligations.

    Args:
        obligations: [N, N] matrix where obligations[i,j] = amount i owes j.
        external_assets: [N] external (non-interbank) assets.
        max_iterations: Maximum fixed-point iterations.
        tol: Convergence tolerance.

    Returns:
        (payments, default_indicator)
        payments: [N] actual payment vector.
        default_indicator: [N] binary array (1 = defaulted).
    """
    n = obligations.shape[0]
    total_obligations = obligations.sum(axis=1)  # total each bank owes

    # Relative obligation matrix: Pi[i,j] = fraction of i's debt owed to j
    Pi = np.zeros_like(obligations)
    for i in range(n):
        if total_obligations[i] > 1e-10:
            Pi[i, :] = obligations[i, :] / total_obligations[i]

    # Initialize payments at full obligations
    payments = total_obligations.copy()

    for iteration in range(max_iterations):
        # Assets = external assets + interbank receipts
        receipts = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if total_obligations[j] > 1e-10:
                    receipts[i] += Pi[j, i] * payments[j]

        total_assets = external_assets + receipts

        # Each bank pays min(obligations, total_assets)
        new_payments = np.minimum(total_obligations, total_assets)
        new_payments = np.maximum(new_payments, 0.0)  # non-negative

        if np.max(np.abs(new_payments - payments)) < tol:
            payments = new_payments
            break
        payments = new_payments

    defaults = (payments < total_obligations - tol).astype(float)
    return payments, defaults


def debtrank(
    network: FinancialNetwork,
    shocked_nodes: List[int],
    shock_magnitude: float = 1.0,
) -> Dict[str, float]:
    """
    Compute DebtRank systemic importance metric.

    Measures the fraction of total economic value lost when
    a set of nodes experiences distress.

    Args:
        network: FinancialNetwork instance.
        shocked_nodes: Indices of initially shocked institutions.
        shock_magnitude: Initial shock to equity (0 to 1).

    Returns:
        Dict with debtrank score, num_defaults, cascade_rounds.
    """
    n = network.num_nodes
    adj = network.adjacency.copy()
    np.fill_diagonal(adj, 0)

    # Normalize: leverage matrix L[i,j] = exposure of j to i / equity of j
    sizes = network.node_attributes.get("bank_size", np.ones(n))
    equity = sizes * 0.1  # assume 10% equity ratio

    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if equity[j] > 1e-10 and adj[i, j] > 0:
                L[i, j] = adj[i, j] / equity[j]

    # Initialize distress
    h = np.zeros(n)  # distress level [0, 1]
    for node in shocked_nodes:
        h[node] = min(shock_magnitude, 1.0)

    status = np.zeros(n)  # 0=active, 1=distressed, 2=inactive
    for node in shocked_nodes:
        status[node] = 1

    # Cascade
    total_rounds = 0
    for round_num in range(n):  # max n rounds
        new_h = h.copy()
        any_change = False

        for j in range(n):
            if status[j] == 2:  # already inactive
                continue

            # Sum distress from neighbors
            distress_input = 0.0
            for i in range(n):
                if status[i] == 1:  # recently distressed
                    distress_input += L[i, j] * h[i]

            if distress_input > 1e-10:
                new_h[j] = min(h[j] + distress_input, 1.0)
                if new_h[j] > h[j]:
                    any_change = True

        # Update status
        for i in range(n):
            if status[i] == 1:
                status[i] = 2  # move to inactive
            if new_h[i] > h[i] and status[i] == 0:
                status[i] = 1  # newly distressed

        h = new_h
        total_rounds += 1

        if not any_change:
            break

    # DebtRank = weighted sum of distress
    weights = sizes / sizes.sum()
    initial_distress = np.zeros(n)
    for node in shocked_nodes:
        initial_distress[node] = shock_magnitude
    dr = float(np.sum(weights * (h - initial_distress)))

    return {
        "debtrank": dr,
        "num_defaults": int((h > 0.99).sum()),
        "cascade_rounds": total_rounds,
        "mean_distress": float(h.mean()),
        "max_distress": float(h.max()),
    }


# ============================================================
# Network Comparison Metrics
# ============================================================

class FinancialNetworkMetrics:
    """
    Compare generated financial networks against reference networks
    on domain-specific statistics.
    """

    @staticmethod
    def degree_distribution_distance(net_a: FinancialNetwork, net_b: FinancialNetwork) -> float:
        """KL divergence between degree distributions."""
        deg_a = net_a.degree_sequence
        deg_b = net_b.degree_sequence

        max_deg = max(int(deg_a.max()), int(deg_b.max())) + 1
        hist_a = np.histogram(deg_a, bins=max_deg, range=(0, max_deg), density=True)[0] + 1e-10
        hist_b = np.histogram(deg_b, bins=max_deg, range=(0, max_deg), density=True)[0] + 1e-10

        # Symmetric KL
        kl_ab = np.sum(hist_a * np.log(hist_a / hist_b))
        kl_ba = np.sum(hist_b * np.log(hist_b / hist_a))
        return 0.5 * (kl_ab + kl_ba)

    @staticmethod
    def weight_distribution_distance(net_a: FinancialNetwork, net_b: FinancialNetwork) -> float:
        """KS statistic between edge weight distributions."""
        from scipy.stats import ks_2samp

        w_a = net_a.adjacency[net_a.adjacency > 1e-10].flatten()
        w_b = net_b.adjacency[net_b.adjacency > 1e-10].flatten()

        if len(w_a) == 0 or len(w_b) == 0:
            return 1.0

        stat, _ = ks_2samp(w_a, w_b)
        return float(stat)

    @staticmethod
    def compare(net_a: FinancialNetwork, net_b: FinancialNetwork) -> Dict[str, float]:
        """Full comparison between two financial networks."""
        summary_a = net_a.summary()
        summary_b = net_b.summary()

        return {
            "density_diff": abs(summary_a["density"] - summary_b["density"]),
            "mean_degree_diff": abs(summary_a["mean_degree"] - summary_b["mean_degree"]),
            "degree_dist_kl": FinancialNetworkMetrics.degree_distribution_distance(net_a, net_b),
        }


# ============================================================
# Deep Generative Model: VAE for Financial Networks
# ============================================================

class FinancialNetworkVAE(nn.Module):
    """
    Variational Autoencoder for financial network generation.

    Encodes adjacency + node features into a latent space, then
    decodes to reconstruct the network.

    Args:
        max_nodes: Maximum number of nodes.
        node_feat_dim: Dimension of node features.
        latent_dim: Latent space dimension.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(
        self,
        max_nodes: int = 50,
        node_feat_dim: int = 4,
        latent_dim: int = 64,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim

        input_dim = max_nodes * max_nodes + max_nodes * node_feat_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.adj_head = nn.Linear(hidden_dim, max_nodes * max_nodes)
        self.feat_head = nn.Linear(hidden_dim, max_nodes * node_feat_dim)

    def encode(self, adj_flat: torch.Tensor, feat_flat: torch.Tensor):
        x = torch.cat([adj_flat, feat_flat], dim=-1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor):
        h = self.decoder(z)
        adj_logits = self.adj_head(h)
        feat_pred = self.feat_head(h)
        return adj_logits, feat_pred

    def forward(self, adj_flat, feat_flat):
        mu, logvar = self.encode(adj_flat, feat_flat)
        z = self.reparameterize(mu, logvar)
        adj_logits, feat_pred = self.decode(z)
        return adj_logits, feat_pred, mu, logvar

    def loss_function(self, adj_logits, adj_target, feat_pred, feat_target, mu, logvar):
        recon_adj = F.binary_cross_entropy_with_logits(adj_logits, adj_target, reduction="mean")
        recon_feat = F.mse_loss(feat_pred, feat_target)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_adj + recon_feat + 0.01 * kl

    @torch.no_grad()
    def generate(self, num_samples: int = 1, device: torch.device = torch.device("cpu")):
        """Generate financial networks from the prior."""
        self.eval()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        adj_logits, feat_pred = self.decode(z)

        adj = torch.sigmoid(adj_logits).view(num_samples, self.max_nodes, self.max_nodes)
        # Threshold to binary
        adj = (adj > 0.5).float()
        # Remove self-loops
        mask = 1.0 - torch.eye(self.max_nodes, device=device)
        adj = adj * mask

        return adj.cpu().numpy(), feat_pred.cpu().numpy()


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    np.random.seed(42)

    # ---- Classical generators ----
    print("=== Erdős-Rényi Financial Network ===")
    er_gen = ErdosRenyiFinancialGenerator(num_nodes=30, edge_prob=0.1)
    er_net = er_gen.generate()
    print(er_net.summary())

    print("\n=== Fitness Model ===")
    fit_gen = FitnessModelGenerator(num_nodes=30, alpha=-3.0)
    fit_net = fit_gen.generate()
    print(fit_net.summary())

    print("\n=== Core-Periphery Model ===")
    cp_gen = CorePeripheryGenerator(num_core=5, num_periphery=25)
    cp_net = cp_gen.generate()
    print(cp_net.summary())
    core_degrees = cp_net.degree_sequence[:5]
    periph_degrees = cp_net.degree_sequence[5:]
    print(f"Core mean degree: {core_degrees.mean():.1f}")
    print(f"Periphery mean degree: {periph_degrees.mean():.1f}")

    # ---- Eisenberg-Noe clearing ----
    print("\n=== Eisenberg-Noe Clearing Vector ===")
    obligations = cp_net.adjacency.copy()
    external_assets = cp_net.node_attributes["bank_size"] * 0.5
    payments, defaults = eisenberg_noe_clearing(obligations, external_assets)
    print(f"Defaults: {int(defaults.sum())} / {cp_net.num_nodes}")
    print(f"Total payments: {payments.sum():.2f}")
    print(f"Total obligations: {obligations.sum():.2f}")

    # ---- DebtRank ----
    print("\n=== DebtRank ===")
    # Shock the largest core bank
    dr_result = debtrank(cp_net, shocked_nodes=[0], shock_magnitude=1.0)
    print(f"DebtRank results: {dr_result}")

    # ---- Network comparison ----
    print("\n=== Network Comparison ===")
    comparison = FinancialNetworkMetrics.compare(er_net, cp_net)
    print(f"ER vs Core-Periphery: {comparison}")

    # ---- VAE demo ----
    print("\n=== Financial Network VAE ===")
    vae = FinancialNetworkVAE(max_nodes=30, node_feat_dim=2, latent_dim=32)
    print(f"Parameters: {sum(p.numel() for p in vae.parameters()):,}")
    gen_adj, gen_feat = vae.generate(num_samples=2)
    print(f"Generated adjacency shape: {gen_adj.shape}")
    print(f"Generated density: {(gen_adj[0] > 0).sum() / (30*29):.4f}")
