"""
31.5.3 3D Molecule Generation — Implementation

Equivariant neural networks (SchNet, EGNN) and equivariant diffusion
models for generating molecules with 3D coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Optional, Tuple


# ============================================================
# Radial Basis Functions
# ============================================================

class GaussianRBF(nn.Module):
    """
    Gaussian radial basis function expansion for distance encoding.

    Args:
        num_rbf: Number of basis functions.
        cutoff: Maximum distance (Angstroms).
    """

    def __init__(self, num_rbf: int = 50, cutoff: float = 10.0):
        super().__init__()
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, num_rbf)
        self.register_buffer("centers", centers)
        self.width = (cutoff / num_rbf) * 0.5

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        if distances.dim() == 1:
            distances = distances.unsqueeze(-1)
        return torch.exp(-((distances - self.centers) ** 2) / (2 * self.width ** 2))


class CosineCutoff(nn.Module):
    """Smooth cutoff that decays to zero at the cutoff distance."""

    def __init__(self, cutoff: float = 10.0):
        super().__init__()
        self.cutoff = cutoff

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        mask = (distances <= self.cutoff).float()
        return 0.5 * (torch.cos(math.pi * distances / self.cutoff) + 1.0) * mask


# ============================================================
# SchNet Interaction Block
# ============================================================

class SchNetInteraction(nn.Module):
    """
    SchNet continuous-filter convolution block.
    SE(3)-invariant: depends only on pairwise distances.

    Args:
        hidden_dim: Feature dimension.
        num_rbf: Number of radial basis functions.
        cutoff: Distance cutoff.
    """

    def __init__(self, hidden_dim: int = 128, num_rbf: int = 50, cutoff: float = 10.0):
        super().__init__()
        self.rbf = GaussianRBF(num_rbf, cutoff)
        self.cutoff_fn = CosineCutoff(cutoff)
        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.pre_linear = nn.Linear(hidden_dim, hidden_dim)
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        num_atoms = h.size(0)
        rbf_vals = self.rbf(distances)
        cutoff_vals = self.cutoff_fn(distances)
        W = self.filter_net(rbf_vals) * cutoff_vals.unsqueeze(-1)
        messages = self.pre_linear(h[src]) * W
        agg = torch.zeros(num_atoms, h.size(1), device=h.device)
        agg.index_add_(0, dst, messages)
        return h + self.post_mlp(agg)


class SchNet(nn.Module):
    """
    SchNet: SE(3)-invariant molecular representation from 3D coordinates.

    Args:
        num_atom_types: Number of atom categories.
        hidden_dim: Feature dimension.
        num_interactions: Number of interaction layers.
        num_rbf: Radial basis functions.
        cutoff: Distance cutoff.
    """

    def __init__(self, num_atom_types=10, hidden_dim=128, num_interactions=3, num_rbf=50, cutoff=10.0):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_dim, num_rbf, cutoff) for _ in range(num_interactions)
        ])
        self.output_net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, atom_types, positions, edge_index):
        src, dst = edge_index
        distances = (positions[src] - positions[dst]).norm(dim=-1)
        h = self.atom_embedding(atom_types)
        for interaction in self.interactions:
            h = interaction(h, edge_index, distances)
        return h, self.output_net(h).squeeze(-1)


# ============================================================
# EGNN Layer and Model
# ============================================================

class EGNNLayer(nn.Module):
    """
    E(n) Equivariant GNN layer: updates features (invariant)
    and coordinates (equivariant) simultaneously.

    Args:
        hidden_dim: Feature dimension.
        coord_update_clamp: Clamp for coordinate update magnitude.
    """

    def __init__(self, hidden_dim: int = 128, coord_update_clamp: float = 10.0):
        super().__init__()
        self.coord_update_clamp = coord_update_clamp
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
        )
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, pos, edge_index):
        src, dst = edge_index
        num_atoms = h.size(0)
        diff = pos[src] - pos[dst]
        dist_sq = (diff ** 2).sum(dim=-1, keepdim=True)
        msg_input = torch.cat([h[src], h[dst], dist_sq], dim=-1)
        messages = self.message_mlp(msg_input)

        # Equivariant coordinate update
        coord_weights = self.coord_mlp(messages).clamp(-self.coord_update_clamp, self.coord_update_clamp)
        coord_delta = torch.zeros_like(pos)
        coord_delta.index_add_(0, dst, diff * coord_weights)
        pos_new = pos + coord_delta

        # Invariant feature update
        msg_agg = torch.zeros(num_atoms, messages.size(1), device=h.device)
        msg_agg.index_add_(0, dst, messages)
        h_new = h + self.node_mlp(torch.cat([h, msg_agg], dim=-1))
        return h_new, pos_new


class EGNN(nn.Module):
    """
    E(n) Equivariant Graph Neural Network.

    Args:
        num_atom_types: Number of atom categories.
        hidden_dim: Feature dimension.
        num_layers: Number of EGNN layers.
    """

    def __init__(self, num_atom_types=10, hidden_dim=128, num_layers=4):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])

    def forward(self, atom_types, positions, edge_index):
        h = self.atom_embedding(atom_types)
        for layer in self.layers:
            h, positions = layer(h, positions, edge_index)
        return h, positions


# ============================================================
# Equivariant Diffusion Model for 3D Molecules
# ============================================================

class EquivariantDiffusion3D(nn.Module):
    """
    Equivariant diffusion model for joint generation of atom
    types and 3D coordinates.

    Args:
        num_atom_types: Number of atom categories.
        hidden_dim: EGNN hidden dimension.
        num_layers: EGNN depth.
        num_timesteps: Diffusion steps.
    """

    def __init__(self, num_atom_types=10, hidden_dim=128, num_layers=4, num_timesteps=500):
        super().__init__()
        self.num_atom_types = num_atom_types
        self.num_timesteps = num_timesteps

        # Linear beta schedule
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)

        # Time embedding
        self.time_emb = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

        # Input projection (noisy atom types + time → hidden)
        self.atom_input = nn.Linear(num_atom_types + hidden_dim, hidden_dim)

        # EGNN layers for denoising
        self.egnn_layers = nn.ModuleList([EGNNLayer(hidden_dim) for _ in range(num_layers)])

        # Output heads
        self.coord_head = nn.Linear(hidden_dim, 3)
        self.atom_head = nn.Linear(hidden_dim, num_atom_types)

    @staticmethod
    def _center(pos: torch.Tensor) -> torch.Tensor:
        """Remove center of mass."""
        return pos - pos.mean(dim=0, keepdim=True)

    def forward_diffusion(self, atom_types_oh, positions, t):
        """Apply forward noise at timestep t."""
        ab = self.alpha_bar[t]
        eps = torch.randn_like(positions)
        noisy_pos = torch.sqrt(ab) * positions + torch.sqrt(1 - ab) * eps
        noisy_pos = self._center(noisy_pos)
        uniform = torch.ones_like(atom_types_oh) / self.num_atom_types
        noisy_types = ab * atom_types_oh + (1 - ab) * uniform
        return noisy_types, noisy_pos, eps

    def denoise(self, noisy_types, noisy_pos, edge_index, t):
        """Predict coordinate noise and clean atom types."""
        N = noisy_types.size(0)
        t_emb = self.time_emb(t.view(1, 1).float() / self.num_timesteps).expand(N, -1)
        h = self.atom_input(torch.cat([noisy_types, t_emb], dim=-1))

        pos = noisy_pos
        for layer in self.egnn_layers:
            h, pos = layer(h, pos, edge_index)

        return self.coord_head(h), self.atom_head(h)

    def training_loss(self, atom_types, positions, edge_index):
        """
        Compute combined training loss.

        Args:
            atom_types: [N] integer atom types.
            positions: [N, 3] clean coordinates.
            edge_index: [2, E] edges.

        Returns:
            Dict with total_loss, coord_loss, atom_loss.
        """
        device = atom_types.device
        types_oh = F.one_hot(atom_types, self.num_atom_types).float()
        positions = self._center(positions)
        t = torch.randint(0, self.num_timesteps, (1,), device=device)

        noisy_types, noisy_pos, true_noise = self.forward_diffusion(types_oh, positions, t.item())
        pred_noise, pred_atom_logits = self.denoise(noisy_types, noisy_pos, edge_index, t)

        coord_loss = F.mse_loss(pred_noise, true_noise)
        atom_loss = F.cross_entropy(pred_atom_logits, atom_types)
        total_loss = coord_loss + 0.1 * atom_loss

        return {"total_loss": total_loss, "coord_loss": coord_loss, "atom_loss": atom_loss}

    @torch.no_grad()
    def sample(self, num_atoms, edge_index, device=torch.device("cpu")):
        """
        Generate a 3D molecule via iterative denoising.

        Args:
            num_atoms: Number of atoms to generate.
            edge_index: [2, E] graph structure (e.g., fully connected).
            device: Compute device.

        Returns:
            (predicted_atom_types [N], generated_positions [N, 3])
        """
        self.eval()

        # Start from pure noise
        pos = self._center(torch.randn(num_atoms, 3, device=device))
        types = torch.ones(num_atoms, self.num_atom_types, device=device) / self.num_atom_types

        for t_idx in reversed(range(self.num_timesteps)):
            t = torch.tensor([t_idx], device=device)
            pred_noise, pred_logits = self.denoise(types, pos, edge_index, t)

            # DDPM reverse step for coordinates
            alpha = self.alphas[t_idx]
            alpha_bar = self.alpha_bar[t_idx]
            beta = self.betas[t_idx]

            coeff = beta / torch.sqrt(1 - alpha_bar)
            mean = (1 / torch.sqrt(alpha)) * (pos - coeff * pred_noise)

            if t_idx > 0:
                noise = torch.randn_like(pos) * torch.sqrt(beta)
                pos = mean + noise
            else:
                pos = mean

            pos = self._center(pos)

            # Update atom type estimates
            types = F.softmax(pred_logits, dim=-1)

        atom_types = types.argmax(dim=-1)
        return atom_types, pos


# ============================================================
# Utility: Build Fully Connected Edge Index
# ============================================================

def fully_connected_edges(num_atoms: int) -> torch.Tensor:
    """Build fully connected edge index (no self-loops)."""
    src = []
    dst = []
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                src.append(i)
                dst.append(j)
    return torch.tensor([src, dst], dtype=torch.long)


# ============================================================
# 3D Geometry Evaluation
# ============================================================

class Geometry3DEvaluator:
    """
    Evaluate quality of generated 3D molecular geometries.

    Checks bond lengths, bond angles, and inter-atomic distances
    against physical expectations.
    """

    # Typical bond lengths in Angstroms (mean, std)
    BOND_LENGTHS = {
        ("C", "C", 1): (1.54, 0.03),
        ("C", "C", 2): (1.34, 0.03),
        ("C", "C", 3): (1.20, 0.02),
        ("C", "N", 1): (1.47, 0.03),
        ("C", "O", 1): (1.43, 0.03),
        ("C", "O", 2): (1.23, 0.02),
    }

    MIN_DISTANCE = 0.8  # Angstroms — below this is a steric clash

    @staticmethod
    def atom_stability(
        positions: torch.Tensor,
        atom_types: List[str],
        bonds: List[Tuple[int, int, int]],
    ) -> float:
        """
        Fraction of atoms with all bond lengths within 2 sigma
        of their expected values.
        """
        num_atoms = len(atom_types)
        stable = [True] * num_atoms

        for i, j, order in bonds:
            dist = (positions[i] - positions[j]).norm().item()
            key = (atom_types[i], atom_types[j], order)
            key_rev = (atom_types[j], atom_types[i], order)

            expected = Geometry3DEvaluator.BOND_LENGTHS.get(
                key, Geometry3DEvaluator.BOND_LENGTHS.get(key_rev, None),
            )
            if expected is not None:
                mean, std = expected
                if abs(dist - mean) > 2 * std:
                    stable[i] = False
                    stable[j] = False

        return sum(stable) / num_atoms if num_atoms > 0 else 0.0

    @staticmethod
    def steric_clash_fraction(positions: torch.Tensor) -> float:
        """Fraction of atom pairs with distance below the minimum threshold."""
        n = positions.size(0)
        if n < 2:
            return 0.0
        dists = torch.cdist(positions.unsqueeze(0), positions.unsqueeze(0)).squeeze(0)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        pair_dists = dists[mask]
        clashes = (pair_dists < Geometry3DEvaluator.MIN_DISTANCE).sum().item()
        total_pairs = mask.sum().item()
        return clashes / total_pairs if total_pairs > 0 else 0.0


# ============================================================
# Demo
# ============================================================

if __name__ == "__main__":
    print("=== SchNet Demo ===")
    schnet = SchNet(num_atom_types=5, hidden_dim=64, num_interactions=2)
    atom_types = torch.tensor([0, 1, 2, 0, 1])  # 5 atoms
    positions = torch.randn(5, 3)
    edges = fully_connected_edges(5)
    h, pred = schnet(atom_types, positions, edges)
    print(f"SchNet output — features: {h.shape}, predictions: {pred.shape}")

    print("\n=== EGNN Demo ===")
    egnn = EGNN(num_atom_types=5, hidden_dim=64, num_layers=2)
    h_out, pos_out = egnn(atom_types, positions, edges)
    print(f"EGNN output — features: {h_out.shape}, positions: {pos_out.shape}")

    # Verify equivariance: rotate input and check output rotates
    Q = torch.tensor([
        [0.0, -1.0, 0.0],
        [1.0,  0.0, 0.0],
        [0.0,  0.0, 1.0],
    ])
    rotated_pos = positions @ Q.T
    h_rot, pos_rot = egnn(atom_types, rotated_pos, edges)
    expected_pos_rot = pos_out @ Q.T
    equiv_error = (pos_rot - expected_pos_rot).abs().max().item()
    print(f"Equivariance error: {equiv_error:.6f}")

    print("\n=== Equivariant Diffusion Model Demo ===")
    edm = EquivariantDiffusion3D(num_atom_types=5, hidden_dim=64, num_layers=2, num_timesteps=50)
    loss_dict = edm.training_loss(atom_types, positions, edges)
    print(f"Training losses: coord={loss_dict['coord_loss']:.4f}, "
          f"atom={loss_dict['atom_loss']:.4f}, "
          f"total={loss_dict['total_loss']:.4f}")

    # Sample (reduced timesteps for demo speed)
    edm.num_timesteps = 10
    edm.betas = edm.betas[:10]
    edm.alphas = edm.alphas[:10]
    edm.alpha_bar = edm.alpha_bar[:10]
    gen_types, gen_pos = edm.sample(5, edges)
    print(f"Generated types: {gen_types}")
    print(f"Generated positions shape: {gen_pos.shape}")

    print("\n=== 3D Geometry Evaluation ===")
    clash_frac = Geometry3DEvaluator.steric_clash_fraction(gen_pos)
    print(f"Steric clash fraction: {clash_frac:.4f}")
