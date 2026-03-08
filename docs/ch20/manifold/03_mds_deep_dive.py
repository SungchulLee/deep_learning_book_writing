"""Mds deep dive."""
# ---
# title: "Multidimensional Scaling (MDS) Deep Dive"
# description: "Classical MDS from scratch, sklearn metric/non-metric MDS, Shepard diagrams,
#               PCA equivalence proof, timing analysis, and finance correlation-distance maps"
# ---
#
# MDS embeds high-dimensional data in low dimensions by preserving pairwise
# distances.  This script builds from scratch to practical usage:
#
#   Part 1 – Classical MDS via double centering (NumPy)
#   Part 2 – Verify PCA ≡ Classical MDS on Euclidean data
#   Part 3 – Metric MDS via stress minimisation (PyTorch)
#   Part 4 – sklearn MDS on Swiss Roll (metric vs non-metric)
#   Part 5 – MDS on MNIST with timing + PCA speed-up
#   Part 6 – Shepard diagram for quality assessment
#   Part 7 – Finance application: correlation-distance asset map
#
# Notebook source: O'Reilly Hands-On ML Ch 8 (dimensionality reduction)
# Existing theory:  ch20/manifold/mds.md  ← see for full derivations

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

# =====================================================================
# Part 1 – Classical MDS from scratch
# =====================================================================
def classical_mds(D, n_components=2):
    """Classical (metric) MDS via double centering + eigendecomposition.

    Args
    ----
    D : ndarray [n, n]  – symmetric distance matrix (zero diagonal)
    n_components : int  – target dimensionality

    Returns
    -------
    Y            : ndarray [n, n_components] – embedding coordinates
    eigenvalues  : ndarray [n_components]    – top eigenvalues of Gram matrix
    """
    n = D.shape[0]
    D_sq = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n          # centering matrix
    B = -0.5 * H @ D_sq @ H                       # Gram matrix

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]            # descending
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    k = n_components
    Lambda_k = np.diag(np.sqrt(np.maximum(eigenvalues[:k], 0)))
    Q_k = eigenvectors[:, :k]
    return Q_k @ Lambda_k, eigenvalues[:k]


# Quick demo on random 3-D points
np.random.seed(42)
X_demo = np.random.randn(200, 3)
D_demo = squareform(pdist(X_demo))
Y_demo, eig_demo = classical_mds(D_demo, n_components=2)

plt.figure(figsize=(6, 5))
plt.scatter(Y_demo[:, 0], Y_demo[:, 1], s=15, alpha=0.7)
plt.title("Classical MDS (from scratch) on 200 random 3-D points")
plt.xlabel("$z_1$"); plt.ylabel("$z_2$")
plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("mds_classical_scratch.png", dpi=150)
plt.show()


# =====================================================================
# Part 2 – PCA ≡ Classical MDS proof (numerical)
# =====================================================================
from sklearn.decomposition import PCA

X_centered = X_demo - X_demo.mean(axis=0)
Y_pca = PCA(n_components=2).fit_transform(X_centered)
D_cent = squareform(pdist(X_centered))
Y_mds_c, _ = classical_mds(D_cent, n_components=2)

# Align signs (eigenvectors are unique up to sign)
for j in range(2):
    if np.corrcoef(Y_pca[:, j], Y_mds_c[:, j])[0, 1] < 0:
        Y_mds_c[:, j] *= -1

max_diff = np.abs(Y_pca - Y_mds_c).max()
print(f"PCA vs Classical MDS  max |difference| = {max_diff:.2e}")
print("  → Numerically identical (as expected for Euclidean distances)\n")


# =====================================================================
# Part 3 – Metric MDS via PyTorch stress minimisation
# =====================================================================
try:
    import torch

    def metric_mds_torch(D, n_components=2, n_iter=300, lr=0.01, seed=42):
        """Metric MDS by gradient-descending raw stress."""
        torch.manual_seed(seed)
        n = D.shape[0]
        D_t = torch.tensor(D, dtype=torch.float32)
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)

        # warm-start from classical MDS
        Y_init, _ = classical_mds(D, n_components)
        Y = torch.tensor(Y_init, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([Y], lr=lr)
        stress_hist = []

        for _ in range(n_iter):
            opt.zero_grad()
            diff = Y.unsqueeze(0) - Y.unsqueeze(1)
            d_embed = torch.sqrt((diff ** 2).sum(-1) + 1e-12)
            stress = ((D_t[mask] - d_embed[mask]) ** 2).sum()
            stress.backward()
            opt.step()
            stress_hist.append(stress.item())
        return Y.detach().numpy(), stress_hist

    Y_torch, stress_hist = metric_mds_torch(D_demo, n_components=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(stress_hist)
    ax1.set_xlabel("Iteration"); ax1.set_ylabel("Stress")
    ax1.set_title("PyTorch Metric MDS — Stress Convergence")
    ax1.grid(True, alpha=0.3)

    ax2.scatter(Y_torch[:, 0], Y_torch[:, 1], s=15, alpha=0.7)
    ax2.set_title("Metric MDS Embedding (PyTorch)")
    ax2.set_xlabel("$z_1$"); ax2.set_ylabel("$z_2$")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mds_pytorch_stress.png", dpi=150)
    plt.show()
    print(f"Final stress = {stress_hist[-1]:.2f}\n")

except ImportError:
    print("PyTorch not available — skipping Part 3\n")


# =====================================================================
# Part 4 – Swiss Roll: sklearn metric vs non-metric MDS
# =====================================================================
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import MDS

X_swiss, t_swiss = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

mds_metric = MDS(n_components=2, metric=True, random_state=42,
                 normalized_stress="auto")
mds_nonmetric = MDS(n_components=2, metric=False, random_state=42,
                    normalized_stress="auto")

t0 = time.time()
Y_met = mds_metric.fit_transform(X_swiss)
t_met = time.time() - t0

t0 = time.time()
Y_nmet = mds_nonmetric.fit_transform(X_swiss)
t_nmet = time.time() - t0

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(Y_met[:, 0], Y_met[:, 1], c=t_swiss, cmap="hot", s=10)
ax1.set_title(f"Metric MDS ({t_met:.1f}s)\nstress={mds_metric.stress_:.0f}")
ax1.grid(True, alpha=0.3)

ax2.scatter(Y_nmet[:, 0], Y_nmet[:, 1], c=t_swiss, cmap="hot", s=10)
ax2.set_title(f"Non-metric MDS ({t_nmet:.1f}s)\nstress={mds_nonmetric.stress_:.0f}")
ax2.grid(True, alpha=0.3)

plt.suptitle("MDS on Swiss Roll: Metric vs Non-Metric", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("mds_swiss_metric_vs_nonmetric.png", dpi=150, bbox_inches="tight")
plt.show()


# =====================================================================
# Part 5 – MDS on MNIST with timing and PCA speed-up
# =====================================================================
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

print("Loading MNIST...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_mnist = mnist.data.astype(np.float32)
y_mnist = mnist.target.astype(int)

# Subsample (MDS is O(n^2 T))
m = 2000
np.random.seed(42)
idx = np.random.permutation(len(X_mnist))[:m]
X_sub, y_sub = X_mnist[idx], y_mnist[idx]


def plot_digit_scatter(X_2d, y, title="", ax=None):
    """Scatter plot with digits colour-coded."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    for digit in range(10):
        mask = y == digit
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=6, label=str(digit), alpha=0.6)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    return ax


# --- Raw MDS (slow) ---
print(f"Running MDS on {m} MNIST images (no PCA pre-processing)...")
t0 = time.time()
Y_raw = MDS(n_components=2, random_state=42, normalized_stress="auto").fit_transform(X_sub)
t_raw = time.time() - t0
print(f"  Raw MDS: {t_raw:.1f}s")

# --- PCA + MDS (fast) ---
print(f"Running PCA(95% var) + MDS on {m} MNIST images...")
t0 = time.time()
pipe = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("mds", MDS(n_components=2, random_state=42, normalized_stress="auto")),
])
Y_pca_mds = pipe.fit_transform(X_sub)
t_pca = time.time() - t0
n_pca_dims = pipe.named_steps["pca"].n_components_
print(f"  PCA({n_pca_dims}d) + MDS: {t_pca:.1f}s  (speed-up {t_raw/t_pca:.1f}x)\n")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
plot_digit_scatter(Y_raw, y_sub, f"Raw MDS ({t_raw:.1f}s)", ax=ax1)
plot_digit_scatter(Y_pca_mds, y_sub, f"PCA + MDS ({t_pca:.1f}s, {t_raw/t_pca:.1f}x faster)", ax=ax2)
ax1.legend(markerscale=3, fontsize=7)
plt.suptitle(f"MDS on MNIST ({m} samples)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("mds_mnist_timing.png", dpi=150, bbox_inches="tight")
plt.show()


# =====================================================================
# Part 6 – Shepard diagram (embedding quality)
# =====================================================================
def shepard_diagram(D_original, Y_embedded, title="Shepard Diagram"):
    """Plot original vs embedded distances to assess MDS quality."""
    D_embed = squareform(pdist(Y_embedded))
    mask_idx = np.triu_indices_from(D_original, k=1)

    corr = np.corrcoef(D_original[mask_idx], D_embed[mask_idx])[0, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(D_original[mask_idx], D_embed[mask_idx],
               alpha=0.1, s=3, color="C0")
    lims = [0, max(D_original[mask_idx].max(), D_embed[mask_idx].max())]
    ax.plot(lims, lims, "r--", alpha=0.6, label="perfect")
    ax.set_xlabel("Original distance")
    ax.set_ylabel("Embedded distance")
    ax.set_title(f"{title}\n(Pearson r = {corr:.3f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


D_sub = squareform(pdist(X_sub))
fig = shepard_diagram(D_sub, Y_pca_mds, "Shepard Diagram — PCA+MDS on MNIST")
fig.savefig("mds_shepard.png", dpi=150)
plt.show()


# =====================================================================
# Part 7 – Finance: correlation-distance asset map
# =====================================================================
print("Finance application: correlation-distance MDS map")

np.random.seed(42)
n_periods = 500
market = np.random.randn(n_periods)
tech   = 0.8 * market + 0.2 * np.random.randn(n_periods)
semis  = 0.7 * market + 0.5 * tech + 0.2 * np.random.randn(n_periods)
banks  = 0.6 * market + 0.3 * np.random.randn(n_periods)
energy = 0.3 * market + 0.5 * np.random.randn(n_periods)
gold   = -0.1 * market + 0.6 * np.random.randn(n_periods)
bonds  = -0.3 * market + 0.4 * np.random.randn(n_periods)
crypto = 0.2 * market + 0.9 * np.random.randn(n_periods)

returns = np.column_stack([market, tech, semis, banks, energy, gold, bonds, crypto])
names = ["Market", "Tech", "Semis", "Banks", "Energy", "Gold", "Bonds", "Crypto"]

# Correlation distance:  d_ij = sqrt(2(1 - rho_ij))
corr = np.corrcoef(returns.T)
D_corr = np.sqrt(2 * (1 - corr))

Y_assets, eig_assets = classical_mds(D_corr, n_components=2)

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(Y_assets[:, 0], Y_assets[:, 1], s=120, alpha=0.8, zorder=5)
for i, name in enumerate(names):
    ax.annotate(name, (Y_assets[i, 0], Y_assets[i, 1]),
                fontsize=11, ha="center", va="bottom",
                xytext=(0, 10), textcoords="offset points",
                fontweight="bold")
ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
ax.axvline(0, color="gray", linestyle="--", alpha=0.3)
ax.set_xlabel("MDS Dimension 1")
ax.set_ylabel("MDS Dimension 2")
ax.set_title("Asset Structure via Correlation-Distance MDS")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("mds_asset_map.png", dpi=150)
plt.show()

# Print correlation distance matrix
print("\nCorrelation-distance matrix:")
print(f"{'':>8}", end="")
for n in names:
    print(f"{n:>8}", end="")
print()
for i, ni in enumerate(names):
    print(f"{ni:>8}", end="")
    for j in range(len(names)):
        print(f"{D_corr[i, j]:8.2f}", end="")
    print()

print("\nDone — 7 PNG files saved.")


if __name__ == "__main__":
    pass
