"""Tsne deep dive."""
# ---
# title: "t-SNE Deep Dive"
# description: "Detailed exploration of t-SNE parameters and behaviour"
# ---
#
# t-SNE (van der Maaten & Hinton, 2008) is the most popular manifold
# learning method for visualization.  This script explores how perplexity,
# learning rate, and number of iterations affect the final embedding.
#
# Key insights for quant practitioners:
#   • t-SNE is *non-parametric* — you cannot project new points without
#     re-running the algorithm.  For production, use parametric t-SNE or UMAP.
#   • Cluster *sizes* and *distances* in t-SNE plots can be misleading.
#   • Always try several perplexity values (5–50).

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# ========================================================================
# Main
# ========================================================================

# ─── 1.  Effect of perplexity ─────────────────────────────────────────────
print("Generating blobs to show perplexity effect...")
X_blobs, y_blobs = make_blobs(
    n_samples=600, centers=4, cluster_std=[1.0, 2.5, 0.5, 1.5], random_state=42
)
X_blobs = StandardScaler().fit_transform(X_blobs)

perplexities = [5, 15, 30, 50, 100]
fig, axes = plt.subplots(1, len(perplexities), figsize=(20, 4))
for ax, perp in zip(axes, perplexities):
    X_2d = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000).fit_transform(X_blobs)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_blobs, cmap="tab10", s=10)
    ax.set_title(f"perplexity={perp}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("t-SNE: Effect of Perplexity", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("tsne_perplexity_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── 2.  Effect of n_iter (convergence) ───────────────────────────────────
print("Showing convergence with increasing iterations...")
iterations = [250, 500, 1000, 2000, 5000]
fig, axes = plt.subplots(1, len(iterations), figsize=(20, 4))
for ax, n_iter in zip(axes, iterations):
    X_2d = TSNE(n_components=2, perplexity=30, n_iter=n_iter, random_state=42).fit_transform(X_blobs)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_blobs, cmap="tab10", s=10)
    ax.set_title(f"n_iter={n_iter}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("t-SNE: Convergence with Iterations", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("tsne_iterations_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── 3.  PCA speed-up trick on MNIST ──────────────────────────────────────
print("\nMNIST: comparing raw t-SNE vs PCA+t-SNE speed...")
import time

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist.data[:3000].astype(np.float32)
y = mnist.target[:3000].astype(int)

# Direct t-SNE
t0 = time.time()
X_direct = TSNE(n_components=2, random_state=42).fit_transform(X)
t_direct = time.time() - t0

# PCA (95% variance) + t-SNE
t0 = time.time()
X_pca = PCA(n_components=0.95, random_state=42).fit_transform(X)
X_pca_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_pca)
t_pca_tsne = time.time() - t0

print(f"  Direct t-SNE:    {t_direct:.1f}s")
print(f"  PCA + t-SNE:     {t_pca_tsne:.1f}s  (speed-up: {t_direct / t_pca_tsne:.1f}x)")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, X_2d, title in zip(
    axes,
    [X_direct, X_pca_tsne],
    [f"Direct t-SNE ({t_direct:.1f}s)", f"PCA + t-SNE ({t_pca_tsne:.1f}s)"],
):
    for digit in range(10):
        mask = y == digit
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=5, label=str(digit), alpha=0.6)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
axes[0].legend(markerscale=3, fontsize=8)
plt.suptitle("PCA Pre-processing Accelerates t-SNE", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("tsne_pca_speedup.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone.")


if __name__ == "__main__":
    pass