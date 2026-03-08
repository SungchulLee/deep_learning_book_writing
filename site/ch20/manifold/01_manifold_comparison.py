"""Manifold comparison."""
# ---
# title: "Manifold Learning: Method Comparison"
# description: "Compare t-SNE, MDS, Isomap, and LLE on Swiss Roll and MNIST"
# ---
#
# This script demonstrates the four main manifold learning algorithms
# available in scikit-learn, comparing their behavior on both synthetic
# (Swiss Roll) and real (MNIST) data. We also show how PCA pre-processing
# can speed up these methods dramatically.
#
# Reference: O'Reilly Hands-On ML, Chapter 8 – Dimensionality Reduction

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ========================================================================
# Main
# ========================================================================

# ─── 1.  Synthetic data: Swiss Roll ──────────────────────────────────────
X_swiss, t_swiss = make_swiss_roll(n_samples=1500, noise=0.2, random_state=42)

methods = {
    "MDS": MDS(n_components=2, random_state=42, normalized_stress="auto"),
    "Isomap": Isomap(n_components=2, n_neighbors=10),
    "LLE": LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42),
    "t-SNE": TSNE(n_components=2, random_state=42),
}

fig, axes = plt.subplots(1, len(methods), figsize=(16, 4))
for ax, (name, model) in zip(axes, methods.items()):
    t0 = time.time()
    X_2d = model.fit_transform(X_swiss)
    elapsed = time.time() - t0
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=t_swiss, cmap="hot", s=5)
    ax.set_title(f"{name} ({elapsed:.1f}s)")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.grid(True, alpha=0.3)
fig.suptitle("Manifold Learning on Swiss Roll", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("manifold_swiss_roll_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── 2.  Real data: MNIST ─────────────────────────────────────────────────
print("\nLoading MNIST (subset of 2 000 samples for speed)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_mnist, y_mnist = mnist.data[:2000].astype(np.float32), mnist.target[:2000].astype(int)
X_mnist = StandardScaler().fit_transform(X_mnist)

# Pre-reduce with PCA to keep 95% variance → massive speed-up
pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_mnist)
print(f"PCA reduced {X_mnist.shape[1]} → {X_pca.shape[1]} dimensions (95% var).\n")


def plot_digits(X_2d, labels, title=""):
    """Scatter plot coloured by digit label."""
    plt.figure(figsize=(8, 6))
    for digit in range(10):
        mask = labels == digit
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], s=8, label=str(digit), alpha=0.6)
    plt.legend(markerscale=3, fontsize=8)
    plt.title(title, fontsize=13)
    plt.xlabel("$z_1$")
    plt.ylabel("$z_2$")
    plt.grid(True, alpha=0.3)


results = {}
for name, model in methods.items():
    pipe = Pipeline([("pca", PCA(n_components=0.95, random_state=42)), ("manifold", model)])
    t0 = time.time()
    X_2d = pipe.fit_transform(X_mnist)
    elapsed = time.time() - t0
    results[name] = (X_2d, elapsed)
    print(f"PCA + {name}: {elapsed:.1f}s")

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, (X_2d, elapsed)) in zip(axes, results.items()):
    for digit in range(10):
        mask = y_mnist == digit
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=6, label=str(digit), alpha=0.6)
    ax.set_title(f"PCA + {name} ({elapsed:.1f}s)")
    ax.grid(True, alpha=0.3)
axes[0].legend(markerscale=3, fontsize=7)
plt.suptitle("Manifold Learning on MNIST (2 000 samples)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("manifold_mnist_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── 3.  t-SNE deep-dive: digit subset ────────────────────────────────────
print("\nt-SNE close-up on digits {2, 3, 5}...")
mask_subset = np.isin(y_mnist, [2, 3, 5])
X_sub, y_sub = X_mnist[mask_subset], y_mnist[mask_subset]
X_sub_2d = TSNE(n_components=2, random_state=42).fit_transform(X_sub)
plot_digits(X_sub_2d, y_sub, title="t-SNE on digits {2, 3, 5}")
plt.savefig("tsne_digit_subset.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone. Three PNG files saved.")


if __name__ == "__main__":
    pass