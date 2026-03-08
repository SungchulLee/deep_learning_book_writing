"""Kmeans."""
# ---
# title: "K-Means Clustering"
# description: "K-Means from basics to advanced usage with sklearn"
# ---
#
# K-Means is the most widely used clustering algorithm.  This script covers:
#   1. Basic K-Means on synthetic data
#   2. The Elbow method and Silhouette analysis for choosing k
#   3. K-Means on images (colour quantization)
#   4. Mini-Batch K-Means for large datasets
#
# Adapted from: O'Reilly Hands-On ML, Chapter 9 (Unsupervised Learning)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs, load_digits
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

# ========================================================================
# Main
# ========================================================================

# ─── 1.  Basic K-Means ────────────────────────────────────────────────────
np.random.seed(42)
X_blobs, y_blobs = make_blobs(
    n_samples=500, centers=5, cluster_std=[1.0, 0.7, 1.2, 0.9, 1.5], random_state=42
)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
y_pred = kmeans.fit_predict(X_blobs)

plt.figure(figsize=(8, 5))
plt.scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_pred, cmap="tab10", s=15, alpha=0.7)
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X", s=200, c="red", edgecolors="black", linewidth=1.5,
    label="centroids",
)
plt.title(f"K-Means (k=5)  inertia={kmeans.inertia_:.0f}")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("kmeans_basic.png", dpi=150)
plt.show()

# ─── 2.  Elbow Method ─────────────────────────────────────────────────────
k_range = range(2, 12)
inertias = []
sil_scores = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_blobs)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_blobs, km.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(list(k_range), inertias, "bo-")
ax1.set_xlabel("k")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method")
ax1.grid(True, alpha=0.3)

ax2.plot(list(k_range), sil_scores, "ro-")
ax2.set_xlabel("k")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Analysis")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("kmeans_elbow_silhouette.png", dpi=150)
plt.show()

print(f"Best k by silhouette: {list(k_range)[np.argmax(sil_scores)]}")

# ─── 3.  Silhouette Diagram ───────────────────────────────────────────────
def plot_silhouette(X, labels, k, ax):
    """Draw a silhouette diagram."""
    sil_vals = silhouette_samples(X, labels)
    y_lower = 10
    for i in range(k):
        cluster_sil = np.sort(sil_vals[labels == i])
        y_upper = y_lower + len(cluster_sil)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * len(cluster_sil), str(i))
        y_lower = y_upper + 10
    ax.axvline(x=silhouette_score(X, labels), color="red", linestyle="--")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_title(f"k = {k}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, k in zip(axes, [3, 5, 7]):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_blobs)
    plot_silhouette(X_blobs, km.labels_, k, ax)
plt.suptitle("Silhouette Diagrams", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("kmeans_silhouette_diagrams.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── 4.  Mini-Batch K-Means ───────────────────────────────────────────────
import time

X_large, _ = make_blobs(n_samples=50000, centers=10, random_state=42)

t0 = time.time()
km_full = KMeans(n_clusters=10, random_state=42, n_init=10).fit(X_large)
t_full = time.time() - t0

t0 = time.time()
km_mini = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=1024).fit(X_large)
t_mini = time.time() - t0

print(f"\nFull K-Means:       {t_full:.2f}s  inertia={km_full.inertia_:.0f}")
print(f"Mini-Batch K-Means: {t_mini:.2f}s  inertia={km_mini.inertia_:.0f}")
print(f"Speed-up:           {t_full / t_mini:.1f}x")
print("Done.")


if __name__ == "__main__":
    pass