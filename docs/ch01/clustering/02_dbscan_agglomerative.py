"""Dbscan agglomerative."""
# ---
# title: "DBSCAN and Agglomerative Clustering"
# description: "Density-based and hierarchical clustering with sklearn"
# ---
#
# Beyond K-Means, two important clustering paradigms:
#   • DBSCAN  — finds arbitrarily-shaped clusters; handles noise
#   • Agglomerative — bottom-up hierarchical clustering with dendrograms
#
# These methods shine where K-Means fails: non-convex clusters and
# unknown number of clusters.
#
# Adapted from: O'Reilly Hands-On ML, Chapter 9 (Unsupervised Learning)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_moons, make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ========================================================================
# Main
# ========================================================================

# ─── 1.  DBSCAN on non-convex data ────────────────────────────────────────
np.random.seed(42)
X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)
X_moons = StandardScaler().fit_transform(X_moons)

# K-Means fails on moons
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, random_state=42).fit(X_moons)

# DBSCAN succeeds
db = DBSCAN(eps=0.2, min_samples=5).fit(X_moons)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X_moons[:, 0], X_moons[:, 1], c=km.labels_, cmap="tab10", s=10)
ax1.set_title("K-Means (fails on moons)")
ax1.grid(True, alpha=0.3)

colours = db.labels_.copy()
noise_mask = db.labels_ == -1
ax2.scatter(X_moons[~noise_mask, 0], X_moons[~noise_mask, 1],
            c=colours[~noise_mask], cmap="tab10", s=10)
ax2.scatter(X_moons[noise_mask, 0], X_moons[noise_mask, 1],
            c="gray", s=10, marker="x", alpha=0.5, label="noise")
ax2.set_title(f"DBSCAN (eps=0.2, min_samples=5) — {len(set(db.labels_)) - 1} clusters")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dbscan_vs_kmeans_moons.png", dpi=150)
plt.show()

# ─── 2.  Choosing eps with k-distance graph ───────────────────────────────
print("Computing k-distance graph for eps selection...")
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_moons)
distances, _ = nn.kneighbors(X_moons)
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.xlabel("Points (sorted by 5-NN distance)")
plt.ylabel("5-NN distance")
plt.title("k-Distance Graph (knee ≈ optimal eps)")
plt.axhline(y=0.2, color="red", linestyle="--", label="eps=0.2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dbscan_k_distance.png", dpi=150)
plt.show()

# ─── 3.  DBSCAN parameter sensitivity ─────────────────────────────────────
eps_values = [0.05, 0.1, 0.2, 0.3, 0.5]
fig, axes = plt.subplots(1, len(eps_values), figsize=(20, 4))
for ax, eps in zip(axes, eps_values):
    db = DBSCAN(eps=eps, min_samples=5).fit(X_moons)
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    n_noise = (db.labels_ == -1).sum()
    ax.scatter(X_moons[:, 0], X_moons[:, 1], c=db.labels_, cmap="tab10", s=8)
    ax.set_title(f"eps={eps}\nclusters={n_clusters}, noise={n_noise}")
    ax.set_xticks([])
    ax.set_yticks([])
plt.suptitle("DBSCAN: Effect of eps", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("dbscan_eps_sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── 4.  Agglomerative Clustering ─────────────────────────────────────────
X_varied, y_varied = make_blobs(
    n_samples=600, centers=4,
    cluster_std=[1.0, 2.5, 0.5, 1.5], random_state=42
)

linkages = ["ward", "complete", "average", "single"]
fig, axes = plt.subplots(1, len(linkages), figsize=(18, 4))
for ax, linkage in zip(axes, linkages):
    agg = AgglomerativeClustering(n_clusters=4, linkage=linkage)
    labels = agg.fit_predict(X_varied)
    ax.scatter(X_varied[:, 0], X_varied[:, 1], c=labels, cmap="tab10", s=10)
    ax.set_title(f"linkage='{linkage}'")
    ax.grid(True, alpha=0.3)
plt.suptitle("Agglomerative Clustering: Linkage Comparison", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("agglomerative_linkages.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── 5.  Dendrogram (requires scipy) ──────────────────────────────────────
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

# Small subset for readable dendrogram
np.random.seed(42)
idx = np.random.choice(len(X_varied), 30, replace=False)
X_small = X_varied[idx]

Z = scipy_linkage(X_small, method="ward")
plt.figure(figsize=(12, 5))
dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
plt.title("Dendrogram (Ward linkage, 30 samples)")
plt.xlabel("Sample index")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig("agglomerative_dendrogram.png", dpi=150)
plt.show()

print("Done.")


if __name__ == "__main__":
    pass