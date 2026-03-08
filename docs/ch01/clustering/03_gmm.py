"""Gmm."""
# ---
# title: "Gaussian Mixture Models"
# description: "GMM clustering and density estimation with sklearn"
# ---
#
# Gaussian Mixture Models (GMMs) offer a probabilistic alternative to K-Means:
#   • Soft assignments (each point has a probability of belonging to each cluster)
#   • Can model ellipsoidal clusters (not just spherical)
#   • Natural model selection via BIC/AIC
#
# GMMs are trained with Expectation-Maximization (EM), which alternates
# between computing responsibilities (E-step) and updating parameters (M-step).
#
# Adapted from: O'Reilly Hands-On ML, Chapter 9 (Unsupervised Learning)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse

# ========================================================================
# Main
# ========================================================================

np.random.seed(42)

# ─── 1.  Basic GMM vs K-Means ─────────────────────────────────────────────
# Elongated clusters where K-Means struggles
X1 = np.random.randn(300, 2) @ np.array([[2.0, 0.8], [0.8, 0.5]]) + [2, 0]
X2 = np.random.randn(200, 2) @ np.array([[1.0, -0.5], [-0.5, 0.8]]) + [-2, 3]
X3 = np.random.randn(250, 2) * 0.5 + [0, -2]
X = np.vstack([X1, X2, X3])

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
gmm = GaussianMixture(n_components=3, random_state=42).fit(X)


def draw_ellipse(ax, mean, cov, color, n_std=2.0):
    """Draw a covariance ellipse at n_std standard deviations."""
    vals, vecs = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=mean, width=width, height=height, angle=angle,
                  facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
    ax.add_patch(ell)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="tab10", s=8, alpha=0.6)
ax1.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
            marker="X", s=150, c="red", edgecolors="black")
ax1.set_title("K-Means")
ax1.grid(True, alpha=0.3)

y_gmm = gmm.predict(X)
ax2.scatter(X[:, 0], X[:, 1], c=y_gmm, cmap="tab10", s=8, alpha=0.6)
colours = ["C0", "C1", "C2"]
for i in range(3):
    draw_ellipse(ax2, gmm.means_[i], gmm.covariances_[i], colours[i])
ax2.set_title("GMM (with covariance ellipses)")
ax2.grid(True, alpha=0.3)

plt.suptitle("K-Means vs GMM on Elongated Clusters", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("gmm_vs_kmeans.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── 2.  Soft assignments (responsibilities) ──────────────────────────────
probs = gmm.predict_proba(X)
print("Sample responsibilities (first 5 points):")
print(f"  Shape: {probs.shape}")
for i in range(5):
    print(f"  Point {i}: {probs[i].round(3)}")

# ─── 3.  Model selection: BIC and AIC ─────────────────────────────────────
n_components_range = range(1, 10)
bics = []
aics = []

for n in n_components_range:
    gm = GaussianMixture(n_components=n, random_state=42).fit(X)
    bics.append(gm.bic(X))
    aics.append(gm.aic(X))

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(n_components_range), bics, "bo-", label="BIC")
ax.plot(list(n_components_range), aics, "ro-", label="AIC")
ax.set_xlabel("Number of components")
ax.set_ylabel("Information criterion")
ax.set_title("GMM Model Selection")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gmm_bic_aic.png", dpi=150)
plt.show()

best_k = list(n_components_range)[np.argmin(bics)]
print(f"\nBest k by BIC: {best_k}")

# ─── 4.  Covariance types ─────────────────────────────────────────────────
cov_types = ["full", "tied", "diag", "spherical"]
fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, cov_type in zip(axes, cov_types):
    gm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42).fit(X)
    y_pred = gm.predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap="tab10", s=8, alpha=0.6)
    ax.set_title(f"cov_type='{cov_type}'\nBIC={gm.bic(X):.0f}")
    ax.grid(True, alpha=0.3)
plt.suptitle("GMM: Covariance Type Comparison", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("gmm_covariance_types.png", dpi=150, bbox_inches="tight")
plt.show()

# ─── 5.  Bayesian GMM (auto-selects n_components) ─────────────────────────
bgmm = BayesianGaussianMixture(
    n_components=10,       # upper bound
    weight_concentration_prior=0.01,
    random_state=42,
).fit(X)

# Many components will have near-zero weight
weights = bgmm.weights_
active = weights > 0.01
print(f"\nBayesian GMM: {active.sum()} active components out of 10")
print(f"  Weights: {weights.round(3)}")

y_bgmm = bgmm.predict(X)
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_bgmm, cmap="tab10", s=8, alpha=0.6)
plt.title(f"Bayesian GMM ({active.sum()} active components)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("bayesian_gmm.png", dpi=150)
plt.show()

# ─── 6.  GMM for anomaly detection ────────────────────────────────────────
# Use log-likelihood as anomaly score
scores = gmm.score_samples(X)
threshold = np.percentile(scores, 2)  # bottom 2% as anomalies
anomalies = X[scores < threshold]

plt.figure(figsize=(8, 5))
plt.scatter(X[:, 0], X[:, 1], c="C0", s=8, alpha=0.3, label="normal")
plt.scatter(anomalies[:, 0], anomalies[:, 1], c="red", s=30, marker="x",
            linewidths=1.5, label=f"anomalies ({len(anomalies)})")
plt.title("GMM Anomaly Detection (2% threshold)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gmm_anomaly_detection.png", dpi=150)
plt.show()

print("Done.")


if __name__ == "__main__":
    pass