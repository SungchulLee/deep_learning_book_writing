# 가우시안 혼합 모형

가우시안 혼합 모형(GMM)은 K-평균에 대한 확률적 대안으로, 부드러운 군집 배정을 제공한다. 각 데이터 점이 하나의 군집에 딱 배정되는 대신 각 군집에 속할 확률을 가진다. GMM은 성분별 공분산 행렬을 학습하여 모양과 방향이 서로 다른 타원형 군집을 모형화할 수 있고, BIC나 AIC 같은 정보 기준을 통해 원칙 있는 모델 선택을 지원한다. 학습에는 기댓값-최대화(EM) 알고리즘을 사용하며, 이는 책임도를 계산하는 단계(E 단계)와 매개변수를 갱신하는 단계(M 단계)를 번갈아 수행한다.

## 1. 코드

```python
"""Gmm."""
# ---
# title: "Gaussian Mixture Models"
# description: "GMM clustering and density estimation with sklearn"
# ---
#
# 가우시안 혼합 모형(GMM)은 K-평균에 대한 확률적 대안이다.
#   * 부드러운 배정(각 점이 각 군집에 속할 확률을 가진다)
#   * 타원형 군집을 모형화할 수 있다(구형에 국한되지 않음)
#   * BIC/AIC를 통한 자연스러운 모델 선택
#
# GMM은 기댓값-최대화(EM)로 학습하며, 책임도를 계산하는 단계(E 단계)와
# 매개변수를 갱신하는 단계(M 단계)를 번갈아 수행한다.
#
# 출처 각색: O'Reilly Hands-On ML, 9장(비지도 학습)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.datasets import make_blobs
from matplotlib.patches import Ellipse

# ========================================================================
# 메인
# ========================================================================

np.random.seed(42)

# ─── 1.  기본 GMM 대 K-평균 ──────────────────────────────────────────────
# K-평균이 어려움을 겪는 길게 늘어진 군집
X1 = np.random.randn(300, 2) @ np.array([[2.0, 0.8], [0.8, 0.5]]) + [2, 0]
X2 = np.random.randn(200, 2) @ np.array([[1.0, -0.5], [-0.5, 0.8]]) + [-2, 3]
X3 = np.random.randn(250, 2) * 0.5 + [0, -2]
X = np.vstack([X1, X2, X3])

from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
gmm = GaussianMixture(n_components=3, random_state=42).fit(X)


def draw_ellipse(ax, mean, cov, color, n_std=2.0):
    """표준편차 n_std 배 위치에 공분산 타원을 그린다."""
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

# ─── 2.  부드러운 배정(책임도) ───────────────────────────────────────────
probs = gmm.predict_proba(X)
print("Sample responsibilities (first 5 points):")
print(f"  Shape: {probs.shape}")
for i in range(5):
    print(f"  Point {i}: {probs[i].round(3)}")

# ─── 3.  모델 선택: BIC와 AIC ────────────────────────────────────────────
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

# ─── 4.  공분산 유형 ─────────────────────────────────────────────────────
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

# ─── 5.  베이즈 GMM(n_components 자동 선택) ─────────────────────────────
bgmm = BayesianGaussianMixture(
    n_components=10,       # 상한
    weight_concentration_prior=0.01,
    random_state=42,
).fit(X)

# 많은 성분의 가중치가 0에 가까워진다
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

# ─── 6.  이상 탐지를 위한 GMM ────────────────────────────────────────────
# 로그가능도를 이상 점수로 사용한다
scores = gmm.score_samples(X)
threshold = np.percentile(scores, 2)  # 하위 2%를 이상치로 본다
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
```

## 2. 논의

GMM은 데이터 분포를 $K$개 가우시안 성분의 가중합 $p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$으로 모형화한다. 여기서 $\pi_k$는 혼합 가중치, $\boldsymbol{\mu}_k$는 평균, $\boldsymbol{\Sigma}_k$는 공분산 행렬이다. EM 알고리즘은 각 데이터 점에 대한 각 성분의 **책임도(responsibility)** 를 계산하는 단계(E 단계)와 기대 로그가능도를 최대화하도록 매개변수를 갱신하는 단계(M 단계)를 번갈아 수행하여 이 매개변수들을 적합시킨다. 딱 떨어지는 배정을 하는 K-평균과 달리, GMM의 책임도는 각 점에 대해 성분들에 대한 확률 분포를 주므로 군집 경계가 겹치는 미묘한 상황도 표현할 수 있다.

**공분산 유형** 은 각 성분의 모양이 얼마나 유연한지를 조절한다. `full` 공분산에서는 각 성분이 임의의 타원 모양을 가질 수 있어 표현력이 가장 크지만 매개변수도 가장 많이 필요하다. `tied`는 모든 성분이 같은 공분산 행렬을 공유하도록 강제하고, `diag`는 축에 평행한 타원으로 제한하며, `spherical`은 K-평균과 비슷한 등방적 군집으로 축소된다. BIC와 AIC는 적합의 품질과 모델 복잡도의 균형을 잡는 모델 선택 기준을 제공한다. BIC는 매개변수 추가에 더 강한 벌점을 매겨 단순한 모델을 선호하고, AIC는 더 관대하다.

**베이즈 GMM** 변형은 혼합 가중치에 집중 매개변수가 작은 디리클레 사전분포를 두어, 불필요한 성분의 가중치를 0 쪽으로 몰아 자동으로 억제한다. 이는 모델 선택 문제에 우아한 해법을 제공한다. $K$를 넉넉한 상한으로 두고 베이즈 사전분포가 실효 성분 개수를 결정하게 하면 된다. 또한 GMM은 로그가능도 점수가 낮은 데이터 점을 표시함으로써 이상 탐지를 자연스럽게 지원한다. 어떤 성분에도 잘 맞지 않는 점은 이상치일 가능성이 높기 때문이다.

## 연습문제

**연습문제 1.**
코드의 길게 늘어진 군집 데이터셋을 사용하여 `n_components=3`인 GMM을 적합시키고 `predict_proba`로 각 데이터 점의 책임도를 추출하라. 처음 10개 데이터 점 각각에 대해 책임도가 가장 높은 성분과 그 값이 두 번째로 높은 값보다 얼마나 큰지를 보고하라. 그 차이가 작다는 것은 그 점에 대해 무엇을 시사하는가?

??? success "연습문제 1 풀이"
    ```python
    import numpy as np
    from sklearn.mixture import GaussianMixture

    np.random.seed(42)
    X1 = np.random.randn(300, 2) @ np.array([[2.0, 0.8], [0.8, 0.5]]) + [2, 0]
    X2 = np.random.randn(200, 2) @ np.array([[1.0, -0.5], [-0.5, 0.8]]) + [-2, 3]
    X3 = np.random.randn(250, 2) * 0.5 + [0, -2]
    X = np.vstack([X1, X2, X3])

    gmm = GaussianMixture(n_components=3, random_state=42).fit(X)
    probs = gmm.predict_proba(X)

    for i in range(10):
        sorted_probs = np.sort(probs[i])[::-1]
        best = np.argmax(probs[i])
        margin = sorted_probs[0] - sorted_probs[1]
        print(f"Point {i}: best component={best}, "
              f"top prob={sorted_probs[0]:.4f}, margin={margin:.4f}")

    # 차이가 작다는 것은 그 점이 두 성분 사이의 경계 근처에 있어
    # 모델이 어느 군집에 속하는지 확신하지 못한다는 뜻이다.
    ```

---

**연습문제 2.**
GMM 모델 선택에서 BIC와 AIC의 차이를 설명하라. BIC가 AIC보다 적은 성분을 선택하는 경향이 있는 이유는 무엇인가? 어떤 상황에서 BIC보다 AIC를 선호할 수 있는가?

??? success "연습문제 2 풀이"
    BIC와 AIC 모두 과적합을 막기 위해 모델 복잡도에 벌점을 주지만 벌점의 강도가 다르다. AIC는 매개변수 개수를 $p$라 할 때 $2p$의 벌점을 더하고, BIC는 데이터 점 개수를 $n$이라 할 때 $p \ln n$을 더한다. $n \geq 8$인 데이터셋이라면 BIC의 벌점이 AIC의 벌점을 넘어서므로, BIC가 성분이 적은 단순한 모델을 선호하게 된다.

    BIC는 점근적으로 일관적이다. 즉 (참 모델이 후보에 포함되어 있다는 가정 아래) $n \to \infty$일 때 참 모델을 선택한다. 반면 AIC는 점근적으로 효율적이어서 예측 오차를 최소화하는 모델을 선택한다. "참된" 군집 개수를 찾는 것이 아니라 밀도 추정이나 예측이 목표일 때, 또는 BIC의 벌점이 지나치게 보수적일 수 있는 작은 데이터셋을 다룰 때 AIC를 선호할 수 있다.

---

**연습문제 3.**
GMM 기반 이상 탐지 시스템을 구현하라. 데이터셋에 GMM을 적합시킨 뒤, 데이터 범위에서 균등 분포로 새로운 점 50개를 생성하라. `score_samples`로 원래 데이터와 균등 분포 점들의 로그가능도를 계산하라. 두 그룹의 로그가능도 히스토그램을 그리고 이들을 분리하는 임계값을 선택하라.

??? success "연습문제 3 풀이"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.mixture import GaussianMixture

    np.random.seed(42)
    X1 = np.random.randn(300, 2) @ np.array([[2.0, 0.8], [0.8, 0.5]]) + [2, 0]
    X2 = np.random.randn(200, 2) @ np.array([[1.0, -0.5], [-0.5, 0.8]]) + [-2, 3]
    X3 = np.random.randn(250, 2) * 0.5 + [0, -2]
    X = np.vstack([X1, X2, X3])

    gmm = GaussianMixture(n_components=3, random_state=42).fit(X)

    # 데이터 범위에서 균등 분포 이상치를 생성한다
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    anomalies = np.column_stack([
        np.random.uniform(x_min, x_max, 50),
        np.random.uniform(y_min, y_max, 50)
    ])

    scores_normal = gmm.score_samples(X)
    scores_anomaly = gmm.score_samples(anomalies)

    plt.figure(figsize=(8, 4))
    plt.hist(scores_normal, bins=50, alpha=0.6, label="Normal data", density=True)
    plt.hist(scores_anomaly, bins=20, alpha=0.6, label="Uniform anomalies", density=True)
    threshold = np.percentile(scores_normal, 5)
    plt.axvline(x=threshold, color="red", linestyle="--", label=f"Threshold={threshold:.1f}")
    plt.xlabel("Log-likelihood")
    plt.ylabel("Density")
    plt.legend()
    plt.title("GMM Anomaly Detection")
    plt.grid(True, alpha=0.3)
    plt.show()

    detected = (scores_anomaly < threshold).sum()
    print(f"Detected {detected}/{len(anomalies)} uniform points as anomalies")
    ```

## 정리하며

**다룬 것** — 가우시안 혼합 모형

GMM은 데이터 분포를 $K$개 가우시안 성분의 가중합 $p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$으로 모형화한다.

앞의 연습문제 3개로 직접 확인할 수 있다.
