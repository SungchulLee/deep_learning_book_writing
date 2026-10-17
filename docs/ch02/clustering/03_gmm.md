# 가우시안 혼합 모델

가우시안 혼합 모델(GMM)은 K-평균에 대한 확률적 대안으로, 부드러운 군집 배정을 제공한다. 각 데이터 점이 하나의 군집에 딱 배정되는 대신 각 군집에 속할 확률을 가진다. GMM은 성분별 공분산 행렬을 학습하여 모양과 방향이 서로 다른 타원형 군집을 모델링할 수 있고, BIC나 AIC 같은 정보 기준을 통해 원칙 있는 모델 선택을 지원한다. 학습에는 기댓값 최대화(EM) 알고리즘을 사용하며, 이는 책임도를 계산하는 단계(E 단계)와 파라메터를 갱신하는 단계(M 단계)를 번갈아 수행한다.

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
# 파라메터를 갱신하는 단계(M 단계)를 번갈아 수행한다.
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

# 많은 성분의 웨이트가 0에 가까워진다
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

**출력:**

```
Sample responsibilities (first 5 points):
  Shape: (750, 3)
  Point 0: [1. 0. 0.]
  Point 1: [1. 0. 0.]
  Point 2: [0.999 0.    0.001]
  Point 3: [1. 0. 0.]
  Point 4: [0.998 0.    0.002]

Best k by BIC: 3

Bayesian GMM: 3 active components out of 10
  Weights: [0.001 0.001 0.267 0.323 0.407 0.    0.    0.    0.    0.   ]
Done.
```

## 2. 논의

GMM은 데이터 분포를 $K$개 가우시안 성분의 가중합 $p(\mathbf{x}) = \sum_{k=1}^{K} \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$으로 모델링한다. 여기서 $\pi_k$는 혼합 웨이트, $\boldsymbol{\mu}_k$는 평균, $\boldsymbol{\Sigma}_k$는 공분산 행렬이다. EM 알고리즘은 각 데이터 점에 대한 각 성분의 **책임도(responsibility)** 를 계산하는 단계(E 단계)와 기대 로그우도를 최대화하도록 파라메터를 갱신하는 단계(M 단계)를 번갈아 수행하여 이 파라메터들을 적합시킨다. 딱 떨어지는 배정을 하는 K-평균과 달리, GMM의 책임도는 각 점에 대해 성분들에 대한 확률 분포를 주므로 군집 경계가 겹치는 미묘한 상황도 표현할 수 있다.

**공분산 유형** 은 각 성분의 모양이 얼마나 유연한지를 조절한다. `full` 공분산에서는 각 성분이 임의의 타원 모양을 가질 수 있어 표현력이 가장 크지만 파라메터도 가장 많이 필요하다. `tied`는 모든 성분이 같은 공분산 행렬을 공유하도록 강제하고, `diag`는 축에 평행한 타원으로 제한하며, `spherical`은 K-평균과 비슷한 등방적 군집으로 축소된다. BIC와 AIC는 적합의 품질과 모델 복잡도의 균형을 잡는 모델 선택 기준을 제공한다. BIC는 파라메터 추가에 더 강한 패널티를 매겨 단순한 모델을 선호하고, AIC는 더 관대하다.

**베이즈 GMM** 변형은 혼합 웨이트에 집중 파라메터가 작은 디리클레 사전분포를 두어, 불필요한 성분의 웨이트를 0 쪽으로 몰아 자동으로 억제한다. 이는 모델 선택 문제에 우아한 해법을 제공한다. $K$를 넉넉한 상한으로 두고 사전분포와 데이터가 유효 성분 수에 함께 영향을 주도록 한다. 결과는 사전분포 설정과 초기화에도 의존한다. 또한 GMM은 로그우도 점수가 낮은 데이터 점을 표시함으로써 이상 탐지를 자연스럽게 지원한다. 어떤 성분에도 잘 맞지 않는 점은 이상치일 가능성이 높기 때문이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
코드의 길게 늘어진 군집 데이터셋을 사용하여 `n_components=3`인 GMM을 적합시키고 `predict_proba`로 각 데이터 점의 책임도를 추출하라. 처음 10개 데이터 점 각각에 대해 책임도가 가장 높은 성분과 그 값이 두 번째로 높은 값보다 얼마나 큰지를 보고하라. 그 차이가 작다는 것은 그 점에 대해 무엇을 시사하는가?

</div>

??? success "풀이"
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
    ```

    ```
    Point 0: best component=0, top prob=1.0000, margin=1.0000
    Point 1: best component=0, top prob=1.0000, margin=1.0000
    Point 2: best component=0, top prob=0.9985, margin=0.9973
    Point 3: best component=0, top prob=1.0000, margin=1.0000
    Point 4: best component=0, top prob=0.9975, margin=0.9951
    Point 5: best component=0, top prob=0.9809, margin=0.9619
    Point 6: best component=0, top prob=0.9038, margin=0.8081
    Point 7: best component=0, top prob=0.9995, margin=0.9991
    Point 8: best component=0, top prob=0.9337, margin=0.8674
    Point 9: best component=0, top prob=0.8397, margin=0.6795
    ```

    차이가 작다는 것은 그 점이 두 성분 사이의 경계 근처에 있어 모델이 어느 군집에 속하는지 확신하지 못한다는 뜻이다. **그런데 처음 열 점에는 그런 점이 하나도 없다.** 가장 작은 마진이 0.6795이고 열 점 모두 성분 0에 확신을 가지고 붙는다.

    까닭은 데이터를 쌓은 차례에 있다. `np.vstack([X1, X2, X3])`에서 앞의 300개가 모두 `X1`이므로, "처음 열 개"는 한 덩이에서만 뽑은 것이다. 경계를 보려면 다른 데를 보아야 한다.

    ```python
    margins = np.sort(probs, axis=1)[:, ::-1]
    margins = margins[:, 0] - margins[:, 1]
    print(f"최소 마진: 점 {margins.argmin()}, {margins.min():.4f}")
    print(f"마진 0.5 미만: {(margins < 0.5).sum()}개 / {len(X)}개")
    ```

    ```
    최소 마진: 점 580, 0.0857
    마진 0.5 미만: 15개 / 750개
    ```

    750개 가운데 경계에 걸친 점은 **15개뿐**이고 가장 헷갈리는 점 580조차 top 확률이 0.5428이다. 세 덩이가 꽤 잘 갈라져 있다는 뜻이며, 이것이 GMM의 부드러운 배정이 K-평균의 딱 떨어지는 배정과 **이 데이터에서는** 크게 다르지 않은 까닭이다. 부드러운 배정의 값어치는 덩이가 겹칠 때 드러난다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
GMM 모델 선택에서 BIC와 AIC의 차이를 설명하라. BIC가 AIC보다 적은 성분을 선택하는 경향이 있는 이유는 무엇인가? 어떤 상황에서 BIC보다 AIC를 선호할 수 있는가?

</div>

??? success "풀이"
    BIC와 AIC 모두 과적합을 막기 위해 모델 복잡도에 패널티를 주지만 패널티의 강도가 다르다. AIC는 파라메터 개수를 $p$라 할 때 $2p$의 패널티를 더하고, BIC는 데이터 점 개수를 $n$이라 할 때 $p \ln n$을 더한다. $n \geq 8$인 데이터셋이라면 BIC의 패널티가 AIC의 패널티를 넘어서므로, BIC가 성분이 적은 단순한 모델을 선호하게 된다.

    BIC는 점근적으로 일관적이다. 즉 참 모델의 포함과 필요한 정규성 조건 등이 성립하면 표본이 증가할 때 올바른 모델을 선택할 가능성이 커진다. 혼합 모델에는 식별성과 특이성 문제가 있어 이런 설명을 조건 없이 적용해서는 안 된다. 반면 AIC는 점근적으로 효율적이어서 예측 오차를 최소화하는 모델을 선택한다. "참된" 군집 개수를 찾는 것이 아니라 밀도 추정이나 예측이 목표일 때, 또는 BIC의 패널티가 지나치게 보수적일 수 있는 작은 데이터셋을 다룰 때 AIC를 선호할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
GMM 기반 이상 탐지 시스템을 구현하라. 데이터셋에 GMM을 적합시킨 뒤, 데이터 범위에서 균등 분포로 새로운 점 50개를 생성하라. `score_samples`로 원래 데이터와 균등 분포 점들의 로그우도를 계산하라. 두 그룹의 로그우도 히스토그램을 그리고 이들을 분리하는 임계값을 선택하라.

</div>

??? success "풀이"
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

    ```
    Detected 44/50 uniform points as anomalies
    ```

    임계는 정상 데이터 로그우도의 5백분위이므로 $-4.6704$다. 두 가지를 함께 읽어야 한다.

    **놓친 여섯 개는 잘못이 아니다.** 균등 분포로 뿌린 점이라도 세 덩이가 놓인 자리에 떨어지면 정상 데이터와 구별할 길이 없다. 이상치를 "분포 바깥의 점"으로 정의한 이상, 안쪽에 떨어진 것은 원리적으로 못 잡는다.

    **임계를 5백분위로 잡았다는 것은 거짓양성을 5%로 **고른** 것이다.** 실제로 정상 750개 가운데 38개(5.1%)가 함께 걸린다. 탐지율 88%는 이 거짓양성률을 치르고 산 값이며, 임계를 낮추면 38개가 줄지만 44개도 함께 준다. 이 맞바꿈을 한 수로 줄이지 말고 두 수를 늘 함께 적어야 한다.

## 정리하며

**다룬 것** — 확률로 묶기

GMM은 데이터가 가우시안 $K$개에서 나왔다고 보고 그 가중합 $p(\mathbf{x}) = \sum_k \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$을 적합한다. [K-평균](01_kmeans.md)과 가르는 점이 둘이다.

**하나, 배정이 부드럽다.** K-평균은 점마다 군집 하나를 찍지만 GMM은 **책임도**, 곧 성분들에 대한 확률 분포를 준다. 두 성분이 비슷하게 설명하는 점은 책임도가 비슷하게 나오며, 그 모호함 자체가 쓸모 있는 정보다.

**둘, 모양이 자유롭다.** 공분산 $\boldsymbol{\Sigma}_k$를 성분마다 따로 두면 기울어진 타원도 담는다. `spherical`은 성분마다 등방적 공분산을 사용하지만, 그 자체로 K-평균과 같은 알고리즘이 되지는 않는다. 동일한 작은 분산과 혼합 웨이트 등의 조건에서 GMM의 단단한 배정 극한을 K-평균과 연결할 수 있다.

성분 수를 고르는 길이 둘 있다. **BIC/AIC**는 적합도와 복잡도를 맞바꾼다. BIC의 패널티가 $p\ln n$, AIC가 $2p$이므로 $n \ge 8$부터는 BIC가 더 엄해 단순한 모델을 고른다. **베이즈 GMM**은 더 우아하다. $K$를 넉넉히 주고 디리클레 사전분포가 쓸모없는 성분의 웨이트를 0으로 눌러 주면, 위 출력처럼 10개 가운데 3개만 살아남는다.

그리고 GMM은 **밀도를 실제로 추정하므로** 이상 탐지가 덤으로 따라온다. 로그우도가 낮은 점이 곧 "어느 성분에도 잘 맞지 않는 점"이다. 군집화 알고리즘 가운데 이 성질을 그냥 갖는 것은 드물다.

[5장](../../ch05/index.md)이 이 생각을 훨씬 멀리 끌고 간다. 데이터의 분포를 모델링한다는 뼈대는 그대로 두고, 가우시안 몇 개 대신 뉴럴 넷트웍을 쓰는 것이 VAE와 확산 모델이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
