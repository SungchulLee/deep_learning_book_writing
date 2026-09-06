# DBSCAN과 병합 군집화

K-평균 너머에는 그 근본적인 한계를 다루는 두 가지 중요한 군집화 접근법이 있다. DBSCAN(잡음이 있는 응용을 위한 밀도 기반 공간 군집화)은 조밀하게 모인 점들을 묶어 임의 모양의 군집을 발견하며, 잡음을 이상치로 자연스럽게 처리한다. 병합 군집화는 아래에서 위로 군집의 계층을 쌓아 올려, 데이터의 다중 스케일 구조를 드러내는 덴드로그램을 만든다. 이 방법들은 정확히 K-평균이 실패하는 지점, 즉 볼록하지 않은 군집과 군집 개수를 모르는 상황에서 강점을 보인다.

## 코드

```python
"""Dbscan agglomerative."""
# ---
# title: "DBSCAN and Agglomerative Clustering"
# description: "Density-based and hierarchical clustering with sklearn"
# ---
#
# K-평균 너머의 두 가지 중요한 군집화 접근법:
#   * DBSCAN  -- 임의 모양의 군집을 찾고 잡음을 처리한다
#   * 병합 군집화 -- 덴드로그램을 만드는 상향식 계층 군집화
#
# 이 방법들은 K-평균이 실패하는 곳에서 빛을 발한다. 볼록하지 않은 군집과
# 군집 개수를 모르는 경우이다.
#
# 출처 각색: O'Reilly Hands-On ML, 9장(비지도 학습)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_moons, make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ========================================================================
# 메인
# ========================================================================

# ─── 1.  볼록하지 않은 데이터에 대한 DBSCAN ──────────────────────────────
np.random.seed(42)
X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)
X_moons = StandardScaler().fit_transform(X_moons)

# K-평균은 moons 데이터에서 실패한다
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, random_state=42).fit(X_moons)

# DBSCAN은 성공한다
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
ax2.set_title(f"DBSCAN (eps=0.2, min_samples=5) -- {len(set(db.labels_)) - 1} clusters")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dbscan_vs_kmeans_moons.png", dpi=150)
plt.show()

# ─── 2.  k-거리 그래프로 eps 고르기 ──────────────────────────────────────
print("Computing k-distance graph for eps selection...")
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_moons)
distances, _ = nn.kneighbors(X_moons)
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.xlabel("Points (sorted by 5-NN distance)")
plt.ylabel("5-NN distance")
plt.title("k-Distance Graph (knee = optimal eps)")
plt.axhline(y=0.2, color="red", linestyle="--", label="eps=0.2")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("dbscan_k_distance.png", dpi=150)
plt.show()

# ─── 3.  DBSCAN 매개변수 민감도 ──────────────────────────────────────────
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

# ─── 4.  병합 군집화 ─────────────────────────────────────────────────────
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

# ─── 5.  덴드로그램(scipy 필요) ──────────────────────────────────────────
from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

# 읽기 쉬운 덴드로그램을 위한 작은 부분집합
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
```

## 논의

DBSCAN은 군집을 더 성긴 영역으로 둘러싸인 조밀한 점들의 영역으로 정의한다. 두 개의 매개변수가 필요하다. `eps`(이웃 반경)와 `min_samples`(조밀한 영역을 이루는 데 필요한 최소 점 개수)이다. 핵심 점(core point)은 `eps` 안에 `min_samples`개 이상의 이웃을 가진다. 경계 점(border point)은 어떤 핵심 점의 `eps` 안에 있지만 자신은 핵심 점이 아니다. 잡음 점은 둘 중 어디에도 속하지 않는다. 이 정식화는 임의 모양의 군집을 자연스럽게 발견하고 이상치를 자동으로 표시하므로, moons 데이터셋처럼 볼록하지 않은 구조를 가진 데이터에 특히 적합하다.

`eps`의 선택은 DBSCAN 성능에 결정적이다. **k-거리 그래프** 가 원칙 있는 접근법을 제공한다. 각 점의 $k$번째 최근접 이웃까지의 거리를 계산하고, 이 거리들을 정렬한 뒤, 그래프에서 급격히 꺾이는 지점을 찾는다. 그 꺾이는 지점의 거리가 `eps`의 좋은 후보이다. `eps`를 너무 작게 잡으면 군집이 작은 조각들로 쪼개지고 너무 많은 점이 잡음으로 표시된다. 너무 크게 잡으면 서로 다른 군집이 하나로 합쳐진다.

병합 군집화는 상향식 접근을 취한다. 각 점을 하나의 군집으로 시작하여 가장 가까운 두 군집을 차례로 병합하며 원하는 개수가 남을 때까지 진행한다. **연결 기준(linkage criterion)** 이 군집 간 거리를 어떻게 측정할지 결정한다. Ward는 군집 내 총분산을 최소화하고(조밀한 구형 군집을 만든다), 완전 연결은 최대 쌍별 거리를 사용하며, 평균 연결은 평균 쌍별 거리를, 단일 연결은 최소 거리를 사용한다(길게 늘어진 사슬 모양 군집이 나올 수 있다). 덴드로그램은 병합의 전체 이력을 시각화하며, 트리를 다른 높이에서 자름으로써 군집 개수를 선택할 수 있게 해 준다.

## 연습문제

**연습문제 1.**
`make_moons(n_samples=1000, noise=0.1)`로 데이터셋을 생성하고 `eps` 값을 0.1, 0.2, 0.3, 0.5로 하여 DBSCAN을 적용하라. 각 설정에서 찾아진 군집 개수와 잡음 점 개수를 보고하라. 어떤 `eps`가 가장 좋은 결과를 주는가?

??? success "연습문제 1 풀이"
    ```python
    import numpy as np
    from sklearn.cluster import DBSCAN
    from sklearn.datasets import make_moons
    from sklearn.preprocessing import StandardScaler

    X, _ = make_moons(n_samples=1000, noise=0.1, random_state=42)
    X = StandardScaler().fit_transform(X)

    for eps in [0.1, 0.2, 0.3, 0.5]:
        db = DBSCAN(eps=eps, min_samples=5).fit(X)
        n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        n_noise = (db.labels_ == -1).sum()
        print(f"eps={eps}: clusters={n_clusters}, noise={n_noise}")

    # noise=0.1일 때는 보통 eps=0.3이 가장 좋은 결과를 준다(군집 2개, 잡음 소수).
    # eps=0.1은 조각이 너무 많이 생기고, eps=0.5는 두 초승달을
    # 하나의 군집으로 합쳐 버린다.
    ```

---

**연습문제 2.**
병합 군집화에서 Ward, 완전, 평균, 단일 연결의 차이를 설명하라. 각 연결 방식이 가장 적절한 상황을 기술하라.

??? success "연습문제 2 풀이"
    **Ward 연결** 은 군집 내 총분산의 증가가 최소가 되도록 군집을 병합한다. 조밀하고 크기가 비슷한 군집을 만드는 경향이 있으며, 군집이 대략 구형이고 크기가 비슷할 때 가장 좋다. 예를 들어 구매 행동의 분산이 비슷한 고객 그룹을 세분화할 때 적합하다.

    **완전 연결**(최대 거리)은 두 군집 사이에서 가장 먼 점 쌍을 사용한다. 더 조밀하고 구형에 가까운 군집을 만들며 이상치에 강건하다. 잘 분리된 군집을 원하고 길게 늘어진 그룹이 쪼개지는 것을 감수할 수 있을 때 적합하다.

    **평균 연결** 은 두 군집에 걸친 모든 쌍의 평균 거리를 사용한다. 단일 연결과 완전 연결의 절충으로 중간 크기의 군집을 만든다. 군집 모양이 극단적으로 길지도 완전히 구형도 아닌 일반적인 군집화에 잘 맞는다.

    **단일 연결**(최소 거리)은 두 군집 사이에서 가장 가까운 쌍을 사용한다. 길게 늘어진 사슬 모양 군집을 발견할 수 있어 볼록하지 않은 구조를 탐지하는 데 유용하지만(예: 연결된 사슬을 이루는 유전자 발현 패턴), 잡음에 매우 민감하며 잡음 다리를 통해 서로 다른 군집이 합쳐지는 "사슬 효과"를 낳을 수 있다.

---

**연습문제 3.**
위 코드의 중심 4개짜리 `make_blobs` 데이터셋에 Ward 연결로 병합 군집화를 적용하되 군집 개수를 2부터 8까지 바꿔 가며 실행하라. 각각의 실루엣 점수를 계산하여 최적 군집 개수를 찾고, 덴드로그램 기반 접근과 비교하라.

??? success "연습문제 3 풀이"
    ```python
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score
    from scipy.cluster.hierarchy import dendrogram, linkage
    import matplotlib.pyplot as plt

    X, _ = make_blobs(n_samples=600, centers=4,
                      cluster_std=[1.0, 2.5, 0.5, 1.5], random_state=42)

    # 실루엣 분석
    for n in range(2, 9):
        agg = AgglomerativeClustering(n_clusters=n, linkage="ward")
        labels = agg.fit_predict(X)
        sil = silhouette_score(X, labels)
        print(f"n_clusters={n}: silhouette={sil:.4f}")

    # 덴드로그램
    Z = linkage(X, method="ward")
    plt.figure(figsize=(12, 5))
    dendrogram(Z, truncate_mode="level", p=5)
    plt.title("Truncated Dendrogram (Ward)")
    plt.xlabel("Cluster size")
    plt.ylabel("Distance")
    plt.show()

    # 두 방법 모두 최적 군집 개수로 4를 가리켜야 한다.
    # 실루엣 점수는 n_clusters=4에서 정점을 찍고, 덴드로그램은 마지막
    # 병합 전에 네 개의 큰 수직 간격을 보여준다.
    ```
