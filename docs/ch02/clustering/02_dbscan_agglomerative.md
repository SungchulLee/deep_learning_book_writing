# DBSCAN과 병합 군집화

K-평균 너머에는 그 근본적인 한계를 다루는 두 가지 중요한 군집화 접근법이 있다. DBSCAN(잡음이 있는 응용을 위한 밀도 기반 공간 군집화)은 조밀하게 모인 점들을 묶어 임의 모양의 군집을 발견하며, 잡음을 이상치로 자연스럽게 처리한다. 병합 군집화는 아래에서 위로 군집의 계층을 쌓아 올려, 데이터의 다중 스케일 구조를 드러내는 덴드로그램을 만든다. 이 방법들은 정확히 K-평균이 실패하는 지점, 즉 볼록하지 않은 군집과 군집 개수를 모르는 상황에서 강점을 보인다.

## 1. 코드

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

**출력:**

```
Computing k-distance graph for eps selection...
Done.
```

## 2. 논의

DBSCAN은 군집을 더 성긴 영역으로 둘러싸인 조밀한 점들의 영역으로 정의한다. 두 개의 매개변수가 필요하다. `eps`(이웃 반경)와 `min_samples`(조밀한 영역을 이루는 데 필요한 최소 점 개수)이다. 핵심 점(core point)은 `eps` 안에 `min_samples`개 이상의 이웃을 가진다. 경계 점(border point)은 어떤 핵심 점의 `eps` 안에 있지만 자신은 핵심 점이 아니다. 잡음 점은 둘 중 어디에도 속하지 않는다. 이 정식화는 임의 모양의 군집을 자연스럽게 발견하고 이상치를 자동으로 표시하므로, moons 데이터셋처럼 볼록하지 않은 구조를 가진 데이터에 특히 적합하다.

`eps`의 선택은 DBSCAN 성능에 결정적이다. **k-거리 그래프** 가 원칙 있는 접근법을 제공한다. 각 점의 $k$번째 최근접 이웃까지의 거리를 계산하고, 이 거리들을 정렬한 뒤, 그래프에서 급격히 꺾이는 지점을 찾는다. 그 꺾이는 지점의 거리가 `eps`의 좋은 후보이다. `eps`를 너무 작게 잡으면 군집이 작은 조각들로 쪼개지고 너무 많은 점이 잡음으로 표시된다. 너무 크게 잡으면 서로 다른 군집이 하나로 합쳐진다.

병합 군집화는 상향식 접근을 취한다. 각 점을 하나의 군집으로 시작하여 가장 가까운 두 군집을 차례로 병합하며 원하는 개수가 남을 때까지 진행한다. **연결 기준(linkage criterion)** 이 군집 간 거리를 어떻게 측정할지 결정한다. Ward는 군집 내 총분산을 최소화하고(조밀한 구형 군집을 만든다), 완전 연결은 최대 쌍별 거리를 사용하며, 평균 연결은 평균 쌍별 거리를, 단일 연결은 최소 거리를 사용한다(길게 늘어진 사슬 모양 군집이 나올 수 있다). 덴드로그램은 병합의 전체 이력을 시각화하며, 트리를 다른 높이에서 자름으로써 군집 개수를 선택할 수 있게 해 준다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`make_moons(n_samples=1000, noise=0.1)`로 데이터셋을 생성하고 `eps` 값을 0.1, 0.2, 0.3, 0.5로 하여 DBSCAN을 적용하라. 각 설정에서 찾아진 군집 개수와 잡음 점 개수를 보고하라. 어떤 `eps`가 가장 좋은 결과를 주는가?

</div>

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

    ```

    실제로 돌리면 이렇게 나온다.

    ```
    eps=0.1: clusters=8, noise=124
    eps=0.2: clusters=2, noise=5
    eps=0.3: clusters=1, noise=0
    eps=0.5: clusters=1, noise=0
    ```

    **가장 좋은 것은 eps=0.2다.** 초승달 둘을 정확히 갈라내고 잡음은 5개뿐이다.

    양쪽 끝이 어떻게 무너지는지 보라. `eps=0.1`은 반경이 좁아 이웃을 충분히 못 모으므로 군집이 여덟 조각으로 부서지고 124개가 잡음이 된다. **`eps=0.3`에서는 이미 두 초승달이 하나로 붙는다.** 잡음이 0이라는 것이 좋은 신호가 아니라 "모두 한 덩이로 보았다"는 뜻이다.

    곧 쓸 만한 창이 0.2 언저리로 좁다. DBSCAN이 `eps`에 민감하다는 말의 실제 모습이며, 본문의 k-거리 그래프가 필요한 까닭이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
병합 군집화에서 Ward, 완전, 평균, 단일 연결의 차이를 설명하라. 각 연결 방식이 가장 적절한 상황을 기술하라.

</div>

??? success "연습문제 2 풀이"
    **Ward 연결** 은 군집 내 총분산의 증가가 최소가 되도록 군집을 병합한다. 조밀하고 크기가 비슷한 군집을 만드는 경향이 있으며, 군집이 대략 구형이고 크기가 비슷할 때 가장 좋다. 예를 들어 구매 행동의 분산이 비슷한 고객 그룹을 세분화할 때 적합하다.

    **완전 연결**(최대 거리)은 두 군집 사이에서 가장 먼 점 쌍을 사용한다. 더 조밀하고 구형에 가까운 군집을 만들며 이상치에 강건하다. 잘 분리된 군집을 원하고 길게 늘어진 그룹이 쪼개지는 것을 감수할 수 있을 때 적합하다.

    **평균 연결** 은 두 군집에 걸친 모든 쌍의 평균 거리를 사용한다. 단일 연결과 완전 연결의 절충으로 중간 크기의 군집을 만든다. 군집 모양이 극단적으로 길지도 완전히 구형도 아닌 일반적인 군집화에 잘 맞는다.

    **단일 연결**(최소 거리)은 두 군집 사이에서 가장 가까운 쌍을 사용한다. 길게 늘어진 사슬 모양 군집을 발견할 수 있어 볼록하지 않은 구조를 탐지하는 데 유용하지만(예: 연결된 사슬을 이루는 유전자 발현 패턴), 잡음에 매우 민감하며 잡음 다리를 통해 서로 다른 군집이 합쳐지는 "사슬 효과"를 낳을 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
위 코드의 중심 4개짜리 `make_blobs` 데이터셋에 Ward 연결로 병합 군집화를 적용하되 군집 개수를 2부터 8까지 바꿔 가며 실행하라. 각각의 실루엣 점수를 계산하여 최적 군집 개수를 찾고, 덴드로그램 기반 접근과 비교하라.

</div>

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

    ```

    실루엣은 이렇게 나온다.

    ```
    n_clusters=2: 0.5782     n_clusters=6: 0.6317
    n_clusters=3: 0.7092     n_clusters=7: 0.5735
    n_clusters=4: 0.6999     n_clusters=8: 0.5677
    n_clusters=5: 0.6246
    ```

    **참 군집 수가 4인데 꼭대기는 3이다.** 다만 4와의 차이가 0.009로 아주 작다.

    까닭은 자료를 만들 때 준 `cluster_std=[1.0, 2.5, 0.5, 1.5]`에 있다. 퍼짐이 다섯 배까지 차이 나므로, 표준편차 2.5짜리 넓은 덩이가 이웃과 겹쳐 하나로 읽힌다. [K-평균 쪽](01_kmeans.md) 연습문제 1에서 본 것과 같은 일이다.

    **덴드로그램이 실루엣보다 정직한 자리가 여기다.** 덴드로그램은 답 하나를 내놓지 않고 병합이 일어난 높이를 모두 보여 주므로, 3과 4 가운데 어느 쪽을 자를지 사람이 보고 정할 수 있다. 자가 하나뿐일 때 생기는 문제를 그림이 덜어 준다.

## 정리하며

**다룬 것** — K-평균이 못 하는 두 가지를 메우는 방법

[K-평균](01_kmeans.md)의 한계가 둘이었다. 볼록한 군집만 찾고, $k$를 미리 알아야 한다. 이 절의 두 방법이 각각을 메운다.

**DBSCAN은 모양을 묻지 않는다.** 군집을 "성긴 곳에 둘러싸인 조밀한 곳"으로 정의하므로 초승달이든 고리든 찾아내고, 어디에도 속하지 않는 점은 **잡음으로 따로 표시한다**. $k$를 줄 필요도 없다.

대신 `eps`를 주어야 하고, **거기에 매우 민감하다.** 연습문제 1에서 보듯 0.2에서는 초승달 둘을 정확히 가르는데 0.3에서 이미 하나로 붙는다. 쓸 만한 창이 그만큼 좁다. k-거리 그래프의 꺾이는 자리를 보는 것이 그 창을 찾는 표준 방법이다.

**병합 군집화는 답을 하나로 정하지 않는다.** 점 하나짜리 군집에서 시작해 가까운 것끼리 차례로 붙이며 그 이력을 덴드로그램으로 남기므로, **나중에 아무 높이에서나 잘라** 군집 수를 정할 수 있다. 연결 기준이 성격을 정한다 — Ward는 조밀한 구형을, 단일 연결은 사슬 모양을 만들되 잡음 다리에 약하다.

두 절의 연습문제가 함께 말하는 것이 하나 있다. **실루엣은 참 군집 수를 되찾아 주지 않는다.** 갈래마다 퍼짐이 다르면 겹친 둘을 하나로 세는 쪽이 점수가 높다. 그래서 덴드로그램처럼 **판단을 사람에게 남기는 도구**가 여전히 쓸모 있다.

앞의 연습문제 3개로 직접 확인할 수 있다.
