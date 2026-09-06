# K-평균 군집화

K-평균은 가장 널리 쓰이는 군집화 알고리즘으로, 각 점을 가장 가까운 중심에 배정하고 중심을 군집 평균으로 갱신하는 과정을 반복하여 데이터를 $k$개 그룹으로 나눈다. 고객 세분화부터 이미지 압축까지 폭넓게 응용되는 기초적인 비지도 학습 방법이다. 군집 개수를 정하는 법과 대규모 데이터셋으로 확장하는 법을 포함해 K-평균을 이해하는 것은 레이블이 없는 데이터를 다루는 모든 실무자에게 필수적이다.

## 코드

```python
"""Kmeans."""
# ---
# title: "K-Means Clustering"
# description: "K-Means from basics to advanced usage with sklearn"
# ---
#
# K-평균은 가장 널리 쓰이는 군집화 알고리즘이다. 이 스크립트는 다음을 다룬다.
#   1. 합성 데이터에 대한 기본 K-평균
#   2. k를 고르기 위한 엘보 방법과 실루엣 분석
#   3. 이미지에 대한 K-평균(색상 양자화)
#   4. 대규모 데이터셋을 위한 미니배치 K-평균
#
# 출처 각색: O'Reilly Hands-On ML, 9장(비지도 학습)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs, load_digits
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

# ========================================================================
# 메인
# ========================================================================

# ─── 1.  기본 K-평균 ──────────────────────────────────────────────────────
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

# ─── 2.  엘보 방법 ────────────────────────────────────────────────────────
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

# ─── 3.  실루엣 다이어그램 ────────────────────────────────────────────────
def plot_silhouette(X, labels, k, ax):
    """실루엣 다이어그램을 그린다."""
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

# ─── 4.  미니배치 K-평균 ──────────────────────────────────────────────────
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
```

## 논의

K-평균은 두 단계를 번갈아 수행하며 동작한다. 각 데이터 점을 가장 가까운 중심에 배정하는 단계(배정 단계)와, 각 중심을 배정된 점들의 평균으로 다시 계산하는 단계(갱신 단계)이다. 이 과정은 수렴이 보장되지만 전역 최적이 아닌 국소 최적을 찾을 수 있다. 그래서 `n_init`으로 서로 다른 무작위 초기화를 여러 번 시도하여 가장 좋은 결과를 취한다.

적절한 군집 개수 $k$를 고르는 것이 K-평균의 주요 과제 중 하나이다. **엘보 방법** 은 관성(군집 내 제곱합)을 $k$에 대해 그려서, 군집을 더 늘려도 이득이 줄어드는 "꺾이는 지점"을 찾는다. **실루엣 점수** 는 더 원칙 있는 지표로, 각 점이 자기 군집에 얼마나 유사한지를 가장 가까운 이웃 군집과 비교하여 측정하며 값의 범위는 $-1$부터 $1$까지이고 클수록 좋다.

대규모 데이터셋에서는 **미니배치 K-평균** 이 실용적인 대안이 된다. 전체 데이터셋 대신 작은 무작위 부분집합으로 중심을 갱신한다. 계산 시간이 극적으로 줄면서도 표준 K-평균에 거의 필적하는 결과를 낸다. 대가는 관성이 약간 증가하는 것이지만, 수만 개 이상의 점을 가진 데이터셋에서는 속도 향상이 몇 자릿수에 이를 수 있다.

## 연습문제

**연습문제 1.**
위 코드의 blob 데이터셋을 사용하여 $k = 3, 5, 7, 10$으로 K-평균을 실행하고 각각의 실루엣 점수를 계산하라. 실루엣 점수를 $k$의 함수로 그려 최적 군집 개수를 찾아라.

??? success "연습문제 1 풀이"
    ```python
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score
    import matplotlib.pyplot as plt

    np.random.seed(42)
    X, _ = make_blobs(n_samples=500, centers=5, cluster_std=[1.0, 0.7, 1.2, 0.9, 1.5], random_state=42)

    k_values = [3, 5, 7, 10]
    scores = []
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        s = silhouette_score(X, km.labels_)
        scores.append(s)
        print(f"k={k}: silhouette={s:.4f}")

    plt.plot(k_values, scores, "bo-")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title("Silhouette Score vs k")
    plt.grid(True, alpha=0.3)
    plt.show()
    # k=5에서 정점을 찍어 실제 군집 개수를 확인해 준다.
    ```

---

**연습문제 2.**
K-평균이 구형(볼록) 군집을 만드는 경향이 있는 이유를 설명하라. 이 알고리즘은 어떤 조건에서 실패하며, 대신 어떤 군집화 방법을 쓸 수 있는가?

??? success "연습문제 2 풀이"
    K-평균은 유클리드 거리를 이용해 각 점을 가장 가까운 중심에 배정하므로 보로노이 분할, 즉 특징 공간의 볼록 다각형 영역을 만든다. 이는 군집이 대략 구형(등방적)이고 크기가 비슷하다고 암묵적으로 가정하는 셈이다. 군집이 볼록하지 않거나(예: 초승달 모양, 고리 모양), 밀도가 크게 다르거나, 크기 차이가 크면 실패한다. 대안으로는 임의 모양 군집을 위한 DBSCAN, 부드러운 배정을 하는 타원형 군집을 위한 가우시안 혼합 모형, 그래프 구조로 포착 가능한 복잡한 군집 기하를 위한 스펙트럴 군집화가 있다.

---

**연습문제 3.**
미니배치 K-평균 실험을 수정하여 배치 크기 128, 512, 1024, 4096을 시험하라. 각 배치 크기에 대해 실행 시간, 최종 관성, 실루엣 점수를 기록하라. 속도와 품질 사이의 절충을 그려라.

??? success "연습문제 3 풀이"
    ```python
    import time
    import numpy as np
    from sklearn.cluster import KMeans, MiniBatchKMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics import silhouette_score

    X_large, _ = make_blobs(n_samples=50000, centers=10, random_state=42)

    batch_sizes = [128, 512, 1024, 4096]
    results = []

    # 전체 K-평균 기준선
    t0 = time.time()
    km_full = KMeans(n_clusters=10, random_state=42, n_init=10).fit(X_large)
    t_full = time.time() - t0
    sil_full = silhouette_score(X_large, km_full.labels_)
    print(f"Full KMeans: time={t_full:.2f}s, inertia={km_full.inertia_:.0f}, sil={sil_full:.4f}")

    for bs in batch_sizes:
        t0 = time.time()
        km = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=bs).fit(X_large)
        elapsed = time.time() - t0
        sil = silhouette_score(X_large, km.labels_)
        results.append((bs, elapsed, km.inertia_, sil))
        print(f"batch_size={bs}: time={elapsed:.2f}s, inertia={km.inertia_:.0f}, sil={sil:.4f}")

    # 배치 크기가 클수록 전체 K-평균의 품질에 가까워지지만 시간이 더 걸리고,
    # 배치 크기가 작을수록 빠르지만 정확도를 일부 희생한다.
    ```
