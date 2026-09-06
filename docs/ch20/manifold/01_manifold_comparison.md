# 다양체 방법 견줌

다양체 배움 알고리즘은 차원 높은 자료 속에 숨은 낮은 차원 짜임을 드러낸다. 이 보기는 널리 쓰이는 네 방법, 곧 t-SNE, MDS, Isomap, LLE를 만든 자료(스위스 롤)와 실제 자료(MNIST) 모두에서 견준다. 어느 쪽이 어디에 센지 알면 그려 보기와 차원 줄이기에 알맞은 연장을 고를 수 있다.

## 코드

```python
"""다양체 방법 견줌."""
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll, fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS, Isomap, LocallyLinearEmbedding
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# === 만든 자료: 스위스 롤 =============================================
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

# === 실제 자료: MNIST =======================================================
print("\nLoading MNIST (subset of 2 000 samples for speed)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_mnist, y_mnist = mnist.data[:2000].astype(np.float32), mnist.target[:2000].astype(int)
X_mnist = StandardScaler().fit_transform(X_mnist)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_mnist)
print(f"PCA reduced {X_mnist.shape[1]} -> {X_pca.shape[1]} dimensions (95% var).\n")


def plot_digits(X_2d, labels, title=""):
    """숫자 이름표로 색을 입힌 흩뿌림 그림."""
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

# === t-SNE 깊이 보기: 숫자 부분 묶음 ==========================================
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
```

## 논의

다양체 배움 방법마다 다른 기하 직관을 담고 있다. MDS는 둘씩의 거리를 전역으로 지켜, 유클리드가 아닌 잣대로 주성분 분석을 자연스럽게 넓힌 것이 된다. Isomap은 유클리드 거리를 가장 가까운 이웃 그래프에서 셈한 측지 거리로 갈음해 이 생각을 넓히며, 그래서 스위스 롤 같은 굽은 다양체를 "펼" 수 있다. LLE는 국소 선형 이웃에 힘을 쏟아 점마다 이웃의 무게 붙은 아우름으로 다시 세운 뒤 그 무게를 지키는 낮은 차원 배치를 찾는다.

t-SNE는 둘씩의 거리를 조건부 확률로 바꾸고 높은 차원과 낮은 차원 확률 분포 사이 KL 벌어짐을 가장 작게 한다는 점에서 남다르다. 그래서 국소 무리 짜임을 지키는 데 뛰어나며, MNIST 그림에서 숫자 갈래가 그토록 또렷이 갈리는 까닭이다. 다만 t-SNE는 매개변수가 없어(다시 돌리지 않고는 새 점을 쏠 수 없다) 내놓은 그림에서 무리의 크기와 무리 사이 거리는 오해를 부를 수 있다.

이 견줌에서 얻은 실전 관찰 하나는 주성분 분석으로 미리 다듬으면 엄청나게 빨라진다는 것이다. 다양체 방법을 쓰기 앞서 MNIST를 784에서 150쯤 차원으로 줄이면(흩어짐의 95%를 남긴 채) 도는 시간이 한 자릿수만큼 줄고 묻힘 품질에는 거의 영향이 없다. 주성분 분석 뒤에 비선형 방법을 잇는 이 물길은 자료를 살피는 일에서 표준 흐름이다.

## 연습문제

**연습문제 1.**
Isomap과 LLE에 대해 `n_samples=5000`, `n_neighbors=20`으로 스위스 롤 견줌을 돌려라. 방법마다 걸린 시간을 재고 표본 수가 늘 때 어느 알고리즘이 가장 무난하게 버티는지 알려라.

??? success "연습문제 1 풀이"
    표본 수와 이웃 수를 늘린 뒤 맞추기마다 시간을 재어라:
    ```python
    X_swiss, t_swiss = make_swiss_roll(n_samples=5000, noise=0.2, random_state=42)
    for name, model in methods.items():
        if hasattr(model, 'n_neighbors'):
            model.n_neighbors = 20
        t0 = time.time()
        model.fit_transform(X_swiss)
        print(f"{name}: {time.time() - t0:.1f}s")
    ```
    MDS와 t-SNE가 가장 비싸다. 둘씩의 거리 행렬 전체를 다루기 때문이다($O(n^2)$ 공간). Isomap과 LLE는 성긴 이웃 그래프 덕분에 규모를 더 잘 키우지만 Isomap은 최단 길 셈하기가 더 든다. 넷 가운데 표본이 많아지면 흔히 LLE가 가장 빠르다. 성긴 고윳값 문제를 풀기 때문이다.

---

**연습문제 2.**
t-SNE의 무리 크기를 갈래마다의 참 흩어짐으로 읽으면 안 되는 까닭을 설명하라. t-SNE 값 함수의 어떤 성질이 이를 부르는가?

??? success "연습문제 2 풀이"
    t-SNE는 낮은 차원 공간에서는 스튜던트 t 분포를, 높은 차원 공간에서는 정규 분포를 쓴다. 스튜던트 t 분포의 두꺼운 꼬리 덕분에 높은 차원에서 중간이거나 큰 거리를 낮은 차원의 넓은 거리 범위로 나타낼 수 있다. 그래서 속 흩어짐이 큰 무리와 작은 무리가 묻힘에서 비슷한 크기로 보일 수 있다. KL 벌어짐 목표는 높은 차원에서 가까운 점을 멀리 놓는 것(붐빔)에는 벌을 주지만, 먼 점을 가까이 놓거나 들쭉날쭉한 거리에 놓는 것에는 세게 벌을 주지 않는다. 그러므로 t-SNE 무리가 차지한 넓이는 본디 공간에서 그 갈래가 얼마나 퍼져 있는지에 대해 알려 주는 바가 거의 없다.

---

**연습문제 3.**
MNIST 물길을 t-SNE 대신 UMAP(`umap-learn`을 깔아라)을 쓰도록 고쳐라. 2차원 흩뿌림 그림을 눈으로 견주고 빠르기 차이를 재어라. 실전에서 UMAP이 t-SNE보다 나은 점은 무엇인가?

??? success "연습문제 3 풀이"
    ```python
    import umap
    reducer = umap.UMAP(n_components=2, random_state=42)
    pipe = Pipeline([("pca", PCA(n_components=0.95, random_state=42)),
                     ("umap", reducer)])
    t0 = time.time()
    X_umap = pipe.fit_transform(X_mnist)
    print(f"PCA + UMAP: {time.time() - t0:.1f}s")
    ```
    UMAP은 흔히 t-SNE보다 5~10배 빠르고 국소 짜임과 전역 짜임을 모두 더 잘 지키는 묻힘을 낸다. t-SNE와 달리 UMAP은 (`umap.parametric_umap`으로) 매개변수를 가진 바꾸개로 쓸 수 있어 알고리즘 전체를 다시 돌리지 않고도 새 점을 쏠 수 있다. 그래서 표본 밖 쏘기가 필요한 실전 물길에 훨씬 알맞다.
