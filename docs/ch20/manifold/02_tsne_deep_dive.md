# t-SNE 깊이 들여다보기

t-SNE(van der Maaten와 Hinton, 2008)는 차원 높은 자료를 그려 보는 데 가장 널리 쓰이는 다양체 배움 방법이다. 이 보기는 핵심 웃매개변수인 헷갈림도, 되풀이 횟수, 주성분 분석 미리 다듬기가 나온 묻힘의 품질과 빠르기에 어떻게 영향을 주는지 살핀다. 이 매개변수를 알아야 믿을 만한 그림을 만들고 흔한 오해를 피할 수 있다.

## 1. 코드

```python
"""t-SNE 깊이 들여다보기."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml, make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# === 헷갈림도의 영향 ===================================================
print("Generating blobs to show perplexity effect...")
X_blobs, y_blobs = make_blobs(
    n_samples=600, centers=4, cluster_std=[1.0, 2.5, 0.5, 1.5], random_state=42
)
X_blobs = StandardScaler().fit_transform(X_blobs)

perplexities = [5, 15, 30, 50, 100]
fig, axes = plt.subplots(1, len(perplexities), figsize=(20, 4))
for ax, perp in zip(axes, perplexities):
    X_2d = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000).fit_transform(X_blobs)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_blobs, cmap="tab10", s=10)
    ax.set_title(f"perplexity={perp}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("t-SNE: Effect of Perplexity", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("tsne_perplexity_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# === n_iter(모임)의 영향 =========================================
print("Showing convergence with increasing iterations...")
iterations = [250, 500, 1000, 2000, 5000]
fig, axes = plt.subplots(1, len(iterations), figsize=(20, 4))
for ax, n_iter in zip(axes, iterations):
    X_2d = TSNE(n_components=2, perplexity=30, n_iter=n_iter, random_state=42).fit_transform(X_blobs)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y_blobs, cmap="tab10", s=10)
    ax.set_title(f"n_iter={n_iter}")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("t-SNE: Convergence with Iterations", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("tsne_iterations_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# === MNIST에서 주성분 분석으로 빠르게 하는 재주 ============================================
print("\nMNIST: comparing raw t-SNE vs PCA+t-SNE speed...")
import time

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist.data[:3000].astype(np.float32)
y = mnist.target[:3000].astype(int)

t0 = time.time()
X_direct = TSNE(n_components=2, random_state=42).fit_transform(X)
t_direct = time.time() - t0

t0 = time.time()
X_pca = PCA(n_components=0.95, random_state=42).fit_transform(X)
X_pca_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_pca)
t_pca_tsne = time.time() - t0

print(f"  Direct t-SNE:    {t_direct:.1f}s")
print(f"  PCA + t-SNE:     {t_pca_tsne:.1f}s  (speed-up: {t_direct / t_pca_tsne:.1f}x)")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, X_2d, title in zip(
    axes,
    [X_direct, X_pca_tsne],
    [f"Direct t-SNE ({t_direct:.1f}s)", f"PCA + t-SNE ({t_pca_tsne:.1f}s)"],
):
    for digit in range(10):
        mask = y == digit
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=5, label=str(digit), alpha=0.6)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
axes[0].legend(markerscale=3, fontsize=8)
plt.suptitle("PCA Pre-processing Accelerates t-SNE", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("tsne_pca_speedup.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nDone.")


if __name__ == "__main__":
    pass
```

## 2. 논의

t-SNE의 헷갈림도 매개변수는 점마다 헤아리는 실효 이웃 수를 부드럽게 재는 잣대 노릇을 한다. 헷갈림도가 낮으면(5~10) 알고리즘이 아주 국소한 짜임에 매달려 작고 빽빽한 무리를 여럿 만든다. 헷갈림도가 높으면(50~100) 이웃이 넓어져 가까운 무리가 합쳐지고 더 전역적인 짜임이 드러난다. 5와 50 사이 값 여럿을 시험해 여러 자리매김에서 한결같이 남는 짜임을 찾기를 권한다. 그런 짜임이 어떤 매개변수 고름이 만든 헛것이 아니라 참된 결일 가능성이 크다.

모임의 몸가짐도 요긴하다. 되풀이가 너무 적으면(500 미만) 가장 좋게 하기가 자리를 잡지 못해 묻힘이 일그러지거나 잡음이 낀 것처럼 보일 수 있다. 1000번쯤이면 흔히 안정되지만 복잡한 자료 묶음은 2000~5000번이 도움이 될 수 있다. 가장 좋게 하는 동안 KL 벌어짐을 지켜보는 것이(`TSNE(verbose=2)`으로 볼 수 있다) 모임을 확인하는 가장 믿을 만한 길이다.

주성분 분석으로 미리 다듬는 재주는 실전에서 거의 언제나 쓴다. t-SNE는 모든 점 사이의 거리를 셈하므로 차원을 784에서 150쯤으로 줄이면 남아도는 흩어짐이 걷히고 거리 셈하기 값이 크게 준다. 주성분 분석이 짜임이 아니라 잡음을 보태는 흩어짐 작은 방향만 없애므로 마지막 묻힘의 품질은 사실상 같다.

## 연습문제

**연습문제 1.**
방울 자료 묶음에 대해 헷갈림도 2, 5, 10, 30, 50, 100으로 t-SNE를 돌려라. 저마다 알려진 무리 이름표로 묻힘의 실루엣 점수를 셈하라. 어느 헷갈림도에서 실루엣 점수가 가장 큰가?

??? success "연습문제 1 풀이"
    ```python
    from sklearn.metrics import silhouette_score
    for perp in [2, 5, 10, 30, 50, 100]:
        X_2d = TSNE(n_components=2, perplexity=perp, random_state=42).fit_transform(X_blobs)
        score = silhouette_score(X_2d, y_blobs)
        print(f"perplexity={perp:3d}  silhouette={score:.3f}")
    ```
    흔히 이 자료 묶음에서는 헷갈림도 15~30쯤이 실루엣 점수가 가장 높다. 국소한 정밀함과 전역의 얼거리를 저울질하기 때문이다. 헷갈림도가 너무 낮으면 자연스러운 무리가 조각나고, 너무 높으면 서로 다른 무리가 합쳐진다.

---

**연습문제 2.**
t-SNE가 왜 매개변수가 없는지 설명하고 이 한계가 실전에서 무엇을 뜻하는지 적어라. 매개변수를 가진 t-SNE는 이를 어떻게 다루는가?

??? success "연습문제 2 풀이"
    여느 t-SNE는 묻은 점의 자리를 곧바로 가장 좋게 할 뿐, 들임 공간에서 묻힘 공간으로 가는 대응 함수를 배우지 않는다. 그래서 새 자료가 오면 묻힘 전체를 맨바닥에서 다시 셈해야 한다. 곧 `transform` 방법이 없다. 매개변수를 가진 t-SNE는 곧바른 가장 좋게 하기를 대응 $f: \mathbb{R}^d \to \mathbb{R}^2$을 배우는 신경망으로 갈음한다. 한 번 익히면 그 그물이 앞먹임 한 번으로 못 본 점을 쏠 수 있어 표본 밖 쏘기가 필요한 실전 물길에 알맞다.

---

**연습문제 3.**
t-SNE를 돌리기 앞서 설명하는 흩어짐을 50%, 80%, 95%, 99%로 두어 주성분 분석으로 미리 다듬는 것을 견주도록 MNIST 실험을 고쳐라. 네 묻힘을 나란히 그리고 전체 도는 시간을 재어라. 어느 흩어짐 문턱에서 품질이 떨어지기 시작하는가?

??? success "연습문제 3 풀이"
    ```python
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, var_ratio in zip(axes, [0.50, 0.80, 0.95, 0.99]):
        t0 = time.time()
        X_pca = PCA(n_components=var_ratio, random_state=42).fit_transform(X)
        X_2d = TSNE(n_components=2, random_state=42).fit_transform(X_pca)
        elapsed = time.time() - t0
        n_dims = X_pca.shape[1]
        for digit in range(10):
            mask = y == digit
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=5, alpha=0.6)
        ax.set_title(f"{var_ratio:.0%} var ({n_dims}d, {elapsed:.1f}s)")
    plt.tight_layout()
    plt.show()
    ```
    흩어짐을 80~95% 남기면 흔히 99%인 경우와 구별할 수 없는 묻힘을 훨씬 빠르게 얻는다. 50%에서는 낱낱을 가르는 화소 수준의 특징이 버려져 숫자 무리가 겹치기 시작할 수 있다.

## 정리하며

**다룬 것** — t-SNE 깊이 들여다보기

t-SNE의 헷갈림도 매개변수는 점마다 헤아리는 실효 이웃 수를 부드럽게 재는 잣대 노릇을 한다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
