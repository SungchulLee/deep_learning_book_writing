# 2차원 주성분 분석 Sklearn

Scikit-learn은 가운데 맞추기, 쪼개기, 다시 세우기를 말끔한 fit/transform 겉면으로 다루는 실전용 주성분 분석 짜기를 준다. 이 보기는 NumPy 판과 같은 2차원에서 1차원 줄이기를 `sklearn.decomposition.PCA`으로 보이며, 저절로 가운데를 맞추고 흩어짐 통계를 갖추며 `inverse_transform`으로 다시 세우는 편함을 드러낸다.

## 1. 코드

```python
"""2차원 주성분 분석 Sklearn."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# === 만든 2차원 자료 묶음 만들기 ==========================================
rng = np.random.default_rng(42)
n = 150
mean_true = np.array([2.0, -1.0])
cov_true = np.array([[3.0, 2.2],
                     [2.2, 2.0]])
X = rng.multivariate_normal(mean_true, cov_true, size=n)

# === scikit-learn으로 주성분 분석 맞추기 ==============================================
pca = PCA(n_components=1)
scores_1d = pca.fit_transform(X)
X_recon = pca.inverse_transform(scores_1d)

mu = pca.mean_
pc1 = pca.components_[0]
var_ratio = pca.explained_variance_ratio_[0]
reconstruction_error = np.mean((X - X_recon) ** 2)

# === 그려 보기 ===========================================================
score_std = scores_1d.std()
t = np.linspace(-4.0 * score_std, 4.0 * score_std, 2)
axis_pts = mu + np.outer(t, pc1)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], s=25, alpha=0.6, color='C0', label="Original points", zorder=2)
ax.scatter(X_recon[:, 0], X_recon[:, 1], s=18, alpha=0.9, marker="x", color='C1',
           label="Projection (1D->2D)", zorder=3)
step = max(1, n // 40)
for i in range(0, n, step):
    ax.plot([X[i, 0], X_recon[i, 0]], [X[i, 1], X_recon[i, 1]],
            color='gray', linewidth=0.8, alpha=0.6, zorder=1)
ax.plot(axis_pts[:, 0], axis_pts[:, 1], color='C2', linewidth=2.0,
        label="Principal axis (PC1)", zorder=4)
ax.scatter([mu[0]], [mu[1]], s=70, edgecolor="k", facecolor="none",
           linewidth=2, label="Mean", zorder=5)
ax.set_title(f"PCA (sklearn): 2D -> 1D (Explained Var: {var_ratio:.2%})")
ax.set_xlabel("x_1")
ax.set_ylabel("x_2")
ax.axis("equal")
ax.legend(loc="best", framealpha=0.9, fontsize=9)
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig('pca_2d_to_1d_sklearn.png', dpi=150, bbox_inches='tight')
plt.show()

if __name__ == "__main__":
    pass
```

## 2. 논의

sklearn의 PCA 갈래가 주성분 분석 흐름 전체를 감싼다. `fit_transform(X)`을 부르면 속으로 자료의 가운데를 맞추고(평균을 `pca.mean_`에 담는다) 특잇값 쪼개기를 셈해 쏜 점수를 돌려준다. `inverse_transform`은 점수에 주성분을 곱하고 평균을 도로 더해 본디 공간의 나타냄을 다시 세운다. NumPy 판에서 손으로 하던 기록을 없애 준다.

sklearn의 핵심 설계 고름 하나는 `pca.components_`이 주방향을 세로줄이 아니라 가로줄로 담는다는 것이다. 가로줄마다 바탕 벡터라는 관례와는 맞지만, 선형 대수 교과서의 세로줄 관례를 기대한 이는 걸려 넘어질 수 있다. 설명하는 흩어짐 비는 `pca.explained_variance_ratio_`으로 볼 수 있고 성분을 모두 남기면 합이 1.0이다.

실전 부호에는 sklearn PCA가 더 주는 것이 있다. `n_components`을 실수로(예컨대 0.95) 주면 그만큼의 흩어짐을 남길 성분을 알아서 고르고, `svd_solver='randomized'`은 큰 행렬의 셈을 빠르게 하며, `whiten=True`은 성분을 단위 흩어짐으로 다시 재어 특징 잣수에 민감한 알고리즘의 미리 다듬기로 쓸모 있다.

## 연습문제

**연습문제 1.**
같은 자료 묶음에 `PCA(n_components=0.95)`을 써라. sklearn은 성분을 몇 개 고르며 저마다 설명하는 흩어짐 비는 얼마인가?

??? success "연습문제 1 풀이"
    ```python
    pca_auto = PCA(n_components=0.95)
    pca_auto.fit(X)
    print(f"Components selected: {pca_auto.n_components_}")
    print(f"Variance ratios: {pca_auto.explained_variance_ratio_}")
    ```
    상관이 큰 이 2차원 자료 묶음에서는 주성분1이 이미 흩어짐의 95%를 넘게 설명하므로 sklearn이 성분 하나를 고른다. 그 흩어짐 비는 대략 0.96이다.

---

**연습문제 2.**
sklearn 주성분 분석의 다시 세우기 평균 제곱 어긋남을 이론 하한(주성분2 방향의 흩어짐)과 견주어라. 둘은 같은가? 까닭을 설명하라.

??? success "연습문제 2 풀이"
    ```python
    pca_full = PCA(n_components=2).fit(X)
    var_pc2 = pca_full.explained_variance_[1]
    print(f"Reconstruction MSE: {reconstruction_error:.6f}")
    print(f"PC2 variance:       {var_pc2:.6f}")
    ```
    주성분2을 버렸을 때의 다시 세우기 평균 제곱 어긋남은 (치우침 없는 흩어짐 어림개에서 오는 $n/(n-1)$ 인수까지) 주성분2 방향의 흩어짐과 같다. 이는 주성분 분석의 바탕 성질이다. 곧 가장 좋은 $k$차원 쏘기가 다시 세우기 어긋남을 가장 작게 하고, 남는 어긋남은 버린 고윳값의 합과 같다.

---

**연습문제 3.**
`PCA(whiten=True)`을 써서 바꾼 점수의 공분산 행렬을 살펴라. 그것이 항등 행렬임을 확인하고 하얗게 하기가 언제 이로운지 설명하라.

??? success "연습문제 3 풀이"
    ```python
    pca_white = PCA(n_components=2, whiten=True)
    scores_white = pca_white.fit_transform(X)
    cov_white = np.cov(scores_white.T)
    print("Covariance of whitened scores:")
    print(np.round(cov_white, 4))
    ```
    공분산 행렬이 (거의) 항등이며, 이는 하얗게 하기가 특징의 상관을 없애고 흩어짐을 1로 고른다는 것을 확인해 준다. 뒤따르는 알고리즘(예컨대 독립 성분 분석, k-평균, 신경망)이 들임 분포가 방향에 무관하다고 여기거나 특징 잣수에 민감할 때 하얗게 하기가 이롭다.

## 정리하며

**다룬 것** — 2차원 주성분 분석 Sklearn

sklearn의 PCA 갈래가 주성분 분석 흐름 전체를 감싼다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
