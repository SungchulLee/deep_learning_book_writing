# 2차원 주성분 분석 NumPy

주성분 분석(PCA)은 기계 배움에서 가장 바탕이 되는 차원 줄이기 재주이다. 이 보기는 NumPy와 특잇값 쪼개기로 주성분 분석을 맨바닥에서 짜서, 흩어짐이 가장 큰 방향을 찾아 상관 있는 2차원 자료 묶음을 1차원으로 줄이는 법을 보인다. 손으로 하는 절차를 알아 두면 꾸러미나 깊은 배움 방식으로 가기 앞서 필요한 선형 대수 직관이 선다.

## 코드

```python
"""2차원 주성분 분석 NumPy."""
import numpy as np
import matplotlib.pyplot as plt

# === 만든 2차원 자료 묶음 만들기 ==========================================
rng = np.random.default_rng(42)
n = 150
mean_true = np.array([2.0, -1.0])
cov_true = np.array([[3.0, 2.2],
                     [2.2, 2.0]])
X = rng.multivariate_normal(mean_true, cov_true, size=n)

# === 자료의 가운데 맞추기 ========================================================
mu = X.mean(axis=0)
Xc = X - mu

# === 특잇값 쪼개기로 주성분 분석 셈하기 ====================================================
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
V = Vt.T
pc1 = V[:, 0]

explained_variance = (S ** 2) / (n - 1)
total_variance = explained_variance.sum()
variance_ratios = explained_variance / total_variance

# === 차원 줄이기(2차원 -> 1차원) ====================================
scores_1d = Xc @ pc1

# === 다시 세우기(1차원 -> 2차원) ==============================================
X_recon = np.outer(scores_1d, pc1) + mu
reconstruction_error = np.mean((X - X_recon) ** 2)

# === 그려 보기 ===========================================================
t = np.linspace(-4.0, 4.0, 2)
axis_pts = mu + np.outer(t * S[0] / np.sqrt(n), pc1)

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
ax.set_title(f"PCA: 2D -> 1D Projection and Reconstruction\n"
             f"(PC1 explains {variance_ratios[0]:.1%} of variance)")
ax.set_xlabel("x_1")
ax.set_ylabel("x_2")
ax.axis("equal")
ax.legend(loc="best", framealpha=0.9)
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig('pca_2d_to_1d_demo.png', dpi=150, bbox_inches='tight')
plt.show()

if __name__ == "__main__":
    pass
```

## 논의

특잇값 쪼개기 $X_c = U \Sigma V^\top$이 주성분 분석의 일꾼이다. 오른쪽 특이 벡터($V$의 세로줄)가 주방향이고 특잇값 $\sigma_i$은 $\text{var}_i = \sigma_i^2 / (n-1)$으로 방향마다 자료가 퍼진 정도를 담는다. 이 보기의 공분산 행렬은 대각 밖 성분이 크므로($\sqrt{3.0 \times 2.0} \approx 2.45$이 최대인데 2.2) 흩어짐이 한 방향에 몰려 주성분1이 전체 흩어짐의 95%를 넘게 잡는다.

다시 세우기 걸음 $\hat{x} = (x_c \cdot v_1) v_1 + \mu$은 가운데 맞춘 점마다 주성분1에 쏘고 본디 좌표계로 되돌린다. 나온 점은 정확히 주축 위에 놓이며 그림의 회색 쏘기 선이 본디 점에서 다시 세운 점까지의 직교 거리를 보여 준다. 이 거리가 주성분2을 버려 잃은 앎이다.

특잇값 쪼개기 앞에 자료의 가운데를 맞추는 것이 결정적이다. 가운데를 맞추지 않으면 주성분 분석이 자료 구름의 가운데가 아니라 원점을 지나는 방향을 찾아 뜻 없는 쪼개기가 나온다. 이 미리 다듬기가 워낙 중요해 꾸러미 짜기(예컨대 `sklearn.decomposition.PCA`)는 이를 저절로 한다.

## 연습문제

**연습문제 1.**
가운데 맞춘 자료의 공분산 행렬을 $\frac{1}{n-1} X_c^\top X_c$으로 손수 셈하고 그 고윳값이 특잇값 쪼개기의 `explained_variance`와 맞는지 확인하라.

??? success "연습문제 1 풀이"
    ```python
    cov_manual = (Xc.T @ Xc) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov_manual)
    eigvals_sorted = np.sort(eigvals)[::-1]
    print("From SVD:", explained_variance)
    print("From eigh:", eigvals_sorted)
    print("Match:", np.allclose(explained_variance, eigvals_sorted))
    ```
    가운데 맞춘 자료 행렬의 특잇값 쪼개기와 공분산 행렬의 고윳값 쪼개기가 같은 문제를 풀므로 공분산 행렬의 고윳값은 $\sigma_i^2 / (n-1)$과 똑같다.

---

**연습문제 2.**
가운데 맞추기를 건너뛰고 $X$에 곧바로 특잇값 쪼개기를 하면 어떻게 되는가? 실험을 돌려 가운데를 맞췄을 때와 아닐 때의 첫 주방향을 견주어라.

??? success "연습문제 2 풀이"
    ```python
    U_nc, S_nc, Vt_nc = np.linalg.svd(X, full_matrices=False)
    pc1_nc = Vt_nc[0]
    print(f"PC1 (centered):   {pc1}")
    print(f"PC1 (uncentered): {pc1_nc}")
    angle = np.arccos(np.clip(np.abs(pc1 @ pc1_nc), 0, 1)) * 180 / np.pi
    print(f"Angle difference: {angle:.1f} degrees")
    ```
    가운데를 맞추지 않으면 첫 특이 벡터가 흩어짐이 가장 큰 방향이 아니라 평균 쪽을 가리키기 쉽다. 가운데를 맞춘 주성분1과 그렇지 않은 것 사이 각이 꽤 클 수 있어(이 자료 묶음에서는 10~30도) 다시 세우기 어긋남을 가장 작게 하지 못하는 덜 좋은 쏘기가 된다.

---

**연습문제 3.**
주성분 둘을 다 남기도록(곧 2차원으로 쏘고 다시 세우도록) 부호를 넓혀라. 이때 다시 세우기 어긋남이 정확히 0임을 확인하고 까닭을 설명하라.

??? success "연습문제 3 풀이"
    ```python
    scores_2d = Xc @ V  # (n, 2)
    X_recon_2d = scores_2d @ V.T + mu
    error_2d = np.mean((X - X_recon_2d) ** 2)
    print(f"Reconstruction error (2 components): {error_2d:.2e}")
    ```
    $d$차원 공간에서 주성분 $d$개를 모두 남기면 본디 자료를 정확히 되찾으므로 어긋남이 (뜬소수점 정밀도까지) 0이다. 성분을 모두 남기면 주성분 분석은 잃음 있는 눌러 담기가 아니라 직교 바탕 바꿈일 뿐이다.
