# 2차원 주성분 분석 NumPy

주성분 분석(PCA)은 기계 배움에서 가장 바탕이 되는 차원 줄이기 재주이다. 이 보기는 NumPy와 특잇값 쪼개기로 주성분 분석을 맨바닥에서 짜서, 흩어짐이 가장 큰 방향을 찾아 상관 있는 2차원 자료 묶음을 1차원으로 줄이는 법을 보인다. 손으로 하는 절차를 알아 두면 꾸러미나 깊은 배움 방식으로 가기 앞서 필요한 선형 대수 직관이 선다.

## 1. 코드

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

## 2. 논의

특잇값 쪼개기 $X_c = U \Sigma V^\top$이 주성분 분석의 일꾼이다. 오른쪽 특이 벡터($V$의 세로줄)가 주방향이고 특잇값 $\sigma_i$은 $\text{var}_i = \sigma_i^2 / (n-1)$으로 방향마다 자료가 퍼진 정도를 담는다. 이 보기의 공분산 행렬은 대각 밖 성분이 크므로($\sqrt{3.0 \times 2.0} \approx 2.45$이 최대인데 2.2) 흩어짐이 한 방향에 몰려 주성분1이 전체 흩어짐의 95%를 넘게 잡는다.

다시 세우기 걸음 $\hat{x} = (x_c \cdot v_1) v_1 + \mu$은 가운데 맞춘 점마다 주성분1에 쏘고 본디 좌표계로 되돌린다. 나온 점은 정확히 주축 위에 놓이며 그림의 회색 쏘기 선이 본디 점에서 다시 세운 점까지의 직교 거리를 보여 준다. 이 거리가 주성분2을 버려 잃은 앎이다.

특잇값 쪼개기 앞에 자료의 가운데를 맞추는 것이 결정적이다. 가운데를 맞추지 않으면 주성분 분석이 자료 구름의 가운데가 아니라 원점을 지나는 방향을 찾아 뜻 없는 쪼개기가 나온다. 이 미리 다듬기가 워낙 중요해 꾸러미 짜기(예컨대 `sklearn.decomposition.PCA`)는 이를 저절로 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
가운데 맞춘 자료의 공분산 행렬을 $\frac{1}{n-1} X_c^\top X_c$으로 손수 셈하고 그 고윳값이 특잇값 쪼개기의 `explained_variance`와 맞는지 확인하라.

</div>

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

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
가운데 맞추기를 건너뛰고 $X$에 곧바로 특잇값 쪼개기를 하면 어떻게 되는가? 실험을 돌려 가운데를 맞췄을 때와 아닐 때의 첫 주방향을 견주어라.

</div>

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

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
주성분 둘을 다 남기도록(곧 2차원으로 쏘고 다시 세우도록) 부호를 넓혀라. 이때 다시 세우기 어긋남이 정확히 0임을 확인하고 까닭을 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    scores_2d = Xc @ V  # (n, 2)
    X_recon_2d = scores_2d @ V.T + mu
    error_2d = np.mean((X - X_recon_2d) ** 2)
    print(f"Reconstruction error (2 components): {error_2d:.2e}")
    ```
    $d$차원 공간에서 주성분 $d$개를 모두 남기면 본디 자료를 정확히 되찾으므로 어긋남이 (뜬소수점 정밀도까지) 0이다. 성분을 모두 남기면 주성분 분석은 잃음 있는 눌러 담기가 아니라 직교 바탕 바꿈일 뿐이다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
이 코드가 공분산 행렬을 만들지 않고 특잇값 쪼개기를 쓰는 까닭은 무엇인가?

</div>

??? success "연습문제 4 풀이"
    수치적으로 더 정확하고, 차원이 클 때 값싸기 때문이다.

    공분산 행렬은 $X_c^\top X_c/(n-1)$이므로 만드는 순간 **조건수가 제곱된다.** 그러면
    작은 고윳값의 정확도가 크게 떨어진다. 특잇값 쪼개기는 $X_c$에 바로 작용하므로 그
    손실이 없다([유도 연습문제 3](pca_derivation.md)).

    2차원 장난감 자료에서는 어느 쪽이든 상관없다. 그런데 **습관을 들이는 것**이 이
    페이지의 목적이므로 처음부터 옳은 방식으로 적어 두는 것이 낫다.

    두 방식이 같은 답을 준다는 것은 알아 두어야 한다. $X_c = USV^\top$이면 $V$의 열이
    성분이고 고윳값은 $s_j^2/(n-1)$이다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
`np.linalg.svd`의 세 반환값이 각각 무엇인가? 성분은 어디에 있는가?

</div>

??? success "연습문제 5 풀이"
    `U, S, Vt = np.linalg.svd(Xc, full_matrices=False)`일 때 이렇다.

    | 반환값 | 모양 | 무엇인가 |
    |---|---|---|
    | `U` | $(n, r)$ | 왼쪽 특이 벡터 |
    | `S` | $(r,)$ | 특잇값. 큰 것부터 정렬되어 있다 |
    | `Vt` | $(r, d)$ | 오른쪽 특이 벡터를 **행**으로 담은 것 |

    **성분은 `Vt`의 행**이다. 곧 `Vt[0]`이 제1성분이다.

    이 자리에서 실수가 잦다. 이름이 `Vt`(전치된 $V$)인 까닭이 바로 그것이고, 수학에서
    $V$의 **열**이 성분이라 적은 것과 축이 어긋난다. `Vt.T[:, 0]`과 `Vt[0]`이 같은
    것임을 확인해 두면 헷갈리지 않는다.

    점수는 `Xc @ Vt[:k].T`로 얻는다. `U[:, :k] * S[:k]`와 같은 값인데, 새 자료에
    적용할 수 있는 것은 앞쪽이다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
`full_matrices=False`로 두는 까닭은 무엇인가?

</div>

??? success "연습문제 6 풀이"
    쓸데없이 큰 행렬을 만들지 않기 위해서다.

    `True`이면 `U`가 $(n, n)$이 된다. MNIST라면 $60{,}000 \times 60{,}000$이므로
    float64로 **29 GB**다. 메모리가 터진다.

    `False`이면 $r = \min(n, d)$까지만 만들어 `U`가 $(n, r)$이 된다. MNIST에서는
    $(60{,}000, 784)$로 376 MB다.

    버려지는 부분이 무엇인지 알아 두면 좋다. $(n-r)$개의 남는 왼쪽 특이 벡터는 $X_c$의
    열공간에 직교하는 방향들이며 특잇값이 모두 0이다. 곧 **아무 정보도 담지 않는다.**
    주성분 분석에 필요한 것이 하나도 버려지지 않는다.

    그래서 주성분 분석에서는 `False`가 언제나 맞는 선택이다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
2차원에서 1차원으로 줄일 때 잃는 것을 수치로 어떻게 보이겠는가?

</div>

??? success "연습문제 7 풀이"
    세 가지를 함께 보이면 좋다.

    **설명하는 흩어짐.** 제1성분이 잡은 비율이다. 상관이 강한 2차원 자료라면 90%를
    넘기도 한다.

    **다시 세우기 어긋남.** 1차원으로 사영하고 되돌린 뒤 원래와의 거리를 잰다. 버린
    고윳값이 곧 이 값이므로 따로 재지 않고도 안다.

    **그림.** 원래 점과 되돌린 점을 함께 찍고 둘을 선으로 이으면, 각 점이 주축에
    **수직으로** 떨어지는 모습이 보인다. 이것이 주성분 분석이 하는 일의 그림이다.

    셋째가 특히 값지다. 최소 제곱 회귀와 헷갈리기 쉬운데, 회귀는 **세로로** 떨어지는
    거리를 줄이고 주성분 분석은 **수직으로** 떨어지는 거리를 줄인다. 그림에서 이 차이가
    한눈에 보인다.

    같은 자료에 회귀선과 제1성분을 겹쳐 그려 보면 두 직선이 다르다는 것도 알 수 있다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
주성분 분석의 제1성분과 최소 제곱 회귀선은 같은가?

</div>

??? success "연습문제 8 풀이"
    다르다. 무엇을 줄이는지가 다르기 때문이다.

    | | 줄이는 거리 | 대칭인가 |
    |---|---|---|
    | 회귀 ($y$를 $x$로) | 세로 거리 $(y - \hat y)^2$ | 아니다 |
    | 주성분 분석 | 직선까지의 수직 거리 | 그렇다 |

    회귀는 $x$를 주어진 것으로 보고 $y$의 어긋남만 벌한다. 그래서 $x$와 $y$를 바꾸어
    회귀하면 **다른 직선**이 나온다. 주성분 분석은 두 변수를 대등하게 다루므로 축을
    바꾸어도 같은 직선이다.

    상관이 완벽하면 셋이 모두 같아지고, 상관이 약해질수록 두 회귀선이 벌어지며 제1성분은
    그 사이에 놓인다. 상관이 0이면 두 회귀선이 각각 수평과 수직이 된다.

    실용적인 뜻이 있다. **변수 사이에 인과나 방향이 있다면 회귀**가 맞고, 대등한 두
    재기의 공통 방향을 찾는 것이면 주성분 분석이 맞다. 같은 자료에서 두 가지를 물을 수
    있으므로 무엇을 묻는지가 정한다.

    단위에 대한 성질도 다르다. 회귀 계수는 단위를 바꾸면 예측이 그대로이도록 함께
    바뀌지만, 주성분 분석의 방향은 단위를 바꾸면 **실제로 달라진다**. 그래서 단위가
    다른 변수에는 표준화가 필요하다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
성분을 구한 뒤 점수가 정말 상관이 없는지 확인하려면 어떻게 하는가?

</div>

??? success "연습문제 9 풀이"
    점수의 공분산 행렬을 셈해 보면 된다. 대각선만 남고 나머지는 0이어야 한다.

    ```python
    Z = Xc @ Vt.T                    # 모든 성분에 대한 점수
    C = np.cov(Z, rowvar=False)
    print(np.abs(C - np.diag(np.diag(C))).max())   # 0에 가까워야 한다
    ```

    이것이 주성분 분석의 정의에서 바로 나온다. $\Sigma = V \Lambda V^\top$이므로 점수의
    공분산은

    $$\operatorname{Cov}(V^\top x) = V^\top \Sigma V = \Lambda$$

    로 대각 행렬이다. 그리고 대각 원소가 곧 고윳값이므로 `np.diag(C)`와 $s^2/(n-1)$이
    일치하는지도 함께 확인할 수 있다.

    **상관을 없앤다는 것이 독립으로 만든다는 뜻은 아니다.** 이 구별이 중요하다. 주성분
    분석은 2차 통계량만 다루므로 더 높은 차수의 의존은 그대로 남는다. 실제로 MNIST의
    점수들은 상관이 0이면서도 서로 얽혀 있다.

    독립까지 원하면 독립 성분 분석 쪽이며, 그쪽은 흩어짐이 아니라 비가우시안성을 좇는다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
이 코드를 새 자료에 적용하려면 무엇을 저장해 두어야 하는가?

</div>

??? success "연습문제 10 풀이"
    둘이다. **평균**과 **성분**이다.

    ```python
    np.savez('pca.npz', mean=mu, components=Vt[:k])
    # 새 자료에 적용
    Z_new = (X_new - mu) @ Vt[:k].T
    ```

    평균을 빠뜨리는 것이 잦은 실수다. 새 자료를 **학습 자료의 평균**으로 가운데 맞추어야
    하며, 새 자료 자신의 평균을 쓰면 안 된다. 그러면 두 자료가 다른 좌표계에 놓인다.

    표본 하나를 변환할 때 특히 위험하다. 표본 하나의 평균을 빼면 그 표본이 원점으로
    가 버려 정보가 사라진다.

    이것이 `sklearn`의 `fit`과 `transform`을 나누는 까닭이다. `fit`이 평균과 성분을
    배우고 `transform`이 그것을 쓴다. 시험 자료에 `fit_transform`을 쓰는 것이 흔한
    누출 사고이며, 직접 짤 때는 이 구별을 스스로 지켜야 한다.

## 정리하며

**다룬 것** — 2차원 주성분 분석 NumPy

특잇값 쪼개기 $X_c = U \Sigma V^\top$이 주성분 분석의 일꾼이다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
