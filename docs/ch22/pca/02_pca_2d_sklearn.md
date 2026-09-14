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

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
같은 자료 묶음에 `PCA(n_components=0.95)`을 써라. sklearn은 성분을 몇 개 고르며 저마다 설명하는 흩어짐 비는 얼마인가?

</div>

??? success "연습문제 1 풀이"
    ```python
    pca_auto = PCA(n_components=0.95)
    pca_auto.fit(X)
    print(f"Components selected: {pca_auto.n_components_}")
    print(f"Variance ratios: {pca_auto.explained_variance_ratio_}")
    ```
    상관이 큰 이 2차원 자료 묶음에서는 주성분1이 이미 흩어짐의 95%를 넘게 설명하므로 sklearn이 성분 하나를 고른다. 그 흩어짐 비는 대략 0.96이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
sklearn 주성분 분석의 다시 세우기 평균 제곱 어긋남을 이론 하한(주성분2 방향의 흩어짐)과 견주어라. 둘은 같은가? 까닭을 설명하라.

</div>

??? success "연습문제 2 풀이"
    ```python
    pca_full = PCA(n_components=2).fit(X)
    var_pc2 = pca_full.explained_variance_[1]
    print(f"Reconstruction MSE: {reconstruction_error:.6f}")
    print(f"PC2 variance:       {var_pc2:.6f}")
    ```
    주성분2을 버렸을 때의 다시 세우기 평균 제곱 어긋남은 (치우침 없는 흩어짐 어림개에서 오는 $n/(n-1)$ 인수까지) 주성분2 방향의 흩어짐과 같다. 이는 주성분 분석의 바탕 성질이다. 곧 가장 좋은 $k$차원 쏘기가 다시 세우기 어긋남을 가장 작게 하고, 남는 어긋남은 버린 고윳값의 합과 같다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`PCA(whiten=True)`을 써서 바꾼 점수의 공분산 행렬을 살펴라. 그것이 항등 행렬임을 확인하고 하얗게 하기가 언제 이로운지 설명하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    pca_white = PCA(n_components=2, whiten=True)
    scores_white = pca_white.fit_transform(X)
    cov_white = np.cov(scores_white.T)
    print("Covariance of whitened scores:")
    print(np.round(cov_white, 4))
    ```
    공분산 행렬이 (거의) 항등이며, 이는 하얗게 하기가 특징의 상관을 없애고 흩어짐을 1로 고른다는 것을 확인해 준다. 뒤따르는 알고리즘(예컨대 독립 성분 분석, k-평균, 신경망)이 들임 분포가 방향에 무관하다고 여기거나 특징 잣수에 민감할 때 하얗게 하기가 이롭다.


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
`fit`, `transform`, `fit_transform`을 언제 각각 쓰는가?

</div>

??? success "연습문제 4 풀이"
    | 메서드 | 하는 일 | 쓰는 자료 |
    |---|---|---|
    | `fit` | 평균과 성분을 배운다 | 학습 자료만 |
    | `transform` | 배운 것으로 옮긴다 | 아무 자료나 |
    | `fit_transform` | 둘을 한 번에 | 학습 자료만 |

    규칙은 하나다. **`fit`이 들어간 것은 학습 자료에만 쓴다.**

    ```python
    pca = PCA(n_components=1)
    Z_train = pca.fit_transform(X_train)   # 배우고 옮긴다
    Z_test  = pca.transform(X_test)        # 배운 것을 쓴다
    ```

    시험 자료에 `fit_transform`을 쓰면 시험 자료의 평균과 성분을 쓰게 되어 **정보가
    샌다.** 그리고 두 자료가 다른 좌표계에 놓이므로 견줄 수 없게 된다.

    직접 짤 때 평균을 저장해 두어야 했던 것과 같은 이야기다
    ([NumPy 연습문제 7](01_pca_2d_numpy.md)). `sklearn`은 그것을 `fit`이 강제해 준다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
`explained_variance_`와 `explained_variance_ratio_`가 어떻게 다른가?

</div>

??? success "연습문제 5 풀이"
    앞은 고윳값 그 자체이고 뒤는 전체에 대한 비율이다.

    ```python
    pca.explained_variance_        # 고윳값. 단위가 자료의 단위^2
    pca.explained_variance_ratio_  # 합이 1이 되도록 나눈 것
    ```

    관계는 이렇다.

    $$\text{ratio}_j = \frac{\lambda_j}{\sum_{i} \lambda_i}$$

    한 가지 조심할 점이 있다. `n_components`를 작게 두면 분모가 **남은 성분까지 포함한
    전체**다. 곧 `n_components=2`로 두어도 `explained_variance_ratio_`의 합이 1이 아니라
    0.168 같은 값이 된다(MNIST의 경우).

    그러므로 `ratio_.sum()`이 1이 아니라고 놀랄 일이 아니다. 그 값이 곧 **남긴 흩어짐의
    비율**이며 우리가 보고 싶어 하던 수다.

    누적 곡선을 그리려면 `n_components`를 비워 두고 모두 구한 뒤 `np.cumsum`을 쓴다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
`inverse_transform`은 무엇을 되돌리는가? 완전히 되돌아오는가?

</div>

??? success "연습문제 6 풀이"
    점수를 원래 공간으로 되돌린다.

    $$\hat x = \bar x + V_k z$$

    곧 성분을 곱하고 평균을 더한다. `transform`의 역이지만 **정보를 되돌리지는
    못한다.** 버린 $d - k$개 방향의 성분은 사라진 것이므로 복구할 수 없다.

    돌아오는 것은 원래 점이 아니라 **주부분 공간에 수직으로 사영한 점**이다. 그래서

    ```python
    X2 = pca.inverse_transform(pca.transform(X))
    ((X - X2)**2).mean()          # 0이 아니다. 버린 고윳값의 합/d
    ```

    이 값이 버린 고윳값으로 예측된다는 것을 [기본 연습문제 3](pca_fundamentals.md)에서
    확인했다.

    $k = d$이면 아무것도 버리지 않으므로 정확히 되돌아온다. 수치 오차만 남는다. 그것을
    확인해 보는 것이 짜기가 맞는지 보는 좋은 검사다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
직접 짠 것과 `sklearn`의 결과가 부호만 다르게 나왔다. 무엇을 확인하겠는가?

</div>

??? success "연습문제 7 풀이"
    **정상이다.** 고유 벡터의 부호는 정해지지 않는다
    ([유도 연습문제 4](pca_derivation.md)).

    확인할 것은 부호에 딸리지 않는 값들이 일치하는지다.

    | 확인할 것 | 같아야 하는가 |
    |---|---|
    | `explained_variance_` | 그렇다 |
    | 다시 세우기 어긋남 | 그렇다 |
    | 점수의 절댓값 | 그렇다 |
    | 점수의 부호 | 아니다 |

    ```python
    assert np.allclose(np.abs(Z_mine), np.abs(Z_sklearn))
    assert np.allclose(ev_mine, pca.explained_variance_)
    ```

    `sklearn`은 부호를 고정하는 관례를 쓴다. 각 성분에서 절댓값이 가장 큰 원소가 양수가
    되게 맞추므로(`svd_flip`), 같은 자료에 대해서는 재현된다. 직접 짠 코드에 같은 관례를
    넣으면 부호까지 맞출 수 있다.

    부호가 아니라 **성분의 순서나 크기**가 다르면 그때는 진짜 버그다. 가운데 맞추기를
    빠뜨렸거나 축을 잘못 잡은 것이다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
`svd_solver` 옵션은 무엇을 고르는 것인가? 언제 신경 써야 하는가?

</div>

??? success "연습문제 8 풀이"
    쪼개기를 어떤 알고리즘으로 할지 고른다. 기본값 `'auto'`가 자료 크기를 보고 정한다.

    | 값 | 쓰는 때 |
    |---|---|
    | `'full'` | 전체 특잇값 쪼개기. 작은 자료 |
    | `'randomized'` | $k$가 $d$보다 훨씬 작은 큰 자료 |
    | `'arpack'` | 성분 몇 개만, 희소 자료 |

    신경 쓸 때는 **자료가 커서 느리거나 메모리가 모자랄 때**다. MNIST에서 성분 50개만
    필요하다면 `'randomized'`가 훨씬 빠르다. 784개를 모두 구할 이유가 없다.

    무작위 방식은 **어림**이라는 점을 알아 두어야 한다. 근사이므로 `random_state`에 따라
    결과가 조금 달라지고, 앞쪽 성분은 매우 정확하지만 뒤쪽은 덜하다. 누적 흩어짐 곡선을
    끝까지 그리려면 `'full'`이 맞다.

    2차원 장난감 자료에서는 무엇을 고르든 차이가 없다. 이런 옵션이 있다는 것만 알아
    두면 큰 자료로 옮길 때 막히지 않는다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
`n_components`에 소수를 넣을 수 있다. 무슨 뜻인가?

</div>

??? success "연습문제 9 풀이"
    남길 흩어짐의 **비율**을 뜻한다.

    ```python
    PCA(n_components=0.95)    # 흩어짐 95%를 남기는 가장 작은 k
    ```

    `fit` 뒤에 `pca.n_components_`를 보면 실제로 고른 개수가 나온다. MNIST라면 154가
    된다([기본 연습문제 1](pca_fundamentals.md)).

    편리하지만 조심할 점이 있다. **$k$가 자료에 따라 달라진다.** 학습 자료를 바꾸거나
    전처리를 바꾸면 고른 $k$가 달라지므로, 뒤따르는 모델의 입력 차원이 바뀐다. 저장하고
    되불러 올 때 차원이 안 맞는 사고가 이렇게 난다.

    그래서 실험 단계에서는 소수로 두어 적당한 $k$를 찾고, 정해지면 **정수로 못 박아 두는**
    편이 안전하다. 무엇을 골랐는지 기록에 남기기도 좋다.

    정수 대신 `'mle'`를 넣는 선택도 있는데, 차원을 추정하는 방법을 쓴다. 가정이 들어가는
    방법이므로 결과를 그대로 믿기보다 참고로 보는 것이 좋다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
`sklearn`의 주성분 분석을 물길(pipeline) 안에서 쓸 때 무엇을 조심해야 하는가?

</div>

??? success "연습문제 10 풀이"
    가장 큰 것은 **교차 검증 안에서 `fit`이 되게 하는 것**이다.

    흔한 잘못은 이렇다.

    ```python
    Z = PCA(n_components=50).fit_transform(X)      # 전체 자료로 fit
    scores = cross_val_score(clf, Z, y, cv=5)      # 그 뒤에 교차 검증
    ```

    이러면 각 접기의 검증 몫이 주성분 분석을 맞추는 데 이미 쓰였으므로 **정보가 샌다.**
    점수가 낙관적으로 나온다.

    옳은 방식은 물길로 묶는 것이다.

    ```python
    pipe = make_pipeline(StandardScaler(), PCA(n_components=50), LogisticRegression())
    scores = cross_val_score(pipe, X, y, cv=5)     # 접기마다 fit 이 다시 된다
    ```

    이러면 접기마다 훈련 몫으로만 평균과 성분을 배운다.

    새는 정도는 자료 수에 달린다. 표본이 많으면 작지만, 표본이 적고 차원이 높으면
    크게 낙관적이 된다. 주성분 분석은 표지를 보지 않으므로 새는 정도가 지도 학습
    전처리보다 덜하다고 여겨지기도 하는데, **그래도 새는 것은 새는 것이다.**

    순서도 중요하다. 표준화가 주성분 분석보다 **앞**에 와야 한다. 뒤에 두면 이미 상관을
    없앤 점수를 다시 눈금 맞추는 셈이라 뜻이 달라진다.

## 정리하며

**다룬 것** — 2차원 주성분 분석 Sklearn

sklearn의 PCA 갈래가 주성분 분석 흐름 전체를 감싼다.

앞의 연습문제 3개로 스스로 따져 볼 수 있다.
