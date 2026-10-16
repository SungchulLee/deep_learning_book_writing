# NumPy로 구현하는 2차원 PCA

이 예제는 상관된 2차원 데이터를 1차원으로 투영하고 원래 좌표로 재구성한다. 평균을 빼고 SVD로 방향을 구한 뒤, 투영 점수와 재구성 오차의 관계를 확인한다. 코드는 분해와 시각화 예제이며 예측 모델의 테스트 성능을 측정하지는 않는다.

## 1. 예제 코드

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

## 2. 데이터 생성과 SVD

표본 수는 150이고 생성 공분산은 [[3,2.2],[2.2,2]]다. 이 행렬의 고윳값은 (5±√20.36)/2로 약 4.756과 0.244이며 첫 방향의 모집단 분산 비율은 약 95.1%다. 유한 표본에서 계산한 비율은 달라질 수 있으므로 생성 모수만으로 표본의 95% 초과를 보장하지 않는다.

`mu`는 학습 평균, `Xc`는 150×2 중심화 데이터다. 축소 SVD `Xc=U diag(S) Vt`에서 `U`는 150×2, `S`는 길이 2, `Vt`는 2×2다. 성분은 `Vt`의 행 또는 `V=Vt.T`의 열에 있다. `explained_variance=S**2/(n-1)`는 표본 공분산 고윳값이다.

## 3. 투영과 재구성

첫 방향 `pc1`의 점수는 `Xc@pc1`이고 재구성은 점수와 방향의 외적에 평균을 더한 값이다.

$$\hat x_i=\mu+v_1v_1^T(x_i-\mu).$$

재구성점은 평균을 지나는 주축 위에 놓이고 잔차는 그 축에 직교한다. 이 직교 거리는 좌표축에 따른 세로 오차를 최소화하는 y 회귀와 다르다. 회귀를 사용한다고 인과관계가 성립하는 것은 아니다.

## 4. 오차와 전처리 해석

코드의 `reconstruction_error`는 모든 150×2 원소의 제곱 오차 평균이다. 버린 표본 공분산 고윳값을 λ₂라 하면 다음과 같다.

$$\operatorname{MSE}=\frac{n-1}{2n}\lambda_2.$$

표본별 제곱 거리와 원소별 MSE, 공분산 분모를 모두 구분해야 한다. 두 성분을 모두 쓰면 부동소수점 오차 범위에서 원본을 재구성한다. 중심화를 생략한 SVD는 평균 항을 포함한 2차 모멘트를 분해하는 다른 문제다. 원점을 지나는 저랭크 근사를 의도했다면 의미가 있지만 평균 주위 PCA와 혼동하지 않는다.

## 5. 새 데이터와 수치 비교

새 표본은 학습 평균으로 중심화하고 저장한 방향으로 투영한다. k개 성분을 쓰면 평균과 k×d 성분 배열을 함께 저장한다. 피처 표준화도 했다면 그 스케일도 필요하다.

SVD는 공분산을 명시적으로 만들지 않아 조건수 제곱의 영향을 줄일 수 있다. 계산 비용은 n, d와 분해 방식에 따라 달라지므로 언제나 다른 방법보다 빠른 것은 아니다. 개별 방향의 부호와 중복 고윳값의 기저 자유를 고려해 결과를 비교한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
표본 공분산의 고윳값과 SVD 특잇값의 관계를 유도하라. 아래 비교 코드의 의미를 설명하라.

</div>

??? success "연습문제 1 풀이"

    Xc=USVᵀ이면 XcᵀXc/(n−1)=V diag(s_j²/(n−1))Vᵀ다. 따라서 같은 데이터의 정렬된 고윳값은 explained_variance와 수치 오차 범위에서 일치한다. eigh는 오름차순 반환이므로 정렬 방향을 맞추어야 한다.

    ```python
    cov_manual = (Xc.T @ Xc) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov_manual)
    eigvals_sorted = np.sort(eigvals)[::-1]
    print("From SVD:", explained_variance)
    print("From eigh:", eigvals_sorted)
    print("Match:", np.allclose(explained_variance, eigvals_sorted))
    ```


---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
중심화를 생략한 SVD와 중심화 PCA의 차이를 설명하라. 각도 차이를 미리 특정 범위로 단정할 수 있는가?

</div>

??? success "연습문제 2 풀이"

    XᵀX/n은 중심화 공분산에 μμᵀ를 더한 행렬이다. 평균이 방향 선택에 영향을 주지만 첫 방향이 언제나 평균과 일치하거나 각도가 10~30도라는 보장은 없다. 아래 코드는 두 축의 부호를 무시하는 절댓값 내적으로 예각을 계산한다.

    ```python
    U_nc, S_nc, Vt_nc = np.linalg.svd(X, full_matrices=False)
    pc1_nc = Vt_nc[0]
    print(f"PC1 (centered):   {pc1}")
    print(f"PC1 (uncentered): {pc1_nc}")
    angle = np.arccos(np.clip(np.abs(pc1 @ pc1_nc), 0, 1)) * 180 / np.pi
    print(f"Angle difference: {angle:.1f} degrees")
    ```


---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
두 주성분을 모두 남기면 재구성 오차가 0이 되는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"

    2×2 직교 V에 대해 VVᵀ=I이므로 XcVVᵀ+μ=X다. 이는 기저 변환일 뿐 좌표 수를 줄이지 않는다. 실제 계산에는 부동소수점 오차가 남을 수 있다.

    ```python
    scores_2d = Xc @ V  # (n, 2)
    X_recon_2d = scores_2d @ V.T + mu
    error_2d = np.mean((X - X_recon_2d) ** 2)
    print(f"Reconstruction error (2 components): {error_2d:.2e}")
    ```


---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
직접 SVD를 사용할 때 얻는 수치상 이점과 비용상 조건을 설명하라.

</div>

??? success "연습문제 4 풀이"

    공분산 XcᵀXc를 만들면 0이 아닌 특잇값에 대한 조건수가 제곱되어 작은 성분 추정에 불리할 수 있다. SVD는 Xc에 직접 작용한다. 다만 표본이 매우 많고 피처가 적으면 공분산 고유분해가 실용적일 수도 있다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff easy" title="쉬움"></span>
축소 SVD의 반환 모양과 주방향·점수의 위치를 설명하라.

</div>

??? success "연습문제 5 풀이"

    r=min(n,d)일 때 U는 n×r, S는 길이 r, Vt는 r×d다. 주방향은 Vt의 행이고 학습 점수는 Xc@Vt[:k].T=U[:,:k]*S[:k]다. r은 행렬의 실제 랭크와 반드시 같지는 않으며 일부 특잇값은 0일 수 있다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
full_matrices=False가 줄이는 저장 공간을 설명하라. n=60000,d=784인 float64 U를 비교하라.

</div>

??? success "연습문제 6 풀이"

    전체 U는 60000×60000으로 28.8 GB이고 축소 U는 60000×784로 약 376.32 MB다. PCA에는 보통 축소 형태면 충분하다. n<d일 때 축소 Vt는 입력 행공간 밖의 영공간 기저를 생략한다. 별도로 완전한 영공간 기저가 필요한 문제라면 전체 분해가 유용할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
첫 성분만 남길 때 코드의 원소별 MSE와 버린 고윳값의 관계를 적어라.

</div>

??? success "연습문제 7 풀이"

    고윳값이 n−1 분모이므로 표본별 제곱 거리 평균은 (n−1)λ₂/n이다. 원소별 평균은 피처 수 2로 더 나누어 (n−1)λ₂/(2n)이 된다. 분산 비율과 재구성선을 함께 보면 기하와 오차 크기를 연결할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff hard" title="어려움"></span>
PCA 직선과 y를 x로 회귀한 직선의 목적을 비교하라.

</div>

??? success "연습문제 8 풀이"

    PCA는 같은 척도로 표현한 두 좌표의 직교 제곱 거리를 최소화한다. y 회귀는 x를 입력으로 두고 세로 y 잔차를 최소화한다. 특정 데이터에서는 직선이 같을 수도 있으나 일반적으로 목적이 다르다. 변수 단위 선택과 예측 목적을 고려하며 회귀선으로 인과성을 단정하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
점수의 무상관을 아래 공분산 코드로 확인하는 원리를 설명하라.

</div>

??? success "연습문제 9 풀이"

    같은 학습 데이터의 점수 Z=XcV에 대해 ZᵀZ/(n−1)=diag(s_j²/(n−1))다. 대각 밖 값이 수치 오차 범위에서 0인지 비교한다. 상관 없음은 독립을 뜻하지 않으며 새로운 데이터에서도 공분산이 대각이라는 보장은 없다.

    ```python
    Z = Xc @ Vt.T                    # 모든 성분에 대한 점수
    C = np.cov(Z, rowvar=False)
    print(np.abs(C - np.diag(np.diag(C))).max())   # 0에 가까워야 한다
    ```


---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff easy" title="쉬움"></span>
새 데이터 변환에 저장해야 할 값을 설명하라. 아래 코드에 필요한 추가 입력은 무엇인가?

</div>

??? success "연습문제 10 풀이"

    학습 평균 mu와 선택한 방향 Vt[:k]를 저장한다. k와 X_new는 아래 코드 전에 정의해야 하며 새 입력은 학습 때와 같은 피처 순서·단위를 사용한다. 전처리 스케일이 있으면 그것도 저장한다. 테스트에 다시 fit하면 같은 변환이 아니고 평가 데이터의 정보가 전처리에 들어간다.

    ```python
    np.savez('pca.npz', mean=mu, components=Vt[:k])
    # 새 자료에 적용
    Z_new = (X_new - mu) @ Vt[:k].T
    ```


## 정리하며

NumPy PCA는 중심화, SVD, 투영과 평균 복원으로 구성된다. 성분의 행·열 관례와 n−1 분모, 원소별 MSE를 맞추면 고윳값과 재구성 오차를 일관되게 비교할 수 있다. 새 데이터에는 학습 평균과 방향을 그대로 사용한다.
