# 커널 PCA

커널 PCA는 특징 사상 φ(x)가 만드는 공간에서 PCA를 수행한다. 내적 κ(x,x′)=⟨φ(x),φ(x′)⟩만 계산하므로 특징 벡터를 명시적으로 만들지 않아도 된다. 원래 좌표에 대해서는 비선형 변환이지만, 선택한 커널의 특징 공간에서는 선형 분산 최대화다. 커널을 쓰더라도 표본 수에 따른 계산·메모리 비용은 남는다.

## 1. 커널과 특징 공간

유효한 실수 내적 커널은 대칭이며 모든 유한 표본의 그람 행렬 K가 양의 준정부호여야 한다. 이 조건은 힐베르트 공간의 특징 사상을 통해 내적으로 표현할 수 있음을 보장한다. 임의의 유사도 함수가 모두 내적 커널인 것은 아니다.

| 커널 | 식과 조건 |
|---|---|
| 선형 | κ(x,x′)=xᵀx′ |
| 다항 | κ(x,x′)=(xᵀx′+c)ᵖ, c≥0과 양의 정수 p |
| RBF | κ(x,x′)=exp(−||x−x′||²/(2σ²)), σ>0 |
| 시그모이드 | tanh(a xᵀx′+c)는 일반적인 설정에서 양의 준정부호를 보장하지 않음 |

RBF의 표현력이 크다고 상위 몇 개 성분이 임의의 비선형 구조를 자동으로 복원하는 것은 아니다. 커널·스케일·성분 수에 따라 남기는 구조가 달라지며, 특징 공간의 투영 결과가 원래 입력 공간의 어떤 점에 정확히 대응할 필요도 없다.

## 2. 특징 공간의 중심화

학습 특징 평균을 μ_φ=(1/n)Σ_iφ(x_i)라 하면 중심화 특징은 φ_c(x_i)=φ(x_i)−μ_φ다. H=I−11ᵀ/n으로 두면 중심화 그람 행렬은 다음과 같다.

$$K_c=HKH.$$

즉 행 평균과 열 평균을 빼고 전체 평균을 더한다. 원래 입력의 평균을 빼는 것만으로 비선형 특징 공간의 평균이 0이 되지는 않는다.

새 입력 x_*에 대한 중심화 커널 벡터의 i번째 원소는 다음과 같다.

$$k_{c,i}(x_*)=\kappa(x_i,x_*)-\frac1n\sum_j\kappa(x_j,x_*)
-\frac1n\sum_j\kappa(x_i,x_j)+\frac1{n^2}\sum_{j,l}\kappa(x_j,x_l).$$

새 데이터 묶음의 특징 평균을 다시 적합하는 대신 학습 특징 평균과의 내적을 계산하는 식이다.

## 3. 고윳값과 정규화

중심화 특징을 행에 둔 Φ_c에 대해 C_φ=Φ_cᵀΦ_c/n이다. 양의 고윳값 λ_j의 단위 주방향은 학습 특징의 선형 결합으로 쓸 수 있다. K_c의 단위 고유벡터 u_j와 양의 고윳값 ρ_j를 구하면 다음과 같다.

$$K_cu_j=\rho_ju_j,\qquad \lambda_j=\rho_j/n,\qquad
v_j=\Phi_c^Tu_j/\sqrt{\rho_j}.$$

이때 ||v_j||²=u_jᵀK_cu_j/ρ_j=1이다. α_j=u_j/√ρ_j로 두면 새 입력의 점수는 다음과 같다.

$$z_j(x_*)=\alpha_j^Tk_c(x_*).$$

학습 점수 행렬은 U_k diag(√ρ₁,…,√ρ_k)다. 중심화 때문에 K_c는 적어도 한 개의 0 고윳값을 가지므로 가역일 필요가 없다. 양의 고유공간에서 식을 구성하면 충분하다. 0이나 음수 고윳값을 단순히 작은 양수로 바꾸어 정규화하는 것은 정확한 고유공간 처리를 대신하지 못한다.

## 4. 직접 구현 읽기

```python
import numpy as np
from scipy.spatial.distance import cdist

class KernelPCA:
    """비선형 차원 줄이기를 위한 알맹이 주성분 분석.

    고른 알맹이 함수가 정하는, 넌지시 딸린 차원 높은 특징 공간에서
    주성분 분석을 한다.

    인수:
        n_components: 주성분의 수
        kernel: 'rbf', 'poly', 'linear' 가운데 하나
        gamma: 방사 바탕 함수 알맹이의 띠너비
        degree: 다항 알맹이의 차수
        coef0: 다항 알맹이의 치우침
    """

    def __init__(self, n_components, kernel='rbf', gamma=1.0,
                 degree=3, coef0=1.0):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def _compute_kernel(self, X, Y=None):
        """알맹이 행렬 K[i,j] = kappa(X[i], Y[j])을 셈한다."""
        if Y is None:
            Y = X
        if self.kernel == 'rbf':
            dists = cdist(X, Y, metric='sqeuclidean')
            return np.exp(-dists / (2 * self.gamma ** 2))
        elif self.kernel == 'poly':
            return (X @ Y.T + self.coef0) ** self.degree
        elif self.kernel == 'linear':
            return X @ Y.T
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _center_kernel(self, K):
        """알맹이 행렬의 가운데를 맞춘다: K_centered = H K H."""
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        return K - one_n @ K - K @ one_n + one_n @ K @ one_n

    def fit(self, X):
        """익히기 자료에 알맹이 주성분 분석을 맞춘다.

        알맹이 행렬을 셈해 가운데를 맞추고
        으뜸 고유벡터(쌍대 계수 alpha)를 찾는다.
        """
        self.X_train = X.copy()
        n = X.shape[0]

        # 알맹이 행렬을 셈하고 가운데를 맞춘다
        K = self._compute_kernel(X)
        K_centered = self._center_kernel(K)

        # 가운데 맞춘 알맹이 행렬의 고윳값 쪼개기
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

        # 내림차순으로 정렬해 으뜸 성분을 고른다
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx[:self.n_components]]
        eigenvectors = eigenvectors[:, idx[:self.n_components]]

        # 고르게 한다: alpha_j / sqrt(n * lambda_j)
        self.alphas = eigenvectors / np.sqrt(
            np.maximum(eigenvalues, 1e-10)
        )
        self.eigenvalues = eigenvalues / n

        # 시험 때 가운데를 맞추려 통계를 곳간에 담는다
        self.K_train = K
        self.K_train_col_mean = K.mean(axis=0)
        self.K_train_mean = K.mean()

        return self

    def transform(self, X):
        """새 자료를 알맹이 주성분에 쏜다.

        시험 점과 익히기 점 사이 알맹이를 셈해
        알맞게 가운데를 맞추고 쏜다.
        """
        K_test = self._compute_kernel(X, self.X_train)

        # 시험 알맹이 행렬의 가운데를 맞춘다
        K_test_centered = (
            K_test
            - K_test.mean(axis=1, keepdims=True)
            - self.K_train_col_mean[np.newaxis, :]
            + self.K_train_mean
        )

        return K_test_centered @ self.alphas

    def fit_transform(self, X):
        self.fit(X)
        K_centered = self._center_kernel(self.K_train)
        return K_centered @ self.alphas
```

이 `KernelPCA` 클래스의 `gamma`는 RBF의 **폭 σ**다. 양수여야 하며 `_center_kernel`은 행렬 H를 직접 곱하는 방식이라 평균으로 중심화하는 구현보다 불필요한 계산이 많다. `transform`은 학습 커널 평균을 저장해 새 입력을 올바른 학습 기준으로 중심화한다.

고윳값은 `eigh` 결과를 내림차순으로 고르지만, `maximum(eigenvalues,1e-10)`은 비양수 성분을 제거하지 않는다. 유효한 양의 고윳값 수와 목표 차원, 수치 허용 오차를 별도로 확인해야 한다. `eigenvalues` 속성은 그람 고윳값을 n으로 나눈 특징 공분산 값이다. 입력 복사와 전체 커널 행렬도 저장하므로 메모리 비용이 든다.

## 5. 동심원 예제와 두 gamma 관례

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA, KernelPCA

# 겹동그라미를 만든다
X, y = make_circles(n_samples=500, factor=0.3, noise=0.05,
                     random_state=42)

# 선형 주성분 분석(동그라미를 가르지 못한다)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 방사 바탕 함수 알맹이 주성분 분석(동그라미를 가른다)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', s=20)
axes[0].set_title('Original Data')

axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', s=20)
axes[1].set_title('Linear PCA')

axes[2].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='coolwarm', s=20)
axes[2].set_title('Kernel PCA (RBF)')

plt.tight_layout()
plt.show()
```

이 블록은 `sklearn.decomposition.KernelPCA`를 다시 import해 앞의 클래스 이름을 덮어쓴다. 또한 원래 2차원 데이터를 2차원으로 변환해 시각화하므로 1차원 분류 정확도를 측정하는 예제가 아니다. 색으로 쓰는 레이블은 PCA 적합에 사용하지 않는다. 데이터와 커널 설정에 따라 분리가 달라지며 완벽한 분류가 보장되지는 않는다.

scikit-learn의 RBF는 exp(−γ||x−x′||²)다. 따라서 앞의 폭 σ와는 γ=1/(2σ²) 관계다. **작은 σ는 좁은 커널**, **큰 scikit-learn γ도 좁은 커널**이다. 같은 `gamma` 인수 이름을 두 구현 사이에 그대로 옮기면 반대 효과를 낼 수 있다.

## 6. 커널 폭과 성분 수 선택

아주 넓은 RBF는 모든 점의 유사도를 비슷하게 만들고 중심화 후 작은 값의 커널 행렬을 만든다. 넓은 폭에서 적절히 전개한 첫 항은 선형 커널과 관련되지만, 단순히 작은 γ이면 모든 결과가 선형 PCA와 같다고 단정하지 않는다. 아주 좁으면 서로 다른 점의 유사도가 거의 0이 되어 유용한 전역 구조를 구분하기 어려울 수 있다.

```python
from scipy.spatial.distance import pdist

def median_heuristic(X):
    """방사 바탕 함수의 gamma을 둘씩 거리의 가운뎃값으로 둔다."""
    dists = pdist(X, metric='euclidean')
    return np.median(dists)
```

위 함수는 서로 다른 점 쌍의 거리 중앙값 m을 반환한다. 직접 구현의 폭을 σ≈m으로 정하는 출발점이라면 scikit-learn에서는 γ≈1/(2m²)로 환산한다. 중복 데이터로 m=0이면 이 규칙은 사용할 수 없다. 거리 계산 자체도 표본 수에 대해 제곱 비용이다.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# 좋은 쏘기의 대리 잣대로 가르기 정확도를 쓴다
pipe = Pipeline([
    ('kpca', KernelPCA(kernel='rbf')),
    ('clf', SVC())
])

param_grid = {
    'kpca__n_components': [2, 5, 10, 20],
    'kpca__gamma': [0.01, 0.1, 1.0, 10.0],
}

search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
search.fit(X_train, y_train)
print(f"Best params: {search.best_params_}")
```

이 격자 탐색은 재구성 오차가 아니라 **SVC 분류 정확도**를 평가한다. `X_train`, `y_train`을 먼저 정의해야 하며, 이 위치의 `KernelPCA`는 scikit-learn 클래스다. Pipeline 안에서 각 학습 폴드에 커널 PCA를 적합하므로 전체 데이터에 먼저 적합하는 누출을 피한다. SVC 자체도 비선형일 수 있어 분류 성능의 개선을 커널 PCA 하나의 효과로 보려면 기준 비교가 필요하다.

다른 커널 폭의 설명 분산은 서로 다른 특징 공간의 양이다. 비율만으로 원래 입력의 복원 품질을 직접 비교하지 않는다. 비지도 안정성·거리 보존 같은 목적에 맞는 지표도 가능하므로 비지도 방법에는 선택 기준이 전혀 없다는 뜻은 아니다.

## 7. 프리이미지와 역변환

저차원 좌표로부터 특징 공간에서 재구성한 점이 φ(x)의 상에 속한다는 보장은 없다. 대응하는 입력이 있더라도 여러 개일 수 있다. 이 프리이미지(pre-image) 문제를 풀려면 특징 공간 거리 최소화나 별도의 회귀 역변환을 사용할 수 있다. scikit-learn의 `inverse_transform`은 학습된 근사 역변환이지 정확한 함수 역이 아니다. 자세한 정의는 [KernelPCA 문서](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html)를 참고한다.

```python
def approximate_preimage(kpca, z, X_train, gamma, n_iter=100, lr=0.1):
    """기울기 내려가기로 앞그림을 어림한다(방사 바탕 함수 알맹이)."""
    # 익히기 점의 무게 붙은 평균으로 첫자리매김한다
    K_weights = np.exp(-0.5 * np.sum(z ** 2))
    x = X_train.mean(axis=0).copy()

    for _ in range(n_iter):
        k = np.exp(-np.sum((X_train - x) ** 2, axis=1)
                    / (2 * gamma ** 2))
        grad = np.sum(k[:, None] * (X_train - x), axis=0)
        x += lr * grad / (k.sum() + 1e-10)

    return x
```

위 `approximate_preimage`는 이름과 달리 목표 z에 대응하는 점을 찾지 않는다. z로 계산한 `K_weights`는 쓰이지 않고 `kpca`도 사용하지 않는다. 갱신은 커널 가중 평균 쪽으로 이동하는 과정이므로 같은 초기점·폭에서는 z를 바꾸어도 결과가 같다. 목표 특징 재구성과의 거리나 그 미분을 구현한 것으로 설명할 수 없다.

오토인코더는 명시적 디코더가 있어 좌표를 입력 모양으로 변환하기 쉽지만, 정확한 역이나 그럴듯한 샘플 생성을 보장하지 않는다. 일반 오토인코더의 잠재 좌표에 임의 분포를 놓는 것만으로 적절한 생성 모델이 완성되는 것도 아니다.

## 8. 계산 비용과 적용 범위

전체 커널 행렬 저장은 O(n²), 밀집 전체 고유분해는 대략 O(n³)다. 부분 고유분해와 니스트룀 근사, 무작위 특징을 사용하면 비용이 달라진다. float64 커널 하나만 해도 n=1,000이면 8 MB, 10,000이면 800 MB, 60,000이면 28.8 GB다. 분해 작업 공간과 복사본은 추가다. 특정 표본 수를 모든 장치의 절대 한계로 정하지 않는다.

선형 PCA는 n과 d, 분해 방법에 따라 비용이 달라지고, 신경망은 구조와 에포크 수에 따라 달라진다. t-SNE나 UMAP의 비용도 알고리즘과 근사 설정에 의존하므로 고정 지수 하나로 모든 방법의 규모 확장성을 비교하지 않는다.

## 9. 금융 데이터의 비선형 특징

변동성 표면이나 스프레드 곡선의 비선형 특징을 탐색할 때 커널 PCA를 후보로 사용할 수 있다. 아래 코드는 표본별로 평탄화된 표면 배열 `vol_surfaces`가 미리 주어졌다고 가정한다.

```python
# 출렁임 면 살피기
# X: [날 수, 행사가 수 * 만기 수]로 펼친 출렁임 면
# 선형 주성분 분석은 높이/기울기/기간 짜임 결을 잡는다
# 알맹이 주성분 분석은 스마일의 움직임과 국면 바뀜을 잡을 수 있다

from sklearn.decomposition import KernelPCA

# 방사 바탕 함수 알맹이가 비선형 국면 짜임을 잡는다
kpca = KernelPCA(n_components=5, kernel='rbf', gamma=0.1)
vol_factors = kpca.fit_transform(vol_surfaces)

# 요인 1: 전체 출렁임 높이(주성분 분석과 비슷)
# 요인 2~5: 스마일 일그러짐, 기간 짜임 비틀림,
#              국면 옮아감을 잡는 비선형 결
```

데이터 생성·전처리·검증과 요인의 경제적 의미는 이 코드에 포함되지 않는다. 시계열에서는 과거 학습 구간에서 커널과 통계를 정하고 이후 구간을 변환해야 한다. 좋은 시각화나 설명 분산만으로 위험 예측이나 투자 성과의 향상을 주장할 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
양의 그람 고윳값 ρ와 단위 고유벡터 u로 특징 공간 단위 주방향을 구성하라.

</div>

??? success "연습문제 1 풀이"

    중심화 특징 Φ_c에서 v=Φ_cᵀu/√ρ다. ||v||²=uᵀK_cu/ρ=1이고 특징 공분산 고윳값은 ρ/n이다. 따라서 새 점수는 uᵀk_c(x)/√ρ다. ρ=0인 방향에는 이 정규화를 적용하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
선형 커널을 중심화한 커널 PCA와 직접 SVD PCA의 점수 관계를 설명하라.

</div>

??? success "연습문제 2 풀이"

    중심화 선형 커널은 X_cX_cᵀ다. X_c=USVᵀ이면 그 고유벡터는 U, 양의 고윳값은 s_j²이며 커널 PCA 점수는 U_kS_k다. 이는 직접 PCA 점수와 부호·중복 고유공간의 기저 자유를 제외하고 같다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
RBF 커널을 사용하면 동심원이 항상 한 좌표에서 완벽하게 분류되는가?

</div>

??? success "연습문제 3 풀이"

    아니다. 표본 분포·잡음·폭과 선택된 성분에 따라 다르다. 비선형 특징 공간이 분류에 유용한 방향을 제공할 수 있지만 PCA는 레이블을 보지 않는다. 해당 설정의 별도 검증이 필요하고 같은 데이터의 시각적 분리만으로 일반화 정확도를 주장할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
커널 PCA의 성분 수와 폭을 후속 분류 성능으로 고를 때 누출을 피하는 방법을 설명하라.

</div>

??? success "연습문제 4 풀이"

    전처리, 커널 PCA와 분류기를 Pipeline으로 묶고 각 교차 검증 학습 폴드에서 적합한다. 검증 폴드는 그 학습 통계로만 변환한다. 최종 테스트 데이터는 설정 선택에 사용하지 않는다. 다른 폭의 설명 분산 비율을 같은 입력 공간의 정보 보존율로 비교하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
이 페이지의 동심원 코드가 실제로 측정하는 것과 측정하지 않는 것을 구분하라.

</div>

??? success "연습문제 5 풀이"

    500개 점을 원래 2차원, 선형 PCA 2차원과 커널 PCA 2차원으로 표시한다. 분류기를 학습하거나 교차 검증 정확도를 계산하지 않는다. 따라서 그 코드에서 1차원 정확도 100% 같은 수치가 얻어졌다고 설명할 수 없다.



---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff med" title="중간"></span>
직접 구현의 gamma=σ와 scikit-learn gamma=γ의 관계를 구하라. 폭 σ=2는 어떤 γ에 해당하는가?

</div>

??? success "연습문제 6 풀이"

    exp(−r²/(2σ²))=exp(−γr²)이므로 γ=1/(2σ²)다. σ=2이면 γ=1/8=0.125다. σ를 키우면 넓어지고 γ를 키우면 좁아지므로 두 인수의 방향을 혼동하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff hard" title="어려움"></span>
입력 X를 중심화했어도 비선형 커널을 다시 중심화해야 하는 이유를 설명하라.

</div>

??? success "연습문제 7 풀이"

    비선형 φ에서는 표본 평균 φ(x_i)의 평균이 일반적으로 φ(평균 x_i)와 같지 않다. 따라서 원래 입력의 평균 0이 특징 평균 0을 뜻하지 않는다. 학습 그람 행렬은 HKH로 중심화하고 새 입력은 저장한 학습 커널 평균을 사용한다.



---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
프리이미지 문제와 이 페이지의 approximate_preimage 함수가 그 문제를 해결하지 못하는 이유를 설명하라.

</div>

??? success "연습문제 8 풀이"

    특징 공간 재구성점에 정확히 대응하는 입력이 없거나 여러 개일 수 있다. 근사 프리이미지는 그 목표점에 가까운 입력을 찾는다. 이 함수는 kpca와 목표 z를 실제 갱신에 쓰지 않아 목표에 따른 복원을 할 수 없다. 반복 횟수를 늘리는 것만으로 이 누락이 해결되지는 않는다.



---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
표본 60,000개의 float64 전체 커널 행렬 저장 크기를 계산하라.

</div>

??? success "연습문제 9 풀이"

    60000²×8=28,800,000,000바이트, 즉 28.8 GB 또는 약 26.82 GiB다. 커널 하나의 크기이며 중심화 복사와 분해 작업 공간은 추가다. 근사나 대표 표본을 사용할 때는 비용 감소와 품질 영향을 함께 평가한다.



---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
커널 PCA와 오토인코더를 선택할 때 고려할 기준을 설명하라.

</div>

??? success "연습문제 10 풀이"

    커널 PCA는 선택한 커널의 고유분해를 사용하지만 전체 커널 저장 비용과 프리이미지 문제가 있다. 오토인코더는 미니배치와 명시적 디코더를 사용할 수 있으나 구조·학습과 일반화를 검증해야 한다. 데이터 크기, 재구성 필요성, 커널 지식과 계산 예산에 따라 선택하며 어느 쪽도 항상 우수하거나 유일한 해를 보장하지 않는다.



## 정리하며

커널 PCA는 중심화된 커널 행렬의 양의 고유공간에서 특징 공간 PCA를 수행한다. 새 데이터의 중심화와 고유벡터 정규화, RBF 폭의 두 관례를 구분해야 한다. 비선형 표현력과 완벽한 분류·역변환은 별개이며 전체 커널 비용과 목적에 맞는 검증이 필요하다.
