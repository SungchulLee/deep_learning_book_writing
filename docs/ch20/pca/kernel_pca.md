# 알맹이 주성분 분석
알맹이 재주로 하는 비선형 차원 줄이기.

---

## 개요

고전 주성분 분석은 선형 아래 공간만 찾을 수 있다. **알맹이 주성분 분석**은 자료를 차원 높은(어쩌면 무한 차원인) 특징 공간에 넌지시 옮기고 거기서 선형 주성분 분석을 해, 비선형 짜임까지 잡도록 넓힌 것이다. **알맹이 재주** 덕분에 셈이 감당할 만해진다. 곧 특징 공간의 좌표를 결코 대놓고 셈하지 않고 옮긴 점 사이의 안쪽 곱만 셈한다.

알맹이 주성분 분석은 선형 주성분 분석과 자기 부호기 같은 온전한 비선형 방법 사이를 이으며, 닫힌 꼴 풀이로 비선형 차원 줄이기를 준다.

---

## 왜 필요한가

### 선형이라는 한계

주성분 분석은 흩어짐을 가장 크게 하는 아래 공간을 찾지만 그 아래 공간은 **평평해야**(초평면이어야) 한다. 스위스 롤이나 겹동그라미처럼 굽은 다양체 위의 자료에서는 가장 좋은 평평한 쏘기도 요긴한 짜임을 잃는다.

### 특징 옮김이라는 생각

본디 공간 $\mathbb{R}^d$에서 일하는 대신 $\phi: \mathbb{R}^d \to \mathcal{F}$으로 자료를 차원 높은 특징 공간 $\mathcal{F}$에 옮긴 뒤 $\mathcal{F}$에서 선형 주성분 분석을 한다:

$$\mathbf{x} \in \mathbb{R}^d \xrightarrow{\phi} \phi(\mathbf{x}) \in \mathcal{F} \xrightarrow{\text{PCA}} \text{principal components in } \mathcal{F}$$

차원 높은 특징 공간의 선형 아래 공간은 본디 공간에서 **비선형** 다양체에 맞닿는다. 어려움은 $\mathcal{F}$이 아주 차원이 높거나 무한 차원일 수 있어 대놓고 셈할 수 없다는 것이다.

---

## 알맹이 재주

### 알맹이 함수

**알맹이 함수** $\kappa: \mathbb{R}^d \times \mathbb{R}^d \to \mathbb{R}$은 특징 옮김을 대놓고 셈하지 않고 특징 공간의 안쪽 곱을 셈한다:

$$\kappa(\mathbf{x}, \mathbf{x}') = \langle \phi(\mathbf{x}), \phi(\mathbf{x}') \rangle_{\mathcal{F}}$$

### 흔한 알맹이

| 알맹이 | 식 | 특징 공간 |
|--------|---------|---------------|
| **선형** | $\kappa(\mathbf{x}, \mathbf{x}') = \mathbf{x}^T\mathbf{x}'$ | $\mathbb{R}^d$(항등 옮김) |
| **다항** | $\kappa(\mathbf{x}, \mathbf{x}') = (\mathbf{x}^T\mathbf{x}' + c)^p$ | $\mathbb{R}^{\binom{d+p}{p}}$(유한) |
| **방사 바탕 함수(정규)** | $\kappa(\mathbf{x}, \mathbf{x}') = \exp\!\left(-\frac{\|\mathbf{x} - \mathbf{x}'\|^2}{2\gamma^2}\right)$ | $\ell^2$(무한 차원) |
| **에스자** | $\kappa(\mathbf{x}, \mathbf{x}') = \tanh(\alpha \, \mathbf{x}^T\mathbf{x}' + c)$ | (늘 옳은 준양정치 알맹이는 아니다) |

**방사 바탕 함수 알맹이**는 넌지시 딸린 특징 공간이 무한 차원이라 이어진 어떤 비선형 짜임도 어림할 수 있어 특히 힘이 세다.

### 머서 조건

함수 $\kappa$이 옳은 알맹이일 필요충분조건은 그것이 **준양정치**라는 것이다. 곧 아무 점 모임 $\{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(n)}\}$에 대해서도 그람 행렬 $K_{ij} = \kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$이 준양정치이다. 이러면 $\kappa$이 $\mathcal{F}$의 안쪽 곱을 셈하도록 하는 특징 옮김 $\phi$이 있음이 보장된다.

---

## 유도

### 특징 공간의 주성분 분석

$\boldsymbol{\Phi} = [\phi(\mathbf{x}^{(1)}), \ldots, \phi(\mathbf{x}^{(n)})]^T$을 특징 공간 나타냄의 $n \times D$ 행렬이라 하자($D = \dim(\mathcal{F})$은 무한일 수 있다). 특징의 가운데가 맞았다고 하면($\frac{1}{n}\sum_i \phi(\mathbf{x}^{(i)}) = \mathbf{0}$) 특징 공간의 공분산은 다음과 같다:

$$\mathbf{C}_\phi = \frac{1}{n}\boldsymbol{\Phi}^T\boldsymbol{\Phi}$$

주성분 분석은 $\mathbf{C}_\phi \mathbf{v} = \lambda \mathbf{v}$을 채우는 고유벡터 $\mathbf{v}$을 찾는다.

### 나타냄 정리

$\mathbf{C}_\phi \mathbf{v} = \frac{1}{n}\boldsymbol{\Phi}^T(\boldsymbol{\Phi}\mathbf{v}) = \lambda\mathbf{v}$이므로 $\lambda > 0$인 고유벡터는 옮긴 자료가 뻗는 공간 안에 있어야 한다:

$$\mathbf{v} = \sum_{i=1}^n \alpha_i \, \phi(\mathbf{x}^{(i)}) = \boldsymbol{\Phi}^T\boldsymbol{\alpha}$$

여기서 $\boldsymbol{\alpha} \in \mathbb{R}^n$은 어떤 계수 벡터이다.

### 알맹이로 바꾼 고윳값 문제

$\mathbf{C}_\phi\mathbf{v} = \lambda\mathbf{v}$에 $\mathbf{v} = \boldsymbol{\Phi}^T\boldsymbol{\alpha}$을 넣고 왼쪽에 $\boldsymbol{\Phi}$을 곱하면:

$$\frac{1}{n}\boldsymbol{\Phi}\boldsymbol{\Phi}^T\boldsymbol{\Phi}\boldsymbol{\Phi}^T\boldsymbol{\alpha} = \lambda\boldsymbol{\Phi}\boldsymbol{\Phi}^T\boldsymbol{\alpha}$$

$K_{ij} = \kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)})$인 **알맹이(그람) 행렬** $\mathbf{K} = \boldsymbol{\Phi}\boldsymbol{\Phi}^T$을 정하면:

$$\frac{1}{n}\mathbf{K}^2\boldsymbol{\alpha} = \lambda\mathbf{K}\boldsymbol{\alpha}$$

$\mathbf{K}$이 무너지지 않았다면 이는 다음으로 간단해진다:

$$\mathbf{K}\boldsymbol{\alpha} = n\lambda\boldsymbol{\alpha}$$

이는 **$n \times n$ 고윳값 문제**이다. 크기가 (어쩌면 무한인) $\mathcal{F}$의 차원이 아니라 표본 수 $n$에 달렸다.

### 고르게 맞추기

특징 공간의 고유벡터는 $\|\mathbf{v}\| = 1$을 채워야 한다:

$$\|\mathbf{v}\|^2 = \boldsymbol{\alpha}^T\mathbf{K}\boldsymbol{\alpha} = n\lambda \|\boldsymbol{\alpha}\|^2 = 1$$

그러므로 $\boldsymbol{\alpha} \leftarrow \boldsymbol{\alpha} / \sqrt{n\lambda}$으로 고르게 한다.

### 쏘기

점 $\mathbf{x}$을 $j$번째 알맹이 주성분에 쏜 것:

$$z_j = \langle \mathbf{v}_j, \phi(\mathbf{x}) \rangle = \sum_{i=1}^n \alpha_{ji} \, \kappa(\mathbf{x}^{(i)}, \mathbf{x}) = \boldsymbol{\alpha}_j^T \mathbf{k}_\mathbf{x}$$

여기서 $\mathbf{k}_\mathbf{x} = [\kappa(\mathbf{x}^{(1)}, \mathbf{x}), \ldots, \kappa(\mathbf{x}^{(n)}, \mathbf{x})]^T$이다.

---

## 특징 공간에서 가운데 맞추기

주성분 분석에는 가운데 맞춘 자료가 필요하다. $\mathcal{F}$에서 대놓고 가운데를 맞출 수 없으므로 알맹이 행렬의 가운데를 맞춘다:

$$\tilde{\mathbf{K}} = \mathbf{H}\mathbf{K}\mathbf{H}$$

여기서 $\mathbf{H} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}^T$은 가운데 맞춤 행렬이다. 펼치면:

$$\tilde{\mathbf{K}} = \mathbf{K} - \frac{1}{n}\mathbf{1}\mathbf{1}^T\mathbf{K} - \frac{1}{n}\mathbf{K}\mathbf{1}\mathbf{1}^T + \frac{1}{n^2}\mathbf{1}\mathbf{1}^T\mathbf{K}\mathbf{1}\mathbf{1}^T$$

새 시험 점에 대한 가운데 맞춘 알맹이 벡터는 다음과 같다:

$$\tilde{k}_i = \kappa(\mathbf{x}^{(i)}, \mathbf{x}_*) - \frac{1}{n}\sum_{j=1}^n \kappa(\mathbf{x}^{(j)}, \mathbf{x}_*) - \frac{1}{n}\sum_{j=1}^n \kappa(\mathbf{x}^{(i)}, \mathbf{x}^{(j)}) + \frac{1}{n^2}\sum_{j,l} \kappa(\mathbf{x}^{(j)}, \mathbf{x}^{(l)})$$

---

## 구현

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

---

## 보기: 겹동그라미

선형 주성분 분석은 어긋나고 알맹이 주성분 분석은 통하는 고전 보기이다:

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

이 보기에서 선형 주성분 분석은 두 동그라미를 겹치는 범위에 쏘지만 방사 바탕 함수 알맹이 주성분 분석은 갈라지는 자리에 옮긴다.

---

## 초매개변수 선택

### 알맹이 띠너비(방사 바탕 함수의 감마)

방사 바탕 함수의 띠너비 $\gamma$이 알맹이의 "국소함"을 다스린다:

- **$\gamma$이 작으면**(넓은 알맹이): 전역 짜임을 도드라지게 하며 선형 주성분 분석에 다가간다
- **$\gamma$이 크면**(좁은 알맹이): 국소 짜임을 도드라지게 하나 지나치게 맞출 위험이 있다

실전에서는 둘씩 거리의 가운뎃값을 바탕으로 $\gamma$을 잡는 어림짐작을 쓴다:

```python
from scipy.spatial.distance import pdist

def median_heuristic(X):
    """방사 바탕 함수의 gamma을 둘씩 거리의 가운뎃값으로 둔다."""
    dists = pdist(X, metric='euclidean')
    return np.median(dists)
```

### 다시 세우기 어긋남으로 하는 격자 찾기

알맹이 주성분 분석은 본디 공간에서 곧바로 다시 세울 수 없으므로 알맹이 결 맞음이나 뒤따르는 일의 잣대를 쓰는 방식이 있다:

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

---

## 앞그림 문제

### 어려움

알맹이 주성분 분석은 자료를 알맹이 고유벡터가 정하는 공간의 나타냄 $\mathbf{z}$으로 옮긴다. 그러나 본디 공간 $\mathbb{R}^d$으로 되돌아가는 곧바른 역옮김이 없다. 알맹이 주성분 공간의 점 $\mathbf{z}$이 주어질 때 그에 맞닿는 $\mathbf{x} \in \mathbb{R}^d$을 찾는 것을 **앞그림 문제**라 한다.

### 어림 풀이

Mika 외(1999)는 되풀이 붙박이점 방법을 내놓았다. 생각은 $\hat{\phi}$이 특징 공간에서 다시 세운 것일 때 $\|\phi(\mathbf{x}) - \hat{\phi}\|^2$을 가장 작게 하는 $\mathbf{x}$을 찾는 것이다:

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

이는 또렷한 풀개를 배우는 자기 부호기에 견주었을 때 타고난 한계이다.

---

## 다른 방법과의 견줌

| 방법 | 선형성 | 다시 세우기 | 규모 키우기 | 웃매개변수 |
|--------|-----------|---------------|-------------|-----------------|
| **주성분 분석** | 선형 | 정확 | $O(nd^2)$ | $k$뿐 |
| **알맹이 주성분 분석** | 비선형 | 어림(앞그림) | $O(n^3)$ | $k$, 알맹이 매개변수 |
| **자기 부호기** | 비선형 | 정확(풀개) | 한 바퀴마다 $O(ndk)$ | 얼개, 익히기 |
| **t-SNE** | 비선형 | 없음 | $O(n^2)$ 또는 $O(n\log n)$ | 헷갈림도 |
| **UMAP** | 비선형 | 어림 | $O(n^{1.14})$ | $k$, 이웃 수 |

### 언제 알맹이 주성분 분석을 쓰는가

알맹이 주성분 분석은 비선형 짜임을 잡아야 하되 자료 묶음이 작아 $O(n^3)$ 값이 받아들일 만할 때(흔히 $n < 10{,}000$) 가장 쓸모 있다. 더 큰 자료 묶음에는 아무렇게나 하는 어림이나 신경망 방법(자기 부호기)이 더 현실적이다.

---

## 계량 금융에서의 응용

금융에서는 비선형 요인 짜임이 자연스레 나타난다. 예컨대 금리의 기간 짜임은 주성분 분석이 잘 잡아내는 선형 요인 셋(높이, 기울기, 굽음)이 지배한다. 그러나 신용 스프레드와 출렁임 면은 알맹이 주성분 분석으로 나타낼 수 있는 비선형 기댐을 보인다:

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

---

## 요약

| 항목 | 내용 |
|--------|--------|
| **생각** | 알맹이 재주로 넌지시 딸린 차원 높은 특징 공간에서 주성분 분석 |
| **고윳값 문제** | $\mathbf{K}\boldsymbol{\alpha} = n\lambda\boldsymbol{\alpha}$($D \times D$이 아니라 $n \times n$) |
| **쏘기** | $z_j = \sum_i \alpha_{ji}\kappa(\mathbf{x}^{(i)}, \mathbf{x})$ |
| **가운데 맞추기** | 알맹이 행렬의 가운데를 맞춘다: $\tilde{\mathbf{K}} = \mathbf{H}\mathbf{K}\mathbf{H}$ |
| **복잡도** | $\mathbf{K}$의 고윳값 쪼개기에 $O(n^3)$ |
| **한계** | 곧바른 역옮김이 없다(앞그림 문제). $O(n^3)$으로 커진다 |
| **핵심 이점** | 닫힌 꼴 풀이로 하는 비선형 차원 줄이기 |
| **자기 부호기와 견주면** | 더 단순하나(익히기 없음) 규모를 키우기 어렵고 또렷한 풀개가 없다 |

## 연습문제

**연습문제 1.**
쏜 자료의 흩어짐을 가장 크게 해서 첫 주성분을 이끌어 내어라.

??? success "연습문제 1 풀이"
    $X$을 가운데 맞춘 자료 행렬($n \times d$)이라 하자. 쏜 흩어짐을 가장 크게 하는 단위 벡터 $w_1$을 찾는다. 곧 $S = \frac{1}{n} X^\top X$이 공분산 행렬일 때 $\max_{\|w\|=1} w^\top S w$이다. 제약 $w^\top w = 1$ 아래 라그랑주 곱수를 쓰면 $\nabla_w [w^\top S w - \lambda(w^\top w - 1)] = 2Sw - 2\lambda w = 0$이라 $Sw = \lambda w$을 얻는다. 풀이는 가장 큰 고윳값 $\lambda_1$에 딸린 $S$의 고유벡터이다. 쏜 흩어짐은 $w_1^\top S w_1 = \lambda_1$과 같다. $\square$

---

**연습문제 2.**
주성분 분석과 특잇값 쪼개기의 관계를 설명하라. 특잇값 쪼개기로 주성분 분석을 어떻게 효율 좋게 셈하는가?

??? success "연습문제 2 풀이"
    가운데 맞춘 자료 행렬의 특잇값 쪼개기가 $X = U \Sigma V^\top$이므로 공분산 행렬은 $S = \frac{1}{n} X^\top X = \frac{1}{n} V \Sigma^2 V^\top$이다. $V$의 세로줄이 $S$의 고유벡터(주성분)이고 $\sigma_i^2 / n$이 그에 딸린 고윳값(흩어짐)이다. 쏜 자료는 $XV = U\Sigma$이다. 특잇값 쪼개기는 $S$의 고윳값 쪼개기보다 수치가 안정되고 $n < d$일 때 $d \times d$ 행렬을 만들지 않아도 된다.

---

**연습문제 3.**
알맹이 주성분 분석은 언제 선형 주성분 분석보다 나은가? 보기를 들어라.

??? success "연습문제 3 풀이"
    알맹이 주성분 분석은 알맹이 함수 $k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$이 이끄는 특징 공간에서 주성분 분석을 해 비선형 짜임을 잡는다. 보기: 겹동그라미로 놓인 자료는 선형 주성분 분석으로 가를 수 없지만(두 동그라미가 겹치는 구간에 쏘인다) 방사 바탕 함수 알맹이를 쓰면 동그라미가 선형으로 갈라지는 공간에 옮겨져 알맹이 주성분 분석이 그 짜임을 뽑아낸다. 선형 쏘기로는 잡을 수 없는 비선형 다양체 짜임이 자료에 있을 때 알맹이 주성분 분석이 이롭다.

---

**연습문제 4.**
남길 주성분의 수를 어떻게 고르는가? 설명하는 흩어짐을 쓰는 방식을 적어라.

??? success "연습문제 4 풀이"
    앞선 $k$개 성분이 설명하는 흩어짐의 몫은 $\sum_{i=1}^k \lambda_i / \sum_{i=1}^d \lambda_i$이다. **팔꿈치 방법**은 $k$에 대한 쌓아 올린 설명 흩어짐을 그려, 성분을 더 넣어도 얻는 것이 줄어드는 "팔꿈치"에서 $k$을 고른다. 전체 흩어짐의 90~95%를 설명할 만큼 성분을 남기는 것이 흔한 문턱이다. 예컨대 100차원 자료 묶음에서 성분 3개가 흩어짐의 95%를 설명하면 3차원으로 줄여도 앎을 5%만 잃고 차원을 $33\times$ 줄인다.
