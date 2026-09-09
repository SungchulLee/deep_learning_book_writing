# 주성분 분석의 바탕
흩어짐 가장 크게 하기로 하는 선형 차원 줄이기.

---

**주성분 분석(PCA)**은 선형 차원 줄이기에 가장 널리 쓰이는 재주이다. $\mathbb{R}^d$의 자료가 주어질 때 주성분 분석은 자료의 흩어짐이 가장 큰 서로 직교하는 방향 $k < d$개를 찾아 그 낮은 차원 아래 공간에 쏜다. 그 결과는 평균 제곱 다시 세우기 어긋남을 가장 작게 한다는 뜻에서 가장 좋은 선형 눌러 담기이다.

주성분 분석을 알아 두는 것은 이를 비선형으로 넓힌 자기 부호기와 변분 자기 부호기의 바탕으로 꼭 필요하다.

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있다:

- 주성분 분석을 제약 있는 흩어짐 가장 크게 하기 문제로 세우고 고유벡터 풀이를 이끌어 내기
- 고윳값 쪼개기와 특잇값 쪼개기로 주성분 분석을 짜고 저마다의 맞바꿈 이해하기
- 다시 세우기 어긋남과 설명하는 흩어짐 비로 잃은 앎을 값으로 재기
- 실제 상황에서 실림(주방향)과 점수(쏜 좌표) 풀이하기
- 주성분 분석을 선형 자기 부호기와 잇고 왜 비선형으로 넓혀야 하는지 이해하기
- 그림 눌러 담기, 잡음 없애기, 특징 뽑기 같은 실제 문제에 주성분 분석 쓰기

---

## 2. 기하학적 직관

주성분 분석은 단순한 기하 물음에 답한다. 차원 높은 공간에 점 구름이 있을 때 그것을 쏘아 넣을 가장 좋은 낮은 차원 "평평한 것"(아핀 아래 공간)은 무엇인가?

여기서 "가장 좋다"는 자료가 퍼진 정도를 되도록 많이 지킨다는 뜻이다. 첫 주성분은 흩어짐이 가장 큰 방향을, 둘째는 첫째와 직교하는 방향 가운데 흩어짐이 가장 큰 방향을 잡고, 그렇게 이어진다. 뒤로 갈수록 방향마다 설명하는 흩어짐이 줄어든다.

공분산이 $\boldsymbol{\Sigma} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$인 가운데 맞춘 자료 $\mathbf{X} \in \mathbb{R}^{n \times d}$에서 단위 벡터 $\mathbf{v}$에 쏜 것의 흩어짐은 다음과 같다:

$$\operatorname{Var}(\mathbf{X}\mathbf{v}) = \mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v}$$

$\|\mathbf{v}\| = 1$ 아래에서 이를 가장 크게 하면 가장 큰 고윳값에 딸린 $\boldsymbol{\Sigma}$의 고유벡터를 얻는다. 온전한 주성분 분석 풀이는 으뜸 $k$개 고유벡터로 이루어진다.

---

## 3. 핵심 식 한눈에 보기

| 개념 | 식 |
|---------|---------|
| **공분산 행렬** | $\boldsymbol{\Sigma} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$ |
| **고윳값 문제** | $\boldsymbol{\Sigma}\mathbf{v} = \lambda \mathbf{v}$ |
| **쏘기(점수)** | $\mathbf{z} = \mathbf{W}^T \mathbf{x}$ |
| **실림** | $\mathbf{W} = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$ |
| **다시 세우기** | $\hat{\mathbf{x}} = \mathbf{W}\mathbf{W}^T \mathbf{x}$ |
| **다시 세우기 어긋남** | $\sum_{i=k+1}^{d} \lambda_i$ |
| **특잇값 쪼개기** | $\mathbf{X} = \mathbf{U}\mathbf{S}\mathbf{V}^T$ |
| **설명하는 흩어짐 비** | $\text{EVR}_k = \lambda_k / \sum_{i=1}^d \lambda_i$ |

---

## 4. 미리 다듬기

주성분을 셈하기 앞서 자료를 미리 다듬어야 한다. 공분산 행렬은 특징의 잣수와 자리에 기대므로 다듬지 않으면 잘못된 결과가 나올 수 있다.

### 평균으로 가운데 맞추기(꼭 필요)

관측마다 표본 평균을 뺀다:

$$\boldsymbol{\mu} = \frac{1}{n}\sum_{i=1}^n \mathbf{x}^{(i)}, \qquad \mathbf{x}^{(i)} \leftarrow \mathbf{x}^{(i)} - \boldsymbol{\mu}$$

평균으로 가운데를 맞추면 공분산 행렬이 자료 구름의 자리가 아니라 흩어짐을 잡는다. 가운데를 맞추지 않으면 첫 주성분이 그저 자료의 무게 중심 쪽을 가리킬 뿐이다.

### 특징 잣수 맞추기(권함)

특징마다 단위가 다르거나 크기가 크게 다르면 특징마다 단위 흩어짐으로 표준화한다:

$$\sigma_j^2 = \frac{1}{n}\sum_{i=1}^n \left(x_j^{(i)}\right)^2, \qquad x_j^{(i)} \leftarrow x_j^{(i)} / \sigma_j$$

잣수를 맞추지 않으면 미터와 킬로미터로 잰 특징이 섞인 자료 묶음에서 주성분 분석이 킬로미터 잣수의 특징에 지배된다. 표준화하면 주성분 분석이 공분산 행렬이 아니라 **상관 행렬**의 고윳값 쪼개기와 같아진다.

**예외:** 모든 특징이 같은 단위와 잣수를 가질 때(예컨대 그림의 화소 밝기)는 잣수 맞추기가 필요 없고 오히려 해로울 수 있다.

```python
def preprocess(X):
    """주성분 분석을 위해 자료의 가운데를 맞추고 잣수를 맞춘다."""
    mu = X.mean(axis=0)
    X_centered = X - mu

    sigma = X_centered.std(axis=0)
    X_scaled = X_centered / (sigma + 1e-10)  # 0으로 나누는 것을 피한다

    return X_scaled, mu, sigma
```

---

## 5. 실림과 점수

주성분 분석은 통계 문헌에서 특정한 이름을 가진 바탕 결과 둘을 낸다.

### 실림(주방향)

**실림**은 주성분마다를 본디 특징의 선형 아우름으로 정하는 계수이다. $k$번째 주성분에 대해:

$$\text{Loading}_k = \mathbf{v}_k = [v_{k1}, v_{k2}, \ldots, v_{kd}]^T$$

실림 계수 $v_{kj}$은 특징 $j$이 성분 $k$에 보태는 몫을 뜻한다. 절댓값이 크면 영향이 세고 부호는 보태는 방향을 뜻한다. 고유벡터가 단위 벡터이므로 $\|\mathbf{v}_k\| = 1$이다.

온전한 실림 행렬 $\mathbf{W} = [\mathbf{v}_1, \ldots, \mathbf{v}_k] \in \mathbb{R}^{d \times k}$은 주성분을 세로줄로 갖는다.

### 점수(쏜 좌표)

**점수**는 주성분 공간에서 자료 점의 좌표이다. 표본 $\mathbf{x}^{(i)}$과 성분 $k$에 대해:

$$z_{ik} = {\mathbf{x}^{(i)}}^T \mathbf{v}_k = \sum_{j=1}^d x_j^{(i)} v_{kj}$$

온전한 점수 행렬은 다음과 같다:

$$\mathbf{Z} = \mathbf{X} \mathbf{W} \in \mathbb{R}^{n \times k}$$

점수에는 중요한 성질 둘이 있다. 곧 서로 **상관이 없고**($i \neq j$이면 $\operatorname{Cov}(z_i, z_j) = 0$), $k$번째 점수의 흩어짐이 $k$번째 고윳값과 같다($\operatorname{Var}(z_k) = \lambda_k$).

### 쏘기와 다시 세우기

실림과 점수의 관계가 쏘기(부호화)와 다시 세우기(풀기)를 모두 준다:

$$\text{Projection:} \quad \mathbf{Z} = \mathbf{X}\mathbf{W}$$

$$\text{Reconstruction:} \quad \hat{\mathbf{X}} = \mathbf{Z}\mathbf{W}^T = \mathbf{X}\mathbf{W}\mathbf{W}^T$$

다시 세운 표본마다 무게 붙은 주방향의 합이다:

$$\hat{\mathbf{x}}^{(i)} = \sum_{k=1}^{K} z_{ik} \, \mathbf{v}_k = \sum_{k=1}^{K} \left({\mathbf{x}^{(i)}}^T \mathbf{v}_k\right) \mathbf{v}_k$$

### 잣수 맞춘 실림(상관 실림)

어떤 쓰임새에서는 실림에 그에 딸린 고윳값의 제곱근을 곱한다:

$$\text{Scaled Loading}_{kj} = v_{kj} \cdot \sqrt{\lambda_k}$$

잣수 맞춘 실림은 본디 특징과 주성분 사이의 **상관**을 뜻한다. 특징마다 잣수 맞춘 실림의 제곱합이 공통성(남긴 성분이 설명하는 흩어짐의 몫)과 같아 풀이에 쓸모 있다.

```python
def correlation_loadings(loadings, eigenvalues):
    """주성분과의 상관을 나타내도록 실림의 잣수를 맞춘다."""
    return loadings * np.sqrt(eigenvalues)[:, np.newaxis]
```

---

## 6. 다시 세우기 어긋남과 k 고르기

### 잃은 앎을 값으로 재기

주성분 $d$개 가운데 $k$개만 남기면 앎을 잃는다. 다시 세우기 어긋남이 이 잃음을 값으로 잰다:

$$\mathcal{E}_k = \frac{1}{n}\sum_{i=1}^n \left\|\mathbf{x}^{(i)} - \hat{\mathbf{x}}^{(i)}\right\|^2 = \sum_{j=k+1}^{d} \lambda_j$$

**전체 다시 세우기 어긋남은 버린 고윳값의 합과 같다.** 이 산뜻한 결과는 고유벡터가 정규 직교 바탕을 이룬다는 사실에서 곧바로 따라 나온다. 곧 어긋남이 버린 방향마다의 흩어짐으로 말끔히 쪼개진다.

### 가장 좋음

주성분 분석은 계수 $k$짜리 **모든** 선형 쏘기 가운데 다시 세우기 어긋남을 가장 작게 한다:

$$\mathbf{W}^* = \arg\min_{\mathbf{W}} \sum_{i=1}^n \left\|\mathbf{x}^{(i)} - \mathbf{W}\mathbf{W}^T \mathbf{x}^{(i)}\right\|^2 \quad \text{s.t.} \quad \mathbf{W}^T \mathbf{W} = \mathbf{I}$$

같은 계수의 다른 어떤 선형 쏘기도 더 낮은 어긋남을 내지 못한다.

### 설명하는 흩어짐 비

성분 $k$이 잡는 전체 흩어짐의 몫:

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{i=1}^d \lambda_i}$$

앞선 $k$개 성분이 쌓아 설명하는 흩어짐:

$$\text{Cumulative EVR}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}$$

### 성분의 개수 고르기

$k$을 고르는 흔한 전략 셋:

**흩어짐 문턱.** 쌓아 올린 설명 흩어짐 비가 바라는 문턱(흔히 90%나 95%)을 넘는 가장 작은 $k$을 고른다:

```python
def choose_n_components(eigenvalues, threshold=0.95):
    """바라는 설명 흩어짐을 얻는 가장 작은 k을 찾는다."""
    total = eigenvalues.sum()
    cumsum = np.cumsum(eigenvalues)
    k = np.searchsorted(cumsum / total, threshold) + 1
    return k
```

**스크리 그림.** 성분 번호에 대한 고윳값을 그리고 "팔꿈치", 곧 뚝 떨어진 뒤 고윳값이 거의 평평해지는 곳을 찾는다. 팔꿈치 앞의 성분은 신호를, 뒤의 성분은 잡음을 잡는다.

**다시 세우기 어긋남 예산.** 표본마다 받아들일 수 있는 최대 평균 제곱 어긋남을 정하고 그 예산 아래에 머무는 가장 작은 $k$을 고른다:

```python
def choose_by_error(eigenvalues, max_error):
    """받아들일 만한 다시 세우기 어긋남을 얻는 가장 작은 k을 찾는다."""
    cumsum_discarded = eigenvalues.sum() - np.cumsum(eigenvalues)
    k = np.searchsorted(-cumsum_discarded, -max_error) + 1
    return k
```

```python
def plot_scree(eigenvalues):
    """고윳값과 쌓아 올린 설명 흩어짐을 그린다."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(eigenvalues, 'o-')
    ax1.set_xlabel('Component')
    ax1.set_ylabel('Eigenvalue')
    ax1.set_title('Scree Plot')

    cumsum = np.cumsum(eigenvalues) / eigenvalues.sum()
    ax2.plot(cumsum, 'o-')
    ax2.axhline(0.95, color='r', linestyle='--', label='95%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.legend()

    plt.tight_layout()
    plt.show()
```

---

## 7. 선형 자기 부호기와의 이음

주성분 분석은 평균 제곱 어긋남 손실로 익힌 선형 자기 부호기와 정확히 같다.

### 구조

선형 자기 부호기는 다음으로 이루어진다:

- **부호기:** $\mathbf{z} = \mathbf{W}_e^T \mathbf{x}$(치우침 없음, 깨어남 없음)
- **풀개:** $\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{z}$(치우침 없음, 깨어남 없음)

익히기 목표는 다음과 같다:

$$\mathcal{L} = \frac{1}{n}\sum_{i=1}^n \left\|\mathbf{x}^{(i)} - \mathbf{W}_d \mathbf{W}_e^T \mathbf{x}^{(i)}\right\|^2$$

### 같음 정리

평균 제곱 어긋남 손실로 익힌 선형 자기 부호기는 모였을 때 다음을 채운다:

1. 부호기 무게가 $\boldsymbol{\Sigma}$의 으뜸 $k$개 고유벡터와 같은 아래 공간을 뻗는다
2. 가장 좋은 풀이는 무게가 묶여 있다: $\mathbf{W}_d = \mathbf{W}_e$
3. 다시 세운 것이 주성분 분석의 다시 세운 것과 같다
4. 손실이 주성분 분석의 다시 세우기 어긋남과 같다

행렬 $\mathbf{W}_d \mathbf{W}_e^T$은 $\mathbf{W}$이 주성분을 담을 때 $\mathbf{W}\mathbf{W}^T$으로 모인다.

```python
import torch
import torch.nn as nn

class LinearAutoencoder(nn.Module):
    """주성분 분석과 같은 선형 자기 부호기.

    평균 제곱 어긋남 손실을 쓰고 깨어남도 치우침도 없으면 익히기가
    주성분 분석 풀이로 모인다.
    """
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
```

| 갈래 | 주성분 분석(닫힌 꼴) | 선형 자기 부호기 |
|--------|------------------|--------------------|
| **방법** | 고윳값 쪼개기 / 특잇값 쪼개기 | 기울기 내려가기 |
| **빠르기** | 한 번에(빠름) | 되풀이(느림) |
| **정확함** | 정확한 풀이 | 주성분 분석으로 모인다 |
| **GPU 받침** | 제한됨 | 본디 받침 |
| **넓힐 수 있음** | 선형 꼴로 붙박이 | 비선형을 붙이기 쉽다 |

---

## 8. 주성분 분석의 한계

주성분 분석은 **선형 아래 공간**을 찾는다. 실제 자료는 흔히 **비선형 다양체** 위에 있으며 이 어긋남이 주성분 분석의 근본 한계이다.

### 비선형 짜임에서의 어긋남

스위스 롤을 보자. 3차원 공간에 묻힌 2차원 면이다. 주성분 분석은 평평한 2차원 면에 쏘므로 다양체 위에서 멀리 떨어진 점들이 쏜 그림에서 겹친다:

```python
from sklearn.datasets import make_swiss_roll

X, color = make_swiss_roll(n_samples=1000, noise=0.1)
# 속 차원은 2이지만 주성분 분석은 그것을 "펴지" 못한다
```

### 주성분 분석이 놓치는 짜임의 갈래

**굽은 다양체**(스위스 롤, S자 곡선): 주성분 분석이 평평한 아래 공간에 쏘아 다양체 짜임을 무너뜨린다. **다양체 위의 무리**: 굽은 면에 놓인 서로 다른 무리를 주성분 분석이 합쳐 버릴 수 있다. **층층 특징**(그림에서 모서리 → 결 → 물체): 주성분 분석은 선형 바꿈 하나만 쓰므로 층층 짜임을 잡지 못한다.

### 그래도 주성분 분석이 통할 때

이런 한계에도 주성분 분석은 다음일 때 알맞다. 곧 자료가 거의 선형일 때, 성분을 풀이할 수 있어야 할 때, 자료 묶음이 작을 때(자기 부호기는 지나치게 맞춰질 수 있다), 셈이 빨라야 할 때, 비선형 방법과 견줄 바탕이 필요할 때이다.

### 비선형 방법으로 넘어가기

비선형 깨어남 함수를 붙이면 선형 자기 부호기가 굽은 다양체를 배울 수 있는 비선형 자기 부호기가 된다:

```python
# 선형(≈ 주성분 분석)
encoder = nn.Linear(784, 32)
decoder = nn.Linear(32, 784)

# 비선형(다양체를 배울 수 있다)
encoder = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 32)
)
decoder = nn.Sequential(
    nn.Linear(32, 256), nn.ReLU(),
    nn.Linear(256, 784)
)
```

| 갈래 | 주성분 분석 | 비선형 자기 부호기 |
|--------|-----|-----------------------|
| **다양체** | 평평한 아래 공간만 | 굽은 다양체 |
| **특징** | 선형 아우름 | 비선형 특징 |
| **층층 짜임** | 없음 | 여러 층 |
| **풀이** | 닫힌 꼴(정확) | 배움(어림) |

---

## 9. 실전 응용

### 쓰임새 1: 차원 줄이기(2차원 → 1차원)

주성분 분석이 상관 있는 2차원 자료를 1차원 선에 어떻게 쏘는지 보이는 가장 작은 보기이다:

```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

np.random.seed(0)

# 상관 있는 2차원 자료를 만든다
x = np.random.normal(size=(200,))
y = 0.5 * x + 2 + 0.1 * np.random.normal(size=(200,))
X = np.column_stack([x, y])

# 1차원으로 줄이고 다시 세운다
pca = PCA(n_components=1).fit(X)
X_pca = pca.transform(X)
X_reconstructed = pca.inverse_transform(X_pca)

print(f"Original shape:      {X.shape}")           # (200, 2)
print(f"Projected shape:     {X_pca.shape}")        # (200, 1)
print(f"Reconstructed shape: {X_reconstructed.shape}")  # (200, 2)

# 본디 자료와 쏜 자료를 그린다
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(X[:, 0], X[:, 1], alpha=0.3, label="Original")
ax.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1],
           color='red', s=20, label="Projected")
ax.legend()
plt.show()
```

**출력:**

```
Original shape:      (200, 2)
Projected shape:     (200, 1)
Reconstructed shape: (200, 2)
```

### 쓰임새 2: MNIST 눌러 담기

흩어짐의 95%를 남긴 채 784차원 손글씨 숫자 그림을 눌러 담는다:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets

# 주성분 분석 자체는 넘파이로 하므로 텐서가 아니라 배열이 필요하다.
# transform 없이 불러오면 .data가 (N, 28, 28) uint8 텐서이고,
# numpy()로 옮긴 뒤 (N, 784)로 펴서 쓴다
train_set = datasets.MNIST(root='./data', train=True, download=True)
test_set = datasets.MNIST(root='./data', train=False, download=True)
X_train = train_set.data.numpy().reshape(-1, 784).astype(np.float32)
X_test = test_set.data.numpy().reshape(-1, 784).astype(np.float32)

# 흩어짐을 95% 남기는 주성분 분석
pca = PCA(n_components=0.95, svd_solver='full').fit(X_train)
X_reduced = pca.transform(X_test)
X_recovered = pca.inverse_transform(X_reduced)

print(f"Original dim:    {X_test.shape[1]}")       # 784
print(f"Reduced dim:     {X_reduced.shape[1]}")     # ~150
print(f"Compression:     {784 / X_reduced.shape[1]:.1f}x")
```

### 쓰임새 3: 잡음 거르기

주성분 분석은 흩어짐이 큰 아래 공간에 쏘고 거의 잡음만 잡는 흩어짐 작은 성분을 버려 자료의 잡음을 없앤다:

```python
# MNIST에 정규 잡음을 더한다
X_noisy = X_train + 10.0 * np.random.normal(size=X_train.shape)

# 잡음 낀 자료에 흩어짐을 90% 남겨 주성분 분석을 맞춘다
pca = PCA(n_components=0.9, svd_solver='full').fit(X_noisy)
X_filtered = pca.inverse_transform(pca.transform(X_noisy))

print(f"Components used for denoising: {pca.n_components_}")
```

핵심 눈썰미는 신호의 흩어짐은 으뜸 성분에 몰리고 잡음의 흩어짐은 모든 성분에 고루 퍼진다는 것이다. 잘라 내면 신호보다 잡음이 훨씬 많이 걷힌다.

### 쓰임새 4: 고유 얼굴

얼굴 그림에 주성분 분석을 쓰면 **고유 얼굴**, 곧 얼굴 공간의 주성분 방향이 나온다:

```python
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

faces = fetch_lfw_people(min_faces_per_person=60)
print(f"Dataset: {faces.data.shape}")  # (사람 수, 62*47)

# 성분 150개로 주성분 분석을 맞춘다
pca = PCA(n_components=150, svd_solver='randomized').fit(faces.data)

# 성분마다 "고유 얼굴"이다
eigenface_0 = pca.components_[0].reshape(62, 47)

# 성분 150개로 얼굴을 다시 세운다
components = pca.transform(faces.data)
reconstructed = pca.inverse_transform(components)
```

고유 얼굴마다 얼굴 자료 묶음에 걸친 흔들림의 한 결(빛의 방향, 머리 자세, 표정)을 잡는다. 어떤 얼굴이든 고유 얼굴의 무게 붙은 합으로 어림해 나타낼 수 있다.

### 쓰임새 5: PyTorch에서 선형 자기 부호기로 본 주성분 분석

선형 자기 부호기를 평균 제곱 어긋남 손실로 익히면 주성분 분석 풀이로 모인다:

```python
import torch
from torch import nn, optim
from torchvision import datasets, transforms

torch.manual_seed(0)

class PCAAutoencoder(nn.Module):
    """선형 자기 부호기(깨어남 없음, 치우침 없음) ≡ 주성분 분석."""
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=False)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.decoder(self.encoder(out))
        return out.view(x.size())

# 학습
transform = transforms.ToTensor()
train_data = datasets.MNIST('./data', train=True, transform=transform,
                             download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64,
                                            shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PCAAutoencoder(784, 20).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    epoch_loss = 0.0
    for batch_X, _ in train_loader:
        batch_X = batch_X.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(batch_X), batch_X)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: loss={epoch_loss / len(train_loader):.4f}")
```

---

## 10. 주성분 분석 결과 풀이하기

### 겹그림

겹그림은 점수(표본의 자리)와 실림(특징 화살표)을 같은 축에 겹쳐 놓아 한꺼번에 풀이할 수 있게 한다:

```python
import matplotlib.pyplot as plt

def biplot(scores, loadings, feature_names, labels=None):
    """주성분 겹그림: 점수는 흩뿌림으로, 실림은 화살표로."""
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(scores[:, 0], scores[:, 1],
                         c=labels, alpha=0.6, s=50)

    scale = np.abs(scores).max() * 0.8
    for load, name in zip(loadings.T, feature_names):
        ax.arrow(0, 0, load[0] * scale, load[1] * scale,
                 head_width=0.05 * scale, fc='red', ec='red', alpha=0.7)
        ax.text(load[0] * scale * 1.1, load[1] * scale * 1.1,
                name, fontsize=10, ha='center')

    ax.axhline(0, color='gray', ls='--', alpha=0.3)
    ax.axvline(0, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('PCA Biplot')
    plt.tight_layout()
    plt.show()
```

### 실림 풀이하기

성분마다 어느 특징이 가장 많이 보태는지 살핀다:

```python
def interpret_loadings(loadings, feature_names, n_top=5):
    """성분마다 가장 많이 보태는 특징을 보인다."""
    for k in range(loadings.shape[0]):
        abs_load = np.abs(loadings[k])
        top_idx = np.argsort(abs_load)[::-1][:n_top]
        print(f"\n=== PC{k+1} ===")
        for idx in top_idx:
            print(f"  {feature_names[idx]}: {loadings[k, idx]:.4f}")
```

---

## 11. 빠른 참고 짜기

```python
import torch
import numpy as np

def pca(X, n_components):
    """
    고윳값 쪼개기로 하는 주성분 분석.

    인수:
        X: 자료 행렬 [표본 수, 특징 수]
        n_components: 주성분의 수

    반환값:
        W: 주성분(실림) [특징 수, 성분 수]
        Z: 쏜 자료(점수) [표본 수, 성분 수]
        eigenvalues: 성분마다 설명하는 흩어짐
    """
    # 자료의 가운데를 맞춘다
    X_centered = X - X.mean(dim=0)

    # 공분산 행렬
    cov = X_centered.T @ X_centered / (X.shape[0] - 1)

    # 고윳값 쪼개기(대칭 행렬에는 eigh)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)

    # 고윳값 내림차순으로 정렬한다
    idx = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[idx[:n_components]]
    W = eigenvectors[:, idx[:n_components]]

    # 쏜다(점수를 셈한다)
    Z = X_centered @ W

    return W, Z, eigenvalues
```

---

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

## 정리하며

주성분 분석은 선형 차원 줄이기의 바탕 얼거리를 준다. 흩어짐 가장 크게 하기, 다시 세우기 어긋남 가장 작게 하기, 닫힌 꼴로 풀 수 있음이라는 핵심 성질 덕분에 실전의 연장이면서 이론의 바탕이 된다. 주성분 분석과 선형 자기 부호기가 같다는 점이 깊은 만들어 내는 모델로 가는 다리를 놓는다. 곧 자기 부호기와 변분 자기 부호기는 주성분 분석을 비선형으로 넓힌 것으로 볼 수 있다.

뒤이은 절들은 흩어짐 가장 크게 하기 관점에서 주성분 분석을 엄밀히 이끌어 내고, 고윳값 쪼개기와 특잇값 쪼개기 셈법을 자세히 다루며, 확률 꼴과 알맹이 꼴로 넓힌다.
