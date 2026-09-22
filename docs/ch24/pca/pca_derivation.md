# 주성분 분석 이끌어 내기
흩어짐 가장 크게 하기와 어긋남 가장 작게 하기 관점에서 주성분 분석을 엄밀히 이끌어 내기.

---

주성분 분석은 서로 같은 두 가지 꼴로 세울 수 있다. 곧 **최대 흩어짐**(쏜 자료가 가장 넓게 퍼지는 방향 찾기)과 **최소 다시 세우기 어긋남**(본디 자료를 가장 잘 어림하는 계수 $k$짜리 선형 쏘기 찾기)이다. 이 절은 둘 다 이끌어 내고 서로 같음을 밝히며 선형 오토인코더와의 이음을 세운다.

---

## 1. 자리매김과 기호

$\mathbf{X} \in \mathbb{R}^{n \times d}$을 $d$차원 관측 $n$개를 담은 가운데 맞춘 자료 행렬이라 하자. 가로줄 $\mathbf{x}^{(i)}$마다 평균을 뺐다. 곧 $\frac{1}{n}\sum_{i=1}^n \mathbf{x}^{(i)} = \mathbf{0}$이다.

표본 공분산 행렬은 다음과 같다:

$$\boldsymbol{\Sigma} = \frac{1}{n}\mathbf{X}^T\mathbf{X} \in \mathbb{R}^{d \times d}$$

$\boldsymbol{\Sigma}$이 실수 대칭이고 준양정치이므로 스펙트럼 쪼개기가 된다:

$$\boldsymbol{\Sigma} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T = \sum_{i=1}^d \lambda_i \mathbf{v}_i \mathbf{v}_i^T$$

여기서 $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$은 고윳값이고 $\{\mathbf{v}_1, \ldots, \mathbf{v}_d\}$은 정규 직교 고유 바탕이다.

!!! note "관례: $1/n$과 $1/(n-1)$"
    여기서는 $1/n$(모집단 꼴)을 쓴다. $1/(n-1)$(베셀 바로잡기)을 쓰면 고윳값의 크기는 달라지지만 고유벡터의 방향은 그대로여서 어느 관례에서나 주성분은 똑같다.

---

## 2. 이끌어 내기 1: 흩어짐 가장 크게 하기

### 성분 하나(k = 1)

쏜 자료 $\{z^{(i)} = {\mathbf{x}^{(i)}}^T \mathbf{v}\}_{i=1}^n$의 흩어짐을 가장 크게 하는 단위 벡터 $\mathbf{v} \in \mathbb{R}^d$을 찾는다.

자료의 가운데가 맞았으므로 쏜 평균은 0이다:

$$\bar{z} = \frac{1}{n}\sum_{i=1}^n {\mathbf{x}^{(i)}}^T \mathbf{v} = \left(\frac{1}{n}\sum_{i=1}^n \mathbf{x}^{(i)}\right)^T \mathbf{v} = \mathbf{0}^T \mathbf{v} = 0$$

쏜 흩어짐은 다음과 같다:

$$\operatorname{Var}(z) = \frac{1}{n}\sum_{i=1}^n \left({\mathbf{x}^{(i)}}^T \mathbf{v}\right)^2 = \frac{1}{n}\sum_{i=1}^n \mathbf{v}^T \mathbf{x}^{(i)} {\mathbf{x}^{(i)}}^T \mathbf{v} = \mathbf{v}^T \left(\frac{1}{n}\mathbf{X}^T\mathbf{X}\right) \mathbf{v} = \mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v}$$

가장 좋게 하는 문제는 다음과 같다:

$$\max_{\mathbf{v}} \; \mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} \quad \text{subject to} \quad \mathbf{v}^T \mathbf{v} = 1$$

**라그랑주 곱수로 푸는 풀이.** 라그랑주 함수를 세운다:

$$\mathcal{L}(\mathbf{v}, \lambda) = \mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} - \lambda(\mathbf{v}^T \mathbf{v} - 1)$$

기울기를 영으로 놓으면 다음과 같다.

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}} = 2\boldsymbol{\Sigma}\mathbf{v} - 2\lambda\mathbf{v} = \mathbf{0}$$

$$\boldsymbol{\Sigma}\mathbf{v} = \lambda\mathbf{v}$$

이는 $\boldsymbol{\Sigma}$의 고윳값 방정식이다. 가장 좋은 $\mathbf{v}$은 고유벡터여야 하고 거기 딸린 고윳값 $\lambda$이 쏜 흩어짐을 준다:

$$\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} = \mathbf{v}^T \lambda \mathbf{v} = \lambda \|\mathbf{v}\|^2 = \lambda$$

흩어짐을 가장 크게 하려면 **가장 큰 고윳값** $\lambda_1$에 딸린 고유벡터를 고른다. 따라서 $\mathbf{v}_1^* = \mathbf{v}_1$(첫 주성분)이다.

### 성분 여럿(k > 1)

둘째 주성분은 $\mathbf{v}_1$과 직교한다는 조건 아래 흩어짐을 가장 크게 해서 얻는다:

$$\max_{\mathbf{v}} \; \mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} \quad \text{s.t.} \quad \mathbf{v}^T \mathbf{v} = 1, \; \mathbf{v}^T \mathbf{v}_1 = 0$$

라그랑주 함수는 다음과 같다:

$$\mathcal{L} = \mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} - \lambda(\mathbf{v}^T \mathbf{v} - 1) - \mu(\mathbf{v}^T \mathbf{v}_1)$$

$\partial \mathcal{L}/\partial \mathbf{v} = 0$으로 두면:

$$2\boldsymbol{\Sigma}\mathbf{v} - 2\lambda\mathbf{v} - \mu\mathbf{v}_1 = \mathbf{0}$$

왼쪽에 $\mathbf{v}_1^T$을 곱하고 $\mathbf{v}_1^T \mathbf{v} = 0$과 $\boldsymbol{\Sigma}\mathbf{v}_1 = \lambda_1 \mathbf{v}_1$을 쓰면:

$$2\lambda_1 \underbrace{\mathbf{v}_1^T \mathbf{v}}_{= 0} - 2\lambda \underbrace{\mathbf{v}_1^T \mathbf{v}}_{= 0} - \mu \underbrace{\mathbf{v}_1^T \mathbf{v}_1}_{= 1} = 0 \implies \mu = 0$$

$\mu = 0$이므로 정지 조건이 $\boldsymbol{\Sigma}\mathbf{v} = \lambda\mathbf{v}$으로 줄어 $\mathbf{v}$은 또 고유벡터이다. 직교 제약으로 $\mathbf{v}_1$을 빼면 흩어짐을 가장 크게 하는 고름은 $\mathbf{v}_2$(고윳값 $\lambda_2$)이다.

귀납으로 $k$번째 주성분은 $k$번째로 큰 고윳값에 딸린 고유벡터이다.

### 한꺼번에 세우기

성분 $k$개 문제는 행렬 $\mathbf{W} \in \mathbb{R}^{d \times k}$에 대한 가장 좋게 하기 하나로도 적을 수 있다:

$$\max_{\mathbf{W}} \; \operatorname{tr}\!\left(\mathbf{W}^T \boldsymbol{\Sigma} \mathbf{W}\right) \quad \text{s.t.} \quad \mathbf{W}^T \mathbf{W} = \mathbf{I}_k$$

대각합 목표는 쏜 방향마다의 흩어짐을 더한 것이다. 풀이는 $\mathbf{W}^* = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$이고 최댓값은 $\sum_{i=1}^k \lambda_i$이다.

---

## 3. 이끌어 내기 2: 다시 세우기 어긋남 가장 작게 하기

### 정식화

흩어짐을 가장 크게 하는 대신 다시 세우기 제곱 어긋남의 평균을 가장 작게 하는 계수 $k$짜리 쏘기를 찾을 수도 있다:

$$\min_{\mathbf{W}} \; \frac{1}{n}\sum_{i=1}^n \left\|\mathbf{x}^{(i)} - \mathbf{W}\mathbf{W}^T \mathbf{x}^{(i)}\right\|^2 \quad \text{s.t.} \quad \mathbf{W}^T \mathbf{W} = \mathbf{I}_k$$

여기서 $\mathbf{W}\mathbf{W}^T$은 $\mathbf{W}$의 세로줄 공간으로 하는 직교 쏘기이다.

### 유도

제곱 노름을 펼치면:

$$\left\|\mathbf{x} - \mathbf{W}\mathbf{W}^T \mathbf{x}\right\|^2 = \mathbf{x}^T\mathbf{x} - 2\mathbf{x}^T\mathbf{W}\mathbf{W}^T\mathbf{x} + \mathbf{x}^T\mathbf{W}\underbrace{\mathbf{W}^T\mathbf{W}}_{=\mathbf{I}}\mathbf{W}^T\mathbf{x} = \mathbf{x}^T\mathbf{x} - \mathbf{x}^T\mathbf{W}\mathbf{W}^T\mathbf{x}$$

표본에 걸쳐 더하면:

$$\frac{1}{n}\sum_i \left\|\mathbf{x}^{(i)} - \mathbf{W}\mathbf{W}^T\mathbf{x}^{(i)}\right\|^2 = \underbrace{\frac{1}{n}\sum_i \left\|\mathbf{x}^{(i)}\right\|^2}_{\text{constant}} - \operatorname{tr}\!\left(\mathbf{W}^T \boldsymbol{\Sigma} \mathbf{W}\right)$$

이를 가장 작게 하는 것은 $\operatorname{tr}(\mathbf{W}^T \boldsymbol{\Sigma} \mathbf{W})$을 가장 크게 하는 것과 같으며 이는 바로 흩어짐 가장 크게 하기 문제이다. 그러므로 두 꼴은 서로 같다.

### 닫힌 꼴 어긋남

고유 바탕 $\{\mathbf{v}_1, \ldots, \mathbf{v}_d\}$을 쓰면 자료의 전체 흩어짐이 다음처럼 쪼개진다:

$$\frac{1}{n}\sum_i \left\|\mathbf{x}^{(i)}\right\|^2 = \operatorname{tr}(\boldsymbol{\Sigma}) = \sum_{j=1}^d \lambda_j$$

으뜸 $k$개 성분이 잡는 흩어짐은 $\sum_{j=1}^k \lambda_j$이므로 다시 세우기 어긋남은 다음과 같다:

$$\mathcal{E}_k = \sum_{j=1}^d \lambda_j - \sum_{j=1}^k \lambda_j = \sum_{j=k+1}^d \lambda_j$$

**다시 세우기 어긋남은 버린 고윳값의 합과 같다.**

---

## 4. 같음 간추리기

| 꼴 | 목표 | 풀이 |
|-------------|-----------|----------|
| **최대 흩어짐** | $\max_{\mathbf{W}} \operatorname{tr}(\mathbf{W}^T\boldsymbol{\Sigma}\mathbf{W})$ | 으뜸 $k$개 고유벡터 |
| **최소 어긋남** | $\min_{\mathbf{W}} \frac{1}{n}\sum_i \|\mathbf{x}^{(i)} - \mathbf{W}\mathbf{W}^T\mathbf{x}^{(i)}\|^2$ | 으뜸 $k$개 고유벡터 |

둘 다 같은 $\mathbf{W}^* = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$을 준다.

---

## 5. k차원 주성분 분석 알고리즘

가운데 맞춘 자료 $\mathbf{X} \in \mathbb{R}^{n \times d}$과 목표 차원 $k$이 주어질 때:

**걸음 1.** 공분산 행렬 $\boldsymbol{\Sigma} = \frac{1}{n}\mathbf{X}^T\mathbf{X}$을 셈한다.

**걸음 2.** 고윳값을 내림차순으로 정렬해 고윳값 쪼개기 $\boldsymbol{\Sigma} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$을 찾는다.

**걸음 3.** 실림 행렬 $\mathbf{W} = [\mathbf{v}_1, \ldots, \mathbf{v}_k] \in \mathbb{R}^{d \times k}$을 만든다.

**걸음 4.** 점수를 셈한다: $\mathbf{Z} = \mathbf{X}\mathbf{W} \in \mathbb{R}^{n \times k}$.

**걸음 5.** 다시 세운다: $\hat{\mathbf{X}} = \mathbf{Z}\mathbf{W}^T = \mathbf{X}\mathbf{W}\mathbf{W}^T \in \mathbb{R}^{n \times d}$.

다시 세운 $\hat{\mathbf{x}}^{(i)}$은 무게 붙은 주방향의 합이다:

$$\hat{\mathbf{x}}^{(i)} = \sum_{j=1}^k \underbrace{\left({\mathbf{x}^{(i)}}^T \mathbf{v}_j\right)}_{\text{score}} \, \mathbf{v}_j$$

```python
import numpy as np

def pca(X, k):
    """주성분 분석: 흩어짐 가장 크게 하기 / 다시 세우기 어긋남 가장 작게 하기.

    인수:
        X: 가운데 맞춘 자료 [n, d]
        k: 성분의 수

    반환값:
        W: 실림 [d, k]
        Z: 점수 [n, k]
        eigenvalues: 성분마다의 흩어짐 [k]
    """
    n = X.shape[0]
    Sigma = X.T @ X / n

    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
    idx = np.argsort(eigenvalues)[::-1]

    W = eigenvectors[:, idx[:k]]
    eigenvalues = eigenvalues[idx[:k]]
    Z = X @ W

    return W, Z, eigenvalues
```

---

## 6. 이끌어 내기 3: 선형 오토인코더와 같음

### 준비

인코더 $\mathbf{W}_e \in \mathbb{R}^{d \times k}$과 디코더 $\mathbf{W}_d \in \mathbb{R}^{d \times k}$을 가진 선형 오토인코더를 보자:

$$\text{Encode:} \quad \mathbf{z} = \mathbf{W}_e^T \mathbf{x}, \qquad \text{Decode:} \quad \hat{\mathbf{x}} = \mathbf{W}_d \mathbf{z}$$

다시 세운 것은 $\hat{\mathbf{x}} = \mathbf{W}_d \mathbf{W}_e^T \mathbf{x}$이고 평균 제곱 어긋남 손실은 다음과 같다:

$$\mathcal{L}(\mathbf{W}_e, \mathbf{W}_d) = \frac{1}{n}\sum_{i=1}^n \left\|\mathbf{x}^{(i)} - \mathbf{W}_d \mathbf{W}_e^T \mathbf{x}^{(i)}\right\|^2$$

### 인코더가 주어졌을 때 가장 좋은 디코더

$\mathbf{W}_e$을 고정하면 손실이 $\mathbf{W}_d$에 대해 이차이다. 미분해 0으로 두면:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}_d} = -\frac{2}{n}\sum_i \left(\mathbf{x}^{(i)} - \mathbf{W}_d \mathbf{W}_e^T \mathbf{x}^{(i)}\right) {\mathbf{x}^{(i)}}^T \mathbf{W}_e = \mathbf{0}$$

$$\boldsymbol{\Sigma}\mathbf{W}_e = \mathbf{W}_d (\mathbf{W}_e^T \boldsymbol{\Sigma} \mathbf{W}_e)$$

$\mathbf{W}_e^T \boldsymbol{\Sigma} \mathbf{W}_e$의 역이 있으면:

$$\mathbf{W}_d = \boldsymbol{\Sigma}\mathbf{W}_e (\mathbf{W}_e^T \boldsymbol{\Sigma} \mathbf{W}_e)^{-1}$$

### 전역 최적에서

$\mathcal{L}$의 전역 최솟값에서 $\mathbf{W}_e$과 $\mathbf{W}_d$의 세로줄 공간은 $\boldsymbol{\Sigma}$의 으뜸 $k$개 고유벡터가 뻗는 아래 공간과 같아진다. $\mathbf{W}_e$의 세로줄이 고유벡터와 결이 맞는 정규 직교이면 $\mathbf{W}_d = \mathbf{W}_e$이고 다시 세우기 행렬은 다음이 된다:

$$\mathbf{W}_d \mathbf{W}_e^T = \mathbf{W}\mathbf{W}^T$$

이는 주성분 쏘기와 똑같다.

### 실전에서 뜻하는 바

(깨어남 함수도 치우침도 없는) 선형 오토인코더를 평균 제곱 어긋남 손실로 기울기 내려가기로 익히면 주성분 분석 풀이로 모인다. 모였을 때의 손실은 주성분 분석의 다시 세우기 어긋남 $\sum_{j=k+1}^d \lambda_j$과 같다.

```python
import torch
import torch.nn as nn

class LinearAutoencoder(nn.Module):
    """기울기 내려가기로 주성분 분석 풀이를 배운다."""
    def __init__(self, d, k):
        super().__init__()
        self.encoder = nn.Linear(d, k, bias=False)
        self.decoder = nn.Linear(k, d, bias=False)

    def forward(self, x):
        return self.decoder(self.encoder(x))

def verify_equivalence(X, k, epochs=5000, lr=0.01):
    """주성분 분석과 선형 오토인코더의 다시 세우기 어긋남을 견준다."""
    X_centered = X - X.mean(axis=0)

    # 닫힌 꼴 주성분 분석
    Sigma = X_centered.T @ X_centered / X.shape[0]
    eigvals = np.linalg.eigh(Sigma)[0]
    eigvals = np.sort(eigvals)[::-1]
    pca_error = eigvals[k:].sum()

    # 선형 오토인코더
    X_t = torch.tensor(X_centered, dtype=torch.float32)
    model = LinearAutoencoder(X.shape[1], k)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        loss = ((X_t - model(X_t)) ** 2).mean()
        loss.backward()
        opt.step()

    ae_error = loss.item() * X.shape[1]  # 평균 제곱 어긋남 -> 전체 어긋남

    print(f"PCA error:       {pca_error:.6f}")
    print(f"Linear AE error: {ae_error:.6f}")
```

---

## 7. 설명하는 흩어짐 살피기

### 성분마다의 비

$$\text{EVR}_k = \frac{\lambda_k}{\sum_{i=1}^d \lambda_i} = \frac{\lambda_k}{\operatorname{tr}(\boldsymbol{\Sigma})}$$

### 쌓아 올린 비

$$\text{CEVR}_k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i} = 1 - \frac{\mathcal{E}_k}{\operatorname{tr}(\boldsymbol{\Sigma})}$$

쌓아 올린 비는 남긴 전체 흩어짐의 몫을 곧바로 잰다. 그 여집합이 다시 세우기 어긋남의 몫이다.

```python
def explained_variance_analysis(eigenvalues):
    """설명하는 흩어짐 비와 다시 세우기 어긋남을 셈한다."""
    total = eigenvalues.sum()
    evr = eigenvalues / total
    cumulative_evr = np.cumsum(evr)
    reconstruction_error = total - np.cumsum(eigenvalues)

    return evr, cumulative_evr, reconstruction_error
```

---

## 8. 주성분 분석 풀이의 성질

### 상관 없는 점수

점수 벡터끼리는 상관이 없다:

$$\operatorname{Cov}(\mathbf{z}) = \frac{1}{n}\mathbf{Z}^T\mathbf{Z} = \frac{1}{n}\mathbf{W}^T \mathbf{X}^T \mathbf{X} \mathbf{W} = \mathbf{W}^T \boldsymbol{\Sigma} \mathbf{W} = \boldsymbol{\Lambda}_k$$

여기서 $\boldsymbol{\Lambda}_k = \operatorname{diag}(\lambda_1, \ldots, \lambda_k)$이다. 쏜 자료의 공분산이 대각이다. 곧 주성분이 자료의 상관을 없앤다.

### 가장 좋음

**에카르트-영-미르스키 정리.** 계수 $k$짜리 모든 행렬 가운데 $\hat{\mathbf{X}} = \mathbf{X}\mathbf{W}\mathbf{W}^T$이 $\|\mathbf{X} - \hat{\mathbf{X}}\|_F^2$을 가장 작게 한다. 이는 직교 쏘기 가운데 주성분 분석이 가장 좋다는 것보다 센 말이다. 곧 쏘기든 아니든 계수 $k$짜리 어떤 행렬도 더 낮은 프로베니우스 노름 어긋남을 내지 못한다.

### 흩어짐 쪼개기

전체 흩어짐이 남긴 몫과 잃은 몫으로 쪼개진다:

$$\underbrace{\operatorname{tr}(\boldsymbol{\Sigma})}_{\text{total}} = \underbrace{\sum_{i=1}^k \lambda_i}_{\text{retained}} + \underbrace{\sum_{i=k+1}^d \lambda_i}_{\text{reconstruction error}}$$

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
쏜 자료의 흩어짐을 가장 크게 해서 첫 주성분을 이끌어 내어라.

</div>

??? success "연습문제 1 풀이"
    $X$을 가운데 맞춘 자료 행렬($n \times d$)이라 하자. 쏜 흩어짐을 가장 크게 하는 단위 벡터 $w_1$을 찾는다. 곧 $S = \frac{1}{n} X^\top X$이 공분산 행렬일 때 $\max_{\|w\|=1} w^\top S w$이다. 제약 $w^\top w = 1$ 아래 라그랑주 곱수를 쓰면 $\nabla_w [w^\top S w - \lambda(w^\top w - 1)] = 2Sw - 2\lambda w = 0$이라 $Sw = \lambda w$을 얻는다. 풀이는 가장 큰 고윳값 $\lambda_1$에 딸린 $S$의 고유벡터이다. 쏜 흩어짐은 $w_1^\top S w_1 = \lambda_1$과 같다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
주성분 분석과 특잇값 쪼개기의 관계를 설명하라. 특잇값 쪼개기로 주성분 분석을 어떻게 효율 좋게 셈하는가?

</div>

??? success "연습문제 2 풀이"
    가운데 맞춘 자료 행렬의 특잇값 쪼개기가 $X = U \Sigma V^\top$이므로 공분산 행렬은 $S = \frac{1}{n} X^\top X = \frac{1}{n} V \Sigma^2 V^\top$이다. $V$의 세로줄이 $S$의 고유벡터(주성분)이고 $\sigma_i^2 / n$이 그에 딸린 고윳값(흩어짐)이다. 쏜 자료는 $XV = U\Sigma$이다. 특잇값 쪼개기는 $S$의 고윳값 쪼개기보다 수치가 안정되고 $n < d$일 때 $d \times d$ 행렬을 만들지 않아도 된다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
알맹이 주성분 분석은 언제 선형 주성분 분석보다 나은가? 보기를 들어라.

</div>

??? success "연습문제 3 풀이"
    알맹이 주성분 분석은 알맹이 함수 $k(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$이 이끄는 특징 공간에서 주성분 분석을 해 비선형 짜임을 잡는다. 보기: 겹동그라미로 놓인 자료는 선형 주성분 분석으로 가를 수 없지만(두 동그라미가 겹치는 구간에 쏘인다) 방사 바탕 함수 알맹이를 쓰면 동그라미가 선형으로 갈라지는 공간에 옮겨져 알맹이 주성분 분석이 그 짜임을 뽑아낸다. 선형 쏘기로는 잡을 수 없는 비선형 다양체 짜임이 자료에 있을 때 알맹이 주성분 분석이 이롭다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
남길 주성분의 수를 어떻게 고르는가? 설명하는 흩어짐을 쓰는 방식을 적어라.

</div>

??? success "연습문제 4 풀이"
    앞선 $k$개 성분이 설명하는 흩어짐의 몫은 $\sum_{i=1}^k \lambda_i / \sum_{i=1}^d \lambda_i$이다. **팔꿈치 방법**은 $k$에 대한 쌓아 올린 설명 흩어짐을 그려, 성분을 더 넣어도 얻는 것이 줄어드는 "팔꿈치"에서 $k$을 고른다. 전체 흩어짐의 90~95%를 설명할 만큼 성분을 남기는 것이 흔한 문턱이다. 예컨대 100차원 자료 묶음에서 성분 3개가 흩어짐의 95%를 설명하면 3차원으로 줄여도 앎을 5%만 잃고 차원을 $33\times$ 줄인다.


---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
흩어짐을 가장 크게 하는 유도와 다시 세우기 어긋남을 가장 작게 하는 유도가 왜 같은 답을 주는가?

</div>

??? success "연습문제 5 풀이"
    두 목표의 합이 상수이기 때문이다.

    단위 벡터 $w$에 사영할 때, 각 점을 $x_i - \bar x$로 두면 피타고라스로

    $$\underbrace{\|x_i - \bar x\|^2}_{\text{상수}}
      = \underbrace{(w^\top (x_i - \bar x))^2}_{\text{사영의 크기}}
      + \underbrace{\|(x_i - \bar x) - w w^\top (x_i - \bar x)\|^2}_{\text{버린 몫}}$$

    이 식을 모든 $i$에 대해 더하면 왼쪽은 $w$와 무관한 상수이고 오른쪽의 두 항이
    각각 흩어짐과 다시 세우기 어긋남이다.

    $$\text{전체 흩어짐} = \text{설명한 흩어짐} + \text{다시 세우기 어긋남}$$

    그러므로 앞 항을 가장 크게 하는 것과 뒤 항을 가장 작게 하는 것이 **정확히 같은
    문제**다. $\square$

    이 관계가 [기본 연습문제 3](pca_fundamentals.md)에서 수치로 확인한 것의 근거다.
    버린 고윳값의 합이 어긋남과 일치했던 것이 바로 이 분해다.

    두 유도를 모두 적어 두는 까닭은 **일반화되는 방향이 다르기** 때문이다. 흩어짐
    쪽은 여러 변수의 관계를 보는 방법들로, 어긋남 쪽은 오토인코더와 행렬 채우기로
    이어진다.

---

<div class="drillbox" markdown>

**연습문제 6.** <span class="diff hard" title="어려움"></span>
제1성분이 공분산 행렬의 가장 큰 고유 벡터라는 것을 라그랑주 승수로 보이라.

</div>

??? success "연습문제 6 풀이"
    사영한 흩어짐을 가장 크게 하는 문제는 이렇다.

    $$\max_w \; w^\top \Sigma w \quad \text{제약} \quad w^\top w = 1$$

    제약이 없으면 $w$를 키우는 것만으로 목적이 무한히 커지므로 제약이 필요하다.
    라그랑주 함수를 세운다.

    $$\mathcal{L}(w, \lambda) = w^\top \Sigma w - \lambda (w^\top w - 1)$$

    $w$로 미분해 0으로 둔다.

    $$\frac{\partial \mathcal{L}}{\partial w} = 2\Sigma w - 2\lambda w = 0
      \quad \Longrightarrow \quad \Sigma w = \lambda w$$

    곧 **정류점이 곧 고유 벡터**다. 그리고 그때의 목적값이

    $$w^\top \Sigma w = w^\top (\lambda w) = \lambda$$

    이므로 목적값이 곧 고윳값이다. 따라서 가장 큰 고윳값에 딸린 고유 벡터가 답이다.
    $\square$

    두 가지를 짚어 둘 만하다.

    **왜 최댓값이 있는가.** $\Sigma$가 대칭 반양정이므로 고윳값이 모두 실수이고 아래로
    묶인다. 단위 구가 옹골지므로 연속 함수인 목적이 최댓값을 가진다.

    **다음 성분은 어떻게 되는가.** $w_1$에 직교한다는 제약을 더해 같은 일을 하면 두 번째
    고유 벡터가 나온다. 이것이 성분들이 직교하는 까닭이며, 대칭 행렬의 고유 벡터가
    직교한다는 사실의 다른 얼굴이다.

---

<div class="drillbox" markdown>

**연습문제 7.** <span class="diff med" title="중간"></span>
고유 쪼개기와 특잇값 쪼개기 가운데 무엇으로 짜야 하는가?

</div>

??? success "연습문제 7 풀이"
    **특잇값 쪼개기 쪽이 거의 언제나 낫다.**

    자료 행렬 $X_c$(가운데 맞춘 것)의 특잇값 쪼개기가 $X_c = U S V^\top$이면

    $$\Sigma = \frac{X_c^\top X_c}{n-1} = V \frac{S^2}{n-1} V^\top$$

    이므로 $V$의 열이 성분이고 $\lambda_j = s_j^2/(n-1)$이다. 같은 답이다.

    차이는 수치와 비용이다.

    | | 고유 쪼개기 | 특잇값 쪼개기 |
    |---|---|---|
    | $\Sigma$를 만드는가 | 만든다 ($d \times d$) | 안 만든다 |
    | 조건수 | $\Sigma$의 조건수 = $X$의 제곱 | $X$의 조건수 |
    | $d \gg n$일 때 | $784^2$ 행렬을 만든다 | 필요 없다 |

    둘째 줄이 핵심이다. $\Sigma$를 만드는 것은 제곱하는 일이므로 조건수가 제곱되어
    작은 고윳값의 정확도가 크게 나빠진다. 특잇값 쪼개기는 $X$에 바로 작용하므로 그
    손실이 없다.

    MNIST는 $n=60{,}000$, $d=784$라 어느 쪽도 감당할 만하다. 그런데 유전체 자료처럼
    $d$가 수만이고 $n$이 수백이면 $\Sigma$를 만드는 것 자체가 불가능하고, 특잇값
    쪼개기라야 된다.

    아주 큰 자료에서는 처음 $k$개만 구하는 무작위 특잇값 쪼개기를 쓴다.

---

<div class="drillbox" markdown>

**연습문제 8.** <span class="diff med" title="중간"></span>
성분의 부호는 정해지는가?

</div>

??? success "연습문제 8 풀이"
    정해지지 않는다. $w$가 고유 벡터이면 $-w$도 고유 벡터이며 사영한 흩어짐이 같다.

    그래서 같은 자료에 같은 알고리즘을 돌려도 꾸러미나 판마다 부호가 다를 수 있다.
    실무에서 겪는 일이다.

    | 무엇이 바뀌는가 | 무엇이 안 바뀌는가 |
    |---|---|
    | 점수의 부호 | 다시 세우기 |
    | 실림 그림이 뒤집힘 | 설명하는 흩어짐 |
    | 흩뿌림 그림이 거울처럼 뒤집힘 | 점들 사이의 거리 |

    오른쪽 칸이 중요하다. **쓸모 있는 것은 모두 부호에 딸리지 않는다.** 그래서 대개
    걱정하지 않아도 된다.

    성분을 **풀이할 때**는 문제가 된다. "제2성분이 크면 장기 금리가 높다"라고 적었는데
    다시 돌렸을 때 부호가 뒤집히면 글이 틀리게 된다. 그래서 부호를 고정하는 관례를
    두는 것이 좋다. 예컨대 실림의 합이 양수가 되게 하거나, 가장 큰 절댓값을 갖는
    성분이 양수가 되게 맞춘다.

    ```python
    for j in range(k):
        if V[j].sum() < 0:
            V[j] = -V[j]
    ```

    고윳값이 겹치면 부호보다 더한 자유가 생긴다. 그 고유 공간 안에서 어떤 직교 기저든
    답이 되므로 성분 개개의 뜻이 사라진다. 그때는 개별 성분을 풀이하지 않는 것이 맞다.

---

<div class="drillbox" markdown>

**연습문제 9.** <span class="diff med" title="중간"></span>
주성분 분석 풀이가 하나로 정해지는 조건은 무엇인가?

</div>

??? success "연습문제 9 풀이"
    고윳값이 서로 다르면(겹치지 않으면) 부호를 빼고 하나로 정해진다.

    $\lambda_1 > \lambda_2 > \cdots > \lambda_k > \lambda_{k+1}$이면 각 고유 공간이
    1차원이므로 성분마다 방향이 하나씩 정해진다.

    겹치면 달라진다. $\lambda_i = \lambda_{i+1}$이면 그 2차원 고유 공간 안의 **어떤
    직교 기저든** 답이 되므로 성분 개개는 정해지지 않는다. 정해지는 것은 부분 공간뿐이다.

    실제 자료에서 정확히 겹치는 일은 드물지만 **거의 겹치는** 일은 흔하고, 그때는
    수치적으로 불안정해진다. 표본을 조금 바꾸거나 씨앗을 바꾸면 두 성분이 서로 섞인다.

    확인하는 법은 고윳값을 나란히 보는 것이다. MNIST의 앞쪽은 9.705%, 7.096%, 6.169%로
    충분히 떨어져 있어 앞 몇 성분은 안정적이다. 그런데 뒤로 가면 값이 촘촘해지므로
    (100번째가 0.101%) 개별 성분을 풀이하려 해서는 안 된다.

    **$k$번째와 $k+1$번째 고윳값의 틈**을 보는 것이 $k$를 고르는 또 하나의 근거가 되기도
    한다. 틈이 크면 그 자리에서 끊는 것이 안정적이다.

---

<div class="drillbox" markdown>

**연습문제 10.** <span class="diff hard" title="어려움"></span>
주성분 분석이 최선이라는 것은 어떤 뜻에서 참인가? 무엇을 가정하는가?

</div>

??? success "연습문제 10 풀이"
    에카르트–영 정리의 뜻에서 참이다. **계수 $k$의 모든 선형 사영 가운데** 제곱
    어긋남을 가장 작게 한다.

    가정이 그 문장 안에 둘 들어 있다.

    **선형이라는 가정.** 굽은 다양체는 다룰 수 없다. 겹동그라미 자료에서 주성분 분석으로
    1차원을 만들면 선형 분류 정확도가 45.2%로 찍기보다 못한데, 알맹이 주성분 분석은
    같은 1차원으로 100%를 낸다([알맹이 주성분 분석](kernel_pca.md)).

    **제곱 어긋남이라는 가정.** 이것은 가우시안 잡음을 가정하는 것과 같다. 이상값이
    있으면 제곱이 그것을 크게 벌하므로 성분이 이상값 쪽으로 끌려간다. 이상값에 튼튼한
    변형들이 여기서 나온다.

    가정 밖의 것도 짚어 둘 만하다. 주성분 분석은 **표지를 보지 않는다.** 그러므로
    "가르기에 좋은 방향"을 찾아 주지 않는다. 흩어짐이 큰 방향이 가르는 방향과 다를 수
    있고, 실제로 다른 경우가 흔하다. 표지를 쓰려면 선형 판별 분석 쪽이다.

    그래서 "주성분 분석은 최선이다"라는 말을 들을 때 **무엇들 가운데 최선인가**를
    물어야 한다. 계수가 정해진 선형 사영들 가운데서만 최선이며, 그 울타리 밖에는 더
    나은 것이 많다.

## 정리하며

| 결과 | 말 |
|--------|-----------|
| **첫 주성분** | 가장 큰 고윳값에 딸린 $\boldsymbol{\Sigma}$의 고유벡터 |
| **$k$번째 주성분** | $k$번째로 큰 고윳값에 딸린 고유벡터 |
| **쏜 흩어짐** | 고윳값과 같다: $\operatorname{Var}(z_k) = \lambda_k$ |
| **다시 세우기 어긋남** | 버린 고윳값의 합: $\sum_{j > k} \lambda_j$ |
| **최대 흩어짐 ≡ 최소 어긋남** | 같은 풀이: 으뜸 $k$개 고유벡터 |
| **선형 오토인코더 ≡ 주성분 분석** | 평균 제곱 어긋남으로 익힌 선형 오토인코더는 주성분 분석 풀이로 모인다 |
| **점수의 상관 없음** | $\operatorname{Cov}(\mathbf{z}) = \boldsymbol{\Lambda}_k$ |
| **가장 좋음** | 계수 $k$짜리 가장 좋은 어림(에카르트-영-미르스키) |
