# 교차 엔트로피 손실
## 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 최대가능도 추정의 원리에서 교차 엔트로피 손실 유도하기
- 교차 엔트로피의 정보 이론적 해석 이해하기
- 다중 클래스 분류에서 NLL과 교차 엔트로피가 동등함을 증명하기
- KL 발산을 교차 엔트로피 최적화와 연결하기
- 모델 매개변수에 대한 교차 엔트로피 손실의 경사를 완전히 유도하기
- 우아한 경사 공식 $\nabla_\mathbf{z}\mathcal{L} = \hat{\boldsymbol{\pi}} - \mathbf{y}$을 PyTorch로 확인하기

!!! note "함께 볼 것"
    교차 엔트로피는 **3.4절 소프트맥스 회귀**에도 등장한다. 거기서는 소프트맥스 분류기의 자연스러운 손실로 유도하고 PyTorch의 세 인터페이스(`nn.CrossEntropyLoss`, `F.cross_entropy`, `nn.NLLLoss`)를 쓴 N-그램 언어 모델로 보여준다. 이 절은 더 폭넓게 다룬다. 정보 이론적 토대, 행렬 형태를 포함한 전체 배치 경사 유도, NumPy로 바닥부터 만드는 구현, 그리고 초점 손실과의 관계를 살펴본다.

---

## 최대가능도의 틀

### 문제 설정

클래스가 $K$개인 다중 클래스 분류에서는 다음을 갖는다.

- **데이터:** $y^{(i)} \in \{1, \ldots, K\}$인 $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{N}$
- **모델:** 확률 $\hat{\pi}_k^{(i)} = P(Y = k \mid \mathbf{x}^{(i)};\, \boldsymbol{\theta})$을 예측한다
- **목표:** 관측된 데이터의 가능도를 최대로 만드는 매개변수 $\boldsymbol{\theta}$을 찾는다

### 가능도 함수

표본들이 독립이라고 가정하면 가능도는 다음과 같다.

$$\mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^{N} P\bigl(Y = y^{(i)} \mid \mathbf{x}^{(i)};\, \boldsymbol{\theta}\bigr) = \prod_{i=1}^{N} \hat{\pi}_{y^{(i)}}^{(i)}$$

$y_k^{(i)} = \mathbb{1}[y^{(i)} = k]$인 원-핫 부호 $\mathbf{y}^{(i)}$을 쓰면 다음과 같다.

$$\mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^{N} \prod_{k=1}^{K} \bigl(\hat{\pi}_k^{(i)}\bigr)^{y_k^{(i)}}$$

### 로그가능도

로그를 취하면(로그는 단조이므로 로그가능도를 최대화하는 것은 가능도를 최대화하는 것과 같다) 다음과 같다.

$$\ell(\boldsymbol{\theta}) = \log \mathcal{L}(\boldsymbol{\theta}) = \sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log \hat{\pi}_k^{(i)}$$

### 음의 로그가능도 (NLL)

보통 손실 함수는 최소화하므로 **음의 로그가능도**를 다음과 같이 정의한다.

$$\text{NLL}(\boldsymbol{\theta}) = -\ell(\boldsymbol{\theta}) = -\sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log \hat{\pi}_k^{(i)}$$

---

## 교차 엔트로피: 정보 이론의 관점

### 엔트로피와 정보

**엔트로피**는 분포의 평균 불확실성(또는 정보량)을 잰다.

$$H(\mathbf{p}) = -\sum_{k=1}^{K} p_k \log p_k = \mathbb{E}_{X \sim \mathbf{p}}[-\log p_X]$$

엔트로피는 분포가 결정적일 때 최소($= 0$)이고 균등할 때 최대($= \log K$)이다.

### 교차 엔트로피의 정의

참 분포 $\mathbf{p}$과 예측 분포 $\mathbf{q}$ 사이의 **교차 엔트로피**는 다음과 같다.

$$H(\mathbf{p}, \mathbf{q}) = -\sum_{k=1}^{K} p_k \log q_k = \mathbb{E}_{X \sim \mathbf{p}}[-\log q_X]$$

이는 $\mathbf{q}$에 최적화된 부호로 $\mathbf{p}$의 표본을 부호화할 때 필요한 평균 비트 수를 잰다.

### 분류에서의 교차 엔트로피

참 이름표가 원-핫 $\mathbf{y}$(참 클래스 $c$)이고 예측 확률이 $\hat{\boldsymbol{\pi}}$인 표본 하나에 대해 다음과 같다.

$$H(\mathbf{y}, \hat{\boldsymbol{\pi}}) = -\sum_{k=1}^{K} y_k \log \hat{\pi}_k = -\log \hat{\pi}_c$$

$y_c = 1$이고 $k \neq c$에 대해 $y_k = 0$이므로 참 클래스 항만 남는다.

---

## 동등성: NLL = 교차 엔트로피

### 수학적 증명

데이터셋 전체에 대해 다음이 성립한다.

$$\text{Cross-Entropy Loss} = \frac{1}{N} \sum_{i=1}^{N} H(\mathbf{y}^{(i)}, \hat{\boldsymbol{\pi}}^{(i)}) = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log \hat{\pi}_k^{(i)}$$

이는 정확히 $\frac{1}{N} \text{NLL}(\boldsymbol{\theta})$이다.

$$\boxed{\text{Cross-Entropy Loss} = \frac{1}{N} \text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{\pi}_{y^{(i)}}^{(i)}}$$

### 이것이 중요한 이유

| 관점 | 해석 |
|-------------|----------------|
| 통계적 | 최대가능도 추정 |
| 정보 이론적 | 부호화의 비효율을 최소화 |
| 최적화 | 다루기 좋은 경사를 갖는 볼록 손실 |
| 실무적 | 경험적으로 잘 작동한다 |

---

## KL 발산과의 관계

### KL 발산의 정의

$\mathbf{q}$에서 $\mathbf{p}$으로의 **쿨백-라이블러 발산**(상대 엔트로피)은 다음과 같다.

$$D_{KL}(\mathbf{p} \| \mathbf{q}) = \sum_{k=1}^{K} p_k \log \frac{p_k}{q_k} = H(\mathbf{p}, \mathbf{q}) - H(\mathbf{p})$$

### 분해

$$\boxed{H(\mathbf{p}, \mathbf{q}) = H(\mathbf{p}) + D_{KL}(\mathbf{p} \| \mathbf{q})}$$

교차 엔트로피는 $\mathbf{p}$에 내재한 불확실성(엔트로피)에 $\mathbf{p}$ 대신 $\mathbf{q}$을 쓰는 "추가 비용"(KL 발산)을 더한 것과 같다.

### 분류에서 (원-핫 이름표)

$\mathbf{y}$이 원-핫이면 $H(\mathbf{y}) = 0$이다(이름표에 불확실성이 없다). 따라서 다음이 성립한다.

$$H(\mathbf{y}, \hat{\boldsymbol{\pi}}) = D_{KL}(\mathbf{y} \| \hat{\boldsymbol{\pi}})$$

**교차 엔트로피 최소화 = 참 이름표로부터의 KL 발산 최소화.**

---

## 기하학적 해석

참 클래스가 $c$인 표본 하나에 대해 교차 엔트로피 손실은 다음과 같다.

$$\mathcal{L} = -\log \hat{\pi}_c$$

여기서 $\hat{\pi}_c = \sigma(\mathbf{z})_c$은 소프트맥스 확률이다.

**성질:**

- $\hat{\pi}_c = 1$일 때 $\mathcal{L} = 0$이다 (완벽한 예측)
- $\hat{\pi}_c \to 0$일 때 $\mathcal{L} \to \infty$이다 (완전히 틀림)
- $\hat{\pi}_c = 1/K$일 때 $\mathcal{L} = \log K$이다 (균등, 무작위 추측)

```
Loss = -log(p_true)

Loss ↑
  ∞  │╲
     │ ╲
  4  │  ╲
     │   ╲
  2  │    ╲___
     │        ╲____
  0  │             ╲___
     └──────────────────────→ p_true
     0    0.25   0.5    1.0

Key points:
  p_true = 0.01: Loss ≈ 4.6
  p_true = 0.5:  Loss ≈ 0.69
  p_true = 0.9:  Loss ≈ 0.1
  p_true = 1.0:  Loss = 0
```

---

## 경사 유도: 단계별로

### 소프트맥스 회귀 모델

클래스가 $K$개이고 입력 특징이 $\mathbf{x} \in \mathbb{R}^D$인 다중 클래스 분류에서 다음과 같다.

**로짓 (선형 점수):**

$$z_k = \mathbf{w}_k^T \mathbf{x} + b_k = \sum_{d=1}^{D} w_{kd} x_d + b_k$$

**예측 확률 (소프트맥스):**

$$\hat{\pi}_k = \sigma(\mathbf{z})_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

**매개변수:** $\mathbf{W} \in \mathbb{R}^{K \times D}$이고 $\mathbf{b} \in \mathbb{R}^K$인 $\boldsymbol{\theta} = \{\mathbf{W}, \mathbf{b}\}$이다.

참 클래스가 $c$인(원-핫으로는 $\mathbf{y}$) 표본 하나에 대한 **손실 함수**는 다음과 같다.

$$\mathcal{L} = -\log \hat{\pi}_c = -\sum_{k=1}^{K} y_k \log \hat{\pi}_k$$

### 1단계: 로짓에 대한 경사

연쇄 법칙으로 $\frac{\partial \mathcal{L}}{\partial z_j}$을 계산한다.

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{k=1}^{K} \frac{\partial \mathcal{L}}{\partial \hat{\pi}_k} \cdot \frac{\partial \hat{\pi}_k}{\partial z_j}$$

**$\frac{\partial \mathcal{L}}{\partial \hat{\pi}_k}$을 계산하면:**

$$\frac{\partial \mathcal{L}}{\partial \hat{\pi}_k} = -\frac{y_k}{\hat{\pi}_k}$$

**소프트맥스의 야코비 행렬을 쓰면:**

$$\frac{\partial \hat{\pi}_k}{\partial z_j} = \hat{\pi}_k(\delta_{kj} - \hat{\pi}_j)$$

**합치면:**

$$\frac{\partial \mathcal{L}}{\partial z_j} = \sum_{k=1}^{K} \left(-\frac{y_k}{\hat{\pi}_k}\right) \cdot \hat{\pi}_k(\delta_{kj} - \hat{\pi}_j) = -\sum_{k=1}^{K} y_k(\delta_{kj} - \hat{\pi}_j)$$

$$= -\sum_{k=1}^{K} y_k \delta_{kj} + \hat{\pi}_j \sum_{k=1}^{K} y_k$$

$\sum_k y_k = 1$(원-핫)이고 $\sum_k y_k \delta_{kj} = y_j$이므로 다음이 성립한다.

$$\frac{\partial \mathcal{L}}{\partial z_j} = -y_j + \hat{\pi}_j = \hat{\pi}_j - y_j$$

### 아름다운 결과

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{z}} = \hat{\boldsymbol{\pi}} - \mathbf{y}}$$

경사는 그저 **예측 확률과 참 이름표의 차이**이다.

**해석:**

- $\hat{\pi}_c \approx 1$이면(맞는 예측) 경사가 $\approx 0$이다 (작은 갱신)
- $\hat{\pi}_c \approx 0$이면(틀린 예측) 경사가 크다 (큰 갱신)
- 경사가 예측을 참 이름표 쪽으로 "민다"

### 2단계: 가중치에 대한 경사

$z_k = \sum_d w_{kd} x_d + b_k$에 연쇄 법칙을 쓰면 다음과 같다.

$$\frac{\partial \mathcal{L}}{\partial w_{kd}} = \frac{\partial \mathcal{L}}{\partial z_k} \cdot \frac{\partial z_k}{\partial w_{kd}} = (\hat{\pi}_k - y_k) \cdot x_d$$

**행렬 형태로 쓰면:**

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = (\hat{\boldsymbol{\pi}} - \mathbf{y}) \mathbf{x}^T}$$

여기서 $(\hat{\boldsymbol{\pi}} - \mathbf{y}) \in \mathbb{R}^K$이고 $\mathbf{x} \in \mathbb{R}^D$이므로 $\frac{\partial \mathcal{L}}{\partial \mathbf{W}} \in \mathbb{R}^{K \times D}$이다.

### 3단계: 편향에 대한 경사

$$\frac{\partial \mathcal{L}}{\partial b_k} = \frac{\partial \mathcal{L}}{\partial z_k} \cdot \frac{\partial z_k}{\partial b_k} = (\hat{\pi}_k - y_k) \cdot 1$$

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \hat{\boldsymbol{\pi}} - \mathbf{y}}$$

---

## 배치 경사 계산

표본 $N$개의 배치 $\{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^{N}$이 주어졌을 때 전체 손실은 다음과 같다.

$$\mathcal{L}_{\text{total}} = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}^{(i)}$$

$\mathbf{X} \in \mathbb{R}^{N \times D}$(표본이 행), $\hat{\mathbf{P}} \in \mathbb{R}^{N \times K}$(예측 확률), $\mathbf{Y} \in \mathbb{R}^{N \times K}$(원-핫 이름표)이라 하자.

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{1}{N} (\hat{\mathbf{P}} - \mathbf{Y})^T \mathbf{X}}$$

$$\boxed{\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \frac{1}{N} (\hat{\mathbf{P}} - \mathbf{Y})^T \mathbf{1}}$$

### 경사의 성질과 직관

**경사의 크기**는 예측의 확신도에 따라 달라진다.

| 참 클래스 확률 $\hat{\pi}_c$ | 경사의 크기 | 해석 |
|-------------------------------|-------------------|----------------|
| 0.99 | 작음 ($\approx 0.01$) | 확신하고 맞음 → 작은 갱신 |
| 0.50 | 중간 ($\approx 0.50$) | 불확실함 → 적당한 갱신 |
| 0.01 | 큼 ($\approx 0.99$) | 확신했는데 틀림 → 큰 갱신 |

**경사의 방향:** 참 클래스의 경사는 음수여서 로짓을 올리고, 다른 클래스의 경사는 양수여서 로짓을 내린다. 그 결과 참 클래스와 나머지 사이의 간격이 벌어진다.

**경사의 유계성:** $\|\nabla_\mathbf{z} \mathcal{L}\|_2 = \|\hat{\boldsymbol{\pi}} - \mathbf{y}\|_2 \leq \sqrt{2}$이며, 이것이 학습의 안정성에 도움이 된다.

---

## 흔한 변형과 확장

### 이진 교차 엔트로피

이진 분류($K = 2$)에서 교차 엔트로피는 다음으로 간단해진다.

$$\text{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y^{(i)} \log \hat{p}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)}) \right]$$

### 가중 교차 엔트로피

클래스가 불균형할 때는 클래스 빈도로 손실에 가중치를 준다.

$$\text{Weighted CE} = -\frac{1}{N} \sum_{i=1}^{N} w_{y^{(i)}} \log \hat{\pi}_{y^{(i)}}^{(i)}$$

### 이름표 평활화 교차 엔트로피

원-핫 목표 대신 평활화된 목표를 쓴다.

$$\tilde{y}_k = \begin{cases}
1 - \epsilon & \text{if } k = c \text{ (참 갈래)} \\
\frac{\epsilon}{K-1} & \text{otherwise}
\end{cases}$$

### 초점 손실

쉬운 예의 비중을 낮추어 클래스 불균형에 대응한다.

$$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

여기서 $\gamma > 0$은 초점 매개변수이고 $\alpha_t$은 가중 인수이다.

---

## L2 정칙화를 더하면

### 정칙화된 손실

$$\mathcal{L}_{\text{reg}} = \mathcal{L}_{CE} + \frac{\lambda}{2} \|\mathbf{W}\|_F^2$$

### 정칙화된 경사

$$\frac{\partial \mathcal{L}_{\text{reg}}}{\partial \mathbf{W}} = \frac{\partial \mathcal{L}_{CE}}{\partial \mathbf{W}} + \lambda \mathbf{W}$$

---

## PyTorch 구현

### nn.CrossEntropyLoss 이해하기

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch의 CrossEntropyLoss은 다음을 합친 것이다:
# 1. 소프트맥스: 로짓 → 확률
# 2. 로그: 확률 → 로그 확률
# 3. NLL: 참 클래스의 로그 확률을 골라 부호를 바꾸고 평균한다

criterion = nn.CrossEntropyLoss()

# 입력: 확률이 아니라 로짓(날것의 점수)이다!
logits = torch.tensor([[2.0, 1.0, 0.5],   # Sample 1
                       [0.5, 2.5, 1.0]])  # Sample 2

# 목표: 원-핫이 아니라 클래스 인덱스이다!
targets = torch.tensor([0, 1])  # Sample 1: class 0, Sample 2: class 1

loss = criterion(logits, targets)
print(f"CrossEntropyLoss: {loss.item():.4f}")
```

### 직접 계산하기

```python
def cross_entropy_manual(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Manual implementation of cross-entropy loss.

    CE = -mean(log(softmax(logits))[true_class])
       = mean(-log_softmax(logits)[true_class])
       = mean(NLL)
    """
    log_probs = F.log_softmax(logits, dim=1)
    nll = -log_probs[range(len(targets)), targets]
    return nll.mean()

# 동등함을 확인한다
loss_manual = cross_entropy_manual(logits, targets)
loss_pytorch = F.cross_entropy(logits, targets)
print(f"Manual:  {loss_manual.item():.6f}")
print(f"PyTorch: {loss_pytorch.item():.6f}")
print(f"Match:   {torch.allclose(loss_manual, loss_pytorch)}")
```

### CrossEntropyLoss 분해하기

```python
# CrossEntropyLoss = LogSoftmax + NLLLoss

log_softmax = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()

# 동등한 계산
log_probs = log_softmax(logits)
loss_decomposed = nll_loss(log_probs, targets)

print(f"Decomposed: {loss_decomposed.item():.6f}")
print(f"Direct CE:  {criterion(logits, targets).item():.6f}")
```

### PyTorch의 손실 변형들

```python
# 클래스 가중치를 쓰는 경우 (불균형 데이터용)
class_weights = torch.tensor([1.0, 2.0, 3.0])
weighted_ce = nn.CrossEntropyLoss(weight=class_weights)

# 이름표 평활화를 쓰는 경우
smooth_ce = nn.CrossEntropyLoss(label_smoothing=0.1)

# 특정 이름표를 무시하는 경우 (예: 채움값)
ignore_ce = nn.CrossEntropyLoss(ignore_index=-100)

# 초점 손실 (직접 구현)
def focal_loss(logits, targets, alpha=1.0, gamma=2.0):
    """
    Focal Loss for dense object detection.

    Args:
        logits: Raw model outputs
        targets: Ground truth class indices
        alpha: Weighting factor
        gamma: Focusing parameter
    """
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    pt = torch.exp(-ce_loss)  # pt = probability of true class
    focal_weight = alpha * (1 - pt) ** gamma
    return (focal_weight * ce_loss).mean()
```

### 경사 유도 확인하기

```python
def verify_gradient_derivation():
    """
    Verify our analytical gradients match PyTorch autograd.
    """
    torch.manual_seed(42)

    # 준비
    N, D, K = 4, 5, 3  # 4 samples, 5 features, 3 classes

    X = torch.randn(N, D)
    y = torch.randint(0, K, (N,))

    W = torch.randn(K, D, requires_grad=True)
    b = torch.randn(K, requires_grad=True)

    # 순전파
    logits = X @ W.T + b  # (N, K)
    probs = F.softmax(logits, dim=1)

    # 손실 (교차 엔트로피)
    loss = F.cross_entropy(logits, y)

    # Autograd의 역전파
    loss.backward()

    # 우리가 해석적으로 구한 경사
    y_onehot = F.one_hot(y, K).float()
    dz = probs.detach() - y_onehot  # (N, K)

    dW_analytical = (1/N) * dz.T @ X  # (K, D)
    db_analytical = (1/N) * dz.sum(dim=0)  # (K,)

    # 비교한다
    print("Gradient Verification")
    print("=" * 50)
    print(f"dW max error: {(W.grad - dW_analytical).abs().max().item():.2e}")
    print(f"db max error: {(b.grad - db_analytical).abs().max().item():.2e}")
    print(f"Gradients match: {torch.allclose(W.grad, dW_analytical, atol=1e-5)}")

verify_gradient_derivation()
```

### 경사의 흐름 시각화하기

```python
def visualize_gradient_flow():
    """
    Show how gradients flow through softmax + cross-entropy.
    """
    torch.manual_seed(42)

    # 분명히 보기 위해 표본 하나만 쓴다
    logits = torch.tensor([2.0, 1.0, 0.5], requires_grad=True)
    true_class = 0

    # 순전파
    probs = F.softmax(logits, dim=0)
    loss = -torch.log(probs[true_class])

    # 역전파
    loss.backward()

    print("Gradient Flow Visualization")
    print("=" * 50)
    print(f"Logits z:        {logits.detach().numpy().round(4)}")
    print(f"Probabilities π: {probs.detach().numpy().round(4)}")
    print(f"True class:      {true_class}")
    print(f"Loss:            {loss.item():.4f}")
    print()
    print(f"∂L/∂z (autograd):   {logits.grad.numpy().round(4)}")

    # 해석적 결과: π - y
    y_onehot = torch.zeros(3)
    y_onehot[true_class] = 1
    grad_analytical = probs.detach() - y_onehot
    print(f"∂L/∂z (analytical): {grad_analytical.numpy().round(4)}")
    print()
    print("Note: ∂L/∂z = π - y (predicted minus true)")

visualize_gradient_flow()
```

### NumPy로 바닥부터 구현하기

```python
import numpy as np

class SoftmaxRegressionNumPy:
    """
    Softmax regression implemented from scratch.
    Demonstrates the gradient derivations in code.
    """

    def __init__(self, input_dim: int, num_classes: int):
        self.W = np.random.randn(num_classes, input_dim) * 0.01
        self.b = np.zeros(num_classes)

    def softmax(self, z: np.ndarray) -> np.ndarray:
        """수치적으로 안정한 소프트맥스."""
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """순전파: X → 로짓 → 확률."""
        self.z = X @ self.W.T + self.b  # (N, K)
        self.probs = self.softmax(self.z)  # (N, K)
        return self.probs

    def compute_loss(self, probs: np.ndarray, y: np.ndarray) -> float:
        """교차 엔트로피 손실을 계산한다."""
        N = len(y)
        correct_log_probs = -np.log(probs[np.arange(N), y] + 1e-10)
        return np.mean(correct_log_probs)

    def backward(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Backward pass: compute gradients.

        The key insight: ∂L/∂z = π - y (one-hot)
        """
        N = len(y)

        # y를 원-핫 부호로 바꾼다
        y_onehot = np.zeros_like(self.probs)
        y_onehot[np.arange(N), y] = 1

        # 로짓에 대한 경사: dL/dz = π - y
        dz = self.probs - y_onehot  # (N, K)

        # 가중치에 대한 경사: dL/dW = (1/N) * dz^T @ X
        dW = (1/N) * dz.T @ X  # (K, D)

        # 편향에 대한 경사: dL/db = (1/N) * sum(dz)
        db = (1/N) * np.sum(dz, axis=0)  # (K,)

        return dW, db

    def train_step(self, X: np.ndarray, y: np.ndarray, lr: float) -> float:
        """학습 한 단계: 순전파, 역전파, 갱신."""
        probs = self.forward(X)
        loss = self.compute_loss(probs, y)
        dW, db = self.backward(X, y)
        self.W -= lr * dW
        self.b -= lr * db
        return loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        """클래스 이름표를 예측한다."""
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """분류 정확도를 계산한다."""
        predictions = self.predict(X)
        return np.mean(predictions == y)
```

---

## PyTorch 빠른 참조

| 함수 | 입력 | 비고 |
|----------|-------|-------|
| `nn.CrossEntropyLoss()` | 로짓 | 가장 흔하며 수치적으로 안정 |
| `F.cross_entropy()` | 로짓 | 함수형 버전 |
| `nn.NLLLoss()` | 로그 확률 | `log_softmax`와 함께 쓴다 |
| `F.log_softmax()` | 로짓 | 안정한 로그 확률 |
| `nn.BCEWithLogitsLoss()` | 로짓 | 이진 분류 |

---

## 요약

### 근본이 되는 식들

**교차 엔트로피 손실:**

$$\boxed{\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{\pi}_{y^{(i)}}^{(i)} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} y_k^{(i)} \log \hat{\pi}_k^{(i)}}$$

**동등한 표현들:**

$$\text{Cross-Entropy} = \frac{1}{N} \text{NLL} = H(\mathbf{p}, \mathbf{q}) = H(\mathbf{p}) + D_{KL}(\mathbf{p} \| \mathbf{q})$$

### 경사 요약

| 양 | 표본 하나 | 배치 (표본 $N$개) |
|----------|--------------|---------------------|
| 로짓에 대해 | $\hat{\boldsymbol{\pi}} - \mathbf{y}$ | — |
| 가중치에 대해 | $(\hat{\boldsymbol{\pi}} - \mathbf{y})\mathbf{x}^T$ | $\frac{1}{N}(\hat{\mathbf{P}} - \mathbf{Y})^T \mathbf{X}$ |
| 편향에 대해 | $\hat{\boldsymbol{\pi}} - \mathbf{y}$ | $\frac{1}{N}\mathbf{1}^T(\hat{\mathbf{P}} - \mathbf{Y})$ |

### 핵심 통찰

$$\boxed{\text{Gradient} = \text{Predicted} - \text{True}}$$

이 단순한 공식이 소프트맥스와 교차 엔트로피의 조합이 그토록 널리 쓰이는 이유이다.

### 교차 엔트로피를 보는 세 가지 관점

!!! info "세 가지 관점"

    1. **통계적:** 관측된 이름표의 가능도를 최대화한다
    2. **정보 이론적:** 부호화의 비효율을 최소화한다
    3. **기하학적:** (KL 발산을 통해) 분포 사이의 "거리"를 잰다

---

## 참고 문헌

1. Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*, Chapter 2.
2. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*, Chapter 6.2.2.
3. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Chapter 4.3.
4. Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*, Chapter 8.6.
5. Lin, T.-Y., et al. (2017). Focal Loss for Dense Object Detection. *ICCV*.
6. PyTorch Documentation: [nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

## 연습문제

**연습문제 1.**
교차 엔트로피가 $H(p, q) \geq H(p)$이며 등호는 $p = q$일 때에만 성립함을 증명하라(깁스 부등식).

??? success "연습문제 1 풀이"
    $H(p,q) - H(p) = -\sum p_i \log q_i + \sum p_i \log p_i = \sum p_i \log\frac{p_i}{q_i} = D_{\text{KL}}(p\|q) \geq 0$이다.

    KL 발산의 성질에 의해 등호는 $p = q$일 때에만 성립한다. $\square$

---

**연습문제 2.**
$K$개 클래스 분류 문제에서 로짓(소프트맥스 이전의 값)에 대한 교차 엔트로피 손실의 경사를 유도하라.

??? success "연습문제 2 풀이"
    $z_k$을 로짓, $p_k = \text{softmax}(z)_k$이라 하자. 교차 엔트로피는 $L = -\sum_k y_k \log p_k$이다.

    $$
    \frac{\partial L}{\partial z_j} = p_j - y_j
    $$

    이 우아한 결과는 소프트맥스의 야코비 행렬 $\frac{\partial p_k}{\partial z_j} = p_k(\delta_{kj} - p_j)$에서 나온다. 경사는 그저 예측 확률과 참 확률의 차이이다.

---

**연습문제 3.**
PyTorch에서 `nn.CrossEntropyLoss`과 `nn.LogSoftmax` + `nn.NLLLoss`이 동등함을 보여라.

??? success "연습문제 3 풀이"
    ```python
    import torch, torch.nn as nn
    logits = torch.randn(5, 10)
    targets = torch.randint(0, 10, (5,))

    loss1 = nn.CrossEntropyLoss()(logits, targets)
    loss2 = nn.NLLLoss()(nn.LogSoftmax(dim=1)(logits), targets)
    assert torch.allclose(loss1, loss2)
    ```
    `CrossEntropyLoss`은 수치적 안정성을 위해 내부에서 로그 소프트맥스와 NLL을 합쳐 계산한다.

---

**연습문제 4.**
$-\log(\text{softmax}(z)_k)$을 곧바로 계산할 때 생기는 수치적 안정성 문제와, log-sum-exp 기법이 그것을 어떻게 해결하는지 설명하라.

??? success "연습문제 4 풀이"
    곧바로 계산하면 $\text{softmax}(z)_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$은 $z_j$이 클 때 넘칠 수 있다. log-sum-exp 기법은 $m = \max_j z_j$을 빼 준다.

    $$
    \log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}
    $$

    이제 모든 지수가 $\leq 0$이 되어 넘침이 생기지 않는다. 최종 교차 엔트로피는 $-z_k + m + \log\sum_j e^{z_j - m}$이며 수치적 문제 없이 계산된다.
