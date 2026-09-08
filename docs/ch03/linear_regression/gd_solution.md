# 경사 하강법 해

정규 방정식이 정확한 닫힌 형태의 해를 주는 반면, **경사 하강법**은 확장성 있는
반복적 대안을 제공하며 신경망 학습의 토대가 된다. 이 페이지에서는 MSE 손실의 경사를
유도하고, 이를 음의 로그가능도와 연결한 뒤, 네 단계의 PyTorch 구현을 차례로 밟아
나간다. 텐서 연산을 직접 다루는 수준에서 시작해 실전에 쓸 수 있는 학습 파이프라인까지
이른다.

---

## 1. 음의 로그가능도로서의 MSE

### 1.1 되짚어 보기: 가우스 가능도

확률 모델
$y_i \mid \mathbf{x}_i \sim \mathcal{N}(\mathbf{w}^\top\mathbf{x}_i + b,\,\sigma^2)$
아래에서 $n$개 관측의 로그가능도는 다음과 같다.

$$
\ell(\mathbf{w}, b, \sigma^2)
= -\frac{n}{2}\ln(2\pi\sigma^2)

  - \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

$\sigma^2$을 고정했을 때 $(\mathbf{w}, b)$에 대해 $\ell$을 최대화하는 것은 다음을
최소화하는 것과 같다.

$$
\text{MSE}
= \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
$$

!!! tip "핵심 통찰"
    가우스 잡음 가정 아래에서 MSE는 (덧셈 상수와 양의 배율을 제외하면) 음의
    로그가능도이다. 따라서 MSE에 대한 경사 하강법은 통계적으로 정당화되는 최대가능도
    추정이다.

### 1.2 MSE가 무너지는 경우

| 상황 | 문제 | 대안 |
|----------|---------|-------------|
| 이상치 | MSE가 큰 오차를 증폭한다 | 후버 손실, MAE |
| 두꺼운 꼬리 잔차 | 가우스 가정이 성립하지 않는다 | 스튜던트 $t$ 가능도 |
| 이분산성 | $\sigma^2$이 일정하다는 가정이 깨진다 | 가중 MSE, $\sigma(\mathbf{x})$을 명시적으로 모형화 |

```python
import torch
import torch.nn as nn

# 이상치에 강한 손실 대안들
criterion_mse   = nn.MSELoss()       # L2 — 가우스 NLL
criterion_mae   = nn.L1Loss()        # L1 — 라플라스 NLL
criterion_huber = nn.HuberLoss(delta=1.0)  # L2에서 L1으로 매끄럽게 전이
```

---

## 2. 경사 계산

### 2.1 행렬 형태

간결한 표기 $\boldsymbol{\theta} = (b, w_1, \ldots, w_p)^\top$을 쓴 MSE 손실은
다음과 같다.

$$
J(\boldsymbol{\theta})
= \frac{1}{n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\theta}\|^2
$$

그 경사는 ([닫힌 형태의 해](closed_form.md)에서 유도했듯이) 다음과 같다.

$$
\nabla_{\boldsymbol{\theta}} J
= \frac{2}{n}\,\mathbf{X}^\top\!
  \bigl(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}\bigr)
= \frac{2}{n}\,\mathbf{X}^\top\!
  \bigl(\hat{\mathbf{y}} - \mathbf{y}\bigr)
$$

### 2.2 성분별 표현 (w, b 분리)

$$
\frac{\partial J}{\partial w_j}
= \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)\,x_{ij},
\qquad
\frac{\partial J}{\partial b}
= \frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)
$$

벡터 형태로 쓰면 다음과 같다.

$$
\nabla_{\mathbf{w}} J
= \frac{2}{n}\,\mathbf{X}^\top(\hat{\mathbf{y}} - \mathbf{y}),
\qquad
\frac{\partial J}{\partial b}
= \frac{2}{n}\,\mathbf{1}^\top(\hat{\mathbf{y}} - \mathbf{y})
$$

### 2.3 최적 학습률

MSE를 쓰는 선형 회귀에서 갱신이 안정적일 조건은 다음과 같다.

$$
\eta < \frac{2}{\lambda_{\max}(\mathbf{X}^\top\mathbf{X}/n)}
$$

여기서 $\lambda_{\max}$은 정규화된 그람 행렬의 가장 큰 고윳값이다.

```python
def compute_max_learning_rate(X: torch.Tensor) -> float:
    """MSE에 대한 경사 하강법이 안정적일 상한."""
    n = X.shape[0]
    eigenvalues = torch.linalg.eigvalsh(X.T @ X / n)
    return (2.0 / eigenvalues.max().item())
```

---

## 3. 경사 하강법의 변형들

### 3.1 배치 경사 하강법

갱신할 때마다 **전체** 데이터셋을 쓴다.

$$
\boldsymbol{\theta}^{(t+1)}
= \boldsymbol{\theta}^{(t)}

  - \eta\,\nabla_{\boldsymbol{\theta}} J(\boldsymbol{\theta}^{(t)})
$$

```python
def batch_gradient_descent(
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.01,
    n_epochs: int = 100,
) -> dict:
    """경사를 직접 계산하는 배치 GD (autograd 없음)."""
    n, d = X.shape
    w = torch.zeros(d, 1)
    b = torch.zeros(1)
    history = []

    for epoch in range(n_epochs):
        y_pred = X @ w + b
        loss = torch.mean((y - y_pred) ** 2)

        error = y_pred - y
        grad_w = (2.0 / n) * (X.T @ error)
        grad_b = (2.0 / n) * error.sum()

        w = w - lr * grad_w
        b = b - lr * grad_b

        history.append(loss.item())

    return {"w": w, "b": b, "history": history}
```

### 3.2 확률적 경사 하강법 (SGD)

갱신마다 표본 **하나**만 쓴다. 잡음이 많지만 빠르다.

```python
def sgd(
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.01,
    n_epochs: int = 50,
) -> dict:
    n, d = X.shape
    w = torch.zeros(d, 1)
    b = torch.zeros(1)
    history = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in perm:
            xi = X[i : i + 1]
            yi = y[i : i + 1]
            y_pred = xi @ w + b
            error = y_pred - yi
            w = w - lr * 2 * (xi.T @ error)
            b = b - lr * 2 * error.squeeze()

        loss = torch.mean((y - (X @ w + b)) ** 2)
        history.append(loss.item())

    return {"w": w, "b": b, "history": history}
```

### 3.3 미니배치 경사 하강법

실용적인 선택이다. 경사의 품질과 계산량 사이에서 균형을 잡는다.

```python
def mini_batch_gd(
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 0.01,
    n_epochs: int = 100,
    batch_size: int = 32,
) -> dict:
    n, d = X.shape
    w = torch.zeros(d, 1)
    b = torch.zeros(1)
    history = []

    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        X_s, y_s = X[perm], y[perm]

        for start in range(0, n, batch_size):
            X_b = X_s[start : start + batch_size]
            y_b = y_s[start : start + batch_size]
            B = X_b.shape[0]

            y_pred = X_b @ w + b
            error = y_pred - y_b
            w = w - lr * (2.0 / B) * (X_b.T @ error)
            b = b - lr * (2.0 / B) * error.sum()

        loss = torch.mean((y - (X @ w + b)) ** 2)
        history.append(loss.item())

    return {"w": w, "b": b, "history": history}
```

### 3.4 비교

| 변형 | 경사 비용 | 갱신 잡음 | 수렴 |
|---------|---------------|--------------|-------------|
| 배치 | 단계당 $O(np)$ | 없음 | 매끄러움, 선형 속도 $O(\kappa^t)$ |
| SGD | 단계당 $O(p)$ | 큼 | 잡음이 많고, 감쇠를 쓰면 $O(1/t)$ |
| 미니배치 ($B$) | 단계당 $O(Bp)$ | 보통 | 균형 잡힘 |

---

## 4. PyTorch 구현의 네 단계

### 1단계: DataLoader와 함께 경사를 직접 계산하기

```python
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
n, p = 500, 3
X = torch.randn(n, p)
w_true = torch.tensor([2.0, -1.5, 0.5])
y = X @ w_true + 3.0 + 0.3 * torch.randn(n)

n_train = int(0.8 * n)
loader = DataLoader(
    TensorDataset(X[:n_train], y[:n_train].unsqueeze(1)),
    batch_size=32,
    shuffle=True,
)

w = torch.zeros(p, 1)
b = torch.zeros(1)
lr = 0.01

for epoch in range(100):
    for X_b, y_b in loader:
        y_pred = X_b @ w + b
        residual = y_pred - y_b
        B = len(y_b)
        w -= lr * (2.0 / B) * (X_b.T @ residual)
        b -= lr * (2.0 / B) * residual.sum()
```

### 2단계: Autograd (`requires_grad`)

```python
w = torch.zeros(p, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.01

for epoch in range(100):
    y_pred = X[:n_train] @ w + b
    loss = ((y_pred - y[:n_train]) ** 2).mean()

    loss.backward()                     # w.grad, b.grad를 채운다

    with torch.no_grad():               # 갱신이 추적되지 않게 한다
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()                      # 매우 중요: PyTorch는 누적한다
    b.grad.zero_()
```

| 장치 | 목적 |
|-----------|---------|
| `requires_grad=True` | 역방향 자동 미분을 위해 연산을 추적한다 |
| `loss.backward()` | $\partial\mathcal{L}/\partial\theta$을 계산한다 |
| `torch.no_grad()` | 제자리 갱신을 위해 경사 추적을 끈다 |
| `.grad.zero_()` | 누적된 경사를 지운다 |
| `.detach()` | 텐서를 계산 그래프에서 떼어낸다 |

!!! warning "경사 누적"
    PyTorch는 새 경사를 `.grad`에 덮어쓰지 않고 **더한다**.
    `zero_()`를 잊으면 경사가 계속 커져 발산으로 이어진다.

### 3단계: `nn.Module` + 최적화기

```python
class LinearRegression(nn.Module):
    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)

model = LinearRegression(in_features=p)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(100):
    for X_b, y_b in loader:
        y_pred = model(X_b).unsqueeze(1)
        loss = criterion(y_pred, y_b)

        optimizer.zero_grad()           # 1. 경사를 지운다
        loss.backward()                 # 2. 경사를 계산한다
        optimizer.step()                # 3. 매개변수를 갱신한다
```

순전파 → zero_grad → backward → step으로 이어지는 이 **네 줄짜리 패턴**은
로지스틱 회귀, CNN, 트랜스포머를 비롯한 모든 구조에서 그대로 쓰인다.

### 4단계: 실전 파이프라인

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LinearRegressionPipeline:
    """전체 파이프라인: 스케일링, 학습/검증 분할, 조기 종료."""

    def __init__(self, input_dim, lr=0.01, batch_size=32, patience=10):
        self.model = nn.Linear(input_dim, 1)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.batch_size = batch_size
        self.patience = patience
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X_np, y_np, n_epochs=200, val_size=0.2):
        # 스케일링
        X_train, X_val, y_train, y_val = train_test_split(
            X_np, y_np, test_size=val_size, random_state=42
        )
        Xs = torch.FloatTensor(self.scaler_X.fit_transform(X_train))
        ys = torch.FloatTensor(
            self.scaler_y.fit_transform(y_train.reshape(-1, 1))
        )
        Xv = torch.FloatTensor(self.scaler_X.transform(X_val))
        yv = torch.FloatTensor(
            self.scaler_y.transform(y_val.reshape(-1, 1))
        )

        loader = DataLoader(
            TensorDataset(Xs, ys), batch_size=self.batch_size, shuffle=True
        )

        best_val, wait, best_state = float("inf"), 0, None

        for epoch in range(n_epochs):
            # --- 학습 ---
            self.model.train()
            for xb, yb in loader:
                loss = self.criterion(self.model(xb), yb)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # --- 검증 ---
            self.model.eval()
            with torch.no_grad():
                val_loss = self.criterion(self.model(Xv), yv).item()
            self.scheduler.step(val_loss)

            if val_loss < best_val:
                best_val, wait = val_loss, 0
                best_state = self.model.state_dict().copy()
            else:
                wait += 1
                if wait >= self.patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)

    def predict(self, X_np):
        self.model.eval()
        Xs = torch.FloatTensor(self.scaler_X.transform(X_np))
        with torch.no_grad():
            y_scaled = self.model(Xs).numpy()
        return self.scaler_y.inverse_transform(y_scaled)
```

---

## 5. 수렴 분석

### 5.1 배치 GD의 선형 수렴

(선형 회귀의 MSE처럼) 볼록한 이차 함수에 대해 배치 경사 하강법은 **선형 속도**로
수렴한다.

$$
J(\boldsymbol{\theta}^{(t)}) - J(\boldsymbol{\theta}^*)
\leq \left(\frac{\kappa - 1}{\kappa + 1}\right)^{2t}
     \bigl(J(\boldsymbol{\theta}^{(0)}) - J(\boldsymbol{\theta}^*)\bigr)
$$

여기서 $\kappa = \lambda_{\max} / \lambda_{\min}$은
$\mathbf{X}^\top\mathbf{X} / n$의 **조건수**이다. $\kappa$가 크면(조건이 나쁘면)
수렴이 느려진다.

### 5.2 학습률의 영향

| 영역 | 거동 |
|--------|-----------|
| $\eta$가 너무 작음 | 수렴이 느려 많은 에폭이 필요하다 |
| $\eta$가 적절함 | 빠르고 매끄럽게 수렴한다 |
| $\eta$가 너무 큼 | 최솟값 주위에서 진동한다 |
| $\eta > 2/\lambda_{\max}$ | 발산한다 |

### 5.3 실무 지침

| 초매개변수 | 권장 사항 |
|----------------|----------------|
| 배치 크기 | 32–128 (GPU 효율을 위해 2의 거듭제곱) |
| 학습률 | 0.01이나 0.1로 시작; `ReduceLROnPlateau` 사용 |
| 에폭 | 100–500; 검증 손실로 조기 종료 |
| 최적화기 | 기본은 Adam; 세밀한 조절에는 운동량을 넣은 SGD |

---

## 6. 모델 보존

```python
# 저장 (state_dict가 권장 형식이다)
torch.save(model.state_dict(), "linear_regression.pt")

# 불러오기
model2 = LinearRegression(in_features=p)
model2.load_state_dict(
    torch.load("linear_regression.pt", map_location="cpu")
)
model2.eval()
```

!!! warning "`torch.save(model)` 대 `torch.save(model.state_dict())`"
    모델 전체를 저장하면 pickle을 사용하며 클래스 정의가 함께 묻어 들어가므로,
    코드를 리팩터링하면 깨진다. 언제나 `state_dict()`를 저장하고 불러오라.

---

## 7. 어떤 방법을 언제 쓸 것인가

| 방법 | 적합한 상황 | 복잡도 |
|--------|----------|------------|
| 정규 방정식 | $p < 10{,}000$이고 정확한 해가 필요할 때 | $O(np^2 + p^3)$ |
| 배치 GD | 중간 크기 데이터셋 | $O(knp)$ |
| 미니배치 GD | 큰 데이터셋, GPU 학습 | 단계당 $O(kBp)$ |
| SGD | 스트리밍 데이터, 아주 큰 $n$ | 단계당 $O(kp)$ |

---

## 연습문제

**연습문제 1.**
선형 회귀에 대한 경사 하강법을 구현하고 500회 반복 동안의 손실 곡선을 그려라.

??? success "연습문제 1 풀이"
    ```python
    import torch
    w = torch.zeros(d, requires_grad=True)
    lr, losses = 0.01, []
    for i in range(500):
        loss = ((X @ w - y)**2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            w.grad.zero_()
        losses.append(loss.item())
    ```

---

**연습문제 2.**
같은 회귀 문제에서 배치 GD, 미니배치 GD(배치 크기 32), SGD의 수렴을 비교하라. 세 손실 곡선을 모두 그려라.

??? success "연습문제 2 풀이"
    배치 GD는 가장 매끄러운 곡선을 내지만 에폭당 가장 느리다. SGD는 잡음이 가장 많지만 에폭당 갱신 횟수가 가장 많다. 미니배치 GD(B=32)는 잡음과 수렴 속도의 균형을 잡는다. 실제 소요 시간 기준으로는 큰 데이터셋에서 미니배치 GD가 대체로 가장 빠르게 수렴한다.

---

**연습문제 3.**
이차 손실 $L(\mathbf{w}) = \frac{1}{2}\mathbf{w}^\top\mathbf{A}\mathbf{w} - \mathbf{b}^\top\mathbf{w}$에 대한 경사 하강법의 닫힌 형태 최적 학습률을 유도하라.

??? success "연습문제 3 풀이"
    단계 $t$에서 경사는 $\mathbf{g}_t = \mathbf{A}\mathbf{w}_t - \mathbf{b}$이다. 최적 이동 폭은 $\eta$에 대해 $L(\mathbf{w}_t - \eta\mathbf{g}_t)$를 최소화한다.

    $$
    \eta^* = \frac{\mathbf{g}_t^\top\mathbf{g}_t}{\mathbf{g}_t^\top\mathbf{A}\mathbf{g}_t}
    $$

    이것이 정확한 직선 탐색을 사용하는 최급강하법이다.

---

**연습문제 4.**
PyTorch 추상화의 네 단계로 경사 하강법을 구현하라. (1) 경사 직접 계산, (2) autograd, (3) `nn.Linear`, (4) 완전한 `nn.Module`. 넷이 모두 같은 결과를 내는지 확인하라.

??? success "연습문제 4 풀이"
    4단계(가장 높은 추상화):
    ```python
    model = torch.nn.Linear(d, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    for _ in range(500):
        optimizer.zero_grad()
        criterion(model(X), y).backward()
        optimizer.step()
    ```
    네 단계 모두 같은 경사를 계산하므로 (부동소수점 차이를 제외하면) 같은 해로 수렴한다.

## 정리하며

### 정석적인 학습 루프

```
┌──────────────────────────────────────────────────────┐
│  1. 모델 정의          model = nn.Linear(p, 1)       │
│  2. 손실 정의          criterion = nn.MSELoss()      │
│  3. 최적화기 정의      optim.SGD(model.parameters()) │
│  4. 학습 루프:                                       │
│       for batch in loader:                           │
│           y_pred = model(x)                          │
│           loss = criterion(y_pred, y)                │
│           optimizer.zero_grad()                      │
│           loss.backward()                            │
│           optimizer.step()                           │
│  5. 평가               model.eval(); torch.no_grad() │
│  6. 저장               torch.save(state_dict)        │
└──────────────────────────────────────────────────────┘
```

### 핵심 정리

1. 가우스 잡음 아래에서 (상수를 제외하면) **MSE = NLL**이므로, MSE에 대한 경사
   하강법은 통계적으로 정당화된다.
2. 대부분의 응용에서 **미니배치 GD**가 실용적인 선택이다.
3. **학습률**이 가장 중요한 초매개변수이며, $2/\lambda_{\max}$으로 상한을 잡는다.
4. **Autograd**는 경사를 손으로 유도할 필요를 없앤다. 수학은 이해를 위해 여전히
   가치 있지만 계산은 PyTorch가 맡는다.
5. **네 줄짜리 패턴**(순전파 → zero_grad → backward → step)은 모든 PyTorch
   구조에 공통이다.

---

**참고 문헌**

1. Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient
   Descent."
2. Ruder, S. (2016). "An Overview of Gradient Descent Optimization Algorithms."
3. Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, Ch. 8.
4. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, Ch. 3.
