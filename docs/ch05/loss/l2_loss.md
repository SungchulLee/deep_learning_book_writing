# L2 손실 (평균 제곱 오차)

---

## 1. 학습 목표

이 절을 마치면 다음을 할 수 있게 된다.

- 가우시안 잡음 가정 아래 최대가능도 추정에서 L2 손실 유도하기
- 제곱합 오차, 평균 제곱 오차, 평균 제곱근 오차의 관계 정리하기
- L2 손실의 경사가 왜 예측에서 목표를 뺀 것에 비례하는지 보이기
- 교차 엔트로피의 경사 $\hat{\boldsymbol{\pi}} - \mathbf{y}$와 같은 꼴이 나오는 까닭 설명하기
- L2 손실을 최소로 만드는 일이 왜 직교 사영인지 기하로 이해하기
- **손실**로서의 L2와 **정칙화 항**으로서의 L2를 구분하기
- `nn.MSELoss`의 세 가지 `reduction`을 손으로 만든 구현과 맞추어 확인하기

!!! note "함께 볼 것"
    이 절은 회귀에서 가장 기본이 되는 손실 하나를 최대가능도에서 끝까지 유도한다.
    여러 회귀 손실을 나란히 견주어 보는 것은 [7장의 회귀 손실의 비교](../../ch07/loss/02_regression_losses_comparison.md)이고,
    이상치에 견디는 절충안은 [후버 손실](../../ch07/loss/huber_loss.md)이다.
    이름이 비슷하지만 하는 일이 전혀 다른 **L2 정칙화**는 [6장](../../ch06/regularization/l2_regularization.md)에서 다룬다.
    분류 쪽의 짝이 되는 손실은 같은 절의 [교차 엔트로피 손실](cross_entropy.md)이다.

---

## 2. 최대가능도의 틀

### 문제 설정

목표값이 실수인 회귀에서는 다음을 갖는다.

- **데이터:** $y^{(i)} \in \mathbb{R}$인 $\mathcal{D} = \{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^{N}$
- **모델:** 값 $\hat{y}^{(i)} = f(\mathbf{x}^{(i)};\, \boldsymbol{\theta})$을 예측한다
- **목표:** 관측된 데이터의 가능도를 최대로 만드는 매개변수 $\boldsymbol{\theta}$을 찾는다

분류에서는 모델이 확률을 곧바로 내놓았으므로 가능도를 적는 일이 자연스러웠다. 회귀에서 모델이 내놓는 것은 실수 하나뿐이므로, 가능도를 적으려면 **잡음이 어떻게 섞이는지**를 먼저 말해야 한다.

### 가우시안 잡음 가정

참값 둘레에 평균이 0이고 분산이 $\sigma^2$인 정규 잡음이 얹힌다고 놓는다.

$$y^{(i)} = f(\mathbf{x}^{(i)};\, \boldsymbol{\theta}) + \varepsilon^{(i)}, \qquad \varepsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$$

그러면 하나의 관측에 대한 조건부 밀도는 다음과 같다.

$$p\bigl(y^{(i)} \mid \mathbf{x}^{(i)};\, \boldsymbol{\theta}\bigr) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{\bigl(y^{(i)} - \hat{y}^{(i)}\bigr)^2}{2\sigma^2}\right)$$

### 가능도 함수

표본들이 독립이라고 가정하면 가능도는 곱이 된다.

$$\mathcal{L}(\boldsymbol{\theta}) = \prod_{i=1}^{N} p\bigl(y^{(i)} \mid \mathbf{x}^{(i)};\, \boldsymbol{\theta}\bigr) = \prod_{i=1}^{N} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{\bigl(y^{(i)} - \hat{y}^{(i)}\bigr)^2}{2\sigma^2}\right)$$

### 로그가능도

로그를 취하면 곱이 합으로 풀린다.

$$\ell(\boldsymbol{\theta}) = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{N} \bigl(y^{(i)} - \hat{y}^{(i)}\bigr)^2$$

### 음의 로그가능도

부호를 뒤집어 최소화할 대상으로 바꾼다.

$$\text{NLL}(\boldsymbol{\theta}) = \frac{N}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2} \sum_{i=1}^{N} \bigl(y^{(i)} - \hat{y}^{(i)}\bigr)^2$$

여기서 $\boldsymbol{\theta}$에 **딸린 항은 오직 뒤의 제곱합뿐이다.** 앞의 $\frac{N}{2}\log(2\pi\sigma^2)$은 상수이고, $\frac{1}{2\sigma^2}$은 양의 배수라 최솟값의 자리를 옮기지 못한다. 따라서 다음이 성립한다.

$$\arg\min_{\boldsymbol{\theta}} \text{NLL}(\boldsymbol{\theta}) = \arg\min_{\boldsymbol{\theta}} \sum_{i=1}^{N} \bigl(y^{(i)} - \hat{y}^{(i)}\bigr)^2$$

!!! tip "이 절의 요점"
    제곱 오차는 누가 정해 준 규칙이 아니다. **잡음이 정규분포라고 가정하는 순간 따라 나오는 결론**이다.
    교차 엔트로피가 범주분포 가정에서 따라 나온 것과 똑같은 자리에 있다.

---

## 3. L2 손실의 여러 이름

같은 양을 어떻게 묶느냐에 따라 이름이 갈린다.

| 이름 | 식 | 쓰임새 |
|---|---|---|
| 제곱합 오차(SSE) | $\sum_{i} (y^{(i)} - \hat{y}^{(i)})^2$ | 유도할 때. 표본 수에 따라 값이 커진다 |
| 평균 제곱 오차(MSE) | $\frac{1}{N}\sum_{i} (y^{(i)} - \hat{y}^{(i)})^2$ | 학습할 때. 배치 크기에 흔들리지 않는다 |
| 평균 제곱근 오차(RMSE) | $\sqrt{\text{MSE}}$ | 보고할 때. 단위가 $y$와 같아진다 |

벡터 표기로 $\mathbf{r} = \mathbf{y} - \hat{\mathbf{y}}$을 잔차라 하면 제곱합 오차는 잔차의 **L2 노름의 제곱**이다.

$$\text{SSE} = \|\mathbf{y} - \hat{\mathbf{y}}\|_2^2$$

'L2 손실'이라는 이름이 여기서 왔다.

!!! warning "L2 손실과 L2 정칙화는 다른 것이다"
    이름이 같은 것은 둘 다 L2 노름의 제곱을 쓰기 때문이지만, **무엇의 노름인지**가 다르다.

    - **L2 손실**: $\|\mathbf{y} - \hat{\mathbf{y}}\|_2^2$ — **잔차**의 노름. 데이터에 얼마나 맞는지를 잰다
    - **L2 정칙화**: $\lambda\|\boldsymbol{\theta}\|_2^2$ — **매개변수**의 노름. 모델이 얼마나 큰지를 벌한다

    둘은 함께 쓰이며 서로를 대신하지 못한다. 자세한 것은 [6장](../../ch06/regularization/l2_regularization.md)에 있다.

---

## 4. 경사 유도

### 예측에 대한 경사

$\mathcal{L} = \frac{1}{N}\sum_{i}(\hat{y}^{(i)} - y^{(i)})^2$을 예측 하나로 미분한다.

$$\frac{\partial \mathcal{L}}{\partial \hat{y}^{(i)}} = \frac{2}{N}\bigl(\hat{y}^{(i)} - y^{(i)}\bigr)$$

벡터로 묶으면 다음과 같다.

$$\nabla_{\hat{\mathbf{y}}} \mathcal{L} = \frac{2}{N}\bigl(\hat{\mathbf{y}} - \mathbf{y}\bigr)$$

### 교차 엔트로피와 같은 꼴인 까닭

앞 절에서 소프트맥스 회귀의 경사가 $\nabla_{\mathbf{z}}\mathcal{L} = \hat{\boldsymbol{\pi}} - \mathbf{y}$이었다. 여기서도 상수배를 빼면 **예측 빼기 목표**이다. 우연이 아니다.

두 손실 모두 지수족 분포의 음의 로그가능도이고, 둘 다 그 분포의 **자연스러운 연결 함수**를 쓰고 있다. 회귀에서는 항등 함수, 분류에서는 소프트맥스이다. 이 짝이 맞을 때 출력 앞단 $\mathbf{z}$에 대한 경사는 언제나 다음 꼴이 된다.

$$\nabla_{\mathbf{z}}\mathcal{L} \;\propto\; (\text{예측} - \text{목표})$$

역전파를 처음 배울 때 출력층 경사가 늘 이렇게 단순한 것이 이 때문이다.

### 선형 모델에서 매개변수에 대한 경사

$\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\theta}$일 때 연쇄 법칙을 이어 붙이면 다음과 같다.

$$\nabla_{\boldsymbol{\theta}} \mathcal{L} = \frac{2}{N}\mathbf{X}^\top\bigl(\mathbf{X}\boldsymbol{\theta} - \mathbf{y}\bigr)$$

---

## 5. 닫힌 해와 기하

### 정규 방정식

경사를 0으로 놓으면 $\mathbf{X}^\top\mathbf{X}\boldsymbol{\theta} = \mathbf{X}^\top\mathbf{y}$이고, $\mathbf{X}^\top\mathbf{X}$이 가역이면 다음을 얻는다.

$$\boldsymbol{\theta}^* = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$$

L2 손실이 특별한 자리를 차지하는 큰 이유가 이것이다. 선형 모델에서는 **되풀이 없이 한 번에** 답이 나온다. 교차 엔트로피에는 이런 닫힌 해가 없다.

### 직교 사영

$\boldsymbol{\theta}^*$에서 잔차는 $\mathbf{X}$의 모든 열과 직교한다.

$$\mathbf{X}^\top(\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}$$

즉 $\hat{\mathbf{y}}$은 $\mathbf{y}$을 $\mathbf{X}$의 열공간 위로 **수직으로 내린 그림자**이다. 손실을 최소로 만든다는 것과 수직으로 내린다는 것이 같은 말이 된다.

```
        y
        │╲
        │  ╲  잔차 r = y - ŷ  (열공간과 직교)
        │    ╲
   ─────┴──────╲──────────  X의 열공간
              ŷ = Xθ*
```

---

## 6. 이상치에 약한 까닭

오차가 **제곱**되므로, 오차가 10배인 점 하나가 손실에 100배로 기여한다. 이는 가우시안 가정에서 곧장 따라 나온 성질이다. 정규분포는 꼬리가 얇아 큰 오차를 매우 드문 일로 보고, 그런 점을 맞추려 모델을 크게 끌어당긴다.

데이터에 이상치가 섞여 있다면 잡음이 정규분포라는 가정 자체가 어긋난 것이므로, 손실을 바꾸는 편이 옳다. 절대 오차(L1)는 라플라스 잡음 가정에, 후버 손실은 그 사이의 절충에 해당한다.

---

## 7. PyTorch 구현

### 손으로 만든 구현과 맞추어 보기

```python
import torch
import torch.nn as nn

torch.manual_seed(0)

y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])
y_pred = torch.tensor([2.5,  0.0, 2.0, 8.0])

residual = y_pred - y_true

# reduction='mean'(기본값): 제곱 오차의 평균 = MSE
mse_manual = (residual ** 2).mean()
mse_torch = nn.MSELoss()(y_pred, y_true)

# reduction='sum': 제곱합 오차 = SSE. 유도할 때 쓰는 꼴이다
sse_manual = (residual ** 2).sum()
sse_torch = nn.MSELoss(reduction='sum')(y_pred, y_true)

# reduction='none': 줄이지 않고 표본별 오차를 그대로 돌려준다
none_torch = nn.MSELoss(reduction='none')(y_pred, y_true)

print(f"잔차        : {residual}")
print(f"MSE  손수 계산: {mse_manual:.4f} | nn.MSELoss: {mse_torch:.4f}")
print(f"SSE  손수 계산: {sse_manual:.4f} | reduction='sum': {sse_torch:.4f}")
print(f"표본별 제곱 오차: {none_torch}")
print(f"RMSE          : {mse_torch.sqrt():.4f}  (단위가 y와 같다)")

# SSE = N × MSE 임을 확인한다
print(f"SSE == N * MSE : {torch.allclose(sse_torch, len(y_true) * mse_torch)}")
```

**출력:**

```
잔차        : tensor([-0.5000,  0.5000,  0.0000,  1.0000])
MSE  손수 계산: 0.3750 | nn.MSELoss: 0.3750
SSE  손수 계산: 1.5000 | reduction='sum': 1.5000
표본별 제곱 오차: tensor([0.2500, 0.2500, 0.0000, 1.0000])
RMSE          : 0.6124  (단위가 y와 같다)
SSE == N * MSE : True
```

### 경사가 정말 예측 빼기 목표인지 확인하기

```python
import torch
import torch.nn as nn

y_true = torch.tensor([3.0, -0.5, 2.0, 7.0])
y_pred = torch.tensor([2.5,  0.0, 2.0, 8.0], requires_grad=True)

loss = nn.MSELoss()(y_pred, y_true)
loss.backward()

# 유도한 식: dL/dŷ = (2/N)(ŷ - y)
N = len(y_true)
analytic = (2 / N) * (y_pred.detach() - y_true)

print(f"autograd 경사: {y_pred.grad}")
print(f"손으로 유도한 값: {analytic}")
print(f"일치: {torch.allclose(y_pred.grad, analytic)}")
```

**출력:**

```
autograd 경사: tensor([-0.2500,  0.2500,  0.0000,  0.5000])
손으로 유도한 값: tensor([-0.2500,  0.2500,  0.0000,  0.5000])
일치: True
```

### 가우시안 음의 로그가능도와 같은 것임을 수치로 확인하기

앞에서 NLL과 SSE가 상수와 양의 배수만큼만 차이 난다고 보였다. 실제로 그런지 확인한다.

```python
import math
import torch
import torch.nn as nn

torch.manual_seed(0)

y_true = torch.randn(200)
y_pred = y_true + 0.3 * torch.randn(200)   # 잡음이 섞인 예측
sigma = 0.3
N = len(y_true)

# 가우시안 NLL을 정의 그대로 계산한다
sse = ((y_pred - y_true) ** 2).sum()
nll = 0.5 * N * math.log(2 * math.pi * sigma ** 2) + sse / (2 * sigma ** 2)

# PyTorch가 제공하는 가우시안 NLL과 비교한다
var = torch.full_like(y_pred, sigma ** 2)
nll_torch = nn.GaussianNLLLoss(reduction='sum', full=True)(y_pred, y_true, var)

print(f"직접 계산한 NLL   : {nll:.4f}")
print(f"nn.GaussianNLLLoss: {nll_torch:.4f}")
print(f"일치: {torch.allclose(nll, nll_torch)}")

# 상수를 걷어내면 남는 것은 SSE뿐이다
const = 0.5 * N * math.log(2 * math.pi * sigma ** 2)
print(f"\nNLL에서 상수를 뺀 값 x 2σ² : {((nll - const) * 2 * sigma ** 2):.4f}")
print(f"SSE                        : {sse:.4f}")
```

**출력:**

```
직접 계산한 NLL   : 56.1714
nn.GaussianNLLLoss: 56.1714
일치: True

NLL에서 상수를 뺀 값 x 2σ² : 20.3721
SSE                        : 20.3721
```

### 닫힌 해와 경사 하강법이 같은 곳에 이르는지 보기

```python
import torch

torch.manual_seed(0)

# y = 2x + 1 에 잡음을 섞은 데이터를 만든다
N = 100
x = torch.randn(N, 1)
X = torch.cat([x, torch.ones(N, 1)], dim=1)     # 편향을 열로 붙인다
theta_true = torch.tensor([[2.0], [1.0]])
y = X @ theta_true + 0.1 * torch.randn(N, 1)

# 방법 1: 정규 방정식 — 되풀이 없이 한 번에 푼다
theta_closed = torch.linalg.solve(X.T @ X, X.T @ y)

# 방법 2: L2 손실에 경사 하강법을 돌린다
theta_gd = torch.zeros(2, 1, requires_grad=True)
optimizer = torch.optim.SGD([theta_gd], lr=0.1)
for step in range(500):
    loss = ((X @ theta_gd - y) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"참값        : {theta_true.flatten().tolist()}")
print(f"정규 방정식 : {[round(v, 4) for v in theta_closed.flatten().tolist()]}")
print(f"경사 하강법 : {[round(v, 4) for v in theta_gd.detach().flatten().tolist()]}")
print(f"두 해가 같은가: {torch.allclose(theta_closed, theta_gd.detach(), atol=1e-3)}")

# 최적해에서 잔차가 X의 열들과 직교하는지 확인한다
residual = y - X @ theta_closed
print(f"\nX^T r (0에 가까워야 한다): {(X.T @ residual).flatten().tolist()}")
```

**출력:**

```
참값        : [2.0, 1.0]
정규 방정식 : [1.9881, 1.016]
경사 하강법 : [1.9881, 1.016]
두 해가 같은가: True

X^T r (0에 가까워야 한다): [-6.654696335317567e-05, -1.3381242752075195e-05]
```

### 이상치 하나가 손실을 어떻게 흔드는지

```python
import torch
import torch.nn as nn

y_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = torch.tensor([1.1, 2.1, 2.9, 4.1, 5.1])   # 고르게 조금씩 빗나갔다

mse_clean = nn.MSELoss()(y_pred, y_true)
mae_clean = nn.L1Loss()(y_pred, y_true)

# 마지막 한 점만 크게 빗나가게 만든다
y_pred_outlier = y_pred.clone()
y_pred_outlier[-1] = 15.0

mse_dirty = nn.MSELoss()(y_pred_outlier, y_true)
mae_dirty = nn.L1Loss()(y_pred_outlier, y_true)
huber_dirty = nn.HuberLoss(delta=1.0)(y_pred_outlier, y_true)

print(f"이상치 없음 -> MSE {mse_clean:.4f} | MAE {mae_clean:.4f}")
print(f"이상치 하나 -> MSE {mse_dirty:.4f} | MAE {mae_dirty:.4f}")
print(f"MSE는 {mse_dirty / mse_clean:.1f}배로 뛰고, MAE는 {mae_dirty / mae_clean:.1f}배에 그친다")
print(f"후버 손실(delta=1.0): {huber_dirty:.4f}  <- 둘 사이에 놓인다")
```

**출력:**

```
이상치 없음 -> MSE 0.0100 | MAE 0.1000
이상치 하나 -> MSE 20.0080 | MAE 2.0800
MSE는 2000.8배로 뛰고, MAE는 20.8배에 그친다
후버 손실(delta=1.0): 1.9040  <- 둘 사이에 놓인다
```

---

## 연습문제

**연습문제 1.**
잡음의 분산 $\sigma^2$을 모른다고 하자. 로그가능도를 $\sigma^2$으로 미분하여 최대가능도 추정값 $\hat{\sigma}^2$을 구하고, 그것이 최적점에서의 평균 제곱 오차와 같음을 보여라.

??? success "연습문제 1 풀이"
    로그가능도는 다음과 같다.

    $$\ell = -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (y^{(i)} - \hat{y}^{(i)})^2$$

    $s = \sigma^2$으로 두고 미분하면 다음과 같다.

    $$\frac{\partial \ell}{\partial s} = -\frac{N}{2s} + \frac{1}{2s^2}\sum_i (y^{(i)} - \hat{y}^{(i)})^2$$

    0으로 놓고 $s$에 대해 풀면 다음을 얻는다.

    $$\hat{\sigma}^2 = \frac{1}{N}\sum_i (y^{(i)} - \hat{y}^{(i)})^2 = \text{MSE}$$

    즉 잡음의 분산을 최대가능도로 추정한 값이 곧 평균 제곱 오차이다. MSE는 손실이면서 동시에 **남은 잡음의 크기 추정값**이기도 하다.

---

**연습문제 2.**
`reduction='mean'`과 `reduction='sum'`은 최솟값의 위치를 바꾸지 않는다. 그런데도 학습 결과가 달라질 수 있다. 왜 그런가?

??? success "연습문제 2 풀이"
    두 손실은 $N$배만큼만 차이 나므로 최솟값의 **위치**는 같다. 그러나 경사의 **크기**도 $N$배 차이가 난다.

    학습률이 고정되어 있으면 `sum`을 쓸 때 한 걸음의 폭이 $N$배가 되어, 배치 크기를 바꾸는 것만으로 실제 보폭이 달라진다. 배치 크기가 커질수록 발산하기 쉬워진다.

    `mean`이 기본값인 까닭이 이것이다. 배치 크기와 보폭을 서로 떼어 놓아 준다.

---

**연습문제 3.**
$\mathbf{X}^\top\mathbf{X}$이 가역이 아니면 정규 방정식을 쓸 수 없다. 어떤 때 그런 일이 생기며, 어떻게 다룰 수 있는가?

??? success "연습문제 3 풀이"
    특징이 표본보다 많거나($d > N$) 특징들 사이에 완전한 선형 종속이 있으면 $\mathbf{X}^\top\mathbf{X}$이 특이해진다. 이때는 손실을 최소로 만드는 $\boldsymbol{\theta}$이 하나로 정해지지 않는다.

    L2 정칙화를 더하면 해결된다. 능선 회귀의 해는 다음과 같다.

    $$\boldsymbol{\theta}^* = (\mathbf{X}^\top\mathbf{X} + \lambda \mathbf{I})^{-1}\mathbf{X}^\top\mathbf{y}$$

    $\lambda > 0$이면 $\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}$의 고윳값이 모두 $\lambda$ 이상이 되어 반드시 가역이 된다. 손실로서의 L2와 정칙화로서의 L2가 한 식 안에서 각자 다른 일을 하는 예이다.

---

**연습문제 4.**
목표값의 단위를 미터에서 밀리미터로 바꾸면 MSE 값은 어떻게 달라지는가? RMSE는 어떠한가? 학습된 매개변수는 달라지는가?

??? success "연습문제 4 풀이"
    $y$을 1000배 하면 잔차도 1000배가 되므로 MSE는 $1000^2 = 10^6$배가 되고, RMSE는 1000배가 된다. RMSE가 보고용으로 선호되는 까닭이 이것이다. 목표값과 단위가 같아 값의 크기를 그대로 읽을 수 있다.

    선형 모델의 매개변수는 그에 맞추어 1000배가 되지만, 예측값을 다시 미터로 되돌리면 원래와 같은 모델이다. 다만 학습률을 그대로 두면 경사도 함께 커지므로 실제 학습은 불안정해질 수 있다. 목표값을 고르게 하는(정규화하는) 습관이 권장되는 이유이다.

## 정리하며

**다룬 것** — 가우시안 잡음 가정에서 유도한 L2 손실

### 근본이 되는 식들

| 이름 | 식 |
|---|---|
| 가우시안 NLL | $\frac{N}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_i (y^{(i)} - \hat{y}^{(i)})^2$ |
| L2 손실(MSE) | $\frac{1}{N}\sum_i (y^{(i)} - \hat{y}^{(i)})^2$ |
| 예측에 대한 경사 | $\nabla_{\hat{\mathbf{y}}}\mathcal{L} = \frac{2}{N}(\hat{\mathbf{y}} - \mathbf{y})$ |
| 매개변수에 대한 경사(선형) | $\nabla_{\boldsymbol{\theta}}\mathcal{L} = \frac{2}{N}\mathbf{X}^\top(\mathbf{X}\boldsymbol{\theta} - \mathbf{y})$ |
| 닫힌 해 | $\boldsymbol{\theta}^* = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ |

### 핵심 통찰

제곱 오차는 약속이 아니라 **가정의 결과**이다. 잡음이 정규분포라고 놓는 순간 L2 손실이 따라 나온다. 이상치에 약한 것도 같은 가정에서 따라 나오는 성질이지, 고쳐 써야 할 흠이 아니다. 데이터가 그 가정에 맞지 않으면 손실을 바꾸는 것이 옳다.

### 두 손실을 나란히 놓고 보기

| | L2 손실 | 교차 엔트로피 |
|---|---|---|
| 가정하는 분포 | 정규분포 | 범주분포 |
| 연결 함수 | 항등 | 소프트맥스 |
| 출력 앞단 경사 | $\hat{\mathbf{y}} - \mathbf{y}$에 비례 | $\hat{\boldsymbol{\pi}} - \mathbf{y}$ |
| 닫힌 해 | 있다(선형 모델) | 없다 |

두 손실은 서로 다른 규칙이 아니라, **같은 최대가능도 원리를 서로 다른 잡음 가정에 적용한 결과**이다.

앞의 연습문제 4개로 직접 확인할 수 있다.
