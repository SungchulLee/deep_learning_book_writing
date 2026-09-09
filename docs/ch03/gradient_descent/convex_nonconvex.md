# 볼록 최적화와 비볼록 최적화

**볼록** 최적화와 **비볼록** 최적화의 구분은 경사 하강법이 언제 성공하고 어떤 어려움을 마주할지 이해하는 데 근본이 된다. 이 절에서는 이 개념들과 기계학습에서의 함의, 그리고 이론적인 어려움에도 불구하고 비볼록 최적화가 딥러닝에서 놀랍도록 잘 동작하는 이유를 살펴본다.

---

## 1. 볼록성: 정의

### 볼록 집합

집합 $\mathcal{C} \subseteq \mathbb{R}^n$이 **볼록** 이라는 것은 임의의 두 점 $\mathbf{x}, \mathbf{y} \in \mathcal{C}$와 임의의 $\lambda \in [0, 1]$에 대해 다음이 성립한다는 뜻이다.

$$\lambda \mathbf{x} + (1-\lambda)\mathbf{y} \in \mathcal{C}$$

**직관**: 집합 안의 임의의 두 점을 잇는 선분이 온전히 그 집합 안에 놓인다.

```
Convex sets:                 Non-convex sets:
    _____                        ╱╲
   ╱     ╲                      ╱  ╲___
  │       │                    │      ╲
  │   •───•                    │   •   │
  │       │                     ╲__│___╱
   ╲_____╱                         │
                                   •
```

### 볼록 함수

함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$이 **볼록** 이라는 것은 임의의 $\mathbf{x}, \mathbf{y}$와 $\lambda \in [0, 1]$에 대해 다음이 성립한다는 뜻이다.

$$f(\lambda\mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

**직관**: 그래프 위의 임의의 두 점을 잇는 현이 그래프 위쪽에(또는 그래프 위에) 놓인다.

```
Convex function:              Non-convex function:
       │                             │
       │    ╱                        │   ╱╲
       │   ╱                         │  ╱  ╲
       │  ╱                          │ ╱    ╲__╱╲
       │_╱                           │╱          ╲
       └────────                     └────────────
```

### 강볼록 함수

$\mathbf{x} \neq \mathbf{y}$이고 $\lambda \in (0, 1)$일 때 부등식이 엄격하게 성립하면 그 함수는 **강볼록** 이다.

$$f(\lambda\mathbf{x} + (1-\lambda)\mathbf{y}) < \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

### 동치인 조건

두 번 미분 가능한 함수에 대해 볼록성은 다음으로 확인할 수 있다.

1. **1차 조건**(접선이 함수 아래에 놓인다):

   $$f(\mathbf{y}) \geq f(\mathbf{x}) + \nabla f(\mathbf{x})^T(\mathbf{y} - \mathbf{x})$$

2. **2차 조건**(헤세 행렬이 양의 준정부호이다):

   $$\nabla^2 f(\mathbf{x}) \succeq 0 \quad \forall \mathbf{x}$$

---

## 2. 볼록 함수의 예

### 흔한 볼록 함수

| 함수 | 식 | 정의역 |
|----------|---------|--------|
| 일차 | $f(\mathbf{x}) = \mathbf{a}^T\mathbf{x} + b$ | $\mathbb{R}^n$ |
| 이차(양의 준정부호) | $f(\mathbf{x}) = \mathbf{x}^T\mathbf{A}\mathbf{x}$ ($\mathbf{A} \succeq 0$) | $\mathbb{R}^n$ |
| 지수 | $f(x) = e^{ax}$ | $\mathbb{R}$ |
| 음의 로그 | $f(x) = -\log(x)$ | $\mathbb{R}_{++}$ |
| 거듭제곱 | $f(x) = x^p$ ($p \geq 1$ 또는 $p \leq 0$) | $\mathbb{R}_+$ |
| 노름 | $f(\mathbf{x}) = \|\mathbf{x}\|_p$ | $\mathbb{R}^n$ |
| Log-sum-exp | $f(\mathbf{x}) = \log(\sum e^{x_i})$ | $\mathbb{R}^n$ |

### 기계학습에서의 예

**MSE 손실(선형 회귀):**

$$L(\mathbf{w}) = \frac{1}{N}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$$

이는 $\mathbf{w}$에 대해 볼록이다(헤세 행렬이 양의 준정부호인 이차식이다).

**로지스틱 회귀 손실:**

$$L(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^N \log(1 + e^{-y_i \mathbf{w}^T\mathbf{x}_i})$$

이는 $\mathbf{w}$에 대해 볼록이다.

---

## 3. 볼록 최적화의 성질

### 전역 최적성

!!! tip "볼록 함수의 근본적인 성질"
    볼록 함수 $f$에 대해 다음이 성립한다.
    
    - 모든 **국소 최솟값** 이 **전역 최솟값** 이다
    - $f$가 강볼록이면 전역 최솟값은 (존재한다면) **유일하다**

볼록 최적화가 "쉽다"고 여겨지는 이유가 이것이다. 최적이 아닌 국소 최솟값에 갇힐 걱정을 하지 않아도 된다.

### 경사 하강법의 수렴

립시츠 연속인 경사를 가진 볼록 함수에 대해 다음이 성립한다.

$$f(\mathbf{x}_t) - f(\mathbf{x}^*) \leq \frac{\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{2\eta t}$$

이는 $O(1/t)$의 속도로 수렴함을 보장한다.

(헤세 행렬의 고윳값이 아래로 유계인) **강볼록** 함수에 대해서는 다음이 성립한다.

$$f(\mathbf{x}_t) - f(\mathbf{x}^*) \leq \left(1 - \frac{\mu}{L}\right)^t (f(\mathbf{x}_0) - f(\mathbf{x}^*))$$

이는 (지수적으로 빠른) **선형 수렴** 을 준다.

---

## 4. 비볼록 최적화

### 무엇이 함수를 비볼록으로 만드는가?

어느 한 곳에서라도 볼록성 조건이 깨지면 그 함수는 비볼록이다. 흔한 원인은 다음과 같다.

1. **여러 개의 국소 최솟값**: 함수에 높이가 다른 골짜기들이 있다
2. **안장점**: 어떤 방향으로는 볼록이고 다른 방향으로는 오목한 영역
3. **비볼록 제약**: 실현 가능 영역이 비볼록이다

### 신경망의 손실 함수

신경망의 손실 함수는 다음 이유로 **매우 비볼록** 이다.

1. **비선형 활성화**: ReLU, 시그모이드, tanh
2. **가중치 대칭성**: 뉴런의 순서를 바꾸어도 함수가 변하지 않는다
3. **크기 조정의 모호성**: $w_1 \cdot a \cdot w_2 = (cw_1) \cdot a \cdot (w_2/c)$

```python
# 간단한 비볼록 손실 지형
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def neural_loss(w1, w2, x=1, y=2):
    """y = w2 * ReLU(w1 * x)의 손실"""
    hidden = np.maximum(0, w1 * x)  # ReLU
    pred = w2 * hidden
    return (pred - y) ** 2

w1 = np.linspace(-3, 3, 100)
w2 = np.linspace(-3, 3, 100)
W1, W2 = np.meshgrid(w1, w2)
Z = np.vectorize(neural_loss)(W1, W2)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(W1, W2, Z, cmap='viridis', alpha=0.7)
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_zlabel('Loss')
ax.set_title('Non-Convex Neural Network Loss Surface')
plt.show()
```

### 비볼록 최적화의 어려움

1. **전역 최적성 보장이 없다**: 경사 하강법은 국소 최솟값을 찾을 뿐 전역 최솟값이라는 보장이 없다
2. **안장점**: 최적화를 크게 늦출 수 있다
3. **초기화에 민감하다**: 출발점이 다르면 다른 해에 이른다
4. **고원 영역**: 경사가 0에 가까운 평평한 영역

---

## 5. 딥러닝은 왜 잘 동작하는가?

비볼록성에도 불구하고 딥러닝은 놀랍도록 잘 성공한다. 여기에는 몇 가지 요인이 있다.

### 1. 손실 지형의 구조

연구에 따르면 신경망의 손실 지형은 유리한 성질을 가진다.

- **동등한 최솟값이 많다**: 대칭성 때문에 많은 국소 최솟값이 비슷한 손실을 달성한다
- **연결된 골짜기**: 좋은 해들이 연결된 영역을 이룬다
- **고차원의 이점**: 고차원에서는 국소 최솟값보다 안장점이 훨씬 많다

### 2. 과매개변수화

신경망의 매개변수가 데이터 점보다 훨씬 많으면 다음과 같다.

- 보간이 쉬워진다(해가 많이 존재한다)
- 최적화 경로가 "넓게" 유지된다
- 암묵적 정칙화가 좋은 해를 골라낸다

### 3. 정칙화 역할을 하는 SGD

확률적 경사 하강법은 다음과 같이 돕는다.

- 뾰족한 최솟값에서 벗어나게 하는 잡음을 더한다
- 일반화가 더 잘되는 평평한 최솟값을 찾는다
- 암묵적 정칙화를 제공한다

### 4. 좋은 초기화

적절한 초기화(Xavier, He 등)는 유리한 영역에서 출발하게 해 준다.

```python
# ReLU 신경망을 위한 He 초기화
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# tanh 신경망을 위한 Xavier 초기화
nn.init.xavier_normal_(layer.weight)
```

---

## 6. 볼록성 시각화하기

### 1차원 비교

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 4, 200)

# 볼록: 이차함수
convex = (x - 1) ** 2

# 비볼록: 여러 개의 최솟값
non_convex = x**4 - 4*x**2 + x

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(x, convex, 'b-', linewidth=2)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.scatter([1], [0], color='red', s=100, zorder=5, label='Global min')
ax1.set_title('Convex: One Global Minimum')
ax1.set_xlabel('x')
ax1.set_ylabel('f(x)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(x, non_convex, 'b-', linewidth=2)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
# 국소 최솟값 표시
ax2.scatter([-1.35, 1.46], [non_convex[np.argmin(np.abs(x+1.35))], 
                            non_convex[np.argmin(np.abs(x-1.46))]], 
            color='red', s=100, zorder=5, label='Local minima')
ax2.set_title('Non-Convex: Multiple Local Minima')
ax2.set_xlabel('x')
ax2.set_ylabel('f(x)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 2차원 등고선 비교

```python
# 볼록: 타원체
def convex_2d(x, y):
    return x**2 + 2*y**2

# 비볼록: 라스트리긴 함수
def rastrigin(x, y):
    return 20 + x**2 + y**2 - 10*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y))

x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 볼록
Z1 = convex_2d(X, Y)
ax1.contour(X, Y, Z1, levels=20)
ax1.set_title('Convex Function')
ax1.set_xlabel('x')
ax1.set_ylabel('y')

# 비볼록
Z2 = rastrigin(X, Y)
ax2.contour(X, Y, Z2, levels=30)
ax2.set_title('Non-Convex Function (Rastrigin)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')

plt.tight_layout()
plt.show()
```

---

## 7. 볼록성 확인하기

### 헤세 행렬 분석

두 번 미분 가능한 함수에 대해 다음과 같이 한다.

```python
import torch

def check_convexity_numerical(f, x, epsilon=1e-3):
    """
    헤세 고윳값으로 그 자리 볼록함을 살핀다
    """
    n = len(x)
    H = torch.zeros((n, n))
    
    x = x.detach().requires_grad_(True)
    
    # 헤세 행렬 계산
    for i in range(n):
        # 경사
        y = f(x)
        grad = torch.autograd.grad(y, x, create_graph=True)[0]
        
        # 이계 도함수
        for j in range(n):
            H[i, j] = torch.autograd.grad(
                grad[i], x, retain_graph=True
            )[0][j]
    
    # 고윳값 확인
    eigenvalues = torch.linalg.eigvalsh(H)
    
    if torch.all(eigenvalues >= -epsilon):
        return "Locally convex (H ≽ 0)"
    elif torch.all(eigenvalues <= epsilon):
        return "Locally concave (H ≼ 0)"
    else:
        return "Non-convex (indefinite H)"
```

### 볼록성을 보존하는 연산

다음 연산에서 볼록성이 보존된다.

1. **음이 아닌 가중합**: $\alpha_i \geq 0$일 때 $\sum \alpha_i f_i$
2. **아핀 함수와의 합성**: $f(\mathbf{Ax} + \mathbf{b})$
3. **점별 최댓값**: $\max(f_1, f_2, \ldots, f_n)$
4. **부분 최소화**: $g(\mathbf{x}) = \min_{\mathbf{y}} f(\mathbf{x}, \mathbf{y})$

---

## 8. 실무적 함의

### 볼록 문제에서

- 전역 최솟값으로의 **수렴이 보장된다**
- 초기화에 **덜 민감하다**
- **표준 최적화기** 를 쓴다. SGD와 Adam이 잘 동작한다
- **닫힌 형태의 해** 가 존재할 수 있다(예: 선형 회귀)

### 비볼록 문제에서(딥러닝)

- **여러 번 다시 시작한다**: 서로 다른 초기화를 시도한다
- **학습률 워밍업**: 초기 탐색을 조심스럽게 한다
- **정칙화**: 일반화되는 최솟값을 찾는 데 도움이 된다
- **학습을 관찰한다**: 발산과 진동을 살핀다
- **적응적 최적화기를 쓴다**: Adam이 SGD보다 강건한 경우가 많다

### 코드: 두 경우의 최적화 비교

```python
import torch
import torch.nn as nn

# 볼록: 선형 회귀
def train_convex():
    X = torch.randn(100, 10)
    y = X @ torch.randn(10, 1) + 0.1 * torch.randn(100, 1)
    
    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        loss = nn.MSELoss()(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.item()

# 비볼록: 신경망
def train_nonconvex():
    X = torch.randn(100, 10)
    y = torch.sin(X.sum(dim=1, keepdim=True))
    
    model = nn.Sequential(
        nn.Linear(10, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(100):
        loss = nn.MSELoss()(model(X), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return loss.item()

# 분산을 보기 위해 여러 번 실행
convex_results = [train_convex() for _ in range(10)]
nonconvex_results = [train_nonconvex() for _ in range(10)]

print(f"Convex: mean={np.mean(convex_results):.4f}, std={np.std(convex_results):.4f}")
print(f"Non-convex: mean={np.mean(nonconvex_results):.4f}, std={np.std(nonconvex_results):.4f}")
```

---

## 9. 다른 주제와의 연결

- **임계점**: 국소 최솟값, 안장점, 고원 참고
- **초기화**: [Xavier 초기화](../../ch05/feedforward/depth_vs_width.md)에서 다룬다
- **정칙화**: [L2 정칙화](../../ch05/regularization/l2_regularization.md) 참고
- **적응적 최적화기**: 비볼록 문제에 도움이 된다. [Adam](../../ch06/optimizers/adam.md) 참고

---

## 연습문제

**연습문제 1.**
모든 $x$에 대해 2차 조건 $f''(x) \geq 0$을 확인하여 $f(x) = e^x$가 볼록임을 보여라.

??? success "연습문제 1 풀이"
    모든 $x \in \mathbb{R}$에 대해 $f'(x) = e^x$이고 $f''(x) = e^x > 0$이다. 이계 도함수가 어디서나 엄격히 양수이므로 $f$는 강볼록이다. $\square$

---

**연습문제 2.**
$f(x, y) = x^2 - y^2$에 대해 헤세 행렬을 계산하고 그 고윳값을 구한 뒤 원점의 임계점을 분류하라.

??? success "연습문제 2 풀이"
    $\nabla f = (2x, -2y)^\top$이므로 유일한 임계점은 $(0,0)$이다.

    $$
    H = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}
    $$

    고윳값은 $\lambda_1 = 2 > 0$과 $\lambda_2 = -2 < 0$이다. 고윳값의 부호가 섞여 있으므로 $H$는 부정부호이며 원점은 **안장점** 이다. 이 함수는 볼록도 오목도 아니다.

---

**연습문제 3.**
볼록 함수의 모든 국소 최솟값이 전역 최솟값임을 증명하라.

??? success "연습문제 3 풀이"
    $x^*$를 볼록 함수 $f$의 국소 최솟값이라 하자. 그러면 $\|x - x^*\| < \delta$에 대해 $f(x^*) \leq f(x)$인 $\delta > 0$이 존재한다. 모순을 위해 $f(y) < f(x^*)$인 $y$가 존재한다고 가정하자.

    볼록성에 의해 임의의 $t \in (0,1)$에 대해 $f(x^* + t(y - x^*)) \leq (1-t)f(x^*) + tf(y) < f(x^*)$이다.

    충분히 작은 $t$에 대해 점 $x^* + t(y - x^*)$은 $x^*$ 주위의 $\delta$-공 안에 놓이며, 이는 $x^*$의 국소 최소성과 모순이다. $\square$

---

**연습문제 4.**
합성 2차원 분류 과제에 대해 작은 신경망(은닉층 2개, 각 32개 단위)을 서로 다른 무작위 초기화 20개로 학습시켜라. 최종 손실 값의 히스토그램을 그리고 서로 다른 국소 최솟값이 몇 개나 발견되는지 세어라.

??? success "연습문제 4 풀이"
    ```python
    import torch
    import torch.nn as nn

    losses = []
    for seed in range(20):
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(2, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid()
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(1000):
            pred = model(X_train)
            loss = nn.BCELoss()(pred, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())

    # 보통 대부분의 실행이 비슷한 손실 값으로 수렴하는데,
    # 이는 손실 지형에 비슷한 품질의 국소 최솟값이
    # 많다는 것을 시사한다(과매개변수화된 신경망에서
    # 흔히 관찰되는 경험적 사실이다).
    ```

## 정리하며

1. **볼록 = 쉽다**: 국소 최솟값이 곧 전역 최솟값이다
2. **신경망은 비볼록이다**: 국소 최솟값이 여럿이고 안장점이 있다
3. **비볼록도 실무에서는 잘 동작한다**: 좋은 지형 구조, 과매개변수화, SGD 잡음 덕분이다
4. **초기화가 중요하다**: 특히 비볼록 문제에서 그렇다
5. **최적화 ≠ 일반화**: 전역 최솟값을 찾는 것이 언제나 목표는 아니다
6. **정칙화가 도움이 된다**: 일반화되는 해로 최적화를 이끈다

**참고 문헌**

- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Choromanska, A., et al. (2015). The loss surfaces of multilayer networks. AISTATS.
- Li, H., et al. (2018). Visualizing the loss landscape of neural nets. NeurIPS.
- Fort, S., & Ganguli, S. (2019). Emergent properties of the local geometry of neural loss landscapes.
