# 볼록 최적화와 비볼록 최적화

볼록 최적화에서는 목적함수와 허용 영역의 구조 덕분에 국소 최적성과 전역 최적성을 연결할 수 있다. 뉴럴 넷트웍처럼 비볼록인 문제에서는 이 보장이 약해진다. 이 절에서는 두 경우의 정의와 수렴 조건, 실무에서 결과를 해석하는 기준을 살펴본다.

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

### 엄격한 볼록성과 강볼록성

$\mathbf{x}\ne\mathbf{y}$이고 $\lambda\in(0,1)$일 때 볼록성 부등식이 엄격하게 성립하면 **엄격한 볼록 함수**(strictly convex)다.

$\mathbf{x} \neq \mathbf{y}$이고 $\lambda \in (0, 1)$일 때 부등식이 엄격하게 성립하면 그 함수는 **강볼록** 이다.

$$f(\lambda\mathbf{x} + (1-\lambda)\mathbf{y}) < \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

**강볼록 함수**(strongly convex)는 어떤 $\mu>0$에 대해 $f(\mathbf{x})-\frac{\mu}{2}\|\mathbf{x}\|^2$가 볼록인 함수다. 두 번 미분 가능한 경우에는 정의역 전체에서 $\nabla^2 f\succeq\mu I$인 것과 같다. 강볼록이면 엄격히 볼록하지만 역은 일반적으로 성립하지 않는다. 예를 들어 $e^x$는 실수 전체에서 엄격히 볼록하지만 이계 도함수의 양의 하한이 없어 강볼록하지 않다.

### 동치인 조건

볼록한 열린 정의역의 미분 가능한 함수에서는 다음 1차 조건으로 볼록성을 확인한다. 두 번 미분 가능하면 2차 조건도 동치다.

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
| 거듭제곱 | $f(x) = x^p$ ($p \geq 1$ 또는 $p \leq 0$) | $x>0$ |
| 노름 | $f(\mathbf{x}) = \|\mathbf{x}\|_p$, $p\ge1$ | $\mathbb{R}^n$ |
| Log-sum-exp | $f(\mathbf{x}) = \log(\sum e^{x_i})$ | $\mathbb{R}^n$ |

### 기계학습에서의 예

**MSE 로스(선형 회귀):**

$$L(\mathbf{w}) = \frac{1}{N}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|^2$$

이는 $\mathbf{w}$에 대해 볼록이다(헤세 행렬이 양의 준정부호인 이차식이다).

**로지스틱 회귀 로스:**

$$L(\mathbf{w}) = \frac{1}{N}\sum_{i=1}^N \log(1 + e^{-y_i \mathbf{w}^T\mathbf{x}_i})$$

$y_i\in\{-1,1\}$인 이진 로지스틱 회귀의 음의 로그우도다. 이는 $\mathbf{w}$에 대해 볼록하지만, 분리 가능한 데이터에서 정규화가 없으면 유한한 최적 파라미터가 존재하지 않을 수 있다.

---

## 3. 볼록 최적화의 성질

### 전역 최적성

!!! tip "볼록 함수의 근본적인 성질"
    볼록 함수 $f$에 대해 다음이 성립한다.
    
    - 모든 **국소 최솟값** 이 **전역 최솟값** 이다
    - $f$가 엄격히 볼록이면 전역 최솟값은 존재할 경우 **유일하다**

볼록 최적화가 "쉽다"고 여겨지는 이유가 이것이다. 최적이 아닌 국소 최솟값에 갇힐 걱정을 하지 않아도 된다.

### 경사 하강법의 수렴

최솟값 $\mathbf{x}^*$가 존재하고 경사가 $L$-립시츠 연속인 볼록 함수에, $0<\eta\le1/L$인 고정 러닝레이트의 전체 경사 하강법을 적용하면 $t\ge1$에서 다음 경계가 성립한다.

$$f(\mathbf{x}_t) - f(\mathbf{x}^*) \leq \frac{\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{2\eta t}$$

이는 $O(1/t)$의 속도로 수렴함을 보장한다.

추가로 $\mu$-강볼록이고 러닝레이트를 $1/L$로 선택하면 다음과 같은 기하급수적 감소를 얻는다.

$$f(\mathbf{x}_t) - f(\mathbf{x}^*) \leq \left(1 - \frac{\mu}{L}\right)^t (f(\mathbf{x}_0) - f(\mathbf{x}^*))$$

이는 (지수적으로 빠른) **선형 수렴** 을 준다.

---

## 4. 비볼록 최적화

### 목적함수와 제약을 구분하기

볼록성 부등식을 만족하지 않는 함수는 비볼록이다. 서로 다른 높이의 국소 최솟값이나 음의 곡률은 비볼록성을 드러낼 수 있지만, 모든 비볼록 함수에 여러 국소 최솟값이 있는 것은 아니다. 목적함수가 볼록해도 허용 영역이 비볼록이면 최적화 문제 자체는 볼록 문제가 아니다.

### 뉴럴 넷트웍 로스의 구조

여러 계층의 파라미터가 곱으로 결합되면 각 계층이 선형이어도 전체 파라미터에 대한 로스는 일반적으로 비볼록이다. 비선형 활성화도 로스 구조에 영향을 준다. 은닉 유닛 순서의 대칭성은 같은 함수를 나타내는 여러 파라미터를 만들 수 있으며, ReLU 같은 양의 동차 활성화에서는 계층 간 스케일 대칭도 나타난다. 이런 대칭성을 모든 활성화에 적용할 수는 없다.

```python
# 간단한 비볼록 로스 지형
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def neural_loss(w1, w2, x=1, y=2):
    """y = w2 * ReLU(w1 * x)의 로스"""
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

### 비볼록 문제의 어려움

경사 기반 방법이 전역 최솟값에 도달한다는 일반적인 보장은 없다. 국소 최솟값뿐 아니라 안장점이나 경사가 작은 영역에서 진행이 느려질 수 있으며, 초기값과 러닝레이트에 따라 다른 경로를 따른다. 정류점으로 수렴한다는 결과도 별도의 매끄러움과 러닝레이트 조건을 요구한다.

---

## 5. 딥러닝 최적화를 돕는 요인

일부 모델과 데이터에서는 과파라미터화로 낮은 학습 로스를 주는 해가 많아지고, 좋은 초기화와 정규화가 경사 전달을 도울 수 있다. 그러나 파라미터가 많다는 사실만으로 최적화나 일반화가 보장되지는 않는다.

미니배치의 경사 잡음은 정류점 주변의 이동과 해의 선택에 영향을 준다. 특정 조건에서는 정규화와 비슷한 효과를 보이지만, 항상 나쁜 최솟값에서 탈출하거나 일반화가 좋은 해를 찾는다고 해석해서는 안 된다. 로스 지형에 관한 연구 결과의 범위도 모델 구조와 분석 가정에 따라 다르다.

### 초기화의 역할

Xavier와 He 초기화는 활성화 특성에 맞춰 초기 값과 경사의 척도를 조절하는 방법이다. 학습을 돕는 출발점이 될 수 있지만 전역 최적성의 보장은 아니다.

```python
# ReLU 뉴럴 넷트웍을 위한 He 초기화
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# tanh 뉴럴 넷트웍을 위한 Xavier 초기화
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

다음 코드는 선택한 점의 헤세 행렬을 검사한다. 한 점에서 음의 고윳값을 찾으면 비볼록성의 증거가 되지만, 일부 점에서 양의 준정부호였다는 사실만으로 정의역 전체의 볼록성을 증명할 수는 없다.

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

아래 조건에서 볼록성이 보존된다. 각 $f_i$ 또는 $f$가 볼록하고, 정의역 조건도 만족한다고 가정한다.

1. **음이 아닌 가중합**: $\alpha_i \geq 0$일 때 $\sum \alpha_i f_i$
2. **아핀 함수와의 합성**: $f(\mathbf{Ax} + \mathbf{b})$
3. **점별 최댓값**: $\max(f_1, f_2, \ldots, f_n)$
4. **부분 최소화**: $f$가 $(\mathbf{x},\mathbf{y})$에 대해 공동으로 볼록하고 $\mathbf{y}$의 허용 영역이 고정된 볼록 집합이면 $g(\mathbf{x})=\inf_{\mathbf{y}}f(\mathbf{x},\mathbf{y})$는 적절한 유한값 조건에서 볼록하다

---

## 8. 실무적 함의

### 볼록 문제에서

- 최솟값의 존재, 매끄러움, 러닝레이트 등 조건을 확인하면 수렴 보장을 활용할 수 있다
- 초기화에 **덜 민감하다**
- 문제 구조와 제약에 맞는 알고리즘을 선택한다. 볼록성만으로 임의의 SGD·Adam 설정의 수렴이 보장되지는 않는다
- **해석적 해** 가 존재할 수 있다(예: 선형 회귀)

### 비볼록 문제에서(딥러닝)

- **여러 번 다시 시작한다**: 서로 다른 초기화를 시도한다
- **러닝레이트 워밍업**: 초기 탐색을 조심스럽게 한다
- **정규화**: 일반화되는 최솟값을 찾는 데 도움이 된다
- **학습을 관찰한다**: 발산과 진동을 살핀다
- **적응적 옵티마이저를 쓴다**: Adam이 SGD보다 강건한 경우가 많다

### 코드: 두 경우의 최적화 비교

```python
import numpy as np
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

# 비볼록: 뉴럴 넷트웍
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

**출력:**

```
Convex: mean=0.2736, std=0.1007
Non-convex: mean=0.4647, std=0.0284
```

---

## 9. 다른 주제와의 연결

- **임계점**: 국소 최솟값, 안장점, 고원 참고
- **초기화**: [Xavier 초기화](../../ch08/feedforward/depth_vs_width.md)에서 다룬다
- **정규화**: [L2 정규화](../../ch08/regularization/l2_regularization.md) 참고
- **적응적 옵티마이저**: 비볼록 문제에 도움이 된다. [Adam](../../ch09/optimizers/adam.md) 참고

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
모든 $x$에 대해 2차 조건 $f''(x) \geq 0$을 확인하여 $f(x) = e^x$가 볼록임을 보여라.

</div>

??? success "연습문제 1 풀이"
    모든 $x \in \mathbb{R}$에 대해 $f'(x) = e^x$이고 $f''(x) = e^x > 0$이다. 이계 도함수가 어디서나 양수이므로 $f$는 엄격히 볼록이다. 다만 $x\to-\infty$에서 $e^x\to0$이므로 실수 전체에서 양의 곡률 하한을 갖는 강볼록 함수는 아니다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff easy" title="쉬움"></span>
$f(x, y) = x^2 - y^2$에 대해 헤세 행렬을 계산하고 그 고윳값을 구한 뒤 원점의 임계점을 분류하라.

</div>

??? success "연습문제 2 풀이"
    $\nabla f = (2x, -2y)^\top$이므로 유일한 임계점은 $(0,0)$이다.

    $$
    H = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}
    $$

    고윳값은 $\lambda_1 = 2 > 0$과 $\lambda_2 = -2 < 0$이다. 고윳값의 부호가 섞여 있으므로 $H$는 부정부호이며 원점은 **안장점** 이다. 이 함수는 볼록도 오목도 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
볼록 함수의 모든 국소 최솟값이 전역 최솟값임을 증명하라.

</div>

??? success "연습문제 3 풀이"
    $x^*$를 볼록 함수 $f$의 국소 최솟값이라 하자. 그러면 $\|x - x^*\| < \delta$에 대해 $f(x^*) \leq f(x)$인 $\delta > 0$이 존재한다. 모순을 위해 $f(y) < f(x^*)$인 $y$가 존재한다고 가정하자.

    볼록성에 의해 임의의 $t \in (0,1)$에 대해 $f(x^* + t(y - x^*)) \leq (1-t)f(x^*) + tf(y) < f(x^*)$이다.

    충분히 작은 $t$에 대해 점 $x^* + t(y - x^*)$은 $x^*$ 주위의 $\delta$-공 안에 놓이며, 이는 $x^*$의 국소 최소성과 모순이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
합성 2차원 분류 과제에 대해 작은 뉴럴 넷트웍(은닉 레이어 2개, 각 32개 단위)을 서로 다른 무작위 초기화 20개로 학습시켜라. 최종 로스의 히스토그램을 그리고 초기값에 따른 차이를 해석하라. 로스값의 차이만으로 서로 다른 국소 최솟값의 개수를 셀 수 없는 이유도 설명하라.

</div>

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

    # 보통 대부분의 실행이 비슷한 로스 값으로 수렴하는데,
    # 이는 로스 지형에 비슷한 품질의 국소 최솟값이
    # 많다는 것을 시사한다(과파라메터화된 뉴럴 넷트웍에서
    # 흔히 관찰되는 경험적 사실이다).
    ```

## 정리하며

1. **볼록성의 이점**: 국소 최솟값은 전역 최솟값이지만 존재와 수렴 조건은 별도로 확인한다
2. **뉴럴 넷트웍은 비볼록이다**: 국소 최솟값이 여럿이고 안장점이 있다
3. **실무 성능은 검증한다**: 초기화, 모델 구조, 데이터와 옵티마이저가 함께 영향을 준다
4. **초기화가 중요하다**: 특히 비볼록 문제에서 그렇다
5. **최적화 ≠ 일반화**: 전역 최솟값을 찾는 것이 언제나 목표는 아니다
6. **정규화가 도움이 된다**: 일반화되는 해로 최적화를 이끈다

**참고 문헌**

- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*. Cambridge University Press.
- Choromanska, A., et al. (2015). The loss surfaces of multilayer networks. AISTATS.
- Li, H., et al. (2018). Visualizing the loss landscape of neural nets. NeurIPS.
- Fort, S., & Ganguli, S. (2019). Emergent properties of the local geometry of neural loss landscapes.
