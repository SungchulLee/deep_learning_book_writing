# 경사는 왜 가장 가파른 상승 방향인가

경사(gradient)는 "경사 하강법"이라는 이름의 유래가 된 핵심 개념이다. 경사가 왜 최급상승 방향을 가리키는지, 거꾸로 음의 경사가 왜 최급강하 방향을 가리키는지 이해하는 것은 최적화 알고리즘에 대한 직관을 기르는 데 필수적이다.

이 절에서는 유클리드 노름으로 이동 길이를 제한할 때의 결과를 유도한다. 다른 노름이나 파라미터 척도를 사용하면 가장 가파른 방향도 달라질 수 있다.

---

## 1. 경사: 정의와 표기

### 정의

미분 가능한 스칼라 함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$에 대해 점 $\mathbf{x}$에서의 **경사** 는 모든 편도함수를 모은 벡터이다.

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

### 여러 가지 표기

경사에는 서로 다른 표기가 쓰인다.

| 표기 | 의미 |
|----------|---------|
| $\nabla f$ | $f$의 경사(나블라 표기) |
| $\nabla_\theta L$ | $\theta$에 대한 $L$의 경사 |
| $\frac{\partial L}{\partial \theta}$ | 편도함수 표기 |
| $\text{grad } f$ | 경사의 다른 표기 |
| $f'(\mathbf{x})$ | 도함수 표기(1차원) |

### 예제: 두 변수 함수

$f(x, y) = x^2 + 3y^2$에 대해 다음과 같다.

$$\nabla f = \begin{bmatrix} \frac{\partial f}{\partial x} \\ \frac{\partial f}{\partial y} \end{bmatrix} = \begin{bmatrix} 2x \\ 6y \end{bmatrix}$$

점 $(1, 2)$에서는 다음과 같다.

$$\nabla f(1, 2) = \begin{bmatrix} 2 \\ 12 \end{bmatrix}$$

---

## 2. 왜 최급상승인가?

### 방향도함수

**방향도함수** 는 (단위 벡터인) 방향 $\mathbf{u}$로 움직일 때 $f$가 얼마나 빠르게 변하는지를 잰다.

$$D_\mathbf{u} f(\mathbf{x}) = \lim_{h \to 0} \frac{f(\mathbf{x} + h\mathbf{u}) - f(\mathbf{x})}{h}$$

이는 경사와 방향의 **내적** 으로 계산할 수 있다.

$$D_\mathbf{u} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u} = \|\nabla f\| \|\mathbf{u}\| \cos\theta$$

여기서 $\theta$는 $\nabla f$와 $\mathbf{u}$ 사이의 각이다.

### 방향도함수 최대화하기

**질문**: 어느 방향 $\mathbf{u}$로 갈 때 $f$가 가장 빠르게 증가하는가?

**답**: $D_\mathbf{u} f = \|\nabla f\| \cos\theta$를 최대화하면 된다.

$\|\nabla f\|$는 고정되어 있고 $\|\mathbf{u}\| = 1$이므로 다음과 같다.

- $\cos\theta = 1$일 때(즉 $\theta = 0$일 때) 최댓값이 된다
- 이는 $\mathbf{u}$가 $\nabla f$와 평행하다는 뜻이다

**결론**: 경사 $\nabla f$는 **최급상승** 방향을 가리킨다.

### 형식적인 정리

!!! tip "최급상승 방향으로서의 경사"
    $f: \mathbb{R}^n \rightarrow \mathbb{R}$이 $\mathbf{x}$에서 미분 가능하고 $\nabla f(\mathbf{x}) \neq \mathbf{0}$이라 하자. 그러면 다음이 성립한다.
    
    1. 최대 증가 방향은 $\mathbf{u}^* = \frac{\nabla f}{\|\nabla f\|}$이다
    2. 최대 증가율은 $\|\nabla f(\mathbf{x})\|$이다
    3. 최대 감소 방향은 $-\mathbf{u}^*$이다

### 증명 개요

임의의 단위 벡터 $\mathbf{u}$에 대해 다음이 성립한다.

$$D_\mathbf{u} f = \nabla f \cdot \mathbf{u} \leq \|\nabla f\| \cdot \|\mathbf{u}\| = \|\nabla f\|$$

코시-슈바르츠 부등식에 의한 결과이다. 등호는 $\mathbf{u} = \frac{\nabla f}{\|\nabla f\|}$일 때 성립한다.

---

## 3. 기하학적 해석

### 등위집합과 경사

$f$의 **등위집합**(또는 등고선)은 $f$가 같은 값을 갖는 점들의 집합이다.

$$\{x : f(\mathbf{x}) = c\}$$

경사가 0이 아닌 점에서 경사는 매끄러운 등위집합의 접선 또는 접평면에 수직이다.

```
                    ↑ ∇f
                    │
    ────────────────┼────────────── f = c + ε (higher)
                    │
    ────────────────●────────────── f = c (level set)
                    │
    ────────────────│────────────── f = c - ε (lower)
```

경사가 0이 아닌 점에서는 등고선에 수직인 경사 방향으로 움직일 때 일차 변화율이 가장 크다.

### 등고선 그림으로 보기

(포물면인) $f(x, y) = x^2 + y^2$에 대해 다음과 같다.

- 등고선은 동심원이다
- 임의의 점에서의 경사는 원점에서 바깥으로 방사 방향을 가리킨다
- 음의 경사를 따라가면 $(0, 0)$의 최솟값에 이른다

```python
import numpy as np
import matplotlib.pyplot as plt

# 격자 생성
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = X**2 + Y**2

# 등고선 그리기
plt.contour(X, Y, Z, levels=15)

# 선택한 점들에서 경사 벡터 그리기
points = [(-2, 1), (1, 2), (2, -1)]
for px, py in points:
    grad = np.array([2*px, 2*py])
    plt.arrow(px, py, 0.3*grad[0], 0.3*grad[1], 
              head_width=0.15, color='red')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradients Perpendicular to Contours')
plt.axis('equal')
plt.show()
```

---

## 4. 최급상승에서 경사 하강법으로

### 하강 방향

$\nabla f$가 최급상승을 가리키므로 **$-\nabla f$는 최급강하를 가리킨다.**

$f$를 최소화하려면 $-\nabla f$ 방향으로 움직여야 한다.

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \nabla f(\mathbf{x}_t)$$

이것이 **경사 하강법의 갱신 규칙** 이다.

### 하강이 함숫값을 줄이는 이유

**1차 테일러 근사:**

$$f(\mathbf{x} + \Delta\mathbf{x}) \approx f(\mathbf{x}) + \nabla f(\mathbf{x})^T \Delta\mathbf{x}$$

$\Delta\mathbf{x} = -\eta \nabla f$에 대해 다음과 같다.

$$f(\mathbf{x} - \eta\nabla f) \approx f(\mathbf{x}) - \eta \|\nabla f\|^2$$

$\|\nabla f\|^2 \geq 0$이므로 ($\eta$가 작을 때) 다음이 성립한다.

$$f(\mathbf{x}_{t+1}) \leq f(\mathbf{x}_t)$$

경사가 0이 아니고 러닝레이트가 충분히 작으면 로스가 감소한다. 임의의 러닝레이트나 미니배치 경사에 대해 매 단계의 감소가 보장되는 것은 아니다.

---

## 5. 경사 계산하기

### 직접 유도하기

로스 함수 $L(w) = \frac{1}{N}\sum_{i=1}^N (wx_i - y_i)^2$에 대해 다음과 같이 한다.

**1단계**: 전개한다

$$L(w) = \frac{1}{N}\sum_{i=1}^N (w^2x_i^2 - 2wx_iy_i + y_i^2)$$

**2단계**: 항별로 미분한다

$$\frac{dL}{dw} = \frac{1}{N}\sum_{i=1}^N (2wx_i^2 - 2x_iy_i) = \frac{2}{N}\sum_{i=1}^N x_i(wx_i - y_i)$$

### 자동 미분

PyTorch는 경사를 자동으로 계산한다.

```python
import torch

# 경사 추적을 켠 변수 정의
x = torch.tensor([1., 2., 3., 4., 5.])
y = torch.tensor([2., 4., 6., 8., 10.])
w = torch.tensor(0.5, requires_grad=True)

# 순전파
y_pred = w * x
loss = torch.mean((y_pred - y) ** 2)

# 역전파 - 경사를 계산한다
loss.backward()

print(f"Gradient dL/dw = {w.grad.item():.4f}")
```

**출력:**

```
Gradient dL/dw = -33.0000
```

### 심층 뉴럴 넷트웍에서의 연쇄 법칙

중간값을 $h_1=f_1(\theta)$, $h_2=f_2(h_1)$, 로스를 $L=\ell(h_2)$라 두자. 열벡터 경사 표기를 쓰면 연쇄 법칙은 다음과 같다.

$$\nabla_\theta L = J_{f_1}(\theta)^\top J_{f_2}(h_1)^\top \nabla_{h_2}\ell$$

벡터 중간값에서는 각 도함수가 야코비안이므로 곱의 순서와 전치를 함께 확인해야 한다.

이것이 **역전파** 의 토대이다.

---

## 6. 경사의 성질

### 임계점에서

$\nabla f(\mathbf{x}) = \mathbf{0}$일 때 점 $\mathbf{x}$는 **임계점**(또는 정류점)이다. 다음 중 하나일 수 있다.

- 헤세 행렬이 양의 정부호이면 엄격한 국소 최솟값이다.
- 음의 정부호이면 엄격한 국소 최댓값이다.
- 양수와 음수 고윳값이 함께 있으면 안장점이다.
- 0인 고윳값이 있거나 준정부호인 경우에는 이 검사만으로 분류할 수 없는 경우가 있다.

### 경사의 크기

크기 $\|\nabla f\|$는 지형이 얼마나 가파른지를 나타낸다.

- **큰 경사**: 현재 좌표와 척도에서 함수의 일차 변화율이 크다
- **작은 경사**: 일차 변화율이 작다. 임계점과의 거리가 가깝다는 뜻은 아니다
- **경사가 0**: 임계점에 있다

### 경사 방향의 변화

경사의 방향과 크기는 함수의 곡률, 초기값, 러닝레이트에 따라 달라진다. 학습 초기에 항상 큰 경사가 나오거나 수렴 전에 방향이 일관되게 유지되는 것은 아니다. 로스와 경사 노름을 함께 기록해 실제 양상을 확인한다.

정류점 근처에서 경사가 작아질 수 있지만, 작은 경사만으로 좋은 최솟값에 도달했다고 판단하지 않는다.

---

## 7. 실무적 함의

### 왜 경사를 정규화하는가?

때로는 크기를 버리고 **경사의 방향** 만 사용한다.

$$\mathbf{x}_{t+1} = \mathbf{x}_t - \eta \frac{\nabla f}{\|\nabla f\|}$$

장점:

- 경사의 크기와 무관하게 이동 폭이 일정하다
- 작은 경사에서도 일정한 이동 길이를 유지한다

단점:

- 경사의 크기 정보를 사용하지 않는다
- 최솟값 근처에서 지나칠 수 있다

### 경사 클리핑

경사 폭발을 막기 위해 다음과 같이 한다.

```python
max_norm = 1.0
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

예제의 노름 기반 클리핑은 전체 경사 벡터에 같은 비율을 곱하므로 유한한 경사의 방향을 유지한다. 원소별 값을 자르는 클리핑은 방향까지 바꿀 수 있다. 경사 정규화에서는 경사 노름이 0인 경우도 별도로 처리해야 한다.

---

## 8. 흔한 오해

### 오해 1: "경사 하강법은 언제나 전역 최솟값을 찾는다"

**사실**: 볼록 문제에서도 최솟값의 존재와 적절한 러닝레이트 등 수렴 조건이 필요하다. 비볼록 문제에서는 정류점에 접근하거나 진동·발산할 수 있으며, 국소 최솟값에 도달한다는 보장도 일반적으로 없다.

### 오해 2: "경사가 클수록 빠르게 수렴한다"

**사실**: 지나치게 큰 경사는 지나침과 발산을 일으킬 수 있다. 적절한 러닝레이트와 함께 적당한 크기의 경사가 가장 잘 동작한다.

### 오해 3: "경사는 최솟값을 곧바로 가리킨다"

**사실**: 경사는 국소적인 최급상승 방향이고, 음의 경사가 최급강하 방향이다. 이 방향이 최솟값의 위치를 직접 가리키지는 않으며 이동 경로는 굽어 있을 수 있다.

---

## 9. 다른 주제와의 연결

- **계산 그래프**: 계산 그래프 참고
- **임계점**: 국소 최솟값, 안장점, 고원에서 자세히 다룬다
- **모멘텀**: [고전적 모멘텀](../../ch09/optimizers/momentum.md)에서 최급강하를 변형한다

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
다음 각각에 대해 $\nabla f$를 계산하라. (a) $f(x, y) = x^2y + y^3$, (b) $f(x, y, z) = e^{xy} + \sin(z)$, (c) $f(\mathbf{w}) = \|\mathbf{Xw} - \mathbf{y}\|^2$.

</div>

??? success "풀이"
    (a) $\nabla f = (2xy,\; x^2 + 3y^2)^\top$.

    (b) $\nabla f = (ye^{xy},\; xe^{xy},\; \cos z)^\top$.

    (c) 전개하면 $f = (\mathbf{Xw}-\mathbf{y})^\top(\mathbf{Xw}-\mathbf{y}) = \mathbf{w}^\top\mathbf{X}^\top\mathbf{X}\mathbf{w} - 2\mathbf{y}^\top\mathbf{X}\mathbf{w} + \mathbf{y}^\top\mathbf{y}$이다. 미분하면 $\nabla_\mathbf{w} f = 2\mathbf{X}^\top(\mathbf{Xw} - \mathbf{y})$이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
$f(x,y) = x^2 + 4y^2$에 대해 점 $(2, 1)$에서 $\nabla f$가 등위곡선 $f(x,y) = c$에 수직임을 보여라.

</div>

??? success "풀이"
    $(2,1)$에서 $f = 4 + 4 = 8$이다. 등위곡선은 $x^2 + 4y^2 = 8$이다. 곡선 위를 $\mathbf{r}(t) = (x(t), y(t))$로 매개화하면 $2x x' + 8y y' = 0$이다. $(2,1)$에서의 접벡터는 $4x' + 8y' = 0$, 즉 $x' = -2y'$을 만족한다. 접벡터 하나는 $(-2, 1)$이다.

    경사는 $\nabla f = (2x, 8y) = (4, 8)$이다. 내적은 $(-2)(4) + (1)(8) = 0$이다. 내적이 0이므로 $\nabla f$는 접벡터에 수직이며 직교성이 확인된다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff easy" title="쉬움"></span>
$(1, 1)$에서 $f(x,y) = x^2 - xy + y^2$에 대해 경사를 계산하고, 방향 $(1, 0)$의 방향도함수를 구하고, 최급상승 방향을 구하라.

</div>

??? success "풀이"
    $\nabla f = (2x - y,\; -x + 2y)$이다. $(1,1)$에서 $\nabla f = (1, 1)$이다.

    방향 $\mathbf{u} = (1,0)$의 방향도함수는 $D_\mathbf{u} f = \nabla f \cdot \mathbf{u} = 1$이다.

    최급상승 방향은 $\frac{\nabla f}{\|\nabla f\|} = \frac{1}{\sqrt{2}}(1, 1)$이다. 최대 증가율은 $\|\nabla f\| = \sqrt{2}$이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
$w_1 = 0.5, w_2 = -0.3, x_1 = 2, x_2 = 3, y = 1$에서 PyTorch autograd를 사용하여 $L = (w_1 x_1 + w_2 x_2 - y)^2$의 $w_1$과 $w_2$에 대한 경사를 계산하라.

</div>

??? success "풀이"
    ```python
    import torch

    w1 = torch.tensor(0.5, requires_grad=True)
    w2 = torch.tensor(-0.3, requires_grad=True)
    x1, x2, y = 2.0, 3.0, 1.0

    L = (w1 * x1 + w2 * x2 - y) ** 2
    L.backward()

    print(w1.grad, w2.grad)
    # 해석적 계산: 잔차 r = 0.5*2 + (-0.3)*3 - 1 = 1 - 0.9 - 1 = -0.9
    # dL/dw1 = 2r * x1 = 2(-0.9)(2) = -3.6
    # dL/dw2 = 2r * x2 = 2(-0.9)(3) = -5.4
    ```

## 정리하며

1. **경사의 정의**: 편도함수를 모은 벡터
2. **최급상승**: 경사는 최대 증가 방향을 가리킨다
3. **최급강하**: 음의 경사는 같은 길이의 작은 이동 중 가장 큰 일차 감소를 준다
4. **등고선에 수직**: 경사는 등위집합에 직교한다
5. **크기가 중요하다**: 가파른 정도를 나타내며 임계점에서 0이다
6. **자동 미분**: PyTorch가 경사를 효율적으로 계산한다

**참고 문헌**

- Stewart, J. (2015). *Calculus: Early Transcendentals*, Chapter 14.
- Boyd, S., & Vandenberghe, L. (2004). *Convex Optimization*, Chapter 9.
- Ruder, S. (2016). An overview of gradient descent optimization algorithms. arXiv:1609.04747.
