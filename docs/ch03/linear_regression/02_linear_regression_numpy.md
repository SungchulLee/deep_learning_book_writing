# NumPy로 만드는 선형 회귀

선형 회귀는 가장 단순한 지도 학습 모델이자 모든 신경망 학습이 세워지는 토대이다. 예측, 평균제곱오차 손실, 경사를 직접 계산하며 NumPy로 바닥부터 구현해 보면, 프레임워크의 추상화가 끼어들기 전에 학습 루프의 모든 동작을 볼 수 있다. 이 이해는 나중에 더 복잡한 모델을 디버깅하고 따져 보는 데 필수적이다.

## 1. 코드

```python
"""
==============================================================================
02_linear_regression_numpy.py
==============================================================================
어려움: ⭐ (첫걸음)

DESCRIPTION:
    넘파이만으로 선형 회귀를 밑바닥부터 짠다.
    PyTorch의 추상을 쓰기 앞에 수식을 이해하는 데 도움이 된다.

다루는 것:
    - 선형 회귀의 수학 바탕
    - 경사 하강법 알고리즘
    - 손실 함수(평균 제곱 오차)
    - 직접 하는 기울기 셈
    - 학습 루프의 짜임

PREREQUISITES:
    - 기본 선형대수(벡터, 행렬)
    - 기본 미적분(도함수)
    - 튜토리얼 01(도움이 되지만 꼭 필요하지는 않다)

학습 목표:
    - 선형 모델 y = wx + b을 이해한다
    - 손실과 기울기를 직접 계산한다
    - 경사 하강법을 밑바닥부터 짠다
    - 학습이 나아가는 모습을 그림으로 본다

걸리는 때: 20분쯤
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("LINEAR REGRESSION WITH NUMPY")
print("=" * 70)

# ============================================================================
# 1부: 합성 데이터 생성
# ============================================================================
np.random.seed(42)

TRUE_W = 2.0
TRUE_B = 1.0

n_samples = 100
X = np.random.uniform(-10, 10, n_samples)
noise = np.random.normal(0, 2, n_samples)
y = TRUE_W * X + TRUE_B + noise

print(f"Generated {n_samples} data points")
print(f"True parameters: w={TRUE_W}, b={TRUE_B}")

# ============================================================================
# 2부: 모델과 손실 함수 정의
# ============================================================================

def predict(X, w, b):
    return w * X + b

def compute_loss(y_true, y_pred):
    n = len(y_true)
    loss = (1 / n) * np.sum((y_true - y_pred) ** 2)
    return loss

def compute_gradients(X, y_true, y_pred):
    n = len(X)
    error = y_pred - y_true
    grad_w = (2 / n) * np.sum(error * X)
    grad_b = (2 / n) * np.sum(error)
    return grad_w, grad_b

# ============================================================================
# 3부: 학습 루프 (경사 하강법)
# ============================================================================
w = 0.0
b = 0.0
learning_rate = 0.01
n_epochs = 100

loss_history = []

print(f"\n{'Epoch':<8} {'Loss':<12} {'w':<12} {'b':<12}")
print("-" * 50)

for epoch in range(n_epochs):
    y_pred = predict(X, w, b)
    loss = compute_loss(y, y_pred)
    loss_history.append(loss)
    grad_w, grad_b = compute_gradients(X, y, y_pred)
    w = w - learning_rate * grad_w
    b = b - learning_rate * grad_b

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {loss:<12.4f} {w:<12.4f} {b:<12.4f}")

print(f"\nFinal Results:")
print(f"  Learned w: {w:.4f} (True: {TRUE_W})")
print(f"  Learned b: {b:.4f} (True: {TRUE_B})")
print(f"  Final loss: {loss_history[-1]:.4f}")


if __name__ == "__main__":
    pass
```

**출력:**

```
======================================================================
LINEAR REGRESSION WITH NUMPY
======================================================================
Generated 100 data points
True parameters: w=2.0, b=1.0

Epoch    Loss         w            b           
--------------------------------------------------
1        137.0771     1.3719       -0.0039     
10       3.9226       1.9398       0.1484      
20       3.6932       1.9424       0.2973      
30       3.5393       1.9445       0.4193      
40       3.4362       1.9462       0.5192      
50       3.3670       1.9476       0.6009      
60       3.3206       1.9488       0.6679      
70       3.2896       1.9497       0.7227      
80       3.2687       1.9505       0.7676      
90       3.2548       1.9511       0.8043      
100      3.2454       1.9517       0.8344      

Final Results:
  Learned w: 1.9517 (True: 2.0)
  Learned b: 0.8344 (True: 1.0)
  Final loss: 3.2454
```

## 2. 논의

선형 모델 $y = wx + b$는 가중치 $w$(기울기)와 편향 $b$(절편)를 통해 입력 $x$를 예측으로 대응시킨다. 평균제곱오차(MSE) 손실은 예측이 목표값에서 평균적으로 얼마나 벗어나는지를 재며, 이를 최소화하면 참된 데이터 생성 과정에 가까운 매개변수를 얻는다. 경사 하강법은 손실을 줄이는 방향으로 매개변수를 반복적으로 옮기며, 학습률이 이동 폭을 조절한다.

경사를 직접 계산하려면 기초 미적분을 적용하면 된다. $\hat{y}_i = wx_i + b$인 MSE 손실 $L = \frac{1}{n}\sum (y_i - \hat{y}_i)^2$에 대해 편도함수는 $\frac{\partial L}{\partial w} = \frac{2}{n}\sum (\hat{y}_i - y_i)x_i$과 $\frac{\partial L}{\partial b} = \frac{2}{n}\sum (\hat{y}_i - y_i)$이다. 갱신 규칙 $w \leftarrow w - \alpha \frac{\partial L}{\partial w}$($b$도 마찬가지)은 각 매개변수를 최급강하 방향으로 옮긴다.

순전파, 손실 계산, 경사 계산, 매개변수 갱신이라는 학습 루프의 구조는 복잡도와 무관하게 모든 신경망에서 동일하다. PyTorch는 경사 계산 단계를 자동화하지만 개념적인 흐름은 결코 바뀌지 않는다. 이 루프를 NumPy 수준에서 이해해 두면 학습 발산(학습률이 너무 큼), 느린 수렴(학습률이 너무 작음), 잘못된 경사 공식 같은 문제를 진단하기가 훨씬 쉬워진다.

## 연습문제

**연습문제 1.**
학습률 $\alpha \in \{0.001, 0.01, 0.1\}$ 각각에 대해 200 에폭 동안 학습 루프를 실행하라. 손실 곡선을 같은 축에 그리고, 발산하지 않으면서 가장 빠르게 수렴하는 학습률을 찾아라.

??? success "연습문제 1 풀이"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    X = np.random.uniform(-10, 10, 100)
    y = 2.0 * X + 1.0 + np.random.normal(0, 2, 100)

    for lr in [0.001, 0.01, 0.1]:
        w, b = 0.0, 0.0
        losses = []
        for epoch in range(200):
            y_pred = w * X + b
            loss = np.mean((y - y_pred) ** 2)
            losses.append(loss)
            grad_w = (2 / len(X)) * np.sum((y_pred - y) * X)
            grad_b = (2 / len(X)) * np.sum(y_pred - y)
            w -= lr * grad_w
            b -= lr * grad_b
        plt.plot(losses, label=f'lr={lr}')
        print(f"lr={lr}: final loss={losses[-1]:.4f}, w={w:.4f}, b={b:.4f}")

    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('Learning Rate Comparison')
    plt.show()
    ```
    학습률 0.01은 매끄럽게 수렴한다. 0.001은 너무 느리고, 0.1은 처음에는 더 빠르게 수렴할 수 있지만 데이터의 규모에 따라 진동할 수 있다.

---

**연습문제 2.**
연쇄 법칙을 사용하여 MSE 손실 $L = \frac{1}{n}\sum_{i=1}^n (y_i - (wx_i + b))^2$에 대한 경사 $\frac{\partial L}{\partial w}$를 단계별로 유도하라.

??? success "연습문제 2 풀이"
    $L = \frac{1}{n}\sum_{i=1}^n (y_i - wx_i - b)^2$이라 하자. $e_i = y_i - wx_i - b$로 두면 $L = \frac{1}{n}\sum e_i^2$이다.

    연쇄 법칙에 의해 다음과 같다.

    $$
    \frac{\partial L}{\partial w} = \frac{1}{n}\sum_{i=1}^n 2 e_i \cdot \frac{\partial e_i}{\partial w}
    $$

    $e_i = y_i - wx_i - b$이므로 $\frac{\partial e_i}{\partial w} = -x_i$이다. 대입하면 다음과 같다.

    $$
    \frac{\partial L}{\partial w} = \frac{1}{n}\sum_{i=1}^n 2(y_i - wx_i - b)(-x_i) = \frac{2}{n}\sum_{i=1}^n (wx_i + b - y_i) x_i
    $$

    이는 코드에서 쓴 공식 `grad_w = (2/n) * sum((y_pred - y_true) * X)`과 일치한다.

---

**연습문제 3.**
학습 동안 $(w, b)$의 궤적을 추적하여 손실 곡면의 등고선 위에 2차원 경로로 겹쳐 그리도록 코드를 수정하라. $(w, b)$ 값의 격자에서 손실을 계산하는 데 `np.meshgrid`를 사용하라.

??? success "연습문제 3 풀이"
    ```python
    import numpy as np
    import matplotlib.pyplot as plt

    np.random.seed(42)
    X = np.random.uniform(-10, 10, 100)
    y = 2.0 * X + 1.0 + np.random.normal(0, 2, 100)

    # 손실 곡면을 계산한다
    w_range = np.linspace(-1, 4, 100)
    b_range = np.linspace(-2, 4, 100)
    W, B = np.meshgrid(w_range, b_range)
    L = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            pred = W[i, j] * X + B[i, j]
            L[i, j] = np.mean((y - pred) ** 2)

    # 학습하며 궤적을 기록한다
    w, b = 0.0, 0.0
    trajectory_w, trajectory_b = [w], [b]
    for _ in range(100):
        y_pred = w * X + b
        grad_w = (2 / len(X)) * np.sum((y_pred - y) * X)
        grad_b = (2 / len(X)) * np.sum(y_pred - y)
        w -= 0.01 * grad_w
        b -= 0.01 * grad_b
        trajectory_w.append(w)
        trajectory_b.append(b)

    plt.figure(figsize=(8, 6))
    plt.contour(W, B, L, levels=30, cmap='viridis')
    plt.plot(trajectory_w, trajectory_b, 'ro-', markersize=3, linewidth=1)
    plt.plot(trajectory_w[0], trajectory_b[0], 'gs', markersize=10, label='Start')
    plt.plot(trajectory_w[-1], trajectory_b[-1], 'r*', markersize=15, label='End')
    plt.xlabel('w')
    plt.ylabel('b')
    plt.title('Gradient Descent Trajectory on Loss Surface')
    plt.legend()
    plt.colorbar(label='MSE')
    plt.show()
    ```

## 정리하며

**다룬 것** — NumPy로 만드는 선형 회귀

선형 모델 $y = wx + b$는 가중치 $w$(기울기)와 편향 $b$(절편)를 통해 입력 $x$를 예측으로 대응시킨다.

앞의 연습문제 3개로 직접 확인할 수 있다.
