# 순수 NumPy로 만드는 선형 회귀

선형 회귀는 모든 신경망의 밑바탕에 있는 핵심 동작, 즉 순전파, 손실 계산, 경사 계산, 경사 하강법을 통한 매개변수 갱신을 보여주는 가장 단순한 모델이다. NumPy로 바닥부터 구현해 보면 나중에 PyTorch가 자동화해 주는 수학적 토대가 드러나고, 학습이 실제로 어떻게 이루어지는지에 대한 깊은 직관이 생긴다.

## 1. 코드

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# 합성 데이터 생성: y = 2x + 1 + 잡음
n_samples = 100
X = np.random.rand(n_samples, 1) * 10
true_w, true_b = 2.0, 1.0
noise = np.random.randn(n_samples, 1) * 0.5
y = true_w * X + true_b + noise

# 매개변수를 초기화한다
w = np.random.randn(1, 1)
b = np.zeros((1, 1))

def forward(X, w, b):
    return X @ w + b

def compute_loss(y_true, y_pred):
    n = len(y_true)
    return (1 / n) * np.sum((y_true - y_pred) ** 2)

def compute_gradients(X, y_true, y_pred):
    n = len(y_true)
    dw = (2 / n) * X.T @ (y_pred - y_true)
    db = (2 / n) * np.sum(y_pred - y_true)
    return dw, db

# 학습 루프
learning_rate = 0.01
n_epochs = 100
loss_history = []

for epoch in range(n_epochs):
    y_pred = forward(X, w, b)
    loss = compute_loss(y, y_pred)
    loss_history.append(loss)
    dw, db = compute_gradients(X, y, y_pred)
    w = w - learning_rate * dw
    b = b - learning_rate * db

print(f"True values:    w = {true_w:.4f}, b = {true_b:.4f}")
print(f"Learned values: w = {w[0][0]:.4f}, b = {b[0][0]:.4f}")
print(f"Final loss: {loss_history[-1]:.4f}")
```

## 2. 논의

순전파는 선형 모델 $\hat{y} = Xw + b$으로 예측을 계산하며, 여기서 `@` 연산자가 행렬 곱을 수행한다. MSE 손실 함수 $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$은 예측과 참값의 평균 제곱 편차를 잰다. $w$과 $b$에 대해 미분하면 경사 $\frac{\partial L}{\partial w} = \frac{2}{n}X^T(\hat{y} - y)$과 $\frac{\partial L}{\partial b} = \frac{2}{n}\sum(\hat{y} - y)$을 얻는다.

경사 하강법은 경사의 반대 방향으로 움직여 매개변수를 갱신한다. $w \leftarrow w - \alpha \frac{\partial L}{\partial w}$이며 여기서 $\alpha$은 학습률이다. 학습률이 너무 크면 진동하거나 발산하고, 너무 작으면 견디기 힘들 만큼 수렴이 느려진다. 손실 기록은 단조롭게 감소하다가 매개변수가 최적값에 다가가면서 평평해지는 곡선을 보여야 한다.

이렇게 직접 구현하는 것은 교육적으로는 값지지만 더 큰 모델에는 비현실적이다. 신경망이 깊어질수록 경사를 손으로 계산하는 일은 불가능해진다. 다음 단계는 PyTorch 텐서를 쓰고 나아가 autograd를 쓰는 것인데, autograd는 계산 그래프에 연쇄 법칙을 적용해 바로 이 경사들을 자동으로 계산한다.

## 연습문제

**연습문제 1.**
학습률 0.001, 0.01, 0.1로 실험하라. 각각 100 에폭 동안 학습시키고 손실 곡선을 같은 그래프에 그려라. 어느 학습률에서 손실이 발산하는가?

??? success "연습문제 1 풀이"
    ```python
    for lr in [0.001, 0.01, 0.1]:
        w_temp = np.random.randn(1, 1)
        b_temp = np.zeros((1, 1))
        losses = []
        for epoch in range(100):
            y_pred = X @ w_temp + b_temp
            losses.append(compute_loss(y, y_pred))
            dw, db = compute_gradients(X, y, y_pred)
            w_temp -= lr * dw
            b_temp -= lr * db
        plt.plot(losses, label=f'lr={lr}')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    ```
    `lr=0.001`에서는 수렴이 느리지만 안정적이다. `lr=0.01`에서는 빠르게 수렴한다. `lr=0.1`에서는 갱신이 최적점을 지나쳐 버리므로 데이터의 규모에 따라 손실이 진동하거나 발산할 수 있다.

---

**연습문제 2.**
MSE 손실에 대한 경사 $\frac{\partial L}{\partial w}$을 제일원리에서 유도하라. 유도의 각 단계를 보여라.

??? success "연습문제 2 풀이"
    MSE 손실에서 출발한다.

    $$
    L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - (wx_i + b))^2
    $$

    연쇄 법칙을 적용한다.

    $$
    \frac{\partial L}{\partial w} = \frac{1}{n}\sum_{i=1}^{n} 2(y_i - wx_i - b)(-x_i) = \frac{2}{n}\sum_{i=1}^{n} x_i(\hat{y}_i - y_i)
    $$

    행렬 형태로는 $\frac{\partial L}{\partial w} = \frac{2}{n}X^T(\hat{y} - y)$이다. $\square$

---

**연습문제 3.**
입력 특징이 3개인 다변량 선형 회귀를 수행하도록 코드를 확장하라. $y = 2x_1 + 3x_2 - x_3 + 5 + \text{noise}$으로 데이터를 생성하고 경사 하강법이 참 계수를 되찾는지 확인하라.

??? success "연습문제 3 풀이"
    ```python
    n_samples = 200
    X = np.random.randn(n_samples, 3)
    true_w = np.array([[2.0], [3.0], [-1.0]])
    true_b = 5.0
    y = X @ true_w + true_b + np.random.randn(n_samples, 1) * 0.3

    w = np.random.randn(3, 1)
    b = np.zeros((1, 1))

    for epoch in range(500):
        y_pred = X @ w + b
        dw = (2 / n_samples) * X.T @ (y_pred - y)
        db = (2 / n_samples) * np.sum(y_pred - y)
        w -= 0.01 * dw
        b -= 0.01 * db

    print(f"Learned w: {w.flatten()}")  # [2, 3, -1]에 가까워야 한다
    print(f"Learned b: {b[0][0]:.4f}")  # 5에 가까워야 한다
    ```
    같은 경사 하강 알고리즘이 특징의 개수와 무관하게 동작한다. 가중치 행렬의 모양이 $(1,1)$에서 $(3,1)$로 바뀔 뿐 갱신 규칙은 그대로이다.

## 정리하며

**다룬 것** — 순수 NumPy로 만드는 선형 회귀

순전파는 선형 모델 $\hat{y} = Xw + b$으로 예측을 계산하며, 여기서 `@` 연산자가 행렬 곱을 수행한다.

앞의 연습문제 3개로 직접 확인할 수 있다.
