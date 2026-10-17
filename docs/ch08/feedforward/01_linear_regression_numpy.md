# 순수 NumPy로 만드는 선형 회귀

선형 회귀는 모든 뉴럴 넷트웍의 기초에 있는 핵심 동작, 즉 순전파, 로스 계산, 경사 계산, 경사 하강법을 통한 파라메터 갱신을 보여주는 가장 단순한 모델이다. NumPy로 바닥부터 구현해 보면 나중에 PyTorch가 자동화해 주는 수학적 토대가 드러나고, 학습이 실제로 어떻게 이루어지는지에 대한 깊은 직관이 생긴다.

## 1. 코드

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── 1. 답을 아는 데이터를 만든다 ────────────────────────────────────
# 참값을 알고 있어야 학습이 제대로 되었는지 판정할 수 있다.
# 넘파이만 쓰는 까닭은 순전파·로스·그래디언트·갱신이라는 네 스텝이
# 프레임워크의 도움 없이도 그저 행렬 연산일 뿐임을 보이려는 것이다.
n_samples = 100
X = np.random.rand(n_samples, 1) * 10       # [0, 10) 구간
true_w, true_b = 2.0, 1.0
noise = np.random.randn(n_samples, 1) * 0.5
y = true_w * X + true_b + noise

# ── 2. 파라메터를 초기화한다 ────────────────────────────────────────
# 웨이트는 무작위로, 바이어스는 0으로 둔다. 모양을 (1,1)로 잡아야
# 아래 X @ w 가 행렬곱으로 성립한다
w = np.random.randn(1, 1)
b = np.zeros((1, 1))

def forward(X, w, b):
    # b는 (1,1)이지만 방송(broadcast)되어 모든 표본에 같은 값이 더해진다
    return X @ w + b

def compute_loss(y_true, y_pred):
    # 평균제곱오차. n으로 나누므로 표본 수가 달라져도 로스의 자가 같다
    n = len(y_true)
    return (1 / n) * np.sum((y_true - y_pred) ** 2)

def compute_gradients(X, y_true, y_pred):
    # MSE = (1/n)*sum((y - Xw - b)^2) 를 손으로 미분한 것이다.
    #   dL/dw = (2/n) * X^T (y_pred - y_true)
    #   dL/db = (2/n) * sum(y_pred - y_true)
    # 잔차의 부호가 (y_pred - y_true)임에 주의한다. 뒤집으면 아래 갱신이
    # 로스를 키우는 쪽으로 올라가 버린다.
    # X.T를 곱하는 까닭은 각 표본의 잔차를 그 표본의 입력값으로 무게를
    # 주어 더하기 위해서다. 입력이 큰 표본이 웨이트를 더 많이 움직인다
    n = len(y_true)
    dw = (2 / n) * X.T @ (y_pred - y_true)
    db = (2 / n) * np.sum(y_pred - y_true)
    return dw, db

# ── 3. 학습 루프 ────────────────────────────────────────────────────
learning_rate = 0.01
n_epochs = 100
loss_history = []

for epoch in range(n_epochs):
    y_pred = forward(X, w, b)              # 순전파
    loss = compute_loss(y, y_pred)         # 얼마나 틀렸는지
    loss_history.append(loss)
    dw, db = compute_gradients(X, y, y_pred)   # 어느 쪽으로 고칠지
    # 그래디언트의 "반대" 방향으로 옮긴다. 부호를 빼먹으면 발산한다
    w = w - learning_rate * dw
    b = b - learning_rate * db

print(f"True values:    w = {true_w:.4f}, b = {true_b:.4f}")
print(f"Learned values: w = {w[0][0]:.4f}, b = {b[0][0]:.4f}")
# 로스가 0으로 가지 않는 것이 정상이다. 데이터에 표준편차 0.5짜리
# 잡음을 얹었으므로, 아무리 잘 맞춰도 그 잡음의 분산만큼은 남는다
print(f"Final loss: {loss_history[-1]:.4f}")
```

**출력:**

```
True values:    w = 2.0000, b = 1.0000
Learned values: w = 2.0464, b = 0.6563
Final loss: 0.2601
```

## 2. 논의

순전파는 선형 모델 $\hat{y} = Xw + b$으로 예측을 계산하며, 여기서 `@` 연산자가 행렬 곱을 수행한다. MSE 로스 함수 $L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$은 예측과 참값의 평균 제곱 편차를 잰다. $w$과 $b$에 대해 미분하면 경사 $\frac{\partial L}{\partial w} = \frac{2}{n}X^T(\hat{y} - y)$과 $\frac{\partial L}{\partial b} = \frac{2}{n}\sum(\hat{y} - y)$을 얻는다.

경사 하강법은 $w\leftarrow w-\alpha\nabla_wL$처럼 계수를 갱신한다. 적절한 러닝레이트의 전체 배치 선형 최소제곱에서는 로스가 감소하지만, 러닝레이트가 너무 크면 진동하거나 발산할 수 있다. 유한한 반복과 관측 잡음 때문에 계수가 데이터 생성의 참값과 정확히 같아질 필요는 없다.

직접 구현하면 순전파와 경사 계산의 대응을 확인할 수 있다. 모델이 커지면 수동 미분을 작성하고 검증하는 비용이 커지므로 자동 미분을 활용한다. 자동 미분도 같은 연쇄 법칙을 계산 그래프에 적용한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
러닝레이트 0.001, 0.01, 0.1로 실험하라. 각각 100 에포크 동안 학습시키고 로스 곡선을 같은 그래프에 그려라. 어느 러닝레이트에서 로스가 발산하는가?

</div>

??? success "풀이"
    ```python
    for lr in [0.001, 0.01, 0.1]:
        # 러닝레이트마다 웨이트를 새로 뽑는다. 이어 쓰면 두 번째 러닝레이트가
        # 이미 학습된 값에서 출발해 견주는 뜻이 없어진다. 다만 씨앗을
        # 다시 심지 않아 초기값이 셋 다 다르므로, 엄밀히 견주려면
        # 이 줄 앞에 np.random.seed(42)를 넣어야 한다
        w_temp = np.random.randn(1, 1)
        b_temp = np.zeros((1, 1))
        losses = []
        for epoch in range(100):
            y_pred = X @ w_temp + b_temp
            # 갱신 전 로스를 기록하므로, losses[0]은 학습을 시작하기
            # 전의 로스다. 곡선의 첫 점이 높은 것이 정상이다
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
    안정 범위는 절편 열을 포함한 설계행렬 $\tilde X$에 대해 $0<\alpha<1/\lambda_{\max}(\tilde X^\top\tilde X/n)$로 판단할 수 있다. 특정 러닝레이트의 안정성을 데이터 스케일과 무관하게 단정하지 않는다. 제시한 반복 횟수에서 로스 감소와 발산 여부를 구분한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
MSE 로스에 대한 경사 $\frac{\partial L}{\partial w}$을 제일원리에서 유도하라. 유도의 각 단계를 보여라.

</div>

??? success "풀이"
    MSE 로스에서 출발한다.

    $$
    L = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \frac{1}{n}\sum_{i=1}^{n}(y_i - (wx_i + b))^2
    $$

    연쇄 법칙을 적용한다.

    $$
    \frac{\partial L}{\partial w} = \frac{1}{n}\sum_{i=1}^{n} 2(y_i - wx_i - b)(-x_i) = \frac{2}{n}\sum_{i=1}^{n} x_i(\hat{y}_i - y_i)
    $$

    행렬 형태로는 $\frac{\partial L}{\partial w} = \frac{2}{n}X^T(\hat{y} - y)$이다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
입력 피처가 3개인 다변량 선형 회귀를 수행하도록 코드를 확장하라. $y = 2x_1 + 3x_2 - x_3 + 5 + \text{noise}$으로 데이터를 생성하고 경사 하강법이 참 계수를 되찾는지 확인하라.

</div>

??? success "풀이"
    ```python
    n_samples = 200
    X = np.random.randn(n_samples, 3)
    # 정답을 미리 정해 두고 데이터를 만든다. 그래야 학습이 끝난 뒤
    # 되찾은 값과 견주어 볼 수 있다. 현실의 데이터에는 없는 사치다
    true_w = np.array([[2.0], [3.0], [-1.0]])
    true_b = 5.0
    # 잡음이 있으므로 되찾은 값이 정확히 2, 3, -1이 되지는 않는다
    y = X @ true_w + true_b + np.random.randn(n_samples, 1) * 0.3

    # 모양만 (1,1)에서 (3,1)로 커졌다. 갱신 규칙은 아래에서 보듯 그대로다
    w = np.random.randn(3, 1)
    b = np.zeros((1, 1))

    for epoch in range(500):
        y_pred = X @ w + b
        # 연습문제 2에서 유도한 (2/n) X^T (y_pred - y)를 그대로 옮긴 것이다.
        # X.T가 (3, n)이고 오차가 (n, 1)이라 결과가 (3, 1)로 나와,
        # w와 모양이 저절로 맞는다
        dw = (2 / n_samples) * X.T @ (y_pred - y)
        # 바이어스의 그래디언트에는 X가 곱해지지 않는다. 바이어스에 대한 y_pred의
        # 편미분이 1이기 때문이다. 그래서 오차를 그냥 더하면 된다
        db = (2 / n_samples) * np.sum(y_pred - y)
        w -= 0.01 * dw
        b -= 0.01 * db

    print(f"Learned w: {w.flatten()}")  # [2, 3, -1]에 가까워야 한다
    print(f"Learned b: {b[0][0]:.4f}")  # 5에 가까워야 한다
    ```
    같은 경사 하강 알고리즘이 피처의 개수와 무관하게 동작한다. 웨이트 행렬의 모양이 $(1,1)$에서 $(3,1)$로 바뀔 뿐 갱신 규칙은 그대로이다.

## 정리하며

NumPy 예제는 MSE의 경사를 직접 계산해 선형 회귀를 학습한다. 로스는 잔차 제곱의 평균이며 경사에는 배율 `2/n`이 들어간다. 러닝레이트의 안정 범위는 입력 스케일에 의존하고, 유한한 반복에서 얻은 계수가 참 계수와 다를 수 있다.
