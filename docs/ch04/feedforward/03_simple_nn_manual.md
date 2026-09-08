# 역전파를 직접 구현한 간단한 신경망

선형 층 하나로는 동심원을 분류하는 것 같은 비선형 문제를 풀 수 없다. 비선형 활성화 함수(ReLU)를 쓰는 은닉층을 더하면 신경망이 복잡한 결정 경계를 배울 수 있는 능력을 얻는다. 이 튜토리얼은 순전파, 이진 교차 엔트로피 손실, 그리고 완전한 역전파를 바닥부터 구현하며, 연쇄 법칙이 층을 거슬러 오차 신호를 어떻게 전파하는지 보여준다.

## 1. 코드

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_samples = 200

def generate_circle_data(n_samples):
    angles = torch.rand(n_samples) * 2 * np.pi
    radii = torch.zeros(n_samples)
    radii[:n_samples//2] = torch.rand(n_samples//2) * 2
    radii[n_samples//2:] = torch.rand(n_samples//2) * 2 + 3
    X = torch.stack([radii * torch.cos(angles),
                     radii * torch.sin(angles)], dim=1)
    y = torch.zeros(n_samples, 1)
    y[n_samples//2:] = 1
    X += torch.randn_like(X) * 0.3
    return X, y

X, y = generate_circle_data(n_samples)
X, y = X.to(device), y.to(device)

input_size, hidden_size, output_size = 2, 8, 1
w1 = torch.randn(input_size, hidden_size, device=device) * np.sqrt(2.0 / input_size)
b1 = torch.zeros(1, hidden_size, device=device)
w2 = torch.randn(hidden_size, output_size, device=device) * np.sqrt(2.0 / hidden_size)
b2 = torch.zeros(1, output_size, device=device)

def relu(x):
    return torch.maximum(x, torch.tensor(0.0, device=x.device))

def relu_derivative(x):
    return (x > 0).float()

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def forward(X, w1, b1, w2, b2):
    z1 = X @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache

def compute_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    return -torch.mean(y_true * torch.log(y_pred) +
                       (1 - y_true) * torch.log(1 - y_pred))

def backward(X, y_true, y_pred, cache, w2):
    n = X.shape[0]
    a1, a2 = cache['a1'], cache['a2']
    z1 = cache['z1']
    dz2 = a2 - y_true
    dw2 = (1 / n) * a1.T @ dz2
    db2 = (1 / n) * torch.sum(dz2, dim=0, keepdim=True)
    da1 = dz2 @ w2.T
    dz1 = da1 * relu_derivative(z1)
    dw1 = (1 / n) * X.T @ dz1
    db1 = (1 / n) * torch.sum(dz1, dim=0, keepdim=True)
    return dw1, db1, dw2, db2

learning_rate = 0.1
for epoch in range(1000):
    y_pred, cache = forward(X, w1, b1, w2, b2)
    loss = compute_loss(y, y_pred)
    dw1, db1, dw2, db2 = backward(X, y, y_pred, cache, w2)
    with torch.no_grad():
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2

predictions = (y_pred > 0.5).float()
accuracy = (predictions == y).float().mean().item() * 100
print(f"Final Accuracy: {accuracy:.2f}%")
```

## 2. 논의

순전파는 데이터를 두 번의 변환에 흘려보낸다. 입력이 은닉층으로 선형 사상되어 ReLU로 활성화되고, 다시 출력으로 선형 사상되어 시그모이드로 활성화된다. ReLU 활성화가 없으면 두 선형 변환의 합성이 하나의 선형 변환으로 주저앉아, 여기서 쓰는 원형 패턴 같은 비선형 결정 경계를 배울 수 없게 된다.

역전파는 연쇄 법칙을 써서 층마다 경사를 계산한다. 출력층의 경사 $\frac{\partial L}{\partial z_2} = a_2 - y$은 시그모이드 활성화와 이진 교차 엔트로피 손실의 편리한 조합에서 나온다. 그다음 경사가 뒤로 흐른다. $\frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial z_2} \cdot w_2^T$이고, ReLU의 도함수가 문지기 노릇을 하여 활성화 전 값이 양수였던 곳으로만 경사를 통과시킨다. 순전파 중에 중간값을 저장해 두는 이유가 여기 있다. 역전파에 그 값들이 필요하다.

분류에서 이진 교차 엔트로피 손실 $L = -\frac{1}{n}\sum[y\log(\hat{y}) + (1-y)\log(1-\hat{y})]$을 MSE보다 선호하는 것은, 예측이 확신에 차 있으면서 틀렸을 때 더 강한 경사를 주기 때문이다. 엡실론으로 범위를 제한하면 $\log(0)$에서 오는 수치적 문제를 막을 수 있다.

## 연습문제

**연습문제 1.**
ReLU 활성화를 없애고(`a1 = relu(z1)`를 `a1 = z1`로 바꾼다) 다시 학습시켜라. 정확도에 무슨 일이 일어나는가? 그 이유를 수학적으로 설명하라.

??? success "연습문제 1 풀이"
    ReLU가 없으면 신경망은 $\hat{y} = \sigma((Xw_1 + b_1)w_2 + b_2) = \sigma(Xw_1w_2 + b_1w_2 + b_2)$을 계산한다. $w_1w_2$은 또 하나의 행렬일 뿐이고 $b_1w_2 + b_2$은 편향이므로, 이는 시그모이드 출력을 갖는 단층 선형 분류기로 주저앉는다. 하나의 선형 경계로는 동심원을 나눌 수 없으므로 정확도는 50% 언저리를 맴돈다. 신경망이 비선형 함수를 표현할 수 있게 만드는 것이 바로 활성화 함수이다.

---

**연습문제 2.**
은닉층 크기를 8에서 32로, 그리고 8에서 2로 바꾸어 보라. 정확도와 수렴 속도를 비교하라. 90%를 넘는 정확도를 얻는 최소 은닉층 크기는 얼마인가?

??? success "연습문제 2 풀이"
    `hidden_size=32`에서는 모델의 용량이 커서 더 빠르게 수렴하며 95% 이상의 정확도에 이르기 쉽다. `hidden_size=2`에서는 은닉 뉴런 두 개로 원형 경계를 표현할 힘이 모자라 모델이 애를 먹는다. 실험적으로 이 과제에서 90%를 넘기는 최소 크기는 대체로 `hidden_size=4`이지만, 결과는 무작위 초기화에 따라 달라진다.

---

**연습문제 3.**
3층 신경망(은닉층 두 개)의 역전파를 구현하라. 기존 은닉층과 출력 사이에 뉴런 4개짜리 두 번째 은닉층을 추가하라. 경사 계산을 유도하고 구현하라.

??? success "연습문제 3 풀이"
    ```python
    # 추가 매개변수
    w3 = torch.randn(4, 1, device=device) * np.sqrt(2.0 / 4)
    b3 = torch.zeros(1, 1, device=device)
    w2_new = torch.randn(8, 4, device=device) * np.sqrt(2.0 / 8)
    b2_new = torch.zeros(1, 4, device=device)

    # 순전파: X -> z1 -> a1(relu) -> z2 -> a2(relu) -> z3 -> a3(sigmoid)
    # 역전파에는 연쇄 법칙 단계가 하나 더 붙는다:
    # dz3 = a3 - y
    # dw3 = (1/n) * a2.T @ dz3
    # da2 = dz3 @ w3.T
    # dz2 = da2 * relu_derivative(z2)
    # dw2 = (1/n) * a1.T @ dz2
    # da1 = dz2 @ w2_new.T
    # dz1 = da1 * relu_derivative(z1)
    # dw1 = (1/n) * X.T @ dz1
    ```
    층을 하나 더할 때마다 연쇄 법칙을 한 번 더 적용하게 된다. 형태는 언제나 같다. $\frac{\partial L}{\partial z_l}$을 계산하고, 그것으로 $w_l$과 $b_l$의 경사를 구한 뒤, $a_{l-1}$으로 거슬러 전파하며 활성화의 도함수를 적용한다.

## 정리하며

**다룬 것** — 역전파를 직접 구현한 간단한 신경망

순전파는 데이터를 두 번의 변환에 흘려보낸다.

앞의 연습문제 3개로 직접 확인할 수 있다.
