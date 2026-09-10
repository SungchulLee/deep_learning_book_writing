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
    # 각도는 0에서 2pi까지 고르게 뿌리고, 반지름만 두 무리로 갈라
    # 극좌표로 점을 찍는다. 안쪽 무리는 반지름 0~2인 원판(이름표 0),
    # 바깥 무리는 3~5인 고리(이름표 1)다. 두 무리 사이에 폭 1의 빈 띠가
    # 있어 분리는 되지만, 어떤 직선으로도 나눌 수 없다.
    # 은닉층 없이 선형 층 하나로는 못 푸는 문제를 일부러 만든 것이다
    angles = torch.rand(n_samples) * 2 * np.pi
    radii = torch.zeros(n_samples)
    radii[:n_samples//2] = torch.rand(n_samples//2) * 2
    radii[n_samples//2:] = torch.rand(n_samples//2) * 2 + 3
    X = torch.stack([radii * torch.cos(angles),
                     radii * torch.sin(angles)], dim=1)
    y = torch.zeros(n_samples, 1)
    y[n_samples//2:] = 1
    # 빈 띠의 폭이 1인데 잡음의 표준편차가 0.3이라, 경계 근처가 조금
    # 흐려지되 두 무리가 뒤섞이지는 않는다
    X += torch.randn_like(X) * 0.3
    return X, y

X, y = generate_circle_data(n_samples)
X, y = X.to(device), y.to(device)

# 가중치를 직접 들고 다닌다. nn.Module도 optimizer도 쓰지 않으므로
# requires_grad를 켜지 않는다. 아래 backward 함수가 자동 미분을 대신한다.
input_size, hidden_size, output_size = 2, 8, 1
# sqrt(2/fan_in) 배는 He 초기화다. ReLU가 입력의 절반을 0으로 죽이는 만큼
# 분산이 반토막 나므로, 2를 곱해 그 손실을 미리 메워 둔다
w1 = torch.randn(input_size, hidden_size, device=device) * np.sqrt(2.0 / input_size)
# 편향은 모양이 (1, hidden)이다. 배치 축으로 방송되어 표본마다 같은 값이
# 더해진다. 0에서 시작해도 대칭이 깨지지 않는 까닭은 w가 무작위이기 때문이다
b1 = torch.zeros(1, hidden_size, device=device)
w2 = torch.randn(hidden_size, output_size, device=device) * np.sqrt(2.0 / hidden_size)
b2 = torch.zeros(1, output_size, device=device)

def relu(x):
    return torch.maximum(x, torch.tensor(0.0, device=x.device))

def relu_derivative(x):
    # x = 0에서 ReLU는 미분할 수 없다. 여기서는 도함수를 0으로 정하는데,
    # 부분미분 가운데 아무 값이나 골라도 되고 실전에서 차이는 없다
    return (x > 0).float()

def sigmoid(x):
    # x가 크게 음수면 exp(-x)가 넘쳐 inf가 되지만, 1/(1+inf)가 0으로
    # 떨어져 NaN 없이 넘어간다. 대신 정확히 0이 나오므로 아래 손실에서
    # log(0)을 막아 주어야 한다
    return 1 / (1 + torch.exp(-x))

def forward(X, w1, b1, w2, b2):
    z1 = X @ w1 + b1
    a1 = relu(z1)
    z2 = a1 @ w2 + b2
    a2 = sigmoid(z2)
    # 중간값을 남겨 두는 것이 순전파의 숨은 절반이다. 역전파는 z1의
    # 부호와 a1의 값을 다시 필요로 하는데, 여기서 버리면 그때 다시
    # 계산해야 한다. 메모리를 써서 계산을 아끼는 맞바꿈이며,
    # PyTorch의 자동 미분도 속으로 똑같은 일을 한다
    cache = {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}
    return a2, cache

def compute_loss(y_true, y_pred):
    # 예측이 정확히 0이나 1이면 log가 발산한다. 양 끝을 잘라 내
    # 손실이 무한대가 되는 것을 막는다
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    # 이진 교차 엔트로피. 이름표가 1이면 앞항만, 0이면 뒷항만 살아남는다
    return -torch.mean(y_true * torch.log(y_pred) +
                       (1 - y_true) * torch.log(1 - y_pred))

def backward(X, y_true, y_pred, cache, w2):
    n = X.shape[0]
    a1, a2 = cache['a1'], cache['a2']
    z1 = cache['z1']
    # 시그모이드와 교차 엔트로피를 이어 붙이면 도함수가 이렇게 깔끔하게
    # 줄어든다. 손실의 1/y_pred와 시그모이드의 a2(1-a2)가 서로 지워져
    # "예측 빼기 정답"만 남는다. 그래서 여기 시그모이드의 도함수가
    # 따로 보이지 않는다
    dz2 = a2 - y_true
    # 손실이 평균이므로 1/n을 여기서 곱한다. a1.T @ dz2는 표본별 기여를
    # 이미 더하고 있으니, 나누어 주어야 평균 기울기가 된다
    dw2 = (1 / n) * a1.T @ dz2
    db2 = (1 / n) * torch.sum(dz2, dim=0, keepdim=True)
    # 여기서부터가 "역"전파다. 오차 신호를 w2의 전치로 되밀어
    # 은닉층의 활성 a1이 진 책임을 나눈다
    da1 = dz2 @ w2.T
    # ReLU가 문지기 노릇을 한다. 순전파에서 z1이 양수였던 자리로만
    # 신호를 통과시키고 나머지는 막는다. 활성 뒤의 a1이 아니라
    # 활성 전의 z1로 판정해야 한다는 점에 주의하라
    dz1 = da1 * relu_derivative(z1)
    dw1 = (1 / n) * X.T @ dz1
    db1 = (1 / n) * torch.sum(dz1, dim=0, keepdim=True)
    return dw1, db1, dw2, db2

learning_rate = 0.1
for epoch in range(1000):
    # 200개를 통째로 넣는 전배치 경사 하강이다. 미니배치가 없으므로
    # 에포크 하나가 곧 한 걸음이다
    y_pred, cache = forward(X, w1, b1, w2, b2)
    loss = compute_loss(y, y_pred)
    dw1, db1, dw2, db2 = backward(X, y, y_pred, cache, w2)
    # 애초에 requires_grad를 켜지 않았으므로 no_grad는 사실 필요 없다.
    # 다만 이 자리가 optimizer.step()에 해당한다는 표시로 남겨 둔다
    with torch.no_grad():
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1
        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2

# 주의: y_pred는 마지막 갱신 "전"의 예측이다. 갱신 뒤의 정확도를 재려면
# forward를 한 번 더 불러야 한다. 여기서는 이미 수렴한 뒤라 차이가 없다
predictions = (y_pred > 0.5).float()
accuracy = (predictions == y).float().mean().item() * 100
print(f"Final Accuracy: {accuracy:.2f}%")
```

**출력:**

```
Final Accuracy: 100.00%
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
