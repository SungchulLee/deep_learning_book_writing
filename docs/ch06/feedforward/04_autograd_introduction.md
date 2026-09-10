# Autograd 소개

PyTorch의 autograd 체계는 계산 그래프를 따라 자동으로 미분하여 경사를 손으로 계산할 필요를 없앤다. 텐서에 `requires_grad=True`를 설정하면 PyTorch가 그 텐서에 대한 모든 연산을 추적하고, 스칼라 출력에 `.backward()`를 호출하면 그래프를 역방향으로 훑으며 계산된 도함수로 `.grad` 속성을 채운다. 이로써 수십 줄에 이르던 역전파 코드가 함수 호출 한 번으로 줄어든다.

## 1. 코드

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(42)

# 자동 미분 기초: x = 3에서 y = x^2의 dy/dx
# requires_grad=True가 "이 텐서에 얽힌 연산을 기록해 두라"는 표시다.
# 이것을 켠 텐서만 잎(leaf)이 되어 나중에 .grad를 받는다
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
# backward()는 스칼라에만 부를 수 있다. y가 벡터라면 어느 방향으로
# 미분할지 알 수 없어, gradient 인자를 따로 넘겨야 한다
y.backward()
print(f"dy/dx at x=3: {x.grad.item()}")  # 2*3 = 6

# 다변수: z = 3a^2 + 2b
a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
z = 3 * a**2 + 2 * b
z.backward()
print(f"dz/da = {a.grad.item()}, dz/db = {b.grad.item()}")

# 자동 미분을 쓰는 신경망
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

# 앞 절(03)과 같은 문제를 같은 구조로 푼다. 다른 것은 requires_grad를
# 켰다는 점 하나이고, 그 대가로 backward 함수를 손으로 쓰지 않아도 된다.
# 주의: 앞 절과 달리 He 초기화(sqrt(2/fan_in) 곱하기)가 빠져 있다.
# 층이 둘뿐이라 큰 탈은 없지만, 깊어지면 이 차이가 드러난다
w1 = torch.randn(2, 8, device=device, requires_grad=True)
b1 = torch.zeros(1, 8, device=device, requires_grad=True)
w2 = torch.randn(8, 1, device=device, requires_grad=True)
b2 = torch.zeros(1, 1, device=device, requires_grad=True)

def forward(X, w1, b1, w2, b2):
    # 앞 절에서는 역전파에 쓰려고 z1, a1을 cache에 담아 두었다. 여기서는
    # 그럴 필요가 없다. autograd가 연산을 따라가며 필요한 중간값을
    # 스스로 붙들고 있기 때문이다
    z1 = X @ w1 + b1
    a1 = torch.relu(z1)
    z2 = a1 @ w2 + b2
    return torch.sigmoid(z2)

def compute_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    return -torch.mean(y_true * torch.log(y_pred) +
                       (1 - y_true) * torch.log(1 - y_pred))

learning_rate = 0.1
for epoch in range(1000):
    # 기울기는 덮어쓰기가 아니라 누적이므로 걸음마다 지워야 한다.
    # 이 네 줄이 곧 optimizer.zero_grad()가 하는 일이다.
    # 첫 걸음에서는 아직 .grad가 없어 None이라, 검사가 필요하다
    if w1.grad is not None:
        w1.grad.zero_()
        b1.grad.zero_()
        w2.grad.zero_()
        b2.grad.zero_()

    y_pred = forward(X, w1, b1, w2, b2)
    loss = compute_loss(y, y_pred)
    # 이 한 줄이 앞 절의 backward 함수 전체를 대신한다. 연쇄 법칙을
    # 손으로 펼쳐 쓴 열 줄 남짓이 여기로 접혔다
    loss.backward()

    # 여기서는 no_grad가 정말로 필요하다. w1은 requires_grad가 켜진
    # 잎이라, 그냥 빼면 그 뺄셈까지 그래프에 기록되어 다음 backward가
    # 엉키고 메모리도 계속 불어난다. 앞 절(03)에서는 requires_grad를
    # 켜지 않아 no_grad가 장식이었지만, 이 절에서는 문법적 요구다
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        b1 -= learning_rate * b1.grad
        w2 -= learning_rate * w2.grad
        b2 -= learning_rate * b2.grad

with torch.no_grad():
    y_pred_final = forward(X, w1, b1, w2, b2)
    predictions = (y_pred_final > 0.5).float()
    accuracy = (predictions == y).float().mean().item() * 100
print(f"Final Accuracy: {accuracy:.2f}%")
```

**출력:**

```
dy/dx at x=3: 6.0
dz/da = 12.0, dz/db = 2.0
Final Accuracy: 100.00%
```

## 2. 논의

계산 그래프는 순전파 중에 동적으로 만들어진다. `requires_grad=True`인 텐서에 대한 모든 연산은 그 연산과 입력을 기록하는 노드를 그래프에 만든다. 손실에 `.backward()`를 호출하면 PyTorch가 이 그래프를 출력에서 입력 쪽으로 훑으며 각 노드에서 연쇄 법칙을 적용해 경사를 계산한다. 이 경사들은 잎 텐서(매개변수)의 `.grad` 속성에 누적된다.

경사 누적은 중요한 세부 사항이다. 기본적으로 `.backward()`는 기존 경사를 덮어쓰지 않고 거기에 더한다. 매 역전파 전에 `.zero_()`를 호출해야 하는 이유가 여기 있다. 0으로 만들지 않으면 앞선 반복의 경사가 쌓여 잘못된 갱신으로 이어진다. 이런 동작이 있는 것은 경사 누적이 더 큰 배치 크기를 흉내 내거나 특정 최적화 알고리즘을 구현하는 데 유용하기 때문이다.

매개변수 갱신에서 쓰는 `torch.no_grad()` 문맥은 PyTorch가 뺄셈 연산을 추적하지 못하게 막는다. 이것이 없으면 계산 그래프가 반복마다 자라 메모리를 잡아먹다가 결국 멈춰 버린다. 경사 0으로 만들기, 순전파, 역전파, `no_grad` 아래에서의 갱신으로 이어지는 이 패턴이 모든 PyTorch 학습 루프의 본보기가 된다.

## 연습문제

**연습문제 1.**
autograd로 점 $(2, 3)$에서 $f(x, y) = x^2 y + y^3$의 경사를 계산하라. 편도함수를 손으로 계산하여 답을 확인하라.

??? success "연습문제 1 풀이"
    ```python
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)
    f = x**2 * y + y**3
    f.backward()
    print(f"df/dx = {x.grad.item()}")  # 2xy = 2*2*3 = 12
    print(f"df/dy = {y.grad.item()}")  # x^2 + 3y^2 = 4 + 27 = 31
    ```
    손으로 계산하면 $\frac{\partial f}{\partial x} = 2xy = 12$이고 $\frac{\partial f}{\partial y} = x^2 + 3y^2 = 4 + 27 = 31$이다. 둘 다 autograd의 결과와 일치한다.

---

**연습문제 2.**
경사를 0으로 만들지 않고 `.backward()`를 두 번 호출하여 경사 누적을 보여라. 경사가 두 배가 됨을 보여라. 그다음 `.zero_()`로 바로잡아라.

??? success "연습문제 2 풀이"
    ```python
    x = torch.tensor(3.0, requires_grad=True)
    y1 = x ** 2
    y1.backward()
    print(f"After first backward: {x.grad.item()}")  # 6

    y2 = x ** 2
    y2.backward()
    print(f"After second backward: {x.grad.item()}")  # 12 (쌓였다!)

    x.grad.zero_()
    y3 = x ** 2
    y3.backward()
    print(f"After zero + backward: {x.grad.item()}")  # 6 (옳다)
    ```
    0으로 만들지 않으면 두 번째 `.backward()`가 기존의 6에 6을 더해 12가 된다. `.zero_()` 후에는 경사가 올바르게 6으로 계산된다.

---

**연습문제 3.**
직접 하던 매개변수 갱신을 `torch.optim.SGD`로 대체하라. 최적화기를 만들고, 경사를 손수 0으로 만들고 빼던 것을 `optimizer.zero_grad()`와 `optimizer.step()`으로 바꿔라.

??? success "연습문제 3 풀이"
    ```python
    params = [w1, b1, w2, b2]
    optimizer = torch.optim.SGD(params, lr=0.1)

    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = forward(X, w1, b1, w2, b2)
        loss = compute_loss(y, y_pred)
        loss.backward()
        optimizer.step()
    ```
    `optimizer.zero_grad()`이 손수 하던 `.grad.zero_()` 호출을 대신하고, `optimizer.step()`이 손수 쓰던 `with torch.no_grad(): w -= lr * w.grad` 패턴을 대신한다. 이 편이 깔끔하고, 코드를 고치지 않고도 Adam 같은 고급 최적화기를 쓸 수 있다.

## 정리하며

**다룬 것** — Autograd 소개

계산 그래프는 순전파 중에 동적으로 만들어진다.

앞의 연습문제 3개로 직접 확인할 수 있다.
