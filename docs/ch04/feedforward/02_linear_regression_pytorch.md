# PyTorch 텐서로 만드는 선형 회귀

PyTorch 텐서는 NumPy 배열의 GPU 가속 대응물이며 딥러닝 계산의 등뼈를 이룬다. 이 튜토리얼은 앞선 예제의 순수 NumPy 선형 회귀를 PyTorch로 옮기면서 장치 관리, 텐서 연산, 그리고 효율적인 학습과 추론에 꼭 필요해지는 `torch.no_grad()` 컨텍스트 관리자를 소개한다.

## 1. 코드

```python
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

n_samples = 100
X = torch.rand(n_samples, 1, device=device) * 10
true_w, true_b = 2.0, 1.0
noise = torch.randn(n_samples, 1, device=device) * 0.5
y = true_w * X + true_b + noise

w = torch.randn(1, 1, device=device)
b = torch.zeros(1, 1, device=device)

def forward(X, w, b):
    return X @ w + b

def compute_loss(y_true, y_pred):
    n = y_true.shape[0]
    return (1 / n) * torch.sum((y_true - y_pred) ** 2)

def compute_gradients(X, y_true, y_pred):
    n = y_true.shape[0]
    dw = (2 / n) * X.T @ (y_pred - y_true)
    db = (2 / n) * torch.sum(y_pred - y_true)
    return dw, db

learning_rate = 0.01
n_epochs = 100
loss_history = []

for epoch in range(n_epochs):
    y_pred = forward(X, w, b)
    loss = compute_loss(y, y_pred)
    loss_history.append(loss.item())
    dw, db = compute_gradients(X, y, y_pred)
    with torch.no_grad():
        w -= learning_rate * dw
        b -= learning_rate * db

print(f"True values:    w = {true_w:.4f}, b = {true_b:.4f}")
print(f"Learned values: w = {w.item():.4f}, b = {b.item():.4f}")
```

**출력:**

```
True values:    w = 2.0000, b = 1.0000
Learned values: w = 2.0336, b = 0.7309
```

## 2. 논의

NumPy에서 PyTorch로 넘어가는 일은 거의 문법 차원에 그친다. `np.random.randn`은 `torch.randn`이 되고, `np.sum`은 `torch.sum`이 되며, 배열 인덱싱은 똑같이 동작한다. 새로 더해지는 것은 장치 지정(`device=device`)과, 원소가 하나인 텐서에서 파이썬 스칼라를 꺼내는 `.item()` 메서드이다. 한 연산에 들어가는 모든 텐서는 같은 장치에 있어야 하므로 데이터와 매개변수를 같은 `device` 객체 위에 만든다.

`torch.no_grad()` 컨텍스트 관리자는 그 블록 안의 연산에 대해 경사 추적을 끈다. 이는 매개변수 갱신에서 결정적으로 중요하다. 이것이 없으면 PyTorch가 갱신 연산 자체를 추적하는 계산 그래프를 끝없이 키워 메모리를 낭비한다. 뒤의 튜토리얼에서는 직접 계산하던 경사를 autograd로 대체하지만, `torch.no_grad()`은 추론과 매개변수를 직접 다룰 때 여전히 필수적이다.

PyTorch 텐서는 `requires_grad` 깃발을 통해 자동 미분을 지원하지만 이 튜토리얼에서는 아직 켜지 않는다. 여기서 경사를 직접 계산하는 방식은 NumPy 버전을 그대로 옮긴 것이며, PyTorch 텐서가 GPU 계산과 경사 추적 능력을 더한 NumPy 배열의 엄격한 상위집합임을 보여준다.

## 연습문제

**연습문제 1.**
결과를 다시 NumPy로 바꾸어 정확성을 확인하라. 학습 후 `w`와 `b`를 CPU로 옮기고 NumPy 배열로 바꾼 뒤 이미 아는 참값과 비교하라. 각 매개변수의 백분율 오차를 계산하라.

??? success "연습문제 1 풀이"
    ```python
    w_np = w.cpu().numpy()
    b_np = b.cpu().numpy()
    w_error = abs(w_np[0][0] - true_w) / true_w * 100
    b_error = abs(b_np[0][0] - true_b) / true_b * 100
    print(f"Weight error: {w_error:.2f}%")
    print(f"Bias error: {b_error:.2f}%")
    ```
    `lr=0.01`로 100 에폭을 학습한 뒤에는 오차가 작아야 한다(5% 미만). 텐서가 GPU에 있을 때는 `.cpu()` 호출이 필요하며, `.numpy()`가 NumPy 배열로 바꿔 준다.

---

**연습문제 2.**
한 연산에 들어가는 모든 텐서가 같은 장치에 있어야 하는 이유를 설명하라. CPU 텐서와 GPU 텐서를 더하려 하면 PyTorch는 어떤 오류 메시지를 내는가?

??? success "연습문제 2 풀이"
    PyTorch는 "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu."와 같은 메시지와 함께 `RuntimeError`를 낸다. GPU 메모리와 CPU 메모리가 물리적으로 분리된 주소 공간이기 때문이다. 데이터는 `.to(device)`, `.cpu()`, `.cuda()`를 써서 명시적으로 옮겨야 한다. 자동으로 옮겨 준다면 비용이 클 수 있는 메모리 복사가 감춰지므로, PyTorch는 성능 특성이 투명하게 드러나도록 명시적인 장치 관리를 요구한다.

---

**연습문제 3.**
`n_samples=10000`으로 1000 에폭의 시간을 재어 CPU와 (가능하다면) GPU에서 학습 루프를 견주어 보라. 실제 소요 시간은 `time.time()`으로 측정하라. 데이터 크기가 얼마부터 GPU 가속이 값어치를 하는가?

??? success "연습문제 3 풀이"
    ```python
    import time

    for dev in ['cpu', 'cuda']:
        if dev == 'cuda' and not torch.cuda.is_available():
            continue
        device = torch.device(dev)
        X = torch.rand(10000, 1, device=device) * 10
        y = 2.0 * X + 1.0 + torch.randn(10000, 1, device=device) * 0.5
        w = torch.randn(1, 1, device=device)
        b = torch.zeros(1, 1, device=device)

        start = time.time()
        for _ in range(1000):
            y_pred = X @ w + b
            dw = (2 / 10000) * X.T @ (y_pred - y)
            db = (2 / 10000) * torch.sum(y_pred - y)
            with torch.no_grad():
                w -= 0.01 * dw
                b -= 0.01 * db
        elapsed = time.time() - start
        print(f"{dev}: {elapsed:.3f}s")
    ```
    작은 데이터셋(표본 약 1000개 미만)에서는 GPU 커널을 띄우는 부담 때문에 CPU가 더 빠른 경우가 많다. GPU 가속은 대체로 표본이 10,000개를 넘거나 행렬의 차원이 GPU의 병렬성을 채울 만큼 클 때 값어치를 한다.

## 정리하며

**다룬 것** — PyTorch 텐서로 만드는 선형 회귀

NumPy에서 PyTorch로 넘어가는 일은 거의 문법 차원에 그친다.

앞의 연습문제 3개로 직접 확인할 수 있다.
