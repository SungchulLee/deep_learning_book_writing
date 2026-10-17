# PyTorch 텐서로 만드는 선형 회귀

앞의 선형 회귀를 PyTorch 텐서로 옮긴다. 이 단계에서는 autograd를 켜지 않고 경사를 직접 계산하여, 장치 지정과 텐서 연산의 역할을 구분한다.

## 1. 코드

```python
import torch
import matplotlib.pyplot as plt

torch.manual_seed(42)

# GPU가 있으면 쓰고 없으면 CPU를 쓴다. 아래에서 만드는 텐서를 모두
# 같은 device에 두어야 한다. 하나라도 다른 곳에 있으면 연산에서 오류가 난다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 1. 답을 아는 데이터를 만든다 ────────────────────────────────────
n_samples = 100
X = torch.rand(n_samples, 1, device=device) * 10   # [0, 10) 구간에 고르게
true_w, true_b = 2.0, 1.0                          # 되찾아야 할 참값
noise = torch.randn(n_samples, 1, device=device) * 0.5
y = true_w * X + true_b + noise

# ── 2. 파라메터를 손으로 만든다 ─────────────────────────────────────
# requires_grad를 켜지 않는다. 이 예제의 요점이 autograd 없이
# 그래디언트를 손으로 셈해 보는 것이기 때문이다
w = torch.randn(1, 1, device=device)   # (1,1): X @ w 가 되도록 행렬 꼴
b = torch.zeros(1, 1, device=device)

def forward(X, w, b):
    # 행렬곱 뒤에 b를 더한다. b는 (1,1)이지만 방송(broadcast)되어
    # 모든 표본에 같은 값이 더해진다
    return X @ w + b

def compute_loss(y_true, y_pred):
    # 평균제곱오차. n으로 나누므로 표본 수가 달라져도 로스의 자가 같다
    n = y_true.shape[0]
    return (1 / n) * torch.sum((y_true - y_pred) ** 2)

def compute_gradients(X, y_true, y_pred):
    # MSE = (1/n) * sum((y - Xw - b)^2) 를 손으로 미분한 것이다.
    #   dL/dw = (2/n) * X^T (y_pred - y_true)
    #   dL/db = (2/n) * sum(y_pred - y_true)
    # 잔차의 부호에 주의한다. (y_pred - y_true)이지 그 반대가 아니다.
    # 뒤집으면 로스가 커지는 쪽으로 올라간다
    n = y_true.shape[0]
    dw = (2 / n) * X.T @ (y_pred - y_true)
    db = (2 / n) * torch.sum(y_pred - y_true)
    return dw, db

# ── 3. 경사 하강 루프 ───────────────────────────────────────────────
learning_rate = 0.01
n_epochs = 100
loss_history = []

for epoch in range(n_epochs):
    y_pred = forward(X, w, b)
    loss = compute_loss(y, y_pred)
    loss_history.append(loss.item())   # item()으로 수만 담는다

    dw, db = compute_gradients(X, y, y_pred)

    # no_grad가 굳이 필요하지는 않다(requires_grad를 켜지 않았으므로).
    # 그래도 두는 까닭은 파라메터 갱신을 그래프 밖에서 한다는 버릇을
    # 들이기 위해서다. autograd를 쓰는 다음 예제에서는 이것이 필수가 된다
    with torch.no_grad():
        w -= learning_rate * dw   # 그래디언트 반대 방향으로
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

NumPy에서 PyTorch로 넘어가는 일은 거의 문법 차원에 그친다. `np.random.randn`은 `torch.randn`이 되고, `np.sum`은 `torch.sum`이 되며, 배열 인덱싱은 똑같이 동작한다. 새로 더해지는 것은 장치 지정(`device=device`)과, 원소가 하나인 텐서에서 파이썬 스칼라를 꺼내는 `.item()` 메서드이다. 한 연산에 들어가는 모든 텐서는 같은 장치에 있어야 하므로 데이터와 파라메터를 같은 `device` 객체 위에 만든다.

`torch.no_grad()`는 블록 안에서 경사 기록을 끈다. 이 예제의 파라메터는 `requires_grad=False`이므로 갱신에 원래 계산 그래프가 생기지 않는다. 뒤에서 학습 가능한 잎 텐서를 직접 제자리 갱신할 때는 경사 기록을 끄는 것이 필요하며, 그렇지 않으면 자동 미분의 제자리 연산 제약으로 오류가 날 수 있다.

PyTorch와 NumPy는 비슷한 연산을 제공하지만 자료형 변환, 장치, 메모리 공유와 API에 차이가 있다. NumPy 배열의 모든 기능을 그대로 포함하는 상위집합으로 생각하기보다, 배열 연산과 자동 미분을 지원하는 별도의 도구로 이해한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
결과를 다시 NumPy로 바꾸어 정확성을 확인하라. 학습 후 `w`와 `b`를 CPU로 옮기고 NumPy 배열로 바꾼 뒤 이미 아는 참값과 비교하라. 각 파라메터의 백분율 오차를 계산하라.

</div>

??? success "풀이"
    ```python
    w_np = w.cpu().numpy()
    b_np = b.cpu().numpy()
    w_error = abs(w_np[0][0] - true_w) / true_w * 100
    b_error = abs(b_np[0][0] - true_b) / true_b * 100
    print(f"Weight error: {w_error:.2f}%")
    print(f"Bias error: {b_error:.2f}%")
    ```
    파라메터의 참값과 학습값을 비교하되, 잡음과 유한한 반복 때문에 오차가 특정 비율 이하라고 보장할 수는 없다. 경사를 추적하는 텐서를 NumPy로 바꿀 때는 일반적으로 `detach().cpu().numpy()`를 사용한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
한 연산에 들어가는 모든 텐서가 같은 장치에 있어야 하는 이유를 설명하라. CPU 텐서와 GPU 텐서를 더하려 하면 PyTorch는 어떤 오류 메시지를 내는가?

</div>

??? success "풀이"
    PyTorch는 "Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu."와 같은 메시지와 함께 `RuntimeError`를 낸다. GPU 메모리와 CPU 메모리가 물리적으로 분리된 주소 공간이기 때문이다. 데이터는 `.to(device)`, `.cpu()`, `.cuda()`를 써서 명시적으로 옮겨야 한다. 자동으로 옮겨 준다면 비용이 클 수 있는 메모리 복사가 감춰지므로, PyTorch는 성능 특성이 투명하게 드러나도록 명시적인 장치 관리를 요구한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
`n_samples=10000`으로 1000 에포크의 시간을 재어 CPU와 (가능하다면) GPU에서 학습 루프를 비교하여 보라. 실제 소요 시간은 `time.time()`으로 측정하라. 데이터 크기가 얼마부터 GPU 가속이 값어치를 하는가?

</div>

??? success "풀이"
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
    GPU의 비동기 실행 때문에 시간 측정의 시작과 끝에서 동기화가 필요하다. 아래 예시 측정 코드에 동기화가 없다면 순수 계산 시간을 정확히 비교하기 어렵다. 워밍업과 데이터 전송 포함 여부를 명시하고, 표본 수뿐 아니라 피처 차원·장치·배치 크기를 함께 기록한다. 보편적인 손익분기 표본 수는 없다.

## 정리하며

PyTorch로 옮겨도 선형 회귀의 식은 같다. 이 예제는 수동 경사 계산을 유지하므로 텐서 연산과 autograd를 구분할 수 있다. CPU와 GPU 비교에는 비동기 실행의 동기화와 전송 비용을 반영해야 한다.
