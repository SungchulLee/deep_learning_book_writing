# PyTorch 기초

PyTorch는 깊은 배움 연구에서 으뜸가는 얼거리이며 실전 펼치기에도 점점 많이 쓰인다. 이 길잡이는 바탕 벽돌인 텐서 만들기, 셈하기, 꼴 바꾸기, GPU로 빠르게 하기, 저절로 미분하기를 들여오고, 기울기 내려가기로 익히는 온전한 선형 회귀 보기로 맺는다. PyTorch에서 주성분 분석, 자기 부호기, 어떤 신경망 얼개든 다루기 앞서 이 밑감을 익혀야 한다.

## 코드

```python
"""PyTorch 기본."""
import torch
import numpy as np
import matplotlib.pyplot as plt

# === 텐서 만들기 ========================================================
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
matrix = torch.tensor([[1, 2, 3], [4, 5, 6]])
zeros = torch.zeros(2, 3)
random = torch.randn(2, 2)
identity = torch.eye(3)

# === 기본 연산 ========================================================
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])
A = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
mat_product = A @ B

# === 저절로 미분하기 ================================================================
x = torch.tensor([2.0], requires_grad=True)
y = 3 * x**2 + 2 * x + 1
y.backward()
print(f"dy/dx at x=2: {x.grad}")  # 14.0

# === 선형 회귀 =======================================================
torch.manual_seed(42)
true_w, true_b = 2.0, 1.0
x_data = torch.linspace(0, 10, 100).unsqueeze(1)
y_data = true_w * x_data + true_b + 0.5 * torch.randn(100, 1)

w = torch.randn(1, 1, requires_grad=True)
b_param = torch.zeros(1, requires_grad=True)
learning_rate = 0.01

losses = []
for epoch in range(50):
    y_pred = x_data @ w + b_param
    loss = ((y_pred - y_data) ** 2).mean()
    losses.append(loss.item())
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        b_param -= learning_rate * b_param.grad
    w.grad.zero_()
    b_param.grad.zero_()

print(f"Learned w={w.item():.4f}, b={b_param.item():.4f}")

if __name__ == "__main__":
    pass
```

## 논의

PyTorch의 텐서는 NumPy 배열처럼 움직이되 결정적인 것 둘이 더 있다. 곧 셈을 빠르게 하려 GPU에 놓을 수 있고, 저절로 미분하려 연산을 좇을 수 있다. `requires_grad=True` 깃발은 앞먹임 동안 셈 그래프를 세우라고 PyTorch에 이르며, 스칼라 손실에 `.backward()`을 부르면 그 그래프를 거꾸로 훑어 좇던 매개변수마다의 기울기를 셈한다.

선형 회귀 보기는 깊은 배움 전체에서 쓰이는 고갱이 익히기 되풀이를 보여 준다. 곧 앞먹임(어림 셈하기), 손실 셈하기(어긋남 재기), 뒤먹임(기울기 셈하기), 매개변수 새로 고침(무게 고치기)이다. 새로 고치는 걸음에서 `torch.no_grad()` 맥락 다루개가 꼭 필요하다. 그러지 않으면 PyTorch가 새로 고침 연산까지 셈 그래프에 넣어 기억 공간을 낭비하고 다음 되풀이에서 기울기가 틀리게 된다.

텐서를 CPU와 GPU 사이에서 옮길 때는 `.to('cuda')`이나 `.cuda()`을 쓰며, 한 셈에 드는 모든 것이 같은 기기에 있어야 한다. 이 길잡이 같은 작은 문제에는 CPU 셈으로 넉넉하지만, 큰 자료 묶음의 주성분 분석 행렬 연산과 신경망 익히기에서는 GPU로 빠르게 하는 것이 결정적이다.

## 연습문제

**연습문제 1.**
3x3 아무 텐서를 만들고 `torch.det`과 `torch.linalg.eig`으로 행렬식과 고윳값을 셈해, 행렬식이 고윳값의 곱과 같은지 확인하라.

??? success "연습문제 1 풀이"
    ```python
    M = torch.randn(3, 3)
    det = torch.det(M)
    eigvals, _ = torch.linalg.eig(M)
    product = eigvals.prod()
    print(f"det(M) = {det.item():.4f}")
    print(f"Product of eigenvalues = {product.real.item():.4f}")
    print(f"Match: {torch.allclose(det, product.real, atol=1e-4)}")
    ```
    정의상 행렬식은 고윳값의 곱과 같다. `torch.linalg.eig`은 복소 고윳값을 돌려주므로 그 곱의 실수 부분을 취한다.

---

**연습문제 2.**
손으로 매개변수를 고치는 대신 `torch.optim.SGD`을 쓰도록 선형 회귀를 고쳐라. 50번 돈 뒤의 마지막 손실을 견주어 같음을 확인하라.

??? success "연습문제 2 풀이"
    ```python
    w2 = torch.randn(1, 1, requires_grad=True)
    b2 = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.SGD([w2, b2], lr=0.01)
    for epoch in range(50):
        y_pred = x_data @ w2 + b2
        loss = ((y_pred - y_data) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"w={w2.item():.4f}, b={b2.item():.4f}, loss={loss.item():.4f}")
    ```
    관성이 없는 `torch.optim.SGD`은 손으로 하는 것과 같은 $w \leftarrow w - \eta \nabla_w L$ 새로 고침을 하므로 결과가 똑같다.

---

**연습문제 3.**
저절로 미분하기로 점 $(x, y) = (1, 2)$에서 $f(x, y) = x^2 y + y^3$의 기울기를 셈하라. 편미분을 손으로 셈해 결과를 확인하라.

??? success "연습문제 3 풀이"
    ```python
    x = torch.tensor(1.0, requires_grad=True)
    y = torch.tensor(2.0, requires_grad=True)
    f = x**2 * y + y**3
    f.backward()
    print(f"df/dx = {x.grad.item()}")  # 2*x*y = 2*1*2 = 4
    print(f"df/dy = {y.grad.item()}")  # x^2 + 3*y^2 = 1 + 12 = 13
    ```
    손으로 셈하면 $\partial f/\partial x = 2xy = 4$, $\partial f/\partial y = x^2 + 3y^2 = 13$이며 저절로 미분한 결과와 정확히 맞는다.
