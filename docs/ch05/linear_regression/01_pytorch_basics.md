# PyTorch 기초

PyTorch 텐서는 모든 딥러닝 계산의 기본 자료구조로, NumPy 배열과 비슷하지만 GPU 가속과 자동 미분을 갖추고 있다. 신경망 모델을 만들기 전에 텐서를 만들고, 다루고, 미분하는 법을 이해하는 것이 필수적이다. 이 스크립트는 텐서 생성, 산술, 재구성, 그리고 경사 기반 최적화를 떠받치는 autograd 체계를 소개한다.

## 1. 코드

```python
"""
==============================================================================
01_pytorch_basics.py
==============================================================================
어려움: ⭐ (첫걸음)

DESCRIPTION:
    PyTorch 텐서와 자동 미분(autograd) 들머리.
    이 글은 뒤이은 모든 학습에서 쓸
    밑바탕 벽돌을 다룬다.

다루는 것:
    - 여러 방식으로 텐서 만들기
    - 텐서 셈과 펴 맞추기
    - autograd로 하는 자동 미분
    - 기울기 계산
    - requires_grad 이해하기

PREREQUISITES:
    - 기본 파이썬 지식
    - 도함수(미적분)에 대한 기본 이해

학습 목표:
    - PyTorch 텐서를 만들고 다룬다
    - 텐서 셈을 이해한다
    - autograd로 기울기를 절로 계산한다
    - 셈 그래프를 이해한다

걸리는 때: 15분쯤
==============================================================================
"""

import torch
import numpy as np

print("=" * 70)
print("PART 1: CREATING TENSORS")
print("=" * 70)

# ============================================================================
# 1.1 파이썬 리스트로 텐서 만들기
# ============================================================================
print("\n1.1 Creating tensors from Python lists:")

# 1차원 텐서(벡터)를 만든다
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(f"1D tensor: {tensor_1d}")
print(f"Shape: {tensor_1d.shape}")
print(f"Data type: {tensor_1d.dtype}")

# 2차원 텐서(행렬)를 만든다
tensor_2d = torch.tensor([[1, 2, 3],
                          [4, 5, 6]])
print(f"\n2D tensor:\n{tensor_2d}")
print(f"Shape: {tensor_2d.shape}")  # (2 rows, 3 columns)
print(f"Data type: {tensor_2d.dtype}")

# ============================================================================
# 1.2 특정 데이터형으로 텐서 만들기
# ============================================================================
print("\n1.2 Creating tensors with specific data types:")

# 기본적으로 정수는 torch.int64를, 실수는 torch.float32를 만든다
float_tensor = torch.tensor([1.0, 2.0, 3.0])
print(f"Float tensor: {float_tensor}, dtype: {float_tensor.dtype}")

# 데이터형을 명시적으로 지정할 수 있다
float32_tensor = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"Float32 tensor: {float32_tensor}, dtype: {float32_tensor.dtype}")

# ============================================================================
# 1.3 NumPy 배열로 텐서 만들기
# ============================================================================
print("\n1.3 Creating tensors from NumPy arrays:")

numpy_array = np.array([1, 2, 3, 4, 5])
tensor_from_numpy = torch.from_numpy(numpy_array)
print(f"Tensor from NumPy: {tensor_from_numpy}")

# 텐서를 다시 NumPy로 바꾼다
back_to_numpy = tensor_from_numpy.numpy()
print(f"Back to NumPy: {back_to_numpy}")

# ============================================================================
# 1.4 특별한 텐서 만들기
# ============================================================================
print("\n1.4 Creating special tensors:")

zeros = torch.zeros(2, 3)
print(f"Zeros:\n{zeros}")

ones = torch.ones(2, 3)
print(f"\nOnes:\n{ones}")

random = torch.rand(2, 3)
print(f"\nRandom:\n{random}")

randn = torch.randn(2, 3)
print(f"\nRandom normal:\n{randn}")

arange = torch.arange(0, 10, 2)
print(f"\nArange: {arange}")

linspace = torch.linspace(0, 1, 5)
print(f"Linspace: {linspace}")

print("\n" + "=" * 70)
print("PART 2: TENSOR OPERATIONS")
print("=" * 70)

# ============================================================================
# 2.1 기본 산술 연산
# ============================================================================
print("\n2.1 Basic arithmetic operations:")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"a + b = {a + b}")
print(f"a - b = {a - b}")
print(f"a * b = {a * b}")
print(f"a / b = {a / b}")
print(f"a ** 2 = {a ** 2}")

# ============================================================================
# 2.2 행렬 연산
# ============================================================================
print("\n2.2 Matrix operations:")

A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])

print(f"Element-wise multiplication:\n{A * B}")
print(f"\nMatrix multiplication (A @ B):\n{A @ B}")
print(f"\nTranspose of A:\n{A.T}")

# ============================================================================
# 2.3 텐서 모양 바꾸기
# ============================================================================
print("\n2.3 Reshaping tensors:")

x = torch.arange(12)
print(f"Original: {x}, shape: {x.shape}")

x_reshaped = x.reshape(3, 4)
print(f"\nReshaped to 3x4:\n{x_reshaped}")

x_reshaped2 = x.reshape(2, 6)
print(f"\nReshaped to 2x6:\n{x_reshaped2}")

x_reshaped3 = x.reshape(4, -1)
print(f"\nReshaped to 4x-1 (becomes 4x3):\n{x_reshaped3}")

print("\n" + "=" * 70)
print("PART 3: AUTOMATIC DIFFERENTIATION (AUTOGRAD)")
print("=" * 70)

# ============================================================================
# 3.1 requires_grad 이해하기
# ============================================================================
print("\n3.1 Understanding requires_grad:")

x = torch.tensor([2.0], requires_grad=True)
print(f"x = {x}")
print(f"x.requires_grad = {x.requires_grad}")

y = x ** 2
print(f"\ny = x^2 = {y}")
print(f"y.requires_grad = {y.requires_grad}")

# ============================================================================
# 3.2 경사 계산하기
# ============================================================================
print("\n3.2 Computing gradients:")

y.backward()
print(f"dy/dx at x=2: {x.grad}")

# ============================================================================
# 3.3 좀 더 복잡한 예제
# ============================================================================
print("\n3.3 More complex example:")

x = torch.tensor([3.0], requires_grad=True)
y = 3 * x**2 + 2 * x + 1
print(f"\nf(x) = 3x^2 + 2x + 1")
print(f"f(3) = {y.item()}")

y.backward()
print(f"df/dx at x=3: {x.grad.item()}")
print(f"Expected (6*3 + 2 = 20): 20")

# ============================================================================
# 3.4 경사 누적
# ============================================================================
print("\n3.4 Gradient accumulation:")

x = torch.tensor([2.0], requires_grad=True)

y = x ** 2
y.backward()
print(f"First backward: x.grad = {x.grad}")

y = x ** 3
y.backward()
print(f"Second backward (accumulated): x.grad = {x.grad}")

# ============================================================================
# 3.5 경사를 0으로 만들기
# ============================================================================
print("\n3.5 Zeroing gradients:")

x = torch.tensor([2.0], requires_grad=True)

y = x ** 2
y.backward()
print(f"First backward: x.grad = {x.grad}")

x.grad.zero_()
print(f"After zeroing: x.grad = {x.grad}")

y = x ** 3
y.backward()
print(f"Second backward (after zeroing): x.grad = {x.grad}")

# ============================================================================
# 3.6 계산 그래프에서 떼어내기
# ============================================================================
print("\n3.6 Detaching from the computation graph:")

x = torch.tensor([2.0], requires_grad=True)
y = x ** 2
y_detached = y.detach()
print(f"y_detached.requires_grad = {y_detached.requires_grad}")

# ============================================================================
# 3.7 경사를 쓰지 않는 문맥
# ============================================================================
print("\n3.7 No gradient context:")

x = torch.tensor([2.0], requires_grad=True)

with torch.no_grad():
    y = x ** 2
    print(f"Inside no_grad context, y.requires_grad = {y.requires_grad}")

print("\n" + "=" * 70)
print("PART 4: PRACTICAL EXAMPLE - SIMPLE DERIVATIVE")
print("=" * 70)

# ============================================================================
# 4.1 경사 하강법으로 함수의 최솟값 찾기
# ============================================================================
print("\n4.1 Finding minimum of f(x) = (x-3)^2:")

x = torch.tensor([0.0], requires_grad=True)
learning_rate = 0.1
num_steps = 20

print(f"{'Step':<6} {'x':<10} {'f(x)':<10} {'df/dx':<10}")
print("-" * 40)

for step in range(num_steps):
    f = (x - 3) ** 2

    if x.grad is not None:
        x.grad.zero_()
    f.backward()

    print(f"{step:<6} {x.item():<10.4f} {f.item():<10.4f} {x.grad.item():<10.4f}")

    with torch.no_grad():
        x -= learning_rate * x.grad

print(f"\nFinal x: {x.item():.6f}")
print(f"Expected minimum at x=3")
print(f"Final f(x): {((x - 3) ** 2).item():.6f}")


if __name__ == "__main__":
    pass
```

## 2. 논의

PyTorch의 텐서는 NumPy의 배열과 같은 역할을 하지만 두 가지 결정적인 능력을 더 가진다. 계산을 가속하기 위해 GPU로 옮길 수 있고, autograd 엔진을 통해 자동 미분을 지원한다. 텐서를 만드는 일은 간단하다. 파이썬 리스트, NumPy 배열, 또는 `torch.zeros`, `torch.randn`, `torch.linspace` 같은 팩토리 함수 어느 쪽에서든 만들 수 있다. 덧셈, 곱셈, 행렬 곱, 재구성 같은 표준 연산은 NumPy의 대응물을 거의 그대로 따르므로, 수치 계산에 익숙한 사람이라면 자연스럽게 넘어올 수 있다.

PyTorch를 단순한 GPU 가속 배열 라이브러리가 아니라 딥러닝 프레임워크로 만들어 주는 것이 autograd 체계이다. 텐서를 `requires_grad=True`로 만들면 그 텐서에 대한 모든 연산이 계산 그래프에 기록된다. 스칼라 출력에 `.backward()`를 호출하면 그 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 `.grad` 속성을 채운다. 이것이 바로 신경망의 역전파를 떠받치는 구조이다. 순전파 계산을 정의하고, 손실을 계산하고, `loss.backward()`를 호출한 뒤, 매개변수 갱신에 필요한 경사를 읽어 오면 된다.

올바르게 사용하려면 몇 가지 실무적인 사항이 중요하다. 경사는 기본적으로 누적되므로, 오래된 값이 섞이지 않도록 새 역전파를 하기 전에 `.grad.zero_()`를 호출해야 한다. `torch.no_grad()` 컨텍스트 관리자는 경사 추적을 끄는데, 이는 (갱신 자체가 그래프에 기록되지 않도록) 매개변수 갱신에도, (메모리와 시간을 아끼기 위해) 추론에도 중요하다. `.detach()`로 텐서를 떼어내면 그래프에서 완전히 분리되어, 저장소는 공유하지만 경사를 전파하지 않는 평범한 값이 된다.

## 연습문제

**연습문제 1.**
`float32`로 텐서 $x = [1, 2, 3, 4, 5]$를 만들어라. 원소별로 $y = x^3 + 2x$를 계산한 뒤 autograd로 각 원소에서의 경사 $\frac{dy}{dx}$를 계산하라. 결과를 해석적 도함수 $3x^2 + 2$와 대조하여 확인하라.

??? success "연습문제 1 풀이"
    ```python
    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    y = (x ** 3 + 2 * x).sum()  # sum to get a scalar for backward
    y.backward()
    print(f"Autograd gradient: {x.grad}")
    print(f"Analytical (3x^2 + 2): {3 * x.detach() ** 2 + 2}")
    # 둘 다 [5, 14, 29, 50, 77]을 내야 한다
    ```

---

**연습문제 2.**
경사 하강법의 매개변수 갱신 단계에서 `torch.no_grad()`를 쓰는 이유와, 그것을 빠뜨리면 어떤 문제가 생기는지 설명하라.

??? success "연습문제 2 풀이"
    매개변수 갱신 `x -= lr * x.grad`는 `requires_grad=True`인 텐서에 대한 제자리 산술 연산이다. `torch.no_grad()`가 없으면 PyTorch가 이 뺄셈을 계산 그래프에 기록하려 하는데, 그래프에 속한 잎 변수를 제자리에서 수정하는 것은 (이전에 계산한 경사를 무효화하므로) 허용되지 않아 오류가 난다. 갱신을 `torch.no_grad()`로 감싸면 PyTorch가 이를 그래프 밖의 평범한 수치 연산으로 취급한다. 갱신 규칙은 우리가 미분하려는 함수의 일부가 아니므로 이것이 올바른 의미이다.

---

**연습문제 3.**
원점에서 시작하여 경사 하강법으로 $f(x_1, x_2) = (x_1 - 2)^2 + (x_2 + 1)^2$의 최솟값을 찾아라. 궤적을 출력하고 $(2, -1)$로 수렴함을 확인하라.

??? success "연습문제 3 풀이"
    ```python
    x = torch.tensor([0.0, 0.0], requires_grad=True)
    lr = 0.1

    for step in range(50):
        f = (x[0] - 2) ** 2 + (x[1] + 1) ** 2
        if x.grad is not None:
            x.grad.zero_()
        f.backward()
        with torch.no_grad():
            x -= lr * x.grad
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}: x = [{x[0].item():.4f}, {x[1].item():.4f}], f(x) = {f.item():.6f}")

    # 최종 x는 대략 [2.0, -1.0]이어야 한다
    print(f"Final: x1 = {x[0].item():.6f}, x2 = {x[1].item():.6f}")
    ```

## 정리하며

**다룬 것** — PyTorch 기초

PyTorch의 텐서는 NumPy의 배열과 같은 역할을 하지만 두 가지 결정적인 능력을 더 가진다.

앞의 연습문제 3개로 직접 확인할 수 있다.
