# PyTorch Autograd 기초

이 스크립트는 PyTorch autograd의 기초을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""
================================================================================
1단계 - 보기 2: PyTorch 자동 미분 기초
================================================================================

학습 목표:
- PyTorch의 자동 미분(autograd)을 이해한다
- 셈 그래프를 배운다
- 직접 계산와 절로 계산를 견준다
- requires_grad과 backward() 방법을 익힌다

어려움: ⭐ 첫걸음

걸리는 때: 25~35분

PREREQUISITES:
- 보기 01(직접 하는 경사 하강법)을 마쳤을 것
- 기울기를 기본으로 이해하고 있을 것

================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("PYTORCH AUTOGRAD: AUTOMATIC DIFFERENTIATION")
print("="*80)

# ============================================================================
# 1부: 텐서와 requires_grad 소개
# ============================================================================
print("\n" + "="*80)
print("PART 1: TENSORS WITH GRADIENT TRACKING")
print("="*80)

# 평범한 텐서(경사 추적 없음)
x_no_grad = torch.tensor([1.0, 2.0, 3.0])
print(f"\nRegular tensor: {x_no_grad}")
print(f"requires_grad: {x_no_grad.requires_grad}")  # False by default

# 경사 추적을 켠 텐서
x_with_grad = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"\nTensor with gradient tracking: {x_with_grad}")
print(f"requires_grad: {x_with_grad.requires_grad}")  # True
print("\n→ When requires_grad=True, PyTorch tracks all operations on this tensor!")

# ============================================================================
# 2부: 간단한 예 - 경사 계산하기
# ============================================================================
print("\n" + "="*80)
print("PART 2: SIMPLE GRADIENT COMPUTATION")
print("="*80)

# x = 2에서 y = 3x² + 2x + 1의 경사를 계산해 보자
print("\nFunction: y = 3x² + 2x + 1")
print("Derivative: dy/dx = 6x + 2")
print("At x = 2: dy/dx = 6(2) + 2 = 14")

# PyTorch 방식
x = torch.tensor(2.0, requires_grad=True)
y = 3 * x**2 + 2 * x + 1

print(f"\nUsing PyTorch autograd:")
print(f"x = {x.item()}")
print(f"y = {y.item()}")

# 경사 dy/dx 계산
y.backward()  # This populates x.grad with dy/dx

print(f"Computed gradient dy/dx = {x.grad.item()}")
print(f"Expected gradient = 14")
print(f"✓ Match!" if abs(x.grad.item() - 14) < 0.001 else "✗ Error!")

# ============================================================================
# 3부: 계산 그래프 이해하기
# ============================================================================
print("\n" + "="*80)
print("PART 3: COMPUTATIONAL GRAPH")
print("="*80)

print("""
requires_grad=True인 텐서에 셈을 하면
PyTorch가 셈 그래프를 세운다.

    x (requires_grad=True)
    ↓
    x² (가운데 셈)
    ↓
    3*x² (가운데 셈)
    ↓
    3*x² + 2x (가운데 셈)
    ↓
    y = 3*x² + 2x + 1 (마지막 출력)

y.backward()을 부르면
- PyTorch가 이 그래프를 거꾸로 훑는다
- 사슬 법칙을 절로 건다
- x.grad에 기울기를 쌓는다
""")

# ============================================================================
# 4부: PyTorch로 하는 경사 하강법 - 예제 01과 같은 문제
# ============================================================================
print("\n" + "="*80)
print("PART 4: LINEAR REGRESSION WITH AUTOGRAD")
print("="*80)

# 예제 01과 같은 데이터셋: y = 2*x
X = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
y = torch.tensor([2, 4, 6, 8, 10], dtype=torch.float32)

print("\nProblem: Fit y = w*x to find w ≈ 2")
print(f"Training data: X = {X.numpy()}")
print(f"               y = {y.numpy()}")

# 경사 추적을 켜고 가중치 초기화
w = torch.tensor(0.0, requires_grad=True)
print(f"\nInitial weight: w = {w.item():.3f}")

# 학습 하이퍼파라미터
learning_rate = 0.01
n_epochs = 100

# 진행 상황 추적
weight_history = [w.item()]
loss_history = []

print("\n" + "-"*60)
print("Training with PyTorch Autograd:")
print("-"*60)
print("Epoch | Weight    | Loss      | Gradient")
print("-"*60)

for epoch in range(n_epochs):
    # 1단계: 순전파
    y_pred = w * X
    
    # 2단계: 손실 계산
    loss = torch.mean((y_pred - y) ** 2)
    
    # 3단계: 역전파 - 경사를 자동으로 계산한다!
    # 중요: 경사가 누적되므로 먼저 0으로 만들어야 한다
    if w.grad is not None:
        w.grad.zero_()
    
    loss.backward()  # This computes dL/dw automatically!
    
    # 4단계: 가중치 갱신
    # 중요: 이 연산이 추적되지 않도록 torch.no_grad()를 쓴다
    with torch.no_grad():
        w -= learning_rate * w.grad
    
    # 이력 저장
    weight_history.append(w.item())
    loss_history.append(loss.item())
    
    # 진행 상황 출력
    if epoch % 10 == 0:
        print(f"{epoch+1:4d}  | {w.item():9.6f} | {loss.item():9.6f} | {w.grad.item():9.6f}")

print("-"*60)
print(f"Final weight: w = {w.item():.6f}")
print(f"Target: 2.0")
print(f"Error: {abs(w.item() - 2.0):.6f}")

# ============================================================================
# 5부: 직접 계산과 autograd 비교
# ============================================================================
print("\n" + "="*80)
print("PART 5: MANUAL vs AUTOGRAD COMPARISON")
print("="*80)

print("""
직접 계산하는 기울기(보기 01에서):
-----------------------------------
def compute_gradient(x, y_true, y_pred):
    # 직접 유도한 식: dL/dw = mean(2 * x * (y_pred - y_true))
    gradient = np.mean(2 * x * (y_pred - y_true))
    return gradient

자동 미분(이 보기):
------------------------
loss = torch.mean((y_pred - y) ** 2)
loss.backward()  # 모든 기울기를 절로 계산한다!
gradient = w.grad  # PyTorch가 대신 채워 준다

자동 미분의 이점:
✓ 직접 미적분할 것이 없다
✓ 복잡한 모델도 절로 다룬다
✓ 벌레가 적다(도함수 실수가 없다)
✓ 효율적인 짜보기
✓ 모델 구조를 고치기 쉽다

직접 할 때:
• 기울기를 배우거나 이해할 때
• 단순한 학습 보기
• 아주 남다른 셈
""")

# ============================================================================
# 6부: 중요한 개념 - 경사 누적
# ============================================================================
print("\n" + "="*80)
print("PART 6: GRADIENT ACCUMULATION (COMMON PITFALL!)")
print("="*80)

print("\nGradients ACCUMULATE by default. Watch what happens:\n")

# 누적을 보여주는 예
x = torch.tensor(3.0, requires_grad=True)

# 첫 번째 역전파
y1 = x ** 2
y1.backward()
print(f"After 1st backward: x.grad = {x.grad.item()}")  # Should be 6

# 경사를 초기화하지 않은 두 번째 역전파
x.grad.zero_()  # Comment this line to see accumulation!
y2 = x ** 3
y2.backward()
print(f"After 2nd backward: x.grad = {x.grad.item()}")  # Should be 27

print("""
종요롭다: 역전파 앞에는 늘 기울기를 0으로 만들어라!
    if w.grad is not None:
        w.grad.zero_()
    loss.backward()
""")

# ============================================================================
# 7부: 계산 그래프에서 떼어내기
# ============================================================================
print("\n" + "="*80)
print("PART 7: DETACHING AND NO_GRAD")
print("="*80)

x = torch.tensor(5.0, requires_grad=True)
y = x ** 2

# 방법 1: detach() - 그래프에서 떼어내되 값은 유지한다
y_detached = y.detach()
print(f"\nOriginal y.requires_grad: {y.requires_grad}")
print(f"Detached y.requires_grad: {y_detached.requires_grad}")

# 방법 2: torch.no_grad() - 추적을 일시적으로 끈다
with torch.no_grad():
    z = x ** 2
    print(f"\nInside no_grad, z.requires_grad: {z.requires_grad}")

print("""
쓰임새:
- detach(): 기울기 없이 값만 쓰고 싶을 때
- no_grad(): 추론이나 매개변수 고치기에
- 둘 다 쓸데없는 그래프 세우기를 막는다(기억 자리를 아낀다!)
""")

# ============================================================================
# 8부: 시각화
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 그림 1: 손실 수렴
axes[0].plot(loss_history, 'b-', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss', fontsize=12)
axes[0].set_title('Loss Convergence (PyTorch Autograd)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

# 그림 2: 가중치 수렴
axes[1].plot(weight_history, 'g-', linewidth=2)
axes[1].axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Target (2.0)')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Weight (w)', fontsize=12)
axes[1].set_title('Weight Convergence to Optimal Value', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_1_basics/pytorch_autograd.png', dpi=150)
print("\n✓ Plot saved as 'pytorch_autograd.png'")
print("\nClose the plot window to continue...")
plt.show()

# ============================================================================
# 9부: 연습 문제
# ============================================================================
print("\n" + "="*80)
print("PRACTICE EXERCISE")
print("="*80)
print("""
다음 함수에 경사 하강법을 짜 보아라.

1. Quadratic: y = ax² + bx + c
   - 매개변수 셋 a, b, c을 쓴다(모두 requires_grad=True)
   - 데이터에 맞춘다: x=[1,2,3,4], y=[3,8,15,24]

2. 비선형: y = a*sin(b*x) + c
   - 잡음 섞인 데이터에 사인 물결을 맞춘다
   - 학습률를 이리저리 바꾸어 보아라

3. Multi-dimensional: z = ax + by + c
   - 입력 특징 2개(x, y)
   - 3 parameters (a, b, c)
""")

# ============================================================================
# 10부: 핵심 요점
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. 자동 미분은 직접 하는 기울기 셈을 없앤다
   - 매개변수에 requires_grad=True를 둔다
   - 셈을 거쳐 셈 그래프를 세운다
   - .backward()을 불러 모든 기울기를 계산한다

2. 역전파 앞에는 늘 기울기를 0으로 만들어라
   - 기울기는 기본으로 쌓인다
   - 이렇게 쓴다: w.grad.zero_()이나 optimizer.zero_grad()

3. 매개변수를 고칠 때는 torch.no_grad()을 써라
   - 미분할 필요 없는 셈을 좇지 않게 한다
   - 추론(시험/예측)에 꼭 필요하다
   - 기억 자리와 셈을 아낀다

4. 잎 텐서가 기울기를 담는다
   - requires_grad=True로 만든 텐서만 그렇다
   - 가운데 텐서는 기본으로 기울기를 담지 않는다

5. 셈 그래프는 그때그때 세워진다
   - 순전파에서 세워진다
   - 역전파 뒤에 지워진다
   - 다음 순전파에서 다시 세워진다
""")

print("="*80)
print("NEXT STEPS")
print("="*80)
print("""
이제 자동 미분을 이해했다!
다음으로 보기 03에 가서 이를 참 회귀 문제에 써 보아라.
""")
print("="*80)


if __name__ == "__main__":
    pass
```

## 논의

이 코드는 `requires_grad=True`인 텐서에 대한 연산을 자동으로 추적하는 PyTorch의 autograd 체계를 보여준다. 스칼라 손실에 `.backward()`를 호출하면 autograd가 계산 그래프를 역방향으로 훑으며 연쇄 법칙을 적용해 모든 잎 텐서의 경사를 계산한다. 이 구조가 PyTorch의 모든 신경망 학습을 떠받친다.

경사 추적을 제어하는 것은 정확성과 성능 모두에 필수적이다. `torch.no_grad()` 컨텍스트 관리자는 매개변수 갱신이나 추론처럼 계산 그래프에 포함되어서는 안 되는 연산에 대해 autograd를 끈다. `.detach()` 메서드는 저장소는 공유하지만 그래프와는 분리된 텐서를 만들며, 값을 기록하거나 NumPy로 변환할 때 유용하다.

## 연습문제

**연습문제 1.**
함수 $f(x) = x^3 - 2x^2 + x$를 생각하자. PyTorch autograd를 사용하여 $f'(3)$을 계산하라.

??? success "연습문제 1 풀이"
    ```python
    import torch

    x = torch.tensor(3.0, requires_grad=True)
    f = x**3 - 2*x**2 + x
    f.backward()
    print(x.grad)  # f'(x) = 3x^2 - 4x + 1 = 27 - 12 + 1 = 16.0
    ```

---


**연습문제 2.**
`retain_graph=True` 없이 같은 계산 그래프에 `.backward()`를 두 번 호출하면 오류가 나는 이유를 설명하라. `retain_graph=True`는 메모리 사용량에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    기본적으로 PyTorch는 메모리를 아끼기 위해 `.backward()` 후에 계산 그래프를 해제한다. `.backward()`를 두 번째로 호출하면 더 이상 존재하지 않는 그래프를 훑으려 하므로 `RuntimeError`가 발생한다. `retain_graph=True`로 두면 그래프가 메모리에 남아 재사용할 수 있지만, 모든 중간 텐서가 할당된 채로 남으므로 메모리 소비가 늘어난다.

---


**연습문제 3.**
잎 텐서 `w`를 만들고 손실을 계산한 뒤, 경사를 초기화하지 않고 `.backward()`를 세 번 호출하며 매번 `w.grad`를 출력하는 코드를 작성하라. 관찰된 값을 설명하라.

??? success "연습문제 3 풀이"
    ```python
    import torch

    w = torch.tensor(2.0, requires_grad=True)
    for i in range(3):
        loss = (w ** 2).sum()
        loss.backward()
        print(f'After backward {i+1}: w.grad = {w.grad}')
    # 출력: 4.0, 8.0, 12.0
    # 경사가 누적된다. 매 backward가 기존 경사에 2*w = 4.0을 더한다.
    ```
