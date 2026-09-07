# 소개

01_introduction.py - PyTorch 텐서와 Autograd 소개

이 튜토리얼은 PyTorch에서 로지스틱 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
"""
================================================================================
01_introduction.py - Introduction to PyTorch Tensors and Autograd
================================================================================

배움 목표:
- Understand PyTorch tensors (the fundamental data structure)
- Learn about automatic differentiation (autograd)
- Implement a simple gradient descent optimization
- Understand the basics of neural network training

PREREQUISITES:
- Basic Python programming
- Understanding of derivatives and gradients
- numpy basics (helpful but not required)

TIME TO COMPLETE: ~30 minutes

DIFFICULTY: ⭐☆☆☆☆ (Beginner)
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("PART 1: PYTORCH TENSORS - The Building Blocks")
print("="*80)

# ============================================================================
# 1.1: 텐서 만들기
# ============================================================================
print("\n1.1: Creating Tensors")
print("-" * 40)

# 여러 방법으로 텐서를 만든다
tensor_from_list = torch.tensor([1.0, 2.0, 3.0])  # From Python list
tensor_from_numpy = torch.from_numpy(np.array([4.0, 5.0, 6.0]))  # From numpy
tensor_zeros = torch.zeros(3)  # All zeros
tensor_ones = torch.ones(3)    # All ones
tensor_random = torch.randn(3)  # Random from standard normal distribution

print(f"From list:   {tensor_from_list}")
print(f"From numpy:  {tensor_from_numpy}")
print(f"Zeros:       {tensor_zeros}")
print(f"Ones:        {tensor_ones}")
print(f"Random:      {tensor_random}")

# 모양과 자료형
print(f"\nShape: {tensor_from_list.shape}")  # torch.Size([3])
print(f"Dtype: {tensor_from_list.dtype}")   # torch.float32
print(f"Device: {tensor_from_list.device}") # cpu or cuda

# ============================================================================
# 1.2: 텐서 연산
# ============================================================================
print("\n1.2: Tensor Operations")
print("-" * 40)

x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

# 원소별 연산
add_result = x + y                    # [5.0, 7.0, 9.0]
multiply_result = x * y               # [4.0, 10.0, 18.0]
dot_product = torch.dot(x, y)         # 1*4 + 2*5 + 3*6 = 32

print(f"x + y = {add_result}")
print(f"x * y = {multiply_result}")
print(f"x · y = {dot_product.item()}")  # .item() extracts Python scalar

# 행렬 연산
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
b = torch.tensor([1.0, 2.0])

matrix_vector_product = torch.matmul(A, b)  # or A @ b
print(f"\nMatrix-vector product:\n{matrix_vector_product}")


print("\n" + "="*80)
print("PART 2: AUTOMATIC DIFFERENTIATION (AUTOGRAD)")
print("="*80)

# ============================================================================
# 2.1: 경사 계산하기
# ============================================================================
print("\n2.1: Computing Gradients")
print("-" * 40)

# 텐서를 만들고 PyTorch에게 그 연산을 추적하라고 알린다
# requires_grad=True는 이 텐서에 대한 경사를 계산하겠다는 뜻이다
x = torch.tensor([2.0], requires_grad=True)

# 함수를 정의한다: y = x^2
y = x ** 2

print(f"x = {x.item()}")
print(f"y = x^2 = {y.item()}")

# 경사 dy/dx 계산
y.backward()  # This computes all gradients automatically!

# 경사는 x.grad에 저장된다
print(f"dy/dx at x=2 is: {x.grad.item()}")
print(f"Mathematical verification: dy/dx = 2x = 2*2 = 4 ✓")

# ============================================================================
# 2.2: 더 복잡한 함수
# ============================================================================
print("\n2.2: More Complex Function")
print("-" * 40)

# 초기화: 새 텐서를 만든다
x = torch.tensor([3.0], requires_grad=True)

# 더 복잡한 함수: y = 3x^2 + 2x + 1
y = 3 * x**2 + 2 * x + 1

print(f"x = {x.item()}")
print(f"y = 3x^2 + 2x + 1 = {y.item()}")

y.backward()
print(f"dy/dx at x=3 is: {x.grad.item()}")
print(f"Mathematical verification: dy/dx = 6x + 2 = 6*3 + 2 = 20 ✓")


print("\n" + "="*80)
print("PART 3: GRADIENT DESCENT OPTIMIZATION")
print("="*80)

# ============================================================================
# 3.1: 함수의 최솟값 찾기
# ============================================================================
print("\n3.1: Finding Minimum of y = (x - 3)^2")
print("-" * 40)

# y = (x - 3)^2을 최소로 만드는 x를 찾고자 한다
# 최솟값은 x = 3에 있으며 그때 y = 0이다

# 초기 추측으로 시작한다
x = torch.tensor([0.0], requires_grad=True)  # Starting point: x = 0
learning_rate = 0.1  # Step size for each update
num_iterations = 50

# 그림을 그리기 위해 기록을 저장한다
x_history = []
y_history = []

print(f"Initial x: {x.item():.4f}")

for iteration in range(num_iterations):
    # 순전파: 함수값을 계산한다
    y = (x - 3) ** 2  # Function to minimize
    
    # 이력 저장
    x_history.append(x.item())
    y_history.append(y.item())
    
    # 역전파: 경사를 계산한다
    y.backward()  # Computes dy/dx
    
    # 갱신 단계: y를 줄이는 방향으로 이동한다
    # x_new = x_old - learning_rate * gradient
    with torch.no_grad():  # We don't need gradients for the update step
        x -= learning_rate * x.grad  # This is gradient descent!
    
    # 다음 반복을 위해 경사를 0으로 만든다 (중요!)
    x.grad.zero_()
    
    # 10회 반복마다 진행 상황을 출력한다
    if (iteration + 1) % 10 == 0:
        print(f"Iteration {iteration+1:2d}: x = {x.item():.4f}, y = {y.item():.4f}")

print(f"\nFinal x: {x.item():.4f}")
print(f"Target x: 3.0000")
print(f"Final y: {y.item():.6f}")

# ============================================================================
# 3.2: 최적화 경로 시각화
# ============================================================================
print("\n3.2: Visualizing the Optimization")
print("-" * 40)

# 그림을 만든다
plt.figure(figsize=(12, 5))

# 그림 1: 함수와 최적화 경로
plt.subplot(1, 2, 1)
x_range = np.linspace(-1, 6, 100)
y_range = (x_range - 3) ** 2

plt.plot(x_range, y_range, 'b-', linewidth=2, label='y = (x-3)²')
plt.plot(x_history, y_history, 'ro-', linewidth=1, markersize=4, 
         label='Optimization path', alpha=0.6)
plt.plot(x_history[0], y_history[0], 'go', markersize=10, label='Start')
plt.plot(x_history[-1], y_history[-1], 'r*', markersize=15, label='End')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Function and Optimization Path', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 그림 2: 수렴 (y가 시간에 따라 줄어드는 모습)
plt.subplot(1, 2, 2)
plt.plot(y_history, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss (y value)', fontsize=12)
plt.title('Convergence: Loss over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to see convergence better

plt.tight_layout()
plt.savefig('/home/claude/pytorch_logistic_regression_tutorial/01_basics/optimization_demo.png', 
            dpi=150, bbox_inches='tight')
print("Plot saved as: optimization_demo.png")


print("\n" + "="*80)
print("PART 4: KEY TAKEAWAYS")
print("="*80)

print("""
1. TENSORS are PyTorch's fundamental data structure
   - Like NumPy arrays but with GPU support and autograd
   - Created with torch.tensor(), torch.zeros(), torch.randn(), etc.

2. AUTOGRAD (Automatic Differentiation)
   - Set requires_grad=True to track operations
   - .backward()을 불러 모든 기울기를 셈한다
   - Gradients stored in .grad attribute

3. GRADIENT DESCENT
   - Iteratively update parameters to minimize a function
   - Update rule: x_new = x_old - learning_rate * gradient
   - Always call .zero_grad() between iterations!

4. TORCH VS NUMPY
   - torch.tensor vs np.array
   - PyTorch can run on GPU (CUDA)
   - PyTorch has built-in automatic differentiation
""")


print("\n" + "="*80)
print("EXERCISES (Try These!)")
print("="*80)

print("""
1. EASY: Change the learning rate to 0.01 and 0.5. 
   How does it affect convergence speed?

2. MEDIUM: Try minimizing y = x^4 - 4x^2 + 4
   This function has two local minima!
   Starting from x=0 and x=3, what do you get?

3. MEDIUM: Implement gradient descent for 2D function:
   y = (x1 - 2)^2 + (x2 + 1)^2
   Minimum should be at (2, -1)

4. HARD: Add momentum to gradient descent:
   velocity = 0.9 * velocity + learning_rate * gradient
   x = x - velocity
   Compare convergence with standard gradient descent.
""")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
You've learned the basics of PyTorch tensors and gradient descent!

Next tutorial: 02_simple_binary_classification.py
- Apply these concepts to a real machine learning problem
- Implement logistic regression from scratch
- Train a binary classifier

Ready to continue? Run:
    python 02_simple_binary_classification.py
""")
print("="*80)


if __name__ == "__main__":
    pass
```

## 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.
