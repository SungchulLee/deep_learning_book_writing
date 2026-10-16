# 소개

텐서 연산과 자동 미분으로 로지스틱 회귀 학습에 필요한 계산 도구를 복습한다. 이 페이지의 최적화 예제는 분류기가 아니라 간단한 이차 함수다.

## 1. 코드

```python
"""
================================================================================
01_introduction.py - PyTorch 텐서와 자동 미분 들머리
================================================================================

학습 목표:
- PyTorch 텐서(밑바탕 데이터 짜임)를 이해한다
- 자동 미분(autograd)을 배운다
- 단순한 경사 하강법 다듬기를 짠다
- 뉴럴 넷트웍 학습의 기초를 이해한다

PREREQUISITES:
- 기본 파이썬 프로그래밍
- 도함수와 그래디언트 이해
- 넘파이 기초(도움이 되지만 꼭 필요하지는 않다)

소요 시간: 30분쯤

어려움: ⭐☆☆☆☆ (첫스텝)
================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다
torch.manual_seed(0)

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

# 모양과 데이터형
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
plt.savefig('optimization_demo.png', 
            dpi=150, bbox_inches='tight')
print("Plot saved as: optimization_demo.png")


print("\n" + "="*80)
print("PART 4: KEY TAKEAWAYS")
print("="*80)

print("""
1. 텐서가 PyTorch의 밑바탕 데이터 짜임이다
   - 넘파이 배열과 비슷하되 GPU 받침과 자동 미분이 있다
   - torch.tensor(), torch.zeros(), torch.randn() 따위로 만든다

2. 자동 미분(autograd)
   - 셈을 좇으려면 requires_grad=True를 둔다
   - .backward()을 불러 모든 그래디언트를 계산한다
   - 그래디언트는 .grad 속성에 담긴다

3. 경사 하강법
   - 함수를 가장 작게 하도록 파라메터를 거듭 고친다
   - 고침 규칙: x_new = x_old - learning_rate * gradient
   - 루프 사이에는 늘 .zero_grad()을 불러라!

4. 토치와 넘파이 견주기
   - torch.tensor와 np.array
   - PyTorch는 GPU(CUDA)에서 돌 수 있다
   - PyTorch에는 붙박이 자동 미분이 있다
""")


print("\n" + "="*80)
print("EXERCISES (Try These!)")
print("="*80)

print("""
1. 쉬움: 러닝레이트를 0.01과 0.5으로 바꾸어라.
   모여드는 속도에 어떤 영향을 주는가?

2. 보통: y = x^4 - 4x^2 + 4을 가장 작게 해 보아라
   이 함수에는 그 자리 골짜기가 둘 있다!
   x=0과 x=3에서 비롯하면 무엇을 얻는가?

3. 보통: 2차원 함수에 경사 하강법을 짜라.
   y = (x1 - 2)^2 + (x2 + 1)^2
   골짜기는 (2, -1)에 있어야 한다

4. 어려움: 경사 하강법에 여세를 더하여라.
   velocity = 0.9 * velocity + learning_rate * gradient
   x = x - velocity
   여느 경사 하강법과 모여드는 모습을 견주어라.
""")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
PyTorch 텐서와 경사 하강법의 기초를 배웠다!

다음 튜토리얼: 02_simple_binary_classification.py
- 이 개념을 참 기계 학습 문제에 쓴다
- 로지스틱 회귀를 밑바닥부터 짠다
- 둘 분류기를 익힌다

이어서 할 채비가 되었는가? 다음을 돌려라.
    python 02_simple_binary_classification.py
""")
print("="*80)


if __name__ == "__main__":
    pass
```

??? note "전체 출력 (124줄)"

    ```
    ================================================================================
    PART 1: PYTORCH TENSORS - The Building Blocks
    ================================================================================

    1.1: Creating Tensors
    ----------------------------------------
    From list:   tensor([1., 2., 3.])
    From numpy:  tensor([4., 5., 6.], dtype=torch.float64)
    Zeros:       tensor([0., 0., 0.])
    Ones:        tensor([1., 1., 1.])
    Random:      tensor([ 1.5410, -0.2934, -2.1788])

    Shape: torch.Size([3])
    Dtype: torch.float32
    Device: cpu

    1.2: Tensor Operations
    ----------------------------------------
    x + y = tensor([5., 7., 9.])
    x * y = tensor([ 4., 10., 18.])
    x · y = 32.0

    Matrix-vector product:
    tensor([ 5., 11.])

    ================================================================================
    PART 2: AUTOMATIC DIFFERENTIATION (AUTOGRAD)
    ================================================================================

    2.1: Computing Gradients
    ----------------------------------------
    x = 2.0
    y = x^2 = 4.0
    dy/dx at x=2 is: 4.0
    Mathematical verification: dy/dx = 2x = 2*2 = 4 ✓

    2.2: More Complex Function
    ----------------------------------------
    x = 3.0
    y = 3x^2 + 2x + 1 = 34.0
    dy/dx at x=3 is: 20.0
    Mathematical verification: dy/dx = 6x + 2 = 6*3 + 2 = 20 ✓

    ================================================================================
    PART 3: GRADIENT DESCENT OPTIMIZATION
    ================================================================================

    3.1: Finding Minimum of y = (x - 3)^2
    ----------------------------------------
    Initial x: 0.0000
    Iteration 10: x = 2.6779, y = 0.1621
    Iteration 20: x = 2.9654, y = 0.0019
    Iteration 30: x = 2.9963, y = 0.0000
    Iteration 40: x = 2.9996, y = 0.0000
    Iteration 50: x = 3.0000, y = 0.0000

    Final x: 3.0000
    Target x: 3.0000
    Final y: 0.000000

    3.2: Visualizing the Optimization
    ----------------------------------------
    Plot saved as: optimization_demo.png

    ================================================================================
    PART 4: KEY TAKEAWAYS
    ================================================================================

    1. 텐서가 PyTorch의 밑바탕 데이터 짜임이다
       - 넘파이 배열과 비슷하되 GPU 받침과 자동 미분이 있다
       - torch.tensor(), torch.zeros(), torch.randn() 따위로 만든다

    2. 자동 미분(autograd)
       - 셈을 좇으려면 requires_grad=True를 둔다
       - .backward()을 불러 모든 그래디언트를 계산한다
       - 그래디언트는 .grad 속성에 담긴다

    3. 경사 하강법
       - 함수를 가장 작게 하도록 파라메터를 거듭 고친다
       - 고침 규칙: x_new = x_old - learning_rate * gradient
       - 루프 사이에는 늘 .zero_grad()을 불러라!

    4. 토치와 넘파이 견주기
       - torch.tensor와 np.array
       - PyTorch는 GPU(CUDA)에서 돌 수 있다
       - PyTorch에는 붙박이 자동 미분이 있다


    ================================================================================
    EXERCISES (Try These!)
    ================================================================================

    1. 쉬움: 러닝레이트를 0.01과 0.5으로 바꾸어라.
       모여드는 속도에 어떤 영향을 주는가?

    2. 보통: y = x^4 - 4x^2 + 4을 가장 작게 해 보아라
       이 함수에는 그 자리 골짜기가 둘 있다!
       x=0과 x=3에서 비롯하면 무엇을 얻는가?

    3. 보통: 2차원 함수에 경사 하강법을 짜라.
       y = (x1 - 2)^2 + (x2 + 1)^2
       골짜기는 (2, -1)에 있어야 한다

    4. 어려움: 경사 하강법에 여세를 더하여라.
       velocity = 0.9 * velocity + learning_rate * gradient
       x = x - velocity
       여느 경사 하강법과 모여드는 모습을 견주어라.


    ================================================================================
    NEXT STEPS
    ================================================================================

    PyTorch 텐서와 경사 하강법의 기초를 배웠다!

    다음 튜토리얼: 02_simple_binary_classification.py
    - 이 개념을 참 기계 학습 문제에 쓴다
    - 로지스틱 회귀를 밑바닥부터 짠다
    - 둘 분류기를 익힌다

    이어서 할 채비가 되었는가? 다음을 돌려라.
        python 02_simple_binary_classification.py

    ================================================================================
    ```


## 2. 논의

### 배열 연산과 자료형

파이썬 리스트와 NumPy 배열에서 텐서를 만들고 원소별 연산, 내적과 행렬 곱을 구분한다. `from_numpy()`는 지원되는 CPU 배열과 저장 공간을 공유하므로 한쪽 수정이 다른 쪽에 영향을 줄 수 있다. 정수형에는 역방향 경사 추적을 사용할 수 없으며 실수·복소수 연산도 지원하는 미분 규칙의 범위에서만 추적된다.

### 스칼라 함수의 미분

$x^2$와 $3x^2+2x+1$의 경사를 구하며 `requires_grad`, `.backward()`, `.grad`의 관계를 확인한다. `.grad`는 기본적으로 누적되므로 독립 업데이트 사이에 초기화한다. 단일 원소 출력은 모양이 $(1,)$이어도 경사 인수를 생략할 수 있다.

### 최적화 예제의 범위

$(x-3)^2$를 미분해 $x$를 업데이트한다. 일반적인 학습에서는 `no_grad()` 안에서 값을 바꾸고 다음 반복에 목적을 새로 계산한다. 이 페이지에는 로지스틱 분류 학습과 옵티마이저 객체가 없으므로 후속 페이지의 학습 루프와 혼동하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이차 함수의 직접 업데이트에서 경사를 초기화하는 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"
    `.backward()`는 기존 `.grad`에 새 경사를 더한다. 새 목적을 계산해도 저장된 경사는 지워지지 않으므로 독립 업데이트 사이에 초기화한다. 이 예제는 옵티마이저 객체 대신 텐서를 직접 업데이트한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`from_numpy()`로 만든 텐서와 `torch.tensor()`로 만든 사본의 차이는 무엇인가?

</div>

??? success "연습문제 2 풀이"
    지원되는 NumPy 배열을 `from_numpy()`로 변환하면 CPU 저장 공간을 공유한다. `torch.tensor()`는 데이터를 복사하므로 원래 배열 수정에 독립적이다. 자료형과 장치의 변환 조건도 함께 확인한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
일차원 이차 함수 최적화에서 로스가 커지는 원인 두 가지를 어떻게 구분할 것인가?

</div>

??? success "연습문제 3 풀이"
    러닝레이트가 곡률에 비해 크면 올바른 경사도 발산할 수 있다. 경사 초기화를 누락하거나 부호가 잘못되어도 다른 업데이트가 된다. 먼저 한 파라미터 지점의 해석적 경사와 비교한 뒤 안정 범위의 러닝레이트로 확인한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
조기 종료를 구현하라. 매 에포크 후 검증 로스를 추적하고, 10 에포크 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 웨이트를 저장하고 복원하라.

</div>

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 로스 추적기를 추가한다.
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
    검증 지표가 더 나아지지 않을 때 중단하고 가장 좋은 체크포인트를 복원하면 과적합을 줄이는 데 도움이 될 수 있다. 검증셋을 반복적인 선택에 사용한 결과와 독립 테스트 성능은 구분한다.

## 정리하며

자동 미분은 기록한 함수의 경사를 계산한다. 원소별 연산과 행렬 연산, 미분과 업데이트를 구분해 후속 분류 예제로 연결한다.
