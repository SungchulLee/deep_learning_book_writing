# NumPy로 직접 구현하는 경사 하강법

이 스크립트는 NumPy로 경사 하강법을 직접 구현하는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""
================================================================================
Level 1 - Example 1: Manual Gradient Descent with NumPy
================================================================================

LEARNING OBJECTIVES:
- Understand gradient descent from first principles
- Manually compute gradients using calculus
- Implement gradient descent without any ML libraries
- Visualize the optimization process

DIFFICULTY: ⭐ Beginner

TIME: 20-30 minutes

PREREQUISITES:
- Basic Python
- Elementary calculus (derivatives)
- Understanding of linear functions

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# 1부: 문제 이해하기
# ============================================================================
print("="*80)
print("GRADIENT DESCENT: LEARNING FROM SCRATCH")
print("="*80)
print("\nProblem: We want to fit a line y = w*x to our data")
print("Goal: Find the best weight 'w' that minimizes prediction error\n")

# ============================================================================
# 2부: 학습 데이터 준비
# ============================================================================
# 참 관계가 y = 2*x인 간단한 데이터셋을 만들어 보자
# 즉 최적 가중치는 w* = 2이다

X_train = np.array([1, 2, 3, 4, 5], dtype=np.float32)  # Input features
y_train = np.array([2, 4, 6, 8, 10], dtype=np.float32)  # True labels (y = 2*x)

print("Training Data:")
print(f"X (inputs):  {X_train}")
print(f"y (targets): {y_train}")
print(f"\nTrue relationship: y = 2*x (so optimal w = 2.0)\n")

# ============================================================================
# 3부: 모델 매개변수 초기화
# ============================================================================
# 가중치를 무작위로 추측하여 시작
w = 0.0  # Initial weight (our model will learn the correct value)

print(f"Initial weight w = {w:.3f}")
print(f"Initial prediction for x=5: y = {w * 5:.3f} (should be 10.0)\n")

# ============================================================================
# 4부: 모델 구성 요소 정의
# ============================================================================

def model_forward(x, weight):
    """
    Forward pass: compute predictions
    
    Our model: y_pred = w * x
    
    Args:
        x: input data (array or scalar)
        weight: current weight parameter
    
    Returns:
        predictions: model output
    """
    return weight * x


def compute_loss(y_true, y_pred):
    """
    Mean Squared Error (MSE) Loss Function
    
    Loss measures how wrong our predictions are:
    L(w) = (1/N) * Σ(y_pred - y_true)²
    
    Args:
        y_true: actual target values
        y_pred: predicted values
    
    Returns:
        loss: average squared error (scalar)
    """
    # 제곱 차이 계산
    squared_errors = (y_pred - y_true) ** 2
    
    # 평균 오차를 반환한다
    return np.mean(squared_errors)


def compute_gradient(x, y_true, y_pred):
    """
    Manually compute gradient dL/dw
    
    Mathematical Derivation:
    ------------------------
    Loss:     L(w) = (1/N) * Σ(w*x - y)²
    
    Derivative:  dL/dw = (1/N) * Σ 2*(w*x - y)*x
                       = (2/N) * Σ x*(w*x - y)
                       = (2/N) * Σ x*(y_pred - y_true)
    
    In code: gradient = mean(2 * x * (y_pred - y_true))
    
    Args:
        x: input data
        y_true: actual targets
        y_pred: predicted values
    
    Returns:
        gradient: dL/dw (tells us which direction to update w)
    """
    # 위에서 유도한 공식으로 경사 계산
    gradient = np.mean(2 * x * (y_pred - y_true))
    return gradient


# ============================================================================
# 5부: 학습 설정
# ============================================================================
learning_rate = 0.01  # How big of a step to take in each iteration
n_iterations = 20     # Number of training iterations (epochs)

print("="*80)
print("TRAINING PROCESS")
print("="*80)
print(f"Learning rate: {learning_rate}")
print(f"Iterations: {n_iterations}\n")

# 시각화를 위해 학습 이력을 저장할 리스트
weight_history = [w]
loss_history = []

# ============================================================================
# 6부: 학습 루프(경사 하강법 알고리즘)
# ============================================================================
print("Epoch | Weight (w) | Loss      | Gradient")
print("-" * 50)

for epoch in range(n_iterations):
    # 1단계: 순전파
    # --------------------
    # 현재 가중치로 예측한다
    y_pred = model_forward(X_train, w)
    
    # 2단계: 손실 계산
    # --------------------
    # 예측이 얼마나 틀렸는지 측정
    loss = compute_loss(y_train, y_pred)
    
    # 3단계: 경사 계산
    # -------------------------
    # dL/dw 계산 - w를 어느 방향으로 옮길지 알려준다
    gradient = compute_gradient(X_train, y_train, y_pred)
    
    # 4단계: 매개변수 갱신(경사 하강 단계)
    # -------------------------------------------------
    # w를 경사의 반대 방향으로 옮긴다
    # 이것이 손실을 줄인다(예측이 좋아진다)
    w = w - learning_rate * gradient
    
    # 나중의 시각화를 위해 이력 저장
    weight_history.append(w)
    loss_history.append(loss)
    
    # 진행 상황 출력
    if epoch % 2 == 0:  # Print every 2 epochs
        print(f"{epoch+1:4d}  | {w:10.6f} | {loss:9.6f} | {gradient:9.6f}")

print("-" * 50)
print(f"\nFinal weight w = {w:.6f}")
print(f"Target weight = 2.0")
print(f"Error: {abs(w - 2.0):.6f}\n")

# ============================================================================
# 7부: 학습된 모델 시험하기
# ============================================================================
print("="*80)
print("TESTING")
print("="*80)

# 학습 데이터에서 시험
print("\nPredictions on training data:")
for x, y_true in zip(X_train, y_train):
    y_pred = model_forward(x, w)
    print(f"  x = {x:.0f} → y_pred = {y_pred:.3f}, y_true = {y_true:.0f}, error = {abs(y_pred-y_true):.3f}")

# 새 데이터에서 시험
print("\nPredictions on new data:")
test_inputs = [6, 7, 8]
for x in test_inputs:
    y_pred = model_forward(x, w)
    y_expected = 2 * x
    print(f"  x = {x:.0f} → y_pred = {y_pred:.3f}, y_expected = {y_expected:.0f}")

# ============================================================================
# 8부: 시각화
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

# 부분 그림 3개를 가진 도표 생성
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 부분 그림 1: 반복에 따른 손실
axes[0].plot(loss_history, 'b-', linewidth=2, marker='o', markersize=4)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Loss (MSE)', fontsize=12)
axes[0].set_title('Loss Decreases Over Time', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')  # Log scale to see improvement better

# 부분 그림 2: 가중치 수렴
axes[1].plot(weight_history, 'g-', linewidth=2, marker='s', markersize=4)
axes[1].axhline(y=2.0, color='r', linestyle='--', linewidth=2, label='Optimal weight (2.0)')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Weight (w)', fontsize=12)
axes[1].set_title('Weight Converges to Optimal Value', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 부분 그림 3: 최종 적합 결과
axes[2].scatter(X_train, y_train, color='blue', s=100, label='Training data', zorder=3)
x_line = np.linspace(0, 6, 100)
y_line = model_forward(x_line, w)
axes[2].plot(x_line, y_line, 'r-', linewidth=2, label=f'Learned line (w={w:.3f})')
axes[2].set_xlabel('x', fontsize=12)
axes[2].set_ylabel('y', fontsize=12)
axes[2].set_title('Fitted Line vs Training Data', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_1_basics/manual_gradient_descent.png', dpi=150)
print("\n✓ Plot saved as 'manual_gradient_descent.png'")
print("\nClose the plot window to continue...")
plt.show()

# ============================================================================
# 9부: 핵심 요점
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. GRADIENT DESCENT is an iterative optimization algorithm
   - Start with random parameters
   - Compute gradient (derivative of loss w.r.t. parameters)
   - Update parameters in opposite direction of gradient
   - Repeat until convergence

2. LEARNING RATE controls step size
   - Too large: might overshoot and diverge
   - Too small: slow convergence
   - Typical values: 0.001 to 0.1

3. LOSS FUNCTION measures prediction quality
   - MSE for regression problems
   - Different losses for different tasks

4. GRADIENT points in the direction of steepest increase
   - We move in opposite direction (-gradient) to decrease loss

5. CONVERGENCE happens when gradients become very small
   - Loss stops decreasing significantly
   - Parameters stabilize around optimal values
""")

print("="*80)
print("EXPERIMENT IDEAS:")
print("="*80)
print("""
Try modifying the code to explore:
1. Different learning rates (0.001, 0.1, 0.5) - what happens?
2. Different initial weights (5.0, -3.0) - does it still converge?
3. More training data points - does it improve accuracy?
4. Noisy data (add random noise to y) - how robust is gradient descent?
""")
print("="*80)


if __name__ == "__main__":
    pass
```

## 논의

CPU 텐서에서 PyTorch와 NumPy의 상호 운용은 매끄럽다. `torch.from_numpy()`는 배열과 메모리를 공유하는 텐서를 만들고, `torch.tensor()`는 항상 복사한다. 어떤 연산이 저장소를 공유하고 어떤 연산이 독립적인 복사본을 만드는지 이해하는 것이 미묘한 버그를 피하는 데 결정적이다.

여기서 보여준 패턴들은 실무적인 PyTorch 개발의 토대이다. 각 개념은 데이터 표현, 자동 미분, 하드웨어 가속을 하나의 일관된 API로 통합하는 텐서 추상화 위에 세워진다.

## 연습문제

**연습문제 1.**
다른 입력 값을 쓰도록 코드를 수정하고 출력이 어떻게 달라지는지 관찰하라.

??? success "연습문제 1 풀이"
    입력 매개변수나 데이터 값을 바꾸고 코드를 다시 실행한다. 출력을 비교하며 각 연산이 데이터를 어떻게 변환하는지에 대한 직관을 기른다.

---


**연습문제 2.**
코드의 어떤 연산이 뷰를 만들고 어떤 연산이 복사본을 만드는지 찾아라. `storage().data_ptr()`을 확인하여 답을 검증하라.

??? success "연습문제 2 풀이"
    슬라이싱, `view()`, `transpose()` 같은 연산은 뷰를 만든다(`data_ptr`가 같다). `clone()`, 불리언 인덱싱, 정수 배열 인덱싱 같은 연산은 복사본을 만든다(`data_ptr`가 다르다).

---


**연습문제 3.**
위에서 보여준 개념 두 가지 이상을 결합한 예제를 추가하여 코드를 확장하라.

??? success "연습문제 3 풀이"
    보여준 연산들을 작은 파이프라인으로 결합한다. 예를 들어 데이터를 만들고, 변환을 적용하고, 그 결과를 간단한 계산에 사용한다. 이렇게 하면 연산들이 어떻게 합성되는지에 대한 이해가 굳어진다.
