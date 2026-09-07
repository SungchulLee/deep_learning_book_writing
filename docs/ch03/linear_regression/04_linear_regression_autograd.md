# Autograd로 만드는 선형 회귀

PyTorch의 autograd 체계는 경사 공식을 손으로 유도하고 코딩할 필요를 없애 준다. 매개변수 텐서에 `requires_grad=True`를 설정하고 손실에 `.backward()`를 호출하면, 프레임워크가 기록된 계산 그래프를 훑으며 정확한 경사를 자동으로 채워 넣는다. 이 튜토리얼은 autograd가 직접 계산하는 방식과 동일한 결과를 내면서도 선형 회귀 학습 루프를 얼마나 간단하게 만들어 주는지 보여준다.

## 코드

```python
"""
==============================================================================
04_linear_regression_autograd.py
==============================================================================
어려움: ⭐⭐ (가운데)

DESCRIPTION:
    PyTorch의 자동 미분(autograd)을 쓰는 선형 회귀.
    이제 손수 기울기를 셈하지 않는다! 미적분은 PyTorch에 맡긴다.

다루는 것:
    - 자동 미분을 위한 requires_grad=True 쓰기
    - 기울기 셈을 위한 .backward()
    - 기울기 쌓기와 0으로 만들기
    - torch.no_grad() 자리

PREREQUISITES:
    - 익힘 01(autograd을 곁들인 PyTorch 기초)
    - 익힘 03(손수 하는 PyTorch 기울기)

배움 목표:
    - 기울기 셈에 autograd을 쓴다
    - 언제 기울기를 0으로 만들지 이해한다
    - 잘 들도록 no_grad() 자리를 쓴다
    - 손수 셈하는 코드와 견준다

걸리는 때: 15분쯤
==============================================================================
"""

import torch
import matplotlib.pyplot as plt

print("=" * 70)
print("LINEAR REGRESSION WITH AUTOGRAD")
print("=" * 70)

# ============================================================================
# 1부: 데이터 생성
# ============================================================================
print("\n" + "=" * 70)
print("PART 1: GENERATE DATA")
print("=" * 70)

torch.manual_seed(42)

TRUE_W = 2.0
TRUE_B = 1.0
n_samples = 100

# 데이터를 생성한다
X = torch.rand(n_samples) * 20 - 10  # Random values between -10 and 10
noise = torch.randn(n_samples) * 2    # Gaussian noise
y = TRUE_W * X + TRUE_B + noise

print(f"Generated {n_samples} samples")
print(f"True parameters: w={TRUE_W}, b={TRUE_B}")

# ============================================================================
# 2부: requires_grad=True로 매개변수 초기화
# ============================================================================
print("\n" + "=" * 70)
print("PART 2: INITIALIZE PARAMETERS WITH AUTOGRAD")
print("=" * 70)

# 핵심 차이: requires_grad=True로 설정한다
# 이는 PyTorch에게 이 텐서들에 대한 연산을 추적하여
# 경사를 자동으로 계산하라고 알려 준다
w = torch.tensor([0.0], requires_grad=True)  # ← requires_grad=True!
b = torch.tensor([0.0], requires_grad=True)  # ← requires_grad=True!

print(f"Parameters initialized:")
print(f"  w: {w.item():.4f}, requires_grad={w.requires_grad}")
print(f"  b: {b.item():.4f}, requires_grad={b.requires_grad}")

# ============================================================================
# 3부: 모델과 손실 정의
# ============================================================================
print("\n" + "=" * 70)
print("PART 3: DEFINE MODEL AND LOSS")
print("=" * 70)

def model(X, w, b):
    """선형 모델: y = w*X + b"""
    return w * X + b

def mse_loss(y_true, y_pred):
    """평균제곱오차 손실"""
    return torch.mean((y_true - y_pred) ** 2)

print("Model and loss functions defined")
print("Note: Same as before, but now PyTorch tracks operations")

# ============================================================================
# 4부: AUTOGRAD를 쓰는 학습 루프
# ============================================================================
print("\n" + "=" * 70)
print("PART 4: TRAINING LOOP WITH AUTOGRAD")
print("=" * 70)

learning_rate = 0.01
n_epochs = 100

loss_history = []
w_history = [w.item()]
b_history = [b.item()]

print(f"Training Configuration:")
print(f"  Learning rate: {learning_rate}")
print(f"  Epochs: {n_epochs}")
print(f"\n{'Epoch':<8} {'Loss':<12} {'w':<12} {'b':<12} {'grad_w':<12} {'grad_b':<12}")
print("-" * 75)

for epoch in range(n_epochs):
    # 1. 순전파: 예측과 손실을 계산한다
    #    PyTorch가 계산 그래프를 자동으로 만든다
    y_pred = model(X, w, b)
    loss = mse_loss(y, y_pred)
    
    # 2. 역전파: 경사를 자동으로 계산한다!
    #    여기서 마법이 일어난다 - 경사 공식을 손으로 쓸 필요가 없다!
    loss.backward()  # Computes gradients via backpropagation
    
    # 이제 w.grad와 b.grad에 경사가 들어 있다
    # (앞에서 직접 계산했던 것과 같은 값이다!)
    
    # 3. 매개변수를 갱신한다
    #    torch.no_grad()를 쓰는 이유는 매개변수 갱신 연산을
    #    계산 그래프에 기록하고 싶지 않기 때문이다
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # 이력 저장
    loss_history.append(loss.item())
    w_history.append(w.item())
    b_history.append(b.item())
    
    # 4. 다음 반복을 위해 경사를 0으로 만든다
    #    매우 중요: 경사는 기본적으로 누적된다!
    #    다음 역전파 전에 반드시 0으로 만들어야 한다
    w.grad.zero_()
    b.grad.zero_()
    
    # 진행 상황 출력
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"{epoch+1:<8} {loss.item():<12.4f} {w.item():<12.4f} "
              f"{b.item():<12.4f} {w.grad.item():<12.4f} {b.grad.item():<12.4f}")

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)
print(f"\nFinal Results:")
print(f"  Learned w: {w.item():.4f} (True: {TRUE_W}, Error: {abs(w.item()-TRUE_W):.4f})")
print(f"  Learned b: {b.item():.4f} (True: {TRUE_B}, Error: {abs(b.item()-TRUE_B):.4f})")
print(f"  Final loss: {loss_history[-1]:.4f}")

# ============================================================================
# 5부: 경사 누적 이해하기
# ============================================================================
print("\n" + "=" * 70)
print("PART 5: UNDERSTANDING GRADIENT ACCUMULATION")
print("=" * 70)

print("""
기울기를 0으로 만들어야 하는 까닭은?

PyTorch은 기본으로 기울기를 쌓는다. 큰 묶음을 흉내내는 기울기 쌓기 같은
앞선 자리에서는 쓸모 있지만,
여느 익힘에서는 되돌이마다 새 기울기를 바란다.

0으로 만들지 않으면 어떻게 되는지 보기:
""")

# 시연
x_demo = torch.tensor([2.0], requires_grad=True)

# 첫 번째 역전파
y = x_demo ** 2
y.backward()
print(f"After first backward: x_demo.grad = {x_demo.grad.item()}")  # Should be 4

# 경사를 초기화하지 않은 두 번째 역전파
y = x_demo ** 2
y.backward()
print(f"After second backward (accumulated): x_demo.grad = {x_demo.grad.item()}")  # 4 + 4 = 8

# 이제 0으로 만들고 다시 해 보자
x_demo.grad.zero_()
y = x_demo ** 2
y.backward()
print(f"After zeroing and third backward: x_demo.grad = {x_demo.grad.item()}")  # Back to 4

print("\nThis is why we call w.grad.zero_() in the training loop!")

# ============================================================================
# 6부: torch.no_grad() 문맥 사용하기
# ============================================================================
print("\n" + "=" * 70)
print("PART 6: UNDERSTANDING torch.no_grad()")
print("=" * 70)

print("""
torch.no_grad()은 기울기 좇기를 잠깐 끈다.
다음 때에 쓴다.
1. Making predictions (inference)
2. Updating parameters (as we did in the training loop)
3. Any operation where you don't need gradients

Benefits:
- Saves memory (no computational graph)
- Faster computation
- Prevents accidental gradient computation

Example:
""")

x = torch.tensor([1.0], requires_grad=True)

# 경사 추적을 켠 경우
y = x ** 2
print(f"With gradients: y.requires_grad = {y.requires_grad}")

# 경사 추적을 끈 경우
with torch.no_grad():
    y_no_grad = x ** 2
    print(f"Inside no_grad: y_no_grad.requires_grad = {y_no_grad.requires_grad}")

print("\nThis is essential for efficient inference and parameter updates!")

# ============================================================================
# 7부: 결과 시각화
# ============================================================================
print("\n" + "=" * 70)
print("PART 7: VISUALIZE RESULTS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 그림 1: 손실 곡선
axes[0, 0].plot(loss_history, linewidth=2, color='purple')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (MSE)')
axes[0, 0].set_title('Training Loss with Autograd')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_yscale('log')

# 그림 2: 매개변수의 변화
axes[0, 1].plot(w_history, label='w (slope)', linewidth=2, color='blue')
axes[0, 1].axhline(y=TRUE_W, color='r', linestyle='--', linewidth=2, label=f'True w={TRUE_W}')
axes[0, 1].plot(b_history, label='b (intercept)', linewidth=2, color='green')
axes[0, 1].axhline(y=TRUE_B, color='orange', linestyle='--', linewidth=2, label=f'True b={TRUE_B}')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Parameter Value')
axes[0, 1].set_title('Parameter Convergence (Autograd)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 그림 3: 최종 적합
with torch.no_grad():  # No gradients needed for visualization
    X_sorted, _ = torch.sort(X)
    y_pred_sorted = model(X_sorted, w, b)

axes[1, 0].scatter(X.numpy(), y.numpy(), alpha=0.5, s=20, label='Data')
axes[1, 0].plot(X_sorted.numpy(), (TRUE_W * X_sorted + TRUE_B).numpy(), 
                'r--', linewidth=2, label=f'True: y={TRUE_W}x+{TRUE_B}')
axes[1, 0].plot(X_sorted.numpy(), y_pred_sorted.numpy(), 
                'g-', linewidth=2, label=f'Learned: y={w.item():.2f}x+{b.item():.2f}')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('y')
axes[1, 0].set_title('Data with Learned Model')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 그림 4: 비교 표
comparison_text = f"""
AUTOGRAD VS MANUAL GRADIENTS

Code Complexity:
  Manual:   5+ lines for gradient formulas
  Autograd: 1 line (loss.backward())

Flexibility:
  Manual:   Hard to extend
  Autograd: Works for any function

Errors:
  Manual:   Easy to make mistakes
  Autograd: Automatic, no mistakes

Performance:
  Manual:   Similar
  Autograd: Highly optimized

Results:
  Final w: {w.item():.4f} (Error: {abs(w.item()-TRUE_W):.4f})
  Final b: {b.item():.4f} (Error: {abs(b.item()-TRUE_B):.4f})
"""
axes[1, 1].text(0.1, 0.95, comparison_text, 
                transform=axes[1, 1].transAxes,
                fontsize=9, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_linear_regression_tutorial/04_autograd_results.png', dpi=100)
print("Saved visualization to: 04_autograd_results.png")
plt.show()

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
고갱이 배움:

1. AUTOGRAD BASICS:
   - 매개변수에 requires_grad=True을 둔다
   - PyTorch builds computational graph automatically
   - .backward()을 불러 모든 기울기를 셈한다

2. TRAINING LOOP STRUCTURE:
   for epoch in range(n_epochs):
       # 순전파
       y_pred = model(X, w, b)
       loss = loss_function(y, y_pred)
       
       # 역전파
       loss.backward()  # ← Computes gradients automatically
       
       # 매개변수 갱신
       with torch.no_grad():
           w -= learning_rate * w.grad
           b -= learning_rate * b.grad
       
       # 경사 초기화
       w.grad.zero_()
       b.grad.zero_()

3. IMPORTANT POINTS:
   ✓ Always zero gradients before backward()
   ✓ 매개변수를 고칠 때는 torch.no_grad()을 써라
   ✓ Gradients accumulate by default
   ✓ Same results as manual computation

4. ADVANTAGES:
   ✓ No manual gradient formulas
   ✓ Less error-prone
   ✓ Works for any differentiable function
   ✓ Scales to complex models

다음 걸음:
- Tutorial 05: Use nn.Module for cleaner code
- Tutorial 06: Multiple input features
- Tutorial 07: Polynomial regression
""")


if __name__ == "__main__":
    pass
```

## 논의

경사를 직접 계산하던 튜토리얼로부터의 핵심 변화는 놀랄 만큼 작다. 손으로 짠 경사 함수를 세 줄로 대체하면 된다. `loss.backward()`, 매개변수 갱신을 위한 `torch.no_grad()` 블록, 그리고 누적된 경사를 초기화하는 `w.grad.zero_()` / `b.grad.zero_()`이다. 데이터 생성, 모델 함수, 손실 함수, 학습 루프 구조 등 나머지는 모두 그대로이다. 이는 autograd가 근본적으로 다른 프로그래밍 방식이 아니라 직접 미분을 그대로 대체하는 수단임을 보여준다.

경사 누적은 PyTorch의 미묘하지만 중요한 기본 동작이다. `.backward()`를 호출할 때마다 기존 `.grad` 텐서를 덮어쓰는 것이 아니라 거기에 더한다. 마이크로배치에 걸친 경사 누적 같은 고급 상황에서는 유용하지만, 표준적인 학습 루프에서는 매 역전파 전에 경사를 명시적으로 0으로 만들어야 한다는 뜻이다. 이 단계를 잊는 것이 초심자가 가장 흔히 하는 실수 중 하나이며, 매개변수가 과거 경사의 총합을 받아 크게 튀어 버린다.

매개변수 갱신에서 `torch.no_grad()` 문맥은 두 가지 역할을 한다. 첫째, PyTorch가 갱신 연산을 계산 그래프에 기록하지 못하게 한다. 기록하면 메모리를 낭비하고 제자리 수정 규칙을 어기게 된다. 둘째, 의도를 드러낸다. 갱신은 미분 대상 함수의 일부가 아니므로 추적되어서는 안 된다. 평가나 추론 시에는 순전파 전체를 `torch.no_grad()`로 감싸면 그래프를 만들 필요가 없어 마찬가지로 메모리와 계산을 아낄 수 있다.

## 연습문제

**익힘 1.**
autograd가 앞 튜토리얼의 직접 계산 공식과 같은 경사를 내는지 확인하라. 동일한 데이터와 매개변수를 만들고 autograd로 순전파와 역전파를 한 번 수행한 뒤, 직접 계산한 경사와 비교하라.

??? success "익힘 1 풀이"
    ```python
    import torch
    torch.manual_seed(42)
    X = torch.rand(100) * 20 - 10
    noise = torch.randn(100) * 2
    y = 2.0 * X + 1.0 + noise
    
    w = torch.tensor([0.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    
    y_pred = w * X + b
    loss = torch.mean((y - y_pred) ** 2)
    loss.backward()
    
    # 직접 계산한 경사
    error = y_pred.detach() - y
    manual_grad_w = (2.0 / len(X)) * (error * X).sum()
    manual_grad_b = (2.0 / len(X)) * error.sum()
    
    print(f'Autograd: grad_w={w.grad.item():.6f}, grad_b={b.grad.item():.6f}')
    print(f'Manual:   grad_w={manual_grad_w.item():.6f}, grad_b={manual_grad_b.item():.6f}')
    # 값들이 수치 정밀도 안에서 일치해야 한다
    ```

---

**익힘 2.**
초기화하지 않고 `.backward()`를 두 번 호출하여 경사 누적을 보여라. 경사가 개별 경사 두 개의 합임을 보여라.

??? success "익힘 2 풀이"
    ```python
    import torch
    x = torch.tensor([3.0], requires_grad=True)
    
    # 첫 번째 역전파
    y1 = x ** 2
    y1.backward()
    grad_after_first = x.grad.item()  # 2*3 = 6
    
    # 0으로 만들지 않은 채 두 번째 역전파
    y2 = x ** 3
    y2.backward()
    grad_after_second = x.grad.item()  # 6 + 3*9 = 6 + 27 = 33
    
    print(f'After first backward (x^2): grad = {grad_after_first}')
    print(f'After second backward (x^3, accumulated): grad = {grad_after_second}')
    print(f'Sum check: {grad_after_first} + {3 * 3.0**2} = {grad_after_first + 3 * 3.0**2}')
    ```

---

**익힘 3.**
매개변수를 직접 갱신하는 대신 `torch.optim.SGD`를 쓰도록 학습 루프를 수정하라. 최종 학습된 매개변수를 비교하여 일치함을 확인하라.

??? success "익힘 3 풀이"
    ```python
    import torch
    torch.manual_seed(42)
    X = torch.rand(100) * 20 - 10
    y = 2.0 * X + 1.0 + torch.randn(100) * 2
    
    w = torch.tensor([0.0], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)
    optimizer = torch.optim.SGD([w, b], lr=0.01)
    
    for epoch in range(100):
        y_pred = w * X + b
        loss = torch.mean((y - y_pred) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Learned w={w.item():.4f}, b={b.item():.4f}')
    # w=2.0, b=1.0에 가까워야 한다
    ```
