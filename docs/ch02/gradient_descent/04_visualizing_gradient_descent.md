# 경사 하강법의 동작 시각화

이 스크립트는 경사 하강법이 동작하는 과정을 시각화하는 방법을 보여준다. 이 개념들을 이해하는 것은 효과적인 PyTorch 프로그래밍과 딥러닝 모델 개발에 필수적이다.

## 코드

```python
"""
================================================================================
Level 1 - Example 4: Visualizing Gradient Descent in Action
================================================================================

배움 목표:
- See gradient descent optimization visually
- Understand loss landscapes
- Observe effect of learning rate
- Visualize convergence paths

어려움: ⭐ 첫걸음

걸리는 때: 25~35분

================================================================================
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

print("="*80)
print("VISUALIZING GRADIENT DESCENT")
print("="*80)

# ============================================================================
# 1부: 간단한 1차원 손실 함수
# ============================================================================
print("\n" + "="*80)
print("PART 1: 1D QUADRATIC LOSS FUNCTION")
print("="*80)

# 손실 함수 정의: L(w) = (w - 3)²
def loss_fn(w):
    return (w - 3) ** 2

def gradient_fn(w):
    return 2 * (w - 3)

# 손실 함수 시각화
w_vals = np.linspace(-2, 8, 100)
loss_vals = [(w - 3) ** 2 for w in w_vals]

plt.figure(figsize=(10, 6))
plt.plot(w_vals, loss_vals, 'b-', linewidth=2, label='Loss function')
plt.axvline(x=3, color='r', linestyle='--', label='Minimum (w=3)')
plt.xlabel('Weight (w)', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Loss Landscape: L(w) = (w - 3)²', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_1_basics/loss_landscape_1d.png', dpi=150)
print("\n✓ Loss landscape visualization saved")
plt.show()

# ============================================================================
# 2부: 서로 다른 학습률로 경사 하강법 수행하기
# ============================================================================
print("\n" + "="*80)
print("PART 2: EFFECT OF LEARNING RATE")
print("="*80)

def run_gradient_descent(w_init, lr, n_steps):
    """Run gradient descent and return trajectory"""
    w = w_init
    trajectory = [w]
    
    for _ in range(n_steps):
        grad = gradient_fn(w)
        w = w - lr * grad
        trajectory.append(w)
    
    return trajectory

# 서로 다른 학습률 시도
learning_rates = [0.1, 0.5, 0.9, 1.1]
w_init = 7.0
n_steps = 15

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, lr in enumerate(learning_rates):
    trajectory = run_gradient_descent(w_init, lr, n_steps)
    loss_trajectory = [loss_fn(w) for w in trajectory]
    
    ax = axes[idx]
    
    # 손실 지형 그리기
    ax.plot(w_vals, loss_vals, 'gray', linewidth=1, alpha=0.5)
    
    # 궤적 그리기
    ax.plot(trajectory, loss_trajectory, 'ro-', linewidth=2, markersize=6)
    ax.plot(trajectory[0], loss_trajectory[0], 'go', markersize=12, label='Start')
    ax.plot(trajectory[-1], loss_trajectory[-1], 'r*', markersize=15, label='End')
    
    ax.axvline(x=3, color='blue', linestyle='--', alpha=0.5, label='Optimum')
    ax.set_xlabel('Weight (w)')
    ax.set_ylabel('Loss')
    ax.set_title(f'Learning Rate = {lr}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 20)
    
    # 주석 추가
    final_w = trajectory[-1]
    if abs(final_w - 3) < 0.5:
        status = "✓ Converged"
        color = 'green'
    elif abs(final_w) > 10:
        status = "✗ Diverged"
        color = 'red'
    else:
        status = "⚠ Oscillating"
        color = 'orange'
    
    ax.text(0.05, 0.95, status, transform=ax.transAxes,
            fontsize=12, fontweight='bold', color=color,
            verticalalignment='top')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_1_basics/learning_rate_effect.png', dpi=150)
print("\n✓ Learning rate comparison saved")
plt.show()

# ============================================================================
# 3부: 2차원 손실 곡면
# ============================================================================
print("\n" + "="*80)
print("PART 3: 2D LOSS SURFACE VISUALIZATION")
print("="*80)

# 선형 회귀를 위한 합성 데이터 생성
torch.manual_seed(42)
X_data = torch.randn(50, 1) * 2
y_data = 3 * X_data + 2 + torch.randn(50, 1) * 0.5

def compute_loss_2d(w, b):
    """Compute MSE loss for given w and b"""
    y_pred = w * X_data + b
    loss = torch.mean((y_pred - y_data) ** 2)
    return loss.item()

# 손실 곡면을 위한 격자 생성
w_range = np.linspace(1, 5, 50)
b_range = np.linspace(0, 4, 50)
W, B = np.meshgrid(w_range, b_range)

# 각 점에서의 손실 계산
Z = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = compute_loss_2d(W[i, j], B[i, j])

# PyTorch로 경사 하강법 실행
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
learning_rate = 0.1
n_steps = 30

# 궤적 저장
trajectory = [(w.item(), b.item())]

for step in range(n_steps):
    # 순전파
    y_pred = w * X_data + b
    loss = torch.mean((y_pred - y_data) ** 2)
    
    # 역전파
    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()
    loss.backward()
    
    # 갱신
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    trajectory.append((w.item(), b.item()))

# 그림
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 등고선 그림
contour = ax1.contour(W, B, Z, levels=20, cmap='viridis')
ax1.clabel(contour, inline=True, fontsize=8)

# 궤적 그리기
w_traj = [p[0] for p in trajectory]
b_traj = [p[1] for p in trajectory]
ax1.plot(w_traj, b_traj, 'r.-', linewidth=2, markersize=8, label='GD path')
ax1.plot(w_traj[0], b_traj[0], 'go', markersize=12, label='Start')
ax1.plot(w_traj[-1], b_traj[-1], 'r*', markersize=15, label='End')
ax1.plot(3, 2, 'b*', markersize=15, label='True optimum')

ax1.set_xlabel('Weight (w)', fontsize=12)
ax1.set_ylabel('Bias (b)', fontsize=12)
ax1.set_title('Loss Surface and Optimization Path', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 3차원 곡면 그림
from mpl_toolkits.mplot3d import Axes3D
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(W, B, Z, cmap='viridis', alpha=0.7)

# 궤적을 3차원으로 그리기
z_traj = [compute_loss_2d(w, b) for w, b in trajectory]
ax2.plot(w_traj, b_traj, z_traj, 'r.-', linewidth=2, markersize=8)

ax2.set_xlabel('Weight (w)')
ax2.set_ylabel('Bias (b)')
ax2.set_zlabel('Loss')
ax2.set_title('3D Loss Surface', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_1_basics/loss_surface_2d.png', dpi=150)
print("\n✓ 2D loss surface visualization saved")
plt.show()

# ============================================================================
# 4부: 수렴 분석
# ============================================================================
print("\n" + "="*80)
print("PART 4: CONVERGENCE ANALYSIS")
print("="*80)

# 반복에 따른 최적점까지의 거리 계산
distances = [np.sqrt((w - 3)**2 + (b - 2)**2) for w, b in trajectory]
losses = [compute_loss_2d(w, b) for w, b in trajectory]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 최적점까지의 거리
ax1.plot(distances, 'b-', linewidth=2, marker='o')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Distance to Optimum', fontsize=12)
ax1.set_title('Convergence: Distance to Optimal Parameters', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# 반복에 따른 손실
ax2.plot(losses, 'g-', linewidth=2, marker='s')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Loss Reduction Over Iterations', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_1_basics/convergence_analysis.png', dpi=150)
print("\n✓ Convergence analysis saved")
plt.show()

print(f"\nFinal parameters: w={w.item():.4f}, b={b.item():.4f}")
print(f"True parameters:  w=3.0000, b=2.0000")
print(f"Final loss: {losses[-1]:.6f}")

# ============================================================================
# 핵심 요점
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. LEARNING RATE is critical:
   • Too small: slow convergence
   • Too large: oscillation or divergence
   • Just right: smooth, fast convergence

2. LOSS LANDSCAPES can be complex:
   • 1D: simple parabolas
   • 2D+: valleys, ridges, saddle points
   • Deep networks: very high-dimensional!

3. GRADIENT DESCENT follows the path of steepest descent:
   • Always moves downhill
   • May take many steps to reach bottom
   • Path depends on starting point and learning rate

4. CONVERGENCE can be monitored:
   • Loss should decrease over time
   • Distance to optimum should decrease
   • Parameters should stabilize

5. VISUALIZATION helps understanding:
   • See what gradient descent is doing
   • Debug optimization problems
   • Choose good hyperparameters
""")

print("="*80)
print("CONGRATULATIONS!")
print("="*80)
print("""
You've completed Level 1! You now understand:
✓ How gradient descent works
✓ PyTorch's automatic differentiation
✓ Training neural networks
✓ Visualizing optimization

Ready for Level 2? Learn about mini-batches, momentum, and more!
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
