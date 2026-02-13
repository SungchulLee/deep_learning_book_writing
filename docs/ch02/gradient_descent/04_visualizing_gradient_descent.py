"""
================================================================================
Level 1 - Example 4: Visualizing Gradient Descent in Action
================================================================================

LEARNING OBJECTIVES:
- See gradient descent optimization visually
- Understand loss landscapes
- Observe effect of learning rate
- Visualize convergence paths

DIFFICULTY: ⭐ Beginner

TIME: 25-35 minutes

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
# PART 1: Simple 1D Loss Function
# ============================================================================
print("\n" + "="*80)
print("PART 1: 1D QUADRATIC LOSS FUNCTION")
print("="*80)

# Define loss function: L(w) = (w - 3)²
def loss_fn(w):
    return (w - 3) ** 2

def gradient_fn(w):
    return 2 * (w - 3)

# Visualize the loss function
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
# PART 2: Gradient Descent with Different Learning Rates
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

# Try different learning rates
learning_rates = [0.1, 0.5, 0.9, 1.1]
w_init = 7.0
n_steps = 15

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, lr in enumerate(learning_rates):
    trajectory = run_gradient_descent(w_init, lr, n_steps)
    loss_trajectory = [loss_fn(w) for w in trajectory]
    
    ax = axes[idx]
    
    # Plot loss landscape
    ax.plot(w_vals, loss_vals, 'gray', linewidth=1, alpha=0.5)
    
    # Plot trajectory
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
    
    # Add annotation
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
# PART 3: 2D Loss Surface
# ============================================================================
print("\n" + "="*80)
print("PART 3: 2D LOSS SURFACE VISUALIZATION")
print("="*80)

# Create synthetic data for linear regression
torch.manual_seed(42)
X_data = torch.randn(50, 1) * 2
y_data = 3 * X_data + 2 + torch.randn(50, 1) * 0.5

def compute_loss_2d(w, b):
    """Compute MSE loss for given w and b"""
    y_pred = w * X_data + b
    loss = torch.mean((y_pred - y_data) ** 2)
    return loss.item()

# Create grid for loss surface
w_range = np.linspace(1, 5, 50)
b_range = np.linspace(0, 4, 50)
W, B = np.meshgrid(w_range, b_range)

# Compute loss at each point
Z = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = compute_loss_2d(W[i, j], B[i, j])

# Run gradient descent with PyTorch
w = torch.tensor(1.5, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
learning_rate = 0.1
n_steps = 30

# Store trajectory
trajectory = [(w.item(), b.item())]

for step in range(n_steps):
    # Forward pass
    y_pred = w * X_data + b
    loss = torch.mean((y_pred - y_data) ** 2)
    
    # Backward pass
    if w.grad is not None:
        w.grad.zero_()
    if b.grad is not None:
        b.grad.zero_()
    loss.backward()
    
    # Update
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    trajectory.append((w.item(), b.item()))

# Plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Contour plot
contour = ax1.contour(W, B, Z, levels=20, cmap='viridis')
ax1.clabel(contour, inline=True, fontsize=8)

# Plot trajectory
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

# 3D surface plot
from mpl_toolkits.mplot3d import Axes3D
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(W, B, Z, cmap='viridis', alpha=0.7)

# Plot trajectory in 3D
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
# PART 4: Convergence Analysis
# ============================================================================
print("\n" + "="*80)
print("PART 4: CONVERGENCE ANALYSIS")
print("="*80)

# Compute distance to optimum over iterations
distances = [np.sqrt((w - 3)**2 + (b - 2)**2) for w, b in trajectory]
losses = [compute_loss_2d(w, b) for w, b in trajectory]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Distance to optimum
ax1.plot(distances, 'b-', linewidth=2, marker='o')
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Distance to Optimum', fontsize=12)
ax1.set_title('Convergence: Distance to Optimal Parameters', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# Loss over iterations
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
# KEY TAKEAWAYS
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
