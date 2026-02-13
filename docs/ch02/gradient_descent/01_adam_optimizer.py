"""
================================================================================
Level 3 - Example 1: Adam Optimizer - Adaptive Moment Estimation
================================================================================

LEARNING OBJECTIVES:
- Understand adaptive learning rate methods
- Implement Adam optimizer from scratch
- Compare Adam with SGD and SGD+Momentum
- Learn when to use Adam vs other optimizers

DIFFICULTY: ⭐⭐⭐ Advanced

TIME: 40-50 minutes

PREREQUISITES:
- Completed Level 1 and Level 2
- Understanding of momentum
- Comfortable with gradient descent variants

================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("ADAM OPTIMIZER: ADAPTIVE MOMENT ESTIMATION")
print("="*80)

# ============================================================================
# PART 1: Understanding Adam
# ============================================================================
print("\n" + "="*80)
print("PART 1: WHAT IS ADAM?")
print("="*80)

print("""
ADAM (Adaptive Moment Estimation) combines:
1. Momentum (moving average of gradients)
2. RMSprop (adaptive learning rates per parameter)
3. Bias correction (accounts for initialization)

KEY INNOVATIONS:
----------------
• Maintains two moving averages:
  - m_t: First moment (mean) of gradients
  - v_t: Second moment (variance) of gradients

• Adapts learning rate for each parameter
• Works well with default hyperparameters
• Most popular optimizer in deep learning!

ALGORITHM:
----------
Initialize: m₀ = 0, v₀ = 0, t = 0

For each iteration:
1. t = t + 1
2. g_t = ∇L(θ_{t-1})                    # Compute gradient
3. m_t = β₁·m_{t-1} + (1-β₁)·g_t         # Update biased first moment
4. v_t = β₂·v_{t-1} + (1-β₂)·g_t²        # Update biased second moment
5. m̂_t = m_t / (1 - β₁ᵗ)                 # Bias correction
6. v̂_t = v_t / (1 - β₂ᵗ)                 # Bias correction
7. θ_t = θ_{t-1} - α·m̂_t / (√v̂_t + ε)  # Parameter update

Default hyperparameters:
- α (learning rate): 0.001
- β₁ (momentum): 0.9
- β₂ (RMSprop): 0.999
- ε (numerical stability): 1e-8
""")

# ============================================================================
# PART 2: Create Challenging Dataset
# ============================================================================
print("\n" + "="*80)
print("PART 2: DATASET - NON-LINEAR FUNCTION")
print("="*80)

torch.manual_seed(42)
np.random.seed(42)

# Non-linear function: y = sin(x) + 0.5*cos(2x)
n_samples = 200
X_numpy = np.linspace(-3, 3, n_samples).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + 0.5 * np.cos(2 * X_numpy) + np.random.randn(n_samples, 1) * 0.1

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"Dataset: {n_samples} samples")
print("Function: y = sin(x) + 0.5*cos(2x) + noise")
print("This is non-linear - we'll use a neural network!")

# ============================================================================
# PART 3: Define Neural Network
# ============================================================================

class NeuralNetwork(nn.Module):
    """
    Multi-layer neural network for non-linear regression
    Architecture: 1 → 20 → 20 → 1
    """
    def __init__(self, input_size=1, hidden_size=20, output_size=1):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# ============================================================================
# PART 4: Training Function
# ============================================================================

def train_model(model, optimizer, criterion, X, y, n_epochs=500):
    """Train model and return loss history"""
    loss_history = []
    
    for epoch in range(n_epochs):
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 100 == 0:
            print(f"  Epoch {epoch+1:4d}: Loss = {loss.item():.6f}")
    
    return loss_history

# ============================================================================
# PART 5: Compare Optimizers
# ============================================================================
print("\n" + "="*80)
print("PART 5: COMPARING OPTIMIZERS")
print("="*80)

n_epochs = 500
lr_sgd = 0.01
lr_adam = 0.001  # Adam typically uses smaller learning rate

print(f"\nTraining for {n_epochs} epochs...\n")

# 1. SGD (vanilla)
print("1. Training with SGD...")
model_sgd = NeuralNetwork()
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=lr_sgd)
criterion = nn.MSELoss()
loss_sgd = train_model(model_sgd, optimizer_sgd, criterion, X, y, n_epochs)

# 2. SGD with Momentum
print("\n2. Training with SGD + Momentum...")
model_momentum = NeuralNetwork()
optimizer_momentum = torch.optim.SGD(model_momentum.parameters(), lr=lr_sgd, momentum=0.9)
loss_momentum = train_model(model_momentum, optimizer_momentum, criterion, X, y, n_epochs)

# 3. Adam
print("\n3. Training with Adam...")
model_adam = NeuralNetwork()
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=lr_adam)
loss_adam = train_model(model_adam, optimizer_adam, criterion, X, y, n_epochs)

# 4. Adam with custom parameters
print("\n4. Training with Adam (custom β values)...")
model_adam_custom = NeuralNetwork()
optimizer_adam_custom = torch.optim.Adam(
    model_adam_custom.parameters(), 
    lr=lr_adam, 
    betas=(0.9, 0.999),  # β₁, β₂
    eps=1e-8
)
loss_adam_custom = train_model(model_adam_custom, optimizer_adam_custom, criterion, X, y, n_epochs)

# ============================================================================
# PART 6: Evaluation
# ============================================================================
print("\n" + "="*80)
print("PART 6: EVALUATION")
print("="*80)

print("\nFinal Loss Comparison:")
print(f"  SGD:               {loss_sgd[-1]:.6f}")
print(f"  SGD + Momentum:    {loss_momentum[-1]:.6f}")
print(f"  Adam:              {loss_adam[-1]:.6f}")
print(f"  Adam (custom):     {loss_adam_custom[-1]:.6f}")

# Compute predictions for visualization
models = {
    'SGD': model_sgd,
    'SGD+Momentum': model_momentum,
    'Adam': model_adam,
    'Adam (custom)': model_adam_custom
}

predictions = {}
with torch.no_grad():
    for name, model in models.items():
        model.eval()
        predictions[name] = model(X).numpy()

# ============================================================================
# PART 7: Visualization
# ============================================================================
print("\n" + "="*80)
print("VISUALIZATION")
print("="*80)

fig = plt.figure(figsize=(16, 10))

# Plot 1: Loss curves (linear)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(loss_sgd, label='SGD', linewidth=2, alpha=0.8)
ax1.plot(loss_momentum, label='SGD+Momentum', linewidth=2, alpha=0.8)
ax1.plot(loss_adam, label='Adam', linewidth=2, alpha=0.8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss (Linear Scale)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Loss curves (log)
ax2 = plt.subplot(2, 3, 2)
ax2.plot(loss_sgd, label='SGD', linewidth=2, alpha=0.8)
ax2.plot(loss_momentum, label='SGD+Momentum', linewidth=2, alpha=0.8)
ax2.plot(loss_adam, label='Adam', linewidth=2, alpha=0.8)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss (Log Scale)')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Early training (first 50 epochs)
ax3 = plt.subplot(2, 3, 3)
ax3.plot(loss_sgd[:50], label='SGD', linewidth=2, alpha=0.8)
ax3.plot(loss_momentum[:50], label='SGD+Momentum', linewidth=2, alpha=0.8)
ax3.plot(loss_adam[:50], label='Adam', linewidth=2, alpha=0.8)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.set_title('Early Training (First 50 Epochs)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: SGD predictions
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(X.numpy(), y.numpy(), alpha=0.3, label='Data', s=20)
ax4.plot(X.numpy(), predictions['SGD'], 'r-', linewidth=2, label='SGD')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('SGD Fit')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: SGD+Momentum predictions
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(X.numpy(), y.numpy(), alpha=0.3, label='Data', s=20)
ax5.plot(X.numpy(), predictions['SGD+Momentum'], 'g-', linewidth=2, label='SGD+Momentum')
ax5.set_xlabel('x')
ax5.set_ylabel('y')
ax5.set_title('SGD+Momentum Fit')
ax5.legend()
ax5.grid(True, alpha=0.3)

# Plot 6: Adam predictions
ax6 = plt.subplot(2, 3, 6)
ax6.scatter(X.numpy(), y.numpy(), alpha=0.3, label='Data', s=20)
ax6.plot(X.numpy(), predictions['Adam'], 'b-', linewidth=2, label='Adam')
ax6.set_xlabel('x')
ax6.set_ylabel('y')
ax6.set_title('Adam Fit')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_gradient_descent_tutorial/level_3_advanced/adam_comparison.png', dpi=150)
print("\n✓ Plot saved as 'adam_comparison.png'")
print("\nClose the plot window to continue...")
plt.show()

# ============================================================================
# PART 8: Understanding Adam's Behavior
# ============================================================================
print("\n" + "="*80)
print("PART 8: WHY ADAM WORKS SO WELL")
print("="*80)

print("""
ADAM'S KEY ADVANTAGES:
----------------------

1. ADAPTIVE LEARNING RATES
   • Different learning rate for each parameter
   • Automatically scales based on gradient history
   • Parameters with large/small gradients get appropriate updates

2. MOMENTUM BENEFITS
   • Smooths out noisy gradients
   • Accelerates in consistent directions
   • Reduces oscillations

3. BIAS CORRECTION
   • Accounts for initialization bias (m₀=0, v₀=0)
   • Particularly important in early training
   • Makes default hyperparameters more robust

4. ROBUST TO HYPERPARAMETERS
   • Default values (α=0.001, β₁=0.9, β₂=0.999) work well
   • Less sensitive to learning rate choice
   • Good across different problems

WHEN ADAM EXCELS:
-----------------
✓ Deep neural networks
✓ Sparse gradients (NLP, RL)
✓ Non-stationary objectives
✓ Noisy gradients
✓ When you want "set it and forget it" performance

WHEN TO USE ALTERNATIVES:
--------------------------
• SGD+Momentum: Better generalization on vision tasks
• Batch GD: Small datasets with convex problems
• RMSprop: RNNs and online learning
• AdaGrad: Sparse features
""")

# ============================================================================
# PART 9: Hyperparameter Sensitivity
# ============================================================================
print("\n" + "="*80)
print("PART 9: ADAM HYPERPARAMETERS")
print("="*80)

print("""
ADAM HYPERPARAMETERS:
---------------------

1. LEARNING RATE (α)
   • Default: 0.001
   • Typical range: 0.0001 to 0.01
   • Start with 0.001, adjust if needed
   • Can use learning rate scheduling

2. BETA_1 (β₁) - First moment decay
   • Default: 0.9
   • Controls momentum
   • Higher = more smoothing
   • Rarely needs tuning

3. BETA_2 (β₂) - Second moment decay
   • Default: 0.999
   • Controls adaptive learning rate
   • Higher = more stable
   • For sparse gradients, try 0.99

4. EPSILON (ε) - Numerical stability
   • Default: 1e-8
   • Prevents division by zero
   • Usually don't need to change

TUNING TIPS:
------------
1. Start with defaults - they work 80% of the time
2. If training unstable: reduce learning rate
3. If training too slow: increase learning rate
4. For sparse problems: reduce β₂ to 0.99
5. For very noisy gradients: increase β₁ to 0.95
""")

# ============================================================================
# PART 10: Key Takeaways
# ============================================================================
print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. ADAM is the most popular optimizer
   • Combines momentum and adaptive learning rates
   • Works well with default hyperparameters
   • Good first choice for most problems

2. ADAM converges faster than SGD
   • Especially beneficial for deep networks
   • Handles different gradient magnitudes well
   • More stable training

3. MOMENTUM matters
   • Even SGD+Momentum significantly outperforms vanilla SGD
   • Accelerates convergence
   • Reduces oscillations

4. LEARNING RATE is still important
   • Adam uses smaller lr (0.001) vs SGD (0.01)
   • Still the most important hyperparameter
   • Use learning rate scheduling for best results

5. CHOOSE optimizer based on task
   • Adam: Default choice, works well everywhere
   • SGD+Momentum: Computer vision, when generalization matters
   • RMSprop: Recurrent networks
   • AdaGrad: Sparse features

6. UNDERSTAND the math
   • First moment (m): Average gradient (momentum)
   • Second moment (v): Average squared gradient (scaling)
   • Bias correction: Fixes initialization issues
""")

print("="*80)
print("EXPERIMENTS TO TRY")
print("="*80)
print("""
1. Different learning rates:
   - Try Adam with lr=0.01, 0.0001
   - How does it affect convergence?

2. Different network architectures:
   - Deeper networks (3-4 layers)
   - Wider networks (50-100 hidden units)
   - Which optimizer works better?

3. Other Adam variants:
   - AdamW: torch.optim.AdamW (weight decay)
   - AMSGrad: torch.optim.Adam(..., amsgrad=True)

4. Learning rate scheduling:
   - torch.optim.lr_scheduler.StepLR
   - torch.optim.lr_scheduler.ReduceLROnPlateau
""")
print("="*80)
