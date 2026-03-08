"""
================================================================================
INTERMEDIATE 01: Comparing Popular Optimizers (SGD, Adam, RMSprop, AdamW)
================================================================================

WHAT YOU'LL LEARN:
- How different optimizers work
- When to use each optimizer
- Adam vs AdamW
- Momentum and adaptive learning rates
- Practical comparison on a real problem

PREREQUISITES:
- Complete all beginner tutorials
- Understand basic optimization concepts

TIME TO COMPLETE: ~25 minutes
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
import matplotlib.pyplot as plt

print("=" * 80)
print("COMPARING POPULAR OPTIMIZERS")
print("=" * 80)

# ============================================================================
# SECTION 1: Overview of Optimization Algorithms
# ============================================================================
print("\n" + "-" * 80)
print("OPTIMIZATION ALGORITHMS OVERVIEW")
print("-" * 80)

print("""
STOCHASTIC GRADIENT DESCENT (SGD):
  • Simplest optimizer: param -= learning_rate × gradient
  • Pros: Simple, works well with momentum
  • Cons: Can be slow, sensitive to learning rate
  • Best for: Well-understood problems, when you have time to tune

SGD WITH MOMENTUM:
  • Adds "velocity" to push through local minima
  • Accumulates gradients over time: v = momentum × v + gradient
  • Smooths out noisy gradients
  • Best for: Deep networks, noisy gradients

ADAM (Adaptive Moment Estimation):
  • Adapts learning rate for each parameter
  • Combines momentum + RMSprop
  • Pros: Works well out-of-the-box, fast convergence
  • Cons: Can generalize worse than SGD, higher memory usage
  • Best for: Quick prototyping, most deep learning tasks

ADAMW (Adam with Weight Decay):
  • Adam with correct weight decay implementation
  • Better generalization than Adam
  • Fixes Adam's weight decay bug
  • Best for: Transformers, modern architectures, when using weight decay

RMSPROP (Root Mean Square Propagation):
  • Adapts learning rate using moving average of squared gradients
  • Good for non-stationary problems
  • Best for: RNNs, online learning
""")

# ============================================================================
# SECTION 2: Create a Non-Trivial Optimization Problem
# ============================================================================
print("\n" + "-" * 80)
print("SETUP: Non-Convex Optimization Problem")
print("-" * 80)

# Set seed for reproducibility
torch.manual_seed(42)

# Generate synthetic data with non-linear pattern
n_samples = 200
X = torch.linspace(-3, 3, n_samples).reshape(-1, 1)
# Non-linear function: y = sin(x) + 0.5*x + noise
y = torch.sin(X) + 0.5 * X + torch.randn(n_samples, 1) * 0.1

print(f"Dataset: {n_samples} samples")
print(f"Task: Learn the non-linear function y = sin(x) + 0.5*x")

# ============================================================================
# SECTION 3: Define a Simple Neural Network
# ============================================================================
print("\n" + "-" * 80)
print("NEURAL NETWORK ARCHITECTURE")
print("-" * 80)

class SimpleNet(nn.Module):
    """
    Simple feedforward network with 2 hidden layers
    Input → 20 neurons → 20 neurons → Output
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Print architecture
model = SimpleNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"Architecture:")
print(model)
print(f"\nTotal parameters: {total_params}")

# ============================================================================
# SECTION 4: Training Function
# ============================================================================

def train_model(model, optimizer_name, optimizer, X, y, epochs=100):
    """Train model and return loss history"""
    criterion = nn.MSELoss()
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X)
        loss = criterion(y_pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Record loss
        loss_history.append(loss.item())
        
        # Print progress
        if (epoch + 1) % 20 == 0:
            print(f"  [{optimizer_name}] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    return loss_history

# ============================================================================
# SECTION 5: Compare Optimizers
# ============================================================================
print("\n" + "-" * 80)
print("TRAINING WITH DIFFERENT OPTIMIZERS")
print("-" * 80)

epochs = 100
learning_rate = 0.01

# Store results for comparison
results = {}
optimizers_to_test = {}

# 1. SGD (vanilla)
print("\n1. SGD (Vanilla - No Momentum):")
model_sgd = SimpleNet()
opt_sgd = optim.SGD(model_sgd.parameters(), lr=learning_rate)
optimizers_to_test['SGD'] = (model_sgd, opt_sgd)

# 2. SGD with Momentum
print("\n2. SGD with Momentum (0.9):")
model_sgd_momentum = SimpleNet()
opt_sgd_momentum = optim.SGD(model_sgd_momentum.parameters(), 
                               lr=learning_rate, momentum=0.9)
optimizers_to_test['SGD+Momentum'] = (model_sgd_momentum, opt_sgd_momentum)

# 3. RMSprop
print("\n3. RMSprop:")
model_rmsprop = SimpleNet()
opt_rmsprop = optim.RMSprop(model_rmsprop.parameters(), lr=learning_rate)
optimizers_to_test['RMSprop'] = (model_rmsprop, opt_rmsprop)

# 4. Adam
print("\n4. Adam:")
model_adam = SimpleNet()
opt_adam = optim.Adam(model_adam.parameters(), lr=learning_rate)
optimizers_to_test['Adam'] = (model_adam, opt_adam)

# 5. AdamW
print("\n5. AdamW (Adam with Weight Decay):")
model_adamw = SimpleNet()
opt_adamw = optim.AdamW(model_adamw.parameters(), lr=learning_rate, 
                         weight_decay=0.01)
optimizers_to_test['AdamW'] = (model_adamw, opt_adamw)

# Train all models
for name, (model, optimizer) in optimizers_to_test.items():
    loss_history = train_model(model, name, optimizer, X, y, epochs)
    results[name] = loss_history

# ============================================================================
# SECTION 6: Analyze Results
# ============================================================================
print("\n" + "-" * 80)
print("RESULTS ANALYSIS")
print("-" * 80)

print("\nFinal Loss (lower is better):")
for name, loss_history in results.items():
    final_loss = loss_history[-1]
    print(f"  {name:15s}: {final_loss:.6f}")

print("\nConvergence Speed (loss after 20 epochs):")
for name, loss_history in results.items():
    early_loss = loss_history[19]  # 20th epoch (0-indexed)
    print(f"  {name:15s}: {early_loss:.6f}")

# ============================================================================
# SECTION 7: Visualize Convergence
# ============================================================================
print("\n" + "-" * 80)
print("VISUALIZATION")
print("-" * 80)

plt.figure(figsize=(12, 5))

# Plot 1: Loss curves
plt.subplot(1, 2, 1)
for name, loss_history in results.items():
    plt.plot(loss_history, label=name, linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (MSE)', fontsize=12)
plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to see differences better

# Plot 2: Final predictions
plt.subplot(1, 2, 2)
# Plot true data
plt.scatter(X.numpy(), y.numpy(), alpha=0.3, s=10, label='True Data', color='black')

# Plot predictions from each optimizer
colors = ['blue', 'orange', 'green', 'red', 'purple']
with torch.no_grad():
    for (name, (model, _)), color in zip(optimizers_to_test.items(), colors):
        y_pred = model(X)
        plt.plot(X.numpy(), y_pred.numpy(), label=name, linewidth=2, 
                color=color, alpha=0.7)

plt.xlabel('X', fontsize=12)
plt.ylabel('Y', fontsize=12)
plt.title('Final Predictions', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = '/home/claude/optimizer_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

# ============================================================================
# SECTION 8: Detailed Optimizer Explanations
# ============================================================================
print("\n" + "-" * 80)
print("DETAILED OPTIMIZER MECHANICS")
print("-" * 80)

print("""
1. SGD (STOCHASTIC GRADIENT DESCENT):
   Update: θ = θ - lr × ∇L
   
   • Simplest algorithm
   • Each step goes directly downhill
   • Can oscillate in narrow valleys
   • Very sensitive to learning rate choice

2. SGD WITH MOMENTUM:
   Velocity: v = β × v + ∇L
   Update: θ = θ - lr × v
   
   • Accumulates past gradients (β typically 0.9)
   • Builds "momentum" to push through small bumps
   • Smooths out noisy gradients
   • Can overshoot optimal values

3. RMSPROP:
   RMS: s = β × s + (1-β) × ∇L²
   Update: θ = θ - (lr / √s) × ∇L
   
   • Adapts learning rate per parameter
   • Divides by root mean square of gradients
   • Prevents learning rate from becoming too small
   • Good for non-stationary objectives

4. ADAM (Adaptive Moment Estimation):
   Momentum: m = β₁ × m + (1-β₁) × ∇L
   Velocity: v = β₂ × v + (1-β₂) × ∇L²
   Update: θ = θ - lr × (m / √v)
   
   • Combines momentum + RMSprop
   • Adapts learning rate for each parameter
   • Includes bias correction
   • Default β₁=0.9, β₂=0.999
   • Most popular optimizer in deep learning

5. ADAMW:
   Same as Adam but with corrected weight decay:
   Update: θ = θ - lr × (m / √v) - lr × λ × θ
   
   • Fixes Adam's weight decay implementation
   • Better regularization
   • Preferred for transformers and large models
""")

# ============================================================================
# SECTION 9: Practical Guidelines
# ============================================================================
print("\n" + "-" * 80)
print("PRACTICAL GUIDELINES: Which Optimizer to Use?")
print("-" * 80)

print("""
🎯 USE ADAM WHEN:
   ✓ Starting a new project (good default)
   ✓ Need fast convergence
   ✓ Limited time for hyperparameter tuning
   ✓ Working with RNNs or transformers
   ✓ Example: Quick prototyping, research experiments

📊 USE SGD + MOMENTUM WHEN:
   ✓ Final model training (often better generalization)
   ✓ Have time to tune learning rate
   ✓ Training CNNs (especially ResNet, VGG)
   ✓ Want most stable long-term performance
   ✓ Example: Production models, ImageNet training

🔬 USE ADAMW WHEN:
   ✓ Training transformers (BERT, GPT, etc.)
   ✓ Using weight decay regularization
   ✓ Need better generalization than Adam
   ✓ Following modern best practices
   ✓ Example: NLP models, large-scale training

⚡ USE RMSPROP WHEN:
   ✓ Training RNNs (historically popular)
   ✓ Non-stationary problems
   ✓ Online learning scenarios
   ✓ Example: Time series, online recommendations

LEARNING RATE RECOMMENDATIONS:
• SGD: 0.01 - 0.1
• SGD + Momentum: 0.01 - 0.1
• Adam: 0.001 (1e-3)
• AdamW: 0.0001 - 0.001 (1e-4 to 1e-3)
• RMSprop: 0.001

Always start with these defaults and adjust based on loss curves!
""")

# ============================================================================
# SECTION 10: Common Hyperparameters
# ============================================================================
print("\n" + "-" * 80)
print("HYPERPARAMETER REFERENCE")
print("-" * 80)

print("""
LEARNING RATE (lr):
  • Most important hyperparameter
  • Too high: Training unstable, loss explodes
  • Too low: Training too slow
  • Use learning rate schedulers to adjust during training

MOMENTUM (SGD):
  • Typical: 0.9 or 0.95
  • Higher = more smoothing but can overshoot

BETAS (Adam/AdamW):
  • β₁ (momentum): typically 0.9
  • β₂ (RMS): typically 0.999
  • Rarely need to change these

WEIGHT DECAY:
  • Regularization strength
  • Typical: 0.0001 to 0.01
  • Higher = more regularization
  • Use AdamW for correct implementation

EPSILON (Adam/RMSprop):
  • Numerical stability constant
  • Default: 1e-8
  • Rarely need to change
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. Different optimizers make different trade-offs:
   • Speed vs stability
   • Ease of use vs final performance
   • Memory efficiency

2. Adam/AdamW are great defaults:
   • Fast convergence
   • Work well out-of-the-box
   • AdamW preferred for modern architectures

3. SGD + Momentum often achieves best final performance:
   • Requires more tuning
   • Better generalization
   • Preferred for production models

4. Always monitor loss curves:
   • Smooth decrease = good
   • Oscillations = learning rate too high
   • Plateau = learning rate too low or converged

5. No single "best" optimizer:
   • Depends on problem, architecture, data
   • Experiment and compare
   • Use learning rate schedulers for better results

NEXT STEPS:
→ Experiment with different learning rates
→ Learn about learning rate schedulers
→ Try combining optimizers with regularization
→ Study advanced optimizers (RAdam, Lookahead, etc.)
""")
print("=" * 80)


if __name__ == "__main__":
    pass
