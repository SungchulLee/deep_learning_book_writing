"""
10_learning_rate_scheduling.py - Dynamic Learning Rate

Learn to adjust learning rate during training for better convergence.
Start with high LR for fast initial progress, then decrease for fine-tuning.

TIME: 30-40 minutes | DIFFICULTY: ⭐⭐⭐⭐☆
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

print("="*70)
print("Learning Rate Scheduling")
print("="*70)

model = nn.Sequential(
    nn.Linear(100, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

optimizer = optim.SGD(model.parameters(), lr=0.1)

print("LEARNING RATE SCHEDULERS IN PyTorch:")
print("-"*70)

# 1. StepLR: Decay LR every N epochs
scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
print("StepLR: Reduces LR by 0.1x every 30 epochs")

# 2. ExponentialLR: Exponential decay
scheduler2 = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
print("ExponentialLR: Multiply LR by 0.9 each epoch")

# 3. CosineAnnealingLR: Cosine decay
scheduler3 = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
print("CosineAnnealingLR: Cosine curve over 100 epochs")

# 4. ReduceLROnPlateau: Reduce when metric plateaus
scheduler4 = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
print("ReduceLROnPlateau: Reduce when val loss doesn't improve for 5 epochs")

# Visualize different schedules
epochs = 100
lrs_step = []
lrs_exp = []
lrs_cos = []

# Simulate StepLR
lr = 0.1
for epoch in range(epochs):
    lrs_step.append(lr)
    if (epoch + 1) % 30 == 0:
        lr *= 0.1

# Simulate ExponentialLR
lr = 0.1
for epoch in range(epochs):
    lrs_exp.append(lr)
    lr *= 0.9

# Simulate CosineAnnealingLR  
import math
for epoch in range(epochs):
    lr = 0.1 * (1 + math.cos(math.pi * epoch / 100)) / 2
    lrs_cos.append(lr)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(lrs_step, label='StepLR', linewidth=2)
plt.plot(lrs_exp, label='ExponentialLR', linewidth=2)
plt.plot(lrs_cos, label='CosineAnnealingLR', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedules Comparison', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.savefig('10_lr_schedules.png', dpi=150)
print("\nSchedules visualization saved!")

print("\n" + "="*70)
print("USAGE PATTERN")
print("="*70)
print("""
# Setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        # Training step
        loss.backward()
        optimizer.step()
    
    # Update learning rate
    scheduler.step()
    
    # Check current LR
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}, LR: {current_lr}")

RECOMMENDATIONS:
- Start with constant LR to establish baseline
- Use StepLR for simple decay
- Use ReduceLROnPlateau for adaptive adjustment
- Use CosineAnnealing for cyclical training
""")
plt.show()
