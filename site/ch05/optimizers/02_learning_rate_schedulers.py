"""
================================================================================
INTERMEDIATE 02: Learning Rate Schedulers
================================================================================

WHAT YOU'LL LEARN:
- Why learning rate scheduling is important
- Different scheduler types (Step, Exponential, Cosine, ReduceLROnPlateau)
- When and how to use each scheduler
- Combining schedulers with optimizers

PREREQUISITES:
- Complete beginner tutorials
- Understand optimizer basics

TIME TO COMPLETE: ~20 minutes
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 80)
print("LEARNING RATE SCHEDULERS")
print("=" * 80)

# ============================================================================
# SECTION 1: Why Use Learning Rate Scheduling?
# ============================================================================
print("\n" + "-" * 80)
print("WHY LEARNING RATE SCHEDULING?")
print("-" * 80)

print("""
The Problem with Fixed Learning Rates:
  
  BEGINNING OF TRAINING:
  • Need large learning rate to make fast progress
  • Escape saddle points
  • Explore the loss landscape
  
  END OF TRAINING:
  • Large learning rate causes oscillation around optimum
  • Can't fine-tune the solution
  • May not converge to best solution
  
  THE SOLUTION:
  Start with high learning rate → Gradually decrease → Fine-tune at end
  
  This is called "Learning Rate Annealing" or "Learning Rate Decay"
""")

# ============================================================================
# SECTION 2: Common Scheduler Types
# ============================================================================
print("\n" + "-" * 80)
print("COMMON SCHEDULER TYPES")
print("-" * 80)

# Create a dummy optimizer for demonstration
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("""
1. STEP LR:
   • Decreases LR by factor every N epochs
   • Simple and predictable
   • Example: lr × 0.1 every 30 epochs

2. EXPONENTIAL LR:
   • Decreases LR by constant factor each epoch
   • Smooth exponential decay
   • Example: lr × 0.95 each epoch

3. COSINE ANNEALING:
   • Follows cosine curve
   • Smooth decrease with periodic restarts option
   • Popular for deep networks

4. REDUCE LR ON PLATEAU:
   • Decreases LR when metric stops improving
   • Adaptive to training progress
   • Example: Reduce by 0.1 if validation loss doesn't improve for 5 epochs
""")

# ============================================================================
# SECTION 3: StepLR - Decrease at Fixed Intervals
# ============================================================================
print("\n" + "-" * 80)
print("1. STEP LR SCHEDULER")
print("-" * 80)

optimizer_step = optim.SGD(model.parameters(), lr=0.1)
# Multiply LR by gamma every step_size epochs
scheduler_step = StepLR(optimizer_step, step_size=10, gamma=0.5)

print(f"Initial LR: {optimizer_step.param_groups[0]['lr']:.6f}")
print("\nLearning rate over 50 epochs:")

lrs_step = []
for epoch in range(50):
    lrs_step.append(optimizer_step.param_groups[0]['lr'])
    
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}: LR = {optimizer_step.param_groups[0]['lr']:.6f}")
    
    # Simulate training step
    optimizer_step.zero_grad()
    loss = torch.tensor(1.0, requires_grad=True)
    loss.backward()
    optimizer_step.step()
    
    # Update learning rate
    scheduler_step.step()

print("\nEXPLANATION:")
print("  step_size=10: LR changes every 10 epochs")
print("  gamma=0.5: LR is multiplied by 0.5 at each step")
print("  Result: 0.1 → 0.05 → 0.025 → 0.0125 → ...")

# ============================================================================
# SECTION 4: ExponentialLR - Smooth Decay
# ============================================================================
print("\n" + "-" * 80)
print("2. EXPONENTIAL LR SCHEDULER")
print("-" * 80)

optimizer_exp = optim.SGD(model.parameters(), lr=0.1)
# Multiply LR by gamma every epoch
scheduler_exp = ExponentialLR(optimizer_exp, gamma=0.95)

print(f"Initial LR: {optimizer_exp.param_groups[0]['lr']:.6f}")
print("\nLearning rate over 50 epochs:")

lrs_exp = []
for epoch in range(50):
    lrs_exp.append(optimizer_exp.param_groups[0]['lr'])
    
    if epoch % 10 == 0:
        print(f"  Epoch {epoch+1}: LR = {optimizer_exp.param_groups[0]['lr']:.6f}")
    
    # Simulate training
    optimizer_exp.zero_grad()
    loss = torch.tensor(1.0, requires_grad=True)
    loss.backward()
    optimizer_exp.step()
    scheduler_exp.step()

print("\nEXPLANATION:")
print("  gamma=0.95: LR is multiplied by 0.95 every epoch")
print("  Smooth exponential decay: lr(t) = lr(0) × gamma^t")
print("  More gradual than StepLR")

# ============================================================================
# SECTION 5: CosineAnnealingLR - Smooth Cosine Curve
# ============================================================================
print("\n" + "-" * 80)
print("3. COSINE ANNEALING LR SCHEDULER")
print("-" * 80)

optimizer_cos = optim.SGD(model.parameters(), lr=0.1)
# Decrease LR following a cosine curve
scheduler_cos = CosineAnnealingLR(optimizer_cos, T_max=50, eta_min=0.001)

print(f"Initial LR: {optimizer_cos.param_groups[0]['lr']:.6f}")
print("\nLearning rate over 50 epochs:")

lrs_cos = []
for epoch in range(50):
    lrs_cos.append(optimizer_cos.param_groups[0]['lr'])
    
    if epoch % 10 == 0:
        print(f"  Epoch {epoch+1}: LR = {optimizer_cos.param_groups[0]['lr']:.6f}")
    
    # Simulate training
    optimizer_cos.zero_grad()
    loss = torch.tensor(1.0, requires_grad=True)
    loss.backward()
    optimizer_cos.step()
    scheduler_cos.step()

print("\nEXPLANATION:")
print("  T_max=50: Complete cosine cycle over 50 epochs")
print("  eta_min=0.001: Minimum learning rate")
print("  Smooth decrease with faster drop at beginning")
print("  Very popular for training vision models")

# ============================================================================
# SECTION 6: ReduceLROnPlateau - Adaptive Scheduling
# ============================================================================
print("\n" + "-" * 80)
print("4. REDUCE LR ON PLATEAU SCHEDULER")
print("-" * 80)

optimizer_plateau = optim.SGD(model.parameters(), lr=0.1)
# Reduce LR when metric plateaus
scheduler_plateau = ReduceLROnPlateau(
    optimizer_plateau, 
    mode='min',           # Minimize the metric
    factor=0.5,           # Multiply LR by 0.5
    patience=5,           # Wait 5 epochs before reducing
    verbose=True,         # Print when LR changes
    min_lr=0.001          # Don't go below this
)

print(f"Initial LR: {optimizer_plateau.param_groups[0]['lr']:.6f}")
print("\nSimulating training with plateaus:")

# Simulate validation losses
# Loss decreases, then plateaus, then decreases again
simulated_losses = (
    [2.0, 1.8, 1.6, 1.4, 1.2] +  # Improving
    [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2] +  # Plateau (7 epochs)
    [1.0, 0.9, 0.8] +  # Improving again
    [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]  # Plateau again
)

lrs_plateau = []
for epoch, val_loss in enumerate(simulated_losses):
    current_lr = optimizer_plateau.param_groups[0]['lr']
    lrs_plateau.append(current_lr)
    
    print(f"  Epoch {epoch+1}: Val Loss = {val_loss:.2f}, LR = {current_lr:.6f}")
    
    # The scheduler needs the validation metric
    scheduler_plateau.step(val_loss)

print("\nEXPLANATION:")
print("  Monitors validation loss (or any metric)")
print("  Reduces LR when no improvement for 'patience' epochs")
print("  More adaptive than time-based schedulers")
print("  Good when you don't know optimal schedule in advance")

# ============================================================================
# SECTION 7: Visualize All Schedulers
# ============================================================================
print("\n" + "-" * 80)
print("VISUALIZATION")
print("-" * 80)

plt.figure(figsize=(14, 5))

# Plot 1: Time-based schedulers comparison
plt.subplot(1, 2, 1)
plt.plot(range(len(lrs_step)), lrs_step, 'b-', label='StepLR', linewidth=2)
plt.plot(range(len(lrs_exp)), lrs_exp, 'r-', label='ExponentialLR', linewidth=2)
plt.plot(range(len(lrs_cos)), lrs_cos, 'g-', label='CosineAnnealingLR', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Time-Based Schedulers', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')

# Plot 2: ReduceLROnPlateau
plt.subplot(1, 2, 2)
plt.plot(range(len(lrs_plateau)), lrs_plateau, 'purple', linewidth=2, label='ReduceLROnPlateau')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Adaptive Scheduler (ReduceLROnPlateau)', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.tight_layout()
plot_path = '/home/claude/scheduler_comparison.png'
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {plot_path}")

# ============================================================================
# SECTION 8: Practical Usage in Training Loop
# ============================================================================
print("\n" + "-" * 80)
print("PRACTICAL USAGE IN TRAINING LOOP")
print("-" * 80)

print("""
BASIC TRAINING LOOP WITH SCHEDULER:

# Setup
model = MyModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    # TRAINING PHASE
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    # VALIDATION PHASE (optional but recommended)
    model.eval()
    val_loss = validate(model, val_loader, criterion)
    
    # UPDATE LEARNING RATE
    scheduler.step()  # For time-based schedulers
    # OR
    scheduler.step(val_loss)  # For ReduceLROnPlateau
    
    # LOG CURRENT LR
    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch}, LR: {current_lr:.6f}')
""")

# ============================================================================
# SECTION 9: Choosing the Right Scheduler
# ============================================================================
print("\n" + "-" * 80)
print("DECISION GUIDE: Which Scheduler to Use?")
print("-" * 80)

print("""
📅 USE STEP LR WHEN:
   ✓ You want simple, predictable scheduling
   ✓ Training for fixed number of epochs
   ✓ Common in traditional vision tasks
   ✓ Example: StepLR(step_size=30, gamma=0.1) for 100 epochs
   
📉 USE EXPONENTIAL LR WHEN:
   ✓ Want smooth, continuous decay
   ✓ Training for many epochs
   ✓ Example: ExponentialLR(gamma=0.95)
   
🌊 USE COSINE ANNEALING WHEN:
   ✓ Training modern deep networks
   ✓ Want smooth, gradual decrease
   ✓ Popular for ImageNet, transformers
   ✓ Can use CosineAnnealingWarmRestarts for periodic restarts
   ✓ Example: CosineAnnealingLR(T_max=epochs, eta_min=1e-6)
   
🎯 USE REDUCE LR ON PLATEAU WHEN:
   ✓ Don't know optimal schedule in advance
   ✓ Want adaptive behavior
   ✓ Have validation metric to monitor
   ✓ Training can be variable length
   ✓ Example: ReduceLROnPlateau(patience=10, factor=0.5)

💡 PRO TIP:
   For best results, combine schedulers with warm-up!
   Start with low LR, increase to base LR, then decay
""")

# ============================================================================
# SECTION 10: Advanced: Warmup + Cosine Annealing
# ============================================================================
print("\n" + "-" * 80)
print("ADVANCED: Learning Rate Warmup")
print("-" * 80)

print("""
WHAT IS WARMUP?
  • Start with very low learning rate
  • Gradually increase to base learning rate
  • Then apply regular scheduling
  
WHY USE WARMUP?
  • Prevents unstable training at start
  • Important for large batch sizes
  • Critical for transformers
  • Helps with batch normalization

EXAMPLE: Warmup + Cosine Annealing
  Epochs 1-10: Linear increase from 1e-6 to 1e-3
  Epochs 11-100: Cosine annealing from 1e-3 to 1e-6
  
PyTorch doesn't have built-in warmup, but you can:
  1. Use a custom scheduler
  2. Manually adjust LR for first N epochs
  3. Use transformers library's get_linear_schedule_with_warmup
""")

# Simple warmup implementation
def get_lr_with_warmup(epoch, warmup_epochs, base_lr, min_lr, total_epochs):
    """Calculate LR with warmup followed by cosine annealing"""
    if epoch < warmup_epochs:
        # Linear warmup
        return min_lr + (base_lr - min_lr) * epoch / warmup_epochs
    else:
        # Cosine annealing
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (base_lr - min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())

# Demonstrate warmup
warmup_epochs = 10
total_epochs = 100
base_lr = 0.1
min_lr = 0.001

lrs_warmup = [get_lr_with_warmup(e, warmup_epochs, base_lr, min_lr, total_epochs) 
              for e in range(total_epochs)]

print(f"\nLR with warmup (first 20 epochs):")
for epoch in range(0, 20, 2):
    print(f"  Epoch {epoch+1}: {lrs_warmup[epoch]:.6f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. Learning rate scheduling improves training:
   • Faster convergence
   • Better final performance
   • More stable training

2. Different schedulers for different needs:
   • StepLR: Simple, predictable drops
   • ExponentialLR: Smooth exponential decay
   • CosineAnnealingLR: Popular for modern networks
   • ReduceLROnPlateau: Adaptive based on metrics

3. General strategy:
   • Start with high LR (explore)
   • Gradually decrease (fine-tune)
   • Optionally use warmup at start

4. Implementation is simple:
   • Create scheduler after optimizer
   • Call scheduler.step() after each epoch
   • ReduceLROnPlateau needs metric: scheduler.step(val_loss)

5. Monitor learning rate during training:
   • Log it to see if schedule is working
   • Adjust if training is unstable or too slow

NEXT STEPS:
→ Try different schedulers on your problem
→ Experiment with warmup strategies
→ Learn about cyclical learning rates
→ Combine with early stopping
""")
print("=" * 80)
