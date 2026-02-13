"""
14_multi_output_network.py - Multi-Task Learning

One network, multiple tasks! Learn to predict multiple outputs
simultaneously from the same input.

EXAMPLE: From a person's photo, predict:
- Age (regression)
- Gender (binary classification)
- Emotion (multi-class classification)

This is common in real applications!

TIME: 35-45 minutes | DIFFICULTY: ⭐⭐⭐⭐☆
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

print("="*80)
print("Multi-Task Learning Example")
print("="*80)

# Synthetic data: predict age AND binary class from features
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data
n_samples = 5000
n_features = 20

X = torch.randn(n_samples, n_features)

# Task 1: Regression (predict age, 18-80)
age = 50 + 15 * X[:, 0] + torch.randn(n_samples) * 5
age = torch.clamp(age, 18, 80).reshape(-1, 1)

# Task 2: Binary classification (predict gender)
gender_logits = 2 * X[:, 1] - X[:, 2] + torch.randn(n_samples) * 0.5
gender = (gender_logits > 0).float().reshape(-1, 1)

# Split data
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
age_train, age_test = age[:split], age[split:]
gender_train, gender_test = gender[:split], gender[split:]

print(f"Features: {n_features}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print(f"\nTasks:")
print(f"  Task 1: Age prediction (regression, range 18-80)")
print(f"  Task 2: Gender classification (binary, 0 or 1)")

print("\n" + "="*80)
print("Multi-Task Architecture")
print("="*80)

class MultiTaskNet(nn.Module):
    """
    Multi-task learning network with shared and task-specific layers.
    
    Architecture:
        Input → Shared layers → Split into task-specific heads
    """
    
    def __init__(self, input_size):
        super().__init__()
        
        # Shared layers (learn general features)
        self.shared = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Task 1 head: Regression (age prediction)
        self.age_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single output for regression
        )
        
        # Task 2 head: Binary classification (gender)
        self.gender_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single output, use BCEWithLogitsLoss
        )
    
    def forward(self, x):
        # Shared feature extraction
        shared_features = self.shared(x)
        
        # Task-specific predictions
        age_pred = self.age_head(shared_features)
        gender_logits = self.gender_head(shared_features)
        
        return age_pred, gender_logits

model = MultiTaskNet(n_features)
print("Multi-task model created!")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "="*80)
print("Training Setup")
print("="*80)

# Separate losses for each task
criterion_age = nn.MSELoss()  # Regression
criterion_gender = nn.BCEWithLogitsLoss()  # Binary classification

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Task weights (balance importance)
weight_age = 1.0
weight_gender = 1.0

print(f"Loss functions:")
print(f"  Age: MSELoss (weight={weight_age})")
print(f"  Gender: BCEWithLogitsLoss (weight={weight_gender})")

print("\n" + "="*80)
print("Training Multi-Task Model")
print("="*80)

epochs = 100
batch_size = 64
losses_total = []
losses_age = []
losses_gender = []

for epoch in range(epochs):
    model.train()
    epoch_loss_total = 0
    epoch_loss_age = 0
    epoch_loss_gender = 0
    
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_age = age_train[i:i+batch_size]
        batch_gender = gender_train[i:i+batch_size]
        
        # Forward pass
        age_pred, gender_logits = model(batch_X)
        
        # Compute losses for each task
        loss_age = criterion_age(age_pred, batch_age)
        loss_gender = criterion_gender(gender_logits, batch_gender)
        
        # Combined loss (weighted sum)
        loss = weight_age * loss_age + weight_gender * loss_gender
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss_total += loss.item()
        epoch_loss_age += loss_age.item()
        epoch_loss_gender += loss_gender.item()
    
    # Record losses
    num_batches = len(X_train) // batch_size
    losses_total.append(epoch_loss_total / num_batches)
    losses_age.append(epoch_loss_age / num_batches)
    losses_gender.append(epoch_loss_gender / num_batches)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] | "
              f"Total: {losses_total[-1]:.4f} | "
              f"Age: {losses_age[-1]:.4f} | "
              f"Gender: {losses_gender[-1]:.4f}")

print("\n" + "="*80)
print("Evaluation")
print("="*80)

model.eval()
with torch.no_grad():
    age_pred, gender_logits = model(X_test)
    gender_pred = (torch.sigmoid(gender_logits) > 0.5).float()
    
    # Age metrics (regression)
    age_mse = criterion_age(age_pred, age_test).item()
    age_mae = torch.abs(age_pred - age_test).mean().item()
    
    # Gender metrics (classification)
    gender_acc = (gender_pred == gender_test).float().mean().item() * 100

print(f"Age Prediction (Regression):")
print(f"  MSE: {age_mse:.4f}")
print(f"  MAE: {age_mae:.4f} years")

print(f"\nGender Classification (Binary):")
print(f"  Accuracy: {gender_acc:.2f}%")

# Visualizations
fig = plt.figure(figsize=(16, 10))

# Loss curves
ax1 = plt.subplot(2, 3, 1)
ax1.plot(losses_total, label='Total Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Total Loss', fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = plt.subplot(2, 3, 2)
ax2.plot(losses_age, label='Age Loss', color='blue', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (MSE)')
ax2.set_title('Age Prediction Loss', fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

ax3 = plt.subplot(2, 3, 3)
ax3.plot(losses_gender, label='Gender Loss', color='red', linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss (BCE)')
ax3.set_title('Gender Classification Loss', fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend()

# Age predictions
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(age_test.numpy(), age_pred.numpy(), alpha=0.5)
ax4.plot([18, 80], [18, 80], 'r--', lw=2)
ax4.set_xlabel('Actual Age')
ax4.set_ylabel('Predicted Age')
ax4.set_title('Age Predictions', fontweight='bold')
ax4.grid(True, alpha=0.3)

# Age residuals
ax5 = plt.subplot(2, 3, 5)
residuals = (age_pred - age_test).numpy()
ax5.hist(residuals, bins=30, edgecolor='black')
ax5.set_xlabel('Prediction Error (years)')
ax5.set_ylabel('Frequency')
ax5.set_title('Age Prediction Errors', fontweight='bold')
ax5.grid(True, alpha=0.3)

# Gender confusion
ax6 = plt.subplot(2, 3, 6)
correct = (gender_pred == gender_test).sum().item()
wrong = len(gender_test) - correct
ax6.bar(['Correct', 'Wrong'], [correct, wrong], color=['green', 'red'], alpha=0.7)
ax6.set_ylabel('Count')
ax6.set_title(f'Gender Classification\nAccuracy: {gender_acc:.1f}%', fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('14_multi_task_results.png', dpi=150)
print("\nVisualization saved!")

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
MULTI-TASK LEARNING:
✓ Share feature extraction across tasks
✓ Separate task-specific heads for predictions
✓ Combine losses with weighted sum
✓ Can improve generalization (regularization effect)

ARCHITECTURE:
  Input → Shared Layers → Split → Task Heads → Outputs

ADVANTAGES:
+ Shared representations (transfer learning effect)
+ Better sample efficiency
+ Implicit regularization
+ Single model deployment

CHALLENGES:
- Balancing task weights
- Tasks may conflict
- More complex training

APPLICATIONS:
- Multi-label classification
- Joint prediction tasks
- Auxiliary task learning
- Cross-domain transfer

TIPS:
1. Start with equal task weights, adjust if needed
2. Monitor each task's performance separately
3. Use task-specific learning rates if needed
4. Consider task uncertainty weighting
""")
plt.show()
