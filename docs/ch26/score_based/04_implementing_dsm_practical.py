"""
MODULE 04: Practical Denoising Score Matching
============================================

DIFFICULTY: Intermediate
TIME: 3-4 hours
PREREQUISITES: Modules 01-03

LEARNING OBJECTIVES:
- Implement production-quality DSM training
- Handle various noise schedules
- Debug common training issues
- Evaluate score network quality

Key formula:
L_DSM(θ) = 𝔼_x 𝔼_ε ||s_θ(x + σε) + ε/σ||²

Author: Sungchul @ Yonsei University
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

print("MODULE 04: Practical DSM Implementation")
print("="*80)

class MLPScore(nn.Module):
    """MLP score network with skip connections"""
    def __init__(self, input_dim=2, hidden_dims=[128, 128]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.SiLU()
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def train_dsm(data, sigma=0.5, epochs=2000, batch_size=256, lr=1e-3):
    """Complete DSM training loop with best practices"""
    model = MLPScore(input_dim=data.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(data))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for (batch,) in loader:
            # Add noise
            noise = torch.randn_like(batch)
            noisy = batch + sigma * noise
            
            # Predict score
            pred_score = model(noisy)
            target_score = -noise / sigma
            
            # Loss
            loss = torch.mean((pred_score - target_score) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        losses.append(epoch_loss / len(loader))
        
        if epoch % 400 == 0:
            print(f"Epoch {epoch}: Loss = {losses[-1]:.6f}, LR = {scheduler.get_last_lr()[0]:.2e}")
    
    return model, losses

# Generate toy data
from sklearn.datasets import make_moons
data, _ = make_moons(n_samples=2000, noise=0.05)

print("\nTraining DSM on moons dataset...")
model, losses = train_dsm(data, sigma=0.1, epochs=2000)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(losses)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

# Score field
x = np.linspace(-2, 3, 20)
y = np.linspace(-1.5, 2, 20)
X, Y = np.meshgrid(x, y)
pos = torch.FloatTensor(np.stack([X.ravel(), Y.ravel()], axis=1))
with torch.no_grad():
    scores = model(pos).numpy().reshape(X.shape + (2,))

ax2.scatter(data[:, 0], data[:, 1], s=1, alpha=0.3, c='blue')
ax2.quiver(X, Y, scores[:,:,0], scores[:,:,1], alpha=0.6, color='red', scale=50)
ax2.set_title('Learned Score Field')
ax2.set_aspect('equal')
plt.tight_layout()
plt.savefig('04_dsm_practical.png', dpi=150)
plt.close()
print("✓ Saved: 04_dsm_practical.png")

print("""
KEY IMPLEMENTATION DETAILS:
1. Layer normalization for stable training
2. AdamW with weight decay for regularization
3. Gradient clipping to prevent explosions
4. Cosine annealing learning rate schedule
5. Batch processing for efficiency

DEBUGGING CHECKLIST:
□ Loss decreasing steadily?
□ Scores point toward data?
□ Gradient norms reasonable (<10)?
□ Learning rate appropriate?
□ Noise level σ matches data scale?
""")

print("\n✓ Module 04 complete!")
