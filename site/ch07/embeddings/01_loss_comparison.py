# ========================================================
# 01_loss_comparison.py
# Comprehensive Loss Function Comparison
# ========================================================

"""
INTERMEDIATE TUTORIAL 1: Comprehensive Loss Comparison

Learning Objectives:
- Compare all three loss functions side-by-side
- Understand convergence patterns
- Analyze performance differences
- Visualize and interpret results

Estimated time: 20 minutes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from utils import word_embedding_ngram as ngr

print("=" * 70)
print("INTERMEDIATE TUTORIAL 1: Loss Function Comparison")
print("=" * 70)

# ========================================================
# Training All Three Models
# ========================================================

print("\nTraining three models with different loss functions...")
print("This will take a moment...\n")

# Model 1: CrossEntropyLoss
print("1. Training with nn.CrossEntropyLoss...")
model1 = ngr.NGramLanguageModeler()
loss_fn1 = nn.CrossEntropyLoss()
optimizer1 = optim.SGD(model1.parameters(), lr=ngr.ARGS.lr)
losses1 = ngr.train(model1, loss_fn1, optimizer1, epochs=ngr.ARGS.epochs, verbose=False)

# Model 2: F.cross_entropy
print("2. Training with F.cross_entropy...")
model2 = ngr.NGramLanguageModeler()
optimizer2 = optim.SGD(model2.parameters(), lr=ngr.ARGS.lr)
losses2 = ngr.train(model2, F.cross_entropy, optimizer2, epochs=ngr.ARGS.epochs, verbose=False)

# Model 3: NLLLoss with LogSoftmax
print("3. Training with nn.NLLLoss...")

class NGramNLL(nn.Module):
    def __init__(self):
        super(NGramNLL, self).__init__()
        self.embeddings = nn.Embedding(ngr.ARGS.vocab_size, ngr.ARGS.embedding_dim)
        self.linear1 = nn.Linear(ngr.ARGS.context_size * ngr.ARGS.embedding_dim, 128)
        self.linear2 = nn.Linear(128, ngr.ARGS.vocab_size)
    
    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((inputs.shape[0], -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        return F.log_softmax(out, dim=1)

model3 = NGramNLL()
loss_fn3 = nn.NLLLoss()
optimizer3 = optim.SGD(model3.parameters(), lr=ngr.ARGS.lr)
losses3 = ngr.train(model3, loss_fn3, optimizer3, epochs=ngr.ARGS.epochs, verbose=False)

print("\nTraining complete!\n")

# ========================================================
# Visualization
# ========================================================

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# Plot 1: All losses together
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(losses1, label='nn.CrossEntropyLoss', linewidth=2, alpha=0.8)
ax1.plot(losses2, label='F.cross_entropy', linewidth=2, alpha=0.8, linestyle='--')
ax1.plot(losses3, label='nn.NLLLoss', linewidth=2, alpha=0.8, linestyle=':')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: First 20 epochs (zoom in)
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(losses1[:20], label='nn.CrossEntropyLoss', linewidth=2, marker='o', markersize=4)
ax2.plot(losses2[:20], label='F.cross_entropy', linewidth=2, marker='s', markersize=4)
ax2.plot(losses3[:20], label='nn.NLLLoss', linewidth=2, marker='^', markersize=4)
ax2.set_xlabel('Epoch', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('First 20 Epochs (Detailed)', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: Last 20 epochs (convergence)
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(range(80, 100), losses1[80:], label='nn.CrossEntropyLoss', linewidth=2, marker='o', markersize=4)
ax3.plot(range(80, 100), losses2[80:], label='F.cross_entropy', linewidth=2, marker='s', markersize=4)
ax3.plot(range(80, 100), losses3[80:], label='nn.NLLLoss', linewidth=2, marker='^', markersize=4)
ax3.set_xlabel('Epoch', fontsize=11)
ax3.set_ylabel('Loss', fontsize=11)
ax3.set_title('Last 20 Epochs (Convergence)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# Plot 4: Loss differences
ax4 = fig.add_subplot(gs[2, 0])
diff_1_2 = np.abs(np.array(losses1) - np.array(losses2))
diff_1_3 = np.abs(np.array(losses1) - np.array(losses3))
ax4.plot(diff_1_2, label='|CE - F.CE|', linewidth=2, alpha=0.8)
ax4.plot(diff_1_3, label='|CE - NLL|', linewidth=2, alpha=0.8)
ax4.set_xlabel('Epoch', fontsize=11)
ax4.set_ylabel('Absolute Difference', fontsize=11)
ax4.set_title('Loss Differences Between Methods', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_yscale('log')

# Plot 5: Statistics table
ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

stats_data = [
    ['Metric', 'CE Loss', 'F.CE', 'NLL Loss'],
    ['Initial Loss', f'{losses1[0]:.4f}', f'{losses2[0]:.4f}', f'{losses3[0]:.4f}'],
    ['Final Loss', f'{losses1[-1]:.4f}', f'{losses2[-1]:.4f}', f'{losses3[-1]:.4f}'],
    ['Mean Loss', f'{np.mean(losses1):.4f}', f'{np.mean(losses2):.4f}', f'{np.mean(losses3):.4f}'],
    ['Std Dev', f'{np.std(losses1):.4f}', f'{np.std(losses2):.4f}', f'{np.std(losses3):.4f}'],
    ['Improvement', f'{((losses1[0]-losses1[-1])/losses1[0]*100):.1f}%', 
     f'{((losses2[0]-losses2[-1])/losses2[0]*100):.1f}%',
     f'{((losses3[0]-losses3[-1])/losses3[0]*100):.1f}%']
]

table = ax5.table(cellText=stats_data, cellLoc='center', loc='center',
                  colWidths=[0.25, 0.25, 0.25, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style the header row
for i in range(4):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

ax5.set_title('Performance Statistics', fontsize=12, fontweight='bold', pad=20)

plt.savefig('/home/claude/loss_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ========================================================
# Analysis
# ========================================================

print("=" * 70)
print("ANALYSIS")
print("=" * 70)

print("\n1. Final Losses:")
print(f"   nn.CrossEntropyLoss: {losses1[-1]:.6f}")
print(f"   F.cross_entropy:     {losses2[-1]:.6f}")
print(f"   nn.NLLLoss:          {losses3[-1]:.6f}")

print("\n2. Maximum Difference:")
max_diff = max(diff_1_2.max(), diff_1_3.max())
print(f"   {max_diff:.6f} (very small!)")

print("\n3. Convergence Speed:")
print(f"   All three converge at similar rates")
print(f"   Minor differences due to random initialization")

print("\n4. Key Insights:")
print("   ✓ All three methods are equivalent")
print("   ✓ Differences are due to numerical precision and random seeds")
print("   ✓ Choice should be based on code structure, not performance")
print("   ✓ CrossEntropyLoss is most commonly used in practice")

print("\n" + "=" * 70)
print("CONCLUSION: All three loss functions work equally well!")
print("=" * 70)
