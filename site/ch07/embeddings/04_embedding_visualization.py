# ========================================================
# 04_embedding_visualization.py
# Advanced Embedding Visualization with t-SNE
# ========================================================

"""
ADVANCED TUTORIAL: Embedding Visualization

Learning Objectives:
- Visualize high-dimensional embeddings in 2D
- Use PCA and t-SNE for dimensionality reduction
- Analyze embedding clusters
- Understand semantic relationships visually

Estimated time: 35 minutes

Prerequisites:
- Complete intermediate tutorials
- Understand what embeddings represent
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from utils.data_loader import (
    load_text_from_file,
    simple_tokenize,
    build_vocabulary,
    create_cbow_dataset
)

print("=" * 70)
print("ADVANCED TUTORIAL: Embedding Visualization")
print("=" * 70)

# ========================================================
# PART 1: Train a CBOW Model
# ========================================================

print("\n" + "=" * 70)
print("PART 1: Training CBOW Model for Visualization")
print("=" * 70)

# Load and prepare data
text = load_text_from_file('../data/sample_text.txt')
tokens = simple_tokenize(text, lowercase=True)
word_to_ix, ix_to_word = build_vocabulary(tokens, min_freq=1)
vocab_size = len(word_to_ix)

print(f"Vocabulary size: {vocab_size}")

# Create CBOW dataset
WINDOW_SIZE = 2
cbow_data = create_cbow_dataset(tokens, WINDOW_SIZE, word_to_ix)


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context):
        embeds = torch.mean(self.embeddings(context), dim=1)
        return self.linear(embeds)


# Train model
EMBEDDING_DIM = 30  # Higher dimension for better embeddings
model = CBOWModel(vocab_size, EMBEDDING_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()

print(f"\nTraining CBOW model with {EMBEDDING_DIM}D embeddings...")

EPOCHS = 150
BATCH_SIZE = 32

for epoch in range(EPOCHS):
    total_loss = 0
    for i in range(0, len(cbow_data), BATCH_SIZE):
        batch = cbow_data[i:i+BATCH_SIZE]
        if len(batch) == 0:
            continue
        
        contexts = torch.stack([item[0] for item in batch])
        targets = torch.cat([item[1] for item in batch])
        
        optimizer.zero_grad()
        outputs = model(contexts)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 30 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(cbow_data)*BATCH_SIZE:.4f}")

print("Training complete!\n")

# ========================================================
# PART 2: Extract Embeddings
# ========================================================

print("=" * 70)
print("PART 2: Extracting Embeddings")
print("=" * 70)

# Get all word embeddings
embeddings = model.embeddings.weight.detach().cpu().numpy()
words = list(word_to_ix.keys())

print(f"\nEmbedding matrix shape: {embeddings.shape}")
print(f"  {len(words)} words × {EMBEDDING_DIM} dimensions")

# ========================================================
# PART 3: PCA Visualization
# ========================================================

print("\n" + "=" * 70)
print("PART 3: PCA (Principal Component Analysis)")
print("=" * 70)

print("\nPCA finds the directions of maximum variance...")
print("Projects high-dimensional data to 2D")

# Apply PCA
pca = PCA(n_components=2, random_state=42)
embeddings_pca = pca.fit_transform(embeddings)

print(f"\nExplained variance ratio:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"  Total: {pca.explained_variance_ratio_.sum():.4f}")

# Plot PCA
fig, ax = plt.subplots(figsize=(14, 10))

# Plot all points
ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
          alpha=0.5, s=30, c='lightblue', edgecolors='black', linewidth=0.5)

# Annotate all words
for i, word in enumerate(words):
    ax.annotate(word, 
               (embeddings_pca[i, 0], embeddings_pca[i, 1]),
               fontsize=8, alpha=0.8,
               xytext=(2, 2), textcoords='offset points')

ax.set_xlabel('First Principal Component', fontsize=12)
ax.set_ylabel('Second Principal Component', fontsize=12)
ax.set_title(f'Word Embeddings Visualization (PCA)\n{vocab_size} words from Shakespeare/Poetry corpus', 
            fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========================================================
# PART 4: t-SNE Visualization
# ========================================================

print("\n" + "=" * 70)
print("PART 4: t-SNE (t-Distributed Stochastic Neighbor Embedding)")
print("=" * 70)

print("\nt-SNE is better at preserving local structure...")
print("Groups similar words together more clearly")
print("(This may take a moment...)\n")

# Apply t-SNE
perplexity = min(30, len(words) - 1)
tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, 
           n_iter=1000, learning_rate=200)
embeddings_tsne = tsne.fit_transform(embeddings)

print("t-SNE complete!\n")

# Plot t-SNE
fig, ax = plt.subplots(figsize=(14, 10))

# Color by word length (just for visualization)
word_lengths = [len(w) for w in words]
scatter = ax.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
                    c=word_lengths, cmap='viridis', 
                    alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

# Annotate words
for i, word in enumerate(words):
    ax.annotate(word,
               (embeddings_tsne[i, 0], embeddings_tsne[i, 1]),
               fontsize=9, alpha=0.85, fontweight='bold',
               xytext=(3, 3), textcoords='offset points')

ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
ax.set_title(f'Word Embeddings Visualization (t-SNE)\nColors indicate word length', 
            fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Word Length', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ========================================================
# PART 5: Comparing PCA and t-SNE
# ========================================================

print("\n" + "=" * 70)
print("PART 5: PCA vs t-SNE Comparison")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# PCA plot
ax1.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1],
           alpha=0.6, s=40, c='blue')
for i, word in enumerate(words[:50]):  # Show first 50 for clarity
    ax1.annotate(word, (embeddings_pca[i, 0], embeddings_pca[i, 1]),
                fontsize=8, alpha=0.7)
ax1.set_xlabel('PC1', fontsize=11)
ax1.set_ylabel('PC2', fontsize=11)
ax1.set_title('PCA: Linear dimensionality reduction', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# t-SNE plot
ax2.scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1],
           alpha=0.6, s=40, c='red')
for i, word in enumerate(words[:50]):  # Show first 50 for clarity
    ax2.annotate(word, (embeddings_tsne[i, 0], embeddings_tsne[i, 1]),
                fontsize=8, alpha=0.7)
ax2.set_xlabel('t-SNE Dim 1', fontsize=11)
ax2.set_ylabel('t-SNE Dim 2', fontsize=11)
ax2.set_title('t-SNE: Non-linear, preserves local structure', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================================
# PART 6: Semantic Analysis
# ========================================================

print("\n" + "=" * 70)
print("PART 6: Analyzing Semantic Clusters")
print("=" * 70)

# Find clusters of similar words in t-SNE space
print("\nLooking for semantic clusters in the visualization...")

# Calculate pairwise distances in t-SNE space
from scipy.spatial.distance import cdist
distances_tsne = cdist(embeddings_tsne, embeddings_tsne, 'euclidean')

# For each word, find nearest neighbors in t-SNE space
print("\nNearest neighbors in embedding space (t-SNE visualization):")
interesting_words = ["beauty", "thy", "love", "shall", "eyes"]

for word in interesting_words:
    if word not in word_to_ix:
        continue
    
    idx = word_to_ix[word]
    # Get nearest neighbors (exclude self)
    distances_from_word = distances_tsne[idx].copy()
    distances_from_word[idx] = float('inf')
    nearest_indices = np.argsort(distances_from_word)[:5]
    
    print(f"\n{word}:")
    for i, near_idx in enumerate(nearest_indices):
        near_word = ix_to_word[near_idx]
        dist = distances_tsne[idx, near_idx]
        print(f"  {i+1}. {near_word} (distance: {dist:.2f})")

# ========================================================
# PART 7: Interactive Analysis Tips
# ========================================================

print("\n" + "=" * 70)
print("PART 7: Interpretation Guide")
print("=" * 70)

print("""
How to interpret the visualizations:

PCA (Principal Component Analysis):
-----------------------------------
✓ Linear transformation
✓ Preserves global structure
✓ Explained variance tells you how much info is retained
✓ Axes have mathematical meaning (principal components)
✗ May not capture complex relationships

t-SNE (t-Distributed Stochastic Neighbor Embedding):
---------------------------------------------------
✓ Non-linear transformation
✓ Excellent at preserving local structure
✓ Creates clear clusters
✓ Similar words appear close together
✗ Distances between clusters are not meaningful
✗ Different runs may produce different layouts
✗ Computationally expensive

What to look for:
----------------
1. Clusters: Groups of related words
2. Outliers: Rare or unique words
3. Proximity: Similar words should be close
4. Patterns: Semantic or syntactic groupings

Example interpretations:
----------------------
- Pronouns (thy, thine, thee) might cluster together
- Abstract concepts (beauty, truth, love) may be near each other
- Common words vs rare words show different patterns
""")

# ========================================================
# KEY TAKEAWAYS
# ========================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Dimensionality Reduction:
   - Necessary to visualize high-D embeddings
   - PCA: Fast, linear, preserves global structure
   - t-SNE: Slower, non-linear, better clusters

2. Visualization Insights:
   - Similar words cluster together
   - Semantic relationships become visible
   - Can validate embedding quality visually

3. Practical Applications:
   - Debug embedding quality
   - Understand what the model learned
   - Present results to non-technical audiences
   - Identify biases or problems

4. Limitations:
   - 2D projection loses information
   - Visualization ≠ actual embedding space
   - Use as qualitative, not quantitative tool

5. Best Practices:
   - Use t-SNE for presentations
   - Use PCA for quick checks
   - Always validate with quantitative metrics too
   - Try different random seeds

Congratulations! You can now visualize and interpret embeddings!
""")

print("=" * 70)
print("END OF TUTORIAL")
print("=" * 70)
