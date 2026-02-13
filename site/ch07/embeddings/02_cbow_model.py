# ========================================================
# 02_cbow_model.py
# Continuous Bag of Words (CBOW) Model
# ========================================================

"""
INTERMEDIATE TUTORIAL 2: Continuous Bag of Words (CBOW)

Learning Objectives:
- Understand CBOW architecture
- Implement CBOW from scratch
- Compare with N-gram model
- Learn context window concept

Estimated time: 30 minutes

What is CBOW?
------------
CBOW (Continuous Bag of Words) predicts a center word from its context.

N-gram:  [word1, word2] → word3
CBOW:    [word1, word2, word4, word5] → word3 (center)

Key differences:
- CBOW uses words from both sides of the target
- More context = better predictions
- Foundation of Word2Vec
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.data_loader import (
    load_text_from_file, 
    simple_tokenize, 
    build_vocabulary,
    create_cbow_dataset,
    print_corpus_stats,
    print_dataset_stats
)

print("=" * 70)
print("INTERMEDIATE TUTORIAL 2: Continuous Bag of Words (CBOW)")
print("=" * 70)

# ========================================================
# PART 1: Load and Prepare Data
# ========================================================

print("\n" + "=" * 70)
print("PART 1: Data Preparation")
print("=" * 70)

# Load text from file
text = load_text_from_file('../data/sample_text.txt')
tokens = simple_tokenize(text, lowercase=True)

print(f"Loaded {len(tokens)} tokens from sample_text.txt")
print(f"First 50 words: {' '.join(tokens[:50])}")

# Build vocabulary
word_to_ix, ix_to_word = build_vocabulary(tokens, min_freq=1)
vocab_size = len(word_to_ix)

print(f"\nVocabulary size: {vocab_size}")
print_corpus_stats(tokens, word_to_ix)

# ========================================================
# PART 2: CBOW Data Creation
# ========================================================

print("\n" + "=" * 70)
print("PART 2: Creating CBOW Training Data")
print("=" * 70)

# CBOW uses a window of context words around the target
WINDOW_SIZE = 2  # 2 words on each side

print(f"\nWindow size: {WINDOW_SIZE}")
print(f"Context size: {WINDOW_SIZE * 2} words (both sides)")
print("\nExample:")
print("  Sentence: [w1, w2, w3, w4, w5]")
print(f"  With window=2: context=[w1,w2,w4,w5] → target=w3")

# Create CBOW dataset
cbow_data = create_cbow_dataset(tokens, WINDOW_SIZE, word_to_ix)
print_dataset_stats(cbow_data, "CBOW")

print("\nFirst 5 training examples:")
for i, (context, target) in enumerate(cbow_data[:5]):
    context_words = [ix_to_word[idx.item()] for idx in context]
    target_word = ix_to_word[target.item()]
    print(f"  {i+1}. Context: {context_words} → Target: {target_word}")

# ========================================================
# PART 3: CBOW Model Architecture
# ========================================================

print("\n" + "=" * 70)
print("PART 3: CBOW Model Architecture")
print("=" * 70)


class CBOWModel(nn.Module):
    """
    Continuous Bag of Words (CBOW) Model
    
    Architecture:
        1. Embedding layer: Maps each context word to a vector
        2. Mean pooling: Average all context embeddings
        3. Linear layer: Project to vocabulary size
    
    The key insight: Average context embeddings to predict center word
    """
    
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        
        # Embedding layer: each word → dense vector
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Linear layer: embedding → vocabulary scores
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context_words):
        """
        Args:
            context_words: Tensor of shape (batch_size, context_size)
        
        Returns:
            Logits of shape (batch_size, vocab_size)
        """
        # Get embeddings for all context words
        # Shape: (batch_size, context_size, embedding_dim)
        embeds = self.embeddings(context_words)
        
        # Average the context word embeddings
        # Shape: (batch_size, embedding_dim)
        mean_embeds = torch.mean(embeds, dim=1)
        
        # Project to vocabulary size
        # Shape: (batch_size, vocab_size)
        out = self.linear(mean_embeds)
        
        return out


# Create model
EMBEDDING_DIM = 50
model = CBOWModel(vocab_size, EMBEDDING_DIM)

print("\nCBOW Model Architecture:")
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")
print(f"  Embeddings: {vocab_size} × {EMBEDDING_DIM} = {vocab_size * EMBEDDING_DIM:,}")
print(f"  Linear: {EMBEDDING_DIM} × {vocab_size} + {vocab_size} = {EMBEDDING_DIM * vocab_size + vocab_size:,}")

# ========================================================
# PART 4: Training
# ========================================================

print("\n" + "=" * 70)
print("PART 4: Training CBOW Model")
print("=" * 70)

# Training configuration
LEARNING_RATE = 0.01
EPOCHS = 100
BATCH_SIZE = 32

loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

print(f"\nTraining configuration:")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Training examples: {len(cbow_data)}")

# Training loop
losses = []
print("\nTraining...")

for epoch in range(EPOCHS):
    total_loss = 0
    
    # Batch training
    for i in range(0, len(cbow_data), BATCH_SIZE):
        # Get batch
        batch = cbow_data[i:i+BATCH_SIZE]
        
        if len(batch) == 0:
            continue
        
        # Prepare batch tensors
        contexts = torch.stack([item[0] for item in batch])
        targets = torch.cat([item[1] for item in batch])
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(contexts)
        
        # Compute loss
        loss = loss_function(outputs, targets)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()
    
    # Record average loss
    avg_loss = total_loss / (len(cbow_data) / BATCH_SIZE)
    losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

print(f"\nTraining complete! Final loss: {losses[-1]:.4f}")

# ========================================================
# PART 5: Visualization
# ========================================================

print("\n" + "=" * 70)
print("PART 5: Training Results")
print("=" * 70)

# Plot training curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(losses, linewidth=2, color='green')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('CBOW Training Loss', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(losses[10:], linewidth=2, color='blue')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('CBOW Loss (After Epoch 10)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================================
# PART 6: Testing the Model
# ========================================================

print("\n" + "=" * 70)
print("PART 6: Making Predictions")
print("=" * 70)

# Test predictions
test_examples = [
    ["thy", "brow", "dig", "deep"],  # Should predict something related
    ["the", "of", "and", "to"],
    ["beauty", "youth", "fair", "proud"]
]

print("\nPredicting center words from context:\n")

with torch.no_grad():
    for context_words in test_examples:
        # Check if all words are in vocabulary
        if all(w in word_to_ix for w in context_words):
            # Convert to tensor
            context_tensor = torch.tensor([word_to_ix[w] for w in context_words]).unsqueeze(0)
            
            # Get prediction
            output = model(context_tensor)
            probs = F.softmax(output, dim=1)
            
            # Get top 5 predictions
            top_probs, top_indices = torch.topk(probs, k=5)
            
            print(f"Context: {context_words}")
            print("Top 5 predictions:")
            for i, (prob, idx) in enumerate(zip(top_probs[0], top_indices[0])):
                word = ix_to_word[idx.item()]
                print(f"  {i+1}. {word}: {prob.item():.4f}")
            print()

# ========================================================
# PART 7: Embedding Analysis
# ========================================================

print("\n" + "=" * 70)
print("PART 7: Analyzing Learned Embeddings")
print("=" * 70)

# Find similar words using cosine similarity
def find_similar_words(word, model, word_to_ix, ix_to_word, top_k=5):
    """Find most similar words to the given word."""
    if word not in word_to_ix:
        return []
    
    word_idx = word_to_ix[word]
    word_embedding = model.embeddings.weight[word_idx]
    
    # Compute similarities with all words
    similarities = F.cosine_similarity(
        word_embedding.unsqueeze(0),
        model.embeddings.weight,
        dim=1
    )
    
    # Exclude the word itself
    similarities[word_idx] = -1
    
    # Get top-k
    top_sim, top_indices = torch.topk(similarities, k=top_k)
    
    results = []
    for sim, idx in zip(top_sim, top_indices):
        results.append((ix_to_word[idx.item()], sim.item()))
    
    return results


# Test similarity
test_words = ["beauty", "thy", "love", "eyes"]

print("\nMost similar words:\n")
for word in test_words:
    if word in word_to_ix:
        similar = find_similar_words(word, model, word_to_ix, ix_to_word)
        print(f"{word}:")
        for sim_word, score in similar:
            print(f"  {sim_word}: {score:.4f}")
        print()

# ========================================================
# KEY TAKEAWAYS
# ========================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. CBOW Architecture:
   - Uses context words from both sides
   - Averages context embeddings
   - Predicts center word
   
2. Advantages:
   - Captures bidirectional context
   - Better for frequent words
   - Faster training than Skip-gram
   
3. Key Insight:
   - Average of context embeddings ≈ center word embedding
   - Words appearing in similar contexts get similar embeddings
   
4. Applications:
   - Word embeddings for NLP tasks
   - Semantic similarity
   - Word analogies
   
5. Next Steps:
   - Skip-gram (complementary to CBOW)
   - Negative sampling for efficiency
   - Full Word2Vec implementation

This is the foundation of Word2Vec's CBOW variant!
""")

print("=" * 70)
print("END OF TUTORIAL")
print("=" * 70)
