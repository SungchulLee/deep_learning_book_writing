# ========================================================
# 01_simple_embeddings.py
# Introduction to Word Embeddings
# ========================================================

"""
TUTORIAL 1: Introduction to Word Embeddings

Learning Objectives:
- Understand what word embeddings are
- Learn how to create an embedding layer in PyTorch
- See how words are mapped to dense vectors
- Understand the difference between one-hot encoding and embeddings

Estimated time: 15 minutes

What are word embeddings?
-------------------------
Word embeddings represent words as dense vectors of real numbers.
Unlike one-hot encoding (sparse, high-dimensional), embeddings are:
  - Dense: All values are non-zero
  - Low-dimensional: Typically 50-300 dimensions
  - Learned: Trained to capture semantic relationships
  - Meaningful: Similar words have similar vectors
"""

import torch
import torch.nn as nn
import numpy as np

print("=" * 70)
print("TUTORIAL 1: Introduction to Word Embeddings")
print("=" * 70)

# ========================================================
# PART 1: The Problem with One-Hot Encoding
# ========================================================

print("\n" + "=" * 70)
print("PART 1: Why Do We Need Embeddings?")
print("=" * 70)

# Small vocabulary for demonstration
vocabulary = ["cat", "dog", "bird", "fish", "lion"]
vocab_size = len(vocabulary)

print(f"\nOur vocabulary: {vocabulary}")
print(f"Vocabulary size: {vocab_size}")

# Create word-to-index mapping
word_to_ix = {word: i for i, word in enumerate(vocabulary)}
print(f"\nWord to index mapping: {word_to_ix}")

# One-hot encoding example
print("\n--- One-Hot Encoding ---")
word = "cat"
word_idx = word_to_ix[word]

# Create one-hot vector
one_hot = torch.zeros(vocab_size)
one_hot[word_idx] = 1
print(f"\nOne-hot vector for '{word}': {one_hot}")
print(f"Vector size: {one_hot.shape[0]}")
print(f"Number of non-zero elements: {one_hot.count_nonzero().item()}")

print("\nProblems with one-hot encoding:")
print("  1. Very sparse (only one element is 1, rest are 0)")
print("  2. High-dimensional (size = vocabulary size)")
print("  3. No semantic meaning (all words equally distant)")
print("  4. Can't capture relationships (cat and lion are treated the same as cat and fish)")

# ========================================================
# PART 2: Creating Word Embeddings
# ========================================================

print("\n" + "=" * 70)
print("PART 2: Creating Word Embeddings with PyTorch")
print("=" * 70)

# Define embedding dimension (much smaller than vocabulary size)
embedding_dim = 3  # In practice, use 50-300

print(f"\nVocabulary size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")

# Create embedding layer
# This is a lookup table: vocab_size x embedding_dim
embeddings = nn.Embedding(vocab_size, embedding_dim)

print(f"\nEmbedding layer created!")
print(f"Shape: {embeddings.weight.shape}")  # (5, 3)
print(f"  → {vocab_size} words, each represented by {embedding_dim} numbers")

# ========================================================
# PART 3: Getting Word Embeddings
# ========================================================

print("\n" + "=" * 70)
print("PART 3: Looking Up Word Embeddings")
print("=" * 70)

# Get embedding for "cat"
word = "cat"
word_idx = word_to_ix[word]
word_tensor = torch.tensor([word_idx], dtype=torch.long)

# Look up the embedding
cat_embedding = embeddings(word_tensor)

print(f"\nWord: '{word}'")
print(f"Index: {word_idx}")
print(f"Embedding vector: {cat_embedding}")
print(f"Vector shape: {cat_embedding.shape}")  # (1, 3)

print("\nKey observations:")
print("  1. Dense: All values are non-zero")
print("  2. Low-dimensional: Only 3 numbers instead of 5 (one-hot)")
print("  3. Learned: These values will be updated during training")

# ========================================================
# PART 4: Multiple Words
# ========================================================

print("\n" + "=" * 70)
print("PART 4: Getting Embeddings for Multiple Words")
print("=" * 70)

# Get embeddings for multiple words
words = ["cat", "dog", "lion"]
word_indices = [word_to_ix[w] for w in words]
word_tensor = torch.tensor(word_indices, dtype=torch.long)

# Batch lookup
batch_embeddings = embeddings(word_tensor)

print(f"\nWords: {words}")
print(f"Indices: {word_indices}")
print(f"\nBatch embeddings shape: {batch_embeddings.shape}")  # (3, 3)
print(f"  → 3 words, each with 3-dimensional embedding")

print("\nIndividual embeddings:")
for i, word in enumerate(words):
    print(f"\n{word}:")
    print(f"  {batch_embeddings[i]}")

# ========================================================
# PART 5: Understanding the Embedding Matrix
# ========================================================

print("\n" + "=" * 70)
print("PART 5: The Embedding Matrix")
print("=" * 70)

print("\nThe embedding layer is just a lookup table!")
print("Let's examine the embedding matrix:\n")

print("Embedding matrix (weight):")
print(embeddings.weight)
print(f"\nShape: {embeddings.weight.shape}")

print("\n--- How lookup works ---")
word = "dog"
idx = word_to_ix[word]
print(f"\nWord: '{word}' → Index: {idx}")
print(f"Embedding is row {idx} of the matrix:")
print(f"  {embeddings.weight[idx]}")

# Verify this is the same as using the embedding layer
retrieved = embeddings(torch.tensor([idx]))
print(f"\nUsing embedding layer: {retrieved[0]}")
print(f"Direct lookup: {embeddings.weight[idx]}")
print(f"Are they the same? {torch.allclose(retrieved[0], embeddings.weight[idx])}")

# ========================================================
# PART 6: Embedding Similarity (Preview)
# ========================================================

print("\n" + "=" * 70)
print("PART 6: Measuring Similarity (Preview)")
print("=" * 70)

print("\nWith embeddings, we can measure word similarity!")

# Calculate cosine similarity between cat and dog
cat_emb = embeddings.weight[word_to_ix["cat"]]
dog_emb = embeddings.weight[word_to_ix["dog"]]
fish_emb = embeddings.weight[word_to_ix["fish"]]

# Cosine similarity
cos = nn.CosineSimilarity(dim=0)
cat_dog_sim = cos(cat_emb, dog_emb)
cat_fish_sim = cos(cat_emb, fish_emb)

print(f"\nCosine similarity:")
print(f"  cat ↔ dog:  {cat_dog_sim:.4f}")
print(f"  cat ↔ fish: {cat_fish_sim:.4f}")

print("\nNote: These similarities are random right now!")
print("After training, similar words (cat, dog) should have higher similarity")
print("than dissimilar words (cat, fish).")

# ========================================================
# PART 7: Key Takeaways
# ========================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. Word embeddings represent words as dense, low-dimensional vectors
   
2. The embedding layer is a learnable lookup table:
   - Input: Word index (integer)
   - Output: Embedding vector (dense)
   
3. Benefits over one-hot encoding:
   - Much smaller dimension
   - Can capture semantic relationships
   - All values are meaningful (not just one 1 and many 0s)
   
4. Initially random, embeddings are trained to be meaningful:
   - Similar words → similar vectors
   - Word relationships captured in vector space
   
5. Common embedding dimensions:
   - Small datasets: 50-100
   - Medium datasets: 100-200
   - Large datasets: 200-300

Next Steps:
----------
In the next tutorial, you'll learn how to TRAIN these embeddings
using an N-gram language model!
""")

print("=" * 70)
print("END OF TUTORIAL 1")
print("=" * 70)

# ========================================================
# EXERCISE (Optional)
# ========================================================

print("\n" + "=" * 70)
print("OPTIONAL EXERCISE")
print("=" * 70)

print("""
Try modifying the code above:

1. Add more words to the vocabulary
2. Change the embedding dimension to 10
3. Compare embeddings of different word pairs
4. Print the entire embedding matrix and observe its structure

Experiment with the code to build your intuition!
""")
