"""
02_text_preprocessing.py
=========================
Text Preprocessing for RNNs

Learn how to convert raw text into numerical format that RNNs can process.
This is a crucial skill for any NLP task!

Topics: Tokenization, vocabularies, word embeddings, padding
Difficulty: Easy
Time: 30-45 minutes
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
import rnn_utils

print("=" * 70)
print("Text Preprocessing for RNNs")
print("=" * 70)

# Sample texts
texts = [
    "I love machine learning",
    "Deep learning is amazing",
    "PyTorch makes deep learning easy",
    "I love PyTorch",
    "Machine learning is the future"
]

print("\n" + "=" * 70)
print("SECTION 1: Tokenization")
print("=" * 70)

print("\nOriginal texts:")
for i, text in enumerate(texts, 1):
    print(f"  {i}. '{text}'")

# Tokenize (split into words)
tokenized = [text.lower().split() for text in texts]

print("\nTokenized (split into words):")
for i, tokens in enumerate(tokenized, 1):
    print(f"  {i}. {tokens}")

# =============================================================================
# SECTION 2: Building Vocabulary
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Building Vocabulary")
print("=" * 70)

# Build vocabulary
vocab = rnn_utils.Vocabulary()
for text in texts:
    vocab.add_sentence(text.lower())

print(f"\nVocabulary size: {len(vocab)} words")
print(f"\nSpecial tokens:")
print(f"  <PAD>: {vocab.word2idx['<PAD>']} (for padding)")
print(f"  <UNK>: {vocab.word2idx['<UNK>']} (unknown words)")
print(f"  <SOS>: {vocab.word2idx['<SOS>']} (start of sequence)")
print(f"  <EOS>: {vocab.word2idx['<EOS>']} (end of sequence)")

print(f"\nWord to Index mapping:")
for word, idx in sorted(vocab.word2idx.items(), key=lambda x: x[1])[:15]:
    print(f"  '{word}' → {idx}")

# =============================================================================
# SECTION 3: Text to Sequences
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Converting Text to Number Sequences")
print("=" * 70)

print("\nConverting texts to sequences of indices:")
sequences = []
for text in texts:
    seq = rnn_utils.text_to_sequence(text, vocab)
    sequences.append(seq)
    print(f"\n'{text}'")
    print(f"  → {seq}")
    print(f"  Words: {[vocab.idx2word[idx] for idx in seq]}")

# =============================================================================
# SECTION 4: Padding
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Padding Sequences")
print("=" * 70)

max_len = 6
padded_sequences = []

print(f"\nPadding all sequences to length {max_len}:")
for text in texts:
    seq = rnn_utils.text_to_sequence(text, vocab, max_length=max_len)
    padded_sequences.append(seq)
    print(f"\n'{text}'")
    print(f"  Padded: {seq}")

# Convert to tensor
seq_tensor = torch.tensor(padded_sequences)
print(f"\nFinal tensor shape: {seq_tensor.shape}")
print(f"  (batch_size={len(texts)}, sequence_length={max_len})")

# =============================================================================
# SECTION 5: Embeddings
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Word Embeddings")
print("=" * 70)

print("""
Word Embeddings convert word indices to dense vectors:
  • Each word → vector of numbers (e.g., 100-dim)
  • Similar words have similar vectors
  • Learned during training!

Example:
  'king' → [0.2, -0.5, 0.8, ...]
  'queen' → [0.3, -0.4, 0.7, ...]  (similar!)
  'car' → [-0.8, 0.3, -0.2, ...]  (different)
""")

embedding_dim = 50
embedding = nn.Embedding(len(vocab), embedding_dim)

print(f"\nEmbedding layer:")
print(f"  Vocabulary size: {len(vocab)}")
print(f"  Embedding dimension: {embedding_dim}")
print(f"  Total parameters: {len(vocab) * embedding_dim:,}")

# Embed sequences
embedded = embedding(seq_tensor)
print(f"\nAfter embedding:")
print(f"  Input shape: {seq_tensor.shape}")
print(f"  Output shape: {embedded.shape}")
print(f"  (batch, seq_len, embedding_dim)")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
✅ Text Preprocessing Pipeline:
   1. Tokenization: text → words
   2. Vocabulary: words → indices
   3. Sequences: texts → lists of indices
   4. Padding: variable → fixed length
   5. Embedding: indices → dense vectors

✅ Key Components:
   • Vocabulary: Maps words ↔ indices
   • Special tokens: <PAD>, <UNK>, <SOS>, <EOS>
   • Embedding layer: Learnable word vectors
   • Padding: Handle variable lengths

✅ Next: 03_time_series_basics.py
   Then: 04_simple_rnn.py (build first RNN!)

Common Vocabulary Sizes:
  • Small datasets: 5,000 - 10,000 words
  • Medium: 20,000 - 50,000 words
  • Large: 100,000+ words (GPT uses 50,257)
""")

print("=" * 70)
print("Tutorial Complete! ✓")
print("=" * 70)
