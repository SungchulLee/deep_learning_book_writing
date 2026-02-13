"""
01_sequence_basics.py
=====================
Understanding Sequences and Sequential Data

This tutorial introduces the fundamental concept of sequences and why
they require special treatment compared to regular data.

What you'll learn:
- What makes data "sequential"
- How to represent sequences as tensors
- Time dependencies and ordering
- Batching sequential data
- Padding and masking

Difficulty: Easy
Estimated Time: 30-45 minutes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("Understanding Sequences")
print("=" * 70)

# =============================================================================
# SECTION 1: What is a Sequence?
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 1: What is a Sequence?")
print("=" * 70)

print("""
A SEQUENCE is an ordered collection of data points where:
1. Order matters (swap elements → different meaning)
2. Elements are related through time or position
3. Past elements influence future predictions

Examples:
---------
✓ Text: "I love PyTorch" vs "PyTorch love I"
✓ Time Series: Stock prices [100, 102, 105, 103]
✓ Audio: Sound waveforms
✓ Video: Frames over time
✗ Images: (Usually not sequential, but CNNs treat spatially)
✗ Tabular Data: (Usually orderless)

Key Difference from Regular Data:
---------------------------------
Regular: Each sample is independent
Sequential: Each element depends on previous elements!
""")

# =============================================================================
# SECTION 2: Simple Sequence Examples
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 2: Simple Sequence Examples")
print("=" * 70)

# Example 1: Number sequence
number_seq = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print("\nExample 1: Number Sequence")
print(f"  Sequence: {number_seq}")
print(f"  Pattern: Each number = previous + 1")
print(f"  Next prediction: {number_seq[-1] + 1}")

# Example 2: Text sequence (words)
text_seq = ["The", "cat", "sat", "on", "the", "mat"]
print("\nExample 2: Text Sequence")
print(f"  Sequence: {' '.join(text_seq)}")
print(f"  Order matters: Rearrange → different meaning!")
wrong_order = ["mat", "the", "cat", "The", "on", "sat"]
print(f"  Wrong order: {' '.join(wrong_order)} (nonsense!)")

# Example 3: Time series (temperature)
temps = [72, 73, 75, 78, 82, 85, 87, 85, 80, 75]
hours = list(range(len(temps)))

plt.figure(figsize=(10, 4))
plt.plot(hours, temps, marker='o')
plt.xlabel('Hour')
plt.ylabel('Temperature (°F)')
plt.title('Temperature Over Time (Sequential Data)')
plt.grid(True)
plt.savefig('/home/claude/pytorch_rnn_tutorial/sequence_example.png', dpi=150, bbox_inches='tight')
print("\nExample 3: Time Series (Temperature)")
print(f"  Data: {temps}")
print(f"  Pattern: Temperature rises then falls (daily cycle)")
print("  Plot saved as 'sequence_example.png'")
plt.close()

# =============================================================================
# SECTION 3: Sequences as PyTorch Tensors
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 3: Sequences as PyTorch Tensors")
print("=" * 70)

print("\nSequence Tensor Shapes:")
print("-" * 50)

# Single sequence
single_seq = torch.tensor([1, 2, 3, 4, 5])
print(f"\n1. Single Sequence:")
print(f"   Data: {single_seq}")
print(f"   Shape: {single_seq.shape}")
print(f"   Interpretation: (sequence_length,)")

# Single sequence with features
single_seq_features = torch.randn(5, 3)  # 5 timesteps, 3 features each
print(f"\n2. Single Sequence with Features:")
print(f"   Shape: {single_seq_features.shape}")
print(f"   Interpretation: (sequence_length, num_features)")
print(f"   Example: 5 timesteps, each with [x, y, z] coordinates")

# Batch of sequences (most common in RNNs)
batch_sequences = torch.randn(32, 10, 5)
print(f"\n3. Batch of Sequences (Standard RNN Input):")
print(f"   Shape: {batch_sequences.shape}")
print(f"   Interpretation: (batch_size, sequence_length, num_features)")
print(f"   Example: 32 sequences, each 10 timesteps, 5 features per step")

print("\nShape Convention for RNNs:")
print("  • batch_first=True:  (batch, seq_len, features)")
print("  • batch_first=False: (seq_len, batch, features)")
print("  • We'll use batch_first=True (more intuitive)")

# =============================================================================
# SECTION 4: Variable Length Sequences
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 4: Handling Variable Length Sequences")
print("=" * 70)

print("\nProblem: Real sequences have different lengths!")

# Example sentences with different lengths
sentences = [
    "I love AI",           # 3 words
    "Deep learning is amazing",  # 4 words
    "PyTorch makes it easy to build neural networks"  # 9 words
]

print("\nExample Sentences:")
for i, sent in enumerate(sentences):
    words = sent.split()
    print(f"  {i+1}. '{sent}' → {len(words)} words")

print("\nSolution: PADDING")
print("-" * 50)

# Simulate padding with <PAD> token (index 0)
max_length = 9
padded_sequences = []

for sent in sentences:
    words = sent.split()
    # Represent as indices (simplified)
    indices = list(range(1, len(words) + 1))
    # Pad to max_length
    padded = indices + [0] * (max_length - len(indices))
    padded_sequences.append(padded)

print("\nPadded Sequences:")
for i, (sent, padded) in enumerate(zip(sentences, padded_sequences)):
    print(f"  {i+1}. {padded}")
    print(f"      Real: {sent.split()}")
    print(f"      Padding: {[0] * (max_length - len(sent.split()))}")

# =============================================================================
# SECTION 5: Time Dependencies
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 5: Understanding Time Dependencies")
print("=" * 70)

print("""
Why RNNs for Sequences?
----------------------
Problem: Each element depends on previous elements

Example: Predicting next word
  "The cat sat on the ___"
  
To predict "mat", model needs to know:
  • Previous word: "the"
  • Context: "cat sat on"
  • Grammar: needs a noun
  
Regular neural networks:
  ✗ Process each word independently
  ✗ No memory of previous words
  ✗ Can't capture temporal patterns

RNNs (Recurrent Neural Networks):
  ✓ Maintain hidden state (memory)
  ✓ Pass information from t-1 to t
  ✓ Can learn temporal patterns
  
Mathematical View:
-----------------
Regular NN:  y = f(x)
RNN:         y_t = f(x_t, h_{t-1})
             
Where h_{t-1} is the hidden state from previous timestep!
""")

# Simple visualization of time dependency
sequence = [10, 12, 15, 19, 24, 30]
differences = [sequence[i] - sequence[i-1] for i in range(1, len(sequence))]

print(f"\nExample: Number Pattern")
print(f"Sequence: {sequence}")
print(f"Differences: {differences}")
print(f"Pattern: Each number increases by (prev_diff + 1)")
print(f"Next prediction: {sequence[-1] + differences[-1] + 1} = {sequence[-1] + differences[-1] + 1}")

# =============================================================================
# SECTION 6: Creating Training Data
# =============================================================================

print("\n" + "=" * 70)
print("SECTION 6: Creating Training Data from Sequences")
print("=" * 70)

print("\nHow to create (input, target) pairs from sequences:")

# Example: Predict next number
full_sequence = list(range(1, 11))
window_size = 3

print(f"\nFull sequence: {full_sequence}")
print(f"Window size: {window_size}")
print("\nTraining pairs (input → target):")
print("-" * 50)

for i in range(len(full_sequence) - window_size):
    input_seq = full_sequence[i:i+window_size]
    target = full_sequence[i+window_size]
    print(f"  {input_seq} → {target}")

print("\nThis is called 'sliding window' approach")
print("RNNs learn to predict next element given previous elements!")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY - Key Takeaways")
print("=" * 70)

print("""
✅ What We Learned:
   1. Sequences are ordered data where order matters
   2. Common in text, time series, audio, video
   3. Elements depend on previous elements
   4. Represented as 3D tensors: (batch, seq_len, features)

✅ Key Concepts:
   • Temporal dependencies: x_t depends on x_{t-1}, x_{t-2}, ...
   • Variable lengths: Need padding for batching
   • Sliding windows: Create training pairs
   • Memory: RNNs maintain hidden state

✅ Next Steps:
   → 02_text_preprocessing.py: Learn text tokenization
   → 03_time_series_basics.py: Time series preparation
   → 04_simple_rnn.py: Build your first RNN!

Important Distinctions:
----------------------
CNNs:  Spatial relationships (images)
RNNs:  Temporal relationships (sequences)
CNNs:  Fixed input size
RNNs:  Variable sequence length (with padding)
CNNs:  No memory
RNNs:  Hidden state memory

Pro Tip:
--------
Always visualize your sequences before feeding to RNNs!
Understanding the data structure is half the battle.
""")

print("=" * 70)
print("Tutorial Complete! ✓")
print("=" * 70)
