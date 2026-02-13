# ========================================================
# 02_ngram_cross_entropy.py
# N-gram Language Model with CrossEntropyLoss
# ========================================================

"""
TUTORIAL 2: N-gram Language Model with CrossEntropyLoss

Learning Objectives:
- Build a complete n-gram language model
- Train word embeddings end-to-end
- Understand CrossEntropyLoss
- Visualize training progress

Estimated time: 20 minutes

What is an N-gram Language Model?
----------------------------------
An n-gram model predicts the next word based on the previous (n-1) words.

Example (trigram, n=3):
  Input: "the cat"
  Output: "sat" (predicted next word)

We'll use embeddings to represent the context words!
"""

import sys
import os

# Add utils directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils import word_embedding_ngram as ngr

print("=" * 70)
print("TUTORIAL 2: N-gram Language Model with CrossEntropyLoss")
print("=" * 70)

# ========================================================
# PART 1: Understanding the Data
# ========================================================

print("\n" + "=" * 70)
print("PART 1: Understanding N-gram Training Data")
print("=" * 70)

# Look at the sample text
print("\nSample text (first 20 words):")
print(" ".join(ngr.ARGS.test_sentence[:20]))

print(f"\nVocabulary size: {ngr.ARGS.vocab_size} unique words")
print(f"Context size: {ngr.ARGS.context_size} words (so this is a {ngr.ARGS.context_size + 1}-gram model)")

# Create training examples
trigrams = ngr.make_context_target_pairs(ngr.ARGS.test_sentence, ngr.ARGS.context_size)

print(f"\nNumber of training examples: {len(trigrams)}")
print("\nFirst 5 training examples:")
for i, (context, target) in enumerate(trigrams[:5]):
    print(f"  {i+1}. Context: {context} → Target: {target}")

print("\nExplanation:")
print("  The model learns to predict the target word from context words")
print("  Each context word is represented as an embedding vector")
print("  The model combines context embeddings to make predictions")

# ========================================================
# PART 2: The Model Architecture
# ========================================================

print("\n" + "=" * 70)
print("PART 2: N-gram Model Architecture")
print("=" * 70)

# Create the model
model = ngr.NGramLanguageModeler()

print("\nModel structure:")
print(model)

print(f"\n--- Layer details ---")
print(f"1. Embedding layer:")
print(f"   - Input: Word indices (integers)")
print(f"   - Output: {ngr.ARGS.embedding_dim}-dimensional vectors")
print(f"   - Parameters: {ngr.ARGS.vocab_size} × {ngr.ARGS.embedding_dim} = {ngr.ARGS.vocab_size * ngr.ARGS.embedding_dim}")

print(f"\n2. First linear layer:")
print(f"   - Input: Concatenated embeddings ({ngr.ARGS.context_size} × {ngr.ARGS.embedding_dim} = {ngr.ARGS.context_size * ngr.ARGS.embedding_dim})")
print(f"   - Output: 128 hidden units")
print(f"   - Activation: ReLU")

print(f"\n3. Second linear layer:")
print(f"   - Input: 128 hidden units")
print(f"   - Output: {ngr.ARGS.vocab_size} (one score per word)")
print(f"   - No activation (raw logits)")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal trainable parameters: {total_params:,}")

# ========================================================
# PART 3: Understanding CrossEntropyLoss
# ========================================================

print("\n" + "=" * 70)
print("PART 3: CrossEntropyLoss Explained")
print("=" * 70)

print("""
CrossEntropyLoss is used for classification tasks (like predicting words).

What it does:
1. Takes model outputs (logits) - raw scores for each word
2. Applies softmax to convert scores to probabilities
3. Computes negative log likelihood of correct word

Formula:
  CrossEntropyLoss = -log(softmax(logits)[correct_class])

Why use it for language modeling?
- Word prediction is a classification problem
- We have vocab_size classes (all possible words)
- We want to maximize probability of correct word
- Minimizing cross-entropy = maximizing correct probability

Key properties:
- Automatically applies softmax (don't do it yourself!)
- Expects raw logits, not probabilities
- Combines LogSoftmax + NLLLoss for numerical stability
""")

# Demonstration with a simple example
print("\n--- Simple Example ---")
print("\nImagine we have 3 words: ['cat', 'dog', 'bird']")
print("Model predicts: cat")
print("Correct answer: dog")

# Create example
vocab_size_example = 3
logits = torch.tensor([[2.0, 5.0, 1.0]])  # Scores for [cat, dog, bird]
target = torch.tensor([1])  # Index 1 = dog

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, target)

print(f"\nLogits (raw scores): {logits[0]}")
print(f"Target: index {target.item()} (dog)")

# Show softmax probabilities
probs = torch.softmax(logits, dim=1)
print(f"\nAfter softmax (probabilities):")
print(f"  cat:  {probs[0, 0]:.4f}")
print(f"  dog:  {probs[0, 1]:.4f}")
print(f"  bird: {probs[0, 2]:.4f}")

print(f"\nCrossEntropyLoss: {loss:.4f}")
print(f"  (Lower is better - we want high probability for correct word)")

# ========================================================
# PART 4: Training the Model
# ========================================================

print("\n" + "=" * 70)
print("PART 4: Training the Model")
print("=" * 70)

# Set up training
loss_function = nn.CrossEntropyLoss()  # Our loss function
optimizer = optim.SGD(model.parameters(), lr=ngr.ARGS.lr)  # SGD optimizer

print("\nTraining configuration:")
print(f"  Loss function: CrossEntropyLoss")
print(f"  Optimizer: SGD")
print(f"  Learning rate: {ngr.ARGS.lr}")
print(f"  Epochs: {ngr.ARGS.epochs}")

print("\nStarting training...")
print("(This will take a few seconds)\n")

# Train the model
losses = ngr.train(model, loss_function, optimizer, epochs=ngr.ARGS.epochs, verbose=True)

# ========================================================
# PART 5: Analyzing Training Results
# ========================================================

print("\n" + "=" * 70)
print("PART 5: Analyzing Results")
print("=" * 70)

# Plot training curve
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(losses, label='Training Loss', linewidth=2, color='blue')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training Progress: N-gram Model with CrossEntropyLoss', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"\nTraining statistics:")
print(f"  Initial loss: {losses[0]:.4f}")
print(f"  Final loss: {losses[-1]:.4f}")
print(f"  Improvement: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

# ========================================================
# PART 6: Examining Learned Embeddings
# ========================================================

print("\n" + "=" * 70)
print("PART 6: Examining Learned Embeddings")
print("=" * 70)

# Pick some interesting words to examine
example_words = ["beauty", "youth", "eyes", "thy", "shall"]

print("\nLearned embeddings for sample words:")
for word in example_words:
    if word in ngr.ARGS.word_to_ix:
        embedding = model.embeddings.weight[ngr.ARGS.word_to_ix[word]]
        print(f"\n{word}:")
        print(f"  {embedding.detach().numpy()}")
    else:
        print(f"\n{word}: (not in vocabulary)")

print("\nThese embeddings were learned from the training data!")
print("Similar words should have similar embedding vectors.")

# Calculate similarity between some words
if "beauty" in ngr.ARGS.word_to_ix and "youth" in ngr.ARGS.word_to_ix:
    beauty_emb = model.embeddings.weight[ngr.ARGS.word_to_ix["beauty"]]
    youth_emb = model.embeddings.weight[ngr.ARGS.word_to_ix["youth"]]
    
    cos = nn.CosineSimilarity(dim=0)
    similarity = cos(beauty_emb, youth_emb)
    
    print(f"\nCosine similarity between 'beauty' and 'youth': {similarity:.4f}")
    print("(Values close to 1 indicate similar words)")

# ========================================================
# PART 7: Making Predictions
# ========================================================

print("\n" + "=" * 70)
print("PART 7: Making Predictions")
print("=" * 70)

# Test the model on some context
test_contexts = [
    ["thy", "beauty's"],
    ["deep", "sunken"],
    ["fair", "child"]
]

print("\nModel predictions:")
for context in test_contexts:
    # Check if context words are in vocabulary
    if all(word in ngr.ARGS.word_to_ix for word in context):
        # Prepare input
        context_idxs = ngr.prepare_sequence(context, ngr.ARGS.word_to_ix)
        
        # Get prediction
        with torch.no_grad():  # No need to track gradients
            logits = model(context_idxs.unsqueeze(0))
            predicted_idx = torch.argmax(logits, dim=1).item()
        
        # Convert index back to word
        ix_to_word = {idx: word for word, idx in ngr.ARGS.word_to_ix.items()}
        predicted_word = ix_to_word[predicted_idx]
        
        # Get top 3 predictions
        top_probs, top_indices = torch.topk(torch.softmax(logits, dim=1), k=3)
        top_words = [ix_to_word[idx.item()] for idx in top_indices[0]]
        
        print(f"\nContext: {context}")
        print(f"  Top prediction: {predicted_word}")
        print(f"  Top 3 predictions: {', '.join(top_words)}")

# ========================================================
# PART 8: Key Takeaways
# ========================================================

print("\n" + "=" * 70)
print("KEY TAKEAWAYS")
print("=" * 70)

print("""
1. N-gram language models predict next word from context
   
2. CrossEntropyLoss is perfect for word prediction:
   - Combines softmax + negative log likelihood
   - Expects raw logits (unnormalized scores)
   - Automatically handles probability conversion
   
3. Training process:
   - Forward pass: context → embeddings → hidden → logits
   - Loss: compare logits with target word
   - Backward pass: compute gradients
   - Update: adjust weights with optimizer
   
4. Word embeddings are learned automatically:
   - Initially random
   - Adjusted during training
   - Capture word relationships
   
5. Loss decreasing = model improving:
   - Better at predicting correct words
   - Learned meaningful word representations

What's Next?
-----------
In the next tutorial, we'll use F.cross_entropy (functional API)
and compare it with nn.CrossEntropyLoss.
""")

print("=" * 70)
print("END OF TUTORIAL 2")
print("=" * 70)

# ========================================================
# EXERCISE (Optional)
# ========================================================

print("\n" + "=" * 70)
print("OPTIONAL EXERCISES")
print("=" * 70)

print("""
Try these modifications:

1. Change the learning rate (ngr.ARGS.lr):
   - Try 0.01, 0.1, 0.0001
   - How does it affect training?

2. Increase epochs to 200:
   - Does the model keep improving?
   - Is there a point of diminishing returns?

3. Change embedding dimension:
   - Try 5, 20, 50
   - How does it affect the loss?

4. Add more context words:
   - Change context_size in word_embedding_ngram.py
   - What happens to training?

Experimentation is the best way to learn!
""")
