# CBOW Model

## Overview

The **Continuous Bag of Words (CBOW)** model predicts the center word given its surrounding context. By averaging context word embeddings before prediction, CBOW is faster to train than Skip-gram and performs well on frequent words, making it the preferred choice for large-scale embedding training when speed is critical.

## Learning Objectives

By the end of this section, you will:

- Derive the CBOW training objective from first principles
- Understand the context averaging mechanism and its implications
- Implement CBOW from scratch in PyTorch
- Recognize when CBOW is preferable to Skip-gram

## Architecture

CBOW predicts the center word given its context:

$$P(\text{center} | \text{context}) = P(w_t | w_{t-k}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+k})$$

```
Input (context words)      Hidden Layer        Output (center word)
["quick", "brown",    →    average of     →    "fox"
 "jumps", "over"]          embeddings          [probability distribution]
```

**Key difference from Skip-gram:** Context embeddings are **averaged** before prediction, collapsing $2k$ vectors into a single representation.

## Mathematical Formulation

### Context Averaging

CBOW first computes the average of all context word embeddings:

$$\hat{\mathbf{v}} = \frac{1}{2k} \sum_{-k \leq j \leq k, j \neq 0} \mathbf{v}_{w_{t+j}}$$

### Prediction

The center word probability is modeled via softmax over this averaged representation:

$$P(w_t | \text{context}) = \frac{\exp(\mathbf{u}_{w_t}^\top \hat{\mathbf{v}})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \hat{\mathbf{v}})}$$

### Training Objective

The CBOW objective maximizes the log probability of center words across the corpus:

$$J_{\text{CBOW}} = \frac{1}{T} \sum_{t=1}^{T} \log P(w_t | w_{t-k}, \ldots, w_{t+k})$$

## Gradient Derivation

For averaged context embedding $\hat{\mathbf{v}}$ and target center word $t$:

**Gradient with respect to the averaged context $\hat{\mathbf{v}}$:**

$$\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{v}}} = -\mathbf{u}_t + \sum_{w=1}^{|V|} P(w|\text{context}) \cdot \mathbf{u}_w$$

**Distribution to individual context words:**

Since $\hat{\mathbf{v}}$ is a simple average, the gradient distributes equally to each context word embedding:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_{t+j}}} = \frac{1}{2k} \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{v}}}$$

This equal distribution is why CBOW is **less effective for rare words**: each context word receives only $\frac{1}{2k}$ of the gradient signal, regardless of its frequency.

## PyTorch Implementation

### Model Definition

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter

class CBOWModel(nn.Module):
    """
    Continuous Bag of Words (CBOW) Model.
    
    Architecture:
        context_words → embeddings → mean → linear → softmax → center_word
    
    The key insight: averaging context embeddings to predict center word.
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
        logits = self.linear(mean_embeds)
        
        return logits
```

### Dataset Construction

```python
def create_cbow_dataset(tokens, window_size, word_to_ix):
    """
    Create CBOW training pairs: (context_words, center_word).
    
    Unlike Skip-gram which creates multiple pairs per position,
    CBOW creates exactly one training example per position.
    """
    data = []
    
    for i in range(window_size, len(tokens) - window_size):
        # Get context: words before and after center word
        context = (tokens[i - window_size:i] + 
                  tokens[i + 1:i + window_size + 1])
        center = tokens[i]
        
        try:
            context_idxs = [word_to_ix[w] for w in context]
            center_idx = word_to_ix[center]
            
            context_tensor = torch.tensor(context_idxs, dtype=torch.long)
            center_tensor = torch.tensor([center_idx], dtype=torch.long)
            
            data.append((context_tensor, center_tensor))
        except KeyError:
            continue
    
    return data
```

### Training Loop

```python
def train_cbow(tokens, embedding_dim=50, window_size=2, epochs=100, lr=0.01, batch_size=32):
    """Train CBOW model on token sequence."""
    
    # Build vocabulary
    word_counts = Counter(tokens)
    word_to_ix = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}
    ix_to_word = {idx: word for word, idx in word_to_ix.items()}
    vocab_size = len(word_to_ix)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataset
    cbow_data = create_cbow_dataset(tokens, window_size, word_to_ix)
    print(f"Training examples: {len(cbow_data)}")
    
    # Initialize model
    model = CBOWModel(vocab_size, embedding_dim)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    
    # Training loop with batching
    losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        
        # Batch training
        for i in range(0, len(cbow_data), batch_size):
            batch = cbow_data[i:i+batch_size]
            
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
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / (len(cbow_data) / batch_size)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, word_to_ix, ix_to_word, losses
```

## Complete Training Example

```python
# Sample text
text = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise."""

# Tokenize
tokens = text.lower().split()
print(f"Total tokens: {len(tokens)}")

# Train CBOW model
print("\n" + "="*50)
print("Training CBOW Model")
print("="*50)
cbow_model, word_to_ix, ix_to_word, cbow_losses = train_cbow(
    tokens, 
    embedding_dim=30,
    window_size=2,
    epochs=100,
    lr=0.01
)

# Find similar words
def find_similar_words(model, word, word_to_ix, ix_to_word, top_k=5):
    """Find most similar words using cosine similarity."""
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
print("\n" + "="*50)
print("Word Similarities (CBOW)")
print("="*50)
test_words = ["beauty", "thy", "deep"]
for word in test_words:
    if word in word_to_ix:
        similar = find_similar_words(cbow_model, word, word_to_ix, ix_to_word)
        print(f"\n{word}:")
        for sim_word, score in similar:
            print(f"  {sim_word}: {score:.4f}")
```

## Why CBOW Is Faster

CBOW generates **one training example per center position** (the full context is collapsed into a single averaged vector), while Skip-gram generates **$2k$ examples** (one per context word). For window size $k = 5$ and corpus of length $T$:

| Model | Training Examples | Updates per Epoch |
|-------|-------------------|-------------------|
| CBOW | $\approx T$ | $T$ |
| Skip-gram | $\approx 2kT = 10T$ | $10T$ |

This 10× reduction in gradient updates makes CBOW significantly faster, though at the cost of less effective rare-word representations.

## Key Takeaways

!!! success "Main Concepts"

    1. **CBOW** predicts the center word from averaged context word embeddings
    2. Context averaging makes CBOW **faster** but less effective for **rare words**
    3. The gradient distributes **equally** to all context words ($\frac{1}{2k}$ each)
    4. CBOW uses a **single embedding matrix** (unlike Skip-gram's two matrices in many implementations)
    5. CBOW is preferred for **large datasets** where training speed matters

!!! warning "Common Pitfalls"

    - Using too large a window size can dilute the averaged context representation
    - CBOW struggles with rare words — consider Skip-gram if rare word quality matters
    - Ensure context tensors have consistent size (handle sentence boundaries)
    - Batch training requires contexts of uniform size — pad or truncate at boundaries

## References

- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality"
