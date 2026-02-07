# Skip-gram Model

## Overview

The Skip-gram model predicts context words given a center word. For each position in the corpus, Skip-gram generates multiple training examples — one for each (center, context) pair — making it particularly effective for learning high-quality embeddings for rare words.

## Learning Objectives

By the end of this section, you will:

- Derive the Skip-gram training objective from first principles
- Implement the Skip-gram model from scratch in PyTorch
- Understand the gradient dynamics of the softmax objective
- Train Skip-gram embeddings on sample text data

## Architecture

Skip-gram predicts context words given a center word:

$$P(\text{context} | \text{center}) = \prod_{-k \leq j \leq k, j \neq 0} P(w_{t+j} | w_t)$$

```
Input (center word)     Hidden Layer     Output (context words)
    "fox"          →    embedding    →   ["quick", "brown", "jumps", "over"]
  [one-hot]             [dense]           [probability distribution × 4]
```

The key assumption is that context words are **conditionally independent** given the center word. While this is linguistically unrealistic, it simplifies training and works well in practice.

## Mathematical Formulation

For center word $w_c$ and context word $w_o$, the probability is modeled using softmax:

$$P(w_o | w_c) = \frac{\exp(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_c})}$$

Where:

- $\mathbf{v}_{w_c} \in \mathbb{R}^d$: Input (center) embedding of word $w_c$
- $\mathbf{u}_{w_o} \in \mathbb{R}^d$: Output (context) embedding of word $w_o$
- $V$: Full vocabulary

## Training Objective

The Skip-gram objective maximizes the log probability of observing context words across the entire corpus:

$$J_{\text{Skip-gram}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{-k \leq j \leq k, j \neq 0} \log P(w_{t+j} | w_t)$$

Equivalently, we minimize the negative log-likelihood:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-k \leq j \leq k, j \neq 0} \log \frac{\exp(\mathbf{u}_{w_{t+j}}^\top \mathbf{v}_{w_t})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_t})}$$

## Gradient Derivation

For a single (center, context) pair with center word $c$ and context word $o$, the per-example loss is:

$$\mathcal{L} = -\log \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_{w=1}^{|V|} \exp(\mathbf{u}_w^\top \mathbf{v}_c)}$$

**Gradient with respect to center embedding $\mathbf{v}_c$:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_c} = -\mathbf{u}_o + \sum_{w=1}^{|V|} P(w|c) \cdot \mathbf{u}_w$$

**Interpretation:** Push the center embedding toward the true context word and away from the expected (probability-weighted) context words.

**Gradient with respect to context embedding $\mathbf{u}_o$ (for the positive word):**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_o} = -(1 - P(o|c)) \cdot \mathbf{v}_c$$

**Gradient with respect to context embedding $\mathbf{u}_w$ (for all other words):**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_w} = P(w|c) \cdot \mathbf{v}_c \quad \text{for } w \neq o$$

## PyTorch Implementation

### Model Definition

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import Counter

class SkipGramModel(nn.Module):
    """
    Skip-gram model: predicts context words from center word.
    
    Architecture:
        center_word → embedding → linear → softmax → context_word_probabilities
    """
    
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramModel, self).__init__()
        
        # Center word embeddings (input)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Context word embeddings (output) — projects to vocabulary size
        self.context_embeddings = nn.Linear(embedding_dim, vocab_size, bias=False)
        
    def forward(self, center_word):
        """
        Args:
            center_word: Tensor of shape (batch_size,) containing center word indices
        
        Returns:
            Logits of shape (batch_size, vocab_size) for each context word
        """
        # Get center word embedding: (batch_size, embedding_dim)
        center_emb = self.center_embeddings(center_word)
        
        # Project to vocabulary size: (batch_size, vocab_size)
        logits = self.context_embeddings(center_emb)
        
        return logits
    
    def get_word_embedding(self, word_idx):
        """Get the learned embedding for a word."""
        return self.center_embeddings.weight[word_idx].detach()
```

### Dataset Construction

```python
def create_skipgram_dataset(tokens, window_size, word_to_ix):
    """
    Create Skip-gram training pairs: (center_word, context_word).
    
    For each center word, we create one pair for each context word
    within the window. This means a single center word generates
    up to 2 * window_size training examples.
    """
    data = []
    
    for i in range(window_size, len(tokens) - window_size):
        center = tokens[i]
        
        # Get all context words within the window
        context_words = (tokens[i - window_size:i] + 
                        tokens[i + 1:i + window_size + 1])
        
        try:
            center_idx = word_to_ix[center]
            
            # Create a pair for each context word
            for context_word in context_words:
                if context_word in word_to_ix:
                    context_idx = word_to_ix[context_word]
                    
                    center_tensor = torch.tensor([center_idx], dtype=torch.long)
                    context_tensor = torch.tensor([context_idx], dtype=torch.long)
                    
                    data.append((center_tensor, context_tensor))
        except KeyError:
            continue
    
    return data
```

### Training Loop

```python
def train_skipgram(tokens, embedding_dim=50, window_size=2, epochs=100, lr=0.01):
    """Train Skip-gram model on a token sequence."""
    # Build vocabulary
    word_counts = Counter(tokens)
    word_to_ix = {word: idx for idx, (word, _) in enumerate(word_counts.most_common())}
    ix_to_word = {idx: word for word, idx in word_to_ix.items()}
    vocab_size = len(word_to_ix)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataset
    dataset = create_skipgram_dataset(tokens, window_size, word_to_ix)
    print(f"Training pairs: {len(dataset)}")
    
    # Initialize model
    model = SkipGramModel(vocab_size, embedding_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        
        for center_tensor, context_tensor in dataset:
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(center_tensor)
            
            # Compute loss
            loss = loss_function(logits, context_tensor)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataset)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model, word_to_ix, ix_to_word, losses
```

## Training Dynamics

### Why Skip-gram Works Well for Rare Words

Each occurrence of a rare word generates $2k$ training examples (one per context word). Since the center word appears in all of them, its embedding receives substantial gradient signal even from a single corpus occurrence. In contrast, CBOW only generates one training example per center position, giving rare words fewer gradient updates.

### Subsampling Frequent Words

Very frequent words (e.g., "the", "a", "is") provide little informational value and slow training. Word2Vec applies **subsampling** to discard frequent words with probability:

$$P(\text{discard } w) = 1 - \sqrt{\frac{t}{f(w)}}$$

where $f(w)$ is the word frequency and $t$ is a threshold (typically $10^{-5}$). This accelerates training and slightly improves embedding quality for rare words.

## Key Takeaways

!!! success "Main Concepts"

    1. **Skip-gram** predicts context words from the center word
    2. Each position generates **multiple training examples** (one per context word)
    3. The **softmax objective** is exact but computationally expensive for large vocabularies
    4. Skip-gram is **better for rare words** because each occurrence generates many gradient updates
    5. **Subsampling** frequent words improves both speed and quality

!!! tip "Best Practices"

    - Use window size 5–10 for semantic relationships
    - Apply subsampling with threshold $t = 10^{-5}$ for large corpora
    - After training, use center embeddings or average center + context
    - Always normalize embeddings before computing cosine similarity

## References

- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality"
- Goldberg, Y., & Levy, O. (2014). "word2vec Explained"
