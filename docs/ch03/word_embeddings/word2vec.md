# Word2Vec: Skip-gram and CBOW

## Overview

Word2Vec, introduced by Mikolov et al. (2013), is a family of models for learning word embeddings from large text corpora. The two main architectures are **Skip-gram** (predicting context from a center word) and **CBOW** (Continuous Bag of Words, predicting a center word from context). These models revolutionized NLP by producing embeddings that capture semantic relationships through simple, efficient training objectives.

## Learning Objectives

By the end of this section, you will:

- Understand the Skip-gram and CBOW architectures
- Derive the training objectives for both models
- Implement both models from scratch in PyTorch
- Compare the strengths and weaknesses of each approach
- Train word embeddings on sample text data

## The Distributional Hypothesis Revisited

Word2Vec operationalizes the distributional hypothesis through prediction tasks:

> Words that appear in similar contexts should have similar representations.

**Key Insight:** Rather than explicitly counting co-occurrences, Word2Vec learns embeddings by predicting words from their contexts (or vice versa).

## Context Windows

Both architectures use a **context window** of size $k$ to define which words are related:

For a sequence $w_1, w_2, \ldots, w_T$ and center word $w_t$, the context consists of words within $k$ positions:

$$\text{Context}(w_t) = \{w_{t-k}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+k}\}$$

**Example with $k=2$:**

```
Sentence: "The quick brown fox jumps over the lazy dog"
Center word: "fox" (position 4)
Context window: ["quick", "brown", "jumps", "over"]
```

## Skip-gram Model

### Architecture Overview

Skip-gram predicts context words given a center word:

$$P(\text{context} | \text{center}) = \prod_{-k \leq j \leq k, j \neq 0} P(w_{t+j} | w_t)$$

**Architecture Diagram:**

```
Input (center word)     Hidden Layer     Output (context words)
    "fox"          →    embedding    →   ["quick", "brown", "jumps", "over"]
  [one-hot]             [dense]           [probability distribution × 4]
```

### Mathematical Formulation

For center word $w_c$ and context word $w_o$, the probability is modeled using softmax:

$$P(w_o | w_c) = \frac{\exp(\mathbf{u}_{w_o}^\top \mathbf{v}_{w_c})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_c})}$$

Where:
- $\mathbf{v}_{w_c} \in \mathbb{R}^d$: Input (center) embedding of word $w_c$
- $\mathbf{u}_{w_o} \in \mathbb{R}^d$: Output (context) embedding of word $w_o$
- $V$: Vocabulary

**Note:** Word2Vec uses two embedding matrices—one for center words and one for context words.

### Training Objective

The Skip-gram objective maximizes the log probability of context words:

$$J_{\text{Skip-gram}} = \frac{1}{T} \sum_{t=1}^{T} \sum_{-k \leq j \leq k, j \neq 0} \log P(w_{t+j} | w_t)$$

Equivalently, we minimize the negative log-likelihood:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \sum_{-k \leq j \leq k, j \neq 0} \log \frac{\exp(\mathbf{u}_{w_{t+j}}^\top \mathbf{v}_{w_t})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{v}_{w_t})}$$

### PyTorch Implementation

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
        
        # Context word embeddings (output) - projects to vocabulary size
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


def create_skipgram_dataset(tokens, window_size, word_to_ix):
    """
    Create Skip-gram training pairs: (center_word, context_word).
    
    For each center word, we create one pair for each context word.
    """
    data = []
    
    for i in range(window_size, len(tokens) - window_size):
        center = tokens[i]
        
        # Get all context words
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


# Example usage
def train_skipgram(tokens, embedding_dim=50, window_size=2, epochs=100, lr=0.01):
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

## CBOW Model

### Architecture Overview

CBOW (Continuous Bag of Words) predicts the center word given its context:

$$P(\text{center} | \text{context}) = P(w_t | w_{t-k}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+k})$$

**Architecture Diagram:**

```
Input (context words)      Hidden Layer        Output (center word)
["quick", "brown",    →    average of     →    "fox"
 "jumps", "over"]          embeddings          [probability distribution]
```

### Mathematical Formulation

CBOW averages the context word embeddings and predicts the center word:

$$\hat{\mathbf{v}} = \frac{1}{2k} \sum_{-k \leq j \leq k, j \neq 0} \mathbf{v}_{w_{t+j}}$$

$$P(w_t | \text{context}) = \frac{\exp(\mathbf{u}_{w_t}^\top \hat{\mathbf{v}})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \hat{\mathbf{v}})}$$

**Key Difference from Skip-gram:** Context embeddings are averaged before prediction.

### Training Objective

The CBOW objective maximizes the log probability of center words:

$$J_{\text{CBOW}} = \frac{1}{T} \sum_{t=1}^{T} \log P(w_t | w_{t-k}, \ldots, w_{t+k})$$

### PyTorch Implementation

```python
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


def create_cbow_dataset(tokens, window_size, word_to_ix):
    """
    Create CBOW training pairs: (context_words, center_word).
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
# Sample text (Shakespeare sonnet excerpt)
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
print(f"Sample tokens: {tokens[:10]}")

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

## Skip-gram vs. CBOW Comparison

| Aspect | Skip-gram | CBOW |
|--------|-----------|------|
| **Prediction Task** | Context from center | Center from context |
| **Training Examples** | One per (center, context) pair | One per center word |
| **Speed** | Slower (more examples) | Faster (fewer examples) |
| **Rare Words** | Better performance | Worse performance |
| **Frequent Words** | Good performance | Better performance |
| **Dataset Size** | Better for smaller datasets | Better for larger datasets |
| **Memory** | Higher (more updates) | Lower (batched context) |

### When to Use Each

**Use Skip-gram when:**
- Working with smaller datasets
- Rare words are important
- Quality is more important than speed

**Use CBOW when:**
- Working with large datasets
- Training speed is critical
- Frequent words are more important

## Gradient Derivation

### Skip-gram Gradient

For the softmax loss with center word $c$ and context word $o$:

$$\mathcal{L} = -\log \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_{w=1}^{|V|} \exp(\mathbf{u}_w^\top \mathbf{v}_c)}$$

**Gradient with respect to center embedding $\mathbf{v}_c$:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_c} = -\mathbf{u}_o + \sum_{w=1}^{|V|} P(w|c) \cdot \mathbf{u}_w$$

**Interpretation:** Push the center embedding toward the true context word and away from expected context words.

**Gradient with respect to context embedding $\mathbf{u}_o$:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{u}_o} = -(1 - P(o|c)) \cdot \mathbf{v}_c$$

### CBOW Gradient

For averaged context embedding $\hat{\mathbf{v}}$ and center word $t$:

$$\frac{\partial \mathcal{L}}{\partial \hat{\mathbf{v}}} = -\mathbf{u}_t + \sum_{w=1}^{|V|} P(w|\text{context}) \cdot \mathbf{u}_w$$

The gradient is distributed equally to all context word embeddings:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{c_j}} = \frac{1}{2k} \frac{\partial \mathcal{L}}{\partial \hat{\mathbf{v}}}$$

## The Softmax Bottleneck

Both models face a computational challenge: the softmax normalizer requires summing over the entire vocabulary:

$$\sum_{w=1}^{|V|} \exp(\mathbf{u}_w^\top \mathbf{v}_c)$$

For $|V| = 100,000$, this is extremely expensive.

**Solutions** (covered in next sections):
1. **Negative Sampling:** Approximate with binary classification
2. **Hierarchical Softmax:** Use binary tree structure

## Visualization of Training

```python
import matplotlib.pyplot as plt

# Plot training loss
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(cbow_losses, linewidth=2, color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('CBOW Training Loss')
plt.grid(True, alpha=0.3)

# Visualize embeddings with PCA
from sklearn.decomposition import PCA

embeddings = cbow_model.embeddings.weight.detach().numpy()
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.subplot(1, 2, 2)
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

# Annotate some words
words_to_annotate = ["beauty", "thy", "deep", "eyes", "youth"]
for word in words_to_annotate:
    if word in word_to_ix:
        idx = word_to_ix[word]
        plt.annotate(word, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]),
                    fontsize=10, fontweight='bold')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Word Embeddings (PCA)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## Key Takeaways

!!! success "Main Concepts"

    1. **Skip-gram** predicts context from center word; better for rare words
    2. **CBOW** predicts center from context; faster but less effective for rare words
    3. Both models learn two embedding matrices (center and context)
    4. The softmax computation is expensive and requires approximations
    5. Word embeddings capture semantic relationships through the prediction task

!!! warning "Common Pitfalls"

    - Forgetting to shuffle training data
    - Using too large or too small embedding dimensions
    - Not normalizing embeddings before similarity computation
    - Ignoring the softmax bottleneck for large vocabularies

## Next Steps

The following sections will address:

1. **Negative Sampling:** Efficient training approximation
2. **GloVe:** Global vectors using co-occurrence statistics
3. **FastText:** Subword information for morphologically rich languages
4. **Practical Embeddings:** PyTorch implementation details

## References

- Mikolov, T., et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Mikolov, T., et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality"
- Goldberg, Y., & Levy, O. (2014). "word2vec Explained"
