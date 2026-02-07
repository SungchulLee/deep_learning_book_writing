# Semantic Embedding Methods

## Overview

Semantic embedding methods replace manually-defined attributes with continuous vector representations learned from text corpora. Class names are embedded in a semantic space where similar concepts are close together, enabling zero-shot transfer through linguistic knowledge.

## From Attributes to Embeddings

### Limitations of Attributes

Attribute-based ZSL requires:
- Expert knowledge to define relevant attributes
- Expensive annotation for each class
- Binary or discrete representations may lose nuance
- Attributes may not capture all discriminative information

### Advantages of Embeddings

Semantic embeddings offer:
- **No manual annotation**: Pre-trained on large text corpora
- **Rich representations**: Continuous vectors capture nuanced similarities
- **Compositional semantics**: Relationships between concepts are preserved
- **Scalability**: Any class with a name can be embedded

## Word Embedding Methods

### Word2Vec

Word2Vec learns embeddings by predicting context words:

**Skip-gram**: Predict context from target word
$$P(w_{context} | w_{target}) = \frac{\exp(\mathbf{v}_{context}^\top \mathbf{v}_{target})}{\sum_w \exp(\mathbf{v}_w^\top \mathbf{v}_{target})}$$

**CBOW**: Predict target from context
$$P(w_{target} | w_{context_1}, \ldots, w_{context_k})$$

**Properties**:
- Captures semantic relationships: king - man + woman ≈ queen
- Typical dimensionality: 100-300
- Pre-trained on billions of words

### GloVe (Global Vectors)

GloVe optimizes for word co-occurrence statistics:

$$J = \sum_{i,j} f(X_{ij}) (\mathbf{v}_i^\top \mathbf{v}_j + b_i + b_j - \log X_{ij})^2$$

where $X_{ij}$ is the co-occurrence count and $f$ is a weighting function.

### FastText

FastText extends Word2Vec with subword information:
- Represents words as bag of character n-grams
- Handles out-of-vocabulary words
- Captures morphological information

$$\mathbf{v}_{word} = \sum_{g \in \text{ngrams}(word)} \mathbf{v}_g$$

### Comparison

| Method | Key Feature | OOV Handling | Speed |
|--------|-------------|--------------|-------|
| Word2Vec | Context prediction | None | Fast |
| GloVe | Co-occurrence stats | None | Fast |
| FastText | Subword embeddings | Yes | Medium |
| BERT | Contextual, bidirectional | Yes | Slow |

## Using Embeddings for ZSL

### Class Name Embedding

For a class with name "zebra":

```python
import gensim.downloader as api

# Load pre-trained embeddings
word_vectors = api.load("glove-wiki-gigaword-300")

# Get class embedding
zebra_embedding = word_vectors["zebra"]  # Shape: (300,)
```

### Multi-Word Class Names

For compound names like "polar bear":

```python
def get_class_embedding(class_name, word_vectors):
    """
    Get embedding for potentially multi-word class names.
    """
    words = class_name.lower().split()
    embeddings = []
    
    for word in words:
        if word in word_vectors:
            embeddings.append(word_vectors[word])
    
    if not embeddings:
        raise ValueError(f"No words in '{class_name}' found in vocabulary")
    
    # Average word embeddings
    return np.mean(embeddings, axis=0)
```

### Semantic Structure

Embeddings naturally capture taxonomic relationships:

```python
def visualize_semantic_structure(class_names, word_vectors):
    """
    Visualize class embeddings using t-SNE.
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # Get embeddings
    embeddings = np.array([get_class_embedding(c, word_vectors) 
                          for c in class_names])
    
    # Reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
    
    for i, name in enumerate(class_names):
        plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title("Semantic Embedding Space")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.show()
```

## Visual-Semantic Compatibility Learning

### Formulation

Learn a compatibility function between visual features and semantic embeddings:

$$F: \mathcal{V} \times \mathcal{S} \rightarrow \mathbb{R}$$

The prediction rule is:
$$\hat{y} = \arg\max_{c \in \mathcal{Y}^u} F(\phi(\mathbf{x}), \mathbf{s}_c)$$

### Embedding Space Approaches

**Project Visual to Semantic**:
$$f_v(\mathbf{v}) \in \mathcal{S}$$

Train: $\min_\theta \sum_{(\mathbf{x}, y)} \|f_\theta(\phi(\mathbf{x})) - \mathbf{s}_y\|^2$

**Project Semantic to Visual**:
$$f_s(\mathbf{s}) \in \mathcal{V}$$

**Joint Embedding Space**:
$$f_v(\mathbf{v}), f_s(\mathbf{s}) \in \mathcal{E}$$

### Ranking Loss

Rather than regression, use ranking to ensure correct class scores higher:

$$\mathcal{L} = \sum_{(\mathbf{x}, y)} \sum_{c \neq y} \max(0, \Delta + F(\phi(\mathbf{x}), \mathbf{s}_c) - F(\phi(\mathbf{x}), \mathbf{s}_y))$$

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VisualSemanticEmbedding(nn.Module):
    """
    Visual-Semantic Embedding model for ZSL.
    
    Projects visual features to semantic embedding space.
    """
    
    def __init__(self, visual_dim: int, semantic_dim: int, 
                 embedding_dim: int = 256):
        super().__init__()
        
        self.visual_encoder = nn.Sequential(
            nn.Linear(visual_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
        )
        
        self.semantic_encoder = nn.Sequential(
            nn.Linear(semantic_dim, embedding_dim),
        )
    
    def encode_visual(self, v):
        """Encode visual features."""
        return F.normalize(self.visual_encoder(v), dim=1)
    
    def encode_semantic(self, s):
        """Encode semantic embeddings."""
        return F.normalize(self.semantic_encoder(s), dim=1)
    
    def compatibility(self, v, s):
        """
        Compute compatibility score (cosine similarity).
        """
        v_emb = self.encode_visual(v)
        s_emb = self.encode_semantic(s)
        return torch.sum(v_emb * s_emb, dim=1)
    
    def forward(self, v, s_positive, s_negative):
        """
        Forward pass for ranking loss computation.
        
        Args:
            v: Visual features (batch_size, visual_dim)
            s_positive: Correct class embeddings (batch_size, semantic_dim)
            s_negative: Incorrect class embeddings (batch_size, semantic_dim)
        
        Returns:
            Positive and negative compatibility scores
        """
        v_emb = self.encode_visual(v)
        s_pos_emb = self.encode_semantic(s_positive)
        s_neg_emb = self.encode_semantic(s_negative)
        
        pos_score = torch.sum(v_emb * s_pos_emb, dim=1)
        neg_score = torch.sum(v_emb * s_neg_emb, dim=1)
        
        return pos_score, neg_score


class RankingLoss(nn.Module):
    """
    Margin-based ranking loss for compatibility learning.
    """
    
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, pos_score, neg_score):
        """
        Compute ranking loss.
        
        Loss = max(0, margin + neg_score - pos_score)
        """
        loss = torch.clamp(self.margin + neg_score - pos_score, min=0)
        return loss.mean()


def train_vse_model(model, dataloader, class_embeddings, seen_classes, 
                    epochs=50, lr=0.001, margin=0.2):
    """
    Train Visual-Semantic Embedding model.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = RankingLoss(margin=margin)
    
    # Pre-compute class embeddings tensor
    class_emb_tensor = torch.stack([
        torch.tensor(class_embeddings[c], dtype=torch.float32)
        for c in seen_classes
    ])
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_v, batch_y in dataloader:
            # Get positive embeddings
            s_positive = torch.stack([
                torch.tensor(class_embeddings[y], dtype=torch.float32)
                for y in batch_y
            ])
            
            # Sample negative embeddings (hard negative mining)
            batch_size = len(batch_y)
            neg_indices = torch.randint(0, len(seen_classes), (batch_size,))
            # Ensure negative != positive
            for i, y in enumerate(batch_y):
                while seen_classes[neg_indices[i]] == y:
                    neg_indices[i] = torch.randint(0, len(seen_classes), (1,)).item()
            
            s_negative = class_emb_tensor[neg_indices]
            
            # Forward pass
            pos_score, neg_score = model(batch_v, s_positive, s_negative)
            loss = criterion(pos_score, neg_score)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
```

## ConSE: Convex Combination of Semantic Embeddings

### Concept

ConSE (Convex Combination of Semantic Embeddings) uses seen class probabilities to create a weighted average semantic representation, then finds the nearest unseen class.

### Mathematical Formulation

**Step 1: Seen Class Probabilities**

Train a classifier on seen classes:
$$P(y^s | \mathbf{x}) = \text{softmax}(W^\top \phi(\mathbf{x}))$$

**Step 2: Projected Embedding**

Compute weighted average of seen class embeddings:
$$\hat{\mathbf{z}} = \sum_{c \in \mathcal{Y}^s} P(c | \mathbf{x}) \cdot \mathbf{s}_c$$

**Step 3: Nearest Neighbor**

Find the closest unseen class:
$$\hat{y} = \arg\min_{c \in \mathcal{Y}^u} \|\hat{\mathbf{z}} - \mathbf{s}_c\|$$

### Top-T ConSE

For efficiency and robustness, use only top-T most probable seen classes:

$$\hat{\mathbf{z}} = \sum_{c \in \text{Top}_T(\mathcal{Y}^s | \mathbf{x})} \frac{P(c | \mathbf{x})}{\sum_{c' \in \text{Top}_T} P(c' | \mathbf{x})} \cdot \mathbf{s}_c$$

### PyTorch Implementation

```python
class ConSE(nn.Module):
    """
    Convex Combination of Semantic Embeddings (ConSE).
    
    Uses seen class probabilities to project into semantic space.
    """
    
    def __init__(self, visual_dim: int, n_seen_classes: int, hidden_dim: int = 512):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, n_seen_classes)
        )
    
    def forward(self, x):
        """Get seen class logits."""
        return self.classifier(x)
    
    def predict_unseen(self, x, seen_classes, unseen_classes, 
                       class_embeddings, top_t=None):
        """
        Predict unseen class using ConSE.
        
        Args:
            x: Visual features (batch_size, visual_dim)
            seen_classes: List of seen class names
            unseen_classes: List of unseen class names
            class_embeddings: Dict mapping class names to embeddings
            top_t: Number of top classes to use (None = all)
        
        Returns:
            Predicted unseen class indices
        """
        self.eval()
        
        with torch.no_grad():
            # Get seen class probabilities
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)  # (batch, n_seen)
            
            # Get seen class embeddings
            seen_embs = torch.stack([
                torch.tensor(class_embeddings[c], dtype=torch.float32)
                for c in seen_classes
            ])  # (n_seen, emb_dim)
            
            if top_t is not None:
                # Use only top-T classes
                top_probs, top_indices = torch.topk(probs, k=top_t, dim=1)
                # Renormalize
                top_probs = top_probs / top_probs.sum(dim=1, keepdim=True)
                
                # Compute weighted embeddings
                projected = torch.zeros(x.size(0), seen_embs.size(1))
                for i in range(x.size(0)):
                    for j in range(top_t):
                        projected[i] += top_probs[i, j] * seen_embs[top_indices[i, j]]
            else:
                # Use all classes
                projected = probs @ seen_embs  # (batch, emb_dim)
            
            # Get unseen class embeddings
            unseen_embs = torch.stack([
                torch.tensor(class_embeddings[c], dtype=torch.float32)
                for c in unseen_classes
            ])  # (n_unseen, emb_dim)
            
            # Find nearest unseen class (cosine similarity)
            projected_norm = F.normalize(projected, dim=1)
            unseen_norm = F.normalize(unseen_embs, dim=1)
            
            similarities = projected_norm @ unseen_norm.T  # (batch, n_unseen)
            predictions = torch.argmax(similarities, dim=1)
            
            return predictions


def train_conse_classifier(model, dataloader, seen_classes, epochs=50, lr=0.001):
    """
    Train the seen class classifier for ConSE.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    class_to_idx = {c: i for i, c in enumerate(seen_classes)}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            labels = torch.tensor([class_to_idx[y] for y in batch_y])
            
            logits = model(batch_x)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {total_loss/len(dataloader):.4f}, "
                  f"Acc: {100*correct/total:.2f}%")
```

## Comparison: Compatibility Learning vs ConSE

### Methodology Differences

| Aspect | Compatibility Learning | ConSE |
|--------|----------------------|-------|
| **Learning** | End-to-end visual-semantic | Seen class classifier |
| **Projection** | Learned embedding space | Weighted average |
| **Flexibility** | Any compatibility function | Linear combination |
| **Interpretability** | Lower | Higher (via seen class weights) |

### When to Use Each

**Compatibility Learning**:
- Large training datasets
- End-to-end training preferred
- Complex visual-semantic relationships

**ConSE**:
- Limited training data
- Pre-trained classifiers available
- Interpretable predictions needed

### Performance Comparison

```python
def compare_methods(X_train, y_train, X_test, y_test,
                   seen_classes, unseen_classes, class_embeddings):
    """
    Compare compatibility learning and ConSE.
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create data loaders
    train_data = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(range(len(y_train)))
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    
    visual_dim = X_train.shape[1]
    semantic_dim = len(class_embeddings[seen_classes[0]])
    
    # Method 1: Compatibility Learning
    print("Training Compatibility Model...")
    compat_model = VisualSemanticEmbedding(visual_dim, semantic_dim)
    train_vse_model(compat_model, train_loader, class_embeddings, 
                    seen_classes, epochs=50)
    
    # Method 2: ConSE
    print("\nTraining ConSE Model...")
    conse_model = ConSE(visual_dim, len(seen_classes))
    train_conse_classifier(conse_model, train_loader, seen_classes, epochs=50)
    
    # Evaluate
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    # Compatibility predictions
    compat_model.eval()
    unseen_embs = torch.tensor(
        [class_embeddings[c] for c in unseen_classes],
        dtype=torch.float32
    )
    
    with torch.no_grad():
        v_emb = compat_model.encode_visual(X_test_tensor)
        s_emb = compat_model.encode_semantic(unseen_embs)
        scores = v_emb @ s_emb.T
        compat_preds = torch.argmax(scores, dim=1).numpy()
    
    # ConSE predictions
    conse_preds = conse_model.predict_unseen(
        X_test_tensor, seen_classes, unseen_classes, class_embeddings
    ).numpy()
    
    # Compute accuracies
    compat_acc = (compat_preds == y_test).mean()
    conse_acc = (conse_preds == y_test).mean()
    
    print(f"\nResults:")
    print(f"Compatibility Learning: {compat_acc*100:.2f}%")
    print(f"ConSE: {conse_acc*100:.2f}%")
    print(f"Random Baseline: {100/len(unseen_classes):.2f}%")
```

## Handling Multi-Word and Compound Class Names

### Challenge

Many real classes have compound names:
- "polar bear"
- "fire truck"
- "German shepherd"

### Strategies

**1. Word Averaging**:
$$\mathbf{s}_c = \frac{1}{|words_c|} \sum_{w \in words_c} \mathbf{v}_w$$

**2. Weighted Averaging** (TF-IDF weights):
$$\mathbf{s}_c = \sum_{w \in words_c} \text{tfidf}(w) \cdot \mathbf{v}_w$$

**3. Phrase Embeddings**:
Use phrase-aware models like FastText or train phrase embeddings.

**4. BERT/GPT Embeddings**:
Get contextual embeddings for the full class name.

```python
from transformers import BertTokenizer, BertModel
import torch

def get_bert_embedding(class_name, tokenizer, model):
    """
    Get BERT embedding for a class name.
    """
    # Tokenize
    inputs = tokenizer(class_name, return_tensors='pt', 
                       padding=True, truncation=True)
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        # Use [CLS] token embedding
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    
    return embedding.numpy()
```

## Summary

Semantic embedding methods provide a powerful alternative to attribute-based ZSL:

1. **Pre-trained embeddings** (Word2Vec, GloVe, FastText) capture rich semantic relationships without manual annotation
2. **Compatibility learning** trains end-to-end to align visual and semantic spaces
3. **ConSE** uses seen class probabilities for interpretable zero-shot prediction
4. **Modern approaches** leverage contextual embeddings from BERT/GPT for better semantic representations

The key advantage is scalability: any class with a name can be immediately embedded, enabling zero-shot recognition without additional annotation effort.
