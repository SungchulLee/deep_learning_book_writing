# Attribute-Based Zero-Shot Learning

## Overview

Attribute-based methods represent the foundational approach to zero-shot learning. Each class is described by a set of semantic attributes, and models learn to predict these attributes from visual features. This section covers the two primary attribute-based approaches: Direct Attribute Prediction (DAP) and Indirect Attribute Prediction (IAP).

## Attribute Representation

### Attribute Definition

An attribute is a semantic property that can describe multiple classes. For each class $c$, we have an attribute vector:

$$\mathbf{a}_c = [a_c^{(1)}, a_c^{(2)}, \ldots, a_c^{(M)}]$$

where $M$ is the number of attributes and $a_c^{(m)}$ indicates the presence/strength of attribute $m$ for class $c$.

### Types of Attributes

**Binary Attributes**: $a_c^{(m)} \in \{0, 1\}$
- Presence/absence of a property
- Example: "has_stripes", "can_fly", "has_4_legs"

**Continuous Attributes**: $a_c^{(m)} \in [0, 1]$ or $\mathbb{R}$
- Degree or strength of a property
- Example: "size" = 0.9 for elephant, 0.1 for mouse

**Relative Attributes**: Comparative values
- Example: "larger_than_cat", "faster_than_tortoise"

### Attribute Matrix

The complete attribute information is captured in an attribute matrix:

$$A \in \mathbb{R}^{C \times M}$$

where $C = |\mathcal{Y}^s \cup \mathcal{Y}^u|$ is the total number of classes.

**Example Attribute Matrix:**

| Class | has_fur | has_feathers | has_scales | 4_legs | can_fly | is_large | carnivore |
|-------|---------|--------------|------------|--------|---------|----------|-----------|
| dog | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.3 | 0.8 |
| cat | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.1 | 1.0 |
| bird | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.1 | 0.3 |
| horse | 1.0 | 0.0 | 0.0 | 1.0 | 0.0 | 0.9 | 0.0 |
| eagle | 0.0 | 1.0 | 0.0 | 0.0 | 1.0 | 0.6 | 1.0 |

## Direct Attribute Prediction (DAP)

### Concept

DAP treats each attribute as an independent binary classification problem. Given an image, we predict the probability of each attribute being present, then compare these predictions to known class attribute signatures.

### Mathematical Formulation

**Step 1: Learn Attribute Classifiers**

For each attribute $m = 1, \ldots, M$, learn a classifier:

$$P(a_m = 1 | \mathbf{x}) = \sigma(w_m^\top \phi(\mathbf{x}) + b_m)$$

where $\sigma$ is the sigmoid function and $\phi(\mathbf{x})$ extracts visual features.

**Step 2: Predict Attributes**

For a test image $\mathbf{x}$, predict all attributes:

$$\hat{\mathbf{a}}(\mathbf{x}) = [P(a_1 | \mathbf{x}), P(a_2 | \mathbf{x}), \ldots, P(a_M | \mathbf{x})]$$

**Step 3: Compute Class Probabilities**

Assuming attributes are conditionally independent given the class:

$$P(c | \mathbf{x}) \propto P(c) \prod_{m=1}^{M} P(a_m | \mathbf{x})^{a_c^{(m)}} \cdot (1 - P(a_m | \mathbf{x}))^{1 - a_c^{(m)}}$$

**Step 4: Predict Class**

$$\hat{y} = \arg\max_{c \in \mathcal{Y}^u} P(c | \mathbf{x})$$

### Log-Space Formulation

For numerical stability, work in log-space:

$$\log P(c | \mathbf{x}) = \text{const} + \sum_{m=1}^{M} \left[ a_c^{(m)} \log P(a_m | \mathbf{x}) + (1 - a_c^{(m)}) \log(1 - P(a_m | \mathbf{x})) \right]$$

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DirectAttributePrediction(nn.Module):
    """
    Direct Attribute Prediction (DAP) for Zero-Shot Learning.
    
    Learns independent attribute classifiers and uses them to predict
    unseen classes via attribute matching.
    """
    
    def __init__(self, visual_dim: int, n_attributes: int, hidden_dim: int = 512):
        super().__init__()
        
        self.n_attributes = n_attributes
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Attribute predictors (one per attribute)
        self.attribute_heads = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(n_attributes)
        ])
    
    def forward(self, x):
        """
        Predict attribute probabilities.
        
        Args:
            x: Visual features of shape (batch_size, visual_dim)
        
        Returns:
            Attribute probabilities of shape (batch_size, n_attributes)
        """
        features = self.feature_extractor(x)
        
        # Predict each attribute
        attr_logits = [head(features) for head in self.attribute_heads]
        attr_logits = torch.cat(attr_logits, dim=1)  # (batch, n_attributes)
        
        return torch.sigmoid(attr_logits)
    
    def predict_class(self, x, class_attributes, class_names):
        """
        Predict class using DAP.
        
        Args:
            x: Visual features (batch_size, visual_dim)
            class_attributes: Dict mapping class names to attribute vectors
            class_names: List of candidate class names
        
        Returns:
            Predicted class indices
        """
        # Get attribute probabilities
        attr_probs = self.forward(x)  # (batch, n_attributes)
        
        # Build attribute matrix for candidate classes
        attr_matrix = torch.stack([
            torch.tensor(class_attributes[c], dtype=torch.float32)
            for c in class_names
        ])  # (n_classes, n_attributes)
        
        # Compute log-probabilities for each class
        # P(c|x) ∝ ∏_m P(a_m|x)^a_c^m * (1-P(a_m|x))^(1-a_c^m)
        log_probs = attr_probs.unsqueeze(1)  # (batch, 1, n_attrs)
        attr_matrix = attr_matrix.unsqueeze(0)  # (1, n_classes, n_attrs)
        
        eps = 1e-7  # Numerical stability
        log_scores = (
            attr_matrix * torch.log(log_probs + eps) +
            (1 - attr_matrix) * torch.log(1 - log_probs + eps)
        )
        class_scores = log_scores.sum(dim=2)  # (batch, n_classes)
        
        return torch.argmax(class_scores, dim=1)


def train_dap_model(model, train_loader, class_attributes, epochs=50, lr=0.001):
    """
    Train DAP model with binary cross-entropy loss.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            # Get target attributes for each sample
            batch_attrs = torch.stack([
                torch.tensor(class_attributes[y], dtype=torch.float32)
                for y in batch_y
            ])
            
            # Forward pass
            pred_attrs = model(batch_x)
            loss = criterion(pred_attrs, batch_attrs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
```

## Indirect Attribute Prediction (IAP)

### Concept

IAP takes a different approach: instead of predicting attributes directly, it first learns a classifier for seen classes, then uses attribute similarity to transfer predictions to unseen classes.

### Mathematical Formulation

**Step 1: Learn Seen Class Classifier**

Train a standard classifier for seen classes:

$$P(y^s | \mathbf{x}) = \text{softmax}(W^\top \phi(\mathbf{x})) \quad \text{for } y^s \in \mathcal{Y}^s$$

**Step 2: Transfer via Attribute Similarity**

For an unseen class $c^u$, compute:

$$P(c^u | \mathbf{x}) \propto \sum_{c^s \in \mathcal{Y}^s} P(c^s | \mathbf{x}) \cdot \text{sim}(\mathbf{a}_{c^s}, \mathbf{a}_{c^u})$$

**Similarity Functions:**

Cosine similarity:
$$\text{sim}(\mathbf{a}_1, \mathbf{a}_2) = \frac{\mathbf{a}_1 \cdot \mathbf{a}_2}{\|\mathbf{a}_1\| \|\mathbf{a}_2\|}$$

Exponential of negative distance:
$$\text{sim}(\mathbf{a}_1, \mathbf{a}_2) = \exp(-\gamma \|\mathbf{a}_1 - \mathbf{a}_2\|^2)$$

**Step 3: Predict Class**

$$\hat{y} = \arg\max_{c \in \mathcal{Y}^u} P(c | \mathbf{x})$$

### Intuition

IAP leverages the correlation between seen class predictions and unseen class attributes:

- If an image looks like a "dog" (seen class)
- And "horse" (unseen) is similar to "dog" in attribute space
- Then the image likely contains a "horse"

### PyTorch Implementation

```python
class IndirectAttributePrediction(nn.Module):
    """
    Indirect Attribute Prediction (IAP) for Zero-Shot Learning.
    
    Learns a seen class classifier and transfers to unseen classes
    via attribute similarity.
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
        """
        Compute seen class logits.
        """
        return self.classifier(x)
    
    def predict_class(self, x, seen_classes, unseen_classes, class_attributes,
                     similarity='cosine', temperature=1.0):
        """
        Predict unseen class using IAP.
        
        Args:
            x: Visual features (batch_size, visual_dim)
            seen_classes: List of seen class names
            unseen_classes: List of unseen class names
            class_attributes: Dict mapping class names to attribute vectors
            similarity: 'cosine' or 'euclidean'
            temperature: Softmax temperature
        """
        # Get seen class probabilities
        logits = self.forward(x)
        seen_probs = F.softmax(logits / temperature, dim=1)  # (batch, n_seen)
        
        # Build attribute matrices
        seen_attrs = torch.stack([
            torch.tensor(class_attributes[c], dtype=torch.float32)
            for c in seen_classes
        ])  # (n_seen, n_attrs)
        
        unseen_attrs = torch.stack([
            torch.tensor(class_attributes[c], dtype=torch.float32)
            for c in unseen_classes
        ])  # (n_unseen, n_attrs)
        
        # Compute similarity matrix
        if similarity == 'cosine':
            seen_norm = F.normalize(seen_attrs, dim=1)
            unseen_norm = F.normalize(unseen_attrs, dim=1)
            sim_matrix = seen_norm @ unseen_norm.T  # (n_seen, n_unseen)
        else:  # euclidean
            # Negative squared distance
            diff = seen_attrs.unsqueeze(1) - unseen_attrs.unsqueeze(0)
            sim_matrix = -torch.sum(diff ** 2, dim=2)
            sim_matrix = torch.exp(sim_matrix)  # Convert to similarity
        
        # Transfer probabilities: P(unseen|x) = sum_s P(seen_s|x) * sim(seen_s, unseen)
        unseen_scores = seen_probs @ sim_matrix  # (batch, n_unseen)
        
        return torch.argmax(unseen_scores, dim=1)


def train_iap_model(model, train_loader, seen_classes, epochs=50, lr=0.001):
    """
    Train IAP model with cross-entropy loss on seen classes.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Create label mapping
    class_to_idx = {c: i for i, c in enumerate(seen_classes)}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            # Convert labels to indices
            batch_labels = torch.tensor([class_to_idx[y] for y in batch_y])
            
            # Forward pass
            logits = model(batch_x)
            loss = criterion(logits, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += (predicted == batch_labels).sum().item()
            total += batch_labels.size(0)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Loss: {total_loss/len(train_loader):.4f}, "
                  f"Acc: {100*correct/total:.2f}%")
```

## Comparison: DAP vs IAP

### Theoretical Differences

| Aspect | DAP | IAP |
|--------|-----|-----|
| **Learning objective** | Predict attributes | Classify seen classes |
| **Independence assumption** | Attributes are independent | No independence assumption |
| **Transfer mechanism** | Attribute matching | Attribute similarity |
| **Seen class knowledge** | Only via attributes | Full classification |

### Advantages and Disadvantages

**DAP Advantages:**
- Interpretable intermediate predictions
- Works well when attributes are truly independent
- Can diagnose errors via attribute analysis

**DAP Disadvantages:**
- Independence assumption often violated
- Error propagation from attribute classifiers
- Requires per-attribute training

**IAP Advantages:**
- No independence assumption
- Leverages correlations in seen class distributions
- Single classifier training

**IAP Disadvantages:**
- Depends heavily on seen-unseen attribute overlap
- May not transfer well with limited similarity
- Less interpretable

### Empirical Performance

```python
def compare_dap_iap(X_train, y_train, X_test, y_test, 
                    seen_classes, unseen_classes, class_attributes):
    """
    Compare DAP and IAP on the same dataset.
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(range(len(y_train)))
    )
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize models
    n_attributes = len(class_attributes[seen_classes[0]])
    dap_model = DirectAttributePrediction(X_train.shape[1], n_attributes)
    iap_model = IndirectAttributePrediction(X_train.shape[1], len(seen_classes))
    
    # Train both models
    print("Training DAP...")
    train_dap_model(dap_model, train_loader, class_attributes, epochs=50)
    
    print("\nTraining IAP...")
    train_iap_model(iap_model, train_loader, seen_classes, epochs=50)
    
    # Evaluate
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    
    dap_preds = dap_model.predict_class(X_test_tensor, class_attributes, unseen_classes)
    iap_preds = iap_model.predict_class(X_test_tensor, seen_classes, unseen_classes, 
                                        class_attributes)
    
    dap_acc = (dap_preds.numpy() == y_test).mean()
    iap_acc = (iap_preds.numpy() == y_test).mean()
    
    print(f"\nResults:")
    print(f"DAP Accuracy: {dap_acc*100:.2f}%")
    print(f"IAP Accuracy: {iap_acc*100:.2f}%")
    print(f"Random Baseline: {100/len(unseen_classes):.2f}%")
```

## Attribute Quality and Selection

### Attribute Design Principles

**Discriminative**: Attributes should distinguish between classes
- Bad: "has_atoms" (true for all objects)
- Good: "has_stripes" (distinguishes zebra from horse)

**Detectable**: Attributes should be visually recognizable
- Bad: "born_in_Africa" (not visible)
- Good: "black_and_white" (visible pattern)

**Semantic**: Attributes should capture meaningful properties
- Bad: "feature_42_positive" (arbitrary)
- Good: "is_carnivore" (meaningful biological property)

### Attribute Statistics

Analyze attribute quality:

```python
def analyze_attributes(class_attributes, seen_classes, unseen_classes):
    """
    Analyze attribute discriminativeness and coverage.
    """
    import numpy as np
    
    seen_attrs = np.array([class_attributes[c] for c in seen_classes])
    unseen_attrs = np.array([class_attributes[c] for c in unseen_classes])
    
    # Per-attribute analysis
    n_attrs = seen_attrs.shape[1]
    
    print("Attribute Analysis:")
    print("-" * 50)
    
    for i in range(n_attrs):
        seen_var = np.var(seen_attrs[:, i])
        unseen_var = np.var(unseen_attrs[:, i])
        mean_diff = abs(seen_attrs[:, i].mean() - unseen_attrs[:, i].mean())
        
        print(f"Attribute {i}: "
              f"Seen var={seen_var:.3f}, "
              f"Unseen var={unseen_var:.3f}, "
              f"Mean diff={mean_diff:.3f}")
    
    # Overall coverage: how much of unseen attributes are covered by seen
    similarity_matrix = unseen_attrs @ seen_attrs.T
    max_similarities = similarity_matrix.max(axis=1)
    
    print(f"\nUnseen class coverage (max similarity to seen):")
    for i, c in enumerate(unseen_classes):
        print(f"  {c}: {max_similarities[i]:.3f}")
```

## Summary

Attribute-based methods provide the foundational approach to zero-shot learning:

1. **DAP** learns independent attribute predictors and matches predictions to class signatures
2. **IAP** learns seen class classifiers and transfers via attribute similarity
3. Both methods require carefully designed, discriminative attributes
4. The choice between DAP and IAP depends on attribute independence assumptions and seen-unseen attribute overlap

While attribute-based methods are interpretable and effective, they require expensive manual annotation. The next section covers semantic embedding approaches that eliminate this requirement by using pre-trained word embeddings.
