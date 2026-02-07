# Zero-Shot Learning Fundamentals

## Problem Formulation

Zero-shot learning addresses a fundamental limitation of traditional machine learning: the inability to recognize classes not seen during training. This section formalizes the ZSL problem and establishes the mathematical foundations for all ZSL methods.

### Formal Definition

Let $\mathcal{X}$ denote the input space (e.g., images) and $\mathcal{Y}$ the label space. In ZSL, the label space is partitioned into:

- **Seen classes**: $\mathcal{Y}^s = \{y_1^s, y_2^s, \ldots, y_K^s\}$ — classes with training examples
- **Unseen classes**: $\mathcal{Y}^u = \{y_1^u, y_2^u, \ldots, y_L^u\}$ — classes without training examples

The fundamental constraint is:

$$\mathcal{Y}^s \cap \mathcal{Y}^u = \emptyset$$

### Training and Testing Protocols

**Training Phase:**
- Access to labeled data: $\mathcal{D}^{tr} = \{(\mathbf{x}_i, y_i) : y_i \in \mathcal{Y}^s\}_{i=1}^{N}$
- Access to semantic descriptions for all classes: $\{\mathbf{s}_c\}_{c \in \mathcal{Y}^s \cup \mathcal{Y}^u}$

**Testing Phase (Conventional ZSL):**
- Test instances belong only to unseen classes
- Prediction space: $\mathcal{Y}^u$

**Testing Phase (Generalized ZSL):**
- Test instances may belong to any class
- Prediction space: $\mathcal{Y}^s \cup \mathcal{Y}^u$

## The Semantic Space

The key to ZSL is the semantic space $\mathcal{S}$ that provides a common representation for both seen and unseen classes.

### Attribute-Based Representations

Each class $c$ is described by an attribute vector $\mathbf{a}_c \in \mathbb{R}^M$ where $M$ is the number of attributes:

$$\mathbf{a}_c = [a_c^{(1)}, a_c^{(2)}, \ldots, a_c^{(M)}]$$

Attributes can be:
- **Binary**: $a_c^{(m)} \in \{0, 1\}$ indicating presence/absence
- **Continuous**: $a_c^{(m)} \in [0, 1]$ indicating strength/relevance

**Example: Animal Attributes**

| Animal | has_fur | has_stripes | is_large | has_4_legs | can_fly |
|--------|---------|-------------|----------|------------|---------|
| Dog | 1 | 0 | 0.3 | 1 | 0 |
| Cat | 1 | 0 | 0.1 | 1 | 0 |
| Zebra | 1 | 1 | 0.9 | 1 | 0 |
| Eagle | 0 | 0 | 0.6 | 0 | 1 |

### Word Embedding Representations

Class names are mapped to continuous vector spaces using pre-trained language models:

$$\mathbf{s}_c = \text{Embed}(\text{classname}_c) \in \mathbb{R}^D$$

Common embedding methods:
- Word2Vec (Skip-gram, CBOW)
- GloVe (Global Vectors)
- FastText (subword information)
- BERT, GPT embeddings

**Advantages over attributes:**
1. No manual annotation required
2. Capture rich semantic relationships automatically
3. Pre-trained on massive text corpora
4. Transfer linguistic knowledge to visual domain

### Semantic Similarity

The semantic space induces similarity relationships:

$$\text{sim}(c_1, c_2) = f(\mathbf{s}_{c_1}, \mathbf{s}_{c_2})$$

Common similarity functions:

**Cosine Similarity:**
$$\text{sim}_{\cos}(\mathbf{s}_1, \mathbf{s}_2) = \frac{\mathbf{s}_1 \cdot \mathbf{s}_2}{\|\mathbf{s}_1\| \|\mathbf{s}_2\|}$$

**Euclidean Distance (converted to similarity):**
$$\text{sim}_{\text{euc}}(\mathbf{s}_1, \mathbf{s}_2) = \exp(-\|\mathbf{s}_1 - \mathbf{s}_2\|^2)$$

**Dot Product:**
$$\text{sim}_{\text{dot}}(\mathbf{s}_1, \mathbf{s}_2) = \mathbf{s}_1^\top \mathbf{s}_2$$

## Compatibility Functions

The compatibility function $F: \mathcal{V} \times \mathcal{S} \rightarrow \mathbb{R}$ measures how well visual features align with semantic representations.

### Linear Compatibility

$$F(\mathbf{v}, \mathbf{s}) = \mathbf{v}^\top W \mathbf{s}$$

where $W \in \mathbb{R}^{d_v \times d_s}$ is a learned projection matrix.

### Bilinear Compatibility

For richer interactions:

$$F(\mathbf{v}, \mathbf{s}) = \mathbf{v}^\top W \mathbf{s}$$

where $W$ captures all pairwise interactions between visual and semantic dimensions.

### Neural Compatibility

Using neural networks for non-linear compatibility:

$$F(\mathbf{v}, \mathbf{s}) = g_\theta(\mathbf{v}, \mathbf{s})$$

Common architectures:
- Concatenation followed by MLP
- Cross-attention mechanisms
- Residual compatibility networks

### Embedding-Based Compatibility

Project both modalities to a common space:

$$F(\mathbf{v}, \mathbf{s}) = f_\phi(\mathbf{v})^\top g_\psi(\mathbf{s})$$

Or using cosine similarity:

$$F(\mathbf{v}, \mathbf{s}) = \frac{f_\phi(\mathbf{v}) \cdot g_\psi(\mathbf{s})}{\|f_\phi(\mathbf{v})\| \|g_\psi(\mathbf{s})\|}$$

## Loss Functions for ZSL

### Cross-Entropy Loss

Treat ZSL as classification over seen classes:

$$\mathcal{L}_{CE} = -\sum_{(\mathbf{x}, y) \in \mathcal{D}^{tr}} \log \frac{\exp(F(\phi(\mathbf{x}), \mathbf{s}_y))}{\sum_{c \in \mathcal{Y}^s} \exp(F(\phi(\mathbf{x}), \mathbf{s}_c))}$$

### Ranking Loss (Hinge Loss)

Encourage correct class to have higher compatibility than incorrect classes:

$$\mathcal{L}_{rank} = \sum_{(\mathbf{x}, y)} \sum_{c \neq y} \max(0, \Delta + F(\phi(\mathbf{x}), \mathbf{s}_c) - F(\phi(\mathbf{x}), \mathbf{s}_y))$$

where $\Delta$ is a margin parameter.

### Triplet Loss

Directly optimize embedding distances:

$$\mathcal{L}_{triplet} = \sum_{(\mathbf{x}, y)} \max(0, \|f(\mathbf{x}) - \mathbf{s}_y\|^2 - \|f(\mathbf{x}) - \mathbf{s}_{y^-}\|^2 + \alpha)$$

where $y^-$ is a negative class and $\alpha$ is the margin.

### Contrastive Loss

For paired similarity learning:

$$\mathcal{L}_{cont} = \sum_{i,j} y_{ij} D_{ij}^2 + (1 - y_{ij}) \max(0, m - D_{ij})^2$$

where $y_{ij} = 1$ if pairs are from the same class, 0 otherwise.

## Visual Feature Extraction

Modern ZSL methods use pre-trained CNN features:

### CNN Backbones

| Model | Output Dim | Pre-training | Common Layer |
|-------|------------|--------------|--------------|
| VGG-19 | 4096 | ImageNet | fc7 |
| ResNet-101 | 2048 | ImageNet | avg_pool |
| Inception-v3 | 2048 | ImageNet | pool3 |
| CLIP | 512/768 | 400M pairs | final |

### Feature Extraction Pipeline

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# Load pre-trained ResNet
resnet = models.resnet101(pretrained=True)
# Remove classification head
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])
feature_extractor.eval()

# Standard preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

# Extract features
def extract_features(image):
    with torch.no_grad():
        x = preprocess(image).unsqueeze(0)
        features = feature_extractor(x)
        return features.squeeze()  # Shape: (2048,)
```

## Prediction at Test Time

### Nearest Neighbor in Semantic Space

For a test instance $\mathbf{x}$ with visual features $\mathbf{v} = \phi(\mathbf{x})$:

1. Compute projected features or compatibility scores
2. Find the most compatible class

**Conventional ZSL:**
$$\hat{y} = \arg\max_{c \in \mathcal{Y}^u} F(\mathbf{v}, \mathbf{s}_c)$$

**Generalized ZSL:**
$$\hat{y} = \arg\max_{c \in \mathcal{Y}^s \cup \mathcal{Y}^u} F(\mathbf{v}, \mathbf{s}_c)$$

### Implementation

```python
import numpy as np

def predict_zsl(visual_features, class_embeddings, class_names, 
                compatibility_fn='cosine'):
    """
    Predict class using zero-shot learning.
    
    Args:
        visual_features: (n_samples, d_v) visual feature matrix
        class_embeddings: (n_classes, d_s) semantic embedding matrix
        class_names: list of class names
        compatibility_fn: 'cosine', 'dot', or 'euclidean'
    
    Returns:
        predictions: list of predicted class names
    """
    if compatibility_fn == 'cosine':
        # Normalize for cosine similarity
        v_norm = visual_features / np.linalg.norm(visual_features, axis=1, keepdims=True)
        s_norm = class_embeddings / np.linalg.norm(class_embeddings, axis=1, keepdims=True)
        scores = v_norm @ s_norm.T
    elif compatibility_fn == 'dot':
        scores = visual_features @ class_embeddings.T
    elif compatibility_fn == 'euclidean':
        # Negative distance as score
        scores = -np.sum((visual_features[:, None, :] - class_embeddings[None, :, :]) ** 2, axis=2)
    
    pred_indices = np.argmax(scores, axis=1)
    predictions = [class_names[idx] for idx in pred_indices]
    
    return predictions, scores
```

## Key Challenges

### Domain Shift

Visual features of seen classes may have different distributions than unseen classes:

- **Intra-class variance**: Different instances of the same class look different
- **Inter-class similarity**: Unseen classes may be more similar to each other than to seen classes
- **Projection domain shift**: Learned mappings may not generalize well

### Hubness Problem

In high-dimensional spaces, certain points (hubs) tend to be nearest neighbors of many other points:

$$\text{Hubness}(c) = \sum_{\mathbf{x} \in \text{test}} \mathbb{1}[\text{NN}(\mathbf{x}) = c]$$

This causes some classes to be predicted too frequently.

**Mitigation strategies:**
- Use local scaling (CSLS)
- Apply hubness reduction techniques
- Use regularization in embedding learning

### Semantic Gap

The semantic space may not perfectly capture visual distinctions:

- Visually similar classes may have different semantic representations
- Visually different classes may have similar semantic representations
- Fine-grained distinctions may not be captured by attributes or word embeddings

### Bias in GZSL

Models trained only on seen classes exhibit strong bias toward predicting seen classes:

- Seen class features are in-distribution
- Compatibility scores naturally higher for seen classes
- Unseen classes rarely predicted

This is addressed through calibration and balancing techniques covered in later sections.

## Summary

The ZSL framework consists of:

1. **Input space** $\mathcal{X}$: Images or other data modalities
2. **Visual encoder** $\phi$: Maps inputs to visual features
3. **Semantic space** $\mathcal{S}$: Provides class representations
4. **Compatibility function** $F$: Measures visual-semantic alignment
5. **Prediction rule**: Selects class with highest compatibility

The key insight enabling ZSL is that semantic representations bridge seen and unseen classes, allowing knowledge transfer through shared attributes or embedding similarities.
