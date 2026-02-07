# CLIP: Contrastive Language-Image Pre-training

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the CLIP architecture and its design principles
2. Derive and implement the InfoNCE contrastive loss function
3. Explain the role of temperature scaling in contrastive learning
4. Implement CLIP-style training with symmetric loss
5. Apply CLIP for zero-shot image classification
6. Analyze the trade-offs between batch size, temperature, and learning dynamics

## Introduction

CLIP (Contrastive Language-Image Pre-training) revolutionized vision-language learning by demonstrating that natural language supervision at scale can produce highly transferable visual representations. Rather than training on fixed label sets, CLIP learns to associate images with free-form text descriptions, enabling remarkable zero-shot transfer to new tasks.

The key insight is that the internet contains hundreds of millions of image-text pairs that can serve as a natural supervision signal. By learning to match images with their captions, CLIP develops rich semantic understanding that generalizes across diverse visual concepts.

## Architecture Overview

CLIP consists of two separate encoders that map images and text to a shared embedding space:

```
┌─────────────────────────────────────────────────────────────┐
│                        CLIP Architecture                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Image ──→ [Vision Encoder] ──→ [Projection] ──→ v_i        │
│                    ↓                                          │
│              ResNet/ViT                                       │
│                                                               │
│  Text  ──→ [Text Encoder]  ──→ [Projection] ──→ t_j         │
│                    ↓                                          │
│           Transformer                                         │
│                                                               │
│            Similarity Matrix: S_ij = v_i · t_j / τ           │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Vision Encoder Options

**ResNet Variants:**
- Modified ResNet-50 with attention pooling
- Anti-aliasing blur pooling
- Uses global average pooling followed by projection

**Vision Transformer (ViT):**
- Patches image into 14×14 or 16×16 tokens
- Adds learnable [CLS] token
- Uses [CLS] embedding as image representation

### Text Encoder

- Transformer architecture with 12 layers
- 512-dimensional embeddings
- Maximum context length of 77 tokens
- Uses [EOS] token embedding as text representation
- Byte Pair Encoding (BPE) tokenization

## Mathematical Foundations

### Contrastive Learning Framework

Given a batch of $N$ image-text pairs $\{(I_i, T_i)\}_{i=1}^N$, we compute:

**Image embeddings:** $v_i = \text{normalize}(f_v(I_i)) \in \mathbb{R}^d$

**Text embeddings:** $t_j = \text{normalize}(f_t(T_j)) \in \mathbb{R}^d$

**Similarity matrix:** $S_{ij} = v_i^T \cdot t_j$

The similarity matrix is an $N \times N$ matrix where:
- Diagonal elements $S_{ii}$ represent positive pairs (matched image-text)
- Off-diagonal elements $S_{ij}, i \neq j$ represent negative pairs (mismatched)

### InfoNCE Loss Derivation

The InfoNCE (Noise Contrastive Estimation) loss treats the matching problem as an $N$-way classification task. For each image, we want to identify its matching text among all $N$ candidates.

**Image-to-Text Loss:**

For image $I_i$, the probability of matching with text $T_j$ is:

$$p(T_j | I_i) = \frac{\exp(S_{ij} / \tau)}{\sum_{k=1}^{N} \exp(S_{ik} / \tau)}$$

where $\tau$ is the temperature parameter.

The loss for image $I_i$ is the negative log-likelihood of the correct match:

$$\mathcal{L}_{i2t}^{(i)} = -\log p(T_i | I_i) = -\log \frac{\exp(S_{ii} / \tau)}{\sum_{k=1}^{N} \exp(S_{ik} / \tau)}$$

**Text-to-Image Loss:**

Similarly, for text $T_j$:

$$\mathcal{L}_{t2i}^{(j)} = -\log p(I_j | T_j) = -\log \frac{\exp(S_{jj} / \tau)}{\sum_{k=1}^{N} \exp(S_{kj} / \tau)}$$

**Symmetric CLIP Loss:**

The total loss averages both directions:

$$\mathcal{L}_{CLIP} = \frac{1}{2N} \left( \sum_{i=1}^{N} \mathcal{L}_{i2t}^{(i)} + \sum_{j=1}^{N} \mathcal{L}_{t2i}^{(j)} \right)$$

### Connection to Cross-Entropy

The InfoNCE loss is equivalent to cross-entropy loss over a softmax distribution. For image-to-text:

$$\mathcal{L}_{i2t} = \text{CrossEntropy}(S / \tau, \text{targets})$$

where targets are the indices $[0, 1, 2, ..., N-1]$ (diagonal positions).

### Temperature Parameter Analysis

The temperature $\tau$ controls the sharpness of the probability distribution:

**Low temperature ($\tau \rightarrow 0$):**
- Distribution approaches one-hot (argmax)
- Model becomes overconfident
- Small differences in similarity become large probability differences
- Can lead to training instability

**High temperature ($\tau \rightarrow \infty$):**
- Distribution approaches uniform
- All candidates appear equally likely
- Weak learning signal
- Slow convergence

**CLIP's choice: $\tau = 0.07$** (or learned, initialized to $1/0.07 \approx 14.29$)

The optimal temperature balances:
1. Sharp enough to distinguish positive from negative pairs
2. Soft enough to maintain stable gradients

### Learnable Temperature

CLIP learns $\tau$ as a parameter by parameterizing it as:

$$\tau = \exp(-\log\_scale)$$

where $\log\_scale$ is learned. This ensures $\tau > 0$ and provides stable gradients.

## PyTorch Implementation

### CLIP Model Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class CLIPModel(nn.Module):
    """
    CLIP-style contrastive vision-language model.
    
    This implementation demonstrates the core CLIP training paradigm:
    1. Separate encoders for vision and language
    2. L2-normalized embeddings in shared space
    3. Learnable temperature parameter
    4. Symmetric contrastive loss
    """
    
    def __init__(self, 
                 image_dim: int = 2048,
                 text_dim: int = 768,
                 embed_dim: int = 512,
                 init_temperature: float = 0.07):
        super().__init__()
        
        # Image encoder (projection from backbone features)
        self.image_encoder = nn.Sequential(
            nn.Linear(image_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Text encoder (projection from backbone features)
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, embed_dim * 2),
            nn.BatchNorm1d(embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Learnable temperature: τ = exp(-logit_scale)
        # Initialize to 1/init_temperature
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(1 / init_temperature)
        )
        
        self.embed_dim = embed_dim
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to normalized embeddings."""
        embeddings = self.image_encoder(images)
        return F.normalize(embeddings, p=2, dim=1)
    
    def encode_text(self, texts: torch.Tensor) -> torch.Tensor:
        """Encode texts to normalized embeddings."""
        embeddings = self.text_encoder(texts)
        return F.normalize(embeddings, p=2, dim=1)
    
    def forward(self, images: torch.Tensor, 
                texts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute image and text embeddings.
        
        Args:
            images: Image features (batch, image_dim)
            texts: Text features (batch, text_dim)
        
        Returns:
            (image_embeddings, text_embeddings), each (batch, embed_dim)
        """
        image_emb = self.encode_image(images)
        text_emb = self.encode_text(texts)
        return image_emb, text_emb
    
    def compute_logits(self, image_emb: torch.Tensor,
                      text_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute temperature-scaled similarity logits.
        
        The logit_scale is clamped to prevent numerical instability.
        """
        # Clamp logit_scale to reasonable range
        logit_scale = self.logit_scale.exp().clamp(max=100)
        
        # Compute similarity matrix scaled by temperature
        # Since embeddings are normalized: dot product = cosine similarity
        logits = torch.matmul(image_emb, text_emb.t()) * logit_scale
        
        return logits
    
    @property
    def temperature(self) -> float:
        """Current temperature value."""
        return 1 / self.logit_scale.exp().item()
```

### Contrastive Loss Implementation

```python
class CLIPLoss(nn.Module):
    """
    Symmetric CLIP contrastive loss (InfoNCE).
    
    Computes cross-entropy loss in both directions:
    - Image-to-text: for each image, classify correct text
    - Text-to-image: for each text, classify correct image
    """
    
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute symmetric contrastive loss.
        
        Args:
            logits: Similarity matrix (batch, batch) with temperature scaling
        
        Returns:
            loss: Scalar loss value
            metrics: Dict with individual loss components
        """
        batch_size = logits.shape[0]
        
        # Targets are diagonal indices [0, 1, 2, ..., N-1]
        targets = torch.arange(batch_size, device=logits.device)
        
        # Image-to-text: each row is a classification problem
        # logits[i] = similarities of image i to all texts
        loss_i2t = self.cross_entropy(logits, targets)
        
        # Text-to-image: each column is a classification problem
        # logits.t()[j] = similarities of text j to all images
        loss_t2i = self.cross_entropy(logits.t(), targets)
        
        # Symmetric loss
        loss = (loss_i2t + loss_t2i) / 2
        
        metrics = {
            'loss': loss.item(),
            'i2t_loss': loss_i2t.item(),
            't2i_loss': loss_t2i.item()
        }
        
        return loss, metrics


def compute_accuracy(logits: torch.Tensor) -> dict:
    """
    Compute matching accuracy for both directions.
    
    Accuracy measures how often the correct match has highest similarity.
    """
    batch_size = logits.shape[0]
    targets = torch.arange(batch_size, device=logits.device)
    
    # Image-to-text accuracy
    i2t_preds = logits.argmax(dim=1)
    i2t_acc = (i2t_preds == targets).float().mean().item()
    
    # Text-to-image accuracy
    t2i_preds = logits.argmax(dim=0)
    t2i_acc = (t2i_preds == targets).float().mean().item()
    
    return {
        'i2t_accuracy': i2t_acc,
        't2i_accuracy': t2i_acc,
        'mean_accuracy': (i2t_acc + t2i_acc) / 2
    }
```

### Training Loop

```python
def train_clip(model: CLIPModel,
               train_loader: torch.utils.data.DataLoader,
               num_epochs: int = 30,
               learning_rate: float = 3e-4,
               warmup_epochs: int = 2,
               device: str = 'cuda') -> dict:
    """
    Train CLIP model with symmetric contrastive loss.
    
    Key training details:
    - AdamW optimizer with weight decay
    - Cosine learning rate schedule with warmup
    - Gradient clipping for stability
    """
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1,
        betas=(0.9, 0.98)  # CLIP uses different betas
    )
    
    # Cosine schedule with warmup
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = CLIPLoss()
    
    history = {'loss': [], 'accuracy': [], 'temperature': [], 'lr': []}
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0
        
        for images, texts in train_loader:
            images = images.to(device)
            texts = texts.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            image_emb, text_emb = model(images, texts)
            logits = model.compute_logits(image_emb, text_emb)
            
            # Compute loss
            loss, loss_metrics = criterion(logits)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            epoch_loss += loss.item()
            acc_metrics = compute_accuracy(logits)
            epoch_acc += acc_metrics['mean_accuracy']
            num_batches += 1
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        
        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_acc)
        history['temperature'].append(model.temperature)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}")
        print(f"  Temperature: {model.temperature:.4f}")
    
    return history
```

## Zero-Shot Classification

One of CLIP's most powerful capabilities is zero-shot image classification using natural language prompts.

### How It Works

1. Create text descriptions for each class (e.g., "a photo of a {class}")
2. Encode all class descriptions with text encoder
3. Encode the input image with image encoder
4. Compute similarities between image and all class descriptions
5. Apply softmax to get class probabilities

```python
class ZeroShotClassifier:
    """
    Zero-shot image classifier using CLIP embeddings.
    """
    
    def __init__(self, model: CLIPModel, 
                 class_names: list,
                 templates: list = None):
        """
        Args:
            model: Trained CLIP model
            class_names: List of class names
            templates: Prompt templates (default: "a photo of a {}")
        """
        self.model = model
        self.class_names = class_names
        
        if templates is None:
            templates = [
                "a photo of a {}",
                "a picture of a {}",
                "an image of a {}",
            ]
        self.templates = templates
    
    @torch.no_grad()
    def build_classifier(self, text_encoder_fn) -> torch.Tensor:
        """
        Build zero-shot classifier weights from class descriptions.
        
        Uses multiple templates and averages embeddings for robustness.
        
        Args:
            text_encoder_fn: Function to encode text string to features
        
        Returns:
            Class embeddings (num_classes, embed_dim)
        """
        class_embeddings = []
        
        for class_name in self.class_names:
            # Get embeddings for all templates
            template_embeddings = []
            for template in self.templates:
                text = template.format(class_name)
                text_features = text_encoder_fn(text)
                text_emb = self.model.encode_text(text_features)
                template_embeddings.append(text_emb)
            
            # Average across templates
            class_emb = torch.stack(template_embeddings).mean(dim=0)
            # Re-normalize after averaging
            class_emb = F.normalize(class_emb, p=2, dim=-1)
            class_embeddings.append(class_emb)
        
        return torch.cat(class_embeddings, dim=0)
    
    @torch.no_grad()
    def classify(self, image_features: torch.Tensor,
                class_embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify images using precomputed class embeddings.
        
        Args:
            image_features: Image features (batch, image_dim)
            class_embeddings: Class text embeddings (num_classes, embed_dim)
        
        Returns:
            predictions: Predicted class indices (batch,)
            probabilities: Class probabilities (batch, num_classes)
        """
        # Encode images
        image_emb = self.model.encode_image(image_features)
        
        # Compute similarities
        logits = self.model.compute_logits(image_emb, class_embeddings)
        
        # Convert to probabilities
        probabilities = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        
        return predictions, probabilities
```

### Prompt Engineering for Zero-Shot

The choice of prompt template significantly affects zero-shot performance:

```python
# Different prompt strategies

# Simple template
templates_simple = ["a photo of a {}"]

# Multiple templates (ensemble)
templates_ensemble = [
    "a photo of a {}",
    "a picture of a {}",
    "an image showing a {}",
    "a photograph of a {}",
    "{} in a photo",
]

# Domain-specific templates
templates_imagenet = [
    "a photo of a {}",
    "a bad photo of a {}",
    "a photo of many {}",
    "a sculpture of a {}",
    "a photo of the hard to see {}",
    "a low resolution photo of the {}",
    "a rendering of a {}",
    "graffiti of a {}",
    "a tattoo of a {}",
    "the embroidered {}",
]

# Task-specific templates
templates_action = [
    "a photo of a person {}",  # e.g., "running", "jumping"
    "someone {} in a photo",
]
```

## Batch Size and Negative Samples

Contrastive learning benefits significantly from large batch sizes:

### Why Larger Batches Help

With batch size $N$, each positive pair has $N-1$ negative pairs. More negatives provide:

1. **Better gradient signal**: More contrast to learn from
2. **Harder negatives**: Larger batch more likely to contain challenging negatives
3. **More stable training**: Averaged over more samples

### Effective Negatives

The effective number of negatives scales with batch size:
- Batch size 32: 31 negatives per positive
- Batch size 256: 255 negatives per positive
- Batch size 32768 (CLIP): 32767 negatives per positive

### Gradient Accumulation for Large Effective Batches

```python
def train_with_gradient_accumulation(model, train_loader, 
                                    accumulation_steps: int = 8):
    """
    Simulate large batch training with gradient accumulation.
    
    Effective batch size = batch_size * accumulation_steps
    """
    optimizer.zero_grad()
    
    for i, (images, texts) in enumerate(train_loader):
        # Forward pass
        image_emb, text_emb = model(images, texts)
        logits = model.compute_logits(image_emb, text_emb)
        
        loss, _ = criterion(logits)
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()
        
        # Update weights every accumulation_steps batches
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

## Evaluation and Metrics

### Standard Benchmarks

CLIP is typically evaluated on:

1. **Zero-shot ImageNet**: Classification without training on ImageNet
2. **Retrieval benchmarks**: MS-COCO, Flickr30K
3. **Transfer learning**: Fine-tuning on downstream tasks

### Evaluation Code

```python
@torch.no_grad()
def evaluate_clip(model: CLIPModel,
                 test_loader: torch.utils.data.DataLoader,
                 device: str = 'cuda') -> dict:
    """
    Comprehensive CLIP evaluation.
    """
    model.eval()
    
    all_image_emb = []
    all_text_emb = []
    all_labels = []
    
    for images, texts, labels in test_loader:
        images = images.to(device)
        texts = texts.to(device)
        
        image_emb, text_emb = model(images, texts)
        
        all_image_emb.append(image_emb.cpu())
        all_text_emb.append(text_emb.cpu())
        all_labels.append(labels)
    
    image_emb = torch.cat(all_image_emb, dim=0)
    text_emb = torch.cat(all_text_emb, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Compute full similarity matrix
    similarity = torch.matmul(image_emb, text_emb.t())
    
    metrics = {}
    
    # Retrieval metrics
    for direction, name in [('i2t', 'image_to_text'), ('t2i', 'text_to_image')]:
        sim = similarity if direction == 'i2t' else similarity.t()
        
        for k in [1, 5, 10]:
            recall = compute_recall_at_k(sim, k)
            metrics[f'{name}_recall@{k}'] = recall
    
    return metrics


def compute_recall_at_k(similarity: torch.Tensor, k: int) -> float:
    """Compute Recall@K for retrieval."""
    n = similarity.shape[0]
    ranks = []
    
    for i in range(n):
        sorted_indices = torch.argsort(similarity[i], descending=True)
        rank = (sorted_indices == i).nonzero().item() + 1
        ranks.append(rank)
    
    recall = sum(1 for r in ranks if r <= k) / n
    return recall
```

## Key Insights and Best Practices

### Training Tips

1. **Large batch sizes**: Use gradient accumulation if memory-limited
2. **Temperature learning**: Initialize around 0.07, let model learn optimal value
3. **Warmup**: Essential for stable training with large learning rates
4. **Mixed precision**: Enable for faster training and larger batches
5. **Data augmentation**: Random crops, color jitter for images

### Common Pitfalls

1. **Forgetting L2 normalization**: Critical for cosine similarity
2. **Temperature too low**: Causes gradient vanishing
3. **Small batches**: Insufficient negatives for good contrastive learning
4. **Mismatched encoders**: Ensure both encoders output same dimension

### Scaling Considerations

| Component | CLIP-Base | CLIP-Large |
|-----------|-----------|------------|
| Image Encoder | ResNet-50 | ViT-L/14 |
| Text Encoder | 63M params | 123M params |
| Embed Dim | 512 | 768 |
| Batch Size | 32,768 | 32,768 |
| Training Data | 400M pairs | 400M pairs |

## Summary

CLIP demonstrates that:

1. **Natural language supervision** scales better than fixed label sets
2. **Contrastive learning** creates rich, transferable representations
3. **Temperature scaling** is critical for training dynamics
4. **Zero-shot transfer** emerges from large-scale multimodal pretraining
5. **Batch size** directly impacts contrastive learning quality

## References

1. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
2. van den Oord, A., et al. "Representation Learning with Contrastive Predictive Coding." arXiv 2018.
3. Chen, T., et al. "A Simple Framework for Contrastive Learning of Visual Representations." ICML 2020.
