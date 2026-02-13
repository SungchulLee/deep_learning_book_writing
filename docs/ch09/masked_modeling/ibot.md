# iBOT: Image BERT Pre-training with Online Tokenizer

## Introduction

iBOT (Image BERT Pre-training with Online Tokenizer) extends masked image modeling principles through an innovative framework that learns to discretize visual information while simultaneously predicting masked patches. By jointly training a tokenizer and mask prediction model, iBOT captures more semantic-level patterns compared to approaches predicting raw pixels, achieving state-of-the-art self-supervised learning performance.

The core insight of iBOT is that images naturally possess hierarchical semantic structure that can be captured through learned discrete tokens. Rather than using pre-trained, frozen tokenizers, iBOT learns the tokenization process end-to-end alongside the masking objective, enabling adaptive representations tailored to the pretraining task.

## Key Innovations

- **Online Tokenizer**: Learn discrete visual tokens jointly with main model
- **Semantic Masking**: Predict learned tokens rather than raw pixels
- **Momentum Tokenizer**: Stable token vocabulary through exponential moving average
- **Multi-Head Prediction**: Predict multiple correlated outputs simultaneously
- **Self-Distillation**: Student-teacher framework with updated target tokens

## Architectural Framework

### Components

iBOT consists of four main components:

1. **Image Encoder**: Vision transformer encoding visible patches
2. **Tokenizer**: Convolutional network quantizing features to discrete tokens
3. **Momentum Tokenizer**: EMA-updated tokenizer providing stable targets
4. **Prediction Head**: Classification head for discrete token prediction

### Tokenization Process

The online tokenizer maps continuous features to discrete vocabulary:

$$\mathbf{q} = \text{Tokenizer}(\mathbf{x}) = \arg\min_{e \in E} \|\mathbf{f} - e\|_2$$

where $E = \{e_1, \ldots, e_K\}$ is a learned discrete codebook with $K$ tokens.

## Mathematical Formulation

### Training Objective

iBOT optimizes the joint objective:

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \alpha \mathcal{L}_{\text{momentum}}$$

### Reconstruction Loss

For masked patch positions:

$$\mathcal{L}_{\text{recon}} = -\sum_{i \in M} \log p(\mathbf{q}_i^{\text{target}} | \mathbf{h}_i)$$

where:
- $M$ is the set of masked patch indices
- $\mathbf{q}_i^{\text{target}}$ is the target token (from momentum tokenizer)
- $p(\cdot | \mathbf{h}_i)$ is the classification probability for token at position $i$

### Momentum Target Update

Momentum tokenizer maintains stable targets through exponential averaging:

$$\theta_{\text{tokenizer}}^{\text{momentum}} \leftarrow \tau \theta_{\text{tokenizer}}^{\text{momentum}} + (1-\tau) \theta_{\text{tokenizer}}$$

where $\tau$ is the momentum coefficient (typically 0.999).

## Tokenizer Design

### Discrete Codebook Learning

The codebook $E$ is learned through:

$$\mathcal{L}_{\text{codebook}} = \|\mathbf{z} - \text{sg}(e)\|_2^2 + \beta \|\text{sg}(\mathbf{z}) - e\|_2^2$$

where $\text{sg}$ denotes stop-gradient operation, preventing codebook collapse.

### Token Vocabulary

Unlike frozen tokenizers, iBOT learns vocabulary end-to-end:

**Vocabulary Size**: Typically 16,384 tokens (learned during pretraining)

**Codebook Dimensionality**: Usually 256-512 dimensions

**Token Distribution**: Approximately uniform across vocabulary when properly trained

## Training Procedure

### Two-Stage Masking

iBOT employs block-wise masking to improve stability:

1. **Visible Patches**: Processed normally through encoder
2. **Masked Patches**: Predict discrete token from encoder representation
3. **Validation Masking**: Additional masking prevents trivial solutions

### Gradient Flow

!!! tip "Gradient Isolation"
    Momentum tokenizer uses stop-gradient operations, preventing direct gradient propagation and ensuring stable target representations.

**Regularization**: Additional constraints prevent codebook collapse:

$$\sum_k \mathbb{I}[\text{codebook}_k \text{ used}] = K$$

ensuring vocabulary usage across all tokens.

## Empirical Results

### Pretraining Performance

On ImageNet-21K pretraining:

**ViT-B**: 
- Top-1 Accuracy (iBOT): 87.3%
- Previous SOTA: 86.8%

**ViT-L**:
- Top-1 Accuracy (iBOT): 89.1%
- Improvement over MAE: +1.2%

### Downstream Task Transfer

| Task | Method | iBOT |
|------|--------|------|
| ImageNet-1K | Previous SOTA | 86.8% |
| ImageNet-1K | iBOT | 87.3% |
| COCO Detection | Previous SOTA | 52.1 AP |
| COCO Detection | iBOT | 52.6 AP |

## Comparison with Related Methods

| Aspect | MAE | BEiT | iBOT |
|--------|-----|------|------|
| **Tokenizer** | Fixed | Frozen | Learned |
| **Token Updates** | N/A | Static | Dynamic (EMA) |
| **Prediction Target** | Pixels | Discrete | Discrete |
| **Performance** | Very Good | Excellent | State-of-Art |
| **Complexity** | Low | Medium | High |

## Practical Considerations

!!! warning "Computational Overhead"
    Online tokenizer learning adds computational cost. Consider this trade-off when designing systems with strict efficiency requirements.

**Hyperparameter Sensitivity**: Momentum coefficient $\tau$ requires careful tuning; values too low cause instability, too high slow convergence.

**Codebook Initialization**: Random initialization of codebook tokens must be handled carefully to avoid early collapse.

## Applications in Quantitative Finance

For financial data with learned discrete representations:

- **Market Regimes**: Tokenize market conditions as discrete states
- **Anomaly Detection**: Unusual patterns manifest as low-probability tokens
- **Cross-Asset Transfer**: Shared token vocabulary across securities

## Research Directions

- Theoretically understanding why learned tokenizers outperform frozen ones
- Optimal codebook size and vocabulary strategies
- Multi-scale tokenization hierarchies
- Application to non-image modalities (time series, point clouds)

## Related Topics

- Masked Image Modeling Overview (Chapter 9.1.1)
- SimMIM Framework (Chapter 9.1.2)
- Vision Transformer Architecture (Chapter 6.2)
- Discrete Representation Learning
