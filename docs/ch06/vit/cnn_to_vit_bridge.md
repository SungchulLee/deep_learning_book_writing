# Bridge: From CNNs to Vision Transformers

## Introduction

The transition from convolutional neural networks to Vision Transformers represents a fundamental shift in architectural paradigms for computer vision. Rather than an abrupt discontinuity, this evolution reflects a gradual relaxation of CNN inductive biases, with intermediate architectures serving as conceptual bridges between locality-based and attention-based processing.

Understanding this bridge is crucial for practitioners seeking to leverage the complementary strengths of both paradigms. Modern state-of-the-art systems often employ hybrid approaches that retain CNN efficiency while incorporating transformer flexibility, creating architectures that effectively balance computational constraints with modeling power.

## Evolution of Architectural Paradigms

### Stage 1: Pure CNNs

Traditional CNNs process images through successive local convolutions:

$$\mathbf{y}_{i,j}^{(l)} = f\left(\sum_{p,q} W_{p,q}^{(l)} \mathbf{x}_{i+p, j+q}^{(l-1)}\right)$$

This locality-centric approach ensures:
- Computational efficiency
- Inductive alignment with image structure
- Parameter sharing across space

### Stage 2: Attention-Enhanced CNNs

Augmenting CNNs with attention mechanisms (SE-Net, CBAM) preserves locality while enabling adaptive channel and spatial reweighting:

$$\mathbf{y}_{i,j}^{(l)} = \alpha_{i,j}^{(l)} \odot f\left(\sum_{p,q} W_{p,q}^{(l)} \mathbf{x}_{i+p, j+q}^{(l-1)}\right)$$

where $\odot$ denotes element-wise multiplication and $\alpha_{i,j}^{(l)}$ represents learned attention weights.

### Stage 3: Non-Local Networks

Non-local operations capture dependencies across entire spatial domains:

$$\mathbf{y}_{i,j} = \frac{1}{Z} \sum_{i',j'} w(i,j,i',j') \cdot \mathbf{x}_{i',j'}$$

where $Z$ is a normalization factor and $w$ represents learned interaction weights. This represents a departure from strict locality.

### Stage 4: Hybrid Architectures

Hybrid models combine convolutional stems with transformer bodies:

$$\text{Tokens} = \text{PatchEmbed}(\text{ConvStem}(\mathbf{x}))$$
$$\mathbf{z}^{(l)} = \text{Transformer}(\mathbf{z}^{(l-1)})$$

Early layers leverage CNN inductive biases for efficiency; later layers employ transformer flexibility.

### Stage 5: Pure Vision Transformers

ViT processes images as sequences of patches without convolution:

$$\mathbf{z}^{(l)} = \text{MSA}(\text{LN}(\mathbf{z}^{(l-1)})) + \mathbf{z}^{(l-1)}$$
$$\mathbf{z}^{(l)} = \text{MLP}(\text{LN}(\mathbf{z}^{(l)})) + \mathbf{z}^{(l)}$$

## Key Architectural Bridges

!!! tip "Patch Embedding as Convolution"
    The patch embedding layer in ViT can be understood as a single convolutional layer with non-overlapping kernels, creating a smooth transition from CNN thinking.

**Convolutional Stems**: Replace initial ViT patch embedding with several convolutional layers to capture low-level features, improving performance on smaller datasets.

**Pyramid Structures**: Hierarchical vision models (Swin, PVT) maintain multi-scale processing similar to CNNs while using attention mechanisms.

**Hybrid Self-Attention**: Replace full transformer blocks with convolutional layers in specific positions, balancing computational cost and modeling capacity.

## Comparative Analysis

| Dimension | CNN | Hybrid | ViT |
|-----------|-----|--------|-----|
| **Receptive Field Growth** | Linear | Hybrid | Quadratic |
| **Computational Cost** | Low | Medium | High |
| **Data Requirements** | Low | Medium | High |
| **Adaptivity** | Limited | Moderate | High |
| **Translation Equivariance** | Strict | Relaxed | None |

## Practical Considerations for Practitioners

!!! warning "Data Regime Matters"
    Pure ViTs typically underperform CNNs on small datasets (< 1M images). Hybrid architectures or transfer learning become essential for limited data scenarios.

!!! note "Computational Budget"
    CNNs remain advantageous when computational resources are severely constrained. Hybrid models offer middle-ground solutions.

## Future Directions

Emerging research explores:

- **Neuromorphic Hybrids**: Combining spiking neurons with attention
- **Efficient Transformers**: Linear-complexity attention for high-resolution images
- **Learned Routing**: Dynamically selecting between convolution and attention based on input
- **Multi-modal Fusion**: Bridging vision and language through hybrid architectures

## Related Topics

- CNN Limitations (Chapter 6.1.2)
- Vision Transformer Architecture (Chapter 6.2)
- Attention Mechanisms in CNNs (Chapter 6.1.1)
- Pyramid Vision Transformers (Chapter 6.2.2)
