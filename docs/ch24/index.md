# Autoregressive Models

## Chapter Overview

Autoregressive models form one of the most fundamental and successful paradigms in generative modeling. By decomposing a joint distribution into a product of conditional distributions using the chain rule of probability, these models provide exact likelihood computation, principled training through maximum likelihood estimation, and intuitive generation through sequential sampling.

This chapter provides a comprehensive treatment of autoregressive generative models, from foundational concepts to state-of-the-art architectures.

## Key Concepts

### The Autoregressive Principle

At its core, an autoregressive model factorizes the joint distribution as:

$$P(\mathbf{x}) = \prod_{i=1}^{n} P(x_i | x_1, x_2, \ldots, x_{i-1})$$

This factorization is exact—not an approximation—and forms the basis for:
- **Tractable likelihood**: Direct computation of $\log P(\mathbf{x})$
- **Maximum likelihood training**: Optimize $\mathbb{E}[\log P(\mathbf{x})]$
- **Ancestral sampling**: Generate by sampling each $x_i$ in order

### Fundamental Trade-offs

| Advantage | Limitation |
|-----------|------------|
| Exact likelihood computation | Sequential generation (slow) |
| Stable training (no adversarial dynamics) | Exposure bias (train-test mismatch) |
| High sample quality | Error accumulation in long sequences |
| Natural for sequential data | Ordering must be chosen for non-sequential data |

## Chapter Contents

### 1. [Autoregressive Factorization](factorization.md)

**Mathematical foundations and design principles**

- Chain rule of probability and its implications
- Parameterizing conditional distributions
- Discrete vs. continuous output modeling
- Training with teacher forcing
- Sampling strategies (temperature, top-k, nucleus)
- Computational considerations

**Key equations:**
- Joint probability factorization
- Categorical and Gaussian conditionals
- Log-likelihood decomposition

### 2. [PixelCNN](pixelcnn.md)

**Autoregressive image generation**

- Raster scan ordering for images
- Masked convolutions (Type A and Type B)
- The blind spot problem and Gated PixelCNN
- PixelCNN++ improvements (discretized logistics mixture)
- Conditional generation with class labels
- Applications: inpainting, density estimation, anomaly detection

**Key techniques:**
- Masked convolution implementation
- Vertical and horizontal stacks
- Discretized logistic mixture likelihood

### 3. [WaveNet](wavenet.md)

**Autoregressive audio synthesis**

- Raw waveform modeling challenges
- Dilated causal convolutions
- Exponentially growing receptive fields
- μ-law quantization for audio
- Global and local conditioning mechanisms
- Fast generation techniques

**Key architectures:**
- WaveNet residual block with gated activation
- Parallel WaveNet (distillation)
- WaveRNN, WaveGlow variants

### 4. [Autoregressive Transformers](transformers.md)

**Modern sequence modeling with attention**

- From RNNs to Transformers
- Causal (masked) self-attention
- GPT architecture and training
- Positional encodings (learned, sinusoidal, RoPE)
- Efficient generation with KV-cache
- Modern improvements (GQA, SwiGLU, RMSNorm)

**Key techniques:**
- Causal masking implementation
- Multi-head attention
- KV-caching for generation

## Comparison of Architectures

| Model | Data Type | Key Innovation | Generation Speed |
|-------|-----------|----------------|------------------|
| AR(p) | Time series | Linear regression | Fast |
| PixelCNN | Images | Masked convolutions | Very slow |
| WaveNet | Audio | Dilated causal convolutions | Slow → Fast (variants) |
| Transformer | Sequences | Causal self-attention | Moderate (with cache) |

## Historical Development

```
1927    AR models (Yule)
        ↓
1986    RNNs with backpropagation
        ↓
2000    NADE (Neural Autoregressive Density Estimator)
        ↓
2016    PixelCNN, WaveNet
        ↓
2017    Attention Is All You Need (Transformer)
        ↓
2018    GPT (Generative Pre-trained Transformer)
        ↓
2020    GPT-3, scaling laws
        ↓
2023    LLaMA, GPT-4, modern LLMs
```

## Connection to Other Generative Models

Autoregressive models relate to other generative paradigms:

- **Normalizing Flows**: Some flows (MAF, IAF) have autoregressive structure
- **VAEs**: VAE decoders are often autoregressive
- **Diffusion Models**: Can be viewed as continuous autoregression over noise levels
- **Energy-Based Models**: Autoregressive models define implicit energy functions

## Applications in Quantitative Finance

| Application | Model Type | Key Benefit |
|-------------|------------|-------------|
| Time series forecasting | Transformer, WaveNet | Long-range dependencies |
| Scenario generation | Any AR model | Exact density, diverse samples |
| Text analysis | Transformer | Financial document understanding |
| Anomaly detection | PixelCNN | Likelihood-based scoring |
| Synthetic data | WaveNet-style | Realistic tick data generation |

## Prerequisites

Before studying this chapter, ensure familiarity with:

- **Probability theory**: Chain rule, conditional distributions
- **Deep learning basics**: Neural networks, backpropagation, PyTorch
- **Previous chapters**: 
  - Ch. 3.4: Attention mechanisms
  - Ch. 3.7-3.8: RNNs, LSTMs
  - Ch. 9: Representation learning

## Learning Objectives

After completing this chapter, you should be able to:

1. **Explain** the autoregressive factorization and its implications
2. **Implement** masked convolutions for images (PixelCNN)
3. **Design** dilated causal convolutions for sequential data (WaveNet)
4. **Build** Transformer-based autoregressive models
5. **Apply** appropriate sampling strategies for generation
6. **Evaluate** autoregressive models using likelihood metrics
7. **Choose** the right architecture for different data types

## Recommended Study Path

1. **Start with fundamentals**: Read [Autoregressive Factorization](factorization.md) thoroughly
2. **Pick your modality**:
   - Images → [PixelCNN](pixelcnn.md)
   - Audio/Time series → [WaveNet](wavenet.md)
   - Text/General sequences → [Transformers](transformers.md)
3. **Implement**: Code along with the PyTorch examples
4. **Experiment**: Try the exercises at the end of each section

## References

### Foundational Papers

1. Larochelle, H., & Murray, I. (2011). The Neural Autoregressive Distribution Estimator. *AISTATS*.
2. van den Oord, A., et al. (2016). Pixel Recurrent Neural Networks. *ICML*.
3. van den Oord, A., et al. (2016). WaveNet: A Generative Model for Raw Audio. *arXiv*.
4. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
5. Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training. *OpenAI*.

### Survey and Review

6. Papamakarios, G., et al. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.
7. Yang, L., et al. (2022). Diffusion Models: A Comprehensive Survey of Methods and Applications. *arXiv*.

### Textbooks

8. Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.
9. Goodfellow, I., et al. (2016). Deep Learning. MIT Press.
10. Murphy, K. P. (2022). Probabilistic Machine Learning: An Introduction. MIT Press.

---

## Quick Reference: PyTorch Implementations

### Basic AR Model
```python
# See factorization.md for complete implementation
class ARModel(nn.Module):
    def forward(self, x):
        logits = self.network(x[:, :-1])
        return logits
    
    def loss(self, x):
        return F.cross_entropy(self.forward(x), x[:, 1:])
```

### Masked Convolution (PixelCNN)
```python
# See pixelcnn.md for complete implementation
class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('mask', self.create_mask(mask_type))
```

### Causal Convolution (WaveNet)
```python
# See wavenet.md for complete implementation
class CausalConv1d(nn.Module):
    def forward(self, x):
        x_padded = F.pad(x, (self.padding, 0))
        return self.conv(x_padded)
```

### Causal Attention (Transformer)
```python
# See transformers.md for complete implementation
class CausalSelfAttention(nn.Module):
    def forward(self, x):
        scores = Q @ K.T / sqrt(d_k)
        scores = scores.masked_fill(causal_mask, -inf)
        return softmax(scores) @ V
```
