# Transformer Machine Translation

## Learning Objectives

- Understand multi-head self-attention for MT
- Appreciate why Transformers replaced RNN-based MT
- Review WMT benchmark results

## From RNNs to Transformers

RNN-based MT processes tokens sequentially, preventing parallelization during training. The Transformer (Vaswani et al., 2017) replaces recurrence entirely with self-attention, enabling full parallelism.

## Multi-Head Self-Attention

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

### Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

where each head applies attention with different learned projections:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Multiple heads allow the model to attend to different aspects simultaneously — one head might capture syntactic structure while another captures semantic similarity.

## Transformer Architecture for MT

### Encoder

Each layer contains: multi-head self-attention + feed-forward network, with residual connections and layer normalization:

$$\mathbf{h}' = \text{LayerNorm}(\mathbf{h} + \text{MultiHead}(\mathbf{h}, \mathbf{h}, \mathbf{h}))$$
$$\mathbf{h}'' = \text{LayerNorm}(\mathbf{h}' + \text{FFN}(\mathbf{h}'))$$

### Decoder

Each layer adds **cross-attention** over encoder outputs:

$$\mathbf{s}' = \text{LayerNorm}(\mathbf{s} + \text{MaskedMultiHead}(\mathbf{s}, \mathbf{s}, \mathbf{s}))$$
$$\mathbf{s}'' = \text{LayerNorm}(\mathbf{s}' + \text{MultiHead}(\mathbf{s}', \mathbf{h}, \mathbf{h}))$$
$$\mathbf{s}''' = \text{LayerNorm}(\mathbf{s}'' + \text{FFN}(\mathbf{s}''))$$

Masked self-attention prevents the decoder from attending to future positions during training.

## Key Innovations

### Positional Encoding

Since self-attention is permutation-invariant, positional encodings inject sequence order:

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{\text{model}}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{\text{model}}})$$

### Label Smoothing

Softens the training target from one-hot to a smoothed distribution, improving generalization:

$$q(k) = (1 - \epsilon) \cdot \mathbf{1}[k = y] + \frac{\epsilon}{|\mathcal{V}|}$$

## WMT Benchmark Results

| Model | EN-DE BLEU | EN-FR BLEU | Parameters |
|-------|-----------|-----------|------------|
| Transformer Base | 27.3 | 38.1 | 65M |
| Transformer Big | 28.4 | 41.0 | 213M |
| mBART | 30.5 | -- | 680M |
| NLLB-200 | 31.2 | 43.5 | 3.3B |

## Advantages Over RNN-Based MT

| Aspect | RNN | Transformer |
|--------|-----|-------------|
| Parallelism | Sequential | Fully parallel |
| Long-range deps | Gradient issues | Direct attention |
| Training speed | Slow | Fast |
| Memory | O(1) per step | O(n^2) attention |

## References

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
2. Ott, M., et al. (2018). Scaling Neural Machine Translation. *WMT*.
