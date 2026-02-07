# CLS Token

## Introduction

The **CLS token** (classification token) is a learnable embedding prepended to the sequence of patch tokens before they enter the transformer encoder. After processing through all transformer layers, the CLS token's output serves as the aggregate representation of the entire image for classification. This mechanism, borrowed directly from BERT in NLP, provides a clean separation between feature extraction (via self-attention over all patches) and classification (via the CLS token's output).

---

## Mechanism

### How the CLS Token Works

The CLS token participates in self-attention alongside all patch tokens. Because self-attention allows every token to attend to every other token, the CLS token can gather information from all patches simultaneously:

$$\mathbf{z}_0 = [\mathbf{x}_{\text{cls}} \;;\; \mathbf{z}_0^{(1)} \;;\; \cdots \;;\; \mathbf{z}_0^{(N)}]$$

At each transformer layer $\ell$, the CLS token attends to all patch tokens and is attended to by all patch tokens:

$$\alpha_{\text{cls} \to j}^{(\ell)} = \text{softmax}\left(\frac{\mathbf{q}_{\text{cls}}^{(\ell)} (\mathbf{k}_j^{(\ell)})^\top}{\sqrt{d_k}}\right), \quad j = 0, 1, \ldots, N$$

Through successive layers, the CLS token progressively aggregates information, evolving from a randomly initialized vector into a rich summary of the image content.

```
Layer 0:  [CLS₀]  [P₁]  [P₂]  ... [Pₙ]     ← CLS is random initialization
              ↓     ↓     ↓         ↓
Layer 1:  [CLS₁]  [P₁]  [P₂]  ... [Pₙ]     ← CLS begins gathering patch info
              ↓     ↓     ↓         ↓
   ...       ...   ...   ...       ...
              ↓     ↓     ↓         ↓
Layer L:  [CLSₗ]  [P₁]  [P₂]  ... [Pₙ]     ← CLS is a rich image summary
              │
              ▼
         MLP Head → ŷ                          ← Classification from CLS only
```

### Mathematical Formulation

After $L$ transformer layers, the classification head operates only on the CLS token:

$$\hat{y} = W_{\text{head}} \cdot \text{LN}(\mathbf{z}_L^{(0)}) + \mathbf{b}_{\text{head}}$$

where $\mathbf{z}_L^{(0)}$ is the CLS token output from the final transformer layer, $\text{LN}$ is layer normalization, and $W_{\text{head}} \in \mathbb{R}^{C \times d}$ projects to $C$ classes.

### Why a Separate Token?

The CLS token serves a specific architectural purpose: it provides a **task-agnostic aggregation point** that is not tied to any particular spatial location. This design choice has several advantages:

1. **No spatial bias**: Unlike using the output at a specific patch position, the CLS token has no inherent spatial association
2. **Flexible aggregation**: The attention mechanism learns which patches are most relevant for the task
3. **Clean interface**: The CLS token provides a single vector output for downstream tasks, decoupling feature extraction from classification
4. **Multi-task adaptability**: Different task heads can be attached to the same CLS representation

---

## PyTorch Implementation

```python
import torch
import torch.nn as nn


class ViTWithCLS(nn.Module):
    """
    Simplified ViT focusing on CLS token mechanics.
    """
    def __init__(
        self,
        num_patches=196,
        embed_dim=768,
        depth=12,
        num_heads=12,
        num_classes=1000,
    ):
        super().__init__()

        # Learnable CLS token: shape (1, 1, embed_dim)
        # Initialized from N(0, 0.02) following standard practice
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Position embeddings for CLS + patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, patch_embeddings):
        """
        Parameters
        ----------
        patch_embeddings : torch.Tensor
            Shape (B, N, embed_dim) from patch embedding layer.

        Returns
        -------
        torch.Tensor
            Class logits, shape (B, num_classes).
        """
        B = patch_embeddings.shape[0]

        # Expand CLS token for batch: (1, 1, d) → (B, 1, d)
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # Prepend CLS token: (B, N, d) → (B, N+1, d)
        x = torch.cat([cls_tokens, patch_embeddings], dim=1)

        # Add position embeddings (CLS gets position 0)
        x = x + self.pos_embed

        # Transformer encoder
        x = self.encoder(x)
        x = self.norm(x)

        # Extract CLS token output (position 0)
        cls_output = x[:, 0]  # (B, embed_dim)

        return self.head(cls_output)
```

---

## CLS Token vs. Global Average Pooling

An important alternative to the CLS token is **Global Average Pooling** (GAP), which averages all patch token outputs:

$$\mathbf{z}_{\text{GAP}} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{z}_L^{(i)}$$

### Comparison

| Aspect | CLS Token | Global Average Pooling |
|---|---|---|
| **Aggregation** | Learned via attention | Fixed uniform average |
| **Parameters** | Adds 1 extra token ($d$ params) | Zero additional parameters |
| **Sequence length** | $N + 1$ (slightly more compute) | $N$ |
| **Spatial weighting** | Learned (non-uniform) | Uniform |
| **Training stability** | Requires careful initialization | More stable |
| **Performance (ViT)** | Standard, well-established | Comparable, sometimes better |

### When to Prefer Each

The CLS token is preferred when:

- Following the standard ViT recipe (most pretrained models use CLS)
- The task benefits from learned non-uniform spatial aggregation
- Multiple task heads will share the backbone (CLS provides a clean interface)

GAP is preferred when:

- Training from scratch on smaller datasets (more stable)
- Dense prediction tasks are also needed (patch tokens must be meaningful)
- Computational budget is tight (one fewer token in attention)

```python
class ViTWithGAP(nn.Module):
    """ViT variant using Global Average Pooling instead of CLS token."""
    def __init__(self, num_patches=196, embed_dim=768, depth=12,
                 num_heads=12, num_classes=1000):
        super().__init__()
        # No CLS token needed
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=4 * embed_dim, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, patch_embeddings):
        x = patch_embeddings + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)

        # Global average pooling over patch tokens
        x = x.mean(dim=1)  # (B, embed_dim)
        return self.head(x)
```

---

## Attention Analysis of the CLS Token

### Visualizing CLS Attention

The attention weights from the CLS token to patch tokens reveal which regions of the image the model focuses on for classification:

```python
def extract_cls_attention(model, x):
    """
    Extract attention weights from CLS token across all layers.

    Returns
    -------
    list of torch.Tensor
        Attention maps, each of shape (num_heads, N+1) showing
        CLS token's attention to all positions.
    """
    attention_maps = []

    def hook_fn(module, input, output):
        # For nn.MultiheadAttention, we need to call with need_weights=True
        pass

    # Alternative: manually compute attention for analysis
    B = x.shape[0]
    cls_tokens = model.cls_token.expand(B, -1, -1)
    tokens = torch.cat([cls_tokens, x], dim=1)
    tokens = tokens + model.pos_embed

    for layer in model.encoder.layers:
        # Pre-norm
        normed = layer.norm1(tokens)
        # Compute attention weights explicitly
        Q = normed @ layer.self_attn.in_proj_weight[:768].T
        K = normed @ layer.self_attn.in_proj_weight[768:1536].T
        attn_weights = (Q @ K.transpose(-2, -1)) / (768 ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        # Store CLS row (attention from CLS to all tokens)
        attention_maps.append(attn_weights[:, 0, :].detach())

        # Continue forward pass
        tokens = layer(tokens)

    return attention_maps
```

### Attention Patterns Across Layers

Empirically, CLS attention exhibits characteristic patterns across depth:

- **Early layers** (1–3): CLS attends broadly and relatively uniformly, gathering general information
- **Middle layers** (4–8): Attention becomes more structured, focusing on semantically meaningful regions
- **Late layers** (9–12): Attention concentrates on the most discriminative patches for the classification task

This progressive refinement mirrors how CNNs build hierarchical features, but through a fundamentally different mechanism (dynamic attention vs. fixed convolutions).

---

## Multi-Task CLS Tokens

For multi-task learning, multiple CLS tokens can be used, each specializing in a different task:

```python
class MultiTaskViT(nn.Module):
    """ViT with multiple CLS tokens for multi-task learning."""
    def __init__(self, num_patches, embed_dim, depth, num_heads, task_configs):
        """
        Parameters
        ----------
        task_configs : dict
            Mapping from task name to number of classes.
            e.g., {'classification': 1000, 'detection': 80, 'segmentation': 21}
        """
        super().__init__()
        self.num_tasks = len(task_configs)

        # One CLS token per task
        self.cls_tokens = nn.Parameter(
            torch.zeros(1, self.num_tasks, embed_dim)
        )
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_tasks, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=4 * embed_dim, dropout=0.1,
            activation='gelu', batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Task-specific heads
        self.heads = nn.ModuleDict({
            name: nn.Linear(embed_dim, n_classes)
            for name, n_classes in task_configs.items()
        })
        self.task_names = list(task_configs.keys())

    def forward(self, patch_embeddings):
        B = patch_embeddings.shape[0]
        cls_tokens = self.cls_tokens.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patch_embeddings], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x)

        # Each CLS token produces a task-specific output
        outputs = {}
        for i, name in enumerate(self.task_names):
            outputs[name] = self.heads[name](x[:, i])

        return outputs
```

---

## Quantitative Finance Applications

### Portfolio-Level CLS Token

In a transformer model processing individual asset features, a CLS token can serve as a **portfolio-level aggregation** mechanism:

```
Assets:    [CLS]  [Asset₁]  [Asset₂]  ...  [Assetₙ]
              │       │         │              │
              ▼       ▼         ▼              ▼
         Transformer Encoder (cross-asset attention)
              │
              ▼
         [CLS output] → Portfolio risk / return prediction
```

The CLS token attends to all assets and learns to weight them based on their relevance to the portfolio-level prediction task. The attention weights provide interpretable importance scores for each asset's contribution to the aggregate prediction.

### Multi-Horizon Forecasting

Multiple CLS tokens can encode different prediction horizons simultaneously:

```python
# CLS tokens for different forecast horizons
# [CLS_1d] [CLS_5d] [CLS_21d] [Asset₁] [Asset₂] ... [Assetₙ]
# Each CLS token specializes in its horizon's prediction
```

This approach allows a single model to generate forecasts at multiple horizons while sharing the same cross-asset attention computation, reducing redundancy compared to separate models per horizon.

### Risk Factor Decomposition

A CLS token trained to predict portfolio returns naturally decomposes its attention across assets. Analyzing these attention patterns across market regimes reveals:

- Which assets drive portfolio risk during stress periods
- Dynamic factor loading patterns that shift with market conditions
- Regime-dependent cross-asset dependencies

---

## Common Pitfalls

### Pitfall 1: CLS Token Initialization

The CLS token must be initialized with small values. Large initialization creates an outlier token that disrupts early attention patterns:

```python
# ❌ Wrong: Default random initialization (too large)
self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

# ✅ Correct: Small truncated normal
self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
nn.init.trunc_normal_(self.cls_token, std=0.02)
```

### Pitfall 2: Forgetting to Expand for Batch

The CLS token parameter has shape `(1, 1, d)` and must be expanded to match the batch dimension:

```python
# ❌ Wrong: Broadcasting may not work as expected in all contexts
x = torch.cat([self.cls_token, patches], dim=1)  # Shape mismatch if B > 1

# ✅ Correct: Explicitly expand
cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, d)
x = torch.cat([cls_tokens, patches], dim=1)
```

### Pitfall 3: Using CLS for Dense Prediction

The CLS token is designed for sequence-level (image-level) tasks. For dense prediction tasks (segmentation, detection), use the patch token outputs instead:

```python
# For classification: use CLS token
cls_output = x[:, 0]  # (B, d)

# For dense prediction: use patch tokens
patch_outputs = x[:, 1:]  # (B, N, d) — reshape to spatial grid
```

---

## Summary

The CLS token provides a learnable aggregation mechanism for producing a single image-level representation from a sequence of patch tokens:

1. It is **prepended** to the patch sequence and processed jointly through self-attention
2. It **attends to all patches**, learning a weighted aggregation optimized for the downstream task
3. **Global average pooling** is a viable alternative with comparable performance
4. **Multiple CLS tokens** enable multi-task learning with shared feature extraction
5. In quantitative finance, CLS tokens serve as natural **portfolio-level aggregation** points

---

## References

1. Dosovitskiy, A., et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. *ICLR 2021*.
2. Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL 2019*.
3. Zhai, X., et al. (2022). Scaling Vision Transformers. *CVPR 2022*.
4. Touvron, H., et al. (2021). Training Data-Efficient Image Transformers & Distillation through Attention. *ICML 2021*.
