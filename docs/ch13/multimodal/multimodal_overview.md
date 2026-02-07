# Vision-Language Models

## Learning Objectives

By the end of this section, you will be able to:

1. Understand the fundamental architecture of multimodal vision-language models
2. Implement dual-encoder architectures for joint embedding spaces
3. Design cross-modal attention mechanisms for vision-language fusion
4. Apply pretraining objectives for multimodal understanding
5. Build retrieval systems across image and text modalities

## Introduction

Vision-language models represent a paradigm shift in artificial intelligence, enabling systems to understand and reason about the relationship between visual content and natural language. These models bridge the semantic gap between pixels and words, creating unified representations that capture the rich interplay between what we see and how we describe it.

The core insight driving vision-language research is that visual understanding and language understanding are deeply intertwined in human cognition. We don't process images in isolation—we interpret them through the lens of language, concepts, and prior knowledge. Vision-language models attempt to capture this relationship computationally.

## Mathematical Foundations

### Joint Embedding Space

The fundamental goal is to learn a shared embedding space where semantically related images and text are close together. Given:

- **Image encoder**: $f_v: \mathcal{I} \rightarrow \mathbb{R}^d$
- **Text encoder**: $f_t: \mathcal{T} \rightarrow \mathbb{R}^d$

where $\mathcal{I}$ is the image space, $\mathcal{T}$ is the text space, and $d$ is the embedding dimension.

For a matched image-text pair $(I, T)$, we want:

$$\text{sim}(f_v(I), f_t(T)) \gg \text{sim}(f_v(I), f_t(T'))$$

for any mismatched text $T'$.

### Similarity Metrics

**Cosine Similarity** is the most common metric for comparing embeddings:

$$\text{sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|} = \frac{\sum_{i=1}^{d} a_i b_i}{\sqrt{\sum_{i=1}^{d} a_i^2} \sqrt{\sum_{i=1}^{d} b_i^2}}$$

When embeddings are L2-normalized ($\|a\| = \|b\| = 1$), cosine similarity reduces to the dot product:

$$\text{sim}(a, b) = a \cdot b = a^T b$$

This property makes computation efficient and enables fast similarity search.

### Similarity Matrix

For a batch of $N$ image-text pairs, we compute the full similarity matrix:

$$S_{ij} = f_v(I_i)^T \cdot f_t(T_j)$$

This $N \times N$ matrix contains:
- **Diagonal elements** $S_{ii}$: similarities between matched pairs (positive)
- **Off-diagonal elements** $S_{ij}, i \neq j$: similarities between mismatched pairs (negatives)

## Architecture Patterns

### Dual-Encoder Architecture

The simplest and most scalable architecture uses separate encoders for each modality:

```
Image → [Image Encoder] → [Projection] → Image Embedding
                                              ↓
                                         Similarity
                                              ↑
Text  → [Text Encoder]  → [Projection] → Text Embedding
```

**Advantages:**
- Efficient at inference: encode once, compare many times
- Scalable to billions of image-text pairs
- Suitable for retrieval applications

**Limitations:**
- No fine-grained interaction between modalities
- May miss subtle relationships requiring joint reasoning

### Fusion Architectures

More sophisticated models fuse information from both modalities:

**Early Fusion:**
```
[Image Features] + [Text Features] → [Joint Encoder] → Output
```

**Late Fusion:**
```
[Image Encoder] → Image Features ─┐
                                  ├─→ [Fusion Layer] → Output
[Text Encoder]  → Text Features  ─┘
```

**Cross-Attention Fusion:**
```
Image Features → Query ─┐
                        ├─→ Cross-Attention → Fused Features
Text Features  → K, V  ─┘
```

## PyTorch Implementation

### Simple Dual-Encoder Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ImageEncoder(nn.Module):
    """
    Image encoder that maps visual features to embedding space.
    
    In practice, this would use a pretrained CNN (ResNet, ViT) backbone.
    Here we demonstrate the projection architecture.
    """
    
    def __init__(self, input_dim: int = 2048, hidden_dim: int = 1024, 
                 embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image features (batch_size, input_dim)
        
        Returns:
            L2-normalized embeddings (batch_size, embed_dim)
        """
        embeddings = self.encoder(x)
        # L2 normalize for cosine similarity
        return F.normalize(embeddings, p=2, dim=1)


class TextEncoder(nn.Module):
    """
    Text encoder that maps language features to embedding space.
    
    In practice, this would use a pretrained language model (BERT, GPT).
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 1024,
                 embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        return F.normalize(embeddings, p=2, dim=1)


class DualEncoderModel(nn.Module):
    """
    Dual-encoder vision-language model.
    
    Maps images and text to a shared embedding space where
    matched pairs have high cosine similarity.
    """
    
    def __init__(self, image_dim: int = 2048, text_dim: int = 768,
                 embed_dim: int = 512):
        super().__init__()
        
        self.image_encoder = ImageEncoder(image_dim, embed_dim=embed_dim)
        self.text_encoder = TextEncoder(text_dim, embed_dim=embed_dim)
        self.embed_dim = embed_dim
    
    def forward(self, images: torch.Tensor, 
                texts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images and texts to shared space.
        
        Args:
            images: Image features (batch, image_dim)
            texts: Text features (batch, text_dim)
        
        Returns:
            Tuple of (image_embeddings, text_embeddings)
        """
        image_emb = self.image_encoder(images)
        text_emb = self.text_encoder(texts)
        return image_emb, text_emb
    
    def compute_similarity(self, image_emb: torch.Tensor,
                          text_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity matrix between image and text embeddings.
        
        Args:
            image_emb: (batch_i, embed_dim)
            text_emb: (batch_t, embed_dim)
        
        Returns:
            Similarity matrix (batch_i, batch_t)
        """
        # Since embeddings are normalized, dot product = cosine similarity
        return torch.matmul(image_emb, text_emb.t())
```

### Cross-Modal Attention

For deeper interaction between modalities:

```python
class CrossModalAttention(nn.Module):
    """
    Cross-modal attention allowing one modality to attend to another.
    
    This is the key mechanism enabling fine-grained vision-language
    understanding, used in models like ViLBERT, LXMERT, and ALBEF.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8,
                 ffn_dim: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, query: torch.Tensor, 
                key_value: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Cross-attention from query modality to key/value modality.
        
        Args:
            query: Query features (batch, query_len, embed_dim)
            key_value: Key/value features (batch, kv_len, embed_dim)
            mask: Optional attention mask
        
        Returns:
            Attended features (batch, query_len, embed_dim)
        """
        # Cross-attention with residual
        attended, _ = self.cross_attention(query, key_value, key_value,
                                           key_padding_mask=mask)
        x = self.norm1(query + attended)
        
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x
```

## Pretraining Objectives

Vision-language models typically use multiple objectives during pretraining:

### Image-Text Contrastive (ITC)

Align representations using contrastive learning (covered in detail in CLIP section).

### Image-Text Matching (ITM)

Binary classification predicting whether an image-text pair is matched:

$$\mathcal{L}_{ITM} = -\mathbb{E}_{(I,T)}\left[y \log p(match|I,T) + (1-y) \log(1-p(match|I,T))\right]$$

where $y=1$ for matched pairs and $y=0$ for mismatched pairs.

### Masked Language Modeling (MLM)

Predict masked tokens conditioned on image context:

$$\mathcal{L}_{MLM} = -\mathbb{E}_{(I,T)}\left[\sum_{i \in M} \log p(t_i | T_{\backslash M}, I)\right]$$

where $M$ is the set of masked token positions.

### Masked Region Modeling (MRM)

Predict masked image regions conditioned on text:

$$\mathcal{L}_{MRM} = -\mathbb{E}_{(I,T)}\left[\sum_{j \in R} \log p(r_j | I_{\backslash R}, T)\right]$$

where $R$ is the set of masked regions.

## Applications

### Image-Text Retrieval

Given a query in one modality, find the most relevant items in the other:

```python
def retrieve_images(model: DualEncoderModel, 
                   text_query: torch.Tensor,
                   image_gallery: torch.Tensor,
                   top_k: int = 10) -> torch.Tensor:
    """
    Retrieve top-k images for a text query.
    """
    with torch.no_grad():
        text_emb = model.text_encoder(text_query)
        image_embs = model.image_encoder(image_gallery)
        
        # Compute similarities
        similarities = torch.matmul(text_emb, image_embs.t())
        
        # Get top-k indices
        _, top_indices = torch.topk(similarities, k=top_k, dim=1)
        
    return top_indices
```

### Zero-Shot Classification

Classify images using text descriptions without task-specific training:

```python
def zero_shot_classify(model: DualEncoderModel,
                      image: torch.Tensor,
                      class_descriptions: list,
                      text_encoder_fn) -> torch.Tensor:
    """
    Zero-shot image classification using text descriptions.
    
    Args:
        image: Image features
        class_descriptions: List of text descriptions for each class
        text_encoder_fn: Function to encode text to features
    """
    with torch.no_grad():
        # Encode image
        image_emb = model.image_encoder(image)
        
        # Encode class descriptions
        text_features = [text_encoder_fn(desc) for desc in class_descriptions]
        text_features = torch.stack(text_features)
        text_embs = model.text_encoder(text_features)
        
        # Compute class probabilities
        logits = torch.matmul(image_emb, text_embs.t())
        probs = F.softmax(logits, dim=-1)
        
    return probs
```

## Evaluation Metrics

### Retrieval Metrics

**Recall@K**: Fraction of queries where the correct item is in top-K results

$$\text{Recall@K} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\text{rank}_i \leq K]$$

**Mean Reciprocal Rank (MRR)**: Average of reciprocal ranks

$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$

**Median Rank**: Median position of correct items

```python
def compute_retrieval_metrics(similarity_matrix: torch.Tensor,
                             k_values: list = [1, 5, 10]) -> dict:
    """
    Compute retrieval metrics assuming diagonal contains correct matches.
    """
    num_queries = similarity_matrix.shape[0]
    metrics = {}
    
    # Compute ranks
    ranks = []
    for i in range(num_queries):
        sims = similarity_matrix[i]
        sorted_indices = torch.argsort(sims, descending=True)
        rank = (sorted_indices == i).nonzero().item() + 1
        ranks.append(rank)
    
    ranks = torch.tensor(ranks, dtype=torch.float)
    
    # Recall@K
    for k in k_values:
        metrics[f'recall@{k}'] = (ranks <= k).float().mean().item()
    
    # MRR
    metrics['mrr'] = (1.0 / ranks).mean().item()
    
    # Median rank
    metrics['median_rank'] = ranks.median().item()
    
    return metrics
```

## Famous Models and Papers

| Model | Year | Key Innovation |
|-------|------|----------------|
| CLIP | 2021 | Large-scale contrastive pretraining |
| ALIGN | 2021 | Noisy web-scale image-text pairs |
| ViLT | 2021 | Minimal visual processing |
| ALBEF | 2021 | Align before fuse strategy |
| BLIP | 2022 | Bootstrapping with synthetic captions |
| CoCa | 2022 | Combined contrastive and captioning |
| Flamingo | 2022 | Few-shot visual reasoning |

### CLIP: Contrastive Language-Image Pretraining

CLIP (Radford et al., 2021) demonstrated that contrastive pretraining on 400M image-text pairs from the internet produces visual representations that transfer remarkably well to downstream tasks without any task-specific fine-tuning.

**Architecture**: Dual encoder (ViT or ResNet for images, Transformer for text) with learned temperature parameter $\tau$:

$$\mathcal{L}_{\text{CLIP}} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ij}/\tau)} + \log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N}\exp(s_{ji}/\tau)}\right]$$

The first term is image-to-text contrastive loss (for each image, the matching text should score highest). The second is text-to-image (for each text, the matching image should score highest). The symmetric formulation ensures both directions are optimized.

**Key insight**: CLIP learns a joint embedding space where semantic similarity transcends modality—"a photo of a dog" and an actual photo of a dog map to nearby vectors. This enables zero-shot classification by comparing image embeddings against text embeddings of class descriptions.

### DALL·E and Text-to-Image Generation

DALL·E (Ramesh et al., 2021) and its successors use Transformers to generate images from text descriptions:

**DALL·E 1**: Encodes images as discrete tokens via dVAE (discrete variational autoencoder), then trains an autoregressive Transformer to generate image tokens conditioned on text tokens.

**DALL·E 2**: Uses CLIP embeddings as the bridge between text and images. A prior model maps CLIP text embeddings to CLIP image embeddings, then a diffusion decoder generates images from CLIP image embeddings.

**Stable Diffusion**: Uses cross-attention to inject text conditioning into a U-Net diffusion model operating in latent space. At each denoising step, the U-Net's intermediate features (queries) attend to CLIP text encoder output (keys and values).

These models demonstrate that cross-modal attention is not limited to understanding—it is equally powerful for generation.

## Key Takeaways

1. **Dual encoders** provide efficient, scalable image-text matching suitable for retrieval
2. **Cross-modal attention** enables deeper reasoning requiring joint understanding
3. **Multiple pretraining objectives** (ITC, ITM, MLM) capture different aspects of vision-language alignment
4. **L2 normalization** simplifies similarity computation to dot products
5. **Zero-shot transfer** emerges from learning rich cross-modal representations

## Next Steps

- Explore CLIP's contrastive learning approach in detail
- Learn image captioning architectures
- Understand Visual Question Answering systems

## References

1. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021. (CLIP)
2. Jia, C., et al. "Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision." ICML 2021. (ALIGN)
3. Li, J., et al. "Align before Fuse: Vision and Language Representation Learning with Momentum Distillation." NeurIPS 2021. (ALBEF)
4. Li, J., et al. "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding." ICML 2022.
5. Ramesh, A., et al. "Hierarchical Text-Conditional Image Generation with CLIP Latents." 2022. (DALL·E 2)
6. Rombach, R., et al. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022. (Stable Diffusion)
