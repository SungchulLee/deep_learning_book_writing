# Decoder Architecture

## Overview

The **decoder-only Transformer** is the dominant architecture behind modern large language models. Unlike the original encoder-decoder Transformer designed for sequence-to-sequence tasks, decoder-only models use a single stack of Transformer blocks with **causal (masked) self-attention**, ensuring that the prediction at position $t$ depends only on tokens at positions $\leq t$. This simple constraint is what makes autoregressive [next-token prediction](next_token_prediction.md) possible: the model can be trained and sampled left-to-right, one token at a time.

This section develops the decoder-only architecture from input processing through the final output distribution, covering each component — token embeddings, positional encoding, causal self-attention, feed-forward layers, normalization — with mathematical detail. We also discuss the key architectural variations across model families (pre-norm vs. post-norm, RoPE vs. learned positions, GQA vs. MHA) and connect to the inference optimizations in [Section 15.7](../inference/inference_overview.md) that make serving these models practical.

---

## 1. Input Processing

### 1.1 Tokenization

Raw text is first converted into a sequence of **token IDs** $(x_1, x_2, \ldots, x_T)$ using a subword tokenization algorithm. The two dominant approaches are:

| Algorithm | Used By | Mechanism |
|-----------|---------|-----------|
| **Byte Pair Encoding (BPE)** | GPT-2/3/4, LLaMA | Iteratively merges most frequent byte pairs |
| **SentencePiece (Unigram)** | LLaMA-2/3, Gemma | Probabilistic subword segmentation |

Each $x_t \in \{1, 2, \ldots, |\mathcal{V}|\}$ indexes into the vocabulary $\mathcal{V}$. Typical vocabulary sizes range from 32K (LLaMA) to 100K+ (GPT-4). Tokenization details and their impact on model behavior are covered in [Tokenization and Scale](tokenization_scale.md).

### 1.2 Token and Positional Embeddings

Each token ID is mapped to a dense vector via an **embedding matrix** $\mathbf{W}_e \in \mathbb{R}^{|\mathcal{V}| \times d}$, where $d$ is the model dimension. The input representation must also encode position information, since self-attention is permutation-equivariant. The two main approaches:

**Learned absolute embeddings** (GPT-2, GPT-3):

$$\mathbf{h}_t^{(0)} = \mathbf{W}_e[x_t] + \mathbf{p}_t$$

where $\mathbf{p}_t \in \mathbb{R}^d$ is a learned vector for position $t$. This ties the model to a fixed maximum sequence length.

**Rotary Position Embeddings (RoPE)** (LLaMA, Qwen, Mistral):

Instead of adding positional vectors to the input, RoPE applies a rotation to the query and key vectors in attention, encoding relative position directly in the attention computation:

$$\mathbf{q}_t' = R_t \mathbf{q}_t, \quad \mathbf{k}_s' = R_s \mathbf{k}_s$$

where $R_t$ is a block-diagonal rotation matrix with angles proportional to position $t$. The dot product $\mathbf{q}_t'^\top \mathbf{k}_s'$ then depends only on the relative position $t - s$, enabling length generalization beyond the training context. RoPE has become the standard for modern LLMs due to its superior extrapolation properties.

---

## 2. Causal Self-Attention

### 2.1 Single-Head Attention

Given input representations $\mathbf{H} = [\mathbf{h}_1, \ldots, \mathbf{h}_T]^\top \in \mathbb{R}^{T \times d}$, a single attention head computes query, key, and value projections:

$$\mathbf{Q} = \mathbf{H}\mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H}\mathbf{W}_K, \quad \mathbf{V} = \mathbf{H}\mathbf{W}_V$$

where $\mathbf{W}_Q, \mathbf{W}_K \in \mathbb{R}^{d \times d_k}$ and $\mathbf{W}_V \in \mathbb{R}^{d \times d_v}$, with head dimension $d_k = d_v = d / H$ for $H$ heads.

The attention output at position $t$ is:

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V})_t = \sum_{s=1}^{t} \alpha_{ts}\, \mathbf{v}_s$$

where the attention weights are:

$$\alpha_{ts} = \frac{\exp\!\left(\mathbf{q}_t^\top \mathbf{k}_s \,/\, \sqrt{d_k}\right)}{\sum_{s'=1}^{t} \exp\!\left(\mathbf{q}_t^\top \mathbf{k}_{s'} \,/\, \sqrt{d_k}\right)}$$

### 2.2 The Causal Mask

The **causal constraint** — restricting the summation to $s \leq t$ — is what makes the architecture autoregressive. In matrix form, the attention logits are masked before softmax:

$$\text{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right) \mathbf{V}$$

where the mask $\mathbf{M} \in \mathbb{R}^{T \times T}$ has $M_{ts} = 0$ for $s \leq t$ and $M_{ts} = -\infty$ for $s > t$. This ensures that token $t$ cannot attend to any future token, which is essential for the [next-token prediction](next_token_prediction.md) training objective: the model must predict $x_{t+1}$ without seeing it.

### 2.3 Multi-Head Attention

Multiple attention heads allow the model to attend to different aspects of the input simultaneously:

$$\text{MHA}(\mathbf{H}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\, \mathbf{W}_O$$

where each $\text{head}_i = \text{Attn}(\mathbf{H}\mathbf{W}_Q^{(i)}, \mathbf{H}\mathbf{W}_K^{(i)}, \mathbf{H}\mathbf{W}_V^{(i)})$ and $\mathbf{W}_O \in \mathbb{R}^{d \times d}$ is the output projection.

### 2.4 Grouped-Query Attention (GQA)

Standard MHA requires storing separate key-value pairs for each head, which dominates memory during inference (see [KV Cache](../inference/kv_cache.md)). **Grouped-Query Attention** (Ainslie et al., 2023) reduces this cost by sharing key-value heads across groups of query heads:

| Variant | KV Heads | Query Heads | KV Memory | Used By |
|---------|----------|-------------|-----------|---------|
| Multi-Head (MHA) | $H$ | $H$ | $H \cdot d_k$ per token | GPT-3 |
| Multi-Query (MQA) | 1 | $H$ | $d_k$ per token | PaLM |
| Grouped-Query (GQA) | $H/G$ | $H$ | $(H/G) \cdot d_k$ per token | LLaMA-2/3, Mistral |

GQA with $G = 4$–$8$ groups provides near-MHA quality with significantly reduced KV cache memory, enabling longer context windows and larger batch sizes during inference.

---

## 3. Transformer Block

### 3.1 Block Structure

Each Transformer block applies attention followed by a feed-forward network, both with residual connections and normalization. Modern LLMs universally use **Pre-LayerNorm** (pre-LN), where normalization is applied before the sublayer rather than after:

$$\mathbf{h}' = \mathbf{h} + \text{MHA}\!\left(\text{Norm}(\mathbf{h})\right)$$

$$\mathbf{h}'' = \mathbf{h}' + \text{FFN}\!\left(\text{Norm}(\mathbf{h}')\right)$$

Pre-LN improves training stability for deep networks (96+ layers) compared to the original post-LN formulation. The normalization is typically **RMSNorm** (Zhang & Sennrich, 2019) rather than LayerNorm, as it is computationally cheaper (no mean subtraction) and empirically equivalent:

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \boldsymbol{\gamma}$$

where $\boldsymbol{\gamma}$ is a learned scale parameter.

### 3.2 Feed-Forward Network

The position-wise FFN applies a nonlinear transformation independently to each position. Modern LLMs use the **SwiGLU** variant (Shazeer, 2020) rather than the original ReLU or GELU FFN:

$$\text{SwiGLU}(\mathbf{x}) = \left(\mathbf{W}_1 \mathbf{x} \odot \text{SiLU}(\mathbf{W}_{\text{gate}} \mathbf{x})\right) \mathbf{W}_2$$

where $\mathbf{W}_1, \mathbf{W}_{\text{gate}} \in \mathbb{R}^{d \times d_{\text{ff}}}$ and $\mathbf{W}_2 \in \mathbb{R}^{d_{\text{ff}} \times d}$, with $\text{SiLU}(x) = x \cdot \sigma(x)$.

The inner dimension $d_{\text{ff}}$ is conventionally $\approx 2.7d$ for SwiGLU (to match the parameter count of a standard $4d$ FFN with two matrices), yielding three matrices instead of two but with a smaller inner dimension.

The FFN is where most of the model's parameters reside and where factual knowledge is believed to be stored. Each position is processed independently, so the FFN adds no cross-position interaction — all inter-token communication happens through attention.

### 3.3 The Full Forward Pass

Stacking $N$ blocks and applying a final normalization:

$$\mathbf{h}^{(0)} = \text{Embed}(x_1, \ldots, x_T)$$

$$\mathbf{h}^{(\ell)} = \text{Block}_\ell(\mathbf{h}^{(\ell-1)}), \quad \ell = 1, \ldots, N$$

$$\mathbf{h}_{\text{out}} = \text{Norm}(\mathbf{h}^{(N)})$$

The final hidden states $\mathbf{h}_{\text{out}} \in \mathbb{R}^{T \times d}$ encode the model's representation of the entire sequence.

---

## 4. Output and Generation

### 4.1 Language Model Head

The output distribution over the vocabulary at each position is computed by projecting the final hidden state through the (typically tied) embedding matrix:

$$p_\theta(x_{t+1} \mid x_{\leq t}) = \text{softmax}\!\left(\mathbf{W}_e^\top\, \mathbf{h}_{\text{out},t}\right)$$

**Weight tying** — using the same matrix $\mathbf{W}_e$ for both the input embedding and the output projection — reduces parameter count by $|\mathcal{V}| \times d$ (often 500M+ parameters) and provides a useful inductive bias: tokens with similar embeddings produce similar output distributions.

### 4.2 Autoregressive Generation

During inference, the model generates tokens one at a time. Given a prompt $(x_1, \ldots, x_T)$:

1. Compute the forward pass to obtain $p_\theta(x_{T+1} \mid x_{\leq T})$
2. Sample or select $x_{T+1}$ from this distribution
3. Append $x_{T+1}$ to the sequence
4. Repeat from step 1 with the extended sequence

This sequential process is the primary inference bottleneck. The [KV cache](../inference/kv_cache.md) avoids redundant computation by caching key-value pairs from previous steps, and [speculative decoding](../inference/speculative_decoding.md) uses a smaller draft model to generate multiple candidate tokens in parallel.

### 4.3 Decoding Strategies

The choice of how tokens are selected from $p_\theta$ affects output quality dramatically:

**Greedy decoding** selects the most probable token:

$$x_{t+1} = \arg\max_{w \in \mathcal{V}}\; p_\theta(w \mid x_{\leq t})$$

This is deterministic but tends to produce repetitive, generic text.

**Temperature sampling** scales the logits before softmax:

$$p_\tau(w \mid x_{\leq t}) = \frac{\exp(z_w / \tau)}{\sum_{w'} \exp(z_{w'} / \tau)}$$

where $z_w$ are the raw logits and $\tau > 0$ is the temperature. Lower $\tau$ sharpens the distribution (more deterministic); higher $\tau$ flattens it (more diverse).

**Top-$k$ sampling** (Fan et al., 2018) restricts sampling to the $k$ highest-probability tokens, zeroing out all others before renormalization.

**Nucleus (top-$p$) sampling** (Holtzman et al., 2020) dynamically selects the smallest token set whose cumulative probability exceeds threshold $p$:

$$\mathcal{V}_p = \arg\min_{\mathcal{S} \subseteq \mathcal{V}} |\mathcal{S}| \quad \text{s.t.} \quad \sum_{w \in \mathcal{S}} p_\theta(w \mid x_{\leq t}) \geq p$$

Nucleus sampling adapts the effective vocabulary size to the model's confidence at each step. In practice, modern LLM APIs combine temperature scaling with nucleus sampling (typically $\tau \in [0.7, 1.0]$, $p \in [0.9, 0.95]$).

---

## 5. Architectural Variations Across Model Families

The basic decoder-only design is shared across all major LLMs, but families differ in important details:

| Feature | GPT-3 | LLaMA-2 | LLaMA-3 | Mistral | Qwen-2 |
|---------|-------|---------|---------|---------|--------|
| Normalization | Post-LN | Pre-LN (RMSNorm) | Pre-LN (RMSNorm) | Pre-LN (RMSNorm) | Pre-LN (RMSNorm) |
| Position encoding | Learned absolute | RoPE | RoPE | RoPE | RoPE |
| Activation | GELU | SwiGLU | SwiGLU | SwiGLU | SwiGLU |
| Attention | MHA | GQA | GQA | GQA + Sliding Window | GQA |
| Bias terms | Yes | No | No | No | Partial |
| Vocab size | 50,257 | 32,000 | 128,256 | 32,000 | 151,936 |

Key trends in modern architectures:

- **RoPE** has replaced learned absolute positions universally
- **SwiGLU** has replaced GELU in the FFN
- **GQA** has replaced full MHA for inference efficiency
- **RMSNorm** has replaced LayerNorm
- **Bias terms removed** from attention and FFN projections (slight parameter savings, no quality loss)

For detailed comparisons of specific model families, see [Architecture Comparison](architectures.md), [GPT Series](gpt_series.md), and [LLaMA Family](llama.md).

---

## 6. Computational Profile

### 6.1 Parameter Count

For a decoder-only Transformer with $N$ layers, model dimension $d$, $H$ heads, and FFN inner dimension $d_{\text{ff}}$:

| Component | Parameters per Layer | Fraction (typical) |
|-----------|--------------------|--------------------|
| Attention ($\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V, \mathbf{W}_O$) | $4d^2$ (MHA) | ~33% |
| FFN ($\mathbf{W}_1, \mathbf{W}_{\text{gate}}, \mathbf{W}_2$) | $3 \cdot d \cdot d_{\text{ff}} \approx 8d^2$ (SwiGLU) | ~67% |
| Layer norms | $2d$ | Negligible |

Total parameters $\approx N \cdot 12d^2 + |\mathcal{V}| \cdot d$ (the embedding matrix). For representative scales:

| Model | $N$ | $d$ | $H$ | $d_{\text{ff}}$ | Total Parameters |
|-------|-----|-----|-----|-----------------|-----------------|
| GPT-2 Large | 36 | 1,280 | 20 | 5,120 | 774M |
| LLaMA-2 7B | 32 | 4,096 | 32 | 11,008 | 6.7B |
| LLaMA-2 70B | 80 | 8,192 | 64 | 28,672 | 70B |
| GPT-3 175B | 96 | 12,288 | 96 | 49,152 | 175B |

### 6.2 Computational Cost

The dominant cost is the matrix multiplications in attention and FFN. For a sequence of length $T$:

| Operation | FLOPs | Memory |
|-----------|-------|--------|
| Attention QKV projection | $6 \cdot T \cdot d^2$ per layer | $O(T \cdot d)$ |
| Attention dot product | $2 \cdot T^2 \cdot d$ per layer | $O(T^2)$ or $O(T)$ with [Flash Attention](../inference/flash_attention.md) |
| FFN | $\sim 16 \cdot T \cdot d^2$ per layer (SwiGLU) | $O(T \cdot d_{\text{ff}})$ |

The $O(T^2)$ attention cost becomes the bottleneck for long sequences. [Flash Attention](../inference/flash_attention.md) reduces the memory cost from $O(T^2)$ to $O(T)$ without changing the FLOPs, while [Paged Attention](../inference/paged_attention.md) enables efficient memory management during serving.

---

## 7. Context Window and Multi-Turn Dialogue

### 7.1 Context as a Flat Sequence

A key design decision is that the decoder processes the entire context — system prompt, conversation history, and current query — as a single flat token sequence:

$$[\texttt{<system>}]\; s \;[\texttt{<user>}]\; u_1 \;[\texttt{<assistant>}]\; r_1 \;[\texttt{<user>}]\; u_2 \;[\texttt{<assistant>}]\; \ldots$$

Special tokens delimit roles. The causal mask ensures every token in the response attends to all preceding context, including the system prompt, all prior turns, and earlier tokens in the current response.

### 7.2 Context Window Limits

| Model | Context Window | Approximate Token Budget |
|-------|---------------|------------------------|
| GPT-3 | 2,048 | ~1,500 words |
| GPT-3.5 | 4,096 / 16,384 | ~3K / 12K words |
| GPT-4 | 8,192 / 128,000 | ~6K / 96K words |
| LLaMA-2 | 4,096 | ~3K words |
| LLaMA-3 | 8,192 / 128,000 | ~6K / 96K words |
| Claude 3 | 200,000 | ~150K words |

When the conversation exceeds the context window, earlier turns must be truncated or summarized. Strategies for managing long contexts are discussed in [Conversational AI](conversational_ai.md).

---

## 8. PyTorch Implementation

```python
"""
Decoder-Only Transformer Block
===============================
Minimal implementation of a single Transformer decoder block
with Pre-LN, RoPE, GQA, and SwiGLU — matching modern LLM design.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply Rotary Position Embeddings to query or key tensor.

    Parameters
    ----------
    x : Tensor of shape (B, H, T, d_k)
    freqs : Tensor of shape (T, d_k // 2)
        Precomputed rotation frequencies.

    Returns
    -------
    Rotated tensor of same shape.
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, T, d_k//2)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


def precompute_rope_freqs(dim: int, max_len: int, base: float = 10000.0) -> torch.Tensor:
    """Precompute RoPE frequency tensor of shape (max_len, dim // 2)."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len).float()
    return torch.outer(t, freqs)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).

    Query heads are divided into groups, each sharing one set of KV heads.

    Parameters
    ----------
    dim : int
        Model dimension.
    n_heads : int
        Number of query heads.
    n_kv_heads : int
        Number of key-value heads (n_heads must be divisible by n_kv_heads).
    max_len : int
        Maximum sequence length for RoPE precomputation.
    """

    def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_len: int = 4096):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.head_dim = dim // n_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.register_buffer(
            "rope_freqs", precompute_rope_freqs(self.head_dim, max_len)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        freqs = self.rope_freqs[:T]
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Expand KV heads to match query heads (GQA)
        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=1)
            v = v.repeat_interleave(self.n_groups, dim=1)

        # Scaled dot-product attention with causal mask
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn = attn + mask
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(out)


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.w1(x) * F.silu(self.w_gate(x)))


class DecoderBlock(nn.Module):
    """
    Single Transformer decoder block (Pre-LN, GQA, SwiGLU).

    Parameters
    ----------
    dim : int
        Model dimension.
    n_heads : int
        Number of query heads.
    n_kv_heads : int
        Number of KV heads.
    ff_dim : int
        FFN hidden dimension.
    max_len : int
        Maximum sequence length.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        ff_dim: int,
        max_len: int = 4096,
    ):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = GroupedQueryAttention(dim, n_heads, n_kv_heads, max_len)
        self.ff_norm = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ff_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), mask)
        x = x + self.ffn(self.ff_norm(x))
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Complete decoder-only Transformer language model.

    Parameters
    ----------
    vocab_size : int
    dim : int
        Model dimension.
    n_layers : int
    n_heads : int
    n_kv_heads : int
    ff_dim : int
    max_len : int
    """

    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: int = 8,
        ff_dim: int = 11008,
        max_len: int = 4096,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            DecoderBlock(dim, n_heads, n_kv_heads, ff_dim, max_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        # Weight tying: output projection shares embedding weights
        self.lm_head = lambda x: F.linear(x, self.embed.weight)

        # Precompute causal mask
        mask = torch.full((max_len, max_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(
        self, tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : LongTensor of shape (B, T)

        Returns
        -------
        logits : Tensor of shape (B, T, vocab_size)
        """
        B, T = tokens.shape
        h = self.embed(tokens)
        mask = self.causal_mask[:T, :T].unsqueeze(0)

        for layer in self.layers:
            h = layer(h, mask)

        h = self.norm(h)
        return self.lm_head(h)
```

---

## 9. Finance Applications

The decoder architecture has direct relevance to quantitative finance:

| Application | Architectural Feature Leveraged | Notes |
|-------------|-------------------------------|-------|
| Time-series forecasting | Causal attention (no future leakage) | Naturally respects temporal ordering |
| Financial document Q&A | Long context windows (128K+) | Process entire filings without truncation |
| Trading signal generation | Autoregressive token generation | Sequential decision-making formulation |
| Code generation for quant | SwiGLU + deep stacks for reasoning | Complex code synthesis; see [LLM Applications](../agents/llm_applications.md) |
| Risk report generation | Nucleus sampling for controlled output | Low temperature for factual content |

---

## 10. Key Takeaways

1. **The causal mask is the defining feature**: by restricting attention to $s \leq t$, the decoder enables autoregressive generation where each token is predicted from all preceding tokens but no future ones.

2. **Modern LLMs share a common blueprint**: Pre-LN (RMSNorm) → GQA with RoPE → SwiGLU FFN → weight-tied output head. The differences between model families (GPT-4, LLaMA-3, Mistral) are variations on this theme, not fundamentally different architectures.

3. **The FFN stores knowledge, attention routes information**: attention layers handle inter-token communication; FFN layers (with ~67% of parameters) are where factual associations are believed to reside.

4. **GQA and RoPE are now standard**: GQA reduces KV cache memory by sharing key-value heads, enabling longer contexts and larger batches during inference. RoPE enables length generalization beyond training context lengths.

5. **Generation is sequential and slow**: autoregressive decoding requires one forward pass per generated token, making inference optimization (KV cache, Flash Attention, speculative decoding) essential for practical deployment.

---

## Exercises

### Exercise 1: Parameter Counting

For a LLaMA-2 7B model ($N = 32$, $d = 4096$, $H = 32$, $n_{\text{kv}} = 32$, $d_{\text{ff}} = 11008$, $|\mathcal{V}| = 32000$), compute the exact parameter count for (a) one attention layer, (b) one FFN layer, (c) the embedding matrix, (d) the full model. Verify that the total is approximately 6.7B.

### Exercise 2: GQA Memory Savings

Compare the KV cache memory for a sequence of 4096 tokens with (a) full MHA ($H = 32$ KV heads), (b) GQA ($n_{\text{kv}} = 8$), (c) MQA ($n_{\text{kv}} = 1$), all with $d_k = 128$ in float16. What is the memory saving factor for GQA vs. MHA?

### Exercise 3: Causal Mask Verification

Implement the `DecoderOnlyTransformer` above with small dimensions ($d = 64$, $N = 2$, $H = 4$). Feed in a sequence and verify that the output at position $t$ does not change when you modify tokens at positions $> t$.

### Exercise 4: Decoding Strategy Comparison

Using any LLM API, generate 20 completions for the prompt "The three most important factors in portfolio risk management are" using (a) greedy, (b) temperature 0.3, (c) temperature 1.0 with top-$p = 0.9$. Measure Distinct-2 (diversity) and manually rate factual accuracy.

### Exercise 5: RoPE vs. Learned Positions

Implement both learned absolute positional embeddings and RoPE in the model above. Train on sequences of length 128 and evaluate on length 256. Which approach degrades less on the longer sequences?

---

## References

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention Is All You Need. *Advances in Neural Information Processing Systems (NeurIPS)*.
2. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models Are Unsupervised Multitask Learners. *OpenAI Technical Report*.
3. Brown, T. B., Mann, B., Ryder, N., et al. (2020). Language Models Are Few-Shot Learners. *Advances in Neural Information Processing Systems (NeurIPS)*.
4. Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2024). RoFormer: Enhanced Transformer with Rotary Position Embedding. *Neurocomputing*, 568, 127063.
5. Shazeer, N. (2020). GLU Variants Improve Transformer. *arXiv preprint arXiv:2002.05202*.
6. Zhang, B. & Sennrich, R. (2019). Root Mean Square Layer Normalization. *Advances in Neural Information Processing Systems (NeurIPS)*.
7. Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *Proceedings of EMNLP*.
8. Touvron, H., Lavril, T., Izacard, G., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv preprint arXiv:2302.13971*.
9. Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The Curious Case of Neural Text Degeneration. *Proceedings of the 8th International Conference on Learning Representations (ICLR)*.
