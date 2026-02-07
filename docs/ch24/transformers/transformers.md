# Autoregressive Transformers

## Introduction

The Transformer architecture, introduced by Vaswani et al. (2017), has become the dominant paradigm for autoregressive sequence modeling. Unlike RNNs which process sequences step-by-step, Transformers use **self-attention** to compute representations in parallel during training while maintaining the autoregressive property through **causal masking**. This combination—parallel training with sequential generation—underlies modern large language models (LLMs) like GPT, LLaMA, and their successors.

## From RNNs to Transformers

### Limitations of Recurrent Models

Traditional autoregressive models (RNNs, LSTMs, GRUs) have fundamental limitations:

1. **Sequential computation**: Hidden state at time $t$ depends on state at $t-1$
2. **Limited parallelization**: Training cannot fully utilize GPU parallelism
3. **Gradient issues**: Vanishing/exploding gradients over long sequences
4. **Fixed context compression**: All history compressed into fixed-size hidden state

### The Attention Solution

Attention mechanisms address these issues by:
- Computing direct connections between any two positions
- Enabling full parallelization during training
- Providing explicit paths for gradient flow
- Allowing dynamic, content-based context aggregation

## Causal (Masked) Self-Attention

### Standard Self-Attention

Given input sequence $\mathbf{X} \in \mathbb{R}^{T \times d}$, self-attention computes:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

where $\mathbf{Q} = \mathbf{X}\mathbf{W}_Q$, $\mathbf{K} = \mathbf{X}\mathbf{W}_K$, $\mathbf{V} = \mathbf{X}\mathbf{W}_V$.

### Enforcing Causality

For autoregressive modeling, position $t$ must not attend to positions $> t$. This is achieved through a **causal mask**:

$$\text{CausalAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

where $\mathbf{M}$ is an upper-triangular matrix of $-\infty$:

$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}$$

The $-\infty$ values become 0 after softmax, preventing attention to future positions.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention for autoregressive models.
    
    Each position can only attend to itself and previous positions.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
        # Create causal mask (lower triangular)
        # Register as buffer so it's saved with model but not trained
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer('causal_mask', mask)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional additional mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x)  # [batch, seq_len, d_model]
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        # [batch, seq_len, n_heads, d_k] -> [batch, n_heads, seq_len, d_k]
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        # [batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply causal mask
        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply optional attention mask (e.g., padding)
        if attention_mask is not None:
            # attention_mask: [batch, seq_len] -> [batch, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        
        # Softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # [batch, n_heads, seq_len, d_k]
        context = torch.matmul(attention_weights, V)
        
        # Reshape back
        # [batch, seq_len, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = self.W_o(context)
        
        return output
```

## Transformer Decoder Block

### Architecture

A standard Transformer decoder block consists of:
1. Causal self-attention with residual connection and layer norm
2. Feed-forward network with residual connection and layer norm

```python
class TransformerBlock(nn.Module):
    """
    Single Transformer decoder block.
    
    Structure:
        x -> LayerNorm -> CausalAttention -> + -> LayerNorm -> FFN -> +
             |__________________________|      |___________________|
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        max_seq_len: int = 2048
    ):
        super().__init__()
        
        d_ff = d_ff or 4 * d_model  # Standard: FFN is 4x model dimension
        
        # Layer norms (Pre-LN architecture)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Causal self-attention
        self.attention = CausalSelfAttention(
            d_model, n_heads, max_seq_len, dropout
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input [batch_size, seq_len, d_model]
            attention_mask: Optional mask [batch_size, seq_len]
            
        Returns:
            Output [batch_size, seq_len, d_model]
        """
        # Self-attention with residual
        x = x + self.attention(self.ln1(x), attention_mask)
        
        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        
        return x
```

### Pre-LN vs Post-LN

**Post-LN** (original Transformer):
```
x -> Attention -> + -> LayerNorm -> FFN -> + -> LayerNorm
     |____________|                 |______|
```

**Pre-LN** (GPT-2, modern models):
```
x -> LayerNorm -> Attention -> + -> LayerNorm -> FFN -> +
                               |                        |
                               x ---------------------->|
```

Pre-LN is more stable for training deep models and is now standard.

## GPT Architecture

### Complete Model

```python
class GPT(nn.Module):
    """
    GPT-style autoregressive Transformer.
    
    Architecture:
        - Token embedding + positional embedding
        - Stack of Transformer decoder blocks
        - Output projection to vocabulary
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = None,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection (often tied with token embedding)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token indices [batch_size, seq_len]
            attention_mask: Optional mask [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        
        # Get embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        tok_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for next-token prediction.
        """
        # Shift for next-token prediction
        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        
        # Get logits
        logits = self.forward(inputs, attention_mask[:, :-1] if attention_mask is not None else None)
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100  # Ignore padding
        )
        
        return loss
```

### Model Configurations

```python
# GPT-2 configurations
GPT2_CONFIGS = {
    'gpt2-small': {
        'vocab_size': 50257,
        'd_model': 768,
        'n_layers': 12,
        'n_heads': 12,
        'max_seq_len': 1024
    },
    'gpt2-medium': {
        'vocab_size': 50257,
        'd_model': 1024,
        'n_layers': 24,
        'n_heads': 16,
        'max_seq_len': 1024
    },
    'gpt2-large': {
        'vocab_size': 50257,
        'd_model': 1280,
        'n_layers': 36,
        'n_heads': 20,
        'max_seq_len': 1024
    },
    'gpt2-xl': {
        'vocab_size': 50257,
        'd_model': 1600,
        'n_layers': 48,
        'n_heads': 25,
        'max_seq_len': 1024
    }
}

def create_gpt2(config_name: str) -> GPT:
    """Create GPT-2 model from configuration name."""
    config = GPT2_CONFIGS[config_name]
    return GPT(**config)
```

## Positional Encodings

### Learned Positional Embeddings

GPT uses learned positional embeddings—a simple lookup table:

```python
self.position_embedding = nn.Embedding(max_seq_len, d_model)
```

Advantages:
- Simple and effective
- Learned from data

Limitations:
- Fixed maximum length
- No extrapolation beyond training length

### Sinusoidal Positional Encoding

The original Transformer used fixed sinusoidal encodings:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

```python
class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""
    
    def __init__(self, d_model: int, max_seq_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_seq_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        return x + self.pe[:, :x.size(1)]
```

### Rotary Position Embedding (RoPE)

Modern models (LLaMA, etc.) use RoPE, which encodes position through rotation:

```python
class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    
    Encodes position by rotating query and key vectors.
    Enables better length extrapolation.
    """
    
    def __init__(self, d_model: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Compute rotation frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin
        self._precompute_cache(max_seq_len)
    
    def _precompute_cache(self, seq_len: int):
        """Precompute cos and sin values."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, d_model/2]
        
        # Duplicate for both sin and cos
        emb = torch.cat([freqs, freqs], dim=-1)  # [seq_len, d_model]
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: int
    ) -> tuple:
        """
        Apply rotary embedding to query and key.
        
        Args:
            q: Query tensor [batch, n_heads, seq_len, d_k]
            k: Key tensor [batch, n_heads, seq_len, d_k]
            seq_len: Sequence length
            
        Returns:
            (rotated_q, rotated_k)
        """
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        q_rot = self._apply_rotation(q, cos, sin)
        k_rot = self._apply_rotation(k, cos, sin)
        
        return q_rot, k_rot
    
    def _apply_rotation(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotation to tensor."""
        # Split into two halves
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        
        # Rotate
        rotated = torch.cat([-x2, x1], dim=-1)
        
        return x * cos + rotated * sin
```

## Text Generation

### Autoregressive Sampling

```python
@torch.no_grad()
def generate(
    model: GPT,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = None,
    top_p: float = None,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Generate text autoregressively.
    
    Args:
        model: Trained GPT model
        prompt_ids: Starting token indices [1, prompt_len]
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        device: Device to generate on
        
    Returns:
        Generated token indices [1, prompt_len + max_new_tokens]
    """
    model.eval()
    model = model.to(device)
    
    generated = prompt_ids.clone().to(device)
    
    for _ in range(max_new_tokens):
        # Truncate to max sequence length
        context = generated[:, -model.max_seq_len:]
        
        # Get logits for next token
        logits = model(context)
        next_token_logits = logits[:, -1, :]  # [1, vocab_size]
        
        # Apply temperature
        next_token_logits = next_token_logits / temperature
        
        # Apply top-k filtering
        if top_k is not None:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')
        
        # Apply top-p (nucleus) filtering
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            next_token_logits[indices_to_remove] = float('-inf')
        
        # Sample from distribution
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=1)
        
        # Check for end-of-sequence token (if defined)
        # if next_token.item() == eos_token_id:
        #     break
    
    return generated
```

### KV-Cache for Efficient Generation

During generation, we can cache key-value pairs to avoid redundant computation:

```python
class CausalSelfAttentionWithCache(nn.Module):
    """Self-attention with KV-cache for efficient generation."""
    
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        # Cache for generation
        self.cache_k = None
        self.cache_v = None
    
    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_cache: tuple = None
    ) -> tuple:
        """
        Forward pass with optional caching.
        
        Args:
            x: Input [batch, seq_len, d_model]
            use_cache: Whether to use/update cache
            past_cache: Previous (K, V) cache
            
        Returns:
            (output, cache) where cache is (K, V) if use_cache else None
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V for new tokens
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Append to cache if using
        if past_cache is not None:
            past_k, past_v = past_cache
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        
        # Store cache for next iteration
        cache = (K, V) if use_cache else None
        
        # Compute attention
        total_len = K.size(2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Causal mask only for new positions attending to old
        # All past positions are valid
        if seq_len > 1:
            mask = torch.triu(torch.ones(seq_len, total_len, device=x.device), diagonal=total_len - seq_len + 1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(context)
        
        return output, cache


@torch.no_grad()
def generate_with_cache(
    model,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Efficient generation using KV-cache.
    
    Only computes attention for new tokens, using cached K/V for context.
    """
    model.eval()
    
    # Initial forward pass for prompt
    generated = prompt_ids.clone()
    past_cache = None
    
    for _ in range(max_new_tokens):
        # Only process new token(s)
        if past_cache is None:
            input_ids = generated
        else:
            input_ids = generated[:, -1:]
        
        # Forward with cache
        logits, past_cache = model.forward_with_cache(input_ids, past_cache)
        
        # Sample next token
        next_logits = logits[:, -1, :] / temperature
        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        
        generated = torch.cat([generated, next_token], dim=1)
    
    return generated
```

## Modern Architectural Improvements

### Multi-Query Attention (MQA)

Reduces memory and compute by sharing K, V across heads:

```python
class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention: single K, V shared across all heads.
    
    Reduces KV-cache size by n_heads factor.
    """
    
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Multiple query heads
        self.W_k = nn.Linear(d_model, self.d_k, bias=False)  # Single K
        self.W_v = nn.Linear(d_model, self.d_k, bias=False)  # Single V
        self.W_o = nn.Linear(d_model, d_model, bias=False)
```

### Grouped-Query Attention (GQA)

Compromise between MHA and MQA—groups of heads share K, V:

```python
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention: K, V shared within groups.
    
    n_kv_heads < n_heads provides memory/compute savings.
    """
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
```

### SwiGLU Activation

LLaMA and modern models use SwiGLU in the FFN:

```python
class SwiGLU(nn.Module):
    """
    SwiGLU activation: Swish-Gated Linear Unit.
    
    SwiGLU(x) = Swish(xW1) * (xW2)
    where Swish(x) = x * sigmoid(x)
    """
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)
        self.W3 = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W3(F.silu(self.W1(x)) * self.W2(x))
```

### RMSNorm

More efficient than LayerNorm:

```python
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Simpler and faster than LayerNorm (no mean subtraction).
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

## Training Considerations

### Learning Rate Schedule

```python
def get_lr_schedule(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float
) -> float:
    """
    Cosine learning rate schedule with warmup.
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * step / warmup_steps
    elif step > max_steps:
        return min_lr
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### Gradient Checkpointing

For memory-efficient training of large models:

```python
from torch.utils.checkpoint import checkpoint

class GPTWithCheckpointing(GPT):
    """GPT with gradient checkpointing to reduce memory."""
    
    def forward(self, input_ids, attention_mask=None):
        # ... embedding ...
        
        for block in self.blocks:
            # Checkpoint each block to save memory
            x = checkpoint(block, x, attention_mask)
        
        # ... output projection ...
```

## Applications in Quantitative Finance

### Financial Text Generation

```python
class FinancialGPT(GPT):
    """
    GPT fine-tuned for financial text generation.
    
    Applications:
    - Report summarization
    - News generation for backtesting
    - Scenario narrative generation
    """
    
    def __init__(self, financial_vocab_size: int, **kwargs):
        super().__init__(vocab_size=financial_vocab_size, **kwargs)
        
        # Additional financial entity embeddings
        self.entity_embedding = nn.Embedding(1000, kwargs.get('d_model', 768))
```

### Time Series Forecasting

Transformers can model financial time series:

```python
class TimeSeriesTransformer(nn.Module):
    """
    Transformer for financial time series forecasting.
    
    Treats price history as a "sentence" of continuous values.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        forecast_horizon: int = 10
    ):
        super().__init__()
        
        # Project continuous values to d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])
        
        # Output: predict distribution parameters
        self.output_projection = nn.Linear(d_model, input_dim * 2)  # mean, std
        self.forecast_horizon = forecast_horizon
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Args:
            x: Historical prices [batch, seq_len, input_dim]
            
        Returns:
            (mean, std) for forecast distribution
        """
        h = self.input_projection(x)
        
        for block in self.blocks:
            h = block(h)
        
        params = self.output_projection(h[:, -1, :])
        mean, log_std = params.chunk(2, dim=-1)
        
        return mean, F.softplus(log_std)
```

## Summary

Autoregressive Transformers have become the dominant architecture for sequence modeling:

1. **Causal masking** enables autoregressive modeling with parallel training
2. **Self-attention** provides direct connections between any positions
3. **Positional encodings** (learned, sinusoidal, or rotary) encode sequence order
4. **KV-caching** enables efficient generation
5. **Modern improvements** (GQA, SwiGLU, RMSNorm) enhance efficiency and performance

The combination of expressive power, training efficiency, and scalability has made Transformers the foundation of modern language models and increasingly important in other domains including finance.

## References

1. Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.
2. Radford, A., et al. (2018). Improving Language Understanding by Generative Pre-Training. *OpenAI*.
3. Radford, A., et al. (2019). Language Models are Unsupervised Multitask Learners. *OpenAI*.
4. Brown, T., et al. (2020). Language Models are Few-Shot Learners. *NeurIPS*.
5. Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. *arXiv*.
6. Su, J., et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. *arXiv*.

---

## Exercises

1. **Attention Visualization**: Implement attention head visualization to understand what the model attends to.

2. **Position Encoding Comparison**: Compare learned vs. sinusoidal vs. RoPE embeddings on a synthetic task.

3. **KV-Cache Implementation**: Implement full KV-caching for a GPT model and measure speedup.

4. **Financial Fine-tuning**: Fine-tune a small GPT model on financial news and evaluate generation quality.

5. **Architectural Ablation**: Compare model performance with different attention variants (MHA, MQA, GQA).
