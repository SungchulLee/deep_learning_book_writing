# Attention Mechanisms: Complete Python Implementation

A comprehensive collection of attention mechanism implementations in PyTorch, from basic attention to advanced self-attention and cross-attention used in Transformers.

## 📚 Contents

### 1. `attention_basics.py`
Foundation attention mechanisms:
- **BasicAttention**: Additive (Bahdanau) attention mechanism
- **ScaledDotProductAttention**: The fundamental building block of Transformers

### 2. `self_attention.py`
Self-attention implementations:
- **SelfAttention**: Basic self-attention layer
- **MultiHeadSelfAttention**: Multi-head self-attention (Transformer encoder)
- **CausalSelfAttention**: Masked self-attention for autoregressive models (GPT-style)

### 3. `cross_attention.py`
Cross-attention for encoder-decoder architectures:
- **CrossAttention**: Basic cross-attention layer
- **MultiHeadCrossAttention**: Multi-head cross-attention
- **EncoderDecoderAttention**: Complete decoder block with self and cross-attention

### 4. `examples.py`
Practical applications:
- Complete Seq2Seq model with attention
- Vision Transformer (ViT) style attention
- Text generation with causal attention
- Attention pattern visualization

## 🚀 Quick Start

### Installation
```bash
pip install torch numpy matplotlib --break-system-packages
```

### Running Examples

```bash
# Run basic attention demonstrations
python attention_basics.py

# Run self-attention examples
python self_attention.py

# Run cross-attention examples
python cross_attention.py

# Run comprehensive examples
python examples.py
```

## 📖 Key Concepts

### Self-Attention
Queries, keys, and values all come from the same sequence:
```
Input: [x₁, x₂, x₃, ..., xₙ]
Q = K = V = Input
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```

**Use cases:**
- Encoder layers (BERT, RoBERTa)
- Understanding relationships within a sequence
- Image patches in Vision Transformers

### Cross-Attention
Queries come from one sequence, keys/values from another:
```
Decoder: [y₁, y₂, y₃, ..., yₘ]
Encoder: [x₁, x₂, x₃, ..., xₙ]
Q = Decoder, K = V = Encoder
```

**Use cases:**
- Machine translation (Transformer decoder)
- Image captioning (image → text)
- Speech recognition (audio → text)

### Causal (Masked) Attention
Prevents future positions from being attended to:
```
Position i can only attend to positions ≤ i
Used in: GPT, GPT-2, GPT-3, Llama
```

## 🏗️ Architecture Overview

### Attention Formula
```
Attention(Q, K, V) = softmax(QK^T / √d_k) * V
```

Where:
- **Q** (Query): "What am I looking for?"
- **K** (Key): "What do I have?"
- **V** (Value): "What information do I provide?"
- **√d_k**: Scaling factor (dimension of keys)

### Multi-Head Attention
```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ)W^O

where headᵢ = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

Benefits:
- Multiple representation subspaces
- Capture different types of relationships
- More expressive model

## 💡 Usage Examples

### Basic Self-Attention
```python
from self_attention import MultiHeadSelfAttention

# Input: batch of sequences
x = torch.randn(batch_size=2, seq_len=10, embed_dim=64)

# Create attention layer
attn = MultiHeadSelfAttention(embed_dim=64, num_heads=8)

# Forward pass
output, attention_weights = attn(x)
```

### Cross-Attention (Encoder-Decoder)
```python
from cross_attention import MultiHeadCrossAttention

# Decoder queries
query = torch.randn(batch_size=2, decoder_len=5, query_dim=64)

# Encoder keys/values
encoder_out = torch.randn(batch_size=2, encoder_len=10, key_dim=64)

# Create cross-attention
cross_attn = MultiHeadCrossAttention(
    query_dim=64, key_dim=64, embed_dim=64, num_heads=8
)

# Forward pass
output, attention_weights = cross_attn(query, encoder_out)
```

### Complete Seq2Seq Model
```python
from examples import SimpleSeq2Seq

# Create model
model = SimpleSeq2Seq(
    src_vocab_size=1000,
    tgt_vocab_size=1000,
    embed_dim=256,
    num_heads=8,
    num_layers=6
)

# Forward pass
logits = model(src_tokens, tgt_tokens, src_mask, tgt_mask)
```

## 🔍 Understanding the Code

### Shapes Guide

**Self-Attention:**
```
Input:  (batch, seq_len, embed_dim)
Output: (batch, seq_len, embed_dim)
Weights: (batch, num_heads, seq_len, seq_len)
```

**Cross-Attention:**
```
Query:  (batch, query_len, query_dim)
Key/Value: (batch, key_len, key_dim)
Output: (batch, query_len, embed_dim)
Weights: (batch, num_heads, query_len, key_len)
```

### Masking

**Padding Mask:**
```python
mask = (seq != pad_idx)  # Ignore padding tokens
```

**Causal Mask:**
```python
mask = torch.tril(torch.ones(L, L))  # Lower triangular
```

## 🎯 Key Features

- ✅ Clean, educational implementations
- ✅ Detailed comments and docstrings
- ✅ Shape annotations throughout
- ✅ Runnable demonstrations
- ✅ No external dependencies (except PyTorch)
- ✅ Ready for learning and experimentation

## 📊 Performance Tips

1. **Multi-head attention**: Use heads that divide evenly into embed_dim
2. **Scaled dot-product**: Always scale by √d_k to prevent gradient issues
3. **Dropout**: Apply after softmax in attention weights
4. **Layer normalization**: Use pre-norm or post-norm consistently
5. **Causal masking**: Use registered buffers for efficiency

## 🎓 Learning Path

1. Start with `attention_basics.py` - understand core concepts
2. Move to `self_attention.py` - learn self-attention patterns
3. Study `cross_attention.py` - understand encoder-decoder interaction
4. Explore `examples.py` - see real-world applications

## 📝 Mathematical Background

### Attention Score Computation
```
score(qᵢ, kⱼ) = qᵢ · kⱼ / √d_k
```

### Attention Weights (via Softmax)
```
αᵢⱼ = exp(score(qᵢ, kⱼ)) / Σⱼ exp(score(qᵢ, kⱼ))
```

### Output Computation
```
outputᵢ = Σⱼ αᵢⱼ · vⱼ
```

## 🔗 References

- **Attention Is All You Need** (Vaswani et al., 2017)
- **BERT** (Devlin et al., 2018)
- **GPT-2** (Radford et al., 2019)
- **Vision Transformer** (Dosovitskiy et al., 2020)

## 📜 License

MIT License - Free for educational and commercial use.

## 🤝 Contributing

Feel free to:
- Add more attention variants
- Improve documentation
- Add visualization tools
- Optimize implementations

## ⚠️ Notes

- These implementations prioritize clarity over maximum performance
- For production use, consider established libraries (HuggingFace Transformers, PyTorch built-ins)
- GPU recommended for large-scale experiments

---

**Happy Learning! 🚀**

For questions or improvements, feel free to explore and modify the code.
