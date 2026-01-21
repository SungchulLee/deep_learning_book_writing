# Transformer Architecture Overview

## Introduction

The Transformer, introduced in "Attention Is All You Need" (Vaswani et al., 2017), revolutionized sequence modeling by replacing recurrence with self-attention. It has become the foundation for virtually all modern NLP models.

## Core Components

A Transformer consists of:

1. **Input Embedding**: Convert tokens to vectors
2. **Positional Encoding**: Inject position information
3. **Encoder Stack**: Process input sequence
4. **Decoder Stack**: Generate output sequence
5. **Output Projection**: Map to vocabulary

## The Original Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Output Probabilities                    │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                    Linear + Softmax                          │
└─────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│                    DECODER STACK (×N)                        │
│    ┌────────────────────────────────────────────────────┐   │
│    │  Masked Self-Attention → Cross-Attention → FFN     │   │
│    └────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
           ↑                                    ↑
    Output Embedding                    ┌──────┴──────┐
           ↑                            │             │
       Outputs                   ┌──────────────────────────┐
     (shifted right)             │                          │
                                 │    ENCODER STACK (×N)    │
                                 │  ┌────────────────────┐  │
                                 │  │ Self-Attention → FFN│  │
                                 │  └────────────────────┘  │
                                 │                          │
                                 └──────────────────────────┘
                                              ↑
                                       Input Embedding
                                              ↑
                                           Inputs
```

## Three Transformer Paradigms

The Transformer architecture has three major variants:

| Architecture | Example Models | Primary Use |
|--------------|----------------|-------------|
| Encoder-only | BERT, RoBERTa | Understanding |
| Decoder-only | GPT, LLaMA, Claude | Generation |
| Encoder-Decoder | T5, BART | Transformation |

### Encoder-Only (BERT-style)

Uses **only the encoder stack** with bidirectional self-attention:

$$\text{Text} \xrightarrow{\text{Embed + PE}} \mathbf{X} \xrightarrow{\text{Encoder} \times N} \mathbf{X}^{(N)}$$

**Key Characteristics:**
- **No causal mask**: Every token sees every other token
- **Bidirectional context**: Full context in both directions
- **No generation**: Designed for understanding, not producing text

**Layer Composition:**
$$\text{Encoder Layer} = \text{Self-Attention} + \text{FFN}$$

**Typical Tasks:** Classification, NER, QA extraction, semantic similarity

**Training:** Masked Language Modeling (MLM)—predict [MASK] tokens using bidirectional context.

### Decoder-Only (GPT-style)

Uses **only the decoder stack** with causal self-attention:

$$\text{Text} \xrightarrow{\text{Embed + PE}} \mathbf{X} \xrightarrow{\text{Decoder} \times N} \mathbf{X}^{(N)} \xrightarrow{\text{Softmax}} P(t_{n+1})$$

**Key Characteristics:**
- **Causal mask**: Position $i$ only sees positions $1, \ldots, i$
- **Autoregressive**: Generates one token at a time, left to right
- **Unified interface**: All tasks become "generate the answer"

**Layer Composition:**
$$\text{Decoder Layer} = \text{Masked Self-Attention} + \text{FFN}$$

**Typical Tasks:** Text generation, chat, code completion, general-purpose via prompting

**Training:** Autoregressive LM—predict next token at every position.

### Encoder-Decoder (T5-style)

Uses **both encoder and decoder** with cross-attention:

$$\text{Source} \xrightarrow{\text{Encoder}} \mathbf{M} \xrightarrow{\text{Decoder (with cross-attention)}} \text{Target}$$

**Key Characteristics:**
- **Encoder**: Bidirectional self-attention (no mask)
- **Decoder**: Causal self-attention + cross-attention to encoder
- **Memory M**: Encoder output that decoder queries

**Layer Composition:**
$$\text{Encoder Layer} = \text{Self-Attention} + \text{FFN}$$
$$\text{Decoder Layer} = \text{Masked Self-Attention} + \text{Cross-Attention} + \text{FFN}$$

**Typical Tasks:** Translation, summarization, any sequence-to-sequence

## Side-by-Side Comparison

| Aspect | Encoder-Only | Decoder-Only | Encoder-Decoder |
|--------|--------------|--------------|-----------------|
| Self-Attention | Bidirectional | Causal (masked) | Encoder: bidir, Decoder: causal |
| Cross-Attention | None | None | Decoder → Encoder |
| Generation | No | Yes | Yes |
| Understanding | Yes | Limited | Yes |
| Sublayers/layer | 2 | 2 | Encoder: 2, Decoder: 3 |
| Typical depth | 12-24 layers | 12-96+ layers | 6-12 each |

## Attention Types Summary

| Attention Type | Mask | Used In | Purpose |
|----------------|------|---------|---------|
| Self-Attention | No | Encoder (BERT) | Bidirectional understanding |
| Self-Attention | Yes (causal) | Decoder (GPT) | Autoregressive generation |
| Cross-Attention | No | Decoder (T5) | Query encoder memory |

## Encoder Layer Details

Each encoder layer has two sublayers:

```
Input X
    │
    ▼
┌─────────────────────────┐
│   Multi-Head Self-Attn  │
└─────────────────────────┘
    │
    ├──────────────────────┐
    ▼                      │
┌────────┐                 │
│   +    │◄────────────────┘ (Residual)
└────────┘
    │
    ▼
┌─────────────────────────┐
│      Layer Norm         │
└─────────────────────────┘
    │
    ▼
┌─────────────────────────┐
│   Feed-Forward Network  │
└─────────────────────────┘
    │
    ├──────────────────────┐
    ▼                      │
┌────────┐                 │
│   +    │◄────────────────┘ (Residual)
└────────┘
    │
    ▼
┌─────────────────────────┐
│      Layer Norm         │
└─────────────────────────┘
    │
    ▼
Output X'
```

Mathematically:
$$\mathbf{X}' = \text{LayerNorm}(\mathbf{X} + \text{MultiHeadAttn}(\mathbf{X}))$$
$$\mathbf{X}'' = \text{LayerNorm}(\mathbf{X}' + \text{FFN}(\mathbf{X}'))$$

## Decoder Layer Details

Each decoder layer has three sublayers:

1. **Masked Self-Attention**: Attend to previous decoder positions
2. **Cross-Attention**: Attend to encoder output
3. **Feed-Forward Network**: Position-wise transformation

$$\mathbf{Y}' = \text{LayerNorm}(\mathbf{Y} + \text{MaskedSelfAttn}(\mathbf{Y}))$$
$$\mathbf{Y}'' = \text{LayerNorm}(\mathbf{Y}' + \text{CrossAttn}(\mathbf{Y}', \mathbf{M}))$$
$$\mathbf{Y}''' = \text{LayerNorm}(\mathbf{Y}'' + \text{FFN}(\mathbf{Y}''))$$

## Model Dimensions

Standard configurations:

| Component | Base | Large |
|-----------|------|-------|
| $d_{\text{model}}$ | 512 | 1024 |
| $d_{ff}$ | 2048 | 4096 |
| Heads | 8 | 16 |
| Layers | 6 | 6 |
| $d_k = d_v$ | 64 | 64 |

## Modern Trend: Decoder-Only Dominance

Decoder-only models now dominate because:

1. **Simpler architecture**: One stack instead of two
2. **Unified interface**: Everything becomes "generate the answer"
3. **Scales better**: Easier to train very large models
4. **Surprisingly general**: Can do understanding tasks via prompting

Example—translation with decoder-only:
```
Prompt: "Translate to English: Le chat noir"
Output: "The black cat"
```

No separate encoder needed—the decoder handles everything through the unified prompt interface.

## Summary

The Transformer architecture:

$$\text{BERT} \approx \text{Encoder only}$$
$$\text{GPT} \approx \text{Decoder only}$$
$$\text{T5} \approx \text{Encoder-Decoder}$$

Each paradigm excels at different tasks, but decoder-only models have become dominant due to their flexibility, scalability, and unified interface for diverse tasks.

## References

- Vaswani et al., "Attention Is All You Need" (2017)
- Devlin et al., "BERT" (2019)
- Radford et al., "Language Models are Unsupervised Multitask Learners" (GPT-2, 2019)
- Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer" (T5, 2020)
