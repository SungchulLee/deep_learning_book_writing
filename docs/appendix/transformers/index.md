# A6 Transformer Architectures

## Overview

This appendix provides complete PyTorch implementations of transformer architectures that revolutionized natural language processing and, increasingly, all of deep learning. Starting from the original "Attention Is All You Need" model, the implementations trace the divergence into encoder-only (BERT family), decoder-only (GPT family), encoder–decoder (T5), and speech (Wav2Vec) paradigms. These architectures are central to quantitative finance for processing textual data — earnings calls, SEC filings, analyst reports, news — and increasingly for multi-modal financial AI systems.

## Architectures

### Foundational

| Model | Year | Key Innovation | Type |
|-------|------|----------------|------|
| [Original Transformer](transformer.py) | 2017 | Self-attention replacing recurrence entirely | Encoder–Decoder |

### Encoder-Only (Bidirectional)

| Model | Year | Key Innovation | Type |
|-------|------|----------------|------|
| [BERT](bert.py) | 2018 | Masked language modeling + next sentence prediction | Encoder |
| [RoBERTa](roberta.py) | 2019 | Optimized BERT training: larger batches, more data, no NSP | Encoder |
| [ALBERT](albert.py) | 2019 | Parameter sharing, factorized embedding | Encoder |
| [DistilBERT](distilbert.py) | 2019 | Knowledge distillation for 60% size, 97% performance | Encoder |
| [Longformer](longformer.py) | 2020 | Sliding window + global attention for long documents | Encoder |

### Decoder-Only (Autoregressive)

| Model | Year | Key Innovation | Type |
|-------|------|----------------|------|
| [GPT Family](gpt.py) | 2018–2023 | Autoregressive language modeling, scaling laws, RLHF | Decoder |
| [LLaMA](llama.py) | 2023 | RMSNorm, RoPE, SwiGLU, grouped-query attention | Decoder |

### Encoder–Decoder

| Model | Year | Key Innovation | Type |
|-------|------|----------------|------|
| [T5](t5.py) | 2019 | Unified text-to-text framework for all NLP tasks | Encoder–Decoder |

### Speech

| Model | Year | Key Innovation | Type |
|-------|------|----------------|------|
| [Wav2Vec](wav2vec.py) | 2020 | Self-supervised speech representations via contrastive learning | Encoder |

## Key Concepts

### Self-Attention Mechanism

The core operation computes weighted sums over all positions in a sequence:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

- **Multi-head attention**: Parallel attention heads capture different relationship types
- **Causal masking**: Decoder-only models mask future positions to maintain autoregressive property
- **Cross-attention**: Decoder attends to encoder outputs in encoder–decoder architectures

### Pre-training Paradigms

| Paradigm | Models | Objective | Strengths |
|----------|--------|-----------|-----------|
| Masked Language Modeling (MLM) | BERT, RoBERTa | Predict masked tokens from bidirectional context | Understanding, classification |
| Causal Language Modeling (CLM) | GPT, LLaMA | Predict next token left-to-right | Generation, few-shot learning |
| Span Corruption | T5 | Reconstruct corrupted text spans | Flexible text-to-text tasks |
| Contrastive (speech) | Wav2Vec | Distinguish true from distractor speech frames | Speech understanding without transcripts |

### Efficiency and Scaling

- **Distillation**: DistilBERT compresses BERT while retaining most performance
- **Parameter sharing**: ALBERT shares weights across layers, reducing model size
- **Long-context**: Longformer uses $O(n)$ windowed attention instead of $O(n^2)$ full attention
- **Modern decoder optimizations**: LLaMA introduces RoPE, GQA, and SwiGLU for efficient scaling

## Quantitative Finance Applications

- **Sentiment analysis**: BERT/RoBERTa fine-tuned on financial text (FinBERT) for earnings calls and news
- **Document understanding**: Longformer for processing full-length 10-K/10-Q filings
- **Text generation**: GPT/LLaMA for automated report drafting, investment memo generation
- **Named entity recognition**: Extract companies, amounts, dates from financial documents
- **Summarization**: T5 for condensing lengthy analyst reports and regulatory filings
- **Earnings call analysis**: Wav2Vec for processing audio recordings of earnings calls, extracting vocal sentiment cues
- **Multi-task learning**: T5's text-to-text framework for unified financial NLP pipelines

## Prerequisites

- [A5: Sequence Models](../sequence/index.md) — RNN-based seq2seq and attention origins
- [A10: Utility Modules — Attention Mechanisms](../utils/attention.py) — multi-head attention implementation details
- [A10: Utility Modules — Positional Encodings](../utils/positional.py) — sinusoidal, learned, and rotary position encodings
- [A10: Utility Modules — Normalization Layers](../utils/normalization.py) — LayerNorm, RMSNorm
