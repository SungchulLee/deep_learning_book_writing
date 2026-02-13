# Language Modeling Foundations

## Overview

Language modeling is the task of assigning probabilities to sequences of words. This fundamental capability underlies virtually all modern NLP applications: text generation, machine translation, speech recognition, and more.

This section provides a comprehensive treatment of language modeling, from classical statistical methods to modern neural approaches.

---

## Section Contents

### 1. [N-gram Language Models](ngram.md)
- Markov assumption and chain rule
- Unigram, bigram, and trigram models
- Maximum Likelihood Estimation
- Smoothing techniques (Laplace, Add-k, Interpolation, Kneser-Ney)
- Text generation with sampling strategies

### 2. [Neural Language Models](neural_lm.md)
- Feedforward neural language models (Bengio et al.)
- RNN language models with variable-length context
- LSTM language models for long-range dependencies
- Transformer language models (GPT-style)
- Comparison of architectures

### 3. [Perplexity](perplexity.md)
- Information-theoretic foundations
- Cross-entropy and perplexity
- Computing perplexity in PyTorch
- Bits-per-character metric
- Standard benchmarks (PTB, WikiText)

### 4. [Causal vs Masked Language Modeling](causal_vs_masked.md)
- Autoregressive (causal) language modeling
- Masked language modeling (BERT-style)
- Architectural implications
- Task-specific selection criteria

### 5. [Tokenization](tokenization.md)
- Word-level and character-level tokenization
- Byte Pair Encoding (BPE)
- WordPiece algorithm
- SentencePiece and unigram models
- Practical usage with HuggingFace

---

## Learning Path

```
Week 1: N-gram Models
├── Statistical foundations
├── Smoothing techniques
└── Text generation basics

Week 2: Neural Language Models
├── Feedforward LMs
├── RNN/LSTM LMs
└── Transformer LMs

Week 3: Evaluation & Tokenization
├── Perplexity computation
├── Causal vs masked objectives
└── Subword tokenization
```

---

## Prerequisites

- Python programming
- Basic probability and statistics
- Linear algebra fundamentals
- PyTorch basics (tensors, autograd, nn.Module)
- Familiarity with embeddings and neural networks

---

## Key Concepts Summary

| Concept | Definition |
|---------|------------|
| Language Model | Probability distribution over sequences |
| Perplexity | $2^{-\frac{1}{N}\sum\log_2 P(w_i)}$ |
| Causal LM | Left-to-right prediction (GPT) |
| Masked LM | Bidirectional with masked tokens (BERT) |
| BPE | Iterative subword merging algorithm |

---

## Benchmark Results

### Penn Treebank Perplexity

| Model | PPL | Year |
|-------|-----|------|
| Kneser-Ney 5-gram | 141 | 1995 |
| LSTM (2-layer) | 78 | 2016 |
| AWD-LSTM | 57 | 2017 |
| Transformer-XL | 54 | 2019 |
| GPT-2 | ~35 | 2019 |

---

## Applications

Language models power:

1. **Text Generation**: Creative writing, code completion
2. **Machine Translation**: Encoder-decoder models
3. **Speech Recognition**: Acoustic + language model
4. **Dialogue Systems**: Conversational AI
5. **Information Retrieval**: Query understanding
6. **Summarization**: Abstractive summaries
7. **Question Answering**: Reading comprehension

---

## References

1. Jurafsky, D., & Martin, J. H. (2023). *Speech and Language Processing* (3rd ed.).
2. Bengio, Y., et al. (2003). A neural probabilistic language model. *JMLR*.
3. Mikolov, T., et al. (2010). Recurrent neural network based language model.
4. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
5. Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers.
6. Radford, A., et al. (2019). Language models are unsupervised multitask learners.
