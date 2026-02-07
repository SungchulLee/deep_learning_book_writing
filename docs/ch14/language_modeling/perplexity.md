# Perplexity and Language Model Evaluation

## Learning Objectives

By the end of this section, you will be able to:

- Understand perplexity as the standard intrinsic evaluation metric
- Derive perplexity from information theory principles
- Compute perplexity for various model architectures
- Interpret perplexity scores and their limitations

---

## Introduction

Evaluating language models requires answering: "How well does this model predict language?" Perplexity is the gold standard intrinsic metric.

---

## Information-Theoretic Foundations

### Entropy and Cross-Entropy

The **entropy** of a random variable $X$ measures uncertainty:

$$H(X) = -\sum_{x} p(x) \log_2 p(x)$$

The **cross-entropy** between true distribution $p$ and model $q$:

$$H(p, q) = -\sum_{x} p(x) \log_2 q(x) = \mathbb{E}_{p}[-\log_2 q(X)]$$

For language modeling, we estimate cross-entropy on test data:

$$H(p, q) \approx -\frac{1}{N} \sum_{i=1}^{N} \log_2 P_{model}(w_i | w_1, \ldots, w_{i-1})$$

---

## Perplexity Definition

**Perplexity** is the exponentiation of cross-entropy:

$$\text{PPL} = 2^{H(p, q)} = 2^{-\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | \text{context})}$$

Equivalently using natural log:

$$\text{PPL} = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \ln P(w_i | \text{context})\right)$$

### Interpretation

Perplexity represents the **average branching factor**—the effective number of equally likely choices at each step.

| Perplexity | Meaning |
|------------|---------|
| PPL = 1 | Perfect prediction |
| PPL = V | Uniform over vocabulary |
| PPL = 100 | Choosing from ~100 words |

**Lower perplexity = better model**

---

## Computing Perplexity in PyTorch

```python
import torch
import torch.nn as nn
import math

def compute_perplexity(model, data_loader, vocab_size, pad_idx=0):
    """Compute perplexity for neural language model."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum', ignore_index=pad_idx)
    
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            logits = model(inputs)
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
            
            loss = criterion(logits, targets)
            non_pad = (targets != pad_idx).sum().item()
            
            total_loss += loss.item()
            total_tokens += non_pad
    
    avg_nll = total_loss / total_tokens
    return math.exp(avg_nll)
```

---

## Standard Benchmarks

### Penn Treebank Results

| Model | Test PPL |
|-------|----------|
| Kneser-Ney 5-gram | 141 |
| LSTM (2-layer) | 78 |
| AWD-LSTM | 57 |
| Transformer-XL | 54 |
| GPT-2 (small) | ~35 |

---

## Bits-Per-Character (BPC)

Alternative metric normalized by character count:

$$\text{BPC} = -\frac{1}{C} \sum_{i=1}^{N} \log_2 P(w_i | \text{context})$$

Useful for comparing across tokenization schemes.

---

## Key Formulas

| Metric | Formula |
|--------|---------|
| Perplexity | $\text{PPL} = 2^{-\frac{1}{N}\sum \log_2 P(w_i)}$ |
| Cross-Entropy | $H = -\frac{1}{N}\sum \log_2 P(w_i)$ |
| Relationship | $\text{PPL} = 2^H = e^{NLL}$ |

---

## Exercises

1. Manually compute perplexity for a sentence given model probabilities
2. Compare perplexities with different smoothing techniques
3. Analyze how vocabulary size affects perplexity

---

## References

1. Jelinek, F., et al. (1977). Perplexity—a measure of difficulty. *JASA*.
2. Chen, S. F., & Goodman, J. (1999). Smoothing techniques for language modeling.
