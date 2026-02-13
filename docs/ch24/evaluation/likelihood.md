# Likelihood Evaluation

## Overview

Autoregressive models provide exact log-likelihoods, making likelihood-based evaluation straightforward. This section covers how to compute, report, and interpret likelihood metrics.

## Computing Log-Likelihood

$$\log p(x) = \sum_{t=1}^{T} \log p(x_t \mid x_{<t})$$

Each term is the log-probability assigned by the model's softmax output to the actual next token.

```python
def compute_log_likelihood(model, x):
    logits = model(x[:, :-1])  # (B, T-1, V)
    log_probs = F.log_softmax(logits, dim=-1)
    
    targets = x[:, 1:]  # (B, T-1)
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    
    return token_log_probs.sum(dim=-1)  # (B,) per-sequence log-likelihood
```

## Bits Per Dimension (BPD)

The standard metric for comparing density models across different data dimensionalities:

$$\text{BPD} = -\frac{1}{d \ln 2} \sum_{t=1}^{d} \log p(x_t \mid x_{<t})$$

For images: $d = H \times W \times C$.

## Perplexity

The standard metric for language models:

$$\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^{T} \log p(x_t \mid x_{<t})\right)$$

Lower perplexity indicates better modeling. Perplexity can be interpreted as the effective vocabulary size the model is uncertain about at each position.

## Evaluation Pitfalls

- **Test set contamination**: ensure no overlap between training and test data
- **Tokenization effects**: different tokenizations yield different perplexities; compare only with the same tokenizer
- **Sequence length**: evaluate on full sequences, not truncated
