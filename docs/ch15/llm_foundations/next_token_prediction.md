# Next-Token Prediction

## Learning Objectives

- Formalize the autoregressive language modeling objective
- Derive the cross-entropy training loss and its connection to perplexity
- Implement and compare decoding strategies
- Understand why next-token prediction produces general-purpose capabilities

## The Autoregressive Objective

The foundation of all decoder-only LLMs is **next-token prediction**: given a sequence of tokens, predict the next one. For a sequence $\mathbf{x} = (x_1, x_2, \ldots, x_T)$, the model factorizes the joint probability autoregressively:

$$P(\mathbf{x}) = \prod_{t=1}^{T} P(x_t \mid x_{<t}; \theta)$$

where $x_{<t} = (x_1, \ldots, x_{t-1})$ and $\theta$ denotes model parameters.

## Training Loss: Cross-Entropy

The training objective minimizes the negative log-likelihood over the training corpus $\mathcal{D}$:

$$\mathcal{L}(\theta) = -\frac{1}{|\mathcal{D}|} \sum_{\mathbf{x} \in \mathcal{D}} \sum_{t=1}^{T} \log P(x_t \mid x_{<t}; \theta)$$

This is equivalent to minimizing the cross-entropy between the true data distribution and the model distribution:

$$H(p_{\text{data}}, p_\theta) = -\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[\log p_\theta(\mathbf{x})\right]$$

### Perplexity

**Perplexity** is the standard evaluation metric, defined as the exponentiated average negative log-likelihood:

$$\text{PPL}(\mathbf{x}) = \exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(x_t \mid x_{<t}; \theta)\right)$$

Interpretation: a perplexity of $k$ means the model is, on average, as uncertain as choosing uniformly among $k$ options at each step. Lower perplexity indicates a better model.

| Model | Parameters | Perplexity (WikiText-103) |
|-------|-----------|--------------------------|
| GPT-2 Small | 117M | 29.4 |
| GPT-2 Large | 774M | 22.0 |
| GPT-2 XL | 1.5B | 17.5 |

## Decoding Strategies

At inference time, we need to convert the predicted probability distribution into actual token sequences.

### Greedy Decoding

Select the highest-probability token at each step:

$$x_t = \arg\max_{x} P(x \mid x_{<t})$$

Simple but often produces repetitive, low-quality text.

### Temperature Sampling

Adjust the logit distribution sharpness before sampling:

$$P(x_t = v \mid x_{<t}) = \frac{\exp(z_v / \tau)}{\sum_{v'} \exp(z_{v'} / \tau)}$$

where $z_v$ are logits and $\tau$ is the temperature:

- $\tau \to 0$: approaches greedy decoding
- $\tau = 1$: standard sampling from the learned distribution
- $\tau > 1$: more uniform (more random) sampling

### Top-k Sampling

Restrict sampling to the $k$ highest-probability tokens:

$$P'(x_t = v) = \begin{cases} \frac{P(x_t = v)}{\sum_{v' \in V_k} P(x_t = v')} & \text{if } v \in V_k \\ 0 & \text{otherwise} \end{cases}$$

### Nucleus (Top-p) Sampling

Dynamically select the smallest set of tokens whose cumulative probability exceeds threshold $p$:

$$V_p = \min\left\{V' \subseteq V : \sum_{v \in V'} P(x_t = v) \geq p\right\}$$

```python
import torch
import torch.nn.functional as F


def nucleus_sampling(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0):
    # Apply temperature
    logits = logits / temperature

    # Sort in descending order
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above threshold
    sorted_indices_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
    sorted_logits[sorted_indices_to_remove] = float('-inf')

    # Sample from filtered distribution
    probs = F.softmax(sorted_logits, dim=-1)
    next_token_sorted = torch.multinomial(probs, num_samples=1)

    # Map back to original indices
    next_token = sorted_indices[next_token_sorted]
    return next_token
```

## Why Next-Token Prediction Works

A natural question: why does the simple objective of predicting the next token produce models capable of reasoning, translation, code generation, and more?

The key insight is that **predicting the next token well requires learning the structure of the world that generated the text**. Consider:

> "The capital of France is ___"

To correctly predict "Paris," the model must encode geographic knowledge. For:

> "If $x^2 + 3x - 4 = 0$, then $x = ___$"

The model must learn algebraic manipulation. The diversity and scale of internet text means that minimizing next-token prediction loss implicitly requires learning syntax, world knowledge, logical reasoning patterns, mathematical relationships, and programming language semantics.

## Quantitative Finance Application

```python
def generate_financial_analysis(model, tokenizer, prompt, max_tokens=512):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = nucleus_sampling(next_token_logits, p=0.9, temperature=0.7)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0], skip_special_tokens=True)
```

## References

1. Radford, A., et al. (2019). "Language Models are Unsupervised Multitask Learners."
2. Holtzman, A., et al. (2020). "The Curious Case of Neural Text Degeneration." *ICLR*.
3. Fan, A., et al. (2018). "Hierarchical Neural Story Generation." *ACL*.
