# Controlled Generation

## Learning Objectives

- Understand techniques for steering language model outputs toward desired attributes
- Implement prompt engineering, constrained decoding, and RLHF-based alignment
- Apply classifier guidance and PPLM-style control at inference time
- Design guardrails for safe and on-topic generation in production systems

---

## Introduction

Controlled generation refers to techniques that steer a language model's output toward desired properties—specific topics, styles, safety constraints, or factual accuracy—without retraining the base model from scratch. This is crucial for deploying language models in real-world applications where unconstrained generation is insufficient.

---

## Taxonomy of Control Methods

Control methods operate at different stages of the generation pipeline:

| Stage | Method | Modifies |
|-------|--------|----------|
| Pre-training | Conditional training | Model weights |
| Fine-tuning | RLHF, DPO, instruction tuning | Model weights |
| Prompt-time | Prompt engineering, system prompts | Input |
| Decode-time | Constrained decoding, classifier guidance | Sampling |
| Post-generation | Filtering, reranking | Output selection |

---

## Prompt-Based Control

The simplest approach uses carefully crafted prompts to guide generation:

```python
def controlled_prompt(attribute: str, content: str) -> str:
    """Construct a prompt that steers generation toward desired attribute."""
    templates = {
        "formal": f"Write the following in formal academic style:\n{content}\n\nFormal version:",
        "summary": f"Summarize the following text concisely:\n{content}\n\nSummary:",
        "positive": f"Rewrite with a positive sentiment:\n{content}\n\nPositive version:",
        "financial": f"Analyze the following from a quantitative finance perspective:\n{content}\n\nAnalysis:",
    }
    return templates.get(attribute, f"{attribute}:\n{content}\n\nOutput:")
```

### System Prompts and Instruction Following

Modern LLMs use system prompts to establish behavioral constraints:

```python
messages = [
    {"role": "system", "content": "You are a financial analyst. Respond only with factual, quantitative analysis. Never provide investment advice."},
    {"role": "user", "content": "Analyze AAPL's Q3 earnings."}
]
```

---

## Constrained Decoding

### Logit Manipulation

Directly modify logits before sampling to enforce constraints:

```python
import torch
import torch.nn.functional as F
from typing import Set, Optional, Callable

def constrained_decode(
    logits: torch.Tensor,
    allowed_token_ids: Optional[Set[int]] = None,
    banned_token_ids: Optional[Set[int]] = None,
    required_next: Optional[int] = None,
) -> torch.Tensor:
    """
    Apply hard constraints to logits before sampling.

    Args:
        logits: Raw model logits [vocab_size]
        allowed_token_ids: If set, only these tokens can be sampled
        banned_token_ids: These tokens cannot be sampled
        required_next: Force this specific token

    Returns:
        Modified logits
    """
    if required_next is not None:
        mask = torch.full_like(logits, float('-inf'))
        mask[required_next] = logits[required_next]
        return mask

    if allowed_token_ids is not None:
        mask = torch.full_like(logits, float('-inf'))
        for tid in allowed_token_ids:
            mask[tid] = logits[tid]
        return mask

    if banned_token_ids is not None:
        for tid in banned_token_ids:
            logits[tid] = float('-inf')

    return logits
```

### Grammar-Constrained Generation

For structured outputs (JSON, code, SQL), constrain generation to follow a grammar:

```python
class GrammarConstraint:
    """Constrain generation to valid JSON structure."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.state = "start"
        self.bracket_depth = 0

    def get_allowed_tokens(self) -> Set[int]:
        """Return set of valid next token IDs given current parse state."""
        if self.state == "start":
            return self._tokens_for("{")
        elif self.state == "key":
            return self._tokens_for('"') | self._tokens_for("}")
        elif self.state == "value":
            return (self._tokens_for('"') | self._tokens_for_digits() |
                    self._tokens_for("{") | self._tokens_for("["))
        # ... additional states
        return set(range(len(self.tokenizer)))

    def _tokens_for(self, char: str) -> Set[int]:
        """Find all token IDs that start with given character."""
        result = set()
        for tid in range(len(self.tokenizer)):
            token = self.tokenizer.decode([tid])
            if token.startswith(char):
                result.add(tid)
        return result
```

---

## Classifier-Guided Generation (PPLM)

Plug and Play Language Models (PPLM) use an external classifier to guide generation without modifying the LM:

$$\tilde{H}_t = H_t + \alpha \cdot \nabla_{H_t} \log p(\text{attribute} \mid H_t)$$

where $H_t$ are the hidden states and the gradient pushes representations toward the desired attribute.

```python
def pplm_step(
    model,
    classifier,
    hidden_states: torch.Tensor,
    attribute_label: int,
    step_size: float = 0.01,
    num_iterations: int = 3,
) -> torch.Tensor:
    """
    PPLM-style perturbation of hidden states.

    Args:
        model: Language model
        classifier: Attribute classifier on hidden states
        hidden_states: Current hidden states [batch, seq_len, hidden_dim]
        attribute_label: Target attribute class index
        step_size: Gradient step size
        num_iterations: Number of gradient ascent steps

    Returns:
        Perturbed hidden states
    """
    perturbed = hidden_states.clone().detach().requires_grad_(True)

    for _ in range(num_iterations):
        # Classify current hidden states
        logits = classifier(perturbed[:, -1, :])
        log_probs = F.log_softmax(logits, dim=-1)

        # Gradient toward desired attribute
        loss = log_probs[:, attribute_label]
        loss.backward()

        # Update hidden states
        with torch.no_grad():
            grad = perturbed.grad
            perturbed = perturbed + step_size * grad
            perturbed = perturbed.detach().requires_grad_(True)

    return perturbed.detach()
```

---

## RLHF and Alignment

Reinforcement Learning from Human Feedback aligns model outputs with human preferences:

### Reward Model Training

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]$$

where $y_w$ is the preferred and $y_l$ the dispreferred completion.

### PPO Fine-Tuning

The policy is updated to maximize the reward while staying close to the reference model:

$$\mathcal{L}_{PPO} = \mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\text{ref}}} A_t,\; \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{ref}}}, 1-\epsilon, 1+\epsilon\right) A_t\right)\right] - \beta \, D_{KL}\left[\pi_\theta \| \pi_{\text{ref}}\right]$$

### Direct Preference Optimization (DPO)

DPO eliminates the separate reward model by directly optimizing the policy:

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]$$

---

## Applications to Quantitative Finance

Controlled generation is particularly valuable in financial NLP:

- **Report Generation**: Constrain outputs to use only verified data and proper financial terminology
- **Risk Disclosure**: Ensure generated text covers required regulatory topics
- **Sentiment-Controlled Summaries**: Generate summaries with calibrated sentiment indicators
- **Structured Output**: Force generation of valid JSON for downstream quantitative pipelines

---

## Summary

1. **Prompt engineering** is the simplest control mechanism but offers limited guarantees
2. **Constrained decoding** provides hard guarantees on output structure
3. **Classifier guidance** (PPLM) steers generation toward attributes without retraining
4. **RLHF/DPO** provides the most robust alignment but requires preference data and training
5. **Combining methods** (prompts + constrained decoding + post-filtering) is standard practice

---

## References

1. Dathathri, S., et al. (2020). Plug and Play Language Models. *ICLR*.
2. Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*.
3. Rafailov, R., et al. (2023). Direct Preference Optimization. *NeurIPS*.
4. Hokamp, C., & Liu, Q. (2017). Lexically Constrained Decoding for Sequence Generation. *ACL*.
