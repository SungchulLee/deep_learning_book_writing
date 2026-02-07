# Sampling Strategies for Text Generation

## Overview

Autoregressive language models define a probability distribution over the next token given previous tokens. At inference time, we must **decode** from this distribution to generate text. The choice of decoding strategy profoundly impacts output quality, diversity, coherence, and computational cost.

Let $p_\theta(x_t \mid x_{<t})$ denote the model's conditional distribution over token $x_t$ given context $x_{<t} = (x_1, \ldots, x_{t-1})$. The model outputs **logits** $z \in \mathbb{R}^{|V|}$ (one per vocabulary token), which are converted to probabilities via softmax:

$$
p_\theta(x_t = v \mid x_{<t}) = \frac{\exp(z_v)}{\sum_{v' \in V} \exp(z_{v'})}
$$

Decoding strategies fall into two broad categories:

1. **Deterministic methods**: Select tokens via optimization (greedy, beam search)
2. **Stochastic methods**: Sample from (modified) distributions (temperature, top-k, top-p)

---

## 1. Greedy Decoding

The simplest strategy: always select the most probable token.

$$
x_t^* = \arg\max_{v \in V} \, p_\theta(v \mid x_{<t})
$$

```python
import torch
import torch.nn.functional as F
from typing import Optional

def greedy_decode(
    model,
    input_ids: torch.Tensor,
    max_length: int = 50,
    eos_token_id: Optional[int] = None
) -> torch.Tensor:
    """Greedy decoding: always select highest probability token."""
    generated = input_ids.clone()
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs.logits[:, -1, :]  # [batch, vocab]
        
        next_token = logits.argmax(dim=-1, keepdim=True)  # [batch, 1]
        generated = torch.cat([generated, next_token], dim=-1)
        
        if eos_token_id is not None and (next_token == eos_token_id).all():
            break
    
    return generated
```

**Properties**:

- Fast: $O(T \cdot |V|)$ per sequence
- Deterministic: same input always produces same output
- Mode-seeking: finds local maximum, not global optimum
- Prone to repetition and generic outputs

**When to use**: Factual QA, classification tasks, when reproducibility matters.

---

## 2. Temperature Sampling

Temperature scaling modifies the "sharpness" of the probability distribution before sampling. Given logits $z$ and temperature $T > 0$:

$$
p_T(x_t = v \mid x_{<t}) = \frac{\exp(z_v / T)}{\sum_{v'} \exp(z_{v'} / T)}
$$

### Mathematical Analysis

Temperature affects the **entropy** of the distribution. Let $H(p) = -\sum_v p_v \log p_v$ denote entropy.

**Theorem (Temperature and Entropy)**. For any distribution $p$ with logits $z$:

1. $\lim_{T \to 0^+} p_T$ converges to a point mass on $\arg\max_v z_v$
2. $\lim_{T \to \infty} p_T$ converges to the uniform distribution
3. $H(p_T)$ is monotonically increasing in $T$

*Proof sketch*: As $T \to 0$, $z_v/T \to \pm\infty$ depending on whether $z_v$ is maximal, concentrating all mass. As $T \to \infty$, $z_v/T \to 0$ for all $v$, yielding uniform distribution. Monotonicity follows from log-sum-exp properties. $\square$

### Entropy Perspective

An alternative view: temperature scaling is equivalent to raising probabilities to a power:

$$
p_T(v) \propto p_1(v)^{1/T}
$$

This follows because $\exp(z_v/T) = \exp(z_v)^{1/T}$.

```python
def temperature_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    num_samples: int = 1
) -> torch.Tensor:
    """
    Sample from temperature-scaled distribution.
    
    Args:
        logits: Unnormalized log-probabilities [batch, vocab]
        temperature: Scaling factor (0 < T). Lower = more deterministic.
        num_samples: Number of samples to draw
        
    Returns:
        Sampled token indices [batch, num_samples]
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    
    if temperature == 1.0:
        scaled_logits = logits
    else:
        scaled_logits = logits / temperature
    
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=num_samples)
```

### Practical Guidance

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| 0.0–0.3 | Near-deterministic | Code, factual responses |
| 0.5–0.7 | Balanced | General assistant tasks |
| 0.7–1.0 | Creative | Story writing, brainstorming |
| 1.0–1.5 | High variance | Diverse generation, exploration |
| > 1.5 | Often incoherent | Rarely useful |

---

## 3. Top-k Sampling

**Top-k sampling** (Fan et al., 2018) restricts sampling to the $k$ most probable tokens, redistributing probability mass among them.

$$
p_{\text{top-}k}(v \mid x_{<t}) = 
\begin{cases}
\frac{p_\theta(v \mid x_{<t})}{\sum_{v' \in V_k} p_\theta(v' \mid x_{<t})} & \text{if } v \in V_k \\
0 & \text{otherwise}
\end{cases}
$$

where $V_k$ is the set of $k$ tokens with highest probability.

```python
def top_k_sample(
    logits: torch.Tensor,
    k: int = 50,
    temperature: float = 1.0,
    filter_value: float = float('-inf')
) -> torch.Tensor:
    """
    Top-k sampling: sample from k most probable tokens.
    
    Args:
        logits: Unnormalized log-probabilities [batch, vocab]
        k: Number of top tokens to keep
        temperature: Temperature scaling (applied before filtering)
        filter_value: Value to assign to filtered tokens
        
    Returns:
        Sampled token index [batch, 1]
    """
    logits = logits / temperature
    
    # Find k-th largest value (threshold)
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    threshold = top_k_values[:, -1, None]  # [batch, 1]
    
    # Zero out tokens below threshold
    logits = torch.where(logits >= threshold, logits, filter_value)
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Limitation: Fixed k

The fundamental weakness of top-k is that $k$ is fixed regardless of the distribution shape:

- **Peaked distribution** (model is confident): Top-k may include low-probability noise tokens
- **Flat distribution** (model is uncertain): Top-k may exclude reasonable alternatives

**Example**: If $p = (0.9, 0.05, 0.02, 0.01, 0.01, 0.01, ...)$ with $k=10$, we include tokens with negligible probability. If $p = (0.1, 0.1, 0.1, ..., 0.1)$ uniformly over 20 tokens with $k=10$, we arbitrarily exclude half the reasonable options.

---

## 4. Nucleus (Top-p) Sampling

**Nucleus sampling** (Holtzman et al., 2020) addresses top-k's limitation by dynamically adjusting the candidate set based on cumulative probability mass.

### Definition

Given threshold $p \in (0, 1]$, the **nucleus** $V_p$ is the smallest set of tokens whose cumulative probability exceeds $p$:

$$
V_p = \arg\min_{V' \subseteq V} |V'| \quad \text{s.t.} \quad \sum_{v \in V'} p_\theta(v \mid x_{<t}) \geq p
$$

where tokens are added in decreasing probability order.

The sampling distribution is:

$$
p_{\text{nucleus}}(v) = 
\begin{cases}
\frac{p_\theta(v)}{\sum_{v' \in V_p} p_\theta(v')} & \text{if } v \in V_p \\
0 & \text{otherwise}
\end{cases}
$$

### Why "Nucleus"?

The name comes from the observation that high-probability tokens form a "core" (nucleus) of the distribution, while the long tail contains unreliable tokens that the model assigns probability to simply because it must distribute mass across the vocabulary.

```python
def nucleus_sample(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0,
    min_tokens_to_keep: int = 1
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling: sample from smallest set with cumulative prob >= p.
    
    Args:
        logits: Unnormalized log-probabilities [batch, vocab]
        p: Cumulative probability threshold
        temperature: Temperature scaling (applied before filtering)
        min_tokens_to_keep: Always keep at least this many tokens
        
    Returns:
        Sampled token index [batch, 1]
    """
    logits = logits / temperature
    
    # Sort by probability (descending)
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff: first position where cumulative prob exceeds p
    # Shift right to keep the token that crosses threshold
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., min_tokens_to_keep:] = sorted_indices_to_remove[..., :-min_tokens_to_keep].clone()
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False
    
    # Scatter back to original order
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Adaptive Behavior

The key advantage: nucleus size adapts to distribution entropy.

| Distribution Type | Entropy | Nucleus Size |
|------------------|---------|--------------|
| Model confident | Low | Small (few tokens) |
| Model uncertain | High | Large (many tokens) |

This matches human intuition: when many continuations are plausible, include them; when one is clearly best, focus on it.

---

## 5. Combined Sampling Strategies

In practice, multiple strategies are often combined for finer control.

### Top-k + Top-p

Apply both filters (intersection of candidate sets):

```python
def combined_sample(
    logits: torch.Tensor,
    k: int = 50,
    p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Combined top-k and top-p sampling.
    
    Order of operations:
    1. Temperature scaling
    2. Top-k filtering (hard cap on candidates)
    3. Top-p filtering (probability-based refinement)
    4. Sample from remaining distribution
    """
    logits = logits / temperature
    
    # Step 1: Top-k filter
    if k > 0 and k < logits.size(-1):
        top_k_values, _ = torch.topk(logits, k, dim=-1)
        threshold_k = top_k_values[:, -1, None]
        logits = torch.where(logits >= threshold_k, logits, float('-inf'))
    
    # Step 2: Top-p filter
    if p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Order of Operations

The standard order is: **Temperature → Top-k → Top-p → Sample**

Why this order?

1. Temperature first: Changes relative probabilities that top-k/top-p will filter on
2. Top-k second: Provides hard cap (safety net against pathological distributions)
3. Top-p third: Refines within top-k based on probability mass
4. Sample last: Draw from the final filtered distribution

---

## 6. Min-p Sampling

**Min-p sampling** (recent alternative to top-p) keeps all tokens with probability at least $p_{\min} \times p_{\max}$, where $p_{\max}$ is the highest token probability.

### Definition

$$
V_{\text{min-}p} = \{v \in V : p_\theta(v) \geq p_{\min} \cdot \max_{v'} p_\theta(v')\}
$$

### Intuition

Min-p scales the threshold relative to the most probable token. This means:

- When the model is confident ($p_{\max}$ high), threshold is high → few tokens
- When uncertain ($p_{\max}$ low), threshold is low → many tokens

```python
def min_p_sample(
    logits: torch.Tensor,
    p_min: float = 0.1,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Min-p sampling: keep tokens with prob >= p_min * max_prob.
    
    Args:
        logits: Unnormalized log-probabilities [batch, vocab]
        p_min: Minimum probability relative to max (0 < p_min <= 1)
        temperature: Temperature scaling
        
    Returns:
        Sampled token index [batch, 1]
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    
    # Dynamic threshold based on max probability
    max_prob = probs.max(dim=-1, keepdim=True).values
    threshold = p_min * max_prob
    
    # Mask tokens below threshold
    mask = probs < threshold
    logits = logits.masked_fill(mask, float('-inf'))
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Comparison: Top-p vs Min-p

| Aspect | Top-p (Nucleus) | Min-p |
|--------|-----------------|-------|
| Threshold type | Absolute cumulative | Relative to max |
| Very peaked dist. | May include 1 token | Always includes 1+ |
| Flat distribution | Includes many | Includes many |
| Parameter meaning | "Cover this much probability" | "Be within this factor of best" |

---

## 7. Typical Sampling

**Typical sampling** (Meister et al., 2023) selects tokens based on how "typical" or "expected" they are according to information theory.

### Information-Theoretic Foundation

The **information content** (surprisal) of token $v$ is:

$$
I(v) = -\log p_\theta(v \mid x_{<t})
$$

The **entropy** of the distribution is the expected information:

$$
H = \mathbb{E}_{v \sim p}[I(v)] = -\sum_v p_\theta(v) \log p_\theta(v)
$$

A token is **typical** if its information content is close to the entropy. Intuitively, typical tokens are "neither too surprising nor too obvious."

### Definition

The **typical set** $A_\epsilon$ contains tokens whose information content is within $\epsilon$ of the entropy:

$$
A_\epsilon = \{v \in V : |I(v) - H| < \epsilon\}
$$

In practice, we sort by $|I(v) - H|$ and take tokens until cumulative probability exceeds a threshold.

```python
def typical_sample(
    logits: torch.Tensor,
    p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Typical sampling: select tokens with information content close to entropy.
    
    Args:
        logits: Unnormalized log-probabilities [batch, vocab]
        p: Cumulative probability threshold for typical set
        temperature: Temperature scaling
        
    Returns:
        Sampled token index [batch, 1]
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Compute entropy and information content
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)  # [batch, 1]
    information = -log_probs  # [batch, vocab]
    
    # Distance from entropy (how "atypical" each token is)
    deviation = torch.abs(information - entropy)
    
    # Sort by typicality (smallest deviation first)
    sorted_deviation, sorted_indices = torch.sort(deviation, dim=-1)
    sorted_probs = probs.gather(dim=-1, index=sorted_indices)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Find cutoff where cumulative prob exceeds p
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Scatter back and mask
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    logits = logits.masked_fill(indices_to_remove, float('-inf'))
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

### Why Typical Sampling?

Standard sampling methods bias toward high-probability tokens. But the Asymptotic Equipartition Property (AEP) from information theory tells us that for long sequences, almost all probability mass concentrates on **typical sequences**—those with per-token information close to entropy.

Typical sampling operationalizes this insight at the token level.

---

## 8. Eta (η) Sampling

**Eta sampling** combines the adaptivity of typical sampling with a fallback to top-p when the distribution is highly entropic.

### Definition

1. Compute entropy $H$ of the distribution
2. Set threshold: $\eta = \min(\epsilon, \sqrt{\epsilon} \cdot e^{-H})$
3. Keep tokens where $p(v) > \eta$
4. If no tokens remain, fall back to top-p

The key insight: $\eta$ decreases as entropy increases (more uncertainty → lower threshold).

```python
def eta_sample(
    logits: torch.Tensor,
    epsilon: float = 0.0003,
    fallback_p: float = 0.9,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Eta sampling: adaptive threshold based on entropy.
    
    Args:
        logits: Unnormalized log-probabilities [batch, vocab]
        epsilon: Base threshold parameter
        fallback_p: Top-p threshold if eta filtering removes all tokens
        temperature: Temperature scaling
        
    Returns:
        Sampled token index [batch, 1]
    """
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Compute entropy
    entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
    
    # Compute adaptive threshold eta
    eta = torch.minimum(
        torch.tensor(epsilon),
        torch.sqrt(torch.tensor(epsilon)) * torch.exp(-entropy)
    )
    
    # Filter tokens below threshold
    mask = probs < eta
    filtered_logits = logits.masked_fill(mask, float('-inf'))
    
    # Check if any tokens remain
    valid_count = (~mask).sum(dim=-1)
    
    # Fallback to top-p if all filtered
    if (valid_count == 0).any():
        # Apply nucleus sampling as fallback
        return nucleus_sample(logits * temperature, p=fallback_p, temperature=temperature)
    
    probs = F.softmax(filtered_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
```

---

## 9. Mirostat Sampling

**Mirostat** (Basu et al., 2021) dynamically adjusts temperature to maintain a target **perplexity** (equivalently, target cross-entropy/surprisal rate).

### Motivation

Fixed sampling parameters often produce inconsistent output quality across contexts. Mirostat treats decoding as a **control problem**: adjust parameters online to track a desired "surprise level."

### Algorithm (Mirostat-2)

1. Set target surprisal $\tau$ (e.g., 5.0 ≈ perplexity 148)
2. Track running estimate of surprisal
3. Adjust top-k dynamically to steer toward target

```python
def mirostat_v2_sample(
    logits: torch.Tensor,
    tau: float = 5.0,       # Target surprisal
    eta: float = 0.1,       # Learning rate
    mu: float = None        # Current surprisal estimate (state)
) -> tuple[torch.Tensor, float]:
    """
    Mirostat-2 sampling: adaptive targeting of perplexity.
    
    Args:
        logits: Unnormalized log-probabilities [batch, vocab]
        tau: Target surprisal (log2 of target perplexity)
        eta: Learning rate for surprisal tracking
        mu: Current surprisal estimate (initialize to 2*tau)
        
    Returns:
        Tuple of (sampled token index, updated mu)
    """
    if mu is None:
        mu = 2 * tau  # Initialize
    
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log2(probs + 1e-10)  # Use log base 2 for bits
    
    # Sort by probability descending
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    sorted_log_probs = log_probs.gather(dim=-1, index=sorted_indices)
    sorted_surprisals = -sorted_log_probs
    
    # Find k where cumulative meets target
    # Use mu as dynamic threshold
    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
    
    # Mirostat-2: truncate at probability threshold derived from mu
    # k = number of tokens to keep
    prob_threshold = torch.exp2(torch.tensor(-mu))
    
    # Keep tokens with prob >= threshold
    mask = sorted_probs >= prob_threshold
    if mask.sum() == 0:
        mask[..., 0] = True  # Keep at least top token
    
    # Sample from filtered
    filtered_probs = sorted_probs * mask.float()
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    
    sample_idx = torch.multinomial(filtered_probs, num_samples=1)
    token_idx = sorted_indices.gather(dim=-1, index=sample_idx)
    
    # Update mu based on observed surprisal
    observed_surprisal = sorted_surprisals.gather(dim=-1, index=sample_idx).item()
    mu = mu - eta * (observed_surprisal - tau)
    
    return token_idx, mu
```

### Usage Pattern

```python
def generate_with_mirostat(model, input_ids, max_length, tau=5.0, eta=0.1):
    """Generate using Mirostat-2 for consistent perplexity."""
    mu = 2 * tau  # Initial surprisal estimate
    generated = input_ids.clone()
    
    for _ in range(max_length):
        with torch.no_grad():
            logits = model(generated).logits[:, -1, :]
        
        token, mu = mirostat_v2_sample(logits, tau=tau, eta=eta, mu=mu)
        generated = torch.cat([generated, token], dim=-1)
    
    return generated
```

---

## 10. Repetition Penalties

### Simple Repetition Penalty

Discourage repeating previously generated tokens by scaling down their logits:

$$
z'_v = \begin{cases}
z_v / \theta & \text{if } z_v > 0 \text{ and } v \in \text{generated} \\
z_v \cdot \theta & \text{if } z_v < 0 \text{ and } v \in \text{generated} \\
z_v & \text{otherwise}
\end{cases}
$$

where $\theta > 1$ is the penalty factor.

```python
def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    penalty: float = 1.2
) -> torch.Tensor:
    """
    Apply repetition penalty to previously generated tokens.
    
    Positive logits are divided by penalty (reduced probability).
    Negative logits are multiplied by penalty (also reduced probability).
    
    Args:
        logits: Current step logits [batch, vocab]
        generated_ids: Previously generated token IDs [batch, seq_len]
        penalty: Penalty factor (> 1 to discourage, < 1 to encourage)
        
    Returns:
        Modified logits
    """
    # Gather logits for generated tokens
    generated_logits = torch.gather(logits, dim=-1, index=generated_ids)
    
    # Apply penalty based on sign
    penalized = torch.where(
        generated_logits > 0,
        generated_logits / penalty,
        generated_logits * penalty
    )
    
    # Scatter back
    logits = logits.scatter(dim=-1, index=generated_ids, src=penalized)
    return logits
```

### Frequency and Presence Penalties

OpenAI-style additive penalties (rather than multiplicative):

$$
z'_v = z_v - \alpha_{\text{freq}} \cdot \text{count}(v) - \alpha_{\text{pres}} \cdot \mathbf{1}[v \in \text{generated}]
$$

- **Frequency penalty** ($\alpha_{\text{freq}}$): Penalizes proportional to how often token appeared
- **Presence penalty** ($\alpha_{\text{pres}}$): Flat penalty if token appeared at all

```python
def apply_frequency_presence_penalty(
    logits: torch.Tensor,
    generated_ids: torch.Tensor,
    frequency_penalty: float = 0.5,
    presence_penalty: float = 0.5
) -> torch.Tensor:
    """
    Apply frequency and presence penalties (additive, OpenAI-style).
    
    Args:
        logits: Current step logits [batch, vocab]
        generated_ids: Previously generated token IDs [batch, seq_len]
        frequency_penalty: Penalty per occurrence
        presence_penalty: Flat penalty if present at all
        
    Returns:
        Modified logits
    """
    batch_size, vocab_size = logits.shape
    
    # Count occurrences of each token
    token_counts = torch.zeros(batch_size, vocab_size, device=logits.device)
    token_counts.scatter_add_(
        dim=-1, 
        index=generated_ids,
        src=torch.ones_like(generated_ids, dtype=logits.dtype)
    )
    
    # Frequency penalty: proportional to count
    logits = logits - frequency_penalty * token_counts
    
    # Presence penalty: flat if count > 0
    presence_mask = (token_counts > 0).float()
    logits = logits - presence_penalty * presence_mask
    
    return logits
```

### No-Repeat N-gram

Prevent exact repetition of n-gram sequences:

```python
def apply_no_repeat_ngram(
    logits: torch.Tensor,
    generated_ids: list[int],
    n: int = 3
) -> torch.Tensor:
    """
    Prevent generation of any n-gram that already appeared.
    
    Args:
        logits: Current step logits [vocab]
        generated_ids: Previously generated token IDs (list)
        n: N-gram size to block
        
    Returns:
        Modified logits with banned continuations set to -inf
    """
    if len(generated_ids) < n - 1:
        return logits
    
    # Get the last (n-1) tokens
    prefix = tuple(generated_ids[-(n-1):])
    
    # Find all n-grams in history and collect banned continuations
    banned_tokens = set()
    for i in range(len(generated_ids) - n + 1):
        if tuple(generated_ids[i:i+n-1]) == prefix:
            banned_tokens.add(generated_ids[i + n - 1])
    
    # Set banned token logits to -inf
    for token_id in banned_tokens:
        logits[token_id] = float('-inf')
    
    return logits
```

---

## 11. Beam Search

**Beam search** maintains multiple hypotheses (beams) to find approximately optimal sequences under the model.

### Formal Definition

At each step, expand each beam with top-$k$ continuations, keep overall top-$B$ candidates:

$$
\text{score}(x_{1:t}) = \sum_{i=1}^{t} \log p_\theta(x_i \mid x_{<i})
$$

### Length Normalization

Raw log-probability favors shorter sequences. **Length penalty** corrects this:

$$
\text{score}_{\text{LP}}(x_{1:t}) = \frac{\sum_{i=1}^{t} \log p_\theta(x_i \mid x_{<i})}{((5 + t) / 6)^\alpha}
$$

where $\alpha > 0$ encourages longer sequences, $\alpha < 0$ favors shorter.

```python
def beam_search(
    model,
    input_ids: torch.Tensor,
    num_beams: int = 5,
    max_length: int = 50,
    length_penalty: float = 1.0,
    eos_token_id: Optional[int] = None,
    early_stopping: bool = True
) -> list[tuple[float, list[int]]]:
    """
    Beam search decoding.
    
    Args:
        model: Language model with .forward() returning logits
        input_ids: Initial token IDs [1, seq_len]
        num_beams: Number of beams to maintain
        max_length: Maximum tokens to generate
        length_penalty: Exponent for length normalization (> 0)
        eos_token_id: End-of-sequence token ID
        early_stopping: Stop when num_beams hypotheses are complete
        
    Returns:
        List of (score, token_ids) tuples, sorted by score descending
    """
    device = input_ids.device
    initial_seq = input_ids[0].tolist()
    
    # Each beam: (accumulated_log_prob, token_sequence)
    beams = [(0.0, initial_seq)]
    complete_hypotheses = []
    
    for step in range(max_length):
        all_candidates = []
        
        for log_prob, seq in beams:
            # Check if already complete
            if eos_token_id is not None and seq[-1] == eos_token_id:
                complete_hypotheses.append((log_prob, seq))
                continue
            
            # Get next token distribution
            seq_tensor = torch.tensor([seq], device=device)
            with torch.no_grad():
                outputs = model(seq_tensor)
                logits = outputs.logits[0, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)
            
            # Expand with top tokens
            top_log_probs, top_indices = torch.topk(log_probs, num_beams * 2)
            
            for next_log_prob, token_id in zip(top_log_probs.tolist(), top_indices.tolist()):
                new_seq = seq + [token_id]
                new_log_prob = log_prob + next_log_prob
                
                # Length-normalized score for ranking
                length_factor = ((5 + len(new_seq)) / 6) ** length_penalty
                normalized_score = new_log_prob / length_factor
                
                all_candidates.append((normalized_score, new_log_prob, new_seq))
        
        if not all_candidates:
            break
        
        # Keep top beams
        all_candidates.sort(key=lambda x: x[0], reverse=True)
        beams = [(c[1], c[2]) for c in all_candidates[:num_beams]]
        
        # Early stopping check
        if early_stopping and len(complete_hypotheses) >= num_beams:
            break
    
    # Add remaining beams to hypotheses
    complete_hypotheses.extend(beams)
    
    # Sort by length-normalized score
    complete_hypotheses.sort(
        key=lambda x: x[0] / ((5 + len(x[1])) / 6) ** length_penalty,
        reverse=True
    )
    
    return complete_hypotheses
```

### Diverse Beam Search

Standard beam search often produces similar hypotheses. **Diverse beam search** (Vijayakumar et al., 2018) partitions beams into groups with diversity penalty:

$$
\text{score}_g(v) = \text{score}(v) - \lambda \sum_{g' < g} \mathbf{1}[v \in \text{beams}_{g'}]
$$

```python
def diverse_beam_search(
    model,
    input_ids: torch.Tensor,
    num_beams: int = 4,
    num_groups: int = 4,
    diversity_penalty: float = 0.5,
    max_length: int = 50
) -> list[list[tuple[float, list[int]]]]:
    """
    Diverse beam search: generate diverse hypotheses via grouped search.
    
    Args:
        model: Language model
        input_ids: Initial token IDs
        num_beams: Beams per group
        num_groups: Number of beam groups
        diversity_penalty: Penalty for tokens chosen by earlier groups
        max_length: Maximum generation length
        
    Returns:
        List of groups, each containing (score, sequence) tuples
    """
    device = input_ids.device
    all_groups = []
    
    for group_idx in range(num_groups):
        # Track tokens selected by previous groups at each position
        previous_group_tokens = []
        for prev_group in all_groups:
            for _, seq in prev_group:
                previous_group_tokens.extend(seq[len(input_ids[0]):])
        
        # Run beam search with diversity penalty
        group_beams = beam_search_with_penalty(
            model, input_ids, num_beams, max_length,
            penalty_tokens=set(previous_group_tokens),
            penalty_weight=diversity_penalty
        )
        
        all_groups.append(group_beams)
    
    return all_groups
```

---

## 12. Contrastive Search

**Contrastive search** (Su et al., 2022) balances likelihood with distinctiveness from the existing context.

### Objective

$$
x_t = \arg\max_{v \in V_k} \left[ (1 - \alpha) \cdot p_\theta(v \mid x_{<t}) - \alpha \cdot \max_{j < t} \text{sim}(h_v, h_{x_j}) \right]
$$

where:
- $V_k$ is the top-k candidates by probability
- $h_v$ is the hidden representation of token $v$
- $\text{sim}$ is cosine similarity
- $\alpha \in [0, 1]$ balances likelihood vs. distinctiveness

### Intuition

The **degeneration penalty** term $\max_j \text{sim}(h_v, h_{x_j})$ discourages tokens whose representations are too similar to previous context, reducing repetitive patterns.

```python
def contrastive_search(
    model,
    input_ids: torch.Tensor,
    k: int = 4,
    alpha: float = 0.6,
    max_length: int = 50
) -> torch.Tensor:
    """
    Contrastive search: balance probability with representation distinctiveness.
    
    Args:
        model: Model that returns logits and hidden states
        input_ids: Initial token IDs [1, seq_len]
        k: Number of candidate tokens to consider
        alpha: Balance factor (0 = pure likelihood, 1 = pure distinctiveness)
        max_length: Maximum tokens to generate
        
    Returns:
        Generated sequence
    """
    generated = input_ids.clone()
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated, output_hidden_states=True)
        
        logits = outputs.logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        
        # Context hidden states (all positions except last)
        context_hidden = outputs.hidden_states[-1][0, :-1, :]  # [ctx_len, hidden]
        
        # Get top-k candidates
        top_probs, top_indices = torch.topk(probs, k)
        
        best_score = float('-inf')
        best_token = top_indices[0].item()  # Default to top-1
        
        for prob, token_id in zip(top_probs.tolist(), top_indices.tolist()):
            # Get candidate's hidden state by forward pass
            candidate_input = torch.cat(
                [generated, torch.tensor([[token_id]], device=generated.device)],
                dim=-1
            )
            with torch.no_grad():
                candidate_output = model(candidate_input, output_hidden_states=True)
            candidate_hidden = candidate_output.hidden_states[-1][0, -1, :]  # [hidden]
            
            # Max similarity to any context position
            similarities = F.cosine_similarity(
                candidate_hidden.unsqueeze(0),
                context_hidden,
                dim=-1
            )
            max_sim = similarities.max().item()
            
            # Contrastive score
            score = (1 - alpha) * prob - alpha * max_sim
            
            if score > best_score:
                best_score = score
                best_token = token_id
        
        generated = torch.cat(
            [generated, torch.tensor([[best_token]], device=generated.device)],
            dim=-1
        )
    
    return generated
```

---

## 13. Speculative Decoding

**Speculative decoding** (Leviathan et al., 2023; Chen et al., 2023) accelerates inference by using a small "draft" model to propose tokens, verified in parallel by the large target model.

### Key Insight

Autoregressive generation is **memory-bound**: each token requires a full forward pass, but most compute is waiting for memory. If we can verify multiple tokens in one pass, we save wall-clock time.

### Algorithm

1. Draft model generates $\gamma$ tokens autoregressively
2. Target model scores all $\gamma$ tokens in one forward pass
3. Accept/reject each token via rejection sampling
4. If rejected, resample from corrected distribution

### Correctness Guarantee

Speculative decoding produces **exactly the same distribution** as standard sampling from the target model (when using the same random seed).

```python
def speculative_decode(
    draft_model,
    target_model,
    input_ids: torch.Tensor,
    gamma: int = 4,
    temperature: float = 1.0
) -> tuple[torch.Tensor, int]:
    """
    Speculative decoding: draft with small model, verify with large model.
    
    Args:
        draft_model: Small, fast model for drafting
        target_model: Large, accurate model for verification
        input_ids: Initial token IDs [1, seq_len]
        gamma: Number of tokens to draft per iteration
        temperature: Sampling temperature
        
    Returns:
        Tuple of (accepted tokens tensor, number accepted)
    """
    device = input_ids.device
    
    # Step 1: Draft gamma tokens with small model
    draft_ids = []
    draft_probs = []
    current_ids = input_ids.clone()
    
    for _ in range(gamma):
        with torch.no_grad():
            draft_output = draft_model(current_ids)
            draft_logits = draft_output.logits[:, -1, :] / temperature
            probs = F.softmax(draft_logits, dim=-1)
        
        # Sample from draft
        token = torch.multinomial(probs, num_samples=1)
        draft_ids.append(token.item())
        draft_probs.append(probs[0, token.item()].item())
        
        current_ids = torch.cat([current_ids, token], dim=-1)
    
    # Step 2: Verify all tokens with target model in one pass
    with torch.no_grad():
        target_output = target_model(current_ids)
        target_logits = target_output.logits / temperature
    
    # Step 3: Accept/reject via rejection sampling
    accepted = []
    input_len = input_ids.size(1)
    
    for i, (token, q_prob) in enumerate(zip(draft_ids, draft_probs)):
        # Target probability at position input_len + i
        target_probs = F.softmax(target_logits[0, input_len + i - 1, :], dim=-1)
        p_prob = target_probs[token].item()
        
        # Acceptance probability: min(1, p/q)
        accept_prob = min(1.0, p_prob / (q_prob + 1e-10))
        
        if torch.rand(1).item() < accept_prob:
            accepted.append(token)
        else:
            # Reject: sample from (p - q)+ normalized
            # This is the "residual" distribution
            diff = target_probs - F.softmax(target_logits[0, input_len + i - 1, :], dim=-1)
            diff = torch.clamp(diff, min=0)
            if diff.sum() > 0:
                diff = diff / diff.sum()
                corrected_token = torch.multinomial(diff, num_samples=1)
                accepted.append(corrected_token.item())
            break
    
    accepted_tensor = torch.tensor([accepted], device=device)
    return accepted_tensor, len(accepted)
```

### Speedup Analysis

Let $\alpha$ be the average acceptance rate. Expected tokens per iteration:

$$
\mathbb{E}[\text{tokens}] = \sum_{i=1}^{\gamma} \alpha^{i-1}(1-\alpha) \cdot i + \alpha^\gamma \cdot \gamma = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}
$$

Speedup depends on:
- Draft model quality (higher $\alpha$)
- Cost ratio between draft and target models
- $\gamma$ (more drafts = higher potential gain but more wasted work if rejected)

---

## 14. Guided Generation

### Classifier-Free Guidance (CFG)

Originally from diffusion models, CFG can improve text generation:

$$
\tilde{z} = z_{\text{uncond}} + w \cdot (z_{\text{cond}} - z_{\text{uncond}})
$$

where $w > 1$ amplifies the effect of conditioning.

```python
def classifier_free_guidance(
    model,
    input_ids: torch.Tensor,
    guidance_scale: float = 1.5,
    uncond_input_ids: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Apply classifier-free guidance to logits.
    
    Args:
        model: Language model
        input_ids: Conditioned input IDs [1, seq_len]
        guidance_scale: Amplification factor (w > 1 strengthens conditioning)
        uncond_input_ids: Unconditional input (e.g., empty prompt)
        
    Returns:
        Guided logits
    """
    with torch.no_grad():
        # Conditional logits
        cond_logits = model(input_ids).logits[:, -1, :]
        
        # Unconditional logits
        if uncond_input_ids is None:
            # Use beginning-of-sequence as "unconditional"
            uncond_input_ids = input_ids[:, :1]
        uncond_logits = model(uncond_input_ids).logits[:, -1, :]
    
    # Guided logits
    guided_logits = uncond_logits + guidance_scale * (cond_logits - uncond_logits)
    
    return guided_logits
```

---

## 15. Complete Generation Pipeline

Putting it all together:

```python
class TextGenerator:
    """Flexible text generation with multiple sampling strategies."""
    
    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cuda'
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = device
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        typical_p: float = 1.0,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        no_repeat_ngram_size: int = 0,
        eos_token_id: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text with configurable sampling strategy.
        
        Args:
            prompt: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering (0 = disabled)
            top_p: Nucleus sampling threshold (1.0 = disabled)
            min_p: Min-p sampling threshold (0.0 = disabled)
            typical_p: Typical sampling threshold (1.0 = disabled)
            repetition_penalty: Multiplicative penalty for repeated tokens
            frequency_penalty: Additive penalty per token occurrence
            presence_penalty: Additive penalty for token presence
            no_repeat_ngram_size: Block n-gram repetition (0 = disabled)
            eos_token_id: Stop generation at this token
            
        Returns:
            Generated text string
        """
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        generated_ids = input_ids[0].tolist()
        
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        
        for _ in range(max_new_tokens):
            # Forward pass
            input_tensor = torch.tensor([generated_ids], device=self.device)
            with torch.no_grad():
                outputs = self.model(input_tensor)
                logits = outputs.logits[0, -1, :].clone()
            
            # Apply penalties
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    if logits[token_id] > 0:
                        logits[token_id] /= repetition_penalty
                    else:
                        logits[token_id] *= repetition_penalty
            
            if frequency_penalty > 0 or presence_penalty > 0:
                token_counts = {}
                for tid in generated_ids:
                    token_counts[tid] = token_counts.get(tid, 0) + 1
                for tid, count in token_counts.items():
                    logits[tid] -= frequency_penalty * count
                    logits[tid] -= presence_penalty
            
            if no_repeat_ngram_size > 0:
                logits = apply_no_repeat_ngram(logits, generated_ids, no_repeat_ngram_size)
            
            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature
            
            # Filtering strategies (apply in order)
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[-1]] = float('-inf')
            
            if min_p > 0:
                probs = F.softmax(logits, dim=-1)
                threshold = min_p * probs.max()
                logits[probs < threshold] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumsum > top_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                sorted_logits[mask] = float('-inf')
                logits = torch.zeros_like(logits).scatter_(-1, sorted_indices, sorted_logits)
            
            if typical_p < 1.0:
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum()
                deviation = torch.abs(-log_probs - entropy)
                sorted_dev, sorted_idx = torch.sort(deviation)
                sorted_probs = probs[sorted_idx]
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum > typical_p
                mask[1:] = mask[:-1].clone()
                mask[0] = False
                remove_idx = sorted_idx[mask]
                logits[remove_idx] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            generated_ids.append(next_token)
            
            if next_token == eos_token_id:
                break
        
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
```

---

## 16. Comparison and Recommendations

### Strategy Comparison

| Strategy | Diversity | Coherence | Speed | Adaptivity |
|----------|-----------|-----------|-------|------------|
| Greedy | Low | High | ★★★★★ | None |
| Temperature | Tunable | Varies | ★★★★★ | None |
| Top-k | Medium | Good | ★★★★★ | None |
| Top-p (Nucleus) | Medium | Good | ★★★★☆ | Distribution shape |
| Min-p | Medium | Good | ★★★★☆ | Max probability |
| Typical | Medium-High | Good | ★★★★☆ | Entropy |
| Eta | Medium | Good | ★★★★☆ | Entropy |
| Mirostat | Medium | Good | ★★★★☆ | Running perplexity |
| Beam Search | Low | High | ★★☆☆☆ | None |
| Contrastive | High | High | ★★☆☆☆ | Context similarity |
| Speculative | Varies | Varies | ★★★★★* | Draft model quality |

*Speculative decoding speed depends on acceptance rate and model cost ratio.

### Recommended Settings by Task

| Task | Temperature | Top-p | Top-k | Other |
|------|-------------|-------|-------|-------|
| Code generation | 0.2–0.4 | 0.95 | — | Low repetition penalty |
| Creative writing | 0.8–1.0 | 0.9 | 50 | Moderate rep. penalty |
| Dialogue/Chat | 0.7 | 0.9 | 40 | Presence penalty 0.1–0.3 |
| Summarization | 0.3–0.5 | 0.9 | — | No-repeat 3-gram |
| Translation | 0.0 (greedy) | — | — | Or beam search |
| Factual QA | 0.0–0.3 | 0.95 | — | — |

### Decision Flowchart

```
Start
  │
  ├─ Need exact reproducibility? ──Yes──► Greedy (T=0)
  │
  ├─ Factual/code task? ──Yes──► Low temperature (0.2-0.4) + Top-p (0.95)
  │
  ├─ Creative task? ──Yes──► Higher temperature (0.7-1.0) + Top-p (0.9)
  │
  ├─ Experiencing repetition? ──Yes──► Add repetition/frequency penalty
  │
  ├─ Need diverse outputs? ──Yes──► Contrastive or diverse beam search
  │
  └─ Need speed? ──Yes──► Speculative decoding (if draft model available)
```

---

## Summary

1. **Top-p (nucleus) sampling with p=0.9** is a robust default for most tasks
2. **Temperature** provides intuitive control over randomness
3. **Combine strategies**: temperature + top-p + repetition penalty covers most needs
4. **Adaptive methods** (typical, eta, mirostat) can improve consistency
5. **Task-specific tuning** significantly impacts output quality
6. **Speculative decoding** offers significant speedups with no quality loss

---

## References

1. Fan, A., Lewis, M., & Dauphin, Y. (2018). Hierarchical Neural Story Generation. *ACL*.

2. Holtzman, A., Buys, J., Du, L., Forbes, M., & Choi, Y. (2020). The Curious Case of Neural Text Degeneration. *ICLR*.

3. Meister, C., Pimentel, T., Wiher, G., & Cotterell, R. (2023). Locally Typical Sampling. *TACL*.

4. Su, Y., Lan, T., Wang, Y., Yogatama, D., Kong, L., & Collier, N. (2022). A Contrastive Framework for Neural Text Generation. *NeurIPS*.

5. Basu, S., Ramachandran, G. S., Keskar, N. S., & Varshney, L. R. (2021). Mirostat: A Neural Text Decoding Algorithm that Directly Controls Perplexity. *ICLR*.

6. Leviathan, Y., Kalman, M., & Matias, Y. (2023). Fast Inference from Transformers via Speculative Decoding. *ICML*.

7. Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., & Jumper, J. (2023). Accelerating Large Language Model Decoding with Speculative Sampling. *arXiv*.

8. Vijayakumar, A. K., et al. (2018). Diverse Beam Search: Decoding Diverse Solutions from Neural Sequence Models. *AAAI*.
