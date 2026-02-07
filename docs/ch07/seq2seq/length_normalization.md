# Length Normalization

Length normalization addresses a systematic bias in beam search: raw log-probability scores inherently favor shorter sequences. Without correction, the decoder preferentially generates truncated outputs, undermining the quality of sequence generation. This section examines the problem in detail, presents standard normalization strategies, and explores their interaction with coverage penalties for attention-based models.

## The Short Sequence Bias Problem

### Why Beam Search Favors Short Sequences

Beam search ranks hypotheses by cumulative log probability:

$$\text{score}(y_1, \ldots, y_T) = \sum_{t=1}^{T} \log P(y_t | y_{<t}, \mathbf{c})$$

Since each conditional probability satisfies $0 < P(y_t | y_{<t}, \mathbf{c}) \leq 1$, every log probability is non-positive: $\log P(\cdot) \leq 0$. Consequently, each additional token can only decrease the total score. A sequence of length 5 with average token probability 0.3 scores $5 \times \log(0.3) \approx -6.0$, while a length-10 sequence with the same average scores $-12.0$.

This creates a systematic preference for shorter outputs. The decoder learns that generating the end-of-sequence token early yields a higher-ranked hypothesis than continuing to generate useful content, resulting in truncated, incomplete outputs.

### Illustrative Example

Consider two candidate translations:

| Hypothesis | Length | Avg Token Prob | Raw Score | Quality |
|-----------|--------|----------------|-----------|---------|
| "The cat" | 2 | 0.5 | -1.39 | Poor (incomplete) |
| "The cat sat on the mat" | 6 | 0.4 | -5.50 | Good (complete) |

Without normalization, beam search selects "The cat" despite it being a clearly inferior translation. The shorter sequence wins purely because it accumulates fewer negative terms.

## Normalization Strategies

### Simple Length Normalization

The most basic approach divides the cumulative score by sequence length:

$$\text{score}_{norm} = \frac{1}{T} \sum_{t=1}^{T} \log P(y_t | y_{<t})$$

This is equivalent to comparing the **geometric mean** of token probabilities—a per-token average that is length-independent. While intuitive, simple normalization can over-correct: it treats a 2-token sequence and a 50-token sequence with the same per-token probability as equally good, even when the longer sequence is more informative and useful.

```python
def simple_length_normalize(score: float, length: int) -> float:
    """
    Normalize by dividing score by sequence length.
    
    Equivalent to comparing geometric mean probabilities.
    Simple but can over-compensate for length.
    """
    if length == 0:
        return float('-inf')
    return score / length
```

### Google's Length Penalty (Wu et al., 2016)

The standard approach in production systems uses a parameterized penalty that provides sublinear normalization:

$$lp(Y) = \frac{(5 + |Y|)^\alpha}{(5 + 1)^\alpha}$$

$$\text{score}_{norm} = \frac{\text{score}}{lp(Y)}$$

The constant 5 provides smoothing for very short sequences, preventing extreme normalization when $|Y|$ is small. The exponent $\alpha$ controls the strength of normalization:

```python
def google_length_penalty(length: int, alpha: float = 0.6) -> float:
    """
    Google's length penalty from Wu et al. (2016).
    
    The penalty grows sublinearly with length when alpha < 1,
    providing a balance between raw scores and per-token averages.
    
    Args:
        length: Sequence length
        alpha: Normalization exponent (0 = no normalization, 1 = full)
        
    Returns:
        Length penalty divisor
    """
    return ((5.0 + length) ** alpha) / ((5.0 + 1.0) ** alpha)


def normalized_score(log_prob_sum: float, length: int, alpha: float = 0.6) -> float:
    """Compute length-normalized beam score."""
    return log_prob_sum / google_length_penalty(length, alpha)
```

### Effect of $\alpha$ Values

The $\alpha$ parameter interpolates between no normalization ($\alpha = 0$, raw scores) and full per-token normalization ($\alpha = 1$, equivalent to geometric mean):

| $\alpha$ Value | Behavior | Use Case |
|---------|----------|----------|
| 0.0 | No normalization (raw scores) | When brevity is desired |
| 0.5 | Moderate smoothing | General translation |
| 0.6–0.7 | Standard choice | Most seq2seq tasks |
| 1.0 | Full normalization (per-token average) | When length should not matter |

The optimal $\alpha$ depends on the task. For machine translation, $\alpha \in [0.6, 0.7]$ is the established standard, providing enough correction to prevent truncation while still maintaining a mild preference for conciseness.

```python
import numpy as np
import matplotlib.pyplot as plt


def compare_alpha_values():
    """Visualize how different alpha values affect scoring."""
    lengths = np.arange(1, 51)
    
    # Same per-token log probability for all lengths
    per_token_logprob = -1.5
    raw_scores = per_token_logprob * lengths
    
    plt.figure(figsize=(10, 6))
    
    for alpha in [0.0, 0.3, 0.6, 0.7, 1.0]:
        penalties = np.array([
            google_length_penalty(l, alpha) for l in lengths
        ])
        normalized = raw_scores / penalties
        plt.plot(lengths, normalized, label=f'alpha={alpha}')
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Normalized Score')
    plt.title('Length Normalization: Effect of Alpha')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### Alternative Normalization Functions

Several alternatives to Google's penalty have been explored:

```python
def exponential_length_penalty(length: int, beta: float = 0.1) -> float:
    """
    Exponential penalty that grows more aggressively with length.
    
    Provides stronger correction for tasks where very long 
    outputs are unlikely.
    """
    return np.exp(beta * length)


def logarithmic_length_penalty(length: int, gamma: float = 1.0) -> float:
    """
    Logarithmic penalty with gentle normalization.
    
    Grows slowly, providing minimal correction for moderate lengths
    but preventing extreme bias for very long sequences.
    """
    return gamma * np.log(1 + length)


def adaptive_length_penalty(
    length: int, 
    target_length: int, 
    sigma: float = 5.0
) -> float:
    """
    Gaussian-shaped penalty centered on expected target length.
    
    Penalizes sequences that deviate from the expected length
    in either direction, useful when target length is predictable
    (e.g., summarization with target length constraints).
    """
    deviation = (length - target_length) ** 2
    return np.exp(deviation / (2 * sigma ** 2))
```

## Integration with Beam Search

### Combined Scoring Function

In practice, length normalization is combined with other scoring adjustments during beam search:

```python
class BeamScorer:
    """
    Combined scoring function for beam search hypotheses.
    
    Integrates length normalization, coverage penalty, and
    optional additional scoring terms.
    """
    
    def __init__(
        self,
        length_penalty_alpha: float = 0.6,
        coverage_penalty_beta: float = 0.0
    ):
        self.alpha = length_penalty_alpha
        self.beta = coverage_penalty_beta
    
    def score(
        self,
        log_prob_sum: float,
        length: int,
        attention_weights: list = None
    ) -> float:
        """
        Compute final hypothesis score.
        
        final_score = log_prob / lp(length) + beta * cp(attention)
        
        Args:
            log_prob_sum: Cumulative log probability
            length: Sequence length
            attention_weights: List of attention distributions (optional)
            
        Returns:
            Combined normalized score
        """
        # Length normalization
        lp = ((5.0 + length) ** self.alpha) / ((5.0 + 1.0) ** self.alpha)
        normalized = log_prob_sum / lp
        
        # Coverage penalty (for attention-based models)
        if self.beta > 0 and attention_weights:
            import torch
            coverage = torch.stack(attention_weights).sum(dim=0)
            cp = torch.sum(torch.log(torch.clamp(coverage, max=1.0)))
            normalized += self.beta * cp.item()
        
        return normalized
```

### Interaction with Coverage Penalty

The coverage penalty and length normalization address complementary problems. Length normalization prevents the decoder from stopping too early, while coverage penalty prevents it from ignoring parts of the input. Together, they encourage complete, well-formed outputs:

$$\text{score}_{final} = \frac{\sum_{t} \log P(y_t | y_{<t}, \mathbf{x})}{lp(|\mathbf{y}|)} + \beta \cdot cp(\mathbf{y})$$

where the coverage penalty is:

$$cp(\mathbf{y}) = \sum_{j=1}^{T_x} \log\left(\min\left(\sum_{t=1}^{T_y} \alpha_{t,j},\; 1\right)\right)$$

This penalty is 0 when every source position has been attended to at least once (coverage $\geq 1$) and negative when positions are under-attended (coverage $< 1$).

### Scoring During vs. After Search

An important implementation detail: length normalization can be applied **during** beam pruning (affecting which hypotheses survive) or only **after** search (for final hypothesis selection). Applying normalization during search is generally preferred because it prevents premature elimination of promising longer hypotheses:

```python
def beam_step_with_normalization(
    beams: list,
    beam_width: int,
    scorer: 'BeamScorer',
    normalize_during_search: bool = True
) -> list:
    """
    Prune beams with optional in-search normalization.
    
    When normalize_during_search=True, length normalization is applied
    when comparing hypotheses of different lengths. This prevents
    shorter hypotheses from dominating the beam.
    """
    if normalize_during_search:
        beams.sort(
            key=lambda h: scorer.score(h.score, len(h.tokens), h.attention_weights),
            reverse=True
        )
    else:
        # Raw scores only (normalization applied after search)
        beams.sort(key=lambda h: h.score, reverse=True)
    
    return beams[:beam_width]
```

## Empirical Analysis

### Length Distribution Comparison

```python
def analyze_length_effects(
    model,
    test_loader,
    beam_decoder,
    alpha_values: list = [0.0, 0.3, 0.6, 0.7, 1.0]
):
    """
    Compare output length distributions across alpha values.
    
    Helps identify the optimal normalization strength by examining
    how output lengths change relative to reference lengths.
    """
    results = {}
    
    for alpha in alpha_values:
        beam_decoder.length_penalty = alpha
        lengths = []
        
        for src, trg in test_loader:
            tokens, score = beam_decoder.decode(src)
            lengths.append(len(tokens))
        
        results[alpha] = {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'lengths': lengths
        }
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Length distributions
    for alpha in alpha_values:
        axes[0].hist(results[alpha]['lengths'], alpha=0.5, 
                     label=f'alpha={alpha}', bins=20)
    axes[0].set_xlabel('Output Length')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Output Length Distribution by Alpha')
    axes[0].legend()
    
    # Mean lengths
    means = [results[a]['mean_length'] for a in alpha_values]
    axes[1].bar(range(len(alpha_values)), means, 
                tick_label=[f'{a}' for a in alpha_values])
    axes[1].set_xlabel('Alpha Value')
    axes[1].set_ylabel('Mean Output Length')
    axes[1].set_title('Mean Output Length by Alpha')
    
    plt.tight_layout()
    plt.show()
    
    return results
```

## Practical Guidelines

### Selecting $\alpha$

The optimal $\alpha$ depends on the task and should be tuned on a validation set. General recommendations:

| Task | Recommended $\alpha$ | Rationale |
|------|---------------------|-----------|
| Machine translation | 0.6–0.7 | Balance between completeness and conciseness |
| Summarization | 0.8–1.0 | Longer summaries tend to be more informative |
| Dialogue response | 0.5–0.6 | Slight preference for conciseness |
| Code generation | 0.6 | Avoid premature termination of code blocks |

### Common Pitfalls

**$\alpha$ too low** ($< 0.4$): Outputs are truncated, missing important information. The model generates end-of-sequence tokens prematurely.

**$\alpha$ too high** ($> 1.0$): Outputs are excessively long, containing repetition or filler. The normalization over-compensates, making longer sequences disproportionately attractive.

**Inconsistent normalization**: Applying length normalization only for final selection but not during beam pruning allows short hypotheses to dominate the beam, eliminating promising longer candidates before they can complete.

### Interaction with Other Parameters

Length normalization interacts with several other hyperparameters:

| Parameter Interaction | Effect |
|----------------------|--------|
| Higher beam width + higher $\alpha$ | More diverse length exploration |
| Coverage penalty + length penalty | Complementary: length prevents truncation, coverage prevents skipping |
| Repetition penalty + $\alpha$ | High $\alpha$ without repetition control can produce repetitive long outputs |
| Minimum length + $\alpha$ | Minimum length provides a hard floor, $\alpha$ provides soft encouragement |

## Summary

Length normalization is essential for practical beam search, correcting the inherent bias toward shorter sequences caused by the accumulation of negative log probabilities. Google's length penalty with $\alpha \approx 0.6\text{--}0.7$ is the established standard, providing sublinear normalization that balances completeness against conciseness. The normalization should be applied during beam pruning, not just for final selection, to prevent premature elimination of promising longer hypotheses. When combined with coverage penalty in attention-based models, length normalization helps produce outputs that are both complete (all source information addressed) and appropriately sized (neither truncated nor padded with filler).
