# Self-Consistency

## Learning Objectives

- Understand the self-consistency decoding strategy
- Implement majority voting over chain-of-thought paths
- Analyze the cost-accuracy trade-off

## Core Idea

**Self-Consistency** (Wang et al., 2023) samples multiple CoT paths and selects the most common answer:

$$\hat{y} = \arg\max_y \sum_{i=1}^{m} \mathbb{1}[y_i = y]$$

Correct reasoning paths converge on the same answer; errors tend to be diverse.

## Implementation

```python
from collections import Counter


def self_consistency(llm, prompt, n_samples=10, temperature=0.7, extract_fn=None):
    answers = []
    for _ in range(n_samples):
        response = llm(prompt, temperature=temperature)
        answer = extract_fn(response)
        answers.append(answer)

    vote_counts = Counter(answers)
    best_answer = vote_counts.most_common(1)[0][0]
    confidence = vote_counts[best_answer] / n_samples

    return {"answer": best_answer, "confidence": confidence, "votes": dict(vote_counts)}
```

## Key Results

| Benchmark | CoT (greedy) | Self-Consistency (40 paths) | Improvement |
|-----------|-------------|---------------------------|-------------|
| GSM8K | 56.5% | 74.4% | +17.9pp |
| SVAMP | 68.9% | 86.6% | +17.7pp |
| AQuA | 35.8% | 48.3% | +12.5pp |

*PaLM 540B (Wang et al., 2023)*

## Cost-Accuracy Trade-off

| Samples ($m$) | GSM8K Accuracy | Relative Cost |
|--------------|---------------|---------------|
| 1 (greedy) | 56.5% | 1x |
| 5 | 68.2% | 5x |
| 10 | 71.8% | 10x |
| 40 | 74.4% | 40x |

Diminishing returns typically after $m = 10$-$20$.

## References

1. Wang, X., et al. (2023). "Self-Consistency Improves Chain of Thought Reasoning." *ICLR*.
