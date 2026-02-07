# Alignment Overview

## Learning Objectives

- Understand why pretrained LLMs need alignment
- Describe the three-stage alignment pipeline
- Define the three pillars: helpful, harmless, honest

## Why Alignment?

Pretrained LLMs are trained to predict the next token—not to be helpful, safe, or truthful. Without alignment:

- Models may generate toxic, biased, or harmful content
- Models may confidently state falsehoods (hallucination)
- Models may not follow user instructions effectively
- Models may reveal private training data

**Alignment** is the process of training models to behave in accordance with human values and intentions.

## The Three Pillars (HHH)

1. **Helpful**: Provides useful, relevant, accurate responses to user queries
2. **Harmless**: Avoids generating content that could cause harm
3. **Honest**: Acknowledges uncertainty, avoids fabrication, corrects mistakes

## The Alignment Pipeline

```
Pretrained LLM → SFT → Reward Model → RL (PPO/DPO) → Aligned LLM
```

### Stage 1: Supervised Fine-Tuning (SFT)

Train on curated instruction-response pairs:

| Dataset Type | Examples | Purpose |
|-------------|---------|---------|
| Instructions | "Summarize this article" → [summary] | Task following |
| Conversations | Multi-turn dialogues | Conversational ability |
| Safety | Harmful query → refusal | Safety behavior |
| Helpfulness | Complex questions → detailed answers | Quality |

### Stage 2: Reward Modeling

Train a model to predict human preferences between responses:

$$P(y_w \succ y_l \mid x) = \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

### Stage 3: Reinforcement Learning

Optimize the policy to maximize reward while staying close to the SFT model:

$$\max_\pi \mathbb{E}_{x, y \sim \pi}[r(x, y)] - \beta \cdot D_{\text{KL}}[\pi \| \pi_{\text{SFT}}]$$

## Evolution of Alignment

| Method | Year | Key Innovation |
|--------|------|---------------|
| RLHF (InstructGPT) | 2022 | Human feedback + PPO |
| Constitutional AI | 2022 | Self-critique with principles |
| DPO | 2023 | Direct preference optimization (no RM) |
| ORPO | 2024 | Odds-ratio preference optimization |
| Self-Play | 2024 | SPIN, iterative self-improvement |

## References

1. Ouyang, L., et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." *NeurIPS*.
2. Bai, Y., et al. (2022). "Training a Helpful and Harmless Assistant with RLHF." *arXiv*.
