# Reinforcement Learning from Human Feedback (RLHF)

## Learning Objectives

- Understand the three-stage RLHF pipeline
- Formalize each stage mathematically
- Identify practical challenges and data requirements

## Stage 1: Supervised Fine-Tuning (SFT)

Fine-tune the pretrained model on high-quality demonstration data:

$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x, y) \sim \mathcal{D}_{\text{demo}}} \left[\sum_{t=1}^{|y|} \log \pi_\theta(y_t \mid x, y_{<t})\right]$$

Data requirements: 10,000-100,000 high-quality instruction-response pairs.

## Stage 2: Reward Model Training

Collect human comparisons between model responses and train a reward model:

$$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{compare}}} \left[\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))\right]$$

where $y_w$ is the preferred response and $y_l$ is the less preferred response (Bradley-Terry model).

Data requirements: 100,000-500,000 comparison pairs.

## Stage 3: PPO Optimization

Optimize the policy using PPO with the reward model:

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[r_\phi(x, y) - \beta \cdot \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{SFT}}(y \mid x)}\right]$$

The KL penalty $\beta$ prevents the policy from deviating too far from the SFT model, which would cause reward hacking.

## Data Requirements Summary

| Stage | Data Type | Volume | Cost |
|-------|----------|--------|------|
| SFT | Instruction-response pairs | 10K-100K | Moderate |
| RM | Comparison pairs | 100K-500K | High (human labelers) |
| PPO | Online generation + scoring | Ongoing | Compute-intensive |

## Practical Challenges

1. **Reward hacking**: The policy exploits reward model weaknesses
2. **KL collapse**: Policy degenerates if KL penalty is too low
3. **Instability**: PPO training is notoriously unstable for LLMs
4. **Cost**: Requires multiple models in memory simultaneously
5. **Distribution shift**: Reward model trained on SFT outputs may not generalize to PPO outputs

## InstructGPT Results

| Model | Helpful | Truthful | Harmless |
|-------|---------|----------|----------|
| GPT-3 (175B) | Baseline | Baseline | Baseline |
| SFT | +15% | +5% | +10% |
| RLHF | +40% | +20% | +25% |

The 1.3B RLHF model was preferred over the 175B base model by human evaluators.

## References

1. Ouyang, L., et al. (2022). "Training Language Models to Follow Instructions with Human Feedback." *NeurIPS*.
2. Schulman, J., et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv*.
