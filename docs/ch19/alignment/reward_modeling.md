# Reward Modeling

## Learning Objectives

- Understand the Bradley-Terry preference model
- Implement a reward model architecture
- Identify reward model failure modes

## Bradley-Terry Model

The preference probability between two responses follows:

$$P(y_w \succ y_l \mid x) = \sigma(r(x, y_w) - r(x, y_l)) = \frac{1}{1 + e^{-(r(x, y_w) - r(x, y_l))}}$$

This is equivalent to logistic regression on reward differences.

## Architecture

A reward model is typically the SFT model with the language modeling head replaced by a **scalar value head**:

```python
import torch
import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # Use last token's hidden state as sequence representation
        last_hidden = outputs.hidden_states[-1]

        # Find the last non-padding token
        if attention_mask is not None:
            last_idx = attention_mask.sum(dim=1) - 1
            sequence_repr = last_hidden[range(len(last_idx)), last_idx]
        else:
            sequence_repr = last_hidden[:, -1]

        reward = self.value_head(sequence_repr).squeeze(-1)
        return reward
```

## Training

```python
def reward_model_loss(model, chosen_ids, rejected_ids, chosen_mask, rejected_mask):
    r_chosen = model(chosen_ids, chosen_mask)
    r_rejected = model(rejected_ids, rejected_mask)

    # Bradley-Terry loss
    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()

    # Accuracy for monitoring
    accuracy = (r_chosen > r_rejected).float().mean()

    return loss, accuracy
```

## Challenges

### Reward Hacking

The policy finds outputs that score high on the reward model but are not genuinely good:

- Verbose responses (longer = higher reward)
- Sycophantic responses (agreeing with user = higher reward)
- Formulaic safety responses

### Mitigation

1. **Diverse training data**: Cover many edge cases
2. **Ensemble reward models**: Average predictions from multiple models
3. **Reward model updates**: Periodically retrain on new policy outputs
4. **Constrained optimization**: Use KL penalty to limit policy divergence

## References

1. Ouyang, L., et al. (2022). "Training Language Models to Follow Instructions with Human Feedback."
2. Gao, L., et al. (2023). "Scaling Laws for Reward Model Overoptimization." *ICML*.
