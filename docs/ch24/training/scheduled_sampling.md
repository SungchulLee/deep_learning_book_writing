# Scheduled Sampling

## Overview

Scheduled sampling (Bengio et al., 2015) bridges the gap between teacher forcing and free-running generation by gradually replacing ground-truth inputs with the model's own predictions during training.

## Algorithm

At each training step, for each position $t$:

- With probability $\epsilon_i$ (curriculum probability): use ground truth $x_t$
- With probability $1 - \epsilon_i$: use model's prediction $\hat{x}_t = \arg\max p_\theta(\cdot \mid x_{<t})$

The curriculum $\epsilon_i$ decreases over training:

$$\epsilon_i = \max(\epsilon_{\min}, k^i) \quad \text{(exponential decay)}$$
$$\epsilon_i = \max(\epsilon_{\min}, \frac{k}{k + \exp(i/k)}) \quad \text{(inverse sigmoid)}$$

## Implementation

```python
def scheduled_sampling_step(model, x, epsilon):
    batch_size, seq_len = x.shape
    input_tokens = x[:, 0:1]  # Start with ground truth BOS
    
    for t in range(1, seq_len):
        logits = model(input_tokens)
        predicted = logits[:, -1].argmax(dim=-1, keepdim=True)
        
        # Coin flip: use ground truth or prediction
        use_gt = (torch.rand(batch_size, 1) < epsilon).to(x.device)
        next_token = torch.where(use_gt, x[:, t:t+1], predicted)
        input_tokens = torch.cat([input_tokens, next_token], dim=1)
    
    # Compute loss on full sequence
    full_logits = model(input_tokens[:, :-1])
    return F.cross_entropy(full_logits.reshape(-1, vocab_size),
                          x[:, 1:].reshape(-1))
```

## Limitations

Scheduled sampling is biased: the training objective no longer corresponds to maximum likelihood because the input distribution is a mixture of ground truth and model predictions. This can lead to inconsistent training in theory, though it often helps in practice.

## Alternatives

- **Professor forcing**: use a discriminator to match the hidden state distributions of teacher-forced and free-running modes
- **Differentiable sampling**: use Gumbel-softmax or straight-through estimators to make sampling differentiable
