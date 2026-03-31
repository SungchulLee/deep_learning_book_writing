# Teacher Forcing

Training an autoregressive model poses a chicken-and-egg problem: the model predicts the next token based on previous tokens, but during training it has not yet learned to produce good predictions. **Teacher forcing** resolves this by supplying the ground-truth previous tokens as input during training, rather than the model's own (initially poor) predictions. This simple technique enables efficient parallel computation and stable gradient flow, but it introduces a train-test mismatch known as **exposure bias**.

## Mechanism

At training time, the model receives the true sequence shifted by one position:

$$
\text{Input: } (x_1, x_2, \ldots, x_{T-1}) \;\longrightarrow\; \text{Target: } (x_2, x_3, \ldots, x_T)
$$

The model predicts $x_t$ given the true prefix $x_{1:t-1}$ -- not its own previous predictions. Because every input token is known in advance, the entire sequence can be processed in a **single forward pass**, enabling GPU parallelism across all time steps.

## Advantages

1. **Parallel computation**: All positions can be computed simultaneously (no sequential dependency during training)
2. **Stable gradients**: The model always conditions on correct inputs, preventing error accumulation during the forward pass
3. **Fast convergence**: Direct supervision at every position provides a strong, position-specific learning signal

## Exposure Bias

The mismatch between training (conditions on ground truth) and inference (conditions on its own predictions) is called **exposure bias**. During generation, errors in early predictions propagate and compound because the model has never learned to recover from its own mistakes. The severity of exposure bias grows with sequence length.

## Mitigations

Several techniques reduce exposure bias:

- **Scheduled sampling**: Gradually replace ground-truth inputs with model predictions during training, starting with full teacher forcing and slowly increasing the self-prediction rate
- **Sequence-level training**: Use REINFORCE or other reinforcement learning methods to optimize sequence-level metrics (e.g., BLEU) rather than per-token cross-entropy
- **Data augmentation**: Inject noise into teacher-forced inputs to simulate the kinds of errors the model will encounter at inference time
- **Beam search**: At inference time, maintain multiple hypotheses to reduce the impact of individual prediction errors

## Implementation

```python
"""Teacher-forced training step for an autoregressive model."""
import torch.nn.functional as F


def teacher_forced_loss(model, x, vocab_size):
    """Compute cross-entropy loss with teacher forcing.

    Args:
        model: autoregressive model mapping (batch, seq_len) -> (batch, seq_len, vocab_size)
        x: ground-truth token IDs, shape (batch, seq_len)
        vocab_size: size of the vocabulary

    Returns:
        Scalar cross-entropy loss.
    """
    logits = model(x[:, :-1])   # input: all tokens except the last
    targets = x[:, 1:]           # target: all tokens except the first
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        targets.reshape(-1),
    )
```

!!! tip "Teacher Forcing in Transformers"
    In transformer-based models, teacher forcing is implemented via **causal masking**: the self-attention layer masks out future positions so that the prediction for position $t$ depends only on positions $1, \ldots, t-1$. This allows all positions to be computed in parallel while maintaining the autoregressive property.
