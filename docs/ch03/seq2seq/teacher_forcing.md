# Teacher Forcing

## Introduction

Teacher forcing is a training strategy for sequence-to-sequence models where the decoder receives the ground truth token from the previous timestep as input, rather than its own prediction. While this accelerates training, it creates a train-test mismatch known as **exposure bias**.

## The Training Dilemma

During inference, the decoder must use its own predictions:

```
Input: "Hello"
Step 1: <SOS> → "Bonjour" (correct)
Step 2: "Bonjour" → "!" (uses own prediction)
```

But during training with teacher forcing:

```
Step 1: <SOS> → "Bonjour" (target: "Bonjour")
Step 2: "Bonjour" (ground truth!) → "!" (target: "!")
```

The decoder never learns to recover from its own mistakes because it never sees them during training.

## Mathematical Formulation

### With Teacher Forcing

At training time $t$, the decoder receives ground truth $y_{t-1}^*$:

$$h_t = \text{Decoder}(y_{t-1}^*, h_{t-1})$$
$$P(y_t | y_{<t}^*, x) = \text{softmax}(W_o h_t)$$

Loss:
$$\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t^* | y_{<t}^*, x)$$

### Without Teacher Forcing (Free Running)

The decoder uses its own predictions $\hat{y}_{t-1}$:

$$h_t = \text{Decoder}(\hat{y}_{t-1}, h_{t-1})$$
$$P(y_t | \hat{y}_{<t}, x) = \text{softmax}(W_o h_t)$$

## Implementation

### Basic Teacher Forcing

```python
import torch
import torch.nn as nn

def train_with_teacher_forcing(model, src, tgt, criterion):
    """
    Train decoder with full teacher forcing.
    
    Args:
        model: Seq2Seq model
        src: Source sequence (batch, src_len)
        tgt: Target sequence (batch, tgt_len) including <SOS>
        criterion: Loss function
    """
    batch_size = src.size(0)
    tgt_len = tgt.size(1)
    vocab_size = model.decoder.vocab_size
    
    # Encode
    encoder_outputs, hidden = model.encoder(src)
    
    # Decode with teacher forcing
    outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size)
    
    for t in range(tgt_len - 1):
        # Input is ground truth token
        decoder_input = tgt[:, t:t+1]  # (batch, 1)
        
        output, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
        outputs[:, t] = output.squeeze(1)
    
    # Compute loss (ignore <SOS> in target)
    loss = criterion(outputs.view(-1, vocab_size), tgt[:, 1:].reshape(-1))
    
    return loss
```

### Mixed Teacher Forcing

Randomly choose between ground truth and prediction:

```python
def train_with_mixed_forcing(model, src, tgt, criterion, teacher_forcing_ratio=0.5):
    """
    Train with probabilistic teacher forcing.
    """
    batch_size = src.size(0)
    tgt_len = tgt.size(1)
    vocab_size = model.decoder.vocab_size
    
    encoder_outputs, hidden = model.encoder(src)
    
    outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size)
    decoder_input = tgt[:, 0:1]  # <SOS>
    
    for t in range(tgt_len - 1):
        output, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
        outputs[:, t] = output.squeeze(1)
        
        # Decide whether to use teacher forcing
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
        
        if use_teacher_forcing:
            decoder_input = tgt[:, t+1:t+2]  # Ground truth
        else:
            decoder_input = output.argmax(dim=-1)  # Own prediction
    
    loss = criterion(outputs.view(-1, vocab_size), tgt[:, 1:].reshape(-1))
    
    return loss
```

## Scheduled Sampling

Gradually decrease teacher forcing ratio during training:

```python
class ScheduledSampler:
    """
    Scheduled sampling: decay teacher forcing over training.
    """
    
    def __init__(self, schedule='linear', initial_ratio=1.0, 
                 final_ratio=0.0, decay_steps=10000):
        self.schedule = schedule
        self.initial_ratio = initial_ratio
        self.final_ratio = final_ratio
        self.decay_steps = decay_steps
        self.step = 0
    
    def get_ratio(self):
        """Get current teacher forcing ratio."""
        progress = min(self.step / self.decay_steps, 1.0)
        
        if self.schedule == 'linear':
            ratio = self.initial_ratio - progress * (self.initial_ratio - self.final_ratio)
        
        elif self.schedule == 'exponential':
            ratio = self.initial_ratio * (self.final_ratio / self.initial_ratio) ** progress
        
        elif self.schedule == 'inverse_sigmoid':
            k = 5  # Steepness
            ratio = self.initial_ratio - (self.initial_ratio - self.final_ratio) * \
                    (1 / (1 + np.exp(-k * (progress - 0.5))))
        
        return ratio
    
    def update(self):
        """Call after each training step."""
        self.step += 1


# Usage
scheduler = ScheduledSampler(schedule='linear', decay_steps=50000)

for epoch in range(num_epochs):
    for batch in dataloader:
        ratio = scheduler.get_ratio()
        loss = train_with_mixed_forcing(model, src, tgt, criterion, ratio)
        scheduler.update()
```

### Visualization of Schedules

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_schedules():
    """Compare different teacher forcing schedules."""
    steps = np.arange(10000)
    
    schedules = {
        'constant': np.ones_like(steps) * 0.5,
        'linear': 1.0 - steps / 10000,
        'exponential': 1.0 * (0.01) ** (steps / 10000),
        'inverse_sigmoid': 1.0 - 1 / (1 + np.exp(-5 * (steps/10000 - 0.5)))
    }
    
    plt.figure(figsize=(10, 6))
    for name, values in schedules.items():
        plt.plot(steps, values, label=name)
    
    plt.xlabel('Training Step')
    plt.ylabel('Teacher Forcing Ratio')
    plt.title('Teacher Forcing Schedules')
    plt.legend()
    plt.grid(True)
    plt.show()
```

## Curriculum Learning Integration

Combine teacher forcing with curriculum learning:

```python
class CurriculumScheduler:
    """
    Curriculum: start with easy examples + teacher forcing,
    progress to hard examples + free running.
    """
    
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.step = 0
    
    def get_config(self):
        progress = self.step / self.max_steps
        
        return {
            # Sequence length curriculum
            'max_src_len': int(10 + progress * 90),  # 10 → 100
            'max_tgt_len': int(10 + progress * 90),
            
            # Teacher forcing curriculum
            'teacher_forcing_ratio': max(0.0, 1.0 - progress),
            
            # Difficulty curriculum  
            'sample_difficulty': 'easy' if progress < 0.3 else 
                                 'medium' if progress < 0.7 else 'hard'
        }
    
    def update(self):
        self.step += 1
```

## Exposure Bias Analysis

### Measuring Exposure Bias

```python
def measure_exposure_bias(model, test_data):
    """
    Measure the gap between teacher-forced and free-running performance.
    """
    model.eval()
    
    tf_losses = []  # Teacher forcing
    fr_losses = []  # Free running
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for src, tgt in test_data:
            # Teacher forcing loss
            tf_output = forward_teacher_forcing(model, src, tgt)
            tf_loss = criterion(tf_output.view(-1, vocab_size), 
                               tgt[:, 1:].reshape(-1))
            tf_losses.append(tf_loss.mean().item())
            
            # Free running loss
            fr_output = forward_free_running(model, src, tgt.size(1))
            fr_loss = criterion(fr_output.view(-1, vocab_size),
                               tgt[:, 1:].reshape(-1))
            fr_losses.append(fr_loss.mean().item())
    
    tf_avg = np.mean(tf_losses)
    fr_avg = np.mean(fr_losses)
    gap = fr_avg - tf_avg
    
    print(f"Teacher Forcing Loss: {tf_avg:.4f}")
    print(f"Free Running Loss: {fr_avg:.4f}")
    print(f"Exposure Bias Gap: {gap:.4f} ({gap/tf_avg*100:.1f}%)")
    
    return tf_avg, fr_avg, gap
```

### Error Propagation Visualization

```python
def visualize_error_propagation(model, src, tgt):
    """
    Show how errors compound during free-running generation.
    """
    model.eval()
    
    # Teacher forcing: get "ideal" hidden states
    tf_hiddens = []
    encoder_outputs, hidden = model.encoder(src)
    for t in range(tgt.size(1) - 1):
        decoder_input = tgt[:, t:t+1]
        _, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
        tf_hiddens.append(hidden.clone())
    
    # Free running: get actual hidden states  
    fr_hiddens = []
    _, hidden = model.encoder(src)
    decoder_input = tgt[:, 0:1]  # <SOS>
    for t in range(tgt.size(1) - 1):
        output, hidden = model.decoder(decoder_input, hidden, encoder_outputs)
        decoder_input = output.argmax(dim=-1)
        fr_hiddens.append(hidden.clone())
    
    # Compute divergence over time
    divergences = []
    for tf_h, fr_h in zip(tf_hiddens, fr_hiddens):
        div = (tf_h - fr_h).norm().item()
        divergences.append(div)
    
    plt.figure(figsize=(10, 4))
    plt.plot(divergences)
    plt.xlabel('Timestep')
    plt.ylabel('Hidden State Divergence')
    plt.title('Error Propagation: Teacher Forcing vs Free Running')
    plt.show()
    
    return divergences
```

## Alternative Approaches

### Professor Forcing (GAN-based)

Train a discriminator to distinguish teacher-forced from free-running hidden states:

```python
class ProfessorForcing:
    """
    Adversarial training to reduce exposure bias.
    Discriminator classifies hidden states as teacher-forced or free-running.
    """
    
    def __init__(self, hidden_size):
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def train_step(self, model, src, tgt, criterion):
        # Get hidden states from both modes
        tf_hiddens = get_teacher_forcing_hiddens(model, src, tgt)
        fr_hiddens = get_free_running_hiddens(model, src, tgt.size(1))
        
        # Discriminator loss
        tf_pred = self.discriminator(tf_hiddens)
        fr_pred = self.discriminator(fr_hiddens)
        
        d_loss = -torch.log(tf_pred).mean() - torch.log(1 - fr_pred).mean()
        
        # Generator loss: fool discriminator
        g_loss = -torch.log(fr_pred).mean()
        
        return d_loss, g_loss
```

### Sequence-Level Training

Use REINFORCE to optimize sequence-level metrics:

```python
def sequence_level_training(model, src, tgt, reward_fn):
    """
    REINFORCE training to optimize sequence-level reward (e.g., BLEU).
    """
    # Sample from model
    sampled_sequence, log_probs = model.sample(src)
    
    # Compute reward
    reward = reward_fn(sampled_sequence, tgt)  # e.g., BLEU score
    
    # REINFORCE gradient
    baseline = reward.mean()
    loss = -((reward - baseline) * log_probs.sum(dim=1)).mean()
    
    return loss
```

## Best Practices

1. **Start with teacher forcing**: Faster initial convergence
2. **Schedule decay**: Gradually reduce ratio during training
3. **Monitor exposure bias**: Track gap between modes
4. **Use beam search at inference**: Partially mitigates errors
5. **Consider sequence-level training**: For final fine-tuning

## Summary

Teacher forcing trades training efficiency for a train-test mismatch:

| Aspect | Teacher Forcing | Free Running |
|--------|-----------------|--------------|
| Training speed | Fast | Slow |
| Gradient quality | Clean | Noisy |
| Inference match | Poor | Perfect |
| Error recovery | Not learned | Learned |

**Recommended approach**:
1. Train with teacher forcing (ratio=1.0) for initial epochs
2. Apply scheduled sampling to decay ratio
3. Fine-tune with lower ratio or sequence-level objectives
4. Use beam search at inference
