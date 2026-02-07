# Scheduled Sampling

Scheduled sampling (Bengio et al., 2015) provides a principled approach to bridging the train-test gap created by teacher forcing. Rather than switching abruptly from teacher forcing to free running, scheduled sampling **gradually transitions** between the two regimes over the course of training. This allows the model to benefit from stable early learning with teacher forcing while progressively learning to handle its own prediction errors.

## Core Idea

The central insight of scheduled sampling is that the teacher forcing ratio should be treated as a **curriculum parameter** that decreases over training. Early in training, the model benefits from ground truth inputs because its predictions are essentially random. As training progresses and predictions improve, the model should increasingly encounter its own outputs to prepare for inference conditions.

At each decoding timestep, the training procedure independently decides whether to use the ground truth token or the model's prediction:

$$\text{input}_t = \begin{cases} y_{t-1}^* & \text{with probability } 1 - \epsilon_i \\ \hat{y}_{t-1} & \text{with probability } \epsilon_i \end{cases}$$

where $\epsilon_i$ is the sampling probability at training step $i$, starting near 0 (mostly teacher forcing) and increasing toward 1 (mostly free running).

## Decay Schedules

The choice of decay schedule determines how quickly the model transitions from teacher forcing to free running. Three standard schedules are commonly used:

### Linear Decay

The simplest schedule decreases the teacher forcing ratio at a constant rate:

$$\text{ratio}(i) = \max\left(\text{ratio}_{\min},\; \text{ratio}_0 - k \cdot i\right)$$

Linear decay provides a predictable, steady transition. It is easy to configure (choose the number of steps over which to decay from 1.0 to the minimum) and works well as a baseline.

### Exponential Decay

Exponential decay decreases rapidly at first, then slows:

$$\text{ratio}(i) = \text{ratio}_0 \cdot (1 - k)^i$$

This schedule front-loads the transition, spending more training steps in the low-teacher-forcing regime. It is appropriate when the model converges quickly and needs extended practice with its own predictions.

### Inverse Sigmoid Decay

The inverse sigmoid provides a smooth S-shaped transition, slow at the beginning and end with a rapid transition in the middle:

$$\text{ratio}(i) = \frac{k}{k + \exp(i / k)}$$

This schedule is often preferred because it maintains high teacher forcing during the critical early learning phase, transitions smoothly through the mid-training regime, and converges gently to the minimum ratio.

```
Teacher Forcing Ratio vs Training Steps

Ratio
1.0 ├─●──●──●
    │          ╲──── Inverse Sigmoid (slow start, fast middle)
0.8 ├─ ╲
    │    ╲──── Linear (constant rate)
0.6 ├     ╲
    │       ╲
0.4 ├  ╲     ╲
    │    ╲     ──── Exponential (fast start, slow end)
0.2 ├      ──── ╲
    │              ────────
0.0 ├
    └──┬──┬──┬──┬──┬──┬──┬──
       0  1k 2k 3k 4k 5k 6k 7k
              Training Step
```

## PyTorch Implementation

### Teacher Forcing Scheduler

```python
import math
import random
import numpy as np
import torch
import torch.nn as nn


class TeacherForcingScheduler:
    """
    Schedules teacher forcing ratio during training.
    
    Supports various decay strategies to gradually reduce teacher forcing,
    helping the model learn to handle its own predictions.
    
    Args:
        initial_ratio: Starting teacher forcing ratio (default: 1.0)
        min_ratio: Minimum ratio (default: 0.0)
        decay_type: 'linear', 'exponential', 'inverse_sigmoid' (default: 'linear')
        decay_rate: Rate parameter for decay (default: 0.01)
    """
    
    def __init__(
        self,
        initial_ratio: float = 1.0,
        min_ratio: float = 0.0,
        decay_type: str = 'linear',
        decay_rate: float = 0.01
    ):
        self.initial_ratio = initial_ratio
        self.min_ratio = min_ratio
        self.decay_type = decay_type
        self.decay_rate = decay_rate
        self.step_count = 0
        
    def get_ratio(self) -> float:
        """Get current teacher forcing ratio."""
        if self.decay_type == 'linear':
            ratio = self.initial_ratio - self.decay_rate * self.step_count
            
        elif self.decay_type == 'exponential':
            ratio = self.initial_ratio * (1 - self.decay_rate) ** self.step_count
            
        elif self.decay_type == 'inverse_sigmoid':
            # Smooth transition using inverse sigmoid
            k = self.decay_rate * 100  # Scale factor
            ratio = k / (k + math.exp(self.step_count / k))
            ratio = self.initial_ratio * ratio
            
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
        
        return max(ratio, self.min_ratio)
    
    def step(self) -> None:
        """Increment step counter."""
        self.step_count += 1
        
    def reset(self) -> None:
        """Reset scheduler state."""
        self.step_count = 0
```

### Epoch-Based Scheduler

When per-step scheduling introduces too much variance, epoch-level scheduling provides more stable control:

```python
class EpochBasedScheduler:
    """
    Teacher forcing scheduler based on epochs rather than steps.
    
    Useful when you want to maintain consistent teacher forcing
    within epochs and only change between epochs.
    """
    
    def __init__(
        self,
        schedule: dict = None,
        default_ratio: float = 0.5
    ):
        """
        Args:
            schedule: Dict mapping epoch -> ratio, e.g., {0: 1.0, 5: 0.75, 10: 0.5}
            default_ratio: Ratio to use for epochs not in schedule
        """
        self.schedule = schedule or {0: 1.0, 10: 0.5, 20: 0.25}
        self.default_ratio = default_ratio
        
    def get_ratio(self, epoch: int) -> float:
        """Get teacher forcing ratio for given epoch."""
        # Find the most recent scheduled ratio
        applicable_epochs = [e for e in self.schedule.keys() if e <= epoch]
        
        if applicable_epochs:
            return self.schedule[max(applicable_epochs)]
        return self.default_ratio
```

### Training Step with Scheduled Sampling

```python
def train_step_scheduled_sampling(
    model: nn.Module,
    src: torch.Tensor,
    trg: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epsilon: float,
    device: torch.device
) -> float:
    """
    Training step with scheduled sampling.
    
    At each timestep, independently decides whether to use
    ground truth or model prediction based on sampling schedule.
    
    Args:
        model: Seq2seq model
        src: Source sequences (batch_size, src_len)
        trg: Target sequences (batch_size, trg_len)
        criterion: Loss function
        optimizer: Optimizer
        epsilon: Probability of using model's own prediction
        device: Computation device
        
    Returns:
        loss: Training loss value
    """
    model.train()
    optimizer.zero_grad()
    
    batch_size = src.size(0)
    trg_len = trg.size(1)
    vocab_size = model.decoder.output_size
    
    encoder_outputs, hidden, cell = model.encoder(src)
    decoder_input = trg[:, 0].unsqueeze(1)
    
    outputs = torch.zeros(batch_size, trg_len, vocab_size, device=device)
    
    for t in range(1, trg_len):
        if hasattr(model.decoder, 'attention'):
            output, hidden, cell, _ = model.decoder(
                decoder_input, hidden, encoder_outputs, cell
            )
        else:
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
        
        outputs[:, t] = output
        
        # Per-timestep sampling decision
        if random.random() < epsilon:
            # Use model's prediction (building inference robustness)
            decoder_input = output.argmax(dim=-1).unsqueeze(1)
        else:
            # Use ground truth (stable learning signal)
            decoder_input = trg[:, t].unsqueeze(1)
    
    output_dim = outputs.size(-1)
    outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
    trg_flat = trg[:, 1:].contiguous().view(-1)
    
    loss = criterion(outputs, trg_flat)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()
```

### Complete Training Loop

```python
def train_with_scheduled_tf(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    device: torch.device,
    tf_scheduler: TeacherForcingScheduler = None
) -> dict:
    """
    Training loop with scheduled teacher forcing.
    
    Args:
        model: Seq2seq model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Computation device
        tf_scheduler: Teacher forcing scheduler
        
    Returns:
        history: Training history dict
    """
    if tf_scheduler is None:
        tf_scheduler = TeacherForcingScheduler(
            initial_ratio=1.0,
            min_ratio=0.1,
            decay_type='exponential',
            decay_rate=0.02
        )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'tf_ratio': []
    }
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        # Get current teacher forcing ratio
        tf_ratio = tf_scheduler.get_ratio()
        epsilon = 1.0 - tf_ratio  # Sampling probability
        history['tf_ratio'].append(tf_ratio)
        
        for batch in train_loader:
            src, trg, src_lengths, _ = batch
            src, trg = src.to(device), trg.to(device)
            
            loss = train_step_scheduled_sampling(
                model, src, trg, criterion, optimizer,
                epsilon=epsilon,
                device=device
            )
            epoch_loss += loss
        
        # Validation (always without teacher forcing)
        val_loss = evaluate_model(model, val_loader, criterion, device)
        
        history['train_loss'].append(epoch_loss / len(train_loader))
        history['val_loss'].append(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  TF Ratio: {tf_ratio:.3f}")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Update scheduler
        tf_scheduler.step()
    
    return history


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Evaluate model without teacher forcing."""
    model.eval()
    total_loss = 0
    
    for batch in data_loader:
        src, trg, src_lengths, _ = batch
        src, trg = src.to(device), trg.to(device)
        
        # Forward pass without teacher forcing
        outputs = model(src, trg, teacher_forcing_ratio=0.0)
        
        output_dim = outputs.size(-1)
        outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
        trg_flat = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(outputs, trg_flat)
        total_loss += loss.item()
    
    return total_loss / len(data_loader)
```

## Curriculum Learning Integration

A more sophisticated approach treats teacher forcing scheduling as part of a broader curriculum, combining it with sequence difficulty progression:

### Performance-Adaptive Scheduling

Rather than following a fixed decay, adapt the teacher forcing ratio based on model competence:

```python
class CurriculumTeacherForcing:
    """
    Curriculum learning approach to teacher forcing.
    
    Adjusts teacher forcing based on model performance,
    reducing ratio when the model demonstrates competence.
    
    Args:
        initial_ratio: Starting ratio
        threshold_loss: Loss threshold for reducing ratio
        reduction_factor: Factor to multiply ratio when threshold is met
        patience: Number of epochs to wait before reducing
        min_ratio: Minimum ratio
    """
    
    def __init__(
        self,
        initial_ratio: float = 1.0,
        threshold_loss: float = 2.0,
        reduction_factor: float = 0.9,
        patience: int = 3,
        min_ratio: float = 0.1
    ):
        self.ratio = initial_ratio
        self.threshold_loss = threshold_loss
        self.reduction_factor = reduction_factor
        self.patience = patience
        self.min_ratio = min_ratio
        
        self.best_loss = float('inf')
        self.epochs_below_threshold = 0
        
    def get_ratio(self) -> float:
        return self.ratio
    
    def step(self, val_loss: float) -> None:
        """
        Update ratio based on validation loss.
        
        Args:
            val_loss: Current validation loss
        """
        if val_loss < self.threshold_loss:
            self.epochs_below_threshold += 1
            
            if self.epochs_below_threshold >= self.patience:
                # Model is performing well, reduce teacher forcing
                self.ratio = max(
                    self.ratio * self.reduction_factor,
                    self.min_ratio
                )
                self.epochs_below_threshold = 0
                self.threshold_loss *= 0.95  # Also reduce threshold
        else:
            self.epochs_below_threshold = 0
        
        self.best_loss = min(self.best_loss, val_loss)
```

### Full Curriculum Scheduler

Combining teacher forcing decay with sequence difficulty progression provides a comprehensive curriculum:

```python
class CurriculumScheduler:
    """
    Full curriculum: start with easy examples + teacher forcing,
    progress to hard examples + free running.
    
    Three curriculum dimensions change simultaneously:
    1. Sequence length increases (short → long)
    2. Teacher forcing decreases (guided → independent)
    3. Example difficulty increases (easy → hard)
    """
    
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.current_step = 0
    
    def get_config(self):
        progress = self.current_step / self.max_steps
        
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
    
    def step(self):
        self.current_step += 1
```

## Scheduled Sampling Variants

### Token-Level vs Sequence-Level

Standard scheduled sampling makes an independent decision at each timestep. An alternative is **sequence-level** sampling, where the entire sequence uses either teacher forcing or free running:

```python
def sequence_level_sampling(model, src, trg, criterion, epsilon):
    """
    Sequence-level scheduled sampling.
    
    Either the entire sequence uses teacher forcing, or the entire
    sequence uses free running. This avoids the inconsistency of
    mixing ground truth and predictions within a single sequence.
    """
    if random.random() < epsilon:
        # Entire sequence: free running
        outputs = model(src, trg, teacher_forcing_ratio=0.0)
    else:
        # Entire sequence: teacher forcing
        outputs = model(src, trg, teacher_forcing_ratio=1.0)
    
    output_dim = outputs.size(-1)
    outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
    trg_flat = trg[:, 1:].contiguous().view(-1)
    
    return criterion(outputs, trg_flat)
```

### Soft Scheduled Sampling

Instead of a hard binary decision, use the model's prediction distribution as a soft mixture with the ground truth embedding:

```python
def soft_scheduled_sampling_step(model, decoder_input_gt, output_logits, epsilon):
    """
    Soft mixing of ground truth and predicted embeddings.
    
    Rather than choosing one or the other, interpolate between
    ground truth embedding and expected embedding under the model's
    predicted distribution. This provides a smoother training signal.
    """
    # Ground truth embedding
    gt_embedding = model.decoder.embedding(decoder_input_gt)
    
    # Expected embedding under predicted distribution
    probs = torch.softmax(output_logits, dim=-1)
    pred_embedding = probs @ model.decoder.embedding.weight
    
    # Soft interpolation
    mixed_embedding = (1 - epsilon) * gt_embedding + epsilon * pred_embedding
    
    return mixed_embedding
```

## Visualization of Schedule Effects

```python
def plot_schedules():
    """Compare different teacher forcing schedules."""
    steps = np.arange(10000)
    
    schedules = {
        'constant (0.5)': np.ones_like(steps, dtype=float) * 0.5,
        'linear': np.maximum(0.0, 1.0 - steps / 10000),
        'exponential': 1.0 * (0.01) ** (steps / 10000),
        'inverse sigmoid': 1.0 - 1 / (1 + np.exp(-5 * (steps / 10000 - 0.5)))
    }
    
    plt.figure(figsize=(10, 6))
    for name, values in schedules.items():
        plt.plot(steps, values, label=name)
    
    plt.xlabel('Training Step')
    plt.ylabel('Teacher Forcing Ratio')
    plt.title('Teacher Forcing Decay Schedules')
    plt.legend()
    plt.grid(True)
    plt.show()
```

## Practical Recommendations

### Schedule Selection Guide

| Schedule | Characteristics | Best For |
|----------|----------------|----------|
| Linear | Predictable, constant rate | Baseline, well-understood tasks |
| Exponential | Fast initial decay, slow tail | Quick convergence tasks |
| Inverse sigmoid | Slow start, fast middle, slow end | Most seq2seq tasks |
| Performance-adaptive | Data-driven transitions | Variable-difficulty tasks |

### Hyperparameter Guidelines

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| Initial ratio | 0.9–1.0 | Start with mostly teacher forcing |
| Minimum ratio | 0.0–0.2 | Some residual teacher forcing can help |
| Decay steps | 5K–50K | Depends on dataset size |
| Linear decay rate | 1e-4–1e-3 | Per step |
| Exponential decay rate | 0.01–0.05 | Per epoch |

### Common Pitfalls

**Decaying too fast**: If the model hasn't learned basic patterns before teacher forcing is reduced, it generates random tokens, and the training signal from its own predictions is meaningless. The model may never recover.

**Decaying too slow**: If teacher forcing remains high for too long, the model becomes overly dependent on ground truth inputs. The exposure bias gap remains large, and inference quality suffers.

**Ignoring validation performance**: Fixed schedules don't account for task difficulty or model capacity. Performance-adaptive scheduling responds to actual model competence rather than following a predetermined curve.

## Summary

Scheduled sampling provides a principled approach to mitigating exposure bias by gradually transitioning from teacher forcing to free running over the course of training. The key design choices are the decay schedule shape (linear, exponential, or inverse sigmoid), the decay rate, and whether to adapt based on model performance.

The inverse sigmoid schedule is generally recommended as it provides a slow, stable start that preserves early learning, a rapid mid-training transition that efficiently builds robustness, and a gentle convergence that avoids abrupt changes near the end of training.

Performance-adaptive approaches offer the most flexibility, automatically reducing teacher forcing when the model demonstrates competence, but require careful tuning of the threshold and patience parameters. Combining teacher forcing scheduling with sequence difficulty curricula provides the most comprehensive approach, simultaneously controlling what the model sees (easy vs. hard examples), how it is trained (guided vs. independent), and when it is challenged (early vs. late in training).
