# Teacher Forcing

## Introduction

Teacher forcing is a training strategy for sequence-to-sequence models where the ground truth output from the previous timestep is fed as input to the decoder at the current timestep, rather than the model's own prediction. This technique significantly accelerates training convergence but introduces a train-test discrepancy known as **exposure bias**.

The name derives from the analogy of a teacher who guides a student by providing correct answers, rather than letting the student's mistakes compound. While this "teaching" accelerates learning, it creates a fundamental tension: the model never learns to recover from its own errors because it never encounters them during training.

## Conceptual Foundation

During autoregressive sequence generation, the decoder produces tokens sequentially, with each prediction conditioned on previously generated tokens:

$$P(y_t | y_1, y_2, \ldots, y_{t-1}, \mathbf{c})$$

At training time, two strategies exist for providing the conditioning context $y_{<t}$:

**Free Running (Autoregressive)**: Use the model's own predictions from previous steps. If the model predicts $\hat{y}_{t-1}$, use $\hat{y}_{t-1}$ as input for predicting $y_t$.

**Teacher Forcing**: Use the ground truth $y_{t-1}^*$ as input for predicting $y_t$, regardless of what the model predicted.

### The Training-Inference Dilemma

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

The decoder never learns to recover from its own mistakes because it never sees them during training. This creates a distribution mismatch between training and inference conditions.

## Mathematical Analysis

### Training Objective with Teacher Forcing

At training time $t$, the decoder receives ground truth $y_{t-1}^*$:

$$h_t = \text{Decoder}(y_{t-1}^*, h_{t-1})$$
$$P(y_t | y_{<t}^*, x) = \text{softmax}(W_o h_t)$$

The training objective becomes:

$$\mathcal{L}_{TF} = -\sum_{t=1}^{T} \log P_\theta(y_t^* | y_1^*, y_2^*, \ldots, y_{t-1}^*, \mathbf{c})$$

Each prediction conditions on the correct previous tokens, creating independent classification problems at each timestep.

### Training Objective without Teacher Forcing

The decoder uses its own predictions $\hat{y}_{t-1}$:

$$h_t = \text{Decoder}(\hat{y}_{t-1}, h_{t-1})$$
$$P(y_t | \hat{y}_{<t}, x) = \text{softmax}(W_o h_t)$$

$$\mathcal{L}_{FR} = -\sum_{t=1}^{T} \log P_\theta(y_t^* | \hat{y}_1, \hat{y}_2, \ldots, \hat{y}_{t-1}, \mathbf{c})$$

where $\hat{y}_{t-1} = \arg\max P_\theta(y | \hat{y}_{<t-1}, \mathbf{c})$.

### Gradient Flow Comparison

With teacher forcing, gradients for timestep $t$ flow directly to the prediction layer without passing through previous timestep predictions:

$$\frac{\partial \mathcal{L}_t}{\partial \theta} = \frac{\partial \mathcal{L}_t}{\partial P_\theta(y_t)} \cdot \frac{\partial P_\theta(y_t)}{\partial \theta}$$

Without teacher forcing, gradients must flow through the entire sequence of predictions:

$$\frac{\partial \mathcal{L}_t}{\partial \theta} = \sum_{s=1}^{t} \frac{\partial \mathcal{L}_t}{\partial \hat{y}_s} \cdot \frac{\partial \hat{y}_s}{\partial \theta}$$

This explains why teacher forcing provides more stable and efficient gradients, but may not prepare the model for the compounding errors encountered during inference.

## Exposure Bias

The fundamental drawback of teacher forcing is **exposure bias**: the model is only exposed to ground truth sequences during training, but must operate on its own predictions during inference.

### Error Propagation

During inference, a single prediction error at timestep $t$ affects all subsequent predictions:

$$P(\text{error at } t+k | \text{error at } t) > P(\text{error at } t+k)$$

The model never learns to recover from its own mistakes because it never encounters them during training.

### Quantifying the Gap

Let $p_{data}(y_{<t})$ be the distribution of ground truth prefixes and $p_{model}(y_{<t})$ be the distribution of model-generated prefixes. The exposure bias can be measured as:

$$D_{KL}(p_{model}(y_{<t}) \| p_{data}(y_{<t}))$$

This divergence grows with sequence length as errors accumulate, potentially leading to significant quality degradation on long sequences.

## PyTorch Implementation

### Basic Teacher Forcing

```python
import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt


def train_step_teacher_forcing(
    model: nn.Module,
    src: torch.Tensor,
    trg: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    teacher_forcing_ratio: float = 1.0,
    clip: float = 1.0,
    device: torch.device = None
) -> float:
    """
    Single training step with configurable teacher forcing.
    
    Args:
        model: Seq2seq model with encoder and decoder
        src: Source sequences (batch_size, src_len)
        trg: Target sequences (batch_size, trg_len)
        criterion: Loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer
        teacher_forcing_ratio: Probability of using ground truth (0.0 to 1.0)
        clip: Gradient clipping value
        device: Computation device
        
    Returns:
        loss: Training loss value
    """
    model.train()
    optimizer.zero_grad()
    
    batch_size = src.size(0)
    trg_len = trg.size(1)
    vocab_size = model.decoder.output_size
    
    # Encode source
    encoder_outputs, hidden, cell = model.encoder(src)
    
    # Initialize decoder input with <sos>
    decoder_input = trg[:, 0].unsqueeze(1)
    
    # Storage for outputs
    outputs = torch.zeros(batch_size, trg_len, vocab_size, device=device)
    
    # Decode step by step
    for t in range(1, trg_len):
        # Decoder forward pass
        if hasattr(model.decoder, 'attention'):
            output, hidden, cell, _ = model.decoder(
                decoder_input, hidden, encoder_outputs, cell
            )
        else:
            output, hidden, cell = model.decoder(decoder_input, hidden, cell)
        
        outputs[:, t] = output
        
        # Teacher forcing decision
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        
        if use_teacher_forcing:
            # Use ground truth as next input
            decoder_input = trg[:, t].unsqueeze(1)
        else:
            # Use model's prediction as next input
            decoder_input = output.argmax(dim=-1).unsqueeze(1)
    
    # Compute loss (ignore <sos> position)
    output_dim = outputs.size(-1)
    outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
    trg_flat = trg[:, 1:].contiguous().view(-1)
    
    loss = criterion(outputs, trg_flat)
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    
    optimizer.step()
    
    return loss.item()
```

### Mixed Teacher Forcing

Randomly choose between ground truth and prediction at each timestep:

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

## Scheduled Teacher Forcing

Gradually reducing teacher forcing ratio during training can help bridge the train-test gap.

### Teacher Forcing Scheduler

```python
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
            import math
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

### Visualization of Schedules

```python
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

### Training Loop with Scheduled Teacher Forcing

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
        history['tf_ratio'].append(tf_ratio)
        
        for batch in train_loader:
            src, trg, src_lengths, _ = batch
            src, trg = src.to(device), trg.to(device)
            
            loss = train_step_teacher_forcing(
                model, src, trg, criterion, optimizer,
                teacher_forcing_ratio=tf_ratio,
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
        outputs, _ = model(src, trg, teacher_forcing_ratio=0.0)
        
        output_dim = outputs.size(-1)
        outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
        trg_flat = trg[:, 1:].contiguous().view(-1)
        
        loss = criterion(outputs, trg_flat)
        total_loss += loss.item()
    
    return total_loss / len(data_loader)
```

## Scheduled Sampling

Scheduled sampling (Bengio et al., 2015) provides a principled approach to mixing teacher forcing with free running:

```python
class ScheduledSampling:
    """
    Scheduled sampling for seq2seq training.
    
    Instead of using a fixed probability, scheduled sampling
    adaptively mixes ground truth and model predictions based
    on a curriculum schedule.
    
    Args:
        schedule: 'linear', 'exponential', 'inverse_sigmoid'
        k: Schedule parameter
    """
    
    def __init__(self, schedule: str = 'inverse_sigmoid', k: float = 1.0):
        self.schedule = schedule
        self.k = k
        self.step_count = 0
        
    def sample_probability(self) -> float:
        """
        Get probability of sampling from model distribution.
        
        Returns probability of using model's prediction (not ground truth).
        """
        import math
        i = self.step_count
        
        if self.schedule == 'linear':
            # Linear decay from 0 to 1
            epsilon = min(1.0, self.k * i)
            
        elif self.schedule == 'exponential':
            # Exponential decay: epsilon = 1 - k^i
            epsilon = 1.0 - self.k ** i
            
        elif self.schedule == 'inverse_sigmoid':
            # Inverse sigmoid: epsilon = k / (k + exp(k/i))
            if i == 0:
                epsilon = 0.0
            else:
                epsilon = self.k / (self.k + math.exp(self.k / i))
        
        return epsilon
    
    def step(self):
        self.step_count += 1


class ScheduledSamplerAlternative:
    """
    Alternative scheduled sampling implementation with more configuration options.
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


def train_step_scheduled_sampling(
    model: nn.Module,
    src: torch.Tensor,
    trg: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    sampler: ScheduledSampling,
    device: torch.device
) -> float:
    """
    Training step with scheduled sampling.
    
    At each timestep, independently decides whether to use
    ground truth or model prediction based on sampling schedule.
    """
    model.train()
    optimizer.zero_grad()
    
    batch_size = src.size(0)
    trg_len = trg.size(1)
    vocab_size = model.decoder.output_size
    
    epsilon = sampler.sample_probability()
    
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
            # Use model's prediction
            decoder_input = output.argmax(dim=-1).unsqueeze(1)
        else:
            # Use ground truth
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

## Curriculum Learning Integration

A more sophisticated approach treats teacher forcing scheduling as a curriculum, combining it with sequence difficulty progression:

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


class CurriculumScheduler:
    """
    Full curriculum: start with easy examples + teacher forcing,
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
def measure_exposure_bias(model, test_data, vocab_size):
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


def analyze_teacher_forcing_impact(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    tf_ratios: list = [0.0, 0.25, 0.5, 0.75, 1.0]
) -> dict:
    """
    Analyze model performance across different teacher forcing ratios.
    
    Helps identify the train-test gap due to exposure bias.
    
    Args:
        model: Trained seq2seq model
        val_loader: Validation data loader
        criterion: Loss function
        device: Computation device
        tf_ratios: List of teacher forcing ratios to test
        
    Returns:
        results: Dict mapping ratio -> average loss
    """
    model.eval()
    results = {}
    
    for ratio in tf_ratios:
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src, trg, _, _ = batch
                src, trg = src.to(device), trg.to(device)
                
                outputs, _ = model(src, trg, teacher_forcing_ratio=ratio)
                
                output_dim = outputs.size(-1)
                outputs = outputs[:, 1:].contiguous().view(-1, output_dim)
                trg_flat = trg[:, 1:].contiguous().view(-1)
                
                loss = criterion(outputs, trg_flat)
                total_loss += loss.item()
        
        results[ratio] = total_loss / len(val_loader)
        print(f"TF Ratio {ratio:.2f}: Loss = {results[ratio]:.4f}")
    
    # Calculate exposure bias gap
    gap = results[0.0] - results[1.0]
    print(f"\nExposure Bias Gap (TF=0 - TF=1): {gap:.4f}")
    
    return results
```

### Error Propagation Visualization

```python
def visualize_error_propagation(model, src, tgt, encoder_outputs):
    """
    Show how errors compound during free-running generation.
    """
    model.eval()
    
    # Teacher forcing: get "ideal" hidden states
    tf_hiddens = []
    _, hidden = model.encoder(src)
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
    plt.grid(True)
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
    
    def train_step(self, model, src, tgt, criterion, 
                   get_teacher_forcing_hiddens, get_free_running_hiddens):
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

## Practical Guidelines

### When to Use Teacher Forcing

Teacher forcing is most beneficial when:

- Training from scratch with random initialization
- The task has long sequences where error accumulation is problematic
- Training time is limited and fast convergence is important

Consider reducing or eliminating teacher forcing when:

- Fine-tuning a pre-trained model
- The model shows significant train-test gap
- Sequences are short enough that error accumulation is minimal

### Recommended Training Schedule

A typical training progression:

| Phase | Epochs | TF Ratio | Purpose |
|-------|--------|----------|---------|
| Early | 1-10 | 0.9-1.0 | Stable gradient flow and rapid initial learning |
| Mid | 10-30 | 0.5-0.9 | Begin exposing model to its own predictions |
| Late | 30+ | 0.0-0.5 | Minimize exposure bias before deployment |

## Summary

Teacher forcing is a powerful technique for accelerating sequence-to-sequence training by providing ground truth inputs during decoding. The key trade-offs are:

| Aspect | Teacher Forcing | Free Running |
|--------|-----------------|--------------|
| Training speed | Fast | Slow |
| Gradient quality | Clean, stable | Noisy, may vanish |
| Inference match | Poor | Perfect |
| Error recovery | Not learned | Learned |
| Convergence | Rapid | Slow |

**Key takeaways:**

1. **Trade-off**: Teacher forcing provides efficient, stable training at the cost of exposure bias, where the model never learns to recover from its own prediction errors.

2. **Scheduling**: Gradually reducing teacher forcing ratio during training (scheduled sampling) helps bridge the train-test gap while maintaining training stability.

3. **Curriculum Learning**: Combining teacher forcing decay with sequence difficulty progression provides a principled approach to training.

4. **Alternatives**: Advanced techniques like professor forcing and sequence-level training with REINFORCE offer more sophisticated approaches to handling the train-test discrepancy.

5. **Monitoring**: Evaluating model performance across different teacher forcing ratios helps quantify exposure bias and guides scheduling decisions.

**Recommended approach:**

1. Train with teacher forcing (ratio=1.0) for initial epochs
2. Apply scheduled sampling to decay ratio
3. Fine-tune with lower ratio or sequence-level objectives
4. Use beam search at inference to partially mitigate errors

The optimal teacher forcing strategy depends on the specific task, sequence lengths, and computational constraints. A common best practice is to start with high teacher forcing for fast initial convergence, then gradually reduce it to prepare the model for autoregressive inference.
