# Teacher Forcing

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

$$\mathbf{h}_t = \text{Decoder}(y_{t-1}^*, \mathbf{h}_{t-1})$$

$$P(y_t | y_{<t}^*, \mathbf{x}) = \text{softmax}(\mathbf{W}_o \mathbf{h}_t)$$

The training objective becomes:

$$\mathcal{L}_{TF} = -\sum_{t=1}^{T} \log P_\theta(y_t^* | y_1^*, y_2^*, \ldots, y_{t-1}^*, \mathbf{c})$$

Each prediction conditions on the correct previous tokens, creating independent classification problems at each timestep.

### Training Objective without Teacher Forcing

The decoder uses its own predictions $\hat{y}_{t-1}$:

$$\mathbf{h}_t = \text{Decoder}(\hat{y}_{t-1}, \mathbf{h}_{t-1})$$

$$P(y_t | \hat{y}_{<t}, \mathbf{x}) = \text{softmax}(\mathbf{W}_o \mathbf{h}_t)$$

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

$$P(\text{error at } t+k \mid \text{error at } t) > P(\text{error at } t+k)$$

The model has never learned to recover from such errors because it never encountered them during training. Once the decoder generates a wrong token, it enters an unfamiliar hidden state distribution, and subsequent predictions are likely to be further from the training distribution.

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

Randomly choosing between ground truth and prediction at each timestep provides a simple form of curriculum learning:

```python
def train_with_mixed_forcing(model, src, tgt, criterion, teacher_forcing_ratio=0.5):
    """
    Train with probabilistic teacher forcing.
    
    At each timestep, independently decides whether to use ground truth
    or the model's own prediction. With ratio=0.5, the model sees its
    own predictions roughly half the time, building some robustness
    to imperfect inputs.
    """
    batch_size = src.size(0)
    tgt_len = tgt.size(1)
    vocab_size = model.decoder.output_size
    
    encoder_outputs, hidden, cell = model.encoder(src)
    
    outputs = torch.zeros(batch_size, tgt_len - 1, vocab_size)
    decoder_input = tgt[:, 0:1]  # <SOS>
    
    for t in range(tgt_len - 1):
        output, hidden, cell = model.decoder(decoder_input, hidden, cell)
        outputs[:, t] = output
        
        # Per-timestep teacher forcing decision
        use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
        
        if use_teacher_forcing:
            decoder_input = tgt[:, t+1:t+2]  # Ground truth
        else:
            decoder_input = output.argmax(dim=-1).unsqueeze(1)  # Own prediction
    
    loss = criterion(outputs.view(-1, vocab_size), tgt[:, 1:].reshape(-1))
    return loss
```

## Exposure Bias Analysis

### Measuring Exposure Bias

The gap between teacher-forced and free-running performance directly quantifies exposure bias:

```python
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
    A large gap between ratio=1.0 and ratio=0.0 indicates
    severe exposure bias.
    
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
                
                outputs = model(src, trg, teacher_forcing_ratio=ratio)
                
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
import matplotlib.pyplot as plt


def visualize_error_propagation(model, src, tgt, encoder_outputs):
    """
    Show how errors compound during free-running generation.
    
    Computes the divergence between hidden states under teacher forcing
    (ideal trajectory) and free running (actual inference trajectory).
    Growing divergence indicates compounding errors.
    """
    model.eval()
    
    # Teacher forcing: get "ideal" hidden states
    tf_hiddens = []
    _, hidden, cell = model.encoder(src)
    for t in range(tgt.size(1) - 1):
        decoder_input = tgt[:, t:t+1]
        _, hidden, cell = model.decoder(decoder_input, hidden, cell)
        tf_hiddens.append(hidden.clone())
    
    # Free running: get actual hidden states  
    fr_hiddens = []
    _, hidden, cell = model.encoder(src)
    decoder_input = tgt[:, 0:1]  # <SOS>
    for t in range(tgt.size(1) - 1):
        output, hidden, cell = model.decoder(decoder_input, hidden, cell)
        decoder_input = output.argmax(dim=-1).unsqueeze(1)
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

### Professor Forcing (GAN-Based)

Professor forcing (Lamb et al., 2016) trains a discriminator to distinguish teacher-forced from free-running hidden state trajectories, pushing the free-running dynamics to match the teacher-forced dynamics:

```python
class ProfessorForcing:
    """
    Adversarial training to reduce exposure bias.
    Discriminator classifies hidden state trajectories as 
    teacher-forced or free-running.
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

### Sequence-Level Training with REINFORCE

Rather than training with token-level cross-entropy, optimize directly for sequence-level metrics (e.g., BLEU):

```python
def sequence_level_training(model, src, tgt, reward_fn):
    """
    REINFORCE training to optimize sequence-level reward (e.g., BLEU).
    
    Addresses exposure bias by training the model to generate
    complete sequences and evaluating them holistically.
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

Teacher forcing is most beneficial when training from scratch with random initialization, for tasks with long sequences where error accumulation is problematic, and when training time is limited and fast convergence is important.

Consider reducing or eliminating teacher forcing when fine-tuning a pre-trained model, when the model shows a significant train-test gap, or when sequences are short enough that error accumulation is minimal.

### Comparison of Strategies

| Aspect | Teacher Forcing | Free Running |
|--------|-----------------|--------------|
| Training speed | Fast | Slow |
| Gradient quality | Clean, stable | Noisy, may vanish |
| Inference match | Poor | Perfect |
| Error recovery | Not learned | Learned |
| Convergence | Rapid | Slow |

### Recommended Training Schedule

| Phase | Epochs | TF Ratio | Purpose |
|-------|--------|----------|---------|
| Early | 1–10 | 0.9–1.0 | Stable gradient flow and rapid initial learning |
| Mid | 10–30 | 0.5–0.9 | Begin exposing model to its own predictions |
| Late | 30+ | 0.0–0.5 | Minimize exposure bias before deployment |

## Summary

Teacher forcing is a powerful technique for accelerating sequence-to-sequence training by providing ground truth inputs during decoding. The key insight is the fundamental trade-off between training efficiency and inference robustness: teacher forcing provides stable, efficient gradients by creating independent per-step classification problems, but the resulting model has never encountered its own prediction errors and cannot recover from them at inference time.

The exposure bias grows with sequence length as the divergence between training (ground truth context) and inference (predicted context) distributions compounds. Practical mitigation strategies include gradually reducing the teacher forcing ratio during training (scheduled sampling, covered in detail in the next section), using adversarial professor forcing to align hidden state distributions, and employing sequence-level training with REINFORCE to optimize holistic output quality.

The recommended approach is to start with high teacher forcing for fast initial convergence, then gradually reduce it to prepare the model for autoregressive inference, and finally use beam search at inference time to partially compensate for remaining exposure bias.
