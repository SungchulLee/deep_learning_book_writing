# LSTM vs GRU Comparison

## Introduction

LSTM and GRU are the two dominant gated RNN architectures. While both solve the vanishing gradient problem, they differ in complexity, parameter count, and behavior. This section provides a comprehensive comparison to guide architecture selection.

## Architectural Comparison

### Structure Overview

| Aspect | LSTM | GRU |
|--------|------|-----|
| States | 2 (hidden $h$, cell $c$) | 1 (hidden $h$) |
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Gate coupling | Independent | Coupled (update = 1 - forget) |
| Output filtering | Yes (output gate) | No |

### Mathematical Formulations

**LSTM:**
$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)$$
$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)$$
$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)$$
$$\tilde{c}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
$$h_t = o_t \odot \tanh(c_t)$$

**GRU:**
$$z_t = \sigma(W_z[h_{t-1}, x_t] + b_z)$$
$$r_t = \sigma(W_r[h_{t-1}, x_t] + b_r)$$
$$\tilde{h}_t = \tanh(W_h[r_t \odot h_{t-1}, x_t] + b_h)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

## Parameter Count

```python
def count_parameters(input_size, hidden_size):
    """Compare parameter counts for LSTM and GRU."""
    
    # LSTM: 4 gates × (input weights + hidden weights + bias)
    # Each gate: (input_size × hidden_size) + (hidden_size × hidden_size) + hidden_size
    lstm_params = 4 * (input_size * hidden_size + 
                       hidden_size * hidden_size + 
                       hidden_size)
    
    # GRU: 3 components (reset, update, new) × same structure
    gru_params = 3 * (input_size * hidden_size + 
                      hidden_size * hidden_size + 
                      hidden_size)
    
    reduction = (lstm_params - gru_params) / lstm_params * 100
    
    return {
        'lstm': lstm_params,
        'gru': gru_params,
        'reduction': reduction
    }

# Example
params = count_parameters(input_size=256, hidden_size=512)
print(f"LSTM parameters: {params['lstm']:,}")
print(f"GRU parameters:  {params['gru']:,}")
print(f"GRU reduction:   {params['reduction']:.1f}%")
```

**Typical reduction**: GRU has ~25% fewer parameters than LSTM.

## Computational Performance

```python
import torch
import torch.nn as nn
import time

def benchmark_speed(input_size=128, hidden_size=256, seq_length=100, 
                    batch_size=32, num_iterations=100):
    """Benchmark LSTM vs GRU forward pass speed."""
    
    lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
    gru = nn.GRU(input_size, hidden_size, batch_first=True)
    
    x = torch.randn(batch_size, seq_length, input_size)
    
    # Warmup
    for _ in range(10):
        _ = lstm(x)
        _ = gru(x)
    
    # Benchmark LSTM
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(num_iterations):
        _ = lstm(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    lstm_time = (time.time() - start) / num_iterations
    
    # Benchmark GRU
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    for _ in range(num_iterations):
        _ = gru(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    gru_time = (time.time() - start) / num_iterations
    
    print(f"LSTM forward time: {lstm_time*1000:.2f} ms")
    print(f"GRU forward time:  {gru_time*1000:.2f} ms")
    print(f"GRU speedup: {lstm_time/gru_time:.2f}x")
    
    return lstm_time, gru_time

benchmark_speed()
```

**Typical result**: GRU is 10-25% faster than LSTM.

## Memory Capacity

### Cell State Advantage

LSTM's separate cell state provides a dedicated memory channel:

```python
def analyze_memory_capacity():
    """
    LSTM can store information in cell state while outputting 
    different information via hidden state (filtered by output gate).
    GRU must store everything in one state vector.
    """
    
    # LSTM: Cell can store info independently of output
    # h_t = o_t ⊙ tanh(c_t)
    # When o_t ≈ 0, information in c_t is hidden but preserved
    
    # GRU: State is always exposed
    # h_t = (1-z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
    # No way to hide state information
    
    print("LSTM Memory Modes:")
    print("  - Store in cell, expose via output gate")
    print("  - Can hide information temporarily")
    print("  - Cell provides 'scratch space'")
    print()
    print("GRU Memory Modes:")
    print("  - Single state vector")
    print("  - All information directly accessible")
    print("  - More efficient but less flexible")
```

### Effective Memory Length

```python
def compare_memory_length(seq_lengths=[50, 100, 200, 500]):
    """Compare effective memory via gradient analysis."""
    
    results = {'lstm': [], 'gru': []}
    
    for seq_len in seq_lengths:
        # Measure gradient at first timestep when backprop from last
        for name, model_class in [('lstm', nn.LSTM), ('gru', nn.GRU)]:
            model = model_class(64, 128, batch_first=True)
            x = torch.randn(1, seq_len, 64, requires_grad=True)
            
            outputs, _ = model(x)
            loss = outputs[0, -1, :].sum()
            loss.backward()
            
            # Gradient at first timestep
            first_grad = x.grad[0, 0, :].norm().item()
            last_grad = x.grad[0, -1, :].norm().item()
            
            ratio = first_grad / (last_grad + 1e-10)
            results[name].append(ratio)
    
    print("Gradient Retention (first/last timestep):")
    print(f"{'Seq Length':<12} {'LSTM':<12} {'GRU':<12}")
    print("-" * 36)
    for i, seq_len in enumerate(seq_lengths):
        print(f"{seq_len:<12} {results['lstm'][i]:<12.2e} {results['gru'][i]:<12.2e}")
```

## Task-Specific Performance

### Empirical Results Summary

Based on published benchmarks and common experience:

| Task | Better Architecture | Notes |
|------|---------------------|-------|
| Language Modeling | LSTM ≈ GRU | Similar perplexity |
| Machine Translation | LSTM slightly | Long dependencies |
| Speech Recognition | LSTM | Complex temporal patterns |
| Sentiment Analysis | GRU ≈ LSTM | Both work well |
| Time Series | GRU | Often sufficient |
| Small Datasets | GRU | Less overfitting |
| Named Entity Recognition | Both | Task-dependent |

### When to Choose LSTM

- Very long sequences (500+ tokens)
- Tasks requiring complex memory management
- When you need output filtering
- Large datasets (can use extra capacity)
- Baseline comparisons (more established)

### When to Choose GRU

- Shorter sequences
- Limited computational resources
- Smaller datasets (regularization via simplicity)
- Real-time applications (speed matters)
- When LSTM shows no improvement

## Empirical Comparison Framework

```python
def compare_on_task(train_data, val_data, input_size, hidden_size, 
                    num_classes, epochs=20):
    """
    Compare LSTM and GRU on a specific task.
    """
    from torch.utils.data import DataLoader
    
    results = {}
    
    for name, rnn_class in [('LSTM', nn.LSTM), ('GRU', nn.GRU)]:
        # Build model
        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.rnn = rnn_class(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, num_classes)
            
            def forward(self, x):
                out, _ = self.rnn(x)
                return self.fc(out[:, -1, :])
        
        model = Classifier()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Train
        train_losses = []
        val_accs = []
        
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for x, y in train_data:
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss / len(train_data))
            
            # Validate
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for x, y in val_data:
                    preds = model(x).argmax(dim=1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            val_accs.append(correct / total)
        
        results[name] = {
            'train_losses': train_losses,
            'val_accs': val_accs,
            'final_acc': val_accs[-1],
            'params': sum(p.numel() for p in model.parameters())
        }
    
    # Report
    print("\nComparison Results:")
    print(f"{'Metric':<20} {'LSTM':<15} {'GRU':<15}")
    print("-" * 50)
    print(f"{'Final Val Accuracy':<20} {results['LSTM']['final_acc']:.4f}{'':>8} {results['GRU']['final_acc']:.4f}")
    print(f"{'Parameters':<20} {results['LSTM']['params']:,}{'':>5} {results['GRU']['params']:,}")
    
    return results
```

## Hybrid Approaches

### Stacked LSTM + GRU

```python
class HybridRNN(nn.Module):
    """Use GRU for lower layers, LSTM for top layer."""
    
    def __init__(self, input_size, hidden_size, num_gru_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_gru_layers, 
                          batch_first=True)
        self.lstm = nn.LSTM(hidden_size, hidden_size, 1, batch_first=True)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        lstm_out, (h_n, c_n) = self.lstm(gru_out)
        return lstm_out, h_n
```

### Architecture Search

For critical applications, try both:

```python
def architecture_search(data, configs):
    """
    Simple architecture search over LSTM/GRU configurations.
    """
    results = []
    
    for config in configs:
        arch = config['arch']  # 'lstm' or 'gru'
        hidden = config['hidden_size']
        layers = config['num_layers']
        
        # Train and evaluate
        acc = train_and_evaluate(data, arch, hidden, layers)
        
        results.append({
            **config,
            'accuracy': acc
        })
    
    # Sort by accuracy
    results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    print("Architecture Search Results:")
    for r in results[:5]:
        print(f"  {r['arch'].upper()} h={r['hidden_size']} L={r['num_layers']}: {r['accuracy']:.4f}")
    
    return results
```

## Decision Flowchart

```
Start
  │
  ▼
Is sequence length > 500?
  │
  ├─ Yes → Consider LSTM (better long-term memory)
  │
  ▼ No
Is dataset small (< 10k samples)?
  │
  ├─ Yes → Use GRU (fewer parameters, less overfitting)
  │
  ▼ No
Is training speed critical?
  │
  ├─ Yes → Use GRU (faster)
  │
  ▼ No
Try both → Pick based on validation performance
```

## Summary

| Factor | LSTM | GRU | Winner |
|--------|------|-----|--------|
| Parameters | More | Fewer | GRU |
| Speed | Slower | Faster | GRU |
| Memory capacity | Higher | Lower | LSTM |
| Long sequences | Better | Good | LSTM |
| Small datasets | Overfits more | Regularized | GRU |
| Interpretability | Complex | Simpler | GRU |
| Default choice | Traditional | Modern | Tie |

**Practical recommendation**: Start with GRU. Switch to LSTM if:
- Validation performance plateaus
- Task involves very long dependencies
- You have sufficient data and compute

Both architectures are mature and well-optimized—the differences are often smaller than other hyperparameter choices like hidden size, learning rate, or regularization.
