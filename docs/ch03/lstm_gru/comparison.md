# LSTM vs GRU Comparison

## Introduction

LSTM and GRU are the two dominant gated RNN architectures. While both solve the vanishing gradient problem, they differ in complexity, parameter count, and behavior. This section provides a comprehensive comparison—covering theory, empirical results, and practical decision frameworks—to guide architecture selection.

## Architectural Comparison

### Structure Overview

| Aspect | LSTM | GRU |
|--------|------|-----|
| States | 2 (hidden $h$, cell $c$) | 1 (hidden $h$) |
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Gate coupling | Independent | Coupled (update = 1 - forget) |
| Output filtering | Yes (output gate) | No |
| Year introduced | 1997 | 2014 |

### Mathematical Formulations

**LSTM:**
$$f_t = \sigma(W_f[h_{t-1}, x_t] + b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i[h_{t-1}, x_t] + b_i) \quad \text{(input gate)}$$
$$o_t = \sigma(W_o[h_{t-1}, x_t] + b_o) \quad \text{(output gate)}$$
$$\tilde{c}_t = \tanh(W_c[h_{t-1}, x_t] + b_c) \quad \text{(candidate)}$$
$$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(cell update)}$$
$$h_t = o_t \odot \tanh(c_t) \quad \text{(hidden state)}$$

**GRU:**
$$z_t = \sigma(W_z[h_{t-1}, x_t] + b_z) \quad \text{(update gate)}$$
$$r_t = \sigma(W_r[h_{t-1}, x_t] + b_r) \quad \text{(reset gate)}$$
$$\tilde{h}_t = \tanh(W_h[r_t \odot h_{t-1}, x_t] + b_h) \quad \text{(candidate)}$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(state update)}$$

### The Fundamental Design Difference

The key philosophical difference lies in how memory is managed:

**LSTM: Separation of Concerns**
- Cell state $c_t$: Long-term memory storage
- Hidden state $h_t$: Working memory / output
- Output gate: Controls what to expose
- This separation allows storing information without immediately using it

**GRU: Unified State**
- Single hidden state serves both purposes
- Simpler but less flexible
- What you store is what you expose

```python
import torch
import torch.nn as nn

def visualize_architecture_difference():
    """
    Conceptual illustration of LSTM vs GRU information flow.
    """
    print("LSTM Information Flow:")
    print("=" * 50)
    print("""
    x_t ─────┬────────────────────────────────┐
             │                                │
             ▼                                ▼
    ┌────────────────┐                ┌──────────────┐
    │  Gates (f,i,o) │                │  Candidate   │
    │   3 sigmoids   │                │    tanh      │
    └───────┬────────┘                └──────┬───────┘
            │                                │
            ▼                                ▼
    c_{t-1} ──→ [f_t × c_{t-1}] + [i_t × c̃_t] ──→ c_t
                                              │
                                              ▼
                                    [o_t × tanh(c_t)] ──→ h_t
    """)
    
    print("\nGRU Information Flow:")
    print("=" * 50)
    print("""
    x_t ─────┬────────────────────────────────┐
             │                                │
             ▼                                ▼
    ┌────────────────┐                ┌──────────────┐
    │  Gates (z,r)   │                │  Candidate   │
    │   2 sigmoids   │                │    tanh      │
    └───────┬────────┘                └──────┬───────┘
            │                                │
            ▼                                ▼
    h_{t-1} ──→ [(1-z_t) × h_{t-1}] + [z_t × h̃_t] ──→ h_t
    
    Note: Reset gate r_t applied INSIDE candidate computation
    """)
```

## Parameter Count Analysis

### Theoretical Calculation

```python
def count_parameters_detailed(input_size, hidden_size, num_layers=1, bidirectional=False):
    """
    Detailed parameter count comparison for LSTM and GRU.
    
    LSTM has 4 weight matrices per direction per layer:
    - W_ii, W_if, W_ig, W_io (input weights): 4 × input_size × hidden_size
    - W_hi, W_hf, W_hg, W_ho (hidden weights): 4 × hidden_size × hidden_size
    - Biases: 4 × hidden_size × 2 (input and hidden bias)
    
    GRU has 3 weight matrices per direction per layer:
    - W_ir, W_iz, W_in (input weights): 3 × input_size × hidden_size
    - W_hr, W_hz, W_hn (hidden weights): 3 × hidden_size × hidden_size
    - Biases: 3 × hidden_size × 2
    """
    num_directions = 2 if bidirectional else 1
    
    def lstm_params_per_layer(input_dim, hidden_dim):
        # 4 gates, each with input weights, hidden weights, and 2 biases
        input_weights = 4 * input_dim * hidden_dim
        hidden_weights = 4 * hidden_dim * hidden_dim
        biases = 4 * hidden_dim * 2  # bias_ih and bias_hh
        return input_weights + hidden_weights + biases
    
    def gru_params_per_layer(input_dim, hidden_dim):
        # 3 components, each with input weights, hidden weights, and 2 biases
        input_weights = 3 * input_dim * hidden_dim
        hidden_weights = 3 * hidden_dim * hidden_dim
        biases = 3 * hidden_dim * 2
        return input_weights + hidden_weights + biases
    
    lstm_total = 0
    gru_total = 0
    
    for layer in range(num_layers):
        # First layer takes input_size, subsequent layers take hidden_size * num_directions
        layer_input = input_size if layer == 0 else hidden_size * num_directions
        
        lstm_total += lstm_params_per_layer(layer_input, hidden_size) * num_directions
        gru_total += gru_params_per_layer(layer_input, hidden_size) * num_directions
    
    reduction_pct = (lstm_total - gru_total) / lstm_total * 100
    
    return {
        'lstm': lstm_total,
        'gru': gru_total,
        'difference': lstm_total - gru_total,
        'reduction_pct': reduction_pct,
        'gru_to_lstm_ratio': gru_total / lstm_total
    }


def parameter_scaling_analysis():
    """
    Analyze how parameter counts scale with model size.
    """
    configs = [
        (64, 128),    # Small
        (128, 256),   # Medium
        (256, 512),   # Large
        (512, 1024),  # XL
        (1024, 2048), # XXL
    ]
    
    print("Parameter Scaling Analysis")
    print("=" * 70)
    print(f"{'Input':<8} {'Hidden':<8} {'LSTM':<15} {'GRU':<15} {'Reduction':<10}")
    print("-" * 70)
    
    for input_size, hidden_size in configs:
        params = count_parameters_detailed(input_size, hidden_size)
        print(f"{input_size:<8} {hidden_size:<8} {params['lstm']:<15,} {params['gru']:<15,} {params['reduction_pct']:.1f}%")
    
    print("\nKey Insight: GRU consistently has 25% fewer parameters (3/4 ratio)")
    print("This is because GRU has 3 gate/component matrices vs LSTM's 4")

# Run analysis
parameter_scaling_analysis()
```

### Memory Footprint

Beyond parameters, consider activation memory during training:

```python
def memory_footprint_analysis(batch_size, seq_length, hidden_size):
    """
    Compare memory footprint during forward/backward pass.
    """
    # LSTM stores: h_t, c_t, and all gate activations for each timestep
    # Per timestep: h (hidden_size) + c (hidden_size) + 4 gates (4 * hidden_size)
    lstm_activations_per_step = 6 * hidden_size  # h, c, f, i, o, c_tilde
    lstm_total = batch_size * seq_length * lstm_activations_per_step * 4  # float32
    
    # GRU stores: h_t and gate activations
    # Per timestep: h (hidden_size) + 3 components (3 * hidden_size)  
    gru_activations_per_step = 4 * hidden_size  # h, z, r, h_tilde
    gru_total = batch_size * seq_length * gru_activations_per_step * 4
    
    print(f"Activation Memory (batch={batch_size}, seq={seq_length}, hidden={hidden_size}):")
    print(f"  LSTM: {lstm_total / 1024**2:.2f} MB")
    print(f"  GRU:  {gru_total / 1024**2:.2f} MB")
    print(f"  GRU saves: {(lstm_total - gru_total) / 1024**2:.2f} MB ({(lstm_total - gru_total) / lstm_total * 100:.1f}%)")

memory_footprint_analysis(batch_size=32, seq_length=512, hidden_size=512)
```

## Computational Performance

### Comprehensive Benchmarking

```python
import torch
import torch.nn as nn
import time
import numpy as np

def comprehensive_benchmark(device='cpu'):
    """
    Comprehensive speed benchmark across different configurations.
    """
    configs = [
        {'input_size': 64, 'hidden_size': 128, 'seq_length': 50, 'batch_size': 32},
        {'input_size': 128, 'hidden_size': 256, 'seq_length': 100, 'batch_size': 32},
        {'input_size': 256, 'hidden_size': 512, 'seq_length': 200, 'batch_size': 16},
        {'input_size': 512, 'hidden_size': 1024, 'seq_length': 100, 'batch_size': 8},
    ]
    
    results = []
    
    for cfg in configs:
        lstm = nn.LSTM(cfg['input_size'], cfg['hidden_size'], batch_first=True).to(device)
        gru = nn.GRU(cfg['input_size'], cfg['hidden_size'], batch_first=True).to(device)
        
        x = torch.randn(cfg['batch_size'], cfg['seq_length'], cfg['input_size']).to(device)
        
        # Warmup
        for _ in range(5):
            _ = lstm(x)
            _ = gru(x)
        
        num_iterations = 50
        
        # Forward pass benchmark
        if device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(num_iterations):
            _ = lstm(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        lstm_forward = (time.time() - start) / num_iterations
        
        start = time.time()
        for _ in range(num_iterations):
            _ = gru(x)
        if device == 'cuda':
            torch.cuda.synchronize()
        gru_forward = (time.time() - start) / num_iterations
        
        # Backward pass benchmark
        lstm.zero_grad()
        gru.zero_grad()
        
        start = time.time()
        for _ in range(num_iterations):
            out, _ = lstm(x)
            loss = out.sum()
            loss.backward()
            lstm.zero_grad()
        if device == 'cuda':
            torch.cuda.synchronize()
        lstm_backward = (time.time() - start) / num_iterations
        
        start = time.time()
        for _ in range(num_iterations):
            out, _ = gru(x)
            loss = out.sum()
            loss.backward()
            gru.zero_grad()
        if device == 'cuda':
            torch.cuda.synchronize()
        gru_backward = (time.time() - start) / num_iterations
        
        results.append({
            'config': f"i={cfg['input_size']}, h={cfg['hidden_size']}, s={cfg['seq_length']}",
            'lstm_forward': lstm_forward * 1000,
            'gru_forward': gru_forward * 1000,
            'lstm_backward': lstm_backward * 1000,
            'gru_backward': gru_backward * 1000,
            'forward_speedup': lstm_forward / gru_forward,
            'backward_speedup': lstm_backward / gru_backward
        })
    
    print("Comprehensive Benchmark Results")
    print("=" * 90)
    print(f"{'Configuration':<30} {'LSTM Fwd':<12} {'GRU Fwd':<12} {'Speedup':<10} {'LSTM Bwd':<12} {'GRU Bwd':<12}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['config']:<30} {r['lstm_forward']:.2f}ms{'':<5} {r['gru_forward']:.2f}ms{'':<5} "
              f"{r['forward_speedup']:.2f}x{'':<4} {r['lstm_backward']:.2f}ms{'':<5} {r['gru_backward']:.2f}ms")
    
    avg_speedup = np.mean([r['forward_speedup'] for r in results])
    print(f"\nAverage GRU speedup: {avg_speedup:.2f}x")
    
    return results
```

## Gradient Flow Comparison

### Theoretical Analysis

Both LSTM and GRU solve vanishing gradients, but through slightly different mechanisms:

**LSTM Gradient Path:**
$$\frac{\partial c_t}{\partial c_{t-1}} = f_t \quad \text{(direct path through cell state)}$$

**GRU Gradient Path:**
$$\frac{\partial h_t}{\partial h_{t-1}} = (1 - z_t) + z_t \cdot \frac{\partial \tilde{h}_t}{\partial h_{t-1}}$$

The $(1 - z_t)$ term in GRU plays a similar role to $f_t$ in LSTM.

```python
def compare_gradient_flow(seq_lengths=[50, 100, 200, 500, 1000]):
    """
    Empirically compare gradient flow between LSTM and GRU.
    """
    input_size = 64
    hidden_size = 128
    num_trials = 20
    
    results = {'lstm': {}, 'gru': {}}
    
    for seq_len in seq_lengths:
        for name, model_class in [('lstm', nn.LSTM), ('gru', nn.GRU)]:
            gradients = []
            
            for _ in range(num_trials):
                model = model_class(input_size, hidden_size, batch_first=True)
                x = torch.randn(1, seq_len, input_size, requires_grad=True)
                
                outputs, _ = model(x)
                loss = outputs[0, -1, :].sum()
                loss.backward()
                
                # Gradient at first timestep relative to last
                first_grad = x.grad[0, 0, :].norm().item()
                last_grad = x.grad[0, -1, :].norm().item()
                ratio = first_grad / (last_grad + 1e-10)
                gradients.append(ratio)
            
            results[name][seq_len] = {
                'mean': np.mean(gradients),
                'std': np.std(gradients)
            }
    
    print("Gradient Retention Analysis (first/last timestep ratio)")
    print("=" * 70)
    print(f"{'Seq Length':<12} {'LSTM Mean':<15} {'LSTM Std':<12} {'GRU Mean':<15} {'GRU Std':<12}")
    print("-" * 70)
    
    for seq_len in seq_lengths:
        lstm_r = results['lstm'][seq_len]
        gru_r = results['gru'][seq_len]
        print(f"{seq_len:<12} {lstm_r['mean']:<15.4f} {lstm_r['std']:<12.4f} "
              f"{gru_r['mean']:<15.4f} {gru_r['std']:<12.4f}")
    
    print("\nInterpretation:")
    print("- Higher ratio = better gradient flow to early timesteps")
    print("- Both architectures show similar gradient retention")
    print("- LSTM may have slight edge on very long sequences due to cell state")
    
    return results
```

## Memory Capacity: Deep Dive

### The Output Gate Advantage

LSTM's output gate provides a unique capability: **storing information without exposing it**.

```python
def demonstrate_output_gate_advantage():
    """
    Demonstrate LSTM's ability to hide information via output gate.
    
    Scenario: Store a piece of information early, retrieve it much later,
    without contaminating intermediate outputs.
    """
    print("Output Gate Advantage Demonstration")
    print("=" * 60)
    print("""
    Task: Remember a signal from step 0, output it at step 100,
          maintain different outputs for steps 1-99.
    
    LSTM Strategy:
    - Step 0: i_t=1, f_t=0 → store in cell
    - Steps 1-99: f_t=1, o_t=0 → preserve but hide
    - Step 100: o_t=1 → reveal stored information
    
    GRU Challenge:
    - No way to hide stored information
    - Must balance h_t between storage and output
    - Earlier stored info gets mixed with new computations
    
    This is why LSTM can sometimes handle complex memory patterns
    that require selective read/write access.
    """)


def memory_pattern_experiment():
    """
    Empirical test of memory patterns requiring selective access.
    """
    # Create a task where information must be stored then retrieved
    # LSTM should have advantage due to output gate
    
    class MemoryTask:
        """
        Task: Input a key at position 0, query at position T.
        Output the key value only at query position.
        """
        def __init__(self, seq_length=100, vocab_size=10):
            self.seq_length = seq_length
            self.vocab_size = vocab_size
        
        def generate_batch(self, batch_size):
            # Random keys (one-hot at position 0)
            keys = torch.randint(0, self.vocab_size, (batch_size,))
            
            # Input sequence: key at pos 0, zeros elsewhere, query signal at last
            x = torch.zeros(batch_size, self.seq_length, self.vocab_size + 1)
            x[:, 0, :self.vocab_size] = torch.nn.functional.one_hot(
                keys, self.vocab_size
            ).float()
            x[:, -1, -1] = 1.0  # Query signal
            
            # Target: key value at last position
            y = keys
            
            return x, y
    
    return MemoryTask()
```

### Information Bottleneck Analysis

```python
def information_bottleneck_comparison():
    """
    Analyze how each architecture handles information compression.
    """
    print("Information Bottleneck Analysis")
    print("=" * 60)
    print("""
    LSTM State Capacity:
    - Cell state c_t: Primary storage (unbounded in principle)
    - Hidden state h_t: Working memory (bounded by tanh)
    - Total: 2 × hidden_size values
    - Can store in cell without affecting output
    
    GRU State Capacity:
    - Hidden state h_t: All-purpose state
    - Total: 1 × hidden_size values
    - Everything stored affects output
    
    Implications:
    1. LSTM can maintain more independent memory slots
    2. GRU must compress all information into one vector
    3. For complex tasks with multiple concurrent memories, LSTM wins
    4. For simpler tasks, GRU's simplicity is sufficient
    """)
    
    # Empirical capacity test
    hidden_sizes = [64, 128, 256, 512]
    
    print("\nEffective Capacity (bits stored in state):")
    print(f"{'Hidden Size':<15} {'LSTM (est.)':<20} {'GRU (est.)':<20}")
    print("-" * 55)
    
    for h in hidden_sizes:
        # Rough estimate: ~1 bit per hidden unit for robust storage
        # LSTM has cell + hidden, GRU has only hidden
        lstm_capacity = 2 * h  # c_t and h_t
        gru_capacity = h       # only h_t
        print(f"{h:<15} {lstm_capacity:<20} {gru_capacity:<20}")
```

## Task-Specific Performance: Detailed Analysis

### Comprehensive Task Comparison

| Task Category | Best Choice | Reasoning |
|---------------|-------------|-----------|
| **Language Modeling** | LSTM ≈ GRU | Both effective; slight LSTM edge on perplexity |
| **Machine Translation** | LSTM | Long dependencies, complex alignments |
| **Speech Recognition** | LSTM | Continuous signals, precise timing |
| **Sentiment Analysis** | GRU ≈ LSTM | Simpler task, efficiency matters |
| **Time Series Forecasting** | GRU | Often shorter dependencies, speed |
| **Named Entity Recognition** | GRU | Local context usually sufficient |
| **Music Generation** | LSTM | Long-range structure, polyphony |
| **Video Captioning** | LSTM | Multi-modal, complex memory |
| **Dialogue Systems** | GRU | Faster inference, good enough quality |
| **Copy/Recall Tasks** | LSTM | Explicit memory requirements |

### Research Findings Summary

Key papers comparing LSTM and GRU:

1. **Chung et al. (2014)** - Original GRU paper
   - GRU comparable to LSTM on music and speech
   - Faster convergence in some cases

2. **Jozefowicz et al. (2015)** - Large-scale comparison
   - Tested 10,000+ architectures
   - No clear winner across all tasks
   - Task-dependent performance

3. **Greff et al. (2017)** - LSTM variant analysis
   - Forget gate and output activation most critical
   - Simplified variants often work well

```python
def summarize_research_findings():
    """
    Summary of key research findings on LSTM vs GRU.
    """
    findings = """
    Research Findings Summary
    ========================
    
    1. NO UNIVERSAL WINNER
       - Performance is highly task-dependent
       - Differences often within noise margin
       - Hyperparameter tuning often matters more
    
    2. LSTM ADVANTAGES
       - Slight edge on very long sequences (>500 tokens)
       - Better on tasks requiring complex memory access
       - More established, better understood
    
    3. GRU ADVANTAGES  
       - Faster training (15-25%)
       - Fewer hyperparameters
       - Less prone to overfitting on small data
       - Easier to scale
    
    4. PRACTICAL IMPLICATIONS
       - Start with GRU for prototyping (faster iteration)
       - Try LSTM if GRU plateaus
       - Difference rarely exceeds 2-3% on final metrics
       - Other choices (depth, regularization) often matter more
    """
    print(findings)
```

## Practical Decision Framework

### Decision Flowchart (Expanded)

```
                                START
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  Is this a prototype/MVP?   │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                   YES                          NO
                    │                           │
                    ▼                           ▼
              Use GRU              ┌─────────────────────────┐
          (faster iteration)      │  Sequence length > 500? │
                                  └───────────┬─────────────┘
                                              │
                                ┌─────────────┴─────────────┐
                                │                           │
                               YES                          NO
                                │                           │
                                ▼                           ▼
                    ┌───────────────────┐     ┌─────────────────────────┐
                    │   Lean LSTM       │     │  Dataset < 10k samples? │
                    │ (better long-term)│     └───────────┬─────────────┘
                    └───────────────────┘                 │
                                              ┌───────────┴───────────┐
                                              │                       │
                                             YES                      NO
                                              │                       │
                                              ▼                       ▼
                                        Use GRU           ┌─────────────────────┐
                                    (less overfitting)    │ Is inference speed  │
                                                          │     critical?       │
                                                          └─────────┬───────────┘
                                                                    │
                                                      ┌─────────────┴─────────────┐
                                                      │                           │
                                                     YES                          NO
                                                      │                           │
                                                      ▼                           ▼
                                                 Use GRU               Try both, pick based
                                               (faster)               on validation metrics
```

### Configuration Recommendations

```python
def get_recommended_config(task_type, seq_length, dataset_size, 
                           inference_speed_critical=False):
    """
    Get recommended architecture based on task characteristics.
    """
    recommendations = {
        'arch': None,
        'hidden_size': None,
        'num_layers': None,
        'dropout': None,
        'reasoning': []
    }
    
    # Architecture selection
    if seq_length > 500:
        recommendations['arch'] = 'LSTM'
        recommendations['reasoning'].append(
            f"Long sequences ({seq_length}) favor LSTM's cell state"
        )
    elif dataset_size < 10000:
        recommendations['arch'] = 'GRU'
        recommendations['reasoning'].append(
            f"Small dataset ({dataset_size}) - GRU less prone to overfitting"
        )
    elif inference_speed_critical:
        recommendations['arch'] = 'GRU'
        recommendations['reasoning'].append(
            "Speed critical - GRU is 15-25% faster"
        )
    else:
        recommendations['arch'] = 'GRU'
        recommendations['reasoning'].append(
            "Default to GRU, switch to LSTM if performance plateaus"
        )
    
    # Hidden size
    if dataset_size < 5000:
        recommendations['hidden_size'] = 128
    elif dataset_size < 50000:
        recommendations['hidden_size'] = 256
    else:
        recommendations['hidden_size'] = 512
    
    # Layers
    if seq_length > 200:
        recommendations['num_layers'] = 2
    else:
        recommendations['num_layers'] = 1
    
    # Dropout
    if dataset_size < 10000:
        recommendations['dropout'] = 0.5
    elif dataset_size < 100000:
        recommendations['dropout'] = 0.3
    else:
        recommendations['dropout'] = 0.1
    
    return recommendations


# Example usage
config = get_recommended_config(
    task_type='classification',
    seq_length=150,
    dataset_size=25000,
    inference_speed_critical=False
)
print("Recommended Configuration:")
for key, value in config.items():
    print(f"  {key}: {value}")
```

## Hybrid and Advanced Approaches

### Stacked Architectures

```python
class HybridStackedRNN(nn.Module):
    """
    Hybrid architecture: GRU for feature extraction, LSTM for memory.
    
    Rationale:
    - Lower layers extract local features (GRU sufficient, faster)
    - Upper layer handles long-range dependencies (LSTM advantage)
    """
    
    def __init__(self, input_size, hidden_size, num_gru_layers=2, dropout=0.3):
        super().__init__()
        
        # GRU layers for feature extraction
        self.gru = nn.GRU(
            input_size, hidden_size, num_gru_layers,
            batch_first=True, dropout=dropout if num_gru_layers > 1 else 0
        )
        
        # LSTM layer for complex memory
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, 1,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Feature extraction
        gru_out, _ = self.gru(x)
        gru_out = self.dropout(gru_out)
        
        # Memory processing
        lstm_out, (h_n, c_n) = self.lstm(gru_out)
        
        return lstm_out, h_n


class BidirectionalHybrid(nn.Module):
    """
    Bidirectional GRU for context, unidirectional LSTM for generation.
    
    Use case: Encoder-decoder architectures where encoder needs
    bidirectional context but decoder must be causal.
    """
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        
        # Bidirectional GRU encoder (context from both directions)
        self.encoder = nn.GRU(
            input_size, hidden_size, 
            batch_first=True, bidirectional=True
        )
        
        # Bridge layer to compress bidirectional to unidirectional
        self.bridge = nn.Linear(hidden_size * 2, hidden_size)
        
        # Unidirectional LSTM decoder (causal generation)
        self.decoder = nn.LSTM(
            hidden_size, hidden_size,
            batch_first=True
        )
    
    def encode(self, x):
        enc_out, h_n = self.encoder(x)
        # Combine bidirectional hidden states
        h_combined = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        h_bridge = torch.tanh(self.bridge(h_combined))
        return enc_out, h_bridge
```

### Minimal GRU Variants

```python
class MinimalGRU(nn.Module):
    """
    Minimal GRU variant with only update gate.
    
    Even simpler than standard GRU, sometimes works well
    for tasks with clear temporal dependencies.
    
    h_t = (1 - z_t) * h_{t-1} + z_t * tanh(W_h @ x_t)
    
    No reset gate - previous state always fully contributes
    to candidate computation (via initial state).
    """
    
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Update gate
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        
        # Candidate (no reset gate, simpler)
        self.W_h = nn.Linear(input_size, hidden_size)
    
    def forward(self, x, h_0=None):
        batch_size, seq_len, _ = x.shape
        
        if h_0 is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h = h_0
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            # Update gate
            z = torch.sigmoid(self.W_z(torch.cat([x_t, h], dim=-1)))
            
            # Candidate (simplified - no reset gate)
            h_tilde = torch.tanh(self.W_h(x_t))
            
            # Update
            h = (1 - z) * h + z * h_tilde
            
            outputs.append(h)
        
        return torch.stack(outputs, dim=1), h
```

## Summary

### Quick Reference Table

| Factor | LSTM | GRU | Winner |
|--------|------|-----|--------|
| Parameters | 4n² + 4nm | 3n² + 3nm | **GRU** (25% fewer) |
| Speed | Baseline | 15-25% faster | **GRU** |
| Memory capacity | 2 × hidden | 1 × hidden | **LSTM** |
| Long sequences (>500) | Better | Good | **LSTM** |
| Small datasets (<10k) | Overfits more | More robust | **GRU** |
| Interpretability | 3 gates, complex | 2 gates, simpler | **GRU** |
| Research maturity | 1997, extensive | 2014, growing | **LSTM** |
| Default recommendation | For benchmarks | For production | **GRU** |

### Final Recommendations

1. **Start with GRU** for most applications
   - Faster prototyping
   - Often equivalent performance
   - Easier to tune

2. **Switch to LSTM** when:
   - Working with very long sequences
   - GRU performance plateaus
   - Task requires complex memory patterns
   - Reproducing published baselines

3. **Consider both** for production:
   - Run controlled experiments
   - The 2-3% difference might matter at scale

4. **Don't overthink it**:
   - Architecture choice is often less important than data quality, regularization, and hyperparameter tuning
   - Both are well-optimized in modern frameworks
   - The "best" choice is task-dependent and often requires experimentation

```python
def final_wisdom():
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                    LSTM vs GRU: Final Wisdom                   ║
    ╠════════════════════════════════════════════════════════════════╣
    ║                                                                ║
    ║  "The difference between LSTM and GRU is often smaller        ║
    ║   than the difference between a good and bad learning rate."  ║
    ║                                                                ║
    ║  Start simple (GRU), measure carefully, iterate.              ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
```
