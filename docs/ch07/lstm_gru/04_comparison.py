"""
Comparison: RNN vs LSTM vs GRU
===============================

This module compares vanilla RNN, LSTM, and GRU on:
1. Architecture complexity
2. Training performance
3. Vanishing gradient problem
4. When to use which
"""

import numpy as np
import matplotlib.pyplot as plt
import time


# =============================================================================
# Architecture Comparison
# =============================================================================

def compare_architectures():
    """Compare architectural differences."""
    print("=" * 70)
    print("Architecture Comparison: RNN vs LSTM vs GRU")
    print("=" * 70)
    
    comparison_table = """
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ Feature         │ Vanilla RNN  │    LSTM      │     GRU      │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Gates           │      0       │      3       │      2       │
│                 │              │ (f, i, o)    │   (r, z)     │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ States          │      1       │      2       │      1       │
│                 │   (hidden)   │ (h + cell)   │  (hidden)    │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Parameters*     │   4(n²+nm+n) │ 4×[4(n²+nm+n)]│3×[4(n²+nm+n)]│
│                 │              │   (4x RNN)   │   (3x RNN)   │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Complexity      │    Simple    │   Complex    │   Medium     │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Training Speed  │    Fast      │    Slow      │   Medium     │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Memory Usage    │     Low      │     High     │   Medium     │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Long-term Deps  │     Poor     │   Excellent  │    Good      │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ Gradient Flow   │     Poor     │     Good     │    Good      │
└─────────────────┴──────────────┴──────────────┴──────────────┘

* n = hidden_size, m = input_size
"""
    print(comparison_table)
    
    # Calculate exact parameters for a concrete example
    input_size = 100
    hidden_size = 200
    
    # RNN: W_hh (n×n) + W_xh (m×n) + b_h (n)
    rnn_params = hidden_size * hidden_size + input_size * hidden_size + hidden_size
    
    # LSTM: 4 gates × (W + b)
    lstm_params = 4 * (hidden_size * (hidden_size + input_size) + hidden_size)
    
    # GRU: 3 gates × (W + b)
    gru_params = 3 * (hidden_size * (hidden_size + input_size) + hidden_size)
    
    print(f"\nConcrete Example (input={input_size}, hidden={hidden_size}):")
    print(f"  RNN:  {rnn_params:,} parameters")
    print(f"  LSTM: {lstm_params:,} parameters ({lstm_params/rnn_params:.1f}× RNN)")
    print(f"  GRU:  {gru_params:,} parameters ({gru_params/rnn_params:.1f}× RNN)")


# =============================================================================
# Vanishing Gradient Demonstration
# =============================================================================

def demonstrate_vanishing_gradients():
    """Demonstrate the vanishing gradient problem in RNNs."""
    print("\n" + "=" * 70)
    print("Vanishing Gradient Problem")
    print("=" * 70)
    
    print("\nThe Problem:")
    print("In vanilla RNN: h_t = tanh(W_hh @ h_{t-1} + W_xh @ x_t)")
    print("Gradient flows through: ∂h_t/∂h_0 = ∏(t, k=1) ∂h_k/∂h_{k-1}")
    print("                                   = ∏(t, k=1) W_hh · diag(tanh')")
    
    # Simulate gradient magnitudes
    sequence_lengths = [5, 10, 20, 50, 100]
    
    # Vanilla RNN gradients (exponential decay)
    rnn_gradients = {}
    for seq_len in sequence_lengths:
        # Simulate with typical weight magnitude
        gradients = [0.9 ** t for t in range(seq_len)]
        rnn_gradients[seq_len] = gradients
    
    # LSTM/GRU gradients (maintained better)
    lstm_gradients = {}
    for seq_len in sequence_lengths:
        # LSTM maintains gradients better due to additive cell state
        gradients = [0.99 ** t for t in range(seq_len)]
        lstm_gradients[seq_len] = gradients
    
    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, seq_len in enumerate(sequence_lengths):
        ax = axes[idx]
        
        rnn_grad = rnn_gradients[seq_len]
        lstm_grad = lstm_gradients[seq_len]
        
        ax.semilogy(range(seq_len), rnn_grad, 'r-', linewidth=2, label='Vanilla RNN')
        ax.semilogy(range(seq_len), lstm_grad, 'b-', linewidth=2, label='LSTM/GRU')
        
        ax.set_title(f'Sequence Length = {seq_len}')
        ax.set_xlabel('Time Steps Back')
        ax.set_ylabel('Gradient Magnitude (log scale)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig('/home/claude/lstm_gru_module/vanishing_gradients.png', dpi=150, bbox_inches='tight')
    print("\n✓ Vanishing gradient plot saved as 'vanishing_gradients.png'")
    plt.close()
    
    print("\nKey Observations:")
    print("1. RNN gradients decay exponentially with sequence length")
    print("2. LSTM/GRU maintain gradient flow much better")
    print("3. This is why LSTM/GRU can learn long-term dependencies")


# =============================================================================
# Performance Comparison
# =============================================================================

class SimpleRNN:
    """Minimal RNN for comparison."""
    def __init__(self, input_size, hidden_size):
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        
    def forward(self, inputs):
        h = np.zeros((self.W_hh.shape[0], 1))
        for x in inputs:
            h = np.tanh(self.W_hh @ h + self.W_xh @ x + self.b_h)
        return h


class SimpleLSTM:
    """Minimal LSTM for comparison."""
    def __init__(self, input_size, hidden_size):
        n, m = hidden_size, input_size
        self.weights = [np.random.randn(n, n+m) * 0.01 for _ in range(4)]
        self.biases = [np.zeros((n, 1)) for _ in range(4)]
        self.hidden_size = n
        
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))
        
        for x in inputs:
            combined = np.vstack((h, x))
            f = self._sigmoid(self.weights[0] @ combined + self.biases[0])
            i = self._sigmoid(self.weights[1] @ combined + self.biases[1])
            c_tilde = np.tanh(self.weights[2] @ combined + self.biases[2])
            c = f * c + i * c_tilde
            o = self._sigmoid(self.weights[3] @ combined + self.biases[3])
            h = o * np.tanh(c)
        return h
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class SimpleGRU:
    """Minimal GRU for comparison."""
    def __init__(self, input_size, hidden_size):
        n, m = hidden_size, input_size
        self.weights = [np.random.randn(n, n+m) * 0.01 for _ in range(3)]
        self.biases = [np.zeros((n, 1)) for _ in range(3)]
        self.hidden_size = n
        
    def forward(self, inputs):
        h = np.zeros((self.hidden_size, 1))
        
        for x in inputs:
            combined = np.vstack((h, x))
            r = self._sigmoid(self.weights[0] @ combined + self.biases[0])
            z = self._sigmoid(self.weights[1] @ combined + self.biases[1])
            combined_reset = np.vstack((r * h, x))
            h_tilde = np.tanh(self.weights[2] @ combined_reset + self.biases[2])
            h = (1 - z) * h + z * h_tilde
        return h
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def performance_comparison():
    """Compare training speed and memory."""
    print("\n" + "=" * 70)
    print("Performance Comparison")
    print("=" * 70)
    
    input_size = 50
    hidden_size = 100
    sequence_length = 50
    num_iterations = 100
    
    # Create models
    rnn = SimpleRNN(input_size, hidden_size)
    lstm = SimpleLSTM(input_size, hidden_size)
    gru = SimpleGRU(input_size, hidden_size)
    
    # Generate test data
    test_inputs = [np.random.randn(input_size, 1) for _ in range(sequence_length)]
    
    # Benchmark RNN
    start = time.time()
    for _ in range(num_iterations):
        _ = rnn.forward(test_inputs)
    rnn_time = time.time() - start
    
    # Benchmark LSTM
    start = time.time()
    for _ in range(num_iterations):
        _ = lstm.forward(test_inputs)
    lstm_time = time.time() - start
    
    # Benchmark GRU
    start = time.time()
    for _ in range(num_iterations):
        _ = gru.forward(test_inputs)
    gru_time = time.time() - start
    
    print(f"\nBenchmark Results ({num_iterations} iterations):")
    print(f"  RNN:  {rnn_time:.4f}s (baseline)")
    print(f"  LSTM: {lstm_time:.4f}s ({lstm_time/rnn_time:.2f}× slower)")
    print(f"  GRU:  {gru_time:.4f}s ({gru_time/rnn_time:.2f}× slower)")
    
    # Visualize
    models = ['RNN', 'LSTM', 'GRU']
    times = [rnn_time, lstm_time, gru_time]
    colors = ['green', 'blue', 'orange']
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, times, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{time_val:.4f}s',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylabel('Execution Time (seconds)', fontsize=12)
    plt.title(f'Forward Pass Speed Comparison\n({num_iterations} iterations, seq_len={sequence_length})',
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('/home/claude/lstm_gru_module/performance.png', dpi=150, bbox_inches='tight')
    print("\n✓ Performance plot saved as 'performance.png'")
    plt.close()


# =============================================================================
# Decision Guide
# =============================================================================

def decision_guide():
    """Guide for choosing between RNN, LSTM, and GRU."""
    print("\n" + "=" * 70)
    print("Decision Guide: When to Use Which?")
    print("=" * 70)
    
    guide = """
Use Vanilla RNN when:
  ✓ Very short sequences (< 10 steps)
  ✓ Simple patterns, no long-term dependencies
  ✓ Computational resources are extremely limited
  ✓ Baseline comparison needed
  ✗ Do NOT use for: language modeling, long sequences

Use LSTM when:
  ✓ Long sequences with complex dependencies
  ✓ Need maximum modeling capacity
  ✓ Tasks requiring precise long-term memory
  ✓ You have sufficient training data
  ✓ Computational resources available
  ✓ Examples: machine translation, speech recognition, video analysis

Use GRU when:
  ✓ Long sequences (but not as long as LSTM needs)
  ✓ Limited training data (fewer parameters = less overfitting)
  ✓ Faster training is priority
  ✓ Comparable performance to LSTM observed
  ✓ Examples: text generation, time series, music generation

LSTM vs GRU Trade-offs:
  
  Choose LSTM if:
    - You have large datasets
    - Maximum accuracy is critical
    - Training time is not a concern
    
  Choose GRU if:
    - You have limited data
    - Faster training is important
    - Slightly lower accuracy is acceptable
    - You want simpler model interpretation

General Guidelines:
  1. Start with GRU (good balance)
  2. Try LSTM if GRU plateaus early
  3. Consider bidirectional for tasks with future context
  4. Use attention mechanisms for very long sequences
  5. Monitor for overfitting with LSTM on small datasets

Performance Tips:
  • Batch your sequences for efficiency
  • Use gradient clipping (threshold ~5.0)
  • Try layer normalization
  • Consider residual connections for deep models
  • Use learning rate scheduling
"""
    print(guide)


# =============================================================================
# Practical Recommendations
# =============================================================================

def practical_recommendations():
    """Practical tips based on task type."""
    print("\n" + "=" * 70)
    print("Practical Recommendations by Task")
    print("=" * 70)
    
    recommendations = """
Task: Text Generation
  Recommended: GRU (faster, good results)
  Hidden Size: 256-512
  Layers: 2-3
  Notes: Use dropout between layers

Task: Machine Translation
  Recommended: LSTM with attention
  Hidden Size: 512-1024
  Layers: 2-4
  Notes: Bidirectional encoder, unidirectional decoder

Task: Sentiment Analysis
  Recommended: GRU or LSTM (both work)
  Hidden Size: 128-256
  Layers: 1-2
  Notes: Often bidirectional helps

Task: Speech Recognition
  Recommended: LSTM (better for audio)
  Hidden Size: 256-512
  Layers: 3-5
  Notes: Deep bidirectional LSTM works best

Task: Time Series Forecasting
  Recommended: GRU (faster training)
  Hidden Size: 32-128
  Layers: 1-2
  Notes: Simple architectures often sufficient

Task: Video Analysis
  Recommended: LSTM (complex temporal patterns)
  Hidden Size: 512-1024
  Layers: 2-3
  Notes: Combine with CNN features

Hyperparameter Starting Points:
  • Learning Rate: 1e-3 to 1e-4
  • Batch Size: 32-128
  • Gradient Clipping: 5.0
  • Dropout: 0.2-0.5
  • Optimizer: Adam or RMSprop
"""
    print(recommendations)


def main():
    """Run all comparisons."""
    print("\n" + "=" * 70)
    print("RNN vs LSTM vs GRU: Comprehensive Comparison")
    print("=" * 70)
    
    compare_architectures()
    demonstrate_vanishing_gradients()
    performance_comparison()
    decision_guide()
    practical_recommendations()
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - vanishing_gradients.png")
    print("  - performance.png")


if __name__ == "__main__":
    main()
