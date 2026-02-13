# Module 51: Recurrent Neural Networks - LSTM & GRU

## Overview
This module covers Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks, which are advanced RNN architectures designed to handle long-term dependencies in sequential data.

## Files Included

### 1. `01_lstm_theory.py`
- Mathematical formulation of LSTM
- Explanation of gates (forget, input, output)
- Cell state mechanics
- Implementation from scratch using NumPy

### 2. `02_gru_theory.py`
- Mathematical formulation of GRU
- Explanation of gates (reset, update)
- Simplified architecture compared to LSTM
- Implementation from scratch using NumPy

### 3. `03_pytorch_lstm_gru.py`
- Using PyTorch's built-in LSTM and GRU layers
- Practical sequence prediction examples
- Text generation task
- Time series forecasting

### 4. `04_comparison.py`
- Side-by-side comparison of RNN, LSTM, and GRU
- Performance benchmarks
- When to use which architecture
- Vanishing gradient demonstration

### 5. `05_advanced_examples.py`
- Bidirectional LSTM/GRU
- Stacked LSTM/GRU layers
- Attention mechanisms
- Real-world applications

## Quick Start

```bash
# Install requirements
pip install numpy torch matplotlib pandas scikit-learn

# Run examples
python 01_lstm_theory.py
python 02_gru_theory.py
python 03_pytorch_lstm_gru.py
python 04_comparison.py
python 05_advanced_examples.py
```

## Key Concepts

### LSTM (Long Short-Term Memory)
- Solves vanishing gradient problem
- Three gates: forget, input, output
- Cell state + hidden state
- More parameters, more expressive

### GRU (Gated Recurrent Unit)
- Simplified LSTM variant
- Two gates: reset, update
- Only hidden state (no separate cell state)
- Fewer parameters, faster training

## Learning Path
1. Start with theory files (01, 02) to understand the math
2. Explore PyTorch implementations (03)
3. Compare architectures (04)
4. Study advanced techniques (05)

## References
- Hochreiter & Schmidhuber (1997) - LSTM paper
- Cho et al. (2014) - GRU paper
- Understanding LSTM Networks (colah's blog)
