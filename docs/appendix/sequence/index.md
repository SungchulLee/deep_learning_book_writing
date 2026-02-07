# A5 Sequence Models

## Overview

This appendix provides complete PyTorch implementations of recurrent sequence models, from vanilla RNNs through attention-augmented encoder–decoder architectures. These models process sequential data by maintaining hidden states that capture temporal dependencies, making them foundational for time series analysis in quantitative finance. While transformers have largely superseded RNNs in NLP, recurrent architectures remain relevant for streaming financial data, low-latency inference, and problems where the sequential inductive bias is beneficial.

## Architectures

| Model | Year | Key Innovation |
|-------|------|----------------|
| [Vanilla RNN](rnn.py) | 1986 | Shared weights across time steps, hidden state recurrence |
| [LSTM](lstm.py) | 1997 | Gated memory cell: forget, input, output gates solve vanishing gradients |
| [GRU](gru.py) | 2014 | Simplified gating with reset and update gates, fewer parameters than LSTM |
| [Bidirectional](bidirectional.py) | 1997 | Forward and backward passes capture past and future context |
| [Seq2Seq](seq2seq.py) | 2014 | Encoder–decoder for variable-length input/output sequence mapping |
| [Attention Seq2Seq](attention_seq2seq.py) | 2015 | Bahdanau/Luong attention over encoder states eliminates information bottleneck |

## Key Concepts

### The Vanishing Gradient Problem

Vanilla RNNs struggle with long-range dependencies because gradients either vanish or explode during backpropagation through time (BPTT). LSTM and GRU architectures solve this with gating mechanisms that control information flow:

$$h_t = f(h_{t-1}, x_t; \theta)$$

- **LSTM**: Separate cell state $c_t$ with additive updates preserves gradient flow
- **GRU**: Merges cell and hidden state, uses update gate to interpolate between old and new states

### Sequence-to-Sequence Learning

The encoder–decoder framework maps variable-length input sequences to variable-length outputs:

1. **Encoder**: Processes input sequence $x_1, \ldots, x_T$ into a context representation
2. **Decoder**: Generates output sequence $y_1, \ldots, y_{T'}$ conditioned on the context
3. **Attention**: Allows the decoder to focus on different encoder positions at each time step, replacing the fixed-length bottleneck vector

### Teacher Forcing vs. Scheduled Sampling

- **Teacher forcing**: Feed ground-truth tokens as decoder input during training (faster convergence, exposure bias)
- **Scheduled sampling**: Gradually transition from teacher forcing to model predictions during training

## Quantitative Finance Applications

- **Time series forecasting**: LSTM/GRU for price, volume, and volatility prediction with streaming data
- **Order book modeling**: Sequence models for limit order book event streams
- **Sequence-to-sequence**: Map economic indicator sequences to forecast horizons
- **Anomaly detection**: Reconstruction-based anomaly detection using encoder–decoder RNNs
- **Text processing**: Extract financial signals from news feeds and earnings call transcripts

## Prerequisites

- [Ch4: Training Deep Networks](../../ch04/index.md) — gradient clipping, learning rate scheduling
- [A6: Transformer Architectures](../transformers/index.md) — attention mechanisms that evolved from Seq2Seq attention
- [A10: Utility Modules — Attention Mechanisms](../utils/attention.py) — additive and dot-product attention
