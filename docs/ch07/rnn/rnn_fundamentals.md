# RNN Fundamentals

## From Sequences to Recurrence

Sequential data—information where the order of elements carries meaning—pervades real-world applications from natural language to financial time series. Unlike tabular data where rows are independent observations, sequential data exhibits temporal or positional dependencies that fundamentally alter how we must process and model information.

Three defining characteristics distinguish sequential data from other data types. **Ordering matters**: rearranging elements changes the meaning entirely, so "The cat sat on the mat" conveys different information than "mat the on sat cat The" despite containing identical words. **Temporal dependencies** link current observations to previous ones: today's stock price relates to yesterday's, and the next word in a sentence depends on all preceding words. **Variable length** means that unlike images with fixed dimensions, sequences naturally range from single tokens to thousands of timesteps.

### The Autoregressive Factorization

The probabilistic structure underlying sequential modeling decomposes a joint distribution via the chain rule:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

Each element's probability depends on all preceding elements. Recurrent Neural Networks are designed precisely to model this conditional structure by maintaining a compressed summary of the past at each timestep.

### Why Feedforward Networks Fail

A standard feedforward network computes each output independently:

$$y = f(Wx + b)$$

With no mechanism to incorporate information from previous inputs, it cannot capture sequential dependencies. What we need is a formulation that conditions on accumulated history:

$$y_t = f(x_t, h_{t-1})$$

where $h_{t-1}$ encodes all relevant information from timesteps $1$ through $t-1$. This is the hidden state—the memory of the network.

## The Recurrence Principle

Recurrent Neural Networks solve the sequential modeling problem through an elegantly simple idea: maintain a hidden state that accumulates information across timesteps. At each step, the network combines new input with its memory of past inputs to produce both an output and an updated memory.

### Core Equations

The fundamental RNN computation at timestep $t$:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

where:

- $x_t \in \mathbb{R}^{d}$: input at timestep $t$ (dimension $d$)
- $h_t \in \mathbb{R}^{H}$: hidden state at timestep $t$ (dimension $H$)
- $y_t \in \mathbb{R}^{o}$: output at timestep $t$ (dimension $o$)
- $W_{xh} \in \mathbb{R}^{H \times d}$: input-to-hidden weights
- $W_{hh} \in \mathbb{R}^{H \times H}$: hidden-to-hidden weights
- $W_{hy} \in \mathbb{R}^{o \times H}$: hidden-to-output weights
- $b_h, b_y$: bias vectors

### Unrolling Through Time

The recurrence creates a computational graph when "unrolled" across timesteps:

```
    x₁       x₂       x₃       x₄
    ↓        ↓        ↓        ↓
h₀ → RNN → h₁ → RNN → h₂ → RNN → h₃ → RNN → h₄
           ↓        ↓        ↓        ↓
           y₁       y₂       y₃       y₄
```

Critically, the **same weights** $(W_{xh}, W_{hh}, W_{hy})$ are shared across all timesteps—this parameter sharing enables processing sequences of arbitrary length.

### Information Flow

At each timestep, information flows through three paths:

**Direct input path** ($W_{xh} x_t$): new information from the current input directly influences the hidden state.

**Recurrent path** ($W_{hh} h_{t-1}$): historical information carried forward through the hidden state.

**Nonlinear combination** ($\tanh(\cdot)$): the activation function squashes values to $[-1, 1]$ and creates complex interactions between current and past information.

## Representing Sequences as Tensors

PyTorch represents sequential data using consistent tensor conventions:

```python
import torch

# Single sequence: 5 timesteps, each with 3 features
single_sequence = torch.randn(5, 3)    # (seq_len, features)

# Batched sequences (standard RNN input with batch_first=True)
batch_size, seq_len, num_features = 32, 10, 5
batched = torch.randn(batch_size, seq_len, num_features)  # (batch, seq_len, features)
```

The convention `(batch, seq_len, features)` with `batch_first=True` provides the most intuitive interpretation.

### The Sliding Window Approach

Creating training examples from sequential data typically uses sliding windows:

```python
def create_sequences(data, window_size):
    """Create input-target pairs using sliding windows."""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return torch.stack(X), torch.stack(y)

# Example: predict next value from previous 3
data = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
X, y = create_sequences(data, window_size=3)
# X[0] = [1, 2, 3] → y[0] = 4
# X[1] = [2, 3, 4] → y[1] = 5
```

## Sequence Modeling Tasks

Different applications require different input-output relationships:

**Many-to-One**: a full sequence maps to a single prediction. Sentiment classification takes a review $[w_1, w_2, \ldots, w_n]$ and outputs a label (positive/negative).

**One-to-Many**: a single input produces a sequence. Image captioning takes an image feature vector and generates $[w_1, w_2, \ldots, w_n]$.

**Many-to-Many (Synchronized)**: input and output sequences have the same length. Part-of-speech tagging maps $[\text{word}_1, \text{word}_2, \text{word}_3]$ to $[\text{noun}, \text{verb}, \text{noun}]$.

**Many-to-Many (Seq2Seq)**: input and output sequences may differ in length. Machine translation maps English words to French words through an encoder-decoder architecture.

## Computational Profile

For a sequence of length $T$ with hidden size $H$:

| Metric | Complexity |
|--------|------------|
| Time (forward) | $O(T \cdot H^2)$ |
| Space (all hidden states) | $O(T \cdot H)$ |
| Parameters | $O(H^2 + d \cdot H + o \cdot H)$ |

The sequential nature of the recurrence prevents parallelization across timesteps—a key limitation addressed by Transformer architectures. However, RNNs remain parameter-efficient: the same weights process every timestep regardless of sequence length.

## Summary

RNNs introduce recurrence to neural networks, enabling sequential modeling through hidden states that compress historical information into fixed-size vectors, parameter sharing that reuses the same weights at every timestep, and unrolling that creates a computational graph amenable to backpropagation. The architecture elegantly addresses sequential dependencies but faces challenges with long sequences due to vanishing gradients—motivating the gated architectures (LSTM, GRU) covered in subsequent sections.
