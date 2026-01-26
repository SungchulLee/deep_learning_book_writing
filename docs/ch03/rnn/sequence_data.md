# Understanding Sequence Data

## Introduction to Sequential Information

Sequential data represents information where the order of elements carries meaning. Unlike tabular data where rows are independent observations, sequential data exhibits temporal or positional dependencies that fundamentally alter how we must process and model the information.

## What Makes Data Sequential?

Three defining characteristics distinguish sequential data from other data types:

**Ordering Matters**: Rearranging elements changes the meaning entirely. The sentence "The cat sat on the mat" conveys different information than "mat the on sat cat The" despite containing identical words.

**Temporal Dependencies**: Current observations depend on previous observations. Today's stock price relates to yesterday's price; the next word in a sentence relates to words that came before.

**Variable Length**: Unlike images with fixed dimensions, sequences naturally vary in length. Sentences range from single words to thousands; time series span days to decades.

## Types of Sequential Data

### Natural Language

Text represents perhaps the most ubiquitous sequential data. At the character level, the sequence "HELLO" differs completely from "OLLEH". At the word level, grammar and semantics create complex dependencies spanning entire paragraphs.

$$\text{Sentence} = (w_1, w_2, \ldots, w_T) \quad \text{where each } w_t \text{ depends on } w_{1:t-1}$$

### Time Series

Financial markets, weather patterns, sensor readings, and physiological signals all generate time series data. The temporal structure captures trends, seasonality, and autocorrelation patterns essential for forecasting.

$$x_t = f(x_{t-1}, x_{t-2}, \ldots, x_{t-k}) + \epsilon_t$$

### Audio and Speech

Sound waves represent continuous signals sampled at discrete intervals (typically 16kHz or 44.1kHz). Speech recognition must model phoneme sequences, word boundaries, and prosodic patterns.

### Video

Video extends images into the temporal dimension, creating sequences of frames where motion and change carry semantic meaning. Actions unfold over time, requiring models that understand both spatial and temporal patterns.

## Representing Sequences as Tensors

PyTorch represents sequential data using consistent tensor conventions:

### Single Sequence
```python
import torch

# A single sequence of 5 timesteps
single_sequence = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Shape: {single_sequence.shape}")  # torch.Size([5])
```

### Sequence with Features
When each timestep contains multiple features (e.g., position coordinates, multiple sensor readings):

```python
# 5 timesteps, each with 3 features (x, y, z coordinates)
sequence_with_features = torch.randn(5, 3)
print(f"Shape: {sequence_with_features.shape}")  # torch.Size([5, 3])
```

### Batched Sequences (Standard RNN Input)
Neural networks process data in batches for efficiency:

```python
# Batch of 32 sequences, each 10 timesteps, 5 features per step
batch_size = 32
sequence_length = 10
num_features = 5

batched_sequences = torch.randn(batch_size, sequence_length, num_features)
print(f"Shape: {batched_sequences.shape}")  # torch.Size([32, 10, 5])
```

The convention `(batch, seq_len, features)` with `batch_first=True` provides the most intuitive interpretation.

## The Sliding Window Approach

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

# X[0] = [1, 2, 3] -> y[0] = 4
# X[1] = [2, 3, 4] -> y[1] = 5
# X[2] = [3, 4, 5] -> y[2] = 6
```

## Why Standard Neural Networks Fail

Consider predicting the next word in "The cat sat on the ___". A feedforward network receiving one word at a time cannot access previous context. Even receiving all words simultaneously treats them as an unordered set, losing positional information.

**Feedforward Limitation**:
$$y = f(W \cdot x + b)$$

Each input $x$ produces output $y$ independently, with no mechanism to incorporate information from previous inputs.

**What We Need**:
$$y_t = f(x_t, h_{t-1})$$

Where $h_{t-1}$ captures accumulated information from all previous timesteps—a *hidden state* that serves as memory.

## Mathematical Framework

For a sequence $(x_1, x_2, \ldots, x_T)$, we seek a model that captures:

**Joint Probability (Autoregressive Factorization)**:
$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^{T} P(x_t | x_1, \ldots, x_{t-1})$$

This factorization expresses that each element's probability depends on all preceding elements—precisely the structure RNNs are designed to model.

## Sequence Modeling Tasks

Different applications require different input-output relationships:

### Many-to-One
Input: Full sequence → Output: Single prediction

*Example*: Sentiment classification of a review

```
[word₁, word₂, ..., wordₙ] → [positive/negative]
```

### One-to-Many
Input: Single item → Output: Sequence

*Example*: Image captioning

```
[image] → [word₁, word₂, ..., wordₙ]
```

### Many-to-Many (Synchronized)
Input: Sequence → Output: Sequence of same length

*Example*: Part-of-speech tagging

```
[word₁, word₂, word₃] → [noun, verb, noun]
```

### Many-to-Many (Sequence-to-Sequence)
Input: Sequence → Output: Sequence (potentially different length)

*Example*: Machine translation

```
[English words] → [French words]
```

## Summary

Sequential data pervades real-world applications from language to finance to biology. Understanding the fundamental structure—ordered elements with temporal dependencies—motivates the architectural innovations we'll explore with RNNs, LSTMs, and beyond.

Key takeaways:

- Order matters: rearrangement changes meaning
- Elements depend on predecessors
- Tensor shape: `(batch, seq_len, features)`
- Sliding windows create training examples
- Standard networks lack memory for sequential patterns

The next section introduces how Recurrent Neural Networks address these challenges through their hidden state mechanism.
