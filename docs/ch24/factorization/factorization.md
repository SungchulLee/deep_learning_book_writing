# Autoregressive Factorization

## Introduction

Autoregressive models represent one of the most fundamental approaches to generative modeling. The core idea is elegantly simple: model a joint probability distribution by decomposing it into a product of conditional distributions, where each variable is predicted based on all previous variables. This chapter establishes the mathematical foundations that underpin all autoregressive generative models, from classical time series models to modern large language models.

## The Chain Rule of Probability

### Mathematical Foundation

For any joint distribution over a sequence of random variables $\mathbf{x} = (x_1, x_2, \ldots, x_n)$, the chain rule of probability provides an exact factorization:

$$P(\mathbf{x}) = P(x_1, x_2, \ldots, x_n) = \prod_{i=1}^{n} P(x_i | x_1, x_2, \ldots, x_{i-1})$$

This is not an approximation—it is an identity that holds for any probability distribution. The first term $P(x_1)$ is unconditional, while each subsequent term conditions on all preceding variables.

**Example: Three Variables**

For a sequence $(x_1, x_2, x_3)$:

$$P(x_1, x_2, x_3) = P(x_1) \cdot P(x_2|x_1) \cdot P(x_3|x_1, x_2)$$

This decomposition is exact regardless of the underlying data distribution.

### Ordering and Its Implications

The chain rule requires choosing an ordering of variables. Different orderings yield mathematically equivalent factorizations but may have different computational properties:

$$P(x_1, x_2, x_3) = P(x_1) P(x_2|x_1) P(x_3|x_1, x_2)$$
$$P(x_1, x_2, x_3) = P(x_3) P(x_2|x_3) P(x_1|x_2, x_3)$$

Both are correct, but the choice of ordering affects:

1. **Natural structure alignment**: For sequences (text, audio, time series), left-to-right ordering matches temporal causality
2. **Computational efficiency**: Some orderings allow more efficient parallel computation
3. **Inductive bias**: The ordering implicitly encodes assumptions about dependencies

For images, common orderings include raster scan (left-to-right, top-to-bottom), which PixelCNN uses, or multi-scale approaches that process coarse-to-fine.

## The Autoregressive Property

### Definition

A model is **autoregressive** if it parameterizes each conditional distribution $P(x_i | x_{<i})$ with learnable parameters, where $x_{<i} = (x_1, \ldots, x_{i-1})$ denotes all variables preceding position $i$.

The key insight is that we model:

$$P_\theta(\mathbf{x}) = \prod_{i=1}^{n} P_\theta(x_i | x_{<i})$$

where $\theta$ represents the learned parameters. Each conditional distribution can be parameterized by a neural network that takes $x_{<i}$ as input.

### Why "Autoregressive"?

The term comes from time series analysis. In a classical AR(p) model:

$$x_t = \sum_{j=1}^{p} \phi_j x_{t-j} + \epsilon_t$$

The variable $x_t$ "regresses" on its own past values—hence "auto" (self) + "regressive." Neural autoregressive models generalize this to non-linear dependencies and arbitrary data types.

### Tractable Likelihood

A crucial advantage of autoregressive models is **tractable likelihood computation**. Given a data point $\mathbf{x}$, we can compute its exact log-likelihood:

$$\log P_\theta(\mathbf{x}) = \sum_{i=1}^{n} \log P_\theta(x_i | x_{<i})$$

This is a sum of $n$ terms, each of which is the output of a neural network. This enables:

1. **Exact density evaluation**: Unlike VAEs or GANs, we can compute the exact probability of any data point
2. **Maximum likelihood training**: Direct optimization of $\log P_\theta(\mathbf{x})$
3. **Model comparison**: Perplexity and bits-per-dimension are meaningful metrics

## Parameterizing Conditional Distributions

### Discrete Variables

For discrete variables with vocabulary size $V$ (e.g., characters, tokens, pixel intensities), the conditional distribution is typically a categorical:

$$P_\theta(x_i | x_{<i}) = \text{Categorical}(x_i; \pi_\theta(x_{<i}))$$

where $\pi_\theta(x_{<i}) \in \mathbb{R}^V$ is a probability vector produced by applying softmax to neural network outputs:

$$\pi_\theta(x_{<i}) = \text{softmax}(f_\theta(x_{<i}))$$

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiscreteARModel(nn.Module):
    """
    Autoregressive model for discrete sequences.
    
    Each position predicts a categorical distribution over vocabulary.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Embedding layer: convert discrete tokens to continuous vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Context encoder: process previous tokens
        self.encoder = nn.GRU(
            embedding_dim, 
            hidden_dim, 
            batch_first=True
        )
        
        # Output projection: hidden state -> vocabulary logits
        self.output_proj = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute logits for next-token prediction at each position.
        
        Args:
            x: Input sequence [batch_size, seq_len]
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Embed input tokens
        embedded = self.embedding(x)  # [batch, seq_len, embed_dim]
        
        # Encode context
        hidden_states, _ = self.encoder(embedded)  # [batch, seq_len, hidden]
        
        # Project to vocabulary
        logits = self.output_proj(hidden_states)  # [batch, seq_len, vocab]
        
        return logits
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.
        
        Uses teacher forcing: predict x[t] from x[0:t].
        """
        # Shift for next-token prediction
        inputs = x[:, :-1]   # All but last token
        targets = x[:, 1:]   # All but first token
        
        # Get predictions
        logits = self.forward(inputs)
        
        # Cross-entropy loss = negative log-likelihood
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1)
        )
        
        return loss
```

### Continuous Variables

For continuous data (e.g., audio waveforms, real-valued features), common choices include:

**Gaussian Output:**

$$P_\theta(x_i | x_{<i}) = \mathcal{N}(x_i; \mu_\theta(x_{<i}), \sigma^2_\theta(x_{<i}))$$

The network outputs mean and variance parameters.

**Mixture of Logistics:**

$$P_\theta(x_i | x_{<i}) = \sum_{k=1}^{K} \pi_k \cdot \text{Logistic}(x_i; \mu_k, s_k)$$

Used in WaveNet and PixelCNN++ for modeling bounded continuous values.

**Discretization:**

A practical approach is to discretize continuous values (e.g., 8-bit audio → 256 levels) and use categorical distributions.

```python
class ContinuousARModel(nn.Module):
    """
    Autoregressive model for continuous sequences with Gaussian output.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
        
        # Output mean and log-variance
        self.mean_proj = nn.Linear(hidden_dim, input_dim)
        self.logvar_proj = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor):
        """
        Compute Gaussian parameters for each position.
        
        Args:
            x: Input sequence [batch_size, seq_len, input_dim]
            
        Returns:
            mean: [batch_size, seq_len, input_dim]
            logvar: [batch_size, seq_len, input_dim]
        """
        hidden_states, _ = self.encoder(x)
        
        mean = self.mean_proj(hidden_states)
        logvar = self.logvar_proj(hidden_states)
        
        return mean, logvar
    
    def compute_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian negative log-likelihood.
        """
        inputs = x[:, :-1, :]
        targets = x[:, 1:, :]
        
        mean, logvar = self.forward(inputs)
        
        # Gaussian NLL: 0.5 * (log(2π) + logvar + (x-μ)²/var)
        nll = 0.5 * (
            logvar + 
            (targets - mean).pow(2) / logvar.exp() +
            torch.log(torch.tensor(2 * torch.pi))
        )
        
        return nll.mean()
```

## Training Autoregressive Models

### Maximum Likelihood Estimation

Training minimizes the negative log-likelihood:

$$\mathcal{L}(\theta) = -\mathbb{E}_{\mathbf{x} \sim p_{\text{data}}} \left[ \log P_\theta(\mathbf{x}) \right] = -\mathbb{E}_{\mathbf{x}} \left[ \sum_{i=1}^{n} \log P_\theta(x_i | x_{<i}) \right]$$

This decomposes into independent terms, allowing efficient minibatch training.

### Teacher Forcing

During training, we use **teacher forcing**: the model receives the true previous tokens $x_{<i}$ as input when predicting $x_i$, rather than its own predictions. This enables:

1. **Parallelization**: All positions can be computed simultaneously
2. **Stable gradients**: No gradient flow through discrete sampling
3. **Efficient training**: Single forward pass for entire sequence

```python
def train_step(model, optimizer, batch):
    """
    Single training step with teacher forcing.
    
    The model sees true previous tokens at each position.
    """
    optimizer.zero_grad()
    
    # Teacher forcing: use true inputs
    loss = model.compute_loss(batch)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()
```

### Exposure Bias

Teacher forcing creates a train-test mismatch called **exposure bias**: during training, the model sees true data, but during generation, it sees its own potentially erroneous predictions. This can cause error accumulation during generation.

Mitigation strategies include:

1. **Scheduled sampling**: Gradually replace true tokens with model predictions during training
2. **Beam search**: Consider multiple hypotheses during generation
3. **Sequence-level training**: Optimize metrics on generated sequences

## Sampling from Autoregressive Models

### Ancestral Sampling

The standard generation procedure samples from each conditional in sequence:

$$x_1 \sim P_\theta(x_1)$$
$$x_2 \sim P_\theta(x_2 | x_1)$$
$$\vdots$$
$$x_n \sim P_\theta(x_n | x_{<n})$$

This produces exact samples from the learned distribution.

```python
@torch.no_grad()
def sample(model, start_token, max_length, temperature=1.0):
    """
    Generate sequence via ancestral sampling.
    
    Args:
        model: Trained autoregressive model
        start_token: Initial token(s)
        max_length: Maximum sequence length
        temperature: Sampling temperature (higher = more random)
    
    Returns:
        Generated sequence
    """
    model.eval()
    
    # Initialize with start token
    generated = [start_token]
    hidden = None
    
    for _ in range(max_length - 1):
        # Get current input
        x = torch.tensor([[generated[-1]]])
        
        # Compute logits for next token
        embedded = model.embedding(x)
        output, hidden = model.encoder(embedded, hidden)
        logits = model.output_proj(output[:, -1, :])
        
        # Apply temperature
        logits = logits / temperature
        
        # Sample from distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        generated.append(next_token)
    
    return generated
```

### Temperature Scaling

Temperature $\tau$ controls the sharpness of the sampling distribution:

$$P_\tau(x_i | x_{<i}) \propto P(x_i | x_{<i})^{1/\tau}$$

- $\tau < 1$: Sharper distribution, more deterministic (favors high-probability tokens)
- $\tau = 1$: Original distribution
- $\tau > 1$: Flatter distribution, more random (explores low-probability tokens)

### Top-k and Top-p Sampling

To improve sample quality while maintaining diversity:

**Top-k Sampling:** Only consider the $k$ highest probability tokens:

```python
def top_k_sampling(logits, k):
    """Sample from top-k most likely tokens."""
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=-1)
    sampled_idx = torch.multinomial(probs, 1)
    return indices.gather(-1, sampled_idx)
```

**Top-p (Nucleus) Sampling:** Include tokens until cumulative probability exceeds $p$:

```python
def top_p_sampling(logits, p):
    """Sample from smallest set of tokens with cumulative probability >= p."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # Find cutoff
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # Filter and sample
    sorted_logits[sorted_indices_to_remove] = float('-inf')
    probs = F.softmax(sorted_logits, dim=-1)
    sampled_idx = torch.multinomial(probs, 1)
    
    return sorted_indices.gather(-1, sampled_idx)
```

## Computational Considerations

### Sequential Generation Bottleneck

The primary limitation of autoregressive models is **sequential generation**. Generating a sequence of length $n$ requires $n$ forward passes through the network, as each $x_i$ depends on all previous outputs.

| Aspect | Training | Generation |
|--------|----------|------------|
| **Parallelization** | Full (all positions simultaneously) | None (strictly sequential) |
| **Complexity** | $O(1)$ forward passes | $O(n)$ forward passes |
| **Practical speed** | Fast | Slow for long sequences |

### Efficient Architectures

Several architectural innovations address the generation bottleneck:

1. **Caching**: Store intermediate computations (e.g., key-value cache in Transformers)
2. **Parallel prediction**: Predict multiple tokens simultaneously (speculative decoding)
3. **Non-autoregressive models**: Trade likelihood tractability for parallel generation

## Comparison with Other Generative Models

| Property | Autoregressive | VAE | GAN | Flow | Diffusion |
|----------|---------------|-----|-----|------|-----------|
| **Exact likelihood** | ✓ | ✗ (ELBO) | ✗ | ✓ | ✗ |
| **Fast training** | ✓ | ✓ | ✗ | ✓ | ✗ |
| **Fast generation** | ✗ | ✓ | ✓ | ✓ | ✗ |
| **Sample quality** | High | Medium | High | Medium | Very High |
| **Mode coverage** | High | High | Low | High | High |

Autoregressive models excel when:
- Exact likelihood is important (density estimation, compression)
- Data has natural sequential structure (text, audio, time series)
- Training efficiency is prioritized over generation speed

## Applications in Quantitative Finance

### Time Series Forecasting

Financial time series naturally fit the autoregressive paradigm:

$$P(r_t | r_{<t}) = P(r_t | r_{t-1}, r_{t-2}, \ldots)$$

where $r_t$ represents returns, prices, or other financial quantities.

```python
class FinancialARModel(nn.Module):
    """
    Autoregressive model for financial time series.
    
    Outputs parameters of a Student-t distribution to capture
    heavy tails common in financial returns.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        
        # Student-t parameters: location, scale, degrees of freedom
        self.loc_proj = nn.Linear(hidden_dim, 1)
        self.scale_proj = nn.Linear(hidden_dim, 1)
        self.df_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        hidden, _ = self.lstm(x)
        
        loc = self.loc_proj(hidden)
        scale = F.softplus(self.scale_proj(hidden)) + 1e-6
        df = F.softplus(self.df_proj(hidden)) + 2  # df > 2 for finite variance
        
        return loc, scale, df
```

### Scenario Generation

Autoregressive models can generate realistic market scenarios by sampling from the learned joint distribution of multiple assets or factors.

### Order Flow Modeling

High-frequency trading applications model the arrival and characteristics of orders autoregressively, capturing the sequential nature of market microstructure.

## Summary

Autoregressive models provide a principled framework for generative modeling based on the chain rule of probability. Key properties include:

1. **Exact likelihood**: Enables direct MLE training and meaningful density evaluation
2. **Flexibility**: Any conditional distribution can be parameterized with neural networks
3. **Natural fit for sequences**: Text, audio, and time series have inherent order
4. **Sequential generation**: The main computational limitation

The following sections explore specific autoregressive architectures: PixelCNN for images, WaveNet for audio, and Transformers for general sequences.

## References

1. Bengio, Y., & Bengio, S. (2000). Modeling High-Dimensional Discrete Data with Multi-Layer Neural Networks. *NeurIPS*.
2. Larochelle, H., & Murray, I. (2011). The Neural Autoregressive Distribution Estimator. *AISTATS*.
3. Uria, B., Côté, M. A., Gregor, K., Murray, I., & Larochelle, H. (2016). Neural Autoregressive Distribution Estimation. *JMLR*.
4. Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). Normalizing Flows for Probabilistic Modeling and Inference. *JMLR*.

---

## Exercises

1. **Chain Rule Verification**: For a simple 3-variable distribution, verify that different orderings give the same joint probability.

2. **Temperature Exploration**: Implement temperature sampling and visualize how generated sequences change with temperature.

3. **Exposure Bias**: Design an experiment to measure the effect of exposure bias on generation quality.

4. **Financial Application**: Build an autoregressive model for daily stock returns and evaluate its density estimation performance.
