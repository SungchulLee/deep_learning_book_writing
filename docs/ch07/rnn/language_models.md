# Language Models and Sequence Modeling

## What is a Language Model?

A language model assigns probability to sequences. Formally, it estimates the joint probability distribution over a sequence:

$$P(x_1, x_2, \ldots, x_T)$$

where $x_1, \ldots, x_T$ are sequential elements (words, characters, tokens, or in finance: price ticks, returns, order events).

### The Chain Rule Decomposition

We decompose this joint probability using the chain rule of probability:

$$P(x_1, x_2, \ldots, x_T) = P(x_1) \cdot P(x_2 | x_1) \cdot P(x_3 | x_1, x_2) \cdots P(x_T | x_1, \ldots, x_{T-1})$$

More compactly:

$$P(x_1, x_2, \ldots, x_T) = \prod_{t=1}^T P(x_t | x_{<t})$$

where $x_{<t} = \{x_1, x_2, \ldots, x_{t-1}\}$ represents the context (all previous elements).

**Key insight:** The probability of the entire sequence factorizes into predictions of each element conditioned on its history. This makes the problem tractable: instead of modeling $T$-way dependencies directly, we model one-step-ahead prediction repeatedly.

## N-Gram Models: Classical Approach

Before neural networks, n-gram models were the workhorse of sequence modeling.

### Principle

An n-gram model makes a **Markov assumption**: the probability of the next element depends only on the previous $n-1$ elements, not the entire history:

$$P(x_t | x_{<t}) \approx P(x_t | x_{t-n+1}, \ldots, x_{t-1})$$

**Common variants:**
- **Unigram** $(n=1)$: $P(x_t) = P(x_t)$ - context-independent frequencies
- **Bigram** $(n=2)$: $P(x_t | x_{t-1})$ - single-step context
- **Trigram** $(n=3)$: $P(x_t | x_{t-2}, x_{t-1})$ - two-step context

### Counting and Estimation

N-gram probabilities are estimated from data via relative frequency:

$$P(x_t = w | x_{t-1} = v) = \frac{\text{count}(v, w)}{\text{count}(v)}$$

Simply count transitions in the training corpus and normalize.

### Smoothing

A critical problem: **data sparsity**. Most n-gram combinations never appear in training data, resulting in zero counts and zero probabilities.

**Smoothing techniques** assign non-zero probability to unseen n-grams:

1. **Add-one smoothing (Laplace smoothing):**
   $$P(x_t = w | \text{context}) = \frac{\text{count}(\text{context}, w) + 1}{\text{count}(\text{context}) + V}$$
   where $V$ is vocabulary size

2. **Backoff:** If an n-gram is unseen, fall back to lower-order model (e.g., bigram → unigram)

3. **Interpolation:** Mix predictions from multiple n-gram orders with learned weights

!!! note "Limitations of N-Gram Models"
    - Cannot capture dependencies beyond $n-1$ steps
    - Exponential growth in parameters with $n$ (curse of dimensionality)
    - Binary decision at position $n$: context either included or dropped (no gradual importance)
    - Requires extensive smoothing for real vocabularies

## Neural Language Models: RNNs for Sequences

Neural language models overcome n-gram limitations by using **recurrent neural networks** (RNNs) to dynamically maintain a latent context representation.

### RNN Architecture for Language Modeling

An RNN processes sequences element-by-element, maintaining a hidden state $h_t$:

$$h_t = f(h_{t-1}, x_t; \theta)$$

where $f$ is a recurrent cell (Elman, LSTM, or GRU) and $\theta$ are learnable parameters.

At each timestep, an output distribution is computed from the hidden state:

$$P(x_{t+1} | x_1, \ldots, x_t) = \text{softmax}(\mathbf{W} h_t + \mathbf{b})$$

**Advantages over n-grams:**
- Theoretically unlimited context through hidden state compression
- Learned representations capture linguistic (or market) structure
- No explicit limit on dependency range
- Single set of parameters reused across sequence positions (weight sharing)

### Unrolling and Backpropagation Through Time (BPTT)

Training an RNN language model involves:

1. **Forward pass:** Unroll the network over the sequence, computing $h_t$ and output distributions at each step
2. **Loss computation:** Sum cross-entropy losses at each position
3. **Backward pass:** Backpropagate through time (BPTT) to compute gradients
4. **Update:** Gradient descent on parameters

!!! warning "Vanishing/Exploding Gradients"
    Long-term dependencies remain challenging. Gradients propagated backward through time can vanish (exponentially decay) or explode, limiting effective context length in practice. This motivated LSTM and GRU architectures with gating mechanisms.

## Perplexity: Evaluation Metric

Perplexity is the standard evaluation metric for language models, quantifying how well the model predicts a test sequence.

### Definition

For a test sequence $x_1, \ldots, x_T$, perplexity is defined as:

$$\text{PPL} = P(x_1, \ldots, x_T)^{-1/T} = \exp\left(-\frac{1}{T}\sum_{t=1}^T \log P(x_t | x_{<t})\right)$$

### Interpretation

- **Lower perplexity = better model** - the model assigns higher probability to the test sequence
- **Baseline:** Uniform distribution over vocabulary of size $V$ has perplexity $V$
- **Intuition:** Geometric mean of inverse probabilities; represents the average branching factor

**Example:**
- Vocabulary size 10,000
- Uniform model perplexity: 10,000
- A model achieving perplexity of 50 is 200x better than uniform
- Interpret as: the model is "as confused" as if choosing from 50 equally-likely alternatives

### Relationship to Loss

During training, we minimize cross-entropy loss:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log P(x_t | x_{<t})$$

The relationship is direct:

$$\text{PPL} = e^{\mathcal{L}}$$

A decrease in loss directly correlates with decrease in perplexity.

## Teacher Forcing in Training

### The Concept

During training, RNNs are exposed to **ground truth** previous elements. During inference, they receive **model predictions** from previous steps.

**Teacher forcing:** At training time, feed the true previous element $x_t$ regardless of what the model predicted:

$$P(x_{t+1} | x_1, \ldots, x_t) \text{ conditioned on true } x_t$$

versus inference where you condition on $\hat{x}_t$ (model's best guess).

### Benefits and Tradeoffs

**Benefits:**
- Speeds up training convergence
- Provides clean gradient signal at each step
- Decouples error propagation across timesteps

**Drawbacks:**
- **Distribution mismatch:** During inference, the model encounters different distributions (its own errors) than seen during training
- **Exposure bias:** Model relies on teacher-provided context, potentially failing when it must condition on its own predictions
- Can lead to brittle models that struggle with long-horizon generation

### Scheduled Sampling

A middle ground: gradually transition from teacher forcing to model predictions during training:

$$x_t^{\text{input}} = \begin{cases} x_t & \text{with probability } 1 - \epsilon(k) \\ \hat{x}_t & \text{with probability } \epsilon(k) \end{cases}$$

where $\epsilon(k)$ increases over epochs $k$, starting near 0.

This reduces exposure bias while maintaining training efficiency.

## Connection to Financial Time Series

Language modeling directly applies to financial sequences with minimal conceptual changes:

### Modeling Return Sequences

Instead of tokens, model log-returns $r_t = \log(P_t / P_{t-1})$:

$$P(r_t | r_{t-1}, r_{t-2}, \ldots) = \mathcal{N}(\mu_t, \sigma_t^2)$$

where $\mu_t$ and $\sigma_t^2$ are outputs from an RNN predicting mean and variance (volatility) given history.

**Applications:**
- Next-step return forecasting
- Volatility prediction
- Conditional probability distributions for risk management

### Order Flow and Microstructure

Model sequences of trades and order events:
- Bid-ask spreads evolution
- Trade size sequences
- Direction of price movement conditioned on order flow history

RNN language models capture temporal dependencies in market microstructure.

### Multivariate Sequences

Extend to multiple assets with:

$$P(x_t^{(1)}, x_t^{(2)}, \ldots, x_t^{(n)} | \text{history})$$

where superscripts index different assets. An RNN maintains a latent state capturing cross-asset dependencies and correlations.

!!! info "Financial Sequence Modeling Benefits"
    - Captures market regimes and momentum effects
    - Models time-varying correlation structures
    - Enables scenario generation and stress testing
    - Supports adaptive trading strategies
    - Provides probabilistic forecasts with uncertainty quantification

## Bridge to Transformers (Chapter 8)

RNN language models have fundamental limitations despite their success:

1. **Sequential computation:** Must process element-by-element; cannot parallelize efficiently
2. **Vanishing gradients:** Long-term dependencies difficult despite LSTMs/GRUs
3. **Hidden state bottleneck:** All information compressed into fixed-size vector $h_t$

**Transformers** (next chapter) address these limitations through:

- **Self-attention mechanism:** Direct, learnable connections between any pair of positions
- **Parallelizable architecture:** Process all positions simultaneously
- **Scalability:** More effective use of model capacity for longer sequences
- **Transfer learning:** Pre-trained models transferable to diverse downstream tasks

The language modeling objective remains the same - predict $P(x_t | x_{<t})$ - but the architecture fundamentally changes how context is processed and integrated.

Modern language models based on Transformers (BERT, GPT, etc.) have achieved remarkable performance across natural language and, increasingly, financial sequence tasks, building on the language modeling foundations established by RNNs.

## Summary

Language models learn probability distributions over sequences via the chain rule decomposition. Classical n-gram approaches are limited by context and data sparsity. RNN language models learn distributed representations maintaining flexible, learned context, enabling better long-range dependency modeling. Evaluation via perplexity provides a principled metric for model comparison. Teacher forcing enables efficient training but introduces exposure bias. In finance, language modeling principles apply directly to return sequences, order flows, and multivariate market data, enabling forecasting, risk quantification, and scenario generation.
