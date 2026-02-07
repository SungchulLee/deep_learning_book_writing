# Seq2Seq Overview

Sequence-to-sequence (seq2seq) models transform variable-length input sequences into variable-length output sequences, enabling tasks where input and output dimensions differ fundamentally. Introduced by Sutskever et al. (2014) and Cho et al. (2014), the seq2seq paradigm unified machine translation, text summarization, dialogue generation, and numerous other structured prediction problems under a single encoder-decoder framework.

## The Sequence Transduction Problem

Many real-world tasks require mapping one sequence to another of potentially different length. Traditional fixed-size neural networks cannot naturally handle this variability. The seq2seq framework addresses this through a two-stage approach: an **encoder** that reads and compresses the input, and a **decoder** that generates the output conditioned on the encoded representation.

The joint probability of the output sequence $\mathbf{y} = (y_1, \ldots, y_{T'})$ given input $\mathbf{x} = (x_1, \ldots, x_T)$ factorizes autoregressively:

$$P(\mathbf{y} | \mathbf{x}) = \prod_{t=1}^{T'} P(y_t | y_{<t}, \mathbf{x})$$

This factorization is exact—the modeling challenge lies entirely in how faithfully each conditional $P(y_t | y_{<t}, \mathbf{x})$ captures the true data distribution. The encoder-decoder architecture provides a structured approach to parameterizing these conditionals through learned representations.

## Architecture Overview

```
Source Sequence: x₁, x₂, ..., x_T
                    ↓
              ┌──────────┐
              │  Encoder  │  ← Processes input, builds representations
              └──────────┘
                    ↓
              Context Vector(s)  ← Compressed source information
                    ↓
              ┌──────────┐
              │  Decoder  │  ← Generates output autoregressively
              └──────────┘
                    ↓
Target Sequence: y₁, y₂, ..., y_T'
```

The **encoder** processes the source sequence through an embedding layer and recurrent network (LSTM or GRU), producing hidden state representations at each position. The final hidden state—or a function of all hidden states—serves as the **context vector** that summarizes source information for the decoder.

The **decoder** generates the target sequence one token at a time. At each step, it takes the previously generated token (or ground truth during training) and the context, updates its hidden state, and predicts a distribution over the output vocabulary. Generation continues until a special end-of-sequence token is produced or a maximum length is reached.

## Core Components

### Encoder

The encoder reads the input sequence and produces a set of hidden representations. For an input $\mathbf{x} = (x_1, \ldots, x_T)$:

$$\mathbf{e}_t = \text{Embed}(x_t) \in \mathbb{R}^{d_e}$$

$$\mathbf{h}_t^{enc} = f_{enc}(\mathbf{e}_t, \mathbf{h}_{t-1}^{enc})$$

where $f_{enc}$ is an LSTM or GRU cell. Bidirectional encoders process the sequence in both directions, concatenating forward and backward states to capture context from both past and future positions:

$$\mathbf{h}_t = [\overrightarrow{\mathbf{h}}_t ; \overleftarrow{\mathbf{h}}_t] \in \mathbb{R}^{2d_h}$$

### Context Vector

The context vector $\mathbf{c}$ bridges encoder and decoder. In the simplest form, it is the encoder's final hidden state:

$$\mathbf{c} = \mathbf{h}_T^{enc}$$

This creates a fundamental **information bottleneck**: the entire input must be compressed into a single fixed-dimensional vector. For long sequences, this fixed-size representation cannot capture all relevant information—a limitation that motivated attention mechanisms. The mutual information between input and context is bounded by context dimensionality, independent of input length.

### Decoder

The decoder generates tokens autoregressively, conditioning on the context and previously generated tokens:

$$\mathbf{s}_t = f_{dec}([\mathbf{e}_{y_{t-1}} ; \mathbf{c}], \mathbf{s}_{t-1})$$

$$P(y_t | y_{<t}, \mathbf{c}) = \text{softmax}(\mathbf{W}_o \mathbf{s}_t + \mathbf{b}_o)$$

The decoder is initialized with the encoder's final state ($\mathbf{s}_0 = \mathbf{h}_T^{enc}$) and begins generation from a start-of-sequence token.

## Training: Teacher Forcing

During training, the decoder can use either ground truth tokens or its own predictions as input at each step. **Teacher forcing** feeds the correct previous token, providing stable gradients and fast convergence. However, this creates **exposure bias**: the model never encounters its own prediction errors during training, leading to error accumulation at inference time.

The training objective with teacher forcing decomposes into independent per-step classification problems:

$$\mathcal{L}_{TF} = -\sum_{t=1}^{T'} \log P_\theta(y_t^* | y_1^*, \ldots, y_{t-1}^*, \mathbf{c})$$

**Scheduled sampling** bridges the train-test gap by gradually reducing teacher forcing over the course of training. Starting with full teacher forcing for stable initial learning, the ratio decays so the model progressively learns to handle its own predictions:

| Phase | Epochs | TF Ratio | Purpose |
|-------|--------|----------|---------|
| Early | 1–10 | 0.9–1.0 | Stable gradients, rapid learning |
| Mid | 10–30 | 0.5–0.9 | Exposure to own predictions |
| Late | 30+ | 0.0–0.5 | Minimize exposure bias |

## Inference: Decoding Strategies

At inference time, the model must generate output without access to ground truth. The choice of decoding strategy significantly impacts output quality.

**Greedy decoding** selects the highest-probability token at each step. It is fast ($O(TV)$) but locally optimal—early mistakes propagate irreversibly through the sequence.

**Beam search** maintains the $K$ most promising partial hypotheses at each step, exploring multiple paths through the search space. This provides a tractable approximation to exact search, reducing the space from $V^T$ to $O(TKV)$ operations. Beam search requires **length normalization** to prevent bias toward shorter sequences, since cumulative log probabilities are always negative and thus penalize longer outputs.

**Stochastic decoding** methods (temperature sampling, top-$k$, nucleus/top-$p$) introduce randomness for applications where diversity matters more than finding the single highest-probability sequence, such as creative writing and dialogue.

| Strategy | Quality | Speed | Diversity | Best For |
|----------|---------|-------|-----------|----------|
| Greedy | Low | Fastest | None | Real-time, simple tasks |
| Beam search | High | Medium | Low | Translation, summarization |
| Top-$k$ sampling | Good | Fast | Medium | Open-ended dialogue |
| Nucleus (top-$p$) | Good | Fast | Medium | Story generation |

## Practical Considerations

### Gradient Management

RNN-based seq2seq models are susceptible to exploding gradients. **Gradient clipping** is essential:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### Weight Initialization

Proper initialization ensures stable training. Orthogonal initialization for recurrent weights and Xavier initialization for input-to-hidden weights are standard practice. Setting the LSTM forget gate bias to 1.0 prevents early information loss.

### Variable-Length Sequences

Efficient batching requires **padding** shorter sequences and **packing** them to avoid unnecessary computation on padding tokens. PyTorch's `pack_padded_sequence` and `pad_packed_sequence` utilities handle this transparently.

### Common Issues

| Problem | Cause | Solution |
|---------|-------|----------|
| Information bottleneck | Fixed context vector | Attention mechanisms |
| Exposure bias | Teacher forcing train-test gap | Scheduled sampling |
| Short output bias | Negative log-prob accumulation | Length normalization |
| Repetitive output | Mode collapse in decoding | Repetition penalties, n-gram blocking |
| Unknown words | Fixed vocabulary | Subword tokenization (BPE, WordPiece) |
| Vanishing gradients | Long sequences | LSTM/GRU gating, gradient clipping |

## Applications

The seq2seq framework with LSTM/GRU encoders and decoders has been applied across diverse domains:

| Application | Key Techniques | Evaluation |
|-------------|---------------|------------|
| Machine translation | Bidirectional encoder, attention, subword tokenization | BLEU, METEOR |
| Text summarization | Pointer-generator networks, coverage penalty | ROUGE-1/2/L |
| Dialogue systems | Hierarchical encoding, turn-level context | Perplexity, human eval |
| Speech recognition | Conv frontend, CTC loss, attention | WER, CER |
| Code generation | Copy mechanisms, constrained decoding | Execution accuracy |
| Time series forecasting | Multi-step decoder, teacher forcing | MSE, MAE |

## Historical Context and Evolution

The seq2seq paradigm emerged from parallel developments:

**Sutskever et al. (2014)** demonstrated that LSTMs could learn sequence-to-sequence mappings by training on reversed input sequences, achieving breakthrough machine translation results.

**Cho et al. (2014)** introduced the GRU architecture and proposed learning phrase representations using RNN encoder-decoders for statistical machine translation.

Subsequent innovations addressed the core limitations of the basic architecture:

- **Attention mechanisms** (Bahdanau et al., 2015) allowed the decoder to selectively access different encoder positions, resolving the information bottleneck
- **Transformer architecture** (Vaswani et al., 2017) replaced recurrence entirely with self-attention, enabling parallel training
- **Pre-trained models** (BERT, GPT) leveraged large-scale unsupervised pre-training to learn universal representations

Despite the dominance of transformer-based architectures in modern NLP, understanding the seq2seq paradigm with recurrent networks remains essential. The concepts of encoding, decoding, attention, teacher forcing, and beam search carry directly into transformer models and form the conceptual foundation for modern sequence-to-sequence learning.

## Section Roadmap

The remaining pages in this section explore each component in depth:

- **Encoder-Decoder Framework** — Detailed architecture, mathematical formulation, and complete PyTorch implementation of encoders, decoders, and their integration
- **Context Vector** — The information bottleneck problem, information-theoretic analysis, and multi-layer architectures
- **Teacher Forcing** — Training dynamics, exposure bias analysis, and mixed forcing strategies
- **Scheduled Sampling** — Decay schedules (linear, exponential, inverse sigmoid), curriculum learning, and adaptive approaches
- **Beam Search** — Algorithm formulation, batched implementation, coverage penalty, diverse beam search, and practical optimizations
- **Length Normalization** — Short sequence bias, Google's length penalty, normalization strategies, and their interaction with beam search
