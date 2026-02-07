# Chapter 7: Sequence Models

## Overview

Sequence models form the computational backbone of natural language processing, time-series analysis, and any domain where the ordering of observations carries meaning. This chapter traces the evolution of neural sequence modeling from foundational word representations through recurrent architectures, gating mechanisms, attention, and the encoder-decoder paradigm that unified structured prediction under a single framework.

The central challenge is deceptively simple: given a sequence of observations $\mathbf{x} = (x_1, x_2, \ldots, x_T)$, learn a mapping that respects temporal or positional dependencies. The autoregressive factorization

$$P(\mathbf{x}) = \prod_{t=1}^{T} P(x_t \mid x_1, \ldots, x_{t-1})$$

makes the structure explicit—each element depends on its entire history. The architectures in this chapter differ in how they compress, propagate, and selectively access that history.

## Chapter Structure

### 7.1 Word Embeddings

Before modeling sequences, we must represent discrete tokens as continuous vectors amenable to gradient-based learning. This section progresses from the limitations of one-hot encoding through prediction-based methods (Word2Vec, Skip-gram, CBOW), count-based approaches (GloVe), morphology-aware representations (FastText, subword embeddings), and finally contextualized embeddings where the same word receives different vectors depending on its surrounding context. The embedding matrix $\mathbf{E} \in \mathbb{R}^{|V| \times d}$ serves as the interface between symbolic language and the continuous vector spaces that neural networks operate in.

**Quantitative finance connection.** Embedding techniques extend naturally to financial data: mapping ticker symbols, sector codes, or economic indicators into dense vector spaces captures latent relationships (e.g., sector co-movement, supply-chain linkages) that are invisible to one-hot representations.

### 7.2 Recurrent Neural Networks

RNNs introduce the recurrence principle—maintaining a hidden state $h_t$ that accumulates information across timesteps:

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

This section covers the vanilla RNN architecture, hidden state dynamics and their interpretation as learned sufficient statistics, backpropagation through time (BPTT), and the fundamental gradient pathologies—vanishing and exploding gradients—that arise when propagating error signals across long sequences. Practical solutions including gradient clipping, bidirectional processing, and deep (stacked) RNN architectures complete the treatment.

**Quantitative finance connection.** RNNs provide a natural framework for modeling sequential financial data—price series, order flow, and macroeconomic indicators—where each observation depends on the accumulated market state.

### 7.3 LSTM and GRU

Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs) solve the vanishing gradient problem through learned gating mechanisms that control information flow. The LSTM cell state $C_t$ provides an uninterrupted gradient highway across timesteps, while three gates (forget, input, output) regulate what information is discarded, stored, and exposed:

$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f), \quad i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

GRUs achieve comparable performance with fewer parameters by merging the cell and hidden states and using two gates (reset, update) instead of three. This section covers both architectures in detail, including peephole connections and stacked configurations.

**Quantitative finance connection.** The gating mechanism maps directly to regime-aware modeling: the forget gate can learn to discard pre-crisis patterns when market regime shifts are detected, while the cell state maintains long-horizon dependencies critical for carry trades and mean-reversion strategies.

### 7.4 Attention

Attention mechanisms liberate sequence models from the information bottleneck of fixed-size hidden states by allowing the model to dynamically focus on relevant parts of the input at each decoding step. The core computation—query-key matching followed by value aggregation:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

is covered from its origins in Bahdanau (additive) and Luong (multiplicative) attention through self-attention and multi-head attention, which form the foundation of the Transformer architecture covered in Chapter 8.

**Quantitative finance connection.** Attention weights provide interpretable importance scores—identifying which historical timesteps, assets, or features most influence a prediction. This transparency is valuable in risk management and regulatory contexts where model interpretability is required.

### 7.5 Sequence-to-Sequence

The encoder-decoder framework unifies variable-length input-to-output mappings under a single architecture. The encoder compresses the source sequence into context representations; the decoder generates the target sequence autoregressively, conditioned on that context. This section covers the full seq2seq pipeline: encoder-decoder architecture, context vector design and its information bottleneck, training strategies (teacher forcing, scheduled sampling), and inference algorithms (beam search, length normalization). A complete French→English translation implementation ties all concepts together.

**Quantitative finance connection.** Seq2seq architectures apply to any structured prediction task in finance: generating trade execution schedules from order specifications, translating natural-language analyst reports into structured signals, or mapping historical factor exposures to forward-looking portfolio allocations.

## Mathematical Prerequisites

This chapter assumes familiarity with:

- **Linear algebra**: matrix-vector products, eigendecomposition, singular values (Ch. 1)
- **Calculus**: chain rule, Jacobian matrices, gradient computation (Ch. 1)
- **Probability**: conditional distributions, maximum likelihood, cross-entropy loss (Ch. 1)
- **Neural network fundamentals**: feedforward layers, backpropagation, SGD and Adam optimizers (Chs. 2–3)
- **PyTorch basics**: `nn.Module`, autograd, `DataLoader` (Ch. 4)

## Notation

| Symbol | Meaning |
|--------|---------|
| $T$ | Sequence length |
| $d$ | Embedding / input dimension |
| $H$ | Hidden state dimension |
| $V$ | Vocabulary (set) or vocabulary size (scalar, from context) |
| $h_t$ | Hidden state at timestep $t$ |
| $C_t$ | LSTM cell state at timestep $t$ |
| $\sigma(\cdot)$ | Sigmoid activation |
| $\odot$ | Element-wise (Hadamard) product |
| $[a; b]$ | Concatenation of vectors $a$ and $b$ |
| $\alpha_{t,j}$ | Attention weight from decoder step $t$ to encoder position $j$ |

## Code Implementations

Each section pairs theoretical exposition with complete, runnable PyTorch code:

| Implementation | Section | Description |
|----------------|---------|-------------|
| `word2vec.py` | 7.1 | Skip-gram and CBOW training on text corpora |
| `ngram_lm.py` | 7.1 | N-gram neural language model |
| `glove_embedding.py` | 7.1 | GloVe vector loading and analogy evaluation |
| `glove_sentiment.py` | 7.1 | Sentiment classification with pretrained GloVe embeddings |
| `char_rnn_classification.py` | 7.2 | Character-level RNN for surname nationality classification |
| `char_rnn_generation.py` | 7.2 | Character-level RNN for name generation |
| `seq2seq_code.md` | 7.5 | French→English translation with Bahdanau attention |

All implementations are self-contained single files that download their own data, train from scratch, and include evaluation and visualization routines.
