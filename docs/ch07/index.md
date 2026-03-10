# Chapter Overview


!!! warning "Incomplete page"
    This page is missing the required five-section structure (Concept Definition, Explanation, Diagram / Example). Content needs to be reorganized and expanded.

This chapter covers the complete landscape of sequence modeling, from representing words as dense vectors through recurrent architectures to the attention mechanisms that eventually led to the Transformer revolution. Each section builds on the previous, progressing from foundational representations to increasingly sophisticated sequence-to-sequence architectures.

---

## 7.1 Word Embeddings

Dense vector representations that encode words in continuous space, capturing semantic relationships through geometric properties.

- Word Embeddings Overview -- Comprehensive overview of embedding techniques from one-hot encoding to contextualized representations
- Word Embeddings Tutorial -- Hands-on PyTorch tutorial package for learning word embedding techniques
- Quick Start Guide -- Get started with word embeddings in 5 minutes
- One-Hot Encoding -- Why sparse symbolic representations fail and how the distributional hypothesis motivates dense embeddings
- Word2Vec -- The prediction-based framework that revolutionized NLP with Skip-gram and CBOW architectures
- Skip-gram Model -- Predicting context words from a center word, with derivation and PyTorch implementation
- CBOW Model -- Predicting the center word from surrounding context for fast large-scale training
- Negative Sampling -- Efficient training by converting multi-class softmax into binary classification problems
- GloVe -- Global Vectors combining matrix factorization with local context window methods
- FastText -- Subword-aware embeddings using character n-grams for handling out-of-vocabulary words
- Subword Embeddings -- Decomposing words into character n-grams for better generalization on rare and unseen words
- Contextual Embeddings -- Dynamic representations from ELMo, BERT, and GPT that adapt to surrounding context
- Embedding Visualization -- Visualizing high-dimensional embeddings with PCA, t-SNE, and UMAP

## 7.2 Recurrent Neural Networks

Sequential architectures that maintain hidden state across timesteps to model temporal dependencies.

- [RNN Fundamentals](rnn/rnn_fundamentals.md) -- From sequential data characteristics to the autoregressive factorization that motivates recurrence
- RNN Tutorial Package -- Comprehensive PyTorch RNN tutorial from basics to advanced sequence modeling
- [Vanilla RNN](rnn/vanilla_rnn.md) -- The Elman RNN architecture with tanh nonlinearity and from-scratch implementation
- Hidden State Dynamics -- How the hidden state serves as memory and evolves across timesteps
- [Backpropagation Through Time](rnn/bptt.md) -- Computing gradients through unrolled computational graphs
- [Vanishing Gradients](rnn/vanishing_gradients.md) -- Why repeated multiplication causes gradients to shrink exponentially in long sequences
- Exploding Gradients -- Catastrophic training failures from exponentially growing gradients
- Gradient Clipping -- Limiting gradient magnitudes to prevent exploding gradients during RNN training
- Bidirectional RNN -- Processing sequences in both directions for tasks requiring full context
- Deep RNN -- Stacking multiple RNN layers for hierarchical representation learning
- [Language Models](rnn/language_models.md) -- Using RNNs to assign probabilities to sequences via the chain rule decomposition

## 7.3 LSTM and GRU

Gated recurrent architectures that solve the vanishing gradient problem through dedicated memory pathways.

- LSTM and GRU Overview -- Module overview covering LSTM and GRU theory, implementation, and comparison
- [LSTM Architecture](lstm_gru/lstm_architecture.md) -- The cell state memory highway with forget, input, and output gates
- LSTM Gates -- Detailed analysis of how forget, input, and output gates control information flow
- [Cell State and Gradient Flow](lstm_gru/cell_state.md) -- Why the LSTM cell state enables stable gradient propagation over long sequences
- [Peephole Connections](lstm_gru/peephole.md) -- Giving gates direct access to the cell state for more precise memory regulation
- Stacked LSTM and GRU -- Multi-layer recurrent networks for hierarchical sequential representations
- [GRU Architecture](lstm_gru/gru_architecture.md) -- The streamlined two-gate variant with merged cell and hidden state
- [GRU vs LSTM Comparison](lstm_gru/gru_vs_lstm.md) -- Comprehensive comparison of architecture, parameters, and practical decision frameworks

## 7.4 Sequence-to-Sequence Models

Encoder-decoder architectures that transform variable-length input sequences into variable-length output sequences.

- Seq2Seq Overview -- The sequence transduction problem and the encoder-decoder framework
- Seq2Seq Tutorial Package -- Complete implementation with multiple encoder architectures and attention mechanisms
- Encoder-Decoder Framework -- The fundamental paradigm for mapping input sequences to output sequences
- Context Vector -- The fixed-dimensional bridge between encoder and decoder and its information bottleneck
- Teacher Forcing -- Accelerating training by feeding ground truth tokens, and the exposure bias problem
- Beam Search -- Exploring multiple hypotheses simultaneously for higher-quality sequence generation
- [Length Normalization](seq2seq/length_normalization.md) -- Correcting beam search bias toward shorter sequences
- Scheduled Sampling -- Gradually transitioning from teacher forcing to free running during training
- [Code: French to English Translation](seq2seq/seq2seq_code.md) -- Complete implementation of seq2seq translation with Bahdanau attention

## 7.5 Attention Mechanisms

Dynamic focus mechanisms that allow models to selectively attend to relevant parts of the input.

- [Attention Fundamentals](attention/attention_fundamentals.md) -- Attention as soft dictionary lookup and the query-key-value framework
- Attention Mechanisms Tutorial -- Complete Python implementations from basic attention to multi-head self-attention
- Bahdanau Attention -- The first attention mechanism for neural machine translation using additive scoring
- Luong Attention -- Simplified multiplicative scoring functions and global vs local attention strategies
- [Scaled Dot-Product Attention](attention/scaled_dot_product.md) -- The fundamental building block of the Transformer with scaling for stable gradients
- [Self-Attention](attention/self_attention.md) -- Queries, keys, and values from the same sequence for modeling within-sequence dependencies
- [Multi-Head Attention](attention/multi_head_attention.md) -- Parallel attention heads for attending to different representation subspaces
- Attention Patterns -- Characteristic weight distributions across self-attention, cross-attention, and causal attention
- Attention Visualization -- Practical tools for visualizing and interpreting attention matrices
