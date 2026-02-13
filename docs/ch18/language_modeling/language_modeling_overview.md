# Module 37: Language Modeling

## Overview
This module provides a comprehensive introduction to language modeling, from classical statistical approaches to modern neural architectures. Language models learn to predict the probability distribution of sequences of words, forming the foundation for many NLP applications.

## Learning Objectives
By completing this module, students will:
- Understand the mathematical foundations of language modeling
- Implement n-gram models with smoothing techniques
- Build neural language models using RNNs and LSTMs
- Create transformer-based language models
- Evaluate language models using perplexity and other metrics
- Apply language models to text generation tasks

## Prerequisites
- Module 19: Activation Functions
- Module 20: Feedforward Networks
- Module 28: Recurrent Neural Networks
- Module 29: LSTM and GRU
- Module 26: Transformers (for advanced sections)

## Mathematical Foundations

### Language Model Definition
A language model assigns probabilities to sequences of words:
- P(w₁, w₂, ..., wₙ) = probability of sequence
- By chain rule: P(w₁, ..., wₙ) = ∏ᵢ P(wᵢ | w₁, ..., wᵢ₋₁)

### N-gram Models
- Markov assumption: P(wᵢ | w₁, ..., wᵢ₋₁) ≈ P(wᵢ | wᵢ₋ₙ₊₁, ..., wᵢ₋₁)
- Maximum Likelihood Estimate: P(wᵢ | wᵢ₋ₙ₊₁, ..., wᵢ₋₁) = count(wᵢ₋ₙ₊₁, ..., wᵢ) / count(wᵢ₋ₙ₊₁, ..., wᵢ₋₁)

### Neural Language Models
- Learn distributed representations of context
- Model: P(wᵢ | context) = softmax(Wh + b)
- Where h is a learned representation of the context

### Evaluation Metrics
- **Perplexity**: PP = 2^(-1/N ∑ log₂ P(wᵢ | context))
  - Lower perplexity indicates better model
  - Exponential of average negative log-likelihood
- **Cross-entropy**: H = -1/N ∑ log P(wᵢ | context)

## Module Structure

### Beginner Level (01-03)
1. **01_ngram_basics.py** - Unigram and bigram models, basic probability calculations
2. **02_ngram_smoothing.py** - Laplace smoothing, add-k smoothing, interpolation
3. **03_text_generation_ngrams.py** - Generate text using n-gram models

### Intermediate Level (04-06)
4. **04_feedforward_lm.py** - Neural language model with fixed context window
5. **05_rnn_language_model.py** - RNN-based language model with variable-length contexts
6. **06_lstm_language_model.py** - LSTM language model with improved long-term dependencies

### Advanced Level (07-10)
7. **07_transformer_lm.py** - Transformer-based language model (GPT-style)
8. **08_pretrained_lm.py** - Fine-tuning pretrained language models
9. **09_conditional_generation.py** - Controlled text generation with different strategies
10. **10_lm_evaluation.py** - Comprehensive evaluation suite for language models

## Datasets Used
- **Penn Treebank (PTB)**: Standard benchmark for language modeling
- **WikiText-2**: Larger dataset with longer-term dependencies
- **Custom text corpora**: For demonstration purposes

## Key Concepts Covered
- **Statistical Language Models**: N-grams, smoothing, backoff
- **Neural Architectures**: Feedforward, RNN, LSTM, Transformer
- **Training Techniques**: Teacher forcing, gradient clipping, learning rate scheduling
- **Generation Strategies**: Greedy search, beam search, sampling, top-k, nucleus sampling
- **Evaluation**: Perplexity, cross-entropy, human evaluation

## Installation Requirements
```bash
pip install torch torchtext numpy matplotlib seaborn nltk tqdm
```

## Usage Examples

### Training an LSTM Language Model
```python
from tutorial_06_lstm_language_model import train_lstm_lm

model, vocab, train_loss, val_loss = train_lstm_lm(
    text_file='data.txt',
    embedding_dim=256,
    hidden_dim=512,
    num_layers=2,
    epochs=20
)
```

### Generating Text
```python
from tutorial_09_conditional_generation import generate_text

generated = generate_text(
    model=model,
    prompt="The quick brown",
    max_length=50,
    strategy='nucleus',
    temperature=0.8
)
print(generated)
```

## Performance Benchmarks
- **Bigram with Laplace**: Perplexity ~300-500 on PTB
- **LSTM (2-layer, 512 hidden)**: Perplexity ~80-120 on PTB
- **Transformer (6-layer)**: Perplexity ~60-80 on PTB

## Common Pitfalls
1. **Vocabulary size**: Too large leads to slow training, too small loses information
2. **Context length**: Longer contexts need more memory and training time
3. **Smoothing**: Essential for n-gram models to handle unseen sequences
4. **Gradient explosion**: Use gradient clipping for RNN training
5. **Exposure bias**: Difference between training (teacher forcing) and generation

## Extensions and Projects
1. Build a character-level language model
2. Implement different smoothing techniques (Kneser-Ney, Good-Turing)
3. Create a language model for code generation
4. Build a multilingual language model
5. Implement adaptive softmax for large vocabularies

## References
1. Bengio et al. (2003) - "A Neural Probabilistic Language Model"
2. Mikolov et al. (2010) - "Recurrent Neural Network Based Language Model"
3. Merity et al. (2017) - "Regularizing and Optimizing LSTM Language Models"
4. Radford et al. (2019) - "Language Models are Unsupervised Multitask Learners"

## Time Estimate
- **Beginner**: 3-4 hours
- **Intermediate**: 4-5 hours
- **Advanced**: 5-6 hours
- **Total**: 12-15 hours

## Assessment Ideas
1. Implement trigram model with Good-Turing smoothing
2. Compare perplexity of different model architectures
3. Generate coherent text with controlled style/topic
4. Analyze model predictions on ambiguous contexts
5. Build a language model for a specialized domain
