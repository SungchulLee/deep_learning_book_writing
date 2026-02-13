# Quick Start Guide - Module 37: Language Modeling

## Installation

```bash
pip install torch numpy
# For advanced tutorials (08):
pip install transformers
```

## Running Tutorials

### Beginner Level (01-03)
Start with n-gram models - no neural networks required:

```bash
# N-gram basics
python tutorial_01_ngram_basics.py

# Smoothing techniques
python tutorial_02_ngram_smoothing.py

# Text generation
python tutorial_03_text_generation_ngrams.py
```

### Intermediate Level (04-06)
Neural language models:

```bash
# Feedforward neural LM
python tutorial_04_feedforward_lm.py

# RNN language model
python tutorial_05_rnn_language_model.py

# LSTM language model
python tutorial_06_lstm_language_model.py
```

### Advanced Level (07-10)
Modern architectures and techniques:

```bash
# Transformer language model
python tutorial_07_transformer_lm.py

# Pretrained models (requires transformers)
python tutorial_08_pretrained_lm.py

# Generation strategies
python tutorial_09_conditional_generation.py

# Evaluation metrics
python tutorial_10_lm_evaluation.py
```

## Using Your Own Data

Replace `sample_data.txt` with your own text file:

```python
# Load your corpus
with open('your_data.txt', 'r') as f:
    corpus = f.readlines()

# Train a model
from tutorial_06_lstm_language_model import train_lstm
from tutorial_04_feedforward_lm import Vocabulary

vocab = Vocabulary()
vocab.build_from_corpus(corpus)
model = train_lstm(corpus, vocab)
```

## Learning Path

**Week 1**: Tutorials 01-03 (N-grams)
- Understand probability basics
- Implement smoothing
- Generate simple text

**Week 2**: Tutorials 04-05 (Neural basics)
- Learn embeddings
- Build feedforward and RNN models
- Compare with n-grams

**Week 3**: Tutorials 06-07 (Advanced architectures)
- Master LSTM
- Understand transformers
- Achieve state-of-the-art results

**Week 4**: Tutorials 08-10 (Production)
- Use pretrained models
- Implement generation strategies
- Comprehensive evaluation

## Common Issues

1. **Out of memory**: Reduce batch_size or hidden_dim
2. **Slow training**: Use GPU if available: `model.to('cuda')`
3. **Poor perplexity**: Increase training data or model size
4. **Repetitive generation**: Use nucleus sampling with repetition penalty

## Performance Benchmarks

Expected perplexities on small corpus (sample_data.txt):
- Bigram with Laplace smoothing: ~15-25
- LSTM (256 hidden): ~5-10
- Transformer (4 layers): ~3-8

## Next Steps

After completing this module:
1. Module 26: Transformers NLP (for production systems)
2. Module 39-46: Generative models (VAE, GAN, Diffusion)
3. Module 53-54: Transfer and self-supervised learning

## Support

Each tutorial includes:
- Detailed mathematical explanations
- Heavily commented code
- Exercises for practice
- References to papers

Happy learning!
