# Word Embeddings Tutorial with PyTorch

A comprehensive, beginner-friendly tutorial on word embeddings using PyTorch, progressing from basic concepts to advanced implementations.

## 📚 Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Tutorial Roadmap](#tutorial-roadmap)
- [Getting Started](#getting-started)
- [Concepts Explained](#concepts-explained)
- [References](#references)

## 🎯 Overview

This tutorial package provides hands-on PyTorch implementations of word embedding techniques, designed specifically for undergraduate students and beginners in Natural Language Processing (NLP). Each example is fully commented and progressively builds on previous concepts.

**What you'll learn:**
- Word embeddings and their importance in NLP
- N-gram language models
- Different loss functions (Cross Entropy, NLL Loss)
- Skip-gram and CBOW architectures
- Word2Vec implementation from scratch
- Visualizing word embeddings
- Working with pre-trained embeddings

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Extract the tutorial package:**
   ```bash
   unzip word_embeddings_tutorial.zip
   cd word_embeddings_tutorial
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Project Structure

```
word_embeddings_tutorial/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/                              # Sample text data
│   └── sample_text.txt               # Training corpus
│
├── utils/                             # Utility modules
│   ├── __init__.py
│   ├── word_embedding_ngram.py       # N-gram model utilities
│   ├── visualization.py              # Plotting functions
│   └── data_loader.py                # Data loading utilities
│
├── 01_basics/                         # Basic tutorials
│   ├── 01_simple_embeddings.py       # Introduction to embeddings
│   ├── 02_ngram_cross_entropy.py     # N-gram with CrossEntropyLoss
│   ├── 03_ngram_functional.py        # Using F.cross_entropy
│   └── 04_ngram_nll_loss.py          # N-gram with NLLLoss
│
├── 02_intermediate/                   # Intermediate tutorials
│   ├── 01_loss_comparison.py         # Compare different loss functions
│   ├── 02_cbow_model.py              # Continuous Bag of Words
│   ├── 03_skipgram_model.py          # Skip-gram architecture
│   └── 04_embedding_analysis.py      # Analyzing learned embeddings
│
└── 03_advanced/                       # Advanced tutorials
    ├── 01_word2vec_full.py           # Complete Word2Vec implementation
    ├── 02_negative_sampling.py       # Negative sampling technique
    ├── 03_pretrained_embeddings.py   # Using GloVe embeddings
    └── 04_embedding_visualization.py  # t-SNE visualization
```

## 🚀 Tutorial Roadmap

### Level 1: Basics (Start Here!)

**Goal:** Understand what word embeddings are and build your first n-gram model.

1. **01_simple_embeddings.py**
   - Introduction to word embeddings
   - Creating an embedding layer
   - Basic forward pass
   - **Estimated time:** 15 minutes

2. **02_ngram_cross_entropy.py**
   - Build a simple n-gram language model
   - Use CrossEntropyLoss
   - Train on sample text
   - **Estimated time:** 20 minutes

3. **03_ngram_functional.py**
   - Same model with functional API (F.cross_entropy)
   - Understand the difference between classes and functions
   - **Estimated time:** 10 minutes

4. **04_ngram_nll_loss.py**
   - Use Negative Log Likelihood Loss
   - Understand log_softmax + NLLLoss = CrossEntropyLoss
   - **Estimated time:** 15 minutes

### Level 2: Intermediate

**Goal:** Explore different architectures and analyze embeddings.

1. **01_loss_comparison.py**
   - Compare all three loss functions side-by-side
   - Visualize training curves
   - **Estimated time:** 20 minutes

2. **02_cbow_model.py**
   - Implement Continuous Bag of Words (CBOW)
   - Predict center word from context
   - **Estimated time:** 30 minutes

3. **03_skipgram_model.py**
   - Implement Skip-gram model
   - Predict context from center word
   - **Estimated time:** 30 minutes

4. **04_embedding_analysis.py**
   - Find similar words
   - Compute word analogies
   - Visualize embedding space
   - **Estimated time:** 25 minutes

### Level 3: Advanced

**Goal:** Build production-ready models and work with large-scale embeddings.

1. **01_word2vec_full.py**
   - Complete Word2Vec implementation
   - Hierarchical softmax
   - **Estimated time:** 45 minutes

2. **02_negative_sampling.py**
   - Efficient training with negative sampling
   - Understand the sampling distribution
   - **Estimated time:** 40 minutes

3. **03_pretrained_embeddings.py**
   - Load and use pre-trained GloVe embeddings
   - Fine-tune for specific tasks
   - **Estimated time:** 30 minutes

4. **04_embedding_visualization.py**
   - t-SNE dimensionality reduction
   - Interactive visualization
   - Cluster analysis
   - **Estimated time:** 35 minutes

## 🎓 Getting Started

### Quick Start (5 minutes)

Run your first word embedding example:

```bash
cd 01_basics
python 01_simple_embeddings.py
```

This will introduce you to the concept of word embeddings with a minimal example.

### Recommended Learning Path

1. **Complete all Basic tutorials** in order (01_basics/)
2. **Move to Intermediate** tutorials (02_intermediate/)
3. **Challenge yourself** with Advanced tutorials (03_advanced/)
4. **Experiment:** Modify the code, try different hyperparameters!

### Running Examples

Each script is standalone and can be run directly:

```bash
cd 01_basics
python 02_ngram_cross_entropy.py
```

Most scripts will:
- Train a model
- Display training progress
- Show loss curves
- Print example outputs

## 📖 Concepts Explained

### What are Word Embeddings?

Word embeddings are dense vector representations of words where words with similar meanings are closer together in the vector space. Instead of representing words as one-hot vectors (sparse, high-dimensional), embeddings are:
- **Dense:** All dimensions have non-zero values
- **Low-dimensional:** Typically 50-300 dimensions
- **Meaningful:** Capture semantic relationships

**Example:**
```
king - man + woman ≈ queen
```

### N-gram Language Models

An n-gram model predicts the next word based on the previous (n-1) words.

**Example with trigrams (n=3):**
```
Input:  "the cat"
Output: "sat" (predicted next word)
```

### Loss Functions Comparison

| Loss Function | When to Use | Output Required |
|--------------|-------------|-----------------|
| **CrossEntropyLoss** | Classification tasks | Raw logits |
| **F.cross_entropy** | Same as above (functional) | Raw logits |
| **NLLLoss** | When using log_softmax | Log probabilities |

**Mathematical relationship:**
```
CrossEntropyLoss(x, y) = NLLLoss(log_softmax(x), y)
```

### CBOW vs Skip-gram

| Aspect | CBOW | Skip-gram |
|--------|------|-----------|
| **Input** | Context words | Center word |
| **Output** | Center word | Context words |
| **Speed** | Faster | Slower |
| **Performance** | Better on frequent words | Better on rare words |
| **Use Case** | Smaller datasets | Larger datasets |

### Negative Sampling

Instead of computing probabilities over the entire vocabulary (expensive), negative sampling:
1. Takes the positive pair (center, context)
2. Samples k negative words
3. Trains binary classifier: positive vs negative

**Benefits:**
- Much faster training
- Better for large vocabularies
- Similar quality to full softmax

## 🔍 Common Issues & Solutions

### Issue 1: CUDA Out of Memory
**Solution:** Reduce batch size or embedding dimension in the ARGS configuration.

### Issue 2: Loss not decreasing
**Solution:** 
- Check learning rate (try 0.001 to 0.1)
- Ensure data is properly preprocessed
- Increase training epochs

### Issue 3: Poor word analogies
**Solution:**
- Train longer
- Use larger corpus
- Increase embedding dimension
- Try different context window size

## 📚 References

### Papers
- Mikolov et al. (2013). "Efficient Estimation of Word Representations in Vector Space"
- Mikolov et al. (2013). "Distributed Representations of Words and Phrases and their Compositionality"
- Pennington et al. (2014). "GloVe: Global Vectors for Word Representation"

### Resources
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Stanford CS224N: NLP with Deep Learning](http://web.stanford.edu/class/cs224n/)
- [Word2Vec Tutorial](https://arxiv.org/abs/1301.3781)

## 🤝 Contributing

This is an educational resource. If you find errors or have suggestions:
1. Review the code carefully
2. Test your proposed changes
3. Document your modifications

## 📝 License

This tutorial package is provided for educational purposes. Feel free to use and modify for learning.

## 💡 Tips for Success

1. **Read the comments:** Every line is explained in detail
2. **Run before modifying:** Understand the baseline behavior first
3. **Experiment:** Change hyperparameters and observe results
4. **Visualize:** Use the plotting functions to understand what's happening
5. **Compare:** Run multiple variations and compare results
6. **Ask questions:** Research terms you don't understand

## 🎉 Happy Learning!

Word embeddings are fundamental to modern NLP. Master these concepts, and you'll be well-prepared for advanced topics like:
- Transformers (BERT, GPT)
- Attention mechanisms
- Transfer learning in NLP
- Neural machine translation

**Good luck on your NLP journey!** 🚀
