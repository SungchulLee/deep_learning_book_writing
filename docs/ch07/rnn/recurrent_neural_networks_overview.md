# PyTorch RNN Tutorial Package for Undergraduates

A comprehensive, progressively challenging collection of Recurrent Neural Network (RNN) tutorials using PyTorch. Perfect for undergraduate students learning sequence modeling, natural language processing, and time series analysis!

## 📋 Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Tutorial Structure](#tutorial-structure)
- [Quick Start](#quick-start)
- [Key Concepts](#key-concepts)
- [Datasets](#datasets)
- [Usage Examples](#usage-examples)
- [Common Issues](#common-issues)
- [Resources](#resources)

## 🎯 Overview

This tutorial series takes you from RNN basics to advanced sequence modeling:
- **Easy**: Understanding sequences and basic RNN concepts
- **Intermediate**: LSTM, GRU, and practical applications
- **Advanced**: Seq2Seq, attention mechanisms, and generation

**Total Time**: 15-20 hours across all tutorials

## 🔧 Prerequisites

### Knowledge Prerequisites
- Completed CNN tutorials (or equivalent)
- Basic understanding of neural networks
- Python programming
- Basic linear algebra

### Technical Prerequisites
- Python 3.7 or higher
- PyTorch 2.0+
- 4GB RAM minimum (8GB recommended)
- GPU optional (CPU is fine for learning)

## 🚀 Installation

```bash
# Clone or extract the tutorial package
cd pytorch_rnn_tutorial

# Install dependencies
pip install -r requirements.txt
```

### Required Packages
```
torch>=2.0.0
torchtext>=0.15.0
numpy>=1.21.0
matplotlib>=3.5.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

## 📚 Tutorial Structure

### Level 1: Easy - Understanding Sequences (30-45 min each)

#### **01_sequence_basics.py**
- **Topics:** What are sequences? Time dependencies
- **Data:** Simple synthetic sequences
- **Skills:** Sequence representation, batching
- **Output:** Understanding sequential data structure
- **Run:** `python 01_sequence_basics.py`

#### **02_text_preprocessing.py**
- **Topics:** Tokenization, vocabularies, embeddings
- **Data:** Text sentences
- **Skills:** Text to numbers, padding, vocabulary building
- **Output:** Processed text ready for RNNs
- **Run:** `python 02_text_preprocessing.py`

#### **03_time_series_basics.py**
- **Topics:** Time series data preparation
- **Data:** Sine waves, stock prices
- **Skills:** Windowing, normalization, train/test splits
- **Output:** Understanding temporal data
- **Run:** `python 03_time_series_basics.py`

### Level 2: Intermediate - Core RNN Architectures (1-2 hours each)

#### **04_simple_rnn.py**
- **Topics:** Vanilla RNN architecture and training
- **Data:** Name classification (which language?)
- **Architecture:** Basic RNN with single hidden layer
- **Expected Accuracy:** ~70-75%
- **Skills:** RNN forward pass, hidden states
- **Run:** `python 04_simple_rnn.py --epochs 20`

#### **05_lstm_sentiment.py**
- **Topics:** LSTM for sentiment analysis
- **Data:** Movie reviews (IMDB subset)
- **Architecture:** LSTM → Dense → Output
- **Expected Accuracy:** ~85-90%
- **Skills:** LSTM cells, long-term dependencies
- **Run:** `python 05_lstm_sentiment.py --epochs 10`

#### **06_gru_time_series.py**
- **Topics:** GRU for time series prediction
- **Data:** Stock prices or sine wave prediction
- **Architecture:** GRU → Dense → Output
- **Expected MSE:** Low prediction error
- **Skills:** GRU gates, time series forecasting
- **Run:** `python 06_gru_time_series.py --sequence-length 50`

#### **07_bidirectional_rnn.py**
- **Topics:** Bidirectional processing
- **Data:** Text classification
- **Architecture:** Bi-LSTM → Dense
- **Expected Accuracy:** ~88-92%
- **Skills:** Forward and backward context
- **Run:** `python 07_bidirectional_rnn.py`

### Level 3: Advanced - Sequence-to-Sequence (2-3 hours each)

#### **08_seq2seq_basic.py**
- **Topics:** Encoder-Decoder architecture
- **Data:** Number reversal or simple translation
- **Architecture:** Encoder LSTM → Decoder LSTM
- **Expected Accuracy:** ~95%+ on simple tasks
- **Skills:** Seq2Seq fundamentals
- **Run:** `python 08_seq2seq_basic.py`

#### **09_attention_mechanism.py**
- **Topics:** Attention layer implementation
- **Data:** Translation or summarization
- **Architecture:** Seq2Seq + Attention
- **Expected Accuracy:** ~90%+ improvement
- **Skills:** Attention weights, alignment
- **Run:** `python 09_attention_mechanism.py --attention-type additive`

#### **10_text_generation.py**
- **Topics:** Character-level or word-level generation
- **Data:** Shakespeare, recipes, or code
- **Architecture:** LSTM with temperature sampling
- **Output:** Generated text samples
- **Skills:** Autoregressive generation, sampling
- **Run:** `python 10_text_generation.py --temperature 0.8`

## 🎓 Quick Start

### Beginner Path (Week 1)
```bash
# Day 1-2: Understand sequences
python 01_sequence_basics.py
python 02_text_preprocessing.py

# Day 3-4: Time series basics
python 03_time_series_basics.py

# Day 5-7: First RNN model
python 04_simple_rnn.py --epochs 20
```

### Intermediate Path (Week 2-3)
```bash
# Week 2: LSTM and GRU
python 05_lstm_sentiment.py --epochs 10
python 06_gru_time_series.py --sequence-length 50
python 07_bidirectional_rnn.py --epochs 15
```

### Advanced Path (Week 4-5)
```bash
# Week 4-5: Seq2Seq and Generation
python 08_seq2seq_basic.py --epochs 30
python 09_attention_mechanism.py --epochs 25
python 10_text_generation.py --epochs 50
```

## 💡 Key Concepts

### What are RNNs?

**Recurrent Neural Networks** process sequential data by maintaining a hidden state that captures information from previous time steps.

```
Input Sequence:  x₁ → x₂ → x₃ → x₄
                  ↓    ↓    ↓    ↓
Hidden State:    h₁ → h₂ → h₃ → h₄
                  ↓    ↓    ↓    ↓
Output:          y₁   y₂   y₃   y₄
```

### RNN vs CNN

| Feature | CNN | RNN |
|---------|-----|-----|
| Input | Spatial (images) | Sequential (text, time series) |
| Key Operation | Convolution | Recurrence |
| Memory | None | Hidden state |
| Use Cases | Images, video | Text, speech, time series |
| Parameter Sharing | Across space | Across time |

### Core Architectures

#### 1. **Vanilla RNN**
- Simplest recurrent architecture
- Suffers from vanishing gradients
- Good for: Short sequences, learning basics

#### 2. **LSTM (Long Short-Term Memory)**
- Gates: Input, Forget, Output
- Maintains cell state for long-term memory
- Good for: Most sequence tasks, standard choice

#### 3. **GRU (Gated Recurrent Unit)**
- Simplified LSTM (2 gates: Reset, Update)
- Faster training, fewer parameters
- Good for: When LSTM is overkill, speed matters

#### 4. **Bidirectional RNN**
- Processes sequence forward AND backward
- Better context understanding
- Good for: Classification, when full sequence available

### Mathematical Foundations

#### Simple RNN
```
h_t = tanh(W_hh * h_(t-1) + W_xh * x_t + b_h)
y_t = W_hy * h_t + b_y
```

#### LSTM
```
f_t = σ(W_f · [h_(t-1), x_t] + b_f)  # Forget gate
i_t = σ(W_i · [h_(t-1), x_t] + b_i)  # Input gate
o_t = σ(W_o · [h_(t-1), x_t] + b_o)  # Output gate
c̃_t = tanh(W_c · [h_(t-1), x_t] + b_c)  # Cell candidate
c_t = f_t ⊙ c_(t-1) + i_t ⊙ c̃_t  # Cell state
h_t = o_t ⊙ tanh(c_t)  # Hidden state
```

## 📊 Datasets

### Built-in Datasets
1. **Synthetic Sequences** (Tutorial 01)
   - Generated patterns for learning
   - Perfect for understanding basics

2. **Name Dataset** (Tutorial 04)
   - 18 languages, ~20,000 names
   - Multi-class classification

3. **IMDB Reviews** (Tutorial 05)
   - 50,000 movie reviews
   - Binary sentiment classification

4. **Stock Prices** (Tutorial 06)
   - Historical time series data
   - Regression/forecasting

5. **Translation Pairs** (Tutorial 08-09)
   - English-French pairs
   - Sequence-to-sequence

6. **Text Corpus** (Tutorial 10)
   - Shakespeare or custom text
   - Character/word generation

### Custom Data
All tutorials support custom datasets. See individual tutorial documentation.

## 🔨 Usage Examples

### Basic Training
```bash
# Default configuration
python 05_lstm_sentiment.py

# Custom hyperparameters
python 05_lstm_sentiment.py --epochs 20 --lr 0.001 --batch-size 64

# With GPU
python 05_lstm_sentiment.py --device cuda

# Save model
python 05_lstm_sentiment.py --save-model --path ./models/sentiment.pth
```

### Advanced Options
```bash
# Bidirectional LSTM
python 07_bidirectional_rnn.py --hidden-size 256 --num-layers 2

# Attention mechanism
python 09_attention_mechanism.py --attention-type scaled-dot-product

# Text generation with sampling
python 10_text_generation.py --temperature 0.8 --max-length 500
```

### Evaluation Mode
```bash
# Load and evaluate
python 05_lstm_sentiment.py --eval-only --path ./models/sentiment.pth
```

## 🎯 Common Issues and Solutions

### Issue: Exploding/Vanishing Gradients
**Solution:** 
- Use LSTM/GRU instead of vanilla RNN
- Gradient clipping: `torch.nn.utils.clip_grad_norm_()`
- Proper weight initialization

### Issue: Slow Training
**Solution:**
- Reduce sequence length
- Use smaller batch size
- Enable GPU: `--device cuda`
- Use GRU instead of LSTM

### Issue: Overfitting
**Solution:**
- Add dropout: `--dropout 0.5`
- Reduce model size
- More training data
- Early stopping

### Issue: Poor Text Generation
**Solution:**
- Train longer (50+ epochs)
- Adjust temperature (0.5-1.2)
- Use larger model
- More diverse training data

### Issue: Memory Errors
**Solution:**
- Reduce batch size
- Reduce sequence length
- Use gradient accumulation
- Smaller hidden size

## 📖 Learning Path by Background

### If you're new to RNNs
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10
(Complete all in order)
```

### If you know basic RNNs
```
Start at: 05 (LSTM) or 06 (GRU)
Then: 07 → 08 → 09 → 10
```

### If you want specific applications

**Text Classification**
→ 02, 04, 05, 07

**Time Series Prediction**
→ 03, 06

**Text Generation**
→ 02, 05, 10

**Machine Translation**
→ 02, 08, 09

## 🎓 Key Differences: RNN Variants

| Feature | RNN | LSTM | GRU |
|---------|-----|------|-----|
| Parameters | Fewest | Most | Medium |
| Training Speed | Fastest | Slowest | Medium |
| Memory Capacity | Poor (short) | Best (long) | Good (medium) |
| Vanishing Gradient | Yes | No | No |
| When to Use | Simple/Short | Standard choice | Speed matters |

## 🚀 After This Tutorial

### Next Topics
1. **Transformers** - Modern alternative to RNNs
2. **BERT/GPT** - Pre-trained language models
3. **Speech Recognition** - Audio sequence processing
4. **Video Analysis** - Spatiotemporal modeling
5. **Reinforcement Learning** - Sequential decision making

### Real-World Projects
1. Chatbot development
2. Stock price prediction
3. Sentiment analysis API
4. Text summarization
5. Named entity recognition

## 📚 Additional Resources

### Papers
- **LSTM**: Hochreiter & Schmidhuber (1997)
- **GRU**: Cho et al. (2014)
- **Seq2Seq**: Sutskever et al. (2014)
- **Attention**: Bahdanau et al. (2015)

### Online Courses
- Stanford CS224N: NLP with Deep Learning
- Fast.ai: Practical Deep Learning
- DeepLearning.AI: Sequence Models

### Documentation
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/char_rnn_classification_tutorial.html)
- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## 🎯 Assessment Checklist

After completing tutorials, you should be able to:
- ✅ Explain the difference between RNN, LSTM, and GRU
- ✅ Implement a basic RNN from scratch
- ✅ Preprocess text for RNN input
- ✅ Train models for classification and generation
- ✅ Understand vanishing gradient problem
- ✅ Implement attention mechanism
- ✅ Build seq2seq models
- ✅ Generate text with sampling strategies
- ✅ Handle variable-length sequences
- ✅ Evaluate sequence models

## 💡 Pro Tips

1. **Start Simple**: Master vanilla RNN before LSTM
2. **Visualize**: Plot hidden states and attention weights
3. **Experiment**: Change hyperparameters and observe effects
4. **Debug**: Print shapes frequently
5. **Patience**: RNNs take longer to train than CNNs
6. **GPU**: Use for tutorials 08-10 (optional but helpful)
7. **Save Models**: Always save successful models
8. **Read Errors**: PyTorch errors are usually helpful

## 🎨 Visualization Tips

Each tutorial includes visualizations:
- Hidden state evolution
- Attention weight heatmaps
- Training loss curves
- Generated text samples
- Prediction vs ground truth

## 📄 File Structure

```
pytorch_rnn_tutorial/
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── rnn_utils.py                # Shared utilities
├── 01_sequence_basics.py       # Introduction
├── 02_text_preprocessing.py    # Text handling
├── 03_time_series_basics.py   # Time series
├── 04_simple_rnn.py           # Vanilla RNN
├── 05_lstm_sentiment.py       # LSTM basics
├── 06_gru_time_series.py      # GRU application
├── 07_bidirectional_rnn.py    # Bidirectional
├── 08_seq2seq_basic.py        # Seq2Seq
├── 09_attention_mechanism.py  # Attention
└── 10_text_generation.py      # Generation
```

## 🤝 Acknowledgments

Inspired by:
- PyTorch official tutorials
- Stanford CS224N
- Understanding LSTM Networks (colah's blog)
- The Unreasonable Effectiveness of RNNs (Karpathy)

## 📝 Notes

- All code is heavily commented for learning
- GPU optional (CPU sufficient for learning)
- Models intentionally kept simple for clarity
- Focus on understanding over performance

## 🎉 Conclusion

By completing this tutorial series, you'll have:
- Deep understanding of RNN architectures
- Hands-on experience with sequence modeling
- Practical skills for NLP and time series
- Foundation for advanced topics (Transformers, BERT)

**Happy Learning! 🚀**

---

*Last Updated: November 2025*
*For issues or questions, refer to PyTorch documentation and forums.*
