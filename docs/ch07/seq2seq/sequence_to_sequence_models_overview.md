# Sequence-to-Sequence Models: Encoder-Decoder Architectures

A comprehensive implementation of sequence-to-sequence models with encoder-decoder architectures in PyTorch.

## 📚 Overview

This repository contains a complete implementation of seq2seq models for tasks like machine translation, text summarization, and more. It includes:

- **Multiple Encoder Architectures**: LSTM, GRU, Bidirectional, and Convolutional encoders
- **Attention Mechanisms**: Bahdanau (additive) and Luong (multiplicative) attention
- **Advanced Decoding**: Greedy decoding, beam search, and teacher forcing
- **Training Pipeline**: Complete training loop with checkpointing and evaluation
- **Data Preprocessing**: Tokenization, vocabulary building, and data loading utilities
- **Practical Example**: End-to-end English to French translation

## 🗂️ Repository Structure

```
seq2seq_project/
├── encoder.py              # Encoder implementations (LSTM, GRU, Conv)
├── decoder.py              # Decoder implementations with attention
├── seq2seq_model.py        # Complete seq2seq models
├── train.py                # Training script and utilities
├── inference.py            # Inference and evaluation utilities
├── data_preprocessing.py   # Data preprocessing tools
├── example_translation.py  # Complete working example
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install torch numpy matplotlib seaborn
```

### 2. Run the Example

```python
python example_translation.py
```

This will:
- Create a small English-French parallel corpus
- Build vocabularies
- Train a seq2seq model with attention
- Evaluate on test data
- Show example translations

## 📖 Core Components

### Encoder

The encoder processes the input sequence and produces hidden states:

```python
from encoder import BasicEncoder

encoder = BasicEncoder(
    input_size=vocab_size,
    embedding_dim=256,
    hidden_size=512,
    num_layers=2,
    dropout=0.1,
    bidirectional=True,
    rnn_type='LSTM'  # or 'GRU'
)
```

**Features:**
- LSTM or GRU recurrent layers
- Bidirectional encoding
- Variable-length sequence handling
- Dropout regularization

### Decoder

The decoder generates the output sequence token by token:

```python
from decoder import AttentionDecoder

decoder = AttentionDecoder(
    output_size=vocab_size,
    embedding_dim=256,
    hidden_size=512,
    encoder_hidden_size=512,
    num_layers=2,
    dropout=0.1,
    rnn_type='LSTM'
)
```

**Features:**
- Attention mechanism (Bahdanau or Luong)
- Teacher forcing support
- Context vector integration
- Multiple decoding strategies

### Complete Seq2Seq Model

```python
from seq2seq_model import Seq2SeqAttention

model = Seq2SeqAttention(
    encoder=encoder,
    decoder=decoder,
    device=device,
    pad_idx=0
)
```

**Capabilities:**
- Teacher forcing during training
- Greedy decoding for inference
- Beam search for better quality
- Attention weight visualization

## 🎯 Usage Examples

### Training

```python
from train import Seq2SeqTrainer

# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    pad_idx=0,
    clip=1.0
)

# Train
train_losses, val_losses = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=20,
    teacher_forcing_ratio=0.5
)
```

### Inference

```python
from inference import Seq2SeqInference

# Create inference object
inference = Seq2SeqInference(
    model=model,
    src_vocab=src_vocab,
    trg_vocab=trg_vocab,
    device=device
)

# Translate
translation, attention = inference.greedy_decode("Hello, how are you?")
print(translation)

# Or use beam search
translation, score = inference.beam_search_decode(
    "Hello, how are you?",
    beam_width=5
)
```

### Data Preprocessing

```python
from data_preprocessing import Vocabulary, Tokenizer, ParallelDataset

# Create tokenizer
tokenizer = Tokenizer(lower=True, remove_punct=False)

# Build vocabulary
vocab = Vocabulary(max_size=10000, min_freq=2)
vocab.build_vocab(texts, tokenizer.tokenize)

# Create dataset
dataset = ParallelDataset(
    src_texts=src_texts,
    trg_texts=trg_texts,
    src_vocab=src_vocab,
    trg_vocab=trg_vocab,
    src_tokenizer=tokenizer,
    trg_tokenizer=tokenizer
)
```

## 🔬 Model Architectures

### Basic Seq2Seq

```
Input → Embedding → Encoder → Hidden State → Decoder → Output
```

### Seq2Seq with Attention

```
Input → Embedding → Encoder → Hidden States
                                    ↓
                            Attention Mechanism
                                    ↓
Output ← Decoder ← Context Vector + Previous Output
```

### Attention Mechanisms

**Bahdanau (Additive) Attention:**
```
score(s_t, h_i) = v^T * tanh(W_1 * s_t + W_2 * h_i)
```

**Luong (Multiplicative) Attention:**
```
score(s_t, h_i) = s_t^T * W * h_i  (general)
score(s_t, h_i) = s_t^T * h_i      (dot)
```

## 📊 Features

### Training Features
- ✅ Teacher forcing with schedule
- ✅ Gradient clipping
- ✅ Checkpoint saving
- ✅ Early stopping support
- ✅ Learning rate scheduling
- ✅ Validation monitoring

### Inference Features
- ✅ Greedy decoding
- ✅ Beam search
- ✅ Length penalty
- ✅ Attention visualization
- ✅ BLEU score evaluation
- ✅ Batch translation

### Data Features
- ✅ Custom tokenization
- ✅ Vocabulary building
- ✅ Padding and batching
- ✅ Variable-length sequences
- ✅ Special token handling
- ✅ Save/load vocabularies

## 🎓 Key Concepts

### Teacher Forcing
During training, the decoder receives the ground truth previous token instead of its own prediction. This speeds up training but can cause exposure bias.

```python
# With teacher forcing (training)
decoder_input = target[:, t]  # Use ground truth

# Without teacher forcing (inference)
decoder_input = predicted_token  # Use prediction
```

### Attention Mechanism
Attention allows the decoder to focus on different parts of the input sequence at each decoding step:

1. Calculate attention scores between decoder state and all encoder states
2. Apply softmax to get attention weights
3. Create context vector as weighted sum of encoder states
4. Use context vector to inform decoder's prediction

### Beam Search
Instead of taking the single best token at each step (greedy), beam search maintains top-k hypotheses:

```python
translation, score = model.beam_search(
    src_sequence,
    beam_width=5,
    length_penalty=0.6
)
```

## 📈 Performance Tips

1. **Use bidirectional encoder** for better context understanding
2. **Apply dropout** to prevent overfitting
3. **Clip gradients** to avoid exploding gradients
4. **Use attention** for longer sequences
5. **Implement beam search** for better quality outputs
6. **Schedule teacher forcing** decay over epochs

## 🔧 Hyperparameter Tuning

Typical hyperparameters:

```python
# Model architecture
embedding_dim = 256-512
hidden_size = 512-1024
num_layers = 2-4
dropout = 0.1-0.5

# Training
batch_size = 32-128
learning_rate = 0.0001-0.001
teacher_forcing_ratio = 0.5
gradient_clip = 1.0
```

## 📝 Applications

This implementation can be used for:

- **Machine Translation**: Translate between languages
- **Text Summarization**: Generate summaries from documents
- **Dialogue Systems**: Generate responses in conversations
- **Code Generation**: Generate code from natural language
- **Image Captioning**: Generate captions from images (with CNN encoder)
- **Speech Recognition**: Convert speech to text

## 🐛 Common Issues and Solutions

### Out of Memory
- Reduce batch size
- Reduce hidden size or embedding dimension
- Use gradient accumulation
- Enable mixed precision training

### Poor Translation Quality
- Increase model capacity (hidden size, layers)
- Use attention mechanism
- Implement beam search
- Add more training data
- Use pretrained embeddings

### Slow Convergence
- Increase learning rate
- Use Adam optimizer
- Implement learning rate scheduling
- Increase teacher forcing ratio initially

## 📚 References

1. **Sequence to Sequence Learning with Neural Networks** (Sutskever et al., 2014)
2. **Neural Machine Translation by Jointly Learning to Align and Translate** (Bahdanau et al., 2015)
3. **Effective Approaches to Attention-based Neural Machine Translation** (Luong et al., 2015)
4. **Massive Exploration of Neural Machine Translation Architectures** (Britz et al., 2017)

## 🤝 Contributing

Feel free to extend this implementation with:
- Transformer architectures
- More attention variants
- Different RNN types
- Advanced beam search variants
- Multi-head attention
- Layer normalization

## 📄 License

This code is provided for educational purposes. Feel free to use and modify it for your projects.

## 🙏 Acknowledgments

This implementation follows best practices from PyTorch and modern NLP research. It's designed to be educational, modular, and easy to extend.
