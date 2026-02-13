"""
rnn_utils.py
============
Comprehensive utility module for RNN tutorials

Provides shared functions for:
- Argument parsing
- Data loading and preprocessing
- RNN model architectures
- Training and evaluation
- Visualization
- Text processing utilities

Author: PyTorch RNN Tutorial
Date: November 2025
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader

# =============================================================================
# Configuration and Setup
# =============================================================================

def parse_args():
    """Parse command line arguments for RNN training"""
    parser = argparse.ArgumentParser(description='PyTorch RNN Training')
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Sequence parameters
    parser.add_argument('--sequence-length', type=int, default=50)
    parser.add_argument('--embedding-dim', type=int, default=100)
    
    # System configuration
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--log-interval', type=int, default=100)
    
    # Model persistence
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--path', type=str, default='./model.pth')
    parser.add_argument('--eval-only', action='store_true')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    
    return args

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# RNN Model Architectures
# =============================================================================

class SimpleRNN(nn.Module):
    """Basic RNN for sequence classification"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # x: (batch, seq_len, input_size)
        out, hidden = self.rnn(x, hidden)
        # Take the last output
        out = self.fc(out[:, -1, :])
        return out, hidden

class LSTMClassifier(nn.Module):
    """LSTM for sequence classification"""
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, 
                 num_layers=1, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Use last hidden state
        out = self.dropout(hidden[-1])
        out = self.fc(out)
        return out

class GRUPredictor(nn.Module):
    """GRU for sequence prediction (e.g., time series)"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.2):
        super(GRUPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        gru_out, hidden = self.gru(x)
        # Use last output
        out = self.fc(gru_out[:, -1, :])
        return out

class BiLSTM(nn.Module):
    """Bidirectional LSTM"""
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size,
                 num_layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                           batch_first=True, bidirectional=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        # *2 because bidirectional
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # Concatenate final forward and backward hidden states
        hidden_fwd = hidden[-2]
        hidden_bwd = hidden[-1]
        hidden_cat = torch.cat([hidden_fwd, hidden_bwd], dim=1)
        out = self.dropout(hidden_cat)
        out = self.fc(out)
        return out

# =============================================================================
# Training and Evaluation
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, clip=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        if clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy for classification
        if output.dim() > 1 and output.size(1) > 1:
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            total_loss += loss.item()
            
            if output.dim() > 1 and output.size(1) > 1:
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy

# =============================================================================
# Text Processing Utilities
# =============================================================================

class Vocabulary:
    """Build and manage vocabulary for text data"""
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}
        self.word_count = {}
        self.n_words = 4
    
    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.word_count[word] = 1
            self.n_words += 1
        else:
            self.word_count[word] += 1
    
    def __len__(self):
        return self.n_words

def tokenize(text):
    """Simple tokenizer"""
    return text.lower().split()

def text_to_sequence(text, vocab, max_length=None):
    """Convert text to sequence of indices"""
    tokens = tokenize(text)
    sequence = [vocab.word2idx.get(word, vocab.word2idx['<UNK>']) for word in tokens]
    
    if max_length:
        if len(sequence) < max_length:
            sequence += [vocab.word2idx['<PAD>']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
    
    return sequence

# =============================================================================
# Visualization
# =============================================================================

def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2 if train_accs else 1, figsize=(15 if train_accs else 10, 5))
    
    if train_accs is None:
        axes = [axes]
    
    # Loss plot
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot (if provided)
    if train_accs:
        axes[1].plot(train_accs, label='Train Acc')
        axes[1].plot(val_accs, label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    return fig

# =============================================================================
# Model Persistence
# =============================================================================

def save_model(model, path, vocab=None):
    """Save model and optionally vocabulary"""
    save_dict = {'model_state_dict': model.state_dict()}
    if vocab:
        save_dict['vocab'] = vocab
    torch.save(save_dict, path)
    print(f"Model saved to {path}")

def load_model(model_class, path, device, **model_kwargs):
    """Load model"""
    checkpoint = torch.load(path, map_location=device)
    model = model_class(**model_kwargs).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    vocab = checkpoint.get('vocab', None)
    print(f"Model loaded from {path}")
    return model, vocab

# =============================================================================
# Data Generation Utilities
# =============================================================================

def generate_sine_wave(seq_length, num_samples):
    """Generate sine wave data for time series"""
    X = []
    y = []
    for _ in range(num_samples):
        start = np.random.rand() * 2 * np.pi
        x = np.sin(np.linspace(start, start + 2 * np.pi, seq_length + 1))
        X.append(x[:-1])
        y.append(x[1:])
    return np.array(X), np.array(y)

__version__ = "1.0.0"
__author__ = "PyTorch RNN Tutorial"
