"""
PyTorch LSTM and GRU Practical Examples
========================================

This module demonstrates practical usage of LSTM and GRU using PyTorch's
built-in layers for real-world tasks:
1. Text generation (character-level)
2. Time series forecasting
3. Sequence classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# Example 1: Character-Level Text Generation
# =============================================================================

class CharLSTM(nn.Module):
    """Character-level LSTM for text generation."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2):
        super(CharLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        # x shape: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        
        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
            
        out = self.fc(out)  # (batch, seq_len, vocab_size)
        return out, hidden


class CharGRU(nn.Module):
    """Character-level GRU for text generation."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=2):
        super(CharGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers,
                         batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        
        if hidden is None:
            out, hidden = self.gru(x)
        else:
            out, hidden = self.gru(x, hidden)
            
        out = self.fc(out)
        return out, hidden


def text_generation_demo():
    """Demonstrate text generation with LSTM and GRU."""
    print("=" * 70)
    print("Example 1: Character-Level Text Generation")
    print("=" * 70)
    
    # Sample text
    text = "hello world! this is a simple example of text generation using lstm and gru."
    
    # Create character mappings
    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    vocab_size = len(chars)
    
    print(f"\nVocabulary size: {vocab_size}")
    print(f"Text length: {len(text)}")
    print(f"Characters: {chars}")
    
    # Prepare training data
    seq_length = 10
    sequences = []
    targets = []
    
    for i in range(len(text) - seq_length):
        seq = text[i:i+seq_length]
        target = text[i+1:i+seq_length+1]
        sequences.append([char_to_idx[ch] for ch in seq])
        targets.append([char_to_idx[ch] for ch in target])
    
    # Create models
    embedding_dim = 16
    hidden_size = 32
    
    lstm_model = CharLSTM(vocab_size, embedding_dim, hidden_size)
    gru_model = CharGRU(vocab_size, embedding_dim, hidden_size)
    
    # Count parameters
    lstm_params = sum(p.numel() for p in lstm_model.parameters())
    gru_params = sum(p.numel() for p in gru_model.parameters())
    
    print(f"\nLSTM parameters: {lstm_params:,}")
    print(f"GRU parameters:  {gru_params:,}")
    print(f"Difference:      {lstm_params - gru_params:,} ({(1-gru_params/lstm_params)*100:.1f}% fewer)")
    
    # Show model architectures
    print("\nLSTM Architecture:")
    print(lstm_model)
    print("\nGRU Architecture:")
    print(gru_model)


# =============================================================================
# Example 2: Time Series Forecasting
# =============================================================================

class TimeSeriesDataset(Dataset):
    """Dataset for time series prediction."""
    
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+self.seq_length]
        return x.unsqueeze(-1), y.unsqueeze(-1)


class TimeSeriesLSTM(nn.Module):
    """LSTM for time series forecasting."""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(TimeSeriesLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last output
        return out


class TimeSeriesGRU(nn.Module):
    """GRU for time series forecasting."""
    
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(TimeSeriesGRU, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                         batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out


def time_series_demo():
    """Demonstrate time series forecasting."""
    print("\n" + "=" * 70)
    print("Example 2: Time Series Forecasting")
    print("=" * 70)
    
    # Generate synthetic time series (sine wave with noise)
    t = np.linspace(0, 100, 1000)
    data = np.sin(0.1 * t) + 0.1 * np.random.randn(1000)
    
    # Split data
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    print(f"\nTraining samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Create datasets
    seq_length = 20
    train_dataset = TimeSeriesDataset(train_data, seq_length)
    test_dataset = TimeSeriesDataset(test_data, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # Create models
    lstm_model = TimeSeriesLSTM(hidden_size=50)
    gru_model = TimeSeriesGRU(hidden_size=50)
    
    print(f"\nSequence length: {seq_length}")
    print(f"LSTM hidden size: {lstm_model.lstm.hidden_size}")
    print(f"GRU hidden size: {gru_model.gru.hidden_size}")
    
    # Training setup (demonstration only - not actual training)
    criterion = nn.MSELoss()
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)
    
    print("\nModels created and ready for training!")
    print("In practice, you would train for several epochs...")
    
    # Visualize sample prediction (untrained models)
    lstm_model.eval()
    gru_model.eval()
    
    with torch.no_grad():
        sample_x, sample_y = next(iter(test_loader))
        lstm_pred = lstm_model(sample_x)
        gru_pred = gru_model(sample_x)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(t[:200], data[:200], 'b-', linewidth=1.5, label='Actual')
    plt.title('Time Series Data Sample')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    indices = range(len(sample_y))
    plt.plot(indices, sample_y.numpy(), 'go-', label='Actual', markersize=4)
    plt.plot(indices, lstm_pred.numpy(), 'rs-', label='LSTM (untrained)', 
             markersize=4, alpha=0.7)
    plt.plot(indices, gru_pred.numpy(), 'b^-', label='GRU (untrained)', 
             markersize=4, alpha=0.7)
    plt.title('Predictions (Untrained Models)')
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/claude/lstm_gru_module/timeseries.png', dpi=150, bbox_inches='tight')
    print("\n✓ Time series plot saved as 'timeseries.png'")
    plt.close()


# =============================================================================
# Example 3: Sequence Classification
# =============================================================================

class SequenceClassifier(nn.Module):
    """LSTM/GRU for sequence classification."""
    
    def __init__(self, input_size, hidden_size, num_classes, model_type='lstm'):
        super(SequenceClassifier, self).__init__()
        self.model_type = model_type
        
        if model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
            
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        if self.model_type == 'lstm':
            _, (hidden, _) = self.rnn(x)
        else:
            _, hidden = self.rnn(x)
            
        # Use final hidden state for classification
        out = self.fc(hidden.squeeze(0))
        return out


def sequence_classification_demo():
    """Demonstrate sequence classification."""
    print("\n" + "=" * 70)
    print("Example 3: Sequence Classification")
    print("=" * 70)
    
    # Generate synthetic sequence data
    # Class 0: increasing sequences, Class 1: decreasing sequences
    def generate_sequence(seq_len, class_label):
        if class_label == 0:
            return np.cumsum(np.random.randn(seq_len, 1) * 0.1 + 0.1, axis=0)
        else:
            return np.cumsum(np.random.randn(seq_len, 1) * 0.1 - 0.1, axis=0)
    
    # Create dataset
    n_samples = 100
    seq_len = 20
    
    X_train = []
    y_train = []
    
    for i in range(n_samples):
        label = i % 2
        seq = generate_sequence(seq_len, label)
        X_train.append(seq)
        y_train.append(label)
    
    print(f"\nDataset size: {n_samples}")
    print(f"Sequence length: {seq_len}")
    print(f"Number of classes: 2 (increasing/decreasing)")
    
    # Create models
    input_size = 1
    hidden_size = 32
    num_classes = 2
    
    lstm_classifier = SequenceClassifier(input_size, hidden_size, num_classes, 'lstm')
    gru_classifier = SequenceClassifier(input_size, hidden_size, num_classes, 'gru')
    
    print(f"\nLSTM Classifier parameters: {sum(p.numel() for p in lstm_classifier.parameters()):,}")
    print(f"GRU Classifier parameters:  {sum(p.numel() for p in gru_classifier.parameters()):,}")
    
    # Visualize sample sequences
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for i in range(5):
        plt.plot(X_train[i*2], alpha=0.7, label=f'Sample {i+1}')
    plt.title('Class 0: Increasing Sequences')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for i in range(5):
        plt.plot(X_train[i*2+1], alpha=0.7, label=f'Sample {i+1}')
    plt.title('Class 1: Decreasing Sequences')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('/home/claude/lstm_gru_module/classification.png', dpi=150, bbox_inches='tight')
    print("\n✓ Classification plot saved as 'classification.png'")
    plt.close()


# =============================================================================
# PyTorch LSTM/GRU Usage Tips
# =============================================================================

def pytorch_tips():
    """Display PyTorch usage tips."""
    print("\n" + "=" * 70)
    print("PyTorch LSTM/GRU Usage Tips")
    print("=" * 70)
    
    tips = """
1. BATCH_FIRST Parameter:
   - batch_first=True: input shape is (batch, seq, features)
   - batch_first=False: input shape is (seq, batch, features)
   - Recommended: Use batch_first=True for clarity

2. Hidden State Initialization:
   - LSTM returns (output, (hidden, cell))
   - GRU returns (output, hidden)
   - If not provided, initialized to zeros

3. Dropout:
   - dropout parameter applies dropout between RNN layers
   - Only applies if num_layers > 1
   - Don't forget to use model.eval() during inference

4. Bidirectional:
   - bidirectional=True doubles the hidden size
   - Output shape: (batch, seq, hidden_size * 2)
   - Useful for tasks where future context is available

5. Packing Sequences:
   - Use pack_padded_sequence for variable-length sequences
   - Improves efficiency with padded sequences
   - Example:
     packed = pack_padded_sequence(x, lengths, batch_first=True)
     output, hidden = lstm(packed)
     output, _ = pad_packed_sequence(output, batch_first=True)

6. Common Pitfalls:
   - Forgetting to detach hidden states between batches
   - Not matching batch_first parameter across operations
   - Using wrong hidden state dimension for stacked layers

7. Performance Tips:
   - Use GPU: model.cuda() and data.cuda()
   - Use DataLoader with num_workers > 0
   - Consider using torch.compile() for PyTorch 2.0+
"""
    print(tips)


def main():
    """Run all PyTorch examples."""
    print("\n" + "=" * 70)
    print("PyTorch LSTM & GRU Practical Examples")
    print("=" * 70)
    
    # Run examples
    text_generation_demo()
    time_series_demo()
    sequence_classification_demo()
    pytorch_tips()
    
    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - timeseries.png")
    print("  - classification.png")


if __name__ == "__main__":
    main()
