"""
Advanced LSTM/GRU Techniques
=============================

This module covers advanced architectures and techniques:
1. Bidirectional LSTM/GRU
2. Stacked (Multi-layer) Networks
3. Attention Mechanisms
4. Encoder-Decoder Architectures
5. Real-world Application Examples
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# 1. Bidirectional LSTM/GRU
# =============================================================================

class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM for sequence processing."""
    
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BidirectionalLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=True  # Key parameter
        )
        
        # Output layer needs to account for both directions
        self.fc = nn.Linear(hidden_size * 2, input_size)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, (hidden, cell) = self.lstm(x)
        # output: (batch, seq_len, hidden_size * 2)
        
        out = self.fc(output)
        return out, hidden


def bidirectional_demo():
    """Demonstrate bidirectional processing."""
    print("=" * 70)
    print("1. Bidirectional LSTM/GRU")
    print("=" * 70)
    
    print("\nConcept:")
    print("  Forward pass:  processes sequence left-to-right")
    print("  Backward pass: processes sequence right-to-left")
    print("  Output: concatenation of both directions")
    
    print("\nAdvantages:")
    print("  ✓ Access to future context")
    print("  ✓ Better for tasks where full sequence is available")
    print("  ✓ Improved accuracy on many NLP tasks")
    
    print("\nDisadvantages:")
    print("  ✗ 2× slower training")
    print("  ✗ 2× more memory")
    print("  ✗ Cannot be used for online/streaming predictions")
    
    # Create model
    input_size = 10
    hidden_size = 20
    seq_len = 15
    batch_size = 3
    
    model = BidirectionalLSTM(input_size, hidden_size)
    
    # Sample input
    x = torch.randn(batch_size, seq_len, input_size)
    output, hidden = model(x)
    
    print(f"\nExample:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Hidden shape: {hidden.shape}")
    print(f"  Note: hidden has shape (num_layers * 2, batch, hidden_size)")
    
    # Visualize bidirectional flow
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Forward direction
    forward_x = np.arange(seq_len)
    forward_y = np.ones(seq_len) * 2
    ax.plot(forward_x, forward_y, 'b-o', linewidth=3, markersize=10, label='Forward →')
    
    # Backward direction
    backward_y = np.ones(seq_len) * 1
    ax.plot(forward_x, backward_y, 'r-o', linewidth=3, markersize=10, label='← Backward')
    
    # Add arrows
    for i in range(seq_len - 1):
        ax.annotate('', xy=(i+1, 2), xytext=(i, 2),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        ax.annotate('', xy=(i, 1), xytext=(i+1, 1),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Time steps
    for i in range(seq_len):
        ax.text(i, 0.5, f't={i}', ha='center', fontsize=10)
    
    ax.set_xlim(-0.5, seq_len - 0.5)
    ax.set_ylim(0, 3)
    ax.set_xlabel('Time Steps', fontsize=12)
    ax.set_title('Bidirectional LSTM: Forward and Backward Processing', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(['Backward', 'Forward'])
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('/home/claude/lstm_gru_module/bidirectional.png', dpi=150, bbox_inches='tight')
    print("\n✓ Bidirectional diagram saved as 'bidirectional.png'")
    plt.close()


# =============================================================================
# 2. Stacked (Multi-layer) LSTM/GRU
# =============================================================================

class StackedLSTM(nn.Module):
    """Multi-layer stacked LSTM."""
    
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(StackedLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        out = self.fc(output)
        return out


def stacked_demo():
    """Demonstrate stacked architectures."""
    print("\n" + "=" * 70)
    print("2. Stacked (Multi-layer) LSTM/GRU")
    print("=" * 70)
    
    print("\nConcept:")
    print("  Stack multiple LSTM/GRU layers vertically")
    print("  Output of layer i becomes input of layer i+1")
    print("  Enables learning hierarchical representations")
    
    print("\nArchitecture:")
    print("  Layer 1: Low-level features")
    print("  Layer 2: Mid-level patterns")
    print("  Layer 3+: High-level abstractions")
    
    # Compare different depths
    input_size = 50
    hidden_size = 100
    configs = [
        (1, "Single Layer"),
        (2, "2 Layers"),
        (3, "3 Layers"),
        (4, "4 Layers")
    ]
    
    print("\nParameter Count Comparison:")
    for num_layers, name in configs:
        model = StackedLSTM(input_size, hidden_size, num_layers)
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} parameters")
    
    print("\nGuidelines:")
    print("  • 1 layer: Simple tasks, small datasets")
    print("  • 2 layers: Most common choice (good balance)")
    print("  • 3-4 layers: Complex tasks, large datasets")
    print("  • 5+ layers: Rare, may need residual connections")
    
    print("\nTips:")
    print("  • Use dropout between layers (0.2-0.5)")
    print("  • Consider residual connections for deep models")
    print("  • More layers ≠ always better (overfitting risk)")


# =============================================================================
# 3. Attention Mechanism
# =============================================================================

class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism."""
    
    def __init__(self, input_size, hidden_size):
        super(AttentionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Attention components
        self.attention = nn.Linear(hidden_size, 1)
        
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        
        # Compute attention scores
        attention_scores = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden_size)
        
        # Output
        out = self.fc(context)
        
        return out, attention_weights


def attention_demo():
    """Demonstrate attention mechanism."""
    print("\n" + "=" * 70)
    print("3. Attention Mechanism")
    print("=" * 70)
    
    print("\nConcept:")
    print("  Learn to focus on relevant parts of the sequence")
    print("  Compute weighted sum of hidden states")
    print("  Weights determined by learned attention scores")
    
    print("\nFormulation:")
    print("  1. Score each hidden state: e_t = f(h_t)")
    print("  2. Normalize scores: α_t = softmax(e)")
    print("  3. Compute context: c = Σ α_t · h_t")
    
    # Create model and visualize attention
    model = AttentionLSTM(input_size=10, hidden_size=20)
    model.eval()
    
    # Sample sequence
    seq_len = 15
    x = torch.randn(1, seq_len, 10)
    
    with torch.no_grad():
        output, attention_weights = model(x)
    
    # Visualize attention weights
    weights = attention_weights.squeeze().numpy()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.bar(range(seq_len), weights, color='steelblue', alpha=0.8, edgecolor='black')
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Attention Weight', fontsize=12)
    plt.title('Attention Weights Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 1, 2)
    plt.imshow(weights.reshape(1, -1), aspect='auto', cmap='YlOrRd')
    plt.colorbar(label='Weight')
    plt.xlabel('Time Step', fontsize=12)
    plt.title('Attention Heatmap', fontsize=14, fontweight='bold')
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('/home/claude/lstm_gru_module/attention.png', dpi=150, bbox_inches='tight')
    print("\n✓ Attention visualization saved as 'attention.png'")
    plt.close()
    
    print("\nApplications:")
    print("  • Machine translation (align source-target words)")
    print("  • Text summarization (identify important sentences)")
    print("  • Question answering (focus on relevant context)")
    print("  • Image captioning (attend to image regions)")


# =============================================================================
# 4. Encoder-Decoder Architecture
# =============================================================================

class EncoderLSTM(nn.Module):
    """LSTM Encoder."""
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell


class DecoderLSTM(nn.Module):
    """LSTM Decoder."""
    def __init__(self, input_size, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.lstm(x, (hidden, cell))
        output = self.fc(output)
        return output, hidden, cell


class Seq2Seq(nn.Module):
    """Sequence-to-Sequence model."""
    def __init__(self, input_size, hidden_size, output_size):
        super(Seq2Seq, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size)
        self.decoder = DecoderLSTM(output_size, hidden_size, output_size)
        
    def forward(self, src, tgt):
        # Encode
        hidden, cell = self.encoder(src)
        
        # Decode
        output, _, _ = self.decoder(tgt, hidden, cell)
        return output


def encoder_decoder_demo():
    """Demonstrate encoder-decoder architecture."""
    print("\n" + "=" * 70)
    print("4. Encoder-Decoder Architecture (Seq2Seq)")
    print("=" * 70)
    
    print("\nArchitecture:")
    print("  [Encoder] Input sequence → Fixed-size context vector")
    print("  [Decoder] Context vector → Output sequence")
    
    print("\nApplications:")
    print("  • Machine Translation: English → French")
    print("  • Text Summarization: Long text → Summary")
    print("  • Dialogue Systems: Question → Answer")
    print("  • Code Generation: Description → Code")
    
    print("\nKey Concepts:")
    print("  1. Encoder compresses input into context")
    print("  2. Decoder generates output from context")
    print("  3. Often use attention to improve performance")
    
    # Create model
    model = Seq2Seq(input_size=20, hidden_size=50, output_size=20)
    
    # Example
    src = torch.randn(2, 10, 20)  # batch=2, src_len=10
    tgt = torch.randn(2, 12, 20)  # batch=2, tgt_len=12
    
    output = model(src, tgt)
    print(f"\nExample:")
    print(f"  Source shape: {src.shape}")
    print(f"  Target shape: {tgt.shape}")
    print(f"  Output shape: {output.shape}")
    
    # Visualize architecture
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Encoder
    encoder_x = [0, 1, 2, 3, 4]
    encoder_y = [1] * 5
    ax.plot(encoder_x, encoder_y, 'bo-', markersize=15, linewidth=3, label='Encoder')
    for i in range(len(encoder_x) - 1):
        ax.annotate('', xy=(encoder_x[i+1], encoder_y[i+1]), 
                   xytext=(encoder_x[i], encoder_y[i]),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    # Context vector
    ax.plot([5.5], [1], 'gs', markersize=20, label='Context')
    ax.annotate('', xy=(5.5, 1), xytext=(4, 1),
               arrowprops=dict(arrowstyle='->', color='green', lw=3))
    
    # Decoder
    decoder_x = [7, 8, 9, 10, 11]
    decoder_y = [1] * 5
    ax.plot(decoder_x, decoder_y, 'ro-', markersize=15, linewidth=3, label='Decoder')
    for i in range(len(decoder_x) - 1):
        ax.annotate('', xy=(decoder_x[i+1], decoder_y[i+1]), 
                   xytext=(decoder_x[i], decoder_y[i]),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    ax.annotate('', xy=(7, 1), xytext=(5.5, 1),
               arrowprops=dict(arrowstyle='->', color='green', lw=3))
    
    # Labels
    for i, x in enumerate(encoder_x):
        ax.text(x, 0.7, f'x{i+1}', ha='center', fontsize=10)
    ax.text(5.5, 0.7, 'h,c', ha='center', fontsize=10)
    for i, x in enumerate(decoder_x):
        ax.text(x, 0.7, f'y{i+1}', ha='center', fontsize=10)
    
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(0.5, 1.5)
    ax.set_title('Encoder-Decoder (Seq2Seq) Architecture', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/claude/lstm_gru_module/seq2seq.png', dpi=150, bbox_inches='tight')
    print("\n✓ Seq2Seq diagram saved as 'seq2seq.png'")
    plt.close()


# =============================================================================
# 5. Real-world Applications
# =============================================================================

def real_world_applications():
    """Showcase real-world LSTM/GRU applications."""
    print("\n" + "=" * 70)
    print("5. Real-World Applications")
    print("=" * 70)
    
    applications = """
Natural Language Processing:
  
  1. Machine Translation
     Model: Encoder-Decoder with Attention
     Example: Google Translate
     Architecture: Bidirectional LSTM encoder, LSTM decoder
     
  2. Text Generation
     Model: Character/Word-level LSTM
     Example: GPT predecessors, text completion
     Architecture: Stacked LSTM (2-3 layers)
     
  3. Named Entity Recognition
     Model: Bidirectional LSTM-CRF
     Example: Identifying people, places, organizations
     Architecture: BiLSTM + Conditional Random Field
     
  4. Sentiment Analysis
     Model: LSTM/GRU Classifier
     Example: Analyzing product reviews
     Architecture: 1-2 layer LSTM with final classification layer

Speech & Audio:
  
  5. Speech Recognition
     Model: Deep LSTM
     Example: Siri, Google Assistant
     Architecture: 5+ layer bidirectional LSTM
     
  6. Music Generation
     Model: LSTM
     Example: Composing melodies
     Architecture: Character-level LSTM on MIDI

Computer Vision:
  
  7. Video Analysis
     Model: CNN-LSTM
     Example: Action recognition
     Architecture: CNN for frames + LSTM for temporal
     
  8. Image Captioning
     Model: CNN-Encoder + LSTM-Decoder
     Example: Describing images
     Architecture: CNN encoder + LSTM decoder with attention

Time Series:
  
  9. Stock Price Prediction
     Model: LSTM/GRU
     Example: Financial forecasting
     Architecture: 2-layer LSTM with dropout
     
 10. Weather Forecasting
     Model: Stacked LSTM
     Example: Temperature prediction
     Architecture: 3-layer LSTM

Other Domains:
  
 11. Protein Structure Prediction
     Model: Bidirectional LSTM
     Example: AlphaFold predecessors
     Architecture: Deep BiLSTM
     
 12. Anomaly Detection
     Model: Autoencoder LSTM
     Example: Network intrusion detection
     Architecture: LSTM encoder-decoder
"""
    print(applications)


# =============================================================================
# Advanced Tips and Tricks
# =============================================================================

def advanced_tips():
    """Share advanced optimization tips."""
    print("\n" + "=" * 70)
    print("Advanced Tips & Tricks")
    print("=" * 70)
    
    tips = """
1. Gradient Clipping
   • Essential for LSTM/GRU training
   • Prevents exploding gradients
   • Use: torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)

2. Layer Normalization
   • Apply normalization within each layer
   • Stabilizes training
   • Better than batch normalization for RNNs

3. Residual Connections
   • Add skip connections for deep networks
   • h_out = h_in + LSTM(h_in)
   • Improves gradient flow

4. Scheduled Sampling
   • During training, sometimes use predicted outputs
   • Reduces train/test mismatch
   • Gradually increase probability during training

5. Beam Search Decoding
   • For generation tasks
   • Keep top-k candidates at each step
   • Better than greedy decoding

6. Teacher Forcing
   • Use ground truth as decoder input
   • Fast training but can cause exposure bias
   • Consider scheduled sampling as alternative

7. Variational Dropout
   • Same dropout mask across time steps
   • More effective than standard dropout
   • Implemented in PyTorch with variational_dropout

8. Weight Tying
   • Share embedding and output weights
   • Reduces parameters in language models
   • Often improves performance

9. Learning Rate Warmup
   • Gradually increase learning rate at start
   • Prevents early instability
   • Especially useful for deep networks

10. Mixed Precision Training
    • Use FP16 for faster training
    • Automatic with torch.cuda.amp
    • Significant speedup with minimal accuracy loss
"""
    print(tips)


def main():
    """Run all advanced demonstrations."""
    print("\n" + "=" * 70)
    print("Advanced LSTM/GRU Techniques")
    print("=" * 70)
    
    bidirectional_demo()
    stacked_demo()
    attention_demo()
    encoder_decoder_demo()
    real_world_applications()
    advanced_tips()
    
    print("\n" + "=" * 70)
    print("Advanced demonstrations complete!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - bidirectional.png")
    print("  - attention.png")
    print("  - seq2seq.png")


if __name__ == "__main__":
    main()
