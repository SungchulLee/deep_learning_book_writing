# CNN-LSTM for Video Understanding

## Learning Objectives

By the end of this section, you will be able to:

- Understand why recurrent networks suit variable-length videos
- Combine CNN feature extractors with LSTM temporal modeling
- Implement encoder-decoder architectures for video
- Handle long sequences efficiently with attention mechanisms
- Compare CNN-LSTM approaches with 3D CNNs and transformers

## Motivation: Temporal Sequence Modeling

### Why Recurrent Networks?

3D CNNs have fixed temporal receptive fields, but videos have:
- **Variable length**: Different videos have different durations
- **Long-range dependencies**: Early events may affect later understanding
- **Sequential nature**: Actions unfold over time

RNNs/LSTMs naturally handle:
- Variable-length sequences
- Hidden state carries temporal context
- Learnable long-term memory

### Architecture Overview

```
Frame 1 → CNN → f₁ → LSTM → h₁
Frame 2 → CNN → f₂ → LSTM → h₂
   ⋮        ⋮       ⋮       ⋮
Frame T → CNN → f_T → LSTM → h_T → Classification
```

## CNN Feature Extraction

### Pre-trained Backbone

```python
import torch
import torch.nn as nn
import torchvision.models as models

class CNNFeatureExtractor(nn.Module):
    """
    CNN backbone for extracting frame features.
    
    Uses pretrained ImageNet models for transfer learning.
    Removes classification head to get spatial features.
    """
    
    def __init__(self, 
                 backbone: str = 'resnet50',
                 pretrained: bool = True,
                 freeze: bool = False):
        super().__init__()
        
        if backbone == 'resnet50':
            base = models.resnet50(pretrained=pretrained)
            self.feature_dim = 2048
        elif backbone == 'resnet18':
            base = models.resnet18(pretrained=pretrained)
            self.feature_dim = 512
        elif backbone == 'efficientnet_b0':
            base = models.efficientnet_b0(pretrained=pretrained)
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove classification head
        if 'resnet' in backbone:
            self.features = nn.Sequential(*list(base.children())[:-1])
        else:
            self.features = nn.Sequential(*list(base.children())[:-1])
        
        # Optionally freeze backbone
        if freeze:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from frames.
        
        Args:
            x: Frames (B, C, H, W) or (B, T, C, H, W)
        
        Returns:
            features: (B, D) or (B, T, D)
        """
        is_video = x.dim() == 5
        
        if is_video:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
        
        features = self.features(x)
        features = features.flatten(1)  # (B*T, D)
        
        if is_video:
            features = features.view(B, T, -1)
        
        return features
```

### Spatial Attention for Better Features

```python
class SpatialAttentionCNN(nn.Module):
    """
    CNN with spatial attention pooling.
    
    Instead of global average pooling, use attention
    to focus on relevant image regions.
    """
    
    def __init__(self, backbone='resnet50'):
        super().__init__()
        
        base = models.resnet50(pretrained=True)
        # Keep up to last conv layer (before avgpool)
        self.features = nn.Sequential(*list(base.children())[:-2])
        
        # Spatial attention
        self.attention = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        
        self.feature_dim = 2048
    
    def forward(self, x):
        # Extract feature maps: (B, 2048, H', W')
        feature_maps = self.features(x)
        
        # Compute attention weights: (B, 1, H', W')
        attn_weights = self.attention(feature_maps)
        
        # Weighted pooling
        weighted = feature_maps * attn_weights
        features = weighted.sum(dim=[2, 3]) / (attn_weights.sum(dim=[2, 3]) + 1e-8)
        
        return features
```

## LSTM Temporal Modeling

### Basic CNN-LSTM

```python
class CNNLSTM(nn.Module):
    """
    CNN-LSTM for video classification.
    
    Architecture:
        1. CNN extracts features from each frame
        2. LSTM models temporal dependencies
        3. Classification from final hidden state
    """
    
    def __init__(self,
                 num_classes: int = 101,
                 hidden_size: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 bidirectional: bool = False):
        super().__init__()
        
        # CNN feature extractor
        self.cnn = CNNFeatureExtractor(backbone='resnet50', freeze=True)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Classifier
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video tensor (B, T, C, H, W)
        
        Returns:
            Class logits (B, num_classes)
        """
        B, T, C, H, W = x.shape
        
        # Extract CNN features for all frames
        features = self.cnn(x)  # (B, T, D)
        
        # LSTM processing
        # h_n: final hidden state (num_layers * directions, B, H)
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Use final hidden state for classification
        # If bidirectional, concatenate forward and backward
        if self.lstm.bidirectional:
            hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            hidden = h_n[-1]
        
        # Classify
        return self.classifier(hidden)
```

### Sequence-to-Sequence LSTM

For dense predictions (per-frame output):

```python
class Seq2SeqCNNLSTM(nn.Module):
    """
    Sequence-to-sequence CNN-LSTM for dense video prediction.
    
    Output at every timestep, useful for:
    - Frame-level action detection
    - Video captioning
    - Temporal segmentation
    """
    
    def __init__(self, num_classes: int, hidden_size: int = 512):
        super().__init__()
        
        self.cnn = CNNFeatureExtractor()
        
        self.lstm = nn.LSTM(
            input_size=self.cnn.feature_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Per-timestep classifier
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Video (B, T, C, H, W)
        
        Returns:
            Per-frame logits (B, T, num_classes)
        """
        features = self.cnn(x)  # (B, T, D)
        
        # LSTM outputs at all timesteps
        lstm_out, _ = self.lstm(features)  # (B, T, H)
        
        # Classify each timestep
        logits = self.fc(lstm_out)  # (B, T, num_classes)
        
        return logits
```

## Attention Mechanisms

### Temporal Attention

```python
class TemporalAttention(nn.Module):
    """
    Attention over LSTM hidden states.
    
    Learns which timesteps are most important for classification.
    
    Mathematical formulation:
        α_t = softmax(w^T tanh(W h_t + b))
        context = Σ α_t · h_t
    """
    
    def __init__(self, hidden_size: int, attention_size: int = 128):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1)
        )
    
    def forward(self, lstm_outputs: torch.Tensor, 
                mask: torch.Tensor = None) -> tuple:
        """
        Args:
            lstm_outputs: LSTM hidden states (B, T, H)
            mask: Optional mask for variable length (B, T)
        
        Returns:
            context: Attended representation (B, H)
            attention_weights: Weights (B, T)
        """
        # Compute attention scores
        scores = self.attention(lstm_outputs).squeeze(-1)  # (B, T)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Softmax to get weights
        weights = torch.softmax(scores, dim=1)  # (B, T)
        
        # Weighted sum
        context = (weights.unsqueeze(-1) * lstm_outputs).sum(dim=1)  # (B, H)
        
        return context, weights


class CNNLSTMAttention(nn.Module):
    """CNN-LSTM with temporal attention."""
    
    def __init__(self, num_classes: int, hidden_size: int = 512):
        super().__init__()
        
        self.cnn = CNNFeatureExtractor()
        
        self.lstm = nn.LSTM(
            input_size=self.cnn.feature_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        self.attention = TemporalAttention(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Returns:
            logits: Class predictions (B, num_classes)
            attention_weights: For visualization (B, T)
        """
        features = self.cnn(x)
        lstm_out, _ = self.lstm(features)
        
        context, weights = self.attention(lstm_out)
        logits = self.classifier(context)
        
        return logits, weights
```

### Self-Attention LSTM

Combine LSTM with self-attention:

```python
class SelfAttentionLSTM(nn.Module):
    """
    LSTM with self-attention layer.
    
    Self-attention captures pairwise relationships between
    all timesteps, complementing LSTM's sequential processing.
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_heads: int = 8):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)  # (B, T, H)
        
        # Self-attention over LSTM outputs
        attn_out, _ = self.self_attn(lstm_out, lstm_out, lstm_out)
        
        # Residual connection
        out = self.layer_norm(lstm_out + attn_out)
        
        return out, (h_n, c_n)
```

## Handling Variable-Length Videos

### Padding and Packing

```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class VariableLengthCNNLSTM(nn.Module):
    """
    CNN-LSTM that handles variable length videos efficiently.
    
    Uses PyTorch's packed sequences to avoid computing
    on padding positions.
    """
    
    def __init__(self, num_classes: int, hidden_size: int = 512):
        super().__init__()
        
        self.cnn = CNNFeatureExtractor()
        self.lstm = nn.LSTM(
            self.cnn.feature_dim, hidden_size,
            batch_first=True, num_layers=2
        )
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Padded video batch (B, T_max, C, H, W)
            lengths: Actual lengths for each video (B,)
        
        Returns:
            Logits (B, num_classes)
        """
        B, T_max, C, H, W = x.shape
        
        # Extract features
        features = self.cnn(x)  # (B, T_max, D)
        
        # Pack sequences
        packed = pack_padded_sequence(
            features, lengths.cpu(),
            batch_first=True, enforce_sorted=False
        )
        
        # LSTM on packed sequence
        packed_out, (h_n, c_n) = self.lstm(packed)
        
        # Use final hidden state
        hidden = h_n[-1]  # (B, H)
        
        return self.classifier(hidden)


def collate_variable_length(batch):
    """
    Custom collate function for variable-length videos.
    """
    # Sort by length (descending)
    batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
    
    videos = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    lengths = torch.tensor([v.shape[0] for v in videos])
    
    # Pad videos
    max_len = max(lengths)
    padded = torch.zeros(len(videos), max_len, *videos[0].shape[1:])
    for i, v in enumerate(videos):
        padded[i, :v.shape[0]] = v
    
    return padded, labels, lengths
```

## Advanced Architectures

### Bidirectional CNN-LSTM

```python
class BiCNNLSTM(nn.Module):
    """
    Bidirectional CNN-LSTM for video understanding.
    
    Processes video in both directions:
    - Forward: past → present → future
    - Backward: future → present → past
    
    Captures both anticipatory and reactionary patterns.
    """
    
    def __init__(self, num_classes: int, hidden_size: int = 256):
        super().__init__()
        
        self.cnn = CNNFeatureExtractor()
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            self.cnn.feature_dim,
            hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Output size is 2x hidden_size due to bidirection
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x):
        features = self.cnn(x)
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Concatenate final forward and backward hidden states
        # h_n shape: (num_layers * 2, B, H)
        forward_h = h_n[-2]   # Last layer, forward
        backward_h = h_n[-1]  # Last layer, backward
        hidden = torch.cat([forward_h, backward_h], dim=1)
        
        return self.classifier(hidden)
```

### Stacked CNN-LSTM with Skip Connections

```python
class DeepCNNLSTM(nn.Module):
    """
    Deep CNN-LSTM with residual connections.
    
    Stacks multiple LSTM layers with skip connections
    for better gradient flow in deep models.
    """
    
    def __init__(self, num_classes: int, hidden_size: int = 512, num_layers: int = 4):
        super().__init__()
        
        self.cnn = CNNFeatureExtractor()
        
        # Project CNN features to LSTM dimension
        self.input_proj = nn.Linear(self.cnn.feature_dim, hidden_size)
        
        # Stacked LSTM layers with skip connections
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_size, hidden_size, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        features = self.cnn(x)
        x = self.input_proj(features)
        
        # Process through LSTM layers with skip connections
        for lstm, ln in zip(self.lstm_layers, self.layer_norms):
            residual = x
            x, _ = lstm(x)
            x = self.dropout(x)
            x = ln(x + residual)  # Skip connection
        
        # Global average pooling over time
        x = x.mean(dim=1)
        
        return self.classifier(x)
```

## Training Strategies

### Gradual Unfreezing

```python
def train_with_gradual_unfreeze(model, train_loader, epochs):
    """
    Training strategy: gradually unfreeze CNN layers.
    
    Phase 1: Train only LSTM (CNN frozen)
    Phase 2: Unfreeze last CNN block
    Phase 3: Unfreeze entire network
    """
    
    # Phase 1: Freeze CNN
    for param in model.cnn.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3
    )
    
    for epoch in range(epochs // 3):
        train_epoch(model, train_loader, optimizer)
    
    # Phase 2: Unfreeze last CNN layer
    for name, param in model.cnn.features.named_parameters():
        if 'layer4' in name:
            param.requires_grad = True
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )
    
    for epoch in range(epochs // 3, 2 * epochs // 3):
        train_epoch(model, train_loader, optimizer)
    
    # Phase 3: Unfreeze all
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    
    for epoch in range(2 * epochs // 3, epochs):
        train_epoch(model, train_loader, optimizer)
```

### Teacher Forcing (for Seq2Seq)

```python
def train_seq2seq_with_teacher_forcing(model, video, targets, 
                                        teacher_forcing_ratio=0.5):
    """
    Train sequence-to-sequence model with teacher forcing.
    
    Teacher forcing: Use ground truth as input with some probability,
    otherwise use model's own predictions.
    """
    B, T = targets.shape
    outputs = []
    
    # Get CNN features
    features = model.cnn(video)
    
    # Initialize hidden state
    hidden = None
    
    for t in range(T):
        if t == 0:
            # First timestep: use CNN features
            lstm_input = features[:, 0:1]
        else:
            # Decide: teacher forcing or not
            use_teacher = torch.rand(1) < teacher_forcing_ratio
            
            if use_teacher:
                lstm_input = features[:, t:t+1]
            else:
                # Use previous prediction
                lstm_input = predicted_features
        
        # LSTM step
        lstm_out, hidden = model.lstm(lstm_input, hidden)
        output = model.fc(lstm_out)
        outputs.append(output)
        
        # For next iteration (if not using teacher forcing)
        predicted_features = lstm_out
    
    return torch.cat(outputs, dim=1)
```

## Comparison with Other Approaches

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| CNN-LSTM | Variable length, long-term memory | Sequential processing (slow) |
| 3D CNN | Parallel processing, local patterns | Fixed temporal receptive field |
| Transformer | Global attention, parallel | Quadratic complexity |
| CNN-LSTM+Attention | Interpretable, selective focus | Still sequential |

## Summary

CNN-LSTM architectures combine:
- **CNN strength**: Spatial feature extraction with transfer learning
- **LSTM strength**: Temporal sequence modeling with memory

Key design choices:
1. **Backbone**: ResNet-50 balances speed and accuracy
2. **LSTM layers**: 2-3 layers typically sufficient
3. **Bidirectional**: Better for offline processing
4. **Attention**: Improves long sequence handling
5. **Training**: Gradual unfreezing for stability

## Next Steps

- **Action Recognition**: Complete pipeline
- **Video Captioning**: Seq2Seq with attention
- **Temporal Action Detection**: Per-frame predictions
