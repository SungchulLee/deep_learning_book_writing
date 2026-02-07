# WaveNet: Autoregressive Audio Generation

## Introduction

WaveNet, introduced by van den Oord et al. (2016) at DeepMind, represents a breakthrough in autoregressive generative modeling for audio. Unlike PixelCNN which operates on 2D images, WaveNet models 1D audio waveforms at the raw sample level—typically 16,000 or more samples per second. The key innovations are **dilated causal convolutions** that provide exponentially growing receptive fields, enabling the model to capture long-range temporal dependencies essential for natural-sounding audio.

## The Audio Generation Challenge

### Raw Waveform Modeling

Audio is fundamentally a sequence of amplitude values sampled at high frequency:
- Speech: typically 16 kHz (16,000 samples/second)
- Music: typically 44.1 kHz (CD quality)

For autoregressive modeling, we factorize:

$$P(\mathbf{x}) = \prod_{t=1}^{T} P(x_t | x_1, x_2, \ldots, x_{t-1})$$

where $x_t$ is the amplitude at time $t$, typically quantized to 8-bit (256 levels) or 16-bit values.

### The Receptive Field Problem

To generate natural audio, the model must capture dependencies across:
- **Short-term**: Individual phonemes, note attacks (~10-50ms)
- **Medium-term**: Words, musical phrases (~100-500ms)
- **Long-term**: Prosody, melody, rhythm (~1-5 seconds)

At 16 kHz, 1 second requires conditioning on 16,000 previous samples. Standard convolutions with reasonable kernel sizes cannot achieve this without prohibitively deep networks.

## Dilated Causal Convolutions

### Causal Convolutions

A **causal convolution** ensures that predictions for time $t$ only depend on $x_{<t}$, never on future values:

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-k}$$

where $K$ is the kernel size. Unlike standard convolutions that are centered, causal convolutions are shifted to only include past values.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    1D causal convolution.
    
    Ensures output at time t only depends on inputs at times <= t.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True
    ):
        super().__init__()
        
        self.kernel_size = kernel_size
        self.dilation = dilation
        
        # Padding to ensure causality
        # With dilation d and kernel k, we need (k-1)*d padding on the left
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            dilation=dilation,
            bias=bias
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply causal convolution.
        
        Args:
            x: Input tensor [batch, channels, time]
            
        Returns:
            Output tensor [batch, out_channels, time]
        """
        # Pad on the left (past) only
        x_padded = F.pad(x, (self.padding, 0))
        
        return self.conv(x_padded)
```

### Dilated Convolutions

**Dilated convolutions** (also called atrous convolutions) skip input values with a fixed gap:

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t - k \cdot d}$$

where $d$ is the dilation factor. With dilation $d$ and kernel size $K$, the receptive field spans $(K-1) \cdot d + 1$ time steps.

### Exponentially Growing Receptive Field

WaveNet stacks dilated convolutions with exponentially increasing dilation factors:

| Layer | Dilation | Receptive Field Growth |
|-------|----------|----------------------|
| 1     | 1        | +2                   |
| 2     | 2        | +4                   |
| 3     | 4        | +8                   |
| 4     | 8        | +16                  |
| ...   | ...      | ...                  |
| 10    | 512      | +1024                |

With kernel size 2 and 10 layers, the receptive field is $2^{10} = 1024$ samples. Multiple stacks multiply this further.

```python
def compute_receptive_field(kernel_size: int, n_layers: int, n_stacks: int = 1) -> int:
    """
    Compute total receptive field of WaveNet.
    
    Args:
        kernel_size: Convolution kernel size (typically 2)
        n_layers: Number of layers per stack
        n_stacks: Number of dilation stacks
    
    Returns:
        Receptive field in samples
    """
    # Each layer adds (kernel_size - 1) * dilation
    # With exponential dilation: 1, 2, 4, ..., 2^(n_layers-1)
    layers_per_stack = sum((kernel_size - 1) * (2 ** i) for i in range(n_layers))
    
    # Multiple stacks add linearly
    total = layers_per_stack * n_stacks + 1
    
    return total

# Example: kernel=2, 10 layers, 3 stacks
rf = compute_receptive_field(2, 10, 3)
print(f"Receptive field: {rf} samples")  # 3069 samples
print(f"At 16kHz: {rf/16000:.3f} seconds")  # ~0.19 seconds
```

## WaveNet Architecture

### Residual Block

The core building block combines dilated causal convolution with gated activation:

```python
class WaveNetResidualBlock(nn.Module):
    """
    WaveNet residual block with gated activation.
    
    Structure:
        input -> dilated_conv -> [tanh, sigmoid] -> multiply -> 1x1 conv -> + input
                                                        |
                                                        v
                                                    skip output
    """
    
    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        kernel_size: int,
        dilation: int,
        condition_channels: int = 0
    ):
        super().__init__()
        
        self.dilation = dilation
        self.condition_channels = condition_channels
        
        # Dilated causal convolution
        # Output 2x channels: half for filter (tanh), half for gate (sigmoid)
        self.dilated_conv = CausalConv1d(
            residual_channels,
            2 * residual_channels,
            kernel_size,
            dilation=dilation
        )
        
        # Optional conditioning (e.g., speaker embedding, linguistic features)
        if condition_channels > 0:
            self.condition_conv = nn.Conv1d(
                condition_channels,
                2 * residual_channels,
                kernel_size=1
            )
        
        # 1x1 convolution for residual and skip
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor [batch, residual_channels, time]
            condition: Optional conditioning [batch, condition_channels, time]
            
        Returns:
            (residual_output, skip_output)
        """
        # Dilated causal convolution
        h = self.dilated_conv(x)
        
        # Add conditioning if provided
        if condition is not None and self.condition_channels > 0:
            h = h + self.condition_conv(condition)
        
        # Split for gated activation
        h_filter, h_gate = h.chunk(2, dim=1)
        
        # Gated activation: tanh(filter) * sigmoid(gate)
        h = torch.tanh(h_filter) * torch.sigmoid(h_gate)
        
        # Skip connection (before residual)
        skip = self.skip_conv(h)
        
        # Residual connection
        residual = self.residual_conv(h) + x
        
        return residual, skip
```

### Complete WaveNet Model

```python
class WaveNet(nn.Module):
    """
    WaveNet autoregressive model for audio generation.
    
    Architecture:
        - Input embedding (quantized audio -> continuous)
        - Stack(s) of dilated residual blocks
        - Skip connection aggregation
        - Output layers -> categorical distribution over amplitude levels
    """
    
    def __init__(
        self,
        n_classes: int = 256,           # μ-law quantization levels
        residual_channels: int = 32,    # Channels in residual path
        skip_channels: int = 256,       # Channels in skip connections
        n_layers: int = 10,             # Layers per dilation stack
        n_stacks: int = 3,              # Number of dilation stacks
        kernel_size: int = 2,
        condition_channels: int = 0     # Global conditioning channels
    ):
        super().__init__()
        
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_stacks = n_stacks
        
        # Input embedding: one-hot -> continuous
        self.input_conv = CausalConv1d(n_classes, residual_channels, 1)
        
        # Build dilated residual blocks
        self.residual_blocks = nn.ModuleList()
        
        for stack in range(n_stacks):
            for layer in range(n_layers):
                dilation = 2 ** layer
                
                block = WaveNetResidualBlock(
                    residual_channels=residual_channels,
                    skip_channels=skip_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    condition_channels=condition_channels
                )
                self.residual_blocks.append(block)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, n_classes, 1)
        )
        
        # Compute and store receptive field
        self.receptive_field = compute_receptive_field(kernel_size, n_layers, n_stacks)
    
    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute logits for next sample prediction.
        
        Args:
            x: Input audio [batch, time], integer values in [0, n_classes)
            condition: Optional conditioning [batch, condition_channels, time]
            
        Returns:
            Logits [batch, n_classes, time]
        """
        # One-hot encode input
        x_onehot = F.one_hot(x, self.n_classes).float()  # [batch, time, n_classes]
        x_onehot = x_onehot.transpose(1, 2)  # [batch, n_classes, time]
        
        # Initial convolution
        h = self.input_conv(x_onehot)
        
        # Accumulate skip connections
        skip_sum = 0
        
        # Process through residual blocks
        for block in self.residual_blocks:
            h, skip = block(h, condition)
            skip_sum = skip_sum + skip
        
        # Output layers
        logits = self.output_layers(skip_sum)
        
        return logits
    
    def compute_loss(
        self,
        x: torch.Tensor,
        condition: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss.
        
        Uses teacher forcing: input is x[:-1], target is x[1:].
        """
        # Shift for next-sample prediction
        inputs = x[:, :-1]
        targets = x[:, 1:]
        
        # Get logits
        logits = self.forward(inputs, condition)
        
        # Align logits with targets
        T = targets.shape[1]
        logits = logits[:, :, -T:]
        
        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.transpose(1, 2).reshape(-1, self.n_classes),
            targets.reshape(-1)
        )
        
        return loss
```

## μ-Law Quantization

### Why Quantize?

Raw 16-bit audio has 65,536 possible values per sample—too many for a categorical distribution. WaveNet uses **μ-law companding** to quantize to 256 levels while preserving perceptual quality.

### μ-Law Transform

The μ-law algorithm compresses dynamic range:

$$f(x) = \text{sign}(x) \frac{\ln(1 + \mu |x|)}{\ln(1 + \mu)}$$

where $\mu = 255$ for 8-bit quantization. This transformation:
- Compresses loud sounds (large amplitudes)
- Expands quiet sounds (small amplitudes)
- Matches human hearing's logarithmic sensitivity

```python
def mu_law_encode(audio: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """
    Apply μ-law encoding to audio.
    
    Args:
        audio: Float audio in [-1, 1], shape [batch, time]
        mu: Compression parameter (255 for 8-bit)
        
    Returns:
        Quantized audio in [0, mu], shape [batch, time]
    """
    audio = audio.clamp(-1, 1)
    compressed = torch.sign(audio) * torch.log1p(mu * torch.abs(audio)) / torch.log1p(torch.tensor(float(mu)))
    quantized = ((compressed + 1) / 2 * mu + 0.5).long()
    return quantized.clamp(0, mu)


def mu_law_decode(quantized: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """
    Decode μ-law quantized audio.
    
    Args:
        quantized: Quantized audio in [0, mu], shape [batch, time]
        mu: Compression parameter
        
    Returns:
        Float audio in [-1, 1], shape [batch, time]
    """
    normalized = 2 * quantized.float() / mu - 1
    expanded = torch.sign(normalized) * (torch.pow(1 + mu, torch.abs(normalized)) - 1) / mu
    return expanded
```

## Conditioning Mechanisms

### Global Conditioning

Global conditioning provides a single vector for the entire sequence (e.g., speaker identity):

$$\mathbf{h} = \tanh(W_{f,k} * \mathbf{x} + V_f \mathbf{g}) \odot \sigma(W_{g,k} * \mathbf{x} + V_g \mathbf{g})$$

where $\mathbf{g}$ is a global conditioning vector.

```python
class GlobalConditionedWaveNet(WaveNet):
    """WaveNet with global conditioning (e.g., speaker ID)."""
    
    def __init__(self, n_speakers: int, speaker_embedding_dim: int = 64, **kwargs):
        super().__init__(condition_channels=speaker_embedding_dim, **kwargs)
        self.speaker_embedding = nn.Embedding(n_speakers, speaker_embedding_dim)
    
    def forward(self, x: torch.Tensor, speaker_id: torch.Tensor) -> torch.Tensor:
        speaker_emb = self.speaker_embedding(speaker_id)
        T = x.shape[1]
        condition = speaker_emb.unsqueeze(-1).expand(-1, -1, T)
        return super().forward(x, condition)
```

### Local Conditioning

Local conditioning provides time-varying information (e.g., linguistic features for TTS):

```python
class LocalConditionedWaveNet(WaveNet):
    """WaveNet with local conditioning (e.g., mel spectrogram for TTS)."""
    
    def __init__(self, local_condition_channels: int, upsample_scales: list = [16, 16], **kwargs):
        super().__init__(condition_channels=local_condition_channels, **kwargs)
        
        self.upsample = nn.ModuleList()
        for scale in upsample_scales:
            self.upsample.append(
                nn.ConvTranspose1d(
                    local_condition_channels,
                    local_condition_channels,
                    kernel_size=scale * 2,
                    stride=scale,
                    padding=scale // 2
                )
            )
    
    def upsample_condition(self, condition: torch.Tensor, target_length: int) -> torch.Tensor:
        for layer in self.upsample:
            condition = F.relu(layer(condition))
        
        if condition.shape[-1] > target_length:
            condition = condition[:, :, :target_length]
        elif condition.shape[-1] < target_length:
            condition = F.pad(condition, (0, target_length - condition.shape[-1]))
        
        return condition
    
    def forward(self, x: torch.Tensor, local_condition: torch.Tensor) -> torch.Tensor:
        condition = self.upsample_condition(local_condition, x.shape[1])
        return super().forward(x, condition)
```

## Training and Generation

### Training Procedure

```python
def train_wavenet(
    model: WaveNet,
    train_loader: torch.utils.data.DataLoader,
    n_epochs: int,
    learning_rate: float = 1e-3,
    device: str = 'cuda'
):
    """Train WaveNet model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(n_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (audio, condition) in enumerate(train_loader):
            audio = audio.to(device)
            if condition is not None:
                condition = condition.to(device)
            
            audio_quantized = mu_law_encode(audio)
            loss = model.compute_loss(audio_quantized, condition)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}, Loss: {total_loss / len(train_loader):.4f}")
```

### Generation (Inference)

```python
@torch.no_grad()
def generate_audio(
    model: WaveNet,
    n_samples: int,
    condition: torch.Tensor = None,
    temperature: float = 1.0,
    device: str = 'cuda'
) -> torch.Tensor:
    """Generate audio autoregressively."""
    model.eval()
    model = model.to(device)
    
    generated = torch.full((1, 1), 128, dtype=torch.long, device=device)
    
    for i in range(n_samples - 1):
        context_start = max(0, generated.shape[1] - model.receptive_field)
        context = generated[:, context_start:]
        
        cond_context = None
        if condition is not None:
            cond_context = condition[:, :, context_start:generated.shape[1]]
        
        logits = model(context, cond_context)
        logits = logits[:, :, -1] / temperature
        
        probs = F.softmax(logits, dim=1)
        next_sample = torch.multinomial(probs, 1)
        generated = torch.cat([generated, next_sample], dim=1)
    
    audio = mu_law_decode(generated.squeeze(0).cpu())
    return audio
```

## Fast WaveNet and Variants

### The Generation Bottleneck

Standard WaveNet generation is extremely slow:
- Each sample requires a full forward pass
- At 16 kHz: 16,000 forward passes per second of audio
- Real-time factor (RTF) can be 1000x or more

### Parallel WaveNet

Parallel WaveNet (van den Oord et al., 2018) achieves real-time generation through **probability density distillation**:

1. Train a standard autoregressive WaveNet (teacher)
2. Train a parallel student model (Inverse Autoregressive Flow)
3. Student learns to match teacher's distribution in parallel

### Other Fast Variants

| Method | RTF (GPU) | RTF (CPU) | Approach |
|--------|-----------|-----------|----------|
| WaveNet (naive) | ~1000x | ~10000x | Autoregressive |
| Fast WaveNet | ~100x | ~1000x | Caching |
| Parallel WaveNet | ~1x | N/A | Distillation |
| WaveRNN | ~10x | ~1x | Sparse RNN |
| WaveGlow | ~1x | ~10x | Flow-based |

## Applications

### Text-to-Speech (TTS)

WaveNet revolutionized speech synthesis:
- Google Assistant uses WaveNet
- Orders of magnitude more natural than concatenative synthesis
- Enables expressive, controllable speech

### Music Generation

WaveNet can generate music when conditioned appropriately:
- MIDI conditioning for note sequences
- Genre/style embeddings
- Multi-instrument generation

### Audio Super-Resolution

Upsampling low-quality audio to high-quality by conditioning on interpolated low-resolution input.

## Summary

WaveNet demonstrated that autoregressive models can achieve unprecedented quality in audio generation:

1. **Dilated causal convolutions** enable large receptive fields efficiently
2. **Gated activations** capture complex temporal dependencies
3. **μ-law quantization** makes categorical modeling tractable
4. **Conditioning mechanisms** enable controlled generation

While generation speed was initially a limitation, subsequent work achieved real-time synthesis, making autoregressive audio models practical for production systems.

## References

1. van den Oord, A., et al. (2016). WaveNet: A Generative Model for Raw Audio. *arXiv:1609.03499*.
2. Paine, T. L., et al. (2016). Fast Wavenet Generation Algorithm. *arXiv:1611.09482*.
3. van den Oord, A., et al. (2018). Parallel WaveNet: Fast High-Fidelity Speech Synthesis. *ICML*.
4. Prenger, R., Valle, R., & Catanzaro, B. (2019). WaveGlow: A Flow-based Generative Network for Speech Synthesis. *ICASSP*.
5. Kalchbrenner, N., et al. (2018). Efficient Neural Audio Synthesis. *ICML*.

---

## Exercises

1. **Receptive Field Analysis**: Compute and visualize the receptive field of WaveNet configurations with different numbers of layers and stacks.

2. **μ-Law Comparison**: Compare audio quality using 8-bit μ-law versus linear quantization at various bit depths.

3. **Conditioning Experiment**: Train WaveNet models with different conditioning (global speaker ID vs. local linguistic features) and compare.

4. **Financial Audio**: Adapt WaveNet for generating synthetic order flow or tick data, treating price movements as a 1D sequence.
