# WaveNet Architecture

## Overview

WaveNet (van den Oord et al., 2016) is an autoregressive model for raw audio waveforms. It generates audio one sample at a time (16,000 samples per second for speech) using dilated causal convolutions.

## Core Architecture

WaveNet stacks dilated causal convolutions with exponentially increasing dilation rates:

```python
class WaveNet(nn.Module):
    def __init__(self, n_layers=30, n_channels=64, n_classes=256):
        super().__init__()
        self.causal_conv = CausalConv1d(1, n_channels, kernel_size=2)
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** (i % 10)  # Reset dilation every 10 layers
            self.layers.append(
                WaveNetBlock(n_channels, dilation=dilation)
            )
        
        self.output = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(n_channels, n_channels, 1),
            nn.ReLU(),
            nn.Conv1d(n_channels, n_classes, 1)
        )
    
    def forward(self, x):
        x = self.causal_conv(x)
        skip_sum = 0
        for layer in self.layers:
            x, skip = layer(x)
            skip_sum = skip_sum + skip
        return self.output(skip_sum)
```

## Mu-Law Encoding

Raw audio samples are continuous. WaveNet quantizes them using mu-law companding:

$$F(x) = \text{sign}(x) \frac{\ln(1 + \mu|x|)}{\ln(1 + \mu)}$$

with $\mu = 255$, giving 256 quantization levels. This non-linear quantization better preserves the perceptual quality of audio.

## Receptive Field

With dilation rates $1, 2, 4, \ldots, 512$ repeated 3 times (30 layers total):

$$\text{Receptive field} = \sum_{i=0}^{29} 2^{i \bmod 10} \approx 3 \times 1024 = 3072 \text{ samples}$$

At 16 kHz, this covers ~190 ms of audio — enough to capture phoneme-level structure.

## Skip Connections

Each layer contributes a skip connection to the output, allowing the network to combine information from all temporal scales.
