# Dilated Causal Convolutions

## Overview

Dilated causal convolutions are the key building block of WaveNet, enabling exponentially growing receptive fields without proportionally increasing computation or parameter count.

## Causal Convolution

A causal convolution ensures that the output at time $t$ depends only on inputs at times $\leq t$:

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t-k}$$

This is implemented by padding the input on the left and removing the right-most output elements.

```python
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation)
    
    def forward(self, x):
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)
```

## Dilated Convolution

A dilated convolution with dilation factor $d$ skips $d-1$ input elements between each filter tap:

$$y_t = \sum_{k=0}^{K-1} w_k \cdot x_{t - k \cdot d}$$

## Exponential Receptive Field Growth

By stacking layers with exponentially increasing dilation ($d = 1, 2, 4, 8, \ldots$), the receptive field grows exponentially with depth:

$$\text{RF} = 1 + \sum_{l=1}^{L} (K-1) \cdot d_l$$

For kernel size $K=2$ and dilations $1, 2, 4, \ldots, 2^{L-1}$:

$$\text{RF} = 2^L$$

This is vastly more efficient than standard convolutions, which grow linearly: RF $= 1 + L(K-1)$.

## Comparison

| Approach | Layers for RF=1024 | Parameters |
|----------|-------------------|-----------|
| Standard conv (k=2) | 1023 | O(1023) |
| Dilated conv (k=2) | 10 | O(10) |
| Standard conv (k=1024) | 1 | O(1024) |

Dilated convolutions achieve the best balance of receptive field size and parameter efficiency.
