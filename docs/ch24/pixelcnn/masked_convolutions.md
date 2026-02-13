# Masked Convolutions

## Overview

Masked convolutions are the key architectural component that enables PixelCNN to maintain the autoregressive property. By zeroing out specific kernel weights, the convolution is constrained to only access previously generated pixels.

## Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, 
                 mask_type='B', padding=None):
        super().__init__()
        assert mask_type in ('A', 'B')
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if padding is None:
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size, padding=padding)
        
        # Create mask
        mask = torch.ones_like(self.conv.weight)
        h, w = kernel_size
        mask[:, :, h // 2, w // 2 + (mask_type == 'A'):] = 0
        mask[:, :, h // 2 + 1:] = 0
        self.register_buffer('mask', mask)
    
    def forward(self, x):
        self.conv.weight.data *= self.mask
        return self.conv(x)
```

## Type A vs Type B

- **Type A**: masks the current pixel position. Used only in the first layer to ensure the output for pixel $i$ does not depend on pixel $i$ itself.
- **Type B**: includes the current pixel position. Used in all subsequent layers because the output already depends on inputs from previous layers (which maintained the autoregressive constraint).

## Color Channel Ordering

For RGB images, the ordering within each pixel is R → G → B:

$$p(x_{i,R}, x_{i,G}, x_{i,B}) = p(x_{i,R} \mid x_{<i}) \cdot p(x_{i,G} \mid x_{<i}, x_{i,R}) \cdot p(x_{i,B} \mid x_{<i}, x_{i,R}, x_{i,G})$$

This requires separate masks for each color channel, increasing implementation complexity.

## Receptive Field

The effective receptive field after $L$ layers with kernel size $k$ forms a triangular region above and to the left of each pixel. Larger kernels and more layers expand the receptive field but increase computational cost.
