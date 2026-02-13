# Gated PixelCNN

## Overview

Gated PixelCNN (van den Oord et al., 2016) addresses two limitations of the original PixelCNN: the blind spot problem and limited expressiveness from ReLU activations.

## Gated Activation

Replace ReLU with a gated activation unit:

$$y = \tanh(W_f * x) \odot \sigma(W_g * x)$$

where $W_f$ and $W_g$ are separate convolution filters for the "filter" and "gate" respectively, and $\sigma$ is the sigmoid function. This is inspired by LSTM gates and allows the network to model more complex distributions.

## Two-Stack Architecture

To eliminate the blind spot, Gated PixelCNN uses two separate convolution stacks:

### Vertical Stack
Processes all rows above the current position using a standard (non-masked) vertical convolution. Has access to the full width of all rows above.

### Horizontal Stack
Processes the current row up to (and including) the current position using a 1D masked convolution. Receives information from the vertical stack via a connection.

```
Vertical: all rows above (full width)
    ↓
    + → Horizontal: current row, left of current position
              ↓
         Output for current pixel
```

This two-stack design ensures every pixel in the valid receptive field can influence the prediction, eliminating the blind spot.

## Residual Connections

Each gated block includes a residual connection:

```python
class GatedBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.v_conv = nn.Conv2d(channels, 2 * channels, (2, 3), padding=(1, 1))
        self.h_conv = nn.Conv2d(channels, 2 * channels, (1, 3), padding=(0, 1))
        self.v_to_h = nn.Conv2d(2 * channels, 2 * channels, 1)
        self.h_res = nn.Conv2d(channels, channels, 1)
    
    def forward(self, v_input, h_input):
        v = self.v_conv(v_input)[:, :, :v_input.shape[2], :]
        
        h = self.h_conv(h_input)
        h = h + self.v_to_h(v)
        
        h_f, h_g = h.chunk(2, dim=1)
        h = torch.tanh(h_f) * torch.sigmoid(h_g)
        h = self.h_res(h) + h_input  # Residual
        
        v_f, v_g = v.chunk(2, dim=1)
        v = torch.tanh(v_f) * torch.sigmoid(v_g)
        
        return v, h
```

## Conditional Generation

Gated PixelCNN can condition on class labels by adding a class-dependent bias:

$$y = \tanh(W_f * x + V_f^T h) \odot \sigma(W_g * x + V_g^T h)$$

where $h$ is a one-hot class embedding. This enables class-conditional image generation.
