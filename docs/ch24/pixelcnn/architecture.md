# PixelCNN Architecture

## Overview

PixelCNN (van den Oord et al., 2016) is an autoregressive model for images that uses masked convolutions to model the conditional distribution of each pixel given all previously generated pixels in raster scan order.

## Architecture

The model stacks multiple masked convolutional layers, with each layer maintaining the autoregressive property:

```python
import torch
import torch.nn as nn

class PixelCNN(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, 
                 n_layers=7, n_colors=256):
        super().__init__()
        layers = []
        
        # First layer: Type A mask
        layers.append(MaskedConv2d(in_channels, hidden_channels, 
                                    kernel_size=7, mask_type='A'))
        layers.append(nn.ReLU())
        
        # Hidden layers: Type B mask
        for _ in range(n_layers - 2):
            layers.append(MaskedConv2d(hidden_channels, hidden_channels,
                                        kernel_size=7, mask_type='B'))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(MaskedConv2d(hidden_channels, n_colors,
                                    kernel_size=1, mask_type='B'))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)  # logits for each pixel value
    
    def loss(self, x):
        logits = self.forward(x)
        return nn.functional.cross_entropy(logits, x.long())
```

## Sampling

Sampling proceeds pixel by pixel in raster scan order:

```python
@torch.no_grad()
def sample(self, height, width, device):
    x = torch.zeros(1, 1, height, width, device=device)
    
    for i in range(height):
        for j in range(width):
            logits = self.forward(x)
            probs = torch.softmax(logits[:, :, i, j], dim=1)
            pixel = torch.multinomial(probs, 1)
            x[:, :, i, j] = pixel.float() / 255.0
    
    return x
```

## Limitations

The original PixelCNN has a **blind spot**: due to the mask shape, some pixels in the receptive field cannot influence the prediction. Additionally, using ReLU activations limits the model's expressiveness for modeling complex pixel distributions.
