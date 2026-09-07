# ConvNeXt V2

ConvNeXt V2, introduced in the 2023 paper "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders," extends the original ConvNeXt with two key innovations: a self-supervised pretraining strategy using masked autoencoders (FCMAE) and a new normalization layer called Global Response Normalization (GRN). Together, these improvements enable ConvNeXt V2 to scale more effectively and achieve stronger performance across a wide range of model sizes.

## Code

```python
#!/usr/bin/env python3
'''
ConvNeXt V2 - Modern ConvNet with Improved Design
Paper: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders" (2023)
Key: Global Response Normalization (GRN), improved training strategies
'''
import torch
import torch.nn as nn

# ========================================================================
# Main
# ========================================================================

class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))
    
    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class ConvNeXtV2Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
    
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x

class ConvNeXtV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Conv2d(3, 96, kernel_size=4, stride=4)
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(96) for _ in range(3)])
        self.head = nn.Linear(96, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = x.mean([2, 3])
        return self.head(x)

if __name__ == "__main__":
    model = ConvNeXtV2()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Discussion

The Global Response Normalization (GRN) layer is the architectural centerpiece of ConvNeXt V2. It addresses the feature collapse problem that arises when training ConvNets with masked autoencoder pretraining. GRN operates by first computing the $L^2$ norm of each channel's spatial response, then normalizing these norms relative to their mean across channels, and finally using the normalized values to re-scale the original features. The learnable parameters $\gamma$ and $\beta$ provide flexibility, and the residual connection ensures the layer can initially behave as an identity.

The motivation for GRN comes from the observation that masked autoencoder pretraining, which works well for Vision Transformers, causes feature redundancy in ConvNets. Without GRN, many channels learn similar representations, reducing the effective capacity of the model. By encouraging competition among channels through global normalization, GRN promotes feature diversity and makes each channel's contribution more distinctive.

Compared to the original ConvNeXt, the V2 block replaces the layer scale mechanism with GRN placed after the GELU activation in the inverted bottleneck. This change, combined with the fully convolutional masked autoencoder (FCMAE) pretraining strategy, allows ConvNeXt V2 to match or surpass transformer-based models across a spectrum of model sizes, from tiny (4M parameters) to huge (600M+ parameters).

## Exercises

**Exercise 1.**
For an input tensor of shape $(B, H, W, D)$ in the GRN layer, describe the shape of $G_x$ and $N_x$ at each step.

??? success "Solution to Exercise 1"
    Starting with input $x$ of shape $(B, H, W, D)$: (1) $G_x = \|x\|_2$ computed over dimensions $(1, 2)$ (height and width) with keepdim gives shape $(B, 1, 1, D)$. This is the $L^2$ norm of each channel's spatial response. (2) $G_x.\text{mean}(\text{dim}=-1, \text{keepdim}=\text{True})$ averages across the $D$ dimension, producing shape $(B, 1, 1, 1)$. (3) $N_x = G_x / (\text{mean} + \epsilon)$ has shape $(B, 1, 1, D)$, representing the normalized response of each channel relative to the average channel response. The final output $\gamma \cdot (x \cdot N_x) + \beta + x$ has shape $(B, H, W, D)$.

---

**Exercise 2.**
Compare GRN to Squeeze-and-Excitation (SE) blocks. What are the key similarities and differences in how they modulate channel responses?

??? success "Solution to Exercise 2"
    Both GRN and SE blocks perform channel-wise feature recalibration based on global spatial information. SE blocks squeeze spatial dimensions via global average pooling, then learn channel weights through a bottleneck MLP with sigmoid activation, producing weights in $[0, 1]$. GRN instead computes $L^2$ norms over spatial dimensions and normalizes them by their cross-channel mean, producing unbounded scaling factors. Key differences: (1) SE uses a learned nonlinear transformation (MLP) while GRN uses a fixed normalization formula with learned affine parameters. (2) SE weights are bounded by sigmoid, while GRN scaling is unbounded. (3) GRN includes a residual connection by design. (4) GRN has far fewer parameters (just $2D$) compared to SE ($2D^2/r$ where $r$ is the reduction ratio).

---

**Exercise 3.**
Implement a variant of GRN that operates in NCHW format (without requiring the permute to NHWC), making it compatible with standard convolutional layers.

??? success "Solution to Exercise 3"
    ```python
    class GRN_NCHW(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
            self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

        def forward(self, x):
            # x: (B, C, H, W)
            Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)    # (B, C, 1, 1)
            return self.gamma * (x * Nx) + self.beta + x
    ```
    The key change is adjusting the norm dimensions from $(1, 2)$ to $(2, 3)$ and the mean dimension from $-1$ to $1$, and reshaping the learnable parameters to $(1, D, 1, 1)$ for broadcasting compatibility with NCHW tensors.
