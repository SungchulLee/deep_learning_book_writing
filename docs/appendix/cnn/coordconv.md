# CoordConv

CoordConv, introduced in the 2018 paper "An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution," addresses a fundamental limitation of standard convolutions: they are translation equivariant by design, which means they cannot natively encode spatial position. By concatenating coordinate channels to the input, CoordConv allows networks to learn position-dependent filters, dramatically improving performance on tasks requiring spatial awareness such as coordinate regression and object detection.

## 코드

```python
#!/usr/bin/env python3
'''
CoordConv - Adding Spatial Coordinates to CNNs
Paper: "An Intriguing Failing of Convolutional Neural Networks" (2018)
Key: Adds coordinate channels to help with spatial reasoning
'''
import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

class AddCoords(nn.Module):
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r
    
    def forward(self, x):
        batch_size, _, height, width = x.size()
        
        xx_channel = torch.arange(width, dtype=x.dtype, device=x.device).repeat(1, height, 1)
        yy_channel = torch.arange(height, dtype=x.dtype, device=x.device).repeat(1, width, 1).transpose(1, 2)
        
        xx_channel = xx_channel / (width - 1)
        yy_channel = yy_channel / (height - 1)
        
        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1
        
        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1)
        
        ret = torch.cat([x, xx_channel, yy_channel], dim=1)
        
        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel, 2) + torch.pow(yy_channel, 2))
            ret = torch.cat([ret, rr], dim=1)
        
        return ret

class CoordConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, with_r=False):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x

class CoordConvNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = CoordConv(3, 64, 7, 2, 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        
        self.conv2 = CoordConv(64, 128, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.avgpool(x).flatten(1)
        return self.fc(x)

if __name__ == "__main__":
    model = CoordConvNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## 논의

The key insight behind CoordConv is that standard convolutions apply the same learned filter everywhere across the spatial dimensions, making them inherently unable to distinguish between different positions. While this translation equivariance is desirable for many tasks (e.g., recognizing an object regardless of where it appears), it becomes a liability for tasks that require understanding absolute or relative spatial position, such as converting pixel coordinates to one-hot grids or detecting objects at specific locations.

The `AddCoords` module generates normalized coordinate grids ranging from $-1$ to $1$ along both spatial axes. These are concatenated as additional input channels before the convolution, effectively breaking translation equivariance in a controlled way. The optional radial coordinate channel $r = \sqrt{x^2 + y^2}$ provides distance-from-center information, which is useful for tasks with radial symmetry. Importantly, the coordinate channels add no learnable parameters; the network learns through the standard convolution weights how much to rely on positional information.

The overhead of CoordConv is minimal: only 2 (or 3) additional input channels per convolution layer. The original paper demonstrated that this simple modification solved the "coordinate transform" problem that standard CNNs completely failed at, and improved performance on generative models and object detection. CoordConv has since become a standard tool in architectures where spatial awareness is critical.

## 익힘 문제

**익힘 1.**
For an input of shape $(2, 3, 8, 8)$, what is the output shape of `AddCoords` with `with_r=True`? What are the values at position $(0, 0)$ and $(7, 7)$ in the x-coordinate channel?

??? success "익힘 1 풀이"
    The output shape is $(2, 6, 8, 8)$: the original 3 channels plus x-coordinate, y-coordinate, and r-coordinate channels. At position $(0, 0)$: $x = 0/(8-1) \times 2 - 1 = -1$. At position $(7, 7)$: $x = 7/7 \times 2 - 1 = 1$. The coordinate channels span from $-1$ to $1$ linearly across the spatial dimensions.

---

**익힘 2.**
Explain why CoordConv is particularly beneficial for generative models (e.g., GANs). What specific failure mode does it address?

??? success "익힘 2 풀이"
    In generative models, the generator must map from a latent vector to a spatially organized output image. With standard transposed convolutions, the generator has no built-in notion of absolute position, so it must implicitly learn spatial structure from the data. This often leads to repeating patterns and difficulty generating content that varies systematically with position (e.g., placing an object at a specific location). CoordConv gives the generator explicit access to spatial coordinates, allowing it to learn position-dependent generation rules directly. This reduces artifacts like checkerboard patterns and improves the generator's ability to control spatial placement of generated content.

---

**익힘 3.**
Design a `RelativeCoordConv` variant where, instead of absolute coordinates, each position receives coordinates relative to a given reference point $(r_x, r_y)$. This could be useful for tasks like keypoint detection.

??? success "익힘 3 풀이"
    ```python
    class RelativeAddCoords(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, ref_x, ref_y):
            # x: (B, C, H, W), ref_x/ref_y: (B,) in [-1, 1]
            B, _, H, W = x.size()
            xx = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
            yy = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
            # Subtract reference point
            dx = xx - ref_x.view(B, 1, 1, 1)
            dy = yy - ref_y.view(B, 1, 1, 1)
            dr = torch.sqrt(dx ** 2 + dy ** 2)
            return torch.cat([x, dx, dy, dr], dim=1)
    ```
    This provides each spatial position with its displacement from the reference point, enabling the network to reason about relative distances and directions.
