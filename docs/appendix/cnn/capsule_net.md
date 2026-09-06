# Capsule Net

Capsule Networks, introduced in the 2017 paper "Dynamic Routing Between Capsules," represent a fundamentally different approach to visual recognition. Unlike traditional CNNs that use scalar activations, capsules output vectors that encode both the probability and the pose (position, orientation, scale) of detected entities. This design addresses the inability of standard CNNs to model part-whole relationships and spatial hierarchies.

## Code

```python
#!/usr/bin/env python3
'''
CapsNet - Capsule Networks
Paper: "Dynamic Routing Between Capsules" (2017)
Key: Capsules with vector outputs, dynamic routing
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# Main
# ========================================================================

class PrimaryCaps(nn.Module):
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32):
        super().__init__()
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=2, padding=0)
            for _ in range(num_capsules)
        ])
    
    def forward(self, x):
        outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]
        outputs = torch.cat(outputs, dim=-1)
        return self.squash(outputs)
    
    def squash(self, tensor):
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)

class DigitCaps(nn.Module):
    def __init__(self, num_capsules=10, num_routes=32 * 6 * 6, in_channels=8, out_channels=16):
        super().__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(1, 2)
        x = torch.stack([x] * self.num_capsules, dim=2).unsqueeze(4)
        
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, x)
        
        # Routing
        b_ij = torch.zeros(batch_size, self.num_routes, self.num_capsules, 1)
        if x.is_cuda:
            b_ij = b_ij.cuda()
        
        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)
            v_j = self.squash(s_j)
            
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(3, 4), torch.cat([v_j] * self.num_routes, dim=1))
                b_ij = b_ij + a_ij
        
        return v_j.squeeze(1)
    
    def squash(self, tensor):
        squared_norm = (tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * tensor / torch.sqrt(squared_norm + 1e-8)

class CapsNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps(num_capsules=num_classes)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(16 * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )
    
    def forward(self, x, y=None):
        x = F.relu(self.conv1(x))
        x = self.primary_capsules(x)
        x = self.digit_capsules(x)
        
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        
        if y is None:
            _, max_length_indices = classes.max(dim=1)
            y = torch.eye(classes.size(1)).cuda().index_select(dim=0, index=max_length_indices)
        
        reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        
        return classes, reconstructions

if __name__ == "__main__":
    model = CapsNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## Discussion

The fundamental innovation in CapsNet is the concept of a capsule: a group of neurons whose activity vector represents the instantiation parameters of a specific type of entity. The length of the capsule's output vector represents the probability that the entity exists, while the orientation encodes properties such as pose, deformation, and texture. This is in stark contrast to conventional CNNs where scalar activations lose spatial relationship information through pooling operations.

The squash function is the nonlinearity applied to capsule outputs. It shrinks short vectors to near zero and long vectors to just below unit length, ensuring the vector length can be interpreted as a probability. The formula $v = \frac{\|s\|^2}{1 + \|s\|^2} \cdot \frac{s}{\|s\|}$ preserves the direction of the input while normalizing its magnitude.

Dynamic routing is the mechanism by which lower-level capsules decide which higher-level capsule to send their output to. Over multiple iterations, coupling coefficients $c_{ij}$ are refined so that each lower-level capsule routes its output to the higher-level capsule whose current output is most aligned with the prediction from that lower-level capsule. The decoder network serves as a regularizer by reconstructing the input image from the digit capsule outputs, encouraging the capsules to encode meaningful instantiation parameters.

## Exercises

**Exercise 1.**
Compute the output shape of the `PrimaryCaps` layer when the input tensor has shape `(batch=4, channels=256, height=20, width=20)` with `num_capsules=8` and `out_channels=32`.

??? success "Solution to Exercise 1"
    Each of the 8 convolutional capsules applies a $9 \times 9$ convolution with stride 2 and no padding to a $20 \times 20$ feature map. The output spatial size is $\lfloor (20 - 9) / 2 \rfloor + 1 = 6$. Each capsule produces shape $(4, 32, 6, 6)$, which is reshaped to $(4, 1152, 1)$. Concatenating 8 capsules along the last dimension gives $(4, 1152, 8)$. After the squash function, the output shape remains $(4, 1152, 8)$.

---

**Exercise 2.**
Explain why the squash function is preferred over a simple sigmoid or softmax normalization for capsule outputs. What property does it preserve that these alternatives would not?

??? success "Solution to Exercise 2"
    The squash function preserves the direction of the input vector while constraining its magnitude to be between 0 and 1. A sigmoid applied element-wise would independently scale each component, destroying the directional information that encodes pose parameters. A softmax would normalize across components to sum to 1, which is also inappropriate since the components represent different instantiation parameters (e.g., position, rotation), not a probability distribution. The squash function uniquely maintains that the orientation of the vector encodes "what" the entity looks like while the length encodes "whether" it exists.

---

**Exercise 3.**
Modify the `CapsNet` architecture to accept RGB images of size $32 \times 32$ (like CIFAR-10) instead of grayscale $28 \times 28$ images. Specify the necessary changes to layer dimensions and the decoder output size.

??? success "Solution to Exercise 3"
    The required changes are: (1) Change `self.conv1` input channels from 1 to 3: `nn.Conv2d(3, 256, kernel_size=9, stride=1)`. The output spatial size after conv1 becomes $(32 - 9 + 1) = 24$. (2) After `PrimaryCaps` with its $9 \times 9$ stride-2 convolution, the spatial size becomes $\lfloor (24 - 9) / 2 \rfloor + 1 = 8$. Update `DigitCaps` accordingly: `num_routes = 32 * 8 * 8 = 2048`. (3) Change the decoder output from 784 to $3 \times 32 \times 32 = 3072$: replace `nn.Linear(1024, 784)` with `nn.Linear(1024, 3072)`.
