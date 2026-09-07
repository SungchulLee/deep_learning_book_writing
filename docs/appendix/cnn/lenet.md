# LeNet

LeNet-5, published in the 1998 paper "Gradient-Based Learning Applied to Document Recognition" by Yann LeCun et al., is one of the earliest and most influential convolutional neural network architectures. Originally designed for handwritten digit recognition on the MNIST dataset, LeNet demonstrated that neural networks with learned convolutional features could outperform hand-engineered feature extractors. Its design introduced the foundational CNN pattern of alternating convolution and pooling layers followed by fully connected layers.

## 코드

```python
#!/usr/bin/env python3
"""
LeNet-5 - Convolutional Neural Network
Paper: "Gradient-Based Learning Applied to Document Recognition" (1998)
Authors: Yann LeCun et al.
Key: Early CNN architecture using convolution, average pooling, and fully
connected layers; widely used for MNIST digit classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================================
# 메인
# ========================================================================


class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # Fully connected layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        x = F.relu(self.conv1(x))      # -> (batch, 6, 24, 24)
        x = F.avg_pool2d(x, 2)         # -> (batch, 6, 12, 12)

        x = F.relu(self.conv2(x))      # -> (batch, 16, 8, 8)
        x = F.avg_pool2d(x, 2)         # -> (batch, 16, 4, 4)

        x = torch.flatten(x, 1)        # -> (batch, 16*4*4)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = LeNet()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    print(y.shape)  # torch.Size([1, 10])
```

## 논의

LeNet-5 established the template that nearly all modern CNNs follow. The architecture applies two convolutional layers with increasing channel counts (1 to 6 to 16), each followed by average pooling that halves the spatial dimensions. The resulting feature maps are flattened and passed through three fully connected layers that progressively reduce dimensionality (256 to 120 to 84 to 10). Each layer applies a nonlinear activation, allowing the network to learn hierarchical feature representations.

The convolutional layers learn local spatial features: the first layer typically learns edge and gradient detectors, while the second layer combines these into higher-level patterns like curves and corners. Average pooling provides translation invariance and spatial compression, though modern architectures have largely replaced it with max pooling or strided convolutions. The fully connected layers serve as a classifier operating on the learned feature representation.

Despite its simplicity by modern standards, LeNet contains key principles that remain relevant: weight sharing through convolutions (reducing parameters compared to fully connected layers), spatial hierarchy through stacked convolutions and pooling, and end-to-end training with backpropagation. With only about 60,000 parameters, LeNet achieves over 99% accuracy on MNIST, demonstrating remarkable efficiency for its task.

## 익힘 문제

**익힘 1.**
Trace the tensor shapes through the entire LeNet forward pass for an input of shape $(32, 1, 28, 28)$.

??? success "익힘 1 풀이"
    Input: $(32, 1, 28, 28)$. After conv1 ($5 \times 5$, 6 filters): $(32, 6, 24, 24)$ since $28 - 5 + 1 = 24$. After avg_pool2d (kernel 2): $(32, 6, 12, 12)$. After conv2 ($5 \times 5$, 16 filters): $(32, 16, 8, 8)$ since $12 - 5 + 1 = 8$. After avg_pool2d (kernel 2): $(32, 16, 4, 4)$. After flatten: $(32, 256)$ since $16 \times 4 \times 4 = 256$. After fc1: $(32, 120)$. After fc2: $(32, 84)$. After fc3: $(32, 10)$.

---

**익힘 2.**
Calculate the total number of learnable parameters in LeNet-5 (including biases).

??? success "익힘 2 풀이"
    conv1: $1 \times 6 \times 5 \times 5 + 6 = 156$ parameters. conv2: $6 \times 16 \times 5 \times 5 + 16 = 2,416$ parameters. fc1: $256 \times 120 + 120 = 30,840$ parameters. fc2: $120 \times 84 + 84 = 10,164$ parameters. fc3: $84 \times 10 + 10 = 850$ parameters. Total: $156 + 2,416 + 30,840 + 10,164 + 850 = 44,426$ parameters. Note that the vast majority (about 93%) of parameters are in the fully connected layers.

---

**익힘 3.**
Modify LeNet to accept $32 \times 32$ RGB images (like CIFAR-10) and use max pooling instead of average pooling. Update all dimension calculations.

??? success "익힘 3 풀이"
    ```python
    class LeNetCIFAR(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)       # (B,3,32,32) -> (B,6,28,28)
            self.conv2 = nn.Conv2d(6, 16, 5)      # (B,6,14,14) -> (B,16,10,10)
            self.fc1 = nn.Linear(16 * 5 * 5, 120) # after pool: (B,16,5,5) -> 400
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # (B,6,14,14)
            x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # (B,16,5,5)
            x = torch.flatten(x, 1)                      # (B,400)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
    ```
    The input changes from 1 channel to 3, spatial dimensions change from 28 to 32, and the flattened size becomes $16 \times 5 \times 5 = 400$ instead of $16 \times 4 \times 4 = 256$.
