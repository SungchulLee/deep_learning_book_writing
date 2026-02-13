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