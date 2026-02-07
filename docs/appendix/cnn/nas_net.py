#!/usr/bin/env python3
'''
NASNet - Neural Architecture Search Network
Paper: "Learning Transferable Architectures for Scalable Image Recognition" (2018)
Key: Architecture discovered by neural architecture search (AutoML)
'''
import torch
import torch.nn as nn

class NASNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(1056, num_classes)
    
    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

if __name__ == "__main__":
    model = NASNet()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
