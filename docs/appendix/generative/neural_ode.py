#!/usr/bin/env python3
'''
Neural ODEs - Neural Ordinary Differential Equations
Paper: "Neural Ordinary Differential Equations" (2018)
Won NeurIPS 2018 Best Paper Award
Key: Continuous depth models, memory-efficient backpropagation
'''
import torch
import torch.nn as nn

class ODEFunc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 64),
            nn.Tanh(),
            nn.Linear(64, dim),
        )
    
    def forward(self, t, y):
        return self.net(y)

class ODEBlock(nn.Module):
    def __init__(self, odefunc, method='euler', step_size=0.1):
        super().__init__()
        self.odefunc = odefunc
        self.method = method
        self.step_size = step_size
    
    def forward(self, x):
        # Simple Euler integration (for demonstration)
        t = 0
        t_end = 1
        
        while t < t_end:
            dx = self.odefunc(t, x)
            x = x + self.step_size * dx
            t += self.step_size
        
        return x

class NeuralODE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, num_classes=10):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.odeblock = ODEBlock(ODEFunc(hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.odeblock(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = NeuralODE()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x = torch.randn(32, 784)
    print(f"Input: {x.shape}, Output: {model(x).shape}")
