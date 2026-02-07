#!/usr/bin/env python3
'''
CapsNet - Capsule Networks
Paper: "Dynamic Routing Between Capsules" (2017)
Key: Capsules with vector outputs, dynamic routing
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

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
