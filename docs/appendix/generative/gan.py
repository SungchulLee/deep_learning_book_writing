#!/usr/bin/env python3
'''
GAN - Generative Adversarial Networks
Paper: "Generative Adversarial Networks" (2014)
Key: Two networks (Generator and Discriminator) in adversarial training
'''
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super().__init__()
        self.img_size = img_size
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, img_size * img_size * 1),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_size=28):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size * 1, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class GAN(nn.Module):
    def __init__(self, latent_dim=100, img_size=28):
        super().__init__()
        self.generator = Generator(latent_dim, img_size)
        self.discriminator = Discriminator(img_size)
    
    def forward(self, z):
        return self.generator(z)

if __name__ == "__main__":
    model = GAN()
    print(f"Generator Parameters: {sum(p.numel() for p in model.generator.parameters()):,}")
    print(f"Discriminator Parameters: {sum(p.numel() for p in model.discriminator.parameters()):,}")
