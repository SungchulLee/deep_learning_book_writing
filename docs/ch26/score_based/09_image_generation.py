"""
FILE: 09_image_generation.py
DIFFICULTY: Advanced
ESTIMATED TIME: 5-6 hours
PREREQUISITES: 07-08, CNNs, U-Net architecture

LEARNING OBJECTIVES:
    1. Implement U-Net score model for images
    2. Train on MNIST digit generation
    3. Implement predictor-corrector sampling
    4. Understand computational considerations

MATHEMATICAL BACKGROUND:
    For images, score networks typically use U-Net architecture:
    - Encoder: downsample + extract features
    - Decoder: upsample + reconstruct
    - Skip connections: preserve spatial information
    
    The score s_θ(x, t) is a vector field over image space.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for score modeling on small images (28x28).
    
    This is a teaching implementation - production code would be more complex.
    """
    
    def __init__(self, in_channels=1, base_channels=64, time_emb_dim=128):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.SiLU()
        )
        
        # Middle
        self.middle = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*2, 3, padding=1),
            nn.GroupNorm(8, base_channels*2),
            nn.SiLU()
        )
        
        # Decoder
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        self.dec1 = nn.Conv2d(base_channels, in_channels, 3, padding=1)
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t.view(-1, 1))
        
        # Encode
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        
        # Middle
        h = self.middle(h2)
        
        # Decode
        h = self.dec2(h)
        h = h + h1  # Skip connection
        out = self.dec1(h)
        
        return out


def demo_mnist():
    """Train score model on MNIST (simplified demo)."""
    print("Score-Based Image Generation on MNIST")
    print("=" * 80)
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Scale to [-1, 1]
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='/tmp/mnist', train=True, download=True, transform=transform
    )
    
    # Use subset for quick demo
    subset = torch.utils.data.Subset(dataset, range(1000))
    loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)
    
    print(f"Dataset: {len(subset)} images")
    
    # Create model
    model = SimpleUNet(in_channels=1, base_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Simple training (just a few epochs for demo)
    print("\nTraining (demo with few epochs)...")
    for epoch in range(5):
        total_loss = 0
        for images, _ in loader:
            # Add noise (simple DSM)
            noise = torch.randn_like(images) * 0.5
            noisy_images = images + noise
            
            # Random time
            t = torch.rand(len(images))
            
            # Predict score
            pred_score = model(noisy_images, t)
            target_score = -noise / (0.5 ** 2)
            
            loss = nn.functional.mse_loss(pred_score, target_score)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/5 | Loss: {total_loss/len(loader):.6f}")
    
    print("\n✓ Training complete!")
    print("\nNote: Full training would take several hours on GPU.")
    print("This demo shows the architecture and training loop.")


if __name__ == "__main__":
    demo_mnist()
