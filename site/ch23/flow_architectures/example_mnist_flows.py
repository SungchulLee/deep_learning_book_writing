"""
MNIST Normalizing Flows

Train normalizing flows on MNIST digits to demonstrate image generation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from flow_utils import BaseDistribution, FlowSequence
from coupling_flows import CouplingLayer, BatchNorm


class MNISTFlow:
    """Wrapper class for MNIST flow training."""
    
    def __init__(self, n_layers: int = 8, hidden_dim: int = 256,
                 batch_size: int = 128, lr: float = 1e-4, device: str = None):
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.lr = lr
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Build model
        dim = 28 * 28  # MNIST image size
        flows = []
        
        for i in range(n_layers):
            # Alternate masks
            mask = torch.zeros(dim)
            if i % 2 == 0:
                mask[dim // 2:] = 1
            else:
                mask[:dim // 2] = 1
            
            flows.append(CouplingLayer(dim, hidden_dim, mask))
            flows.append(BatchNorm(dim))
        
        base_dist = BaseDistribution(dim)
        self.model = FlowSequence(flows, base_dist).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def get_dataloader(self, train: bool = True):
        """Create MNIST dataloader."""
        # Dequantize and normalize to [0, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x + torch.rand_like(x) / 256.))  # Dequantization
        ])
        
        dataset = datasets.MNIST(
            root='./data',
            train=train,
            download=True,
            transform=transform
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=2,
            pin_memory=True
        )
    
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for images, _ in pbar:
            images = images.to(self.device)
            images = images.view(images.shape[0], -1)  # Flatten
            
            # Compute negative log-likelihood
            log_prob = self.model.log_prob(images)
            loss = -log_prob.mean()
            
            # Add bits per dimension metric
            bpd = loss / (28 * 28 * np.log(2))
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'bpd': f'{bpd.item():.4f}'})
        
        return total_loss / num_batches
    
    def sample_images(self, n_samples: int = 64):
        """Generate sample images."""
        self.model.eval()
        
        with torch.no_grad():
            samples = self.model.sample(n_samples, device=self.device)
            samples = samples.view(n_samples, 1, 28, 28)
            samples = torch.clamp(samples, 0, 1)
        
        return samples
    
    def train(self, n_epochs: int = 50, save_interval: int = 10):
        """Train the flow model."""
        dataloader = self.get_dataloader(train=True)
        
        os.makedirs('samples', exist_ok=True)
        losses = []
        
        print(f"\nTraining for {n_epochs} epochs...")
        print("=" * 50)
        
        for epoch in range(1, n_epochs + 1):
            print(f"\nEpoch {epoch}/{n_epochs}")
            
            avg_loss = self.train_epoch(dataloader)
            losses.append(avg_loss)
            
            print(f"Average loss: {avg_loss:.4f}")
            
            # Generate samples
            if epoch % save_interval == 0 or epoch == 1:
                print("Generating samples...")
                samples = self.sample_images(64)
                self.visualize_samples(samples, f'samples/epoch_{epoch:04d}.png')
            
            # Save checkpoint
            if epoch % 25 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        # Plot training loss
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Negative Log-Likelihood')
        plt.title('MNIST Flow Training Loss')
        plt.grid(True)
        plt.savefig('training_loss.png', dpi=150)
        plt.close()
        
        print("\nTraining complete!")
        
        return losses
    
    def visualize_samples(self, samples, filename='samples.png'):
        """Visualize generated samples."""
        from torchvision.utils import make_grid
        
        grid = make_grid(samples, nrow=8, padding=2)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(grid[0].cpu().numpy(), cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved samples to {filename}")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"Saved checkpoint to {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded checkpoint from {filename}")


import numpy as np

def main():
    """Main training script."""
    print("=" * 50)
    print("MNIST Normalizing Flow Training")
    print("=" * 50)
    
    # Initialize
    mnist_flow = MNISTFlow(
        n_layers=8,
        hidden_dim=256,
        batch_size=128,
        lr=1e-4
    )
    
    # Train
    mnist_flow.train(n_epochs=50, save_interval=10)
    
    # Generate final samples
    print("\nGenerating final samples...")
    samples = mnist_flow.sample_images(64)
    mnist_flow.visualize_samples(samples, 'final_samples.png')
    
    # Save model
    mnist_flow.save_checkpoint('mnist_flow_final.pt')
    
    print("\nDone! Check the 'samples' folder for generated images.")


if __name__ == "__main__":
    main()
