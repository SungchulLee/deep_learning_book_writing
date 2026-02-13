"""
Restricted Boltzmann Machines: Practical Energy-Based Models
==========================================================

RBMs are the most successful practical application of energy-based learning.
They restrict connections to be between visible and hidden layers only (bipartite),
making training tractable through Contrastive Divergence.

Learning Objectives:
-------------------
1. Understand RBM architecture and energy function
2. Implement Contrastive Divergence (CD-k) algorithm
3. Train RBMs on real data (MNIST)
4. Visualize learned features
5. Use RBMs for reconstruction and generation

Key Concepts:
------------
- Bipartite graph: no v-v or h-h connections
- Energy: E(v,h) = -aᵀv - bᵀh - vᵀWh
- Conditionals factor: P(h|v) = ∏ P(hⱼ|v), P(v|h) = ∏ P(vᵢ|h)
- Contrastive Divergence: approximate gradient
- Block Gibbs sampling

Duration: 90-120 minutes
Prerequisites: Modules 01-03
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm import tqdm

torch.manual_seed(42)
np.random.seed(42)

class RestrictedBoltzmannMachine(nn.Module):
    """
    Restricted Boltzmann Machine Implementation.
    
    RBM is a bipartite undirected graphical model with:
    - Visible layer v ∈ {0,1}ⁿ
    - Hidden layer h ∈ {0,1}ᵐ
    - No intra-layer connections
    
    Energy function:
    E(v,h) = -aᵀv - bᵀh - vᵀWh
    
    where W is the weight matrix, a is visible bias, b is hidden bias.
    """
    
    def __init__(self, n_visible, n_hidden, learning_rate=0.01, k=1):
        super().__init__()
        
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k  # CD-k steps
        
        # Initialize parameters
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.a = nn.Parameter(torch.zeros(n_visible))  # visible bias
        self.b = nn.Parameter(torch.zeros(n_hidden))   # hidden bias
        
        self.lr = learning_rate
        
    def sample_hidden(self, v):
        """Sample hidden units given visible units: P(h=1|v) = σ(Wv + b)"""
        activation = F.linear(v, self.W, self.b)
        prob = torch.sigmoid(activation)
        sample = torch.bernoulli(prob)
        return prob, sample
    
    def sample_visible(self, h):
        """Sample visible units given hidden units: P(v=1|h) = σ(Wᵀh + a)"""
        activation = F.linear(h, self.W.t(), self.a)
        prob = torch.sigmoid(activation)
        sample = torch.bernoulli(prob)
        return prob, sample
    
    def energy(self, v, h):
        """Compute energy E(v,h) = -aᵀv - bᵀh - vᵀWh"""
        return -(v @ self.a + h @ self.b + (v @ self.W.t() * h).sum(dim=1))
    
    def free_energy(self, v):
        """
        Compute free energy F(v) = -log Σₕ exp(-E(v,h))
        F(v) = -aᵀv - Σⱼ log(1 + exp(bⱼ + Wⱼv))
        """
        wx_b = F.linear(v, self.W, self.b)
        visible_term = (v * self.a).sum(dim=1)
        hidden_term = wx_b.exp().add(1).log().sum(dim=1)
        return -(visible_term + hidden_term)
    
    def contrastive_divergence(self, v0):
        """
        Contrastive Divergence k steps (CD-k) training.
        
        Approximate gradient: ∇L ≈ E_data[vh] - E_model_k[vh]
        """
        batch_size = v0.shape[0]
        
        # Positive phase: sample from data
        ph0, h0 = self.sample_hidden(v0)
        
        # Negative phase: k steps of Gibbs sampling
        vk, hk = v0, h0
        for _ in range(self.k):
            _, vk = self.sample_visible(hk)
            _, hk = self.sample_hidden(vk)
        
        # Compute positive and negative gradients
        positive_grad = torch.matmul(ph0.t(), v0)
        negative_grad = torch.matmul(hk.t(), vk)
        
        # Update parameters
        self.W.data += self.lr * (positive_grad - negative_grad) / batch_size
        self.a.data += self.lr * (v0 - vk).mean(dim=0)
        self.b.data += self.lr * (ph0 - hk).mean(dim=0)
        
        # Compute reconstruction error for monitoring
        recon_error = ((v0 - vk)**2).sum(dim=1).mean()
        
        return recon_error.item()
    
    def reconstruct(self, v):
        """Reconstruct visible units: v → h → v'"""
        _, h = self.sample_hidden(v)
        _, v_recon = self.sample_visible(h)
        return v_recon

def train_rbm_mnist():
    """Train RBM on MNIST dataset."""
    print("\n" + "="*70)
    print("TRAINING RBM ON MNIST")
    print("="*70)
    
    # Load MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())  # Binarize
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, 
                                   download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # Create RBM
    n_visible = 784
    n_hidden = 256
    rbm = RestrictedBoltzmannMachine(n_visible, n_hidden, learning_rate=0.01, k=1)
    
    print(f"\nRBM Architecture:")
    print(f"  Visible units: {n_visible}")
    print(f"  Hidden units: {n_hidden}")
    print(f"  CD-k steps: {rbm.k}")
    
    # Training loop
    n_epochs = 10
    errors = []
    
    for epoch in range(n_epochs):
        epoch_error = 0
        n_batches = 0
        
        for batch_idx, (data, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            data = data.view(-1, 784)
            error = rbm.contrastive_divergence(data)
            epoch_error += error
            n_batches += 1
        
        avg_error = epoch_error / n_batches
        errors.append(avg_error)
        print(f"Epoch {epoch+1}/{n_epochs}, Reconstruction Error: {avg_error:.4f}")
    
    # Visualize filters
    fig, axes = plt.subplots(8, 16, figsize=(16, 8))
    for i, ax in enumerate(axes.flat):
        if i < n_hidden:
            filter_img = rbm.W[i].detach().numpy().reshape(28, 28)
            ax.imshow(filter_img, cmap='gray')
        ax.axis('off')
    plt.suptitle('Learned RBM Filters (Features)', fontsize=14)
    plt.tight_layout()
    plt.savefig('04_rbm_filters.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Test reconstruction
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)
    test_samples = test_dataset.data[:10].float() / 255.0
    test_samples = (test_samples > 0.5).float().view(10, -1)
    
    reconstructed = rbm.reconstruct(test_samples).detach()
    
    fig, axes = plt.subplots(2, 10, figsize=(15, 3))
    for i in range(10):
        axes[0, i].imshow(test_samples[i].view(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed[i].view(28, 28), cmap='gray')
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Reconstructed', fontsize=12)
    plt.tight_layout()
    plt.savefig('04_rbm_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✓ RBM training complete")
    return rbm

def main():
    print("="*70)
    print("RESTRICTED BOLTZMANN MACHINES")
    print("="*70)
    
    train_rbm_mnist()
    
    print("\n" + "="*70)
    print("MODULE COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✓ RBMs use bipartite architecture for tractable inference")
    print("  ✓ Contrastive Divergence enables practical training")
    print("  ✓ RBMs learn useful feature representations")
    print("\nNext: 05_contrastive_divergence.py")

if __name__ == "__main__":
    main()
