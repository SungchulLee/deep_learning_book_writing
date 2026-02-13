"""
Module 40.3: Sparse Autoencoder

This script introduces sparse autoencoders, which learn sparse representations
by penalizing the activation of hidden units. Sparse coding encourages the model
to learn a overcomplete basis where only a small subset of neurons are active
for any given input, leading to more interpretable features.

Key Concepts:
- L1 regularization on activations
- KL divergence sparsity constraint
- Learning interpretable features
- Sparse vs. dense representations
- Relationship to sparse coding

Mathematical Foundation:
-----------------------
Standard Autoencoder Loss:
L = ||x - f(x)||²

Sparse Autoencoder Loss (L1 regularization):
L = ||x - f(x)||² + λ Σⱼ |hⱼ|

Where:
- hⱼ is the activation of neuron j in latent layer
- λ is sparsity regularization strength
- Σⱼ |hⱼ| encourages many activations to be zero

Alternative: KL Divergence Sparsity:
L = ||x - f(x)||² + β Σⱼ KL(ρ || ρ̂ⱼ)

Where:
- ρ is target sparsity level (e.g., 0.05)
- ρ̂ⱼ is average activation of neuron j
- KL(ρ || ρ̂ⱼ) = ρ log(ρ/ρ̂ⱼ) + (1-ρ) log((1-ρ)/(1-ρ̂ⱼ))
- β is sparsity weight

Sparsity encourages:
1. Selective feature activation
2. Interpretable representations
3. Robustness to noise
4. Better generalization

Time: 50 minutes
Level: Intermediate
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


# =============================================================================
# PART 1: SPARSE AUTOENCODER WITH L1 REGULARIZATION
# =============================================================================

class SparseAutoencoder_L1(nn.Module):
    """
    Sparse Autoencoder using L1 regularization on latent activations.
    
    Loss = Reconstruction Loss + λ * L1(latent activations)
    
    The L1 penalty encourages many latent activations to be exactly zero,
    promoting sparse representations.
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 128):
        super(SparseAutoencoder_L1, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder: Note we often use larger latent_dim for sparse AE
        # to create overcomplete representations
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()  # ReLU naturally promotes sparsity
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning reconstruction and latent activations."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def l1_loss(latent: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 penalty on latent activations.
    
    L1(h) = Σᵢⱼ |hᵢⱼ|
    
    This encourages sparsity by penalizing non-zero activations.
    
    Parameters:
    -----------
    latent : torch.Tensor, shape (batch_size, latent_dim)
        Latent activations
        
    Returns:
    --------
    l1_penalty : torch.Tensor, scalar
        Average L1 norm across batch
    """
    return torch.mean(torch.abs(latent))


# =============================================================================
# PART 2: SPARSE AUTOENCODER WITH KL DIVERGENCE
# =============================================================================

class SparseAutoencoder_KL(nn.Module):
    """
    Sparse Autoencoder using KL divergence sparsity constraint.
    
    This approach constrains the average activation of each neuron
    to be close to a target sparsity level ρ (e.g., 0.05).
    
    Loss = Reconstruction Loss + β * KL_divergence_penalty
    """
    
    def __init__(self, input_dim: int = 784, latent_dim: int = 128):
        super(SparseAutoencoder_KL, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.Sigmoid()  # Sigmoid for KL divergence (outputs in [0,1])
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def kl_divergence_loss(latent: torch.Tensor, rho: float = 0.05) -> torch.Tensor:
    """
    Compute KL divergence sparsity penalty.
    
    For each neuron j, we want its average activation ρ̂ⱼ ≈ ρ.
    
    KL(ρ || ρ̂ⱼ) = ρ log(ρ/ρ̂ⱼ) + (1-ρ) log((1-ρ)/(1-ρ̂ⱼ))
    
    This is minimized when ρ̂ⱼ = ρ, encouraging sparse activations.
    
    Parameters:
    -----------
    latent : torch.Tensor, shape (batch_size, latent_dim)
        Latent activations (should be in [0, 1] from sigmoid)
    rho : float
        Target sparsity level (e.g., 0.05 means 5% activation)
        
    Returns:
    --------
    kl_penalty : torch.Tensor, scalar
        KL divergence penalty averaged over all neurons
    """
    # Compute average activation for each neuron across batch
    # Shape: (latent_dim,)
    rho_hat = torch.mean(latent, dim=0)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    rho_hat = torch.clamp(rho_hat, eps, 1 - eps)
    
    # Compute KL divergence for each neuron
    # KL(ρ || ρ̂) = ρ log(ρ/ρ̂) + (1-ρ) log((1-ρ)/(1-ρ̂))
    kl = rho * torch.log(rho / rho_hat) + \
         (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    
    # Sum over all neurons and average
    return torch.sum(kl)


# =============================================================================
# PART 3: TRAINING FUNCTION FOR SPARSE AUTOENCODERS
# =============================================================================

def train_sparse_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    sparsity_type: str = 'l1',
    sparsity_weight: float = 0.001,
    rho: float = 0.05
) -> Tuple[float, float, float]:
    """
    Train sparse autoencoder for one epoch.
    
    Total Loss = Reconstruction Loss + Sparsity Penalty
    
    Parameters:
    -----------
    model : nn.Module
        Sparse autoencoder model
    train_loader : DataLoader
        Training data
    optimizer : optim.Optimizer
        Optimizer
    device : torch.device
        Device for computation
    epoch : int
        Current epoch
    sparsity_type : str
        'l1' or 'kl' for sparsity constraint type
    sparsity_weight : float
        Weight for sparsity penalty (λ or β)
    rho : float
        Target sparsity for KL divergence
        
    Returns:
    --------
    avg_total_loss : float
        Average total loss
    avg_recon_loss : float
        Average reconstruction loss
    avg_sparsity_loss : float
        Average sparsity penalty
    """
    model.train()
    
    total_loss_sum = 0.0
    recon_loss_sum = 0.0
    sparsity_loss_sum = 0.0
    num_batches = 0
    
    # MSE loss for reconstruction
    recon_criterion = nn.MSELoss()
    
    for batch_idx, (images, _) in enumerate(train_loader):
        # Flatten images
        images = images.view(images.size(0), -1).to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        reconstructed, latent = model(images)
        
        # Reconstruction loss
        recon_loss = recon_criterion(reconstructed, images)
        
        # Sparsity penalty
        if sparsity_type == 'l1':
            sparsity_loss = l1_loss(latent)
        elif sparsity_type == 'kl':
            sparsity_loss = kl_divergence_loss(latent, rho)
        else:
            raise ValueError(f"Unknown sparsity type: {sparsity_type}")
        
        # Total loss = Reconstruction + λ * Sparsity
        total_loss = recon_loss + sparsity_weight * sparsity_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss_sum += total_loss.item()
        recon_loss_sum += recon_loss.item()
        sparsity_loss_sum += sparsity_loss.item()
        num_batches += 1
        
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                  f'Total Loss: {total_loss.item():.6f}, '
                  f'Recon: {recon_loss.item():.6f}, '
                  f'Sparse: {sparsity_loss.item():.6f}')
    
    # Return average losses
    avg_total = total_loss_sum / num_batches
    avg_recon = recon_loss_sum / num_batches
    avg_sparse = sparsity_loss_sum / num_batches
    
    return avg_total, avg_recon, avg_sparse


# =============================================================================
# PART 4: SPARSITY ANALYSIS AND VISUALIZATION
# =============================================================================

def analyze_sparsity(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    num_samples: int = 1000
) -> Tuple[float, np.ndarray]:
    """
    Analyze sparsity of learned representations.
    
    Metrics:
    - Lifetime sparsity: For each neuron, what fraction of samples activate it?
    - Population sparsity: For each sample, what fraction of neurons are active?
    
    Parameters:
    -----------
    model : nn.Module
        Trained sparse autoencoder
    test_loader : DataLoader
        Test data
    device : torch.device
        Device
    num_samples : int
        Number of samples to analyze
        
    Returns:
    --------
    population_sparsity : float
        Average fraction of active neurons per sample
    lifetime_sparsity : np.ndarray, shape (latent_dim,)
        Fraction of samples that activate each neuron
    """
    model.eval()
    
    all_activations = []
    
    with torch.no_grad():
        for images, _ in test_loader:
            if len(all_activations) * test_loader.batch_size >= num_samples:
                break
            
            images = images.view(images.size(0), -1).to(device)
            _, latent = model(images)
            all_activations.append(latent.cpu().numpy())
    
    # Concatenate all activations
    all_activations = np.concatenate(all_activations, axis=0)[:num_samples]
    # Shape: (num_samples, latent_dim)
    
    # Define "active" as activation > threshold (e.g., 0.1)
    threshold = 0.1
    active = all_activations > threshold
    
    # Population sparsity: Average fraction of active neurons per sample
    population_sparsity = np.mean(np.mean(active, axis=1))
    
    # Lifetime sparsity: Fraction of samples that activate each neuron
    lifetime_sparsity = np.mean(active, axis=0)
    
    return population_sparsity, lifetime_sparsity


def visualize_sparsity_analysis(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
):
    """
    Visualize sparsity statistics.
    
    Creates three plots:
    1. Histogram of population sparsity across samples
    2. Histogram of lifetime sparsity across neurons
    3. Activation distribution for latent layer
    """
    model.eval()
    
    # Collect activations
    all_activations = []
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.view(images.size(0), -1).to(device)
            _, latent = model(images)
            all_activations.append(latent.cpu().numpy())
            if len(all_activations) >= 20:  # ~2500 samples
                break
    
    all_activations = np.concatenate(all_activations, axis=0)
    
    # Population and lifetime sparsity
    threshold = 0.1
    active = all_activations > threshold
    population_sparsity_per_sample = np.mean(active, axis=1)
    lifetime_sparsity_per_neuron = np.mean(active, axis=0)
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Population sparsity distribution
    axes[0].hist(population_sparsity_per_sample, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(population_sparsity_per_sample), color='red',
                   linestyle='--', label=f'Mean: {np.mean(population_sparsity_per_sample):.3f}')
    axes[0].set_xlabel('Fraction of Active Neurons', fontsize=11)
    axes[0].set_ylabel('Number of Samples', fontsize=11)
    axes[0].set_title('Population Sparsity\n(per sample)', fontsize=12)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Lifetime sparsity distribution
    axes[1].hist(lifetime_sparsity_per_neuron, bins=50, edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(lifetime_sparsity_per_neuron), color='red',
                   linestyle='--', label=f'Mean: {np.mean(lifetime_sparsity_per_neuron):.3f}')
    axes[1].set_xlabel('Fraction of Samples Activating', fontsize=11)
    axes[1].set_ylabel('Number of Neurons', fontsize=11)
    axes[1].set_title('Lifetime Sparsity\n(per neuron)', fontsize=12)
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Plot 3: Overall activation distribution
    axes[2].hist(all_activations.flatten(), bins=100, edgecolor='black', alpha=0.7)
    axes[2].axvline(threshold, color='red', linestyle='--', 
                   label=f'Threshold: {threshold}')
    axes[2].set_xlabel('Activation Value', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Activation Distribution', fontsize=12)
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sparsity_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved sparsity analysis to 'sparsity_analysis.png'")


def visualize_learned_features(
    model: nn.Module,
    num_features: int = 64
):
    """
    Visualize learned features (decoder weights).
    
    For sparse autoencoders, features are often more interpretable
    than dense autoencoders, showing localized patterns.
    
    Parameters:
    -----------
    model : nn.Module
        Trained sparse autoencoder
    num_features : int
        Number of features to visualize
    """
    model.eval()
    
    # Get decoder weights from first layer
    # Shape: (256, latent_dim) for decoder's first linear layer
    # We want to visualize what each latent dimension reconstructs
    
    # Get all decoder parameters
    decoder_weights = None
    for name, param in model.decoder.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            # First weight matrix: (output_dim, input_dim)
            # For first decoder layer: (256, latent_dim)
            # We want latent_dim features, so take transpose
            decoder_weights = param.detach().cpu().numpy().T
            break
    
    if decoder_weights is None:
        print("Could not extract decoder weights")
        return
    
    # decoder_weights shape: (latent_dim, 256)
    # We need to decode further to get (latent_dim, 784) for visualization
    # Instead, let's create one-hot latent vectors and decode them
    
    latent_dim = model.latent_dim
    num_features = min(num_features, latent_dim)
    
    features = []
    with torch.no_grad():
        for i in range(num_features):
            # Create one-hot vector
            latent = torch.zeros(1, latent_dim)
            latent[0, i] = 1.0  # Activate only neuron i
            
            # Decode to image space
            feature = model.decoder(latent)
            features.append(feature.cpu().numpy().reshape(28, 28))
    
    # Visualize features in grid
    grid_size = int(np.ceil(np.sqrt(num_features)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()
    
    for i in range(num_features):
        axes[i].imshow(features[i], cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Feature {i}', fontsize=8)
    
    # Hide unused subplots
    for i in range(num_features, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Learned Features (Decoder Basis)', fontsize=14)
    plt.tight_layout()
    plt.savefig('learned_features.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved learned features to 'learned_features.png'")


# =============================================================================
# PART 5: DATA LOADING
# =============================================================================

def load_mnist_data(batch_size: int = 128) -> Tuple[DataLoader, DataLoader]:
    """Load MNIST dataset."""
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST(root='./data', train=True,
                                   download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False,
                                  download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=2)
    
    return train_loader, test_loader


# =============================================================================
# PART 6: MAIN EXECUTION
# =============================================================================

def main():
    """Main function to train sparse autoencoder."""
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Hyperparameters
    input_dim = 784
    latent_dim = 128  # Often larger for sparse AE (overcomplete)
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 15
    
    # Sparsity configuration
    sparsity_type = 'kl'  # Options: 'l1' or 'kl'
    sparsity_weight = 0.01  # λ or β
    rho = 0.05  # Target sparsity for KL (5% activation)
    
    print("\n" + "="*60)
    print("SPARSE AUTOENCODER TRAINING")
    print("="*60)
    print(f"Sparsity type: {sparsity_type}")
    print(f"Sparsity weight: {sparsity_weight}")
    if sparsity_type == 'kl':
        print(f"Target sparsity (ρ): {rho}")
    print(f"Latent dimension: {latent_dim}")
    print("="*60 + "\n")
    
    # Load data
    print("Loading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(batch_size)
    
    # Initialize model based on sparsity type
    if sparsity_type == 'l1':
        model = SparseAutoencoder_L1(input_dim, latent_dim).to(device)
    else:  # kl
        model = SparseAutoencoder_KL(input_dim, latent_dim).to(device)
    
    print(f"\nSparse Autoencoder architecture:")
    print(model)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    total_losses = []
    recon_losses = []
    sparsity_losses = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 40)
        
        total_loss, recon_loss, sparse_loss = train_sparse_autoencoder(
            model, train_loader, optimizer, device, epoch,
            sparsity_type, sparsity_weight, rho
        )
        
        total_losses.append(total_loss)
        recon_losses.append(recon_loss)
        sparsity_losses.append(sparse_loss)
        
        print(f"Epoch {epoch} - Total: {total_loss:.6f}, "
              f"Recon: {recon_loss:.6f}, Sparse: {sparse_loss:.6f}")
    
    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, num_epochs + 1)
    axes[0].plot(epochs, total_losses, marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(epochs, recon_losses, marker='s', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(epochs, sparsity_losses, marker='^', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Sparsity Penalty')
    axes[2].set_title(f'Sparsity Penalty ({sparsity_type.upper()})')
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sparse_training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nSaved training curves to 'sparse_training_curves.png'")
    
    # Sparsity analysis
    print("\n" + "="*60)
    print("SPARSITY ANALYSIS")
    print("="*60)
    
    pop_sparsity, life_sparsity = analyze_sparsity(model, test_loader, device)
    print(f"\nPopulation sparsity: {pop_sparsity:.4f}")
    print(f"Lifetime sparsity (mean): {np.mean(life_sparsity):.4f}")
    print(f"Lifetime sparsity (std): {np.std(life_sparsity):.4f}")
    print(f"Inactive neurons (<1% activation): "
          f"{np.sum(life_sparsity < 0.01)}/{len(life_sparsity)}")
    
    # Visualizations
    print("\n" + "="*60)
    print("VISUALIZATIONS")
    print("="*60)
    
    print("\n1. Sparsity analysis...")
    visualize_sparsity_analysis(model, test_loader, device)
    
    print("\n2. Learned features...")
    visualize_learned_features(model, num_features=64)
    
    # Save model
    torch.save(model.state_dict(), f'sparse_autoencoder_{sparsity_type}.pth')
    print(f"\nModel saved to 'sparse_autoencoder_{sparsity_type}.pth'")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()


# =============================================================================
# EXERCISES
# =============================================================================

"""
EXERCISE 1: Sparsity Weight Tuning
-----------------------------------
Train models with different sparsity weights:
l1: λ ∈ {0.0001, 0.001, 0.01, 0.1, 1.0}
kl: β ∈ {0.001, 0.01, 0.1, 1.0, 10.0}

For each:
a) Record final losses (total, reconstruction, sparsity)
b) Measure population and lifetime sparsity
c) Visualize learned features

Questions:
- How does sparsity weight affect reconstruction quality?
- What is the optimal trade-off between sparsity and reconstruction?
- Do stronger constraints lead to more interpretable features?


EXERCISE 2: L1 vs KL Comparison
--------------------------------
Train two models with similar effective sparsity:
- L1 model with λ = 0.01
- KL model with β = 0.1, ρ = 0.05

Compare:
a) Training dynamics (loss curves)
b) Final sparsity levels
c) Learned feature quality
d) Computational cost

Questions:
- Which method produces sparser representations?
- Which features are more interpretable?
- Which method is more stable during training?


EXERCISE 3: Overcomplete Representations
-----------------------------------------
Train sparse autoencoders with different latent dimensions:
latent_dims = [64, 128, 256, 512, 1024]

Note: 784 is input dimension, so >784 is "overcomplete"

Questions:
- How does overcomplete representation affect feature quality?
- Can you learn more diverse features with larger latent dim?
- What happens to sparsity as latent_dim increases?


EXERCISE 4: Feature Selectivity
--------------------------------
Analyze which features activate for which digits:

a) For each digit class (0-9), compute average activation pattern
b) Identify features that are highly selective for specific digits
c) Visualize most and least selective features

Implementation:
- Collect activations for each digit class separately
- Compute mean and variance of activations per feature
- High selectivity: high mean for one class, low for others


EXERCISE 5: Reconstruction from Sparse Codes
---------------------------------------------
Investigate sparse coding properties:

a) Take a test image and get its latent representation
b) Gradually set more neurons to zero (starting from smallest)
c) Reconstruct from increasingly sparse codes
d) Plot: sparsity level vs. reconstruction quality

Questions:
- How many neurons are actually needed for good reconstruction?
- Which neurons are most important?
- Can you identify "basis" vs "refinement" neurons?


ADVANCED CHALLENGE: Dictionary Learning Connection
---------------------------------------------------
Sparse autoencoders are related to dictionary learning:

1. Extract decoder weights as dictionary atoms
2. Compare with K-SVD or OMP dictionary learning
3. Use learned features for classification task
4. Compare with PCA features

Can sparse autoencoder features outperform PCA for downstream tasks?
"""
