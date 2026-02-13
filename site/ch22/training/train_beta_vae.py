"""
Training script for β-VAE (Beta-VAE)
Learns disentangled representations with β parameter
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import sys
sys.path.append('..')
from models.beta_vae import BetaVAE, ConvBetaVAE
from utils.losses import beta_vae_loss
from utils.visualization import (
    visualize_reconstruction,
    visualize_samples,
    visualize_latent_traversal
)


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    train_recon = 0
    train_kl = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, _ in pbar:
        data = data.to(device)
        
        # Flatten data for FC β-VAE
        if isinstance(model, BetaVAE):
            data_input = data.view(data.size(0), -1)
        else:
            data_input = data
        
        # Forward pass
        reconstruction, mu, logvar = model(data_input)
        
        # Compute loss
        if isinstance(model, BetaVAE):
            target = data.view(data.size(0), -1)
        else:
            target = data
        
        loss, recon_loss, kl_loss = model.loss_function(reconstruction, target, mu, logvar)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        train_loss += loss.item()
        train_recon += recon_loss.item()
        train_kl += kl_loss.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item() / data.size(0),
            'recon': recon_loss.item() / data.size(0),
            'kl': kl_loss.item() / data.size(0)
        })
    
    num_samples = len(train_loader.dataset)
    return train_loss / num_samples, train_recon / num_samples, train_kl / num_samples


def test_epoch(model, test_loader, device):
    """Evaluate on test set"""
    model.eval()
    test_loss = 0
    test_recon = 0
    test_kl = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            
            # Flatten data for FC β-VAE
            if isinstance(model, BetaVAE):
                data_input = data.view(data.size(0), -1)
            else:
                data_input = data
            
            # Forward pass
            reconstruction, mu, logvar = model(data_input)
            
            # Compute loss
            if isinstance(model, BetaVAE):
                target = data.view(data.size(0), -1)
            else:
                target = data
            
            loss, recon_loss, kl_loss = model.loss_function(reconstruction, target, mu, logvar)
            
            test_loss += loss.item()
            test_recon += recon_loss.item()
            test_kl += kl_loss.item()
    
    num_samples = len(test_loader.dataset)
    return test_loss / num_samples, test_recon / num_samples, test_kl / num_samples


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    if args.model_type == 'fc':
        model = BetaVAE(
            input_dim=784,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            beta=args.beta
        )
    elif args.model_type == 'conv':
        model = ConvBetaVAE(
            latent_dim=args.latent_dim,
            beta=args.beta,
            img_channels=1
        )
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    
    model = model.to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Beta (disentanglement weight): {args.beta}")
    print(f"  β=1: Standard VAE")
    print(f"  β>1: More disentangled (β=4-10 typical)")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_test_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(model, train_loader, optimizer, device)
        
        # Test
        test_loss, test_recon, test_kl = test_epoch(model, test_loader, device)
        
        # Print statistics
        print(f"Train - Loss: {train_loss:.4f}, Recon: {train_recon:.4f}, KL: {train_kl:.4f}")
        print(f"Test  - Loss: {test_loss:.4f}, Recon: {test_recon:.4f}, KL: {test_kl:.4f}")
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
                'beta': args.beta,
            }, args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")
    
    # Load best model and visualize
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nGenerating visualizations...")
    
    # Standard visualizations
    visualize_reconstruction(model, test_loader, num_images=10, device=device, conditional=False)
    visualize_samples(model, args.latent_dim, num_samples=10, device=device)
    
    # β-VAE specific: Latent traversals to show disentanglement
    print("\nGenerating latent dimension traversals (disentanglement visualization)...")
    num_dims_to_traverse = min(10, args.latent_dim)
    for dim_idx in range(num_dims_to_traverse):
        print(f"  Traversing dimension {dim_idx}...")
        visualize_latent_traversal(model, dim_idx=dim_idx, num_steps=10, range_limit=3.0, device=device)
    
    print("\nTraining complete!")
    print(f"\nInterpretation tip:")
    print(f"  Look at the latent traversal images to see what each dimension encodes.")
    print(f"  With β={args.beta}, dimensions should encode independent factors of variation")
    print(f"  (e.g., one dimension for rotation, another for thickness, etc.)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train β-VAE on MNIST for disentangled representations')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='fc', choices=['fc', 'conv'],
                        help='Type of β-VAE (fc or conv)')
    parser.add_argument('--latent-dim', type=int, default=10,
                        help='Latent dimension (smaller for better disentanglement visualization)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension (for FC β-VAE)')
    parser.add_argument('--beta', type=float, default=4.0,
                        help='Beta parameter (1.0=standard VAE, 4-10=disentangled)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # Checkpoint
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/beta_vae_model.pt',
                        help='Path to save checkpoint')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    import os
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    main(args)
