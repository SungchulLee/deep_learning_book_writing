"""
Training script for basic Autoencoder
"""

import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import sys
sys.path.append('..')
from models.autoencoder import SimpleAutoencoder
from utils.visualization import visualize_reconstruction


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    train_loss = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for data, _ in pbar:
        data = data.to(device)
        data_flat = data.view(data.size(0), -1)
        
        # Forward pass
        reconstruction = model(data_flat)
        loss = model.loss_function(reconstruction, data_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item() / data.size(0)})
    
    return train_loss / len(train_loader.dataset)


def test_epoch(model, test_loader, device):
    """Evaluate on test set"""
    model.eval()
    test_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            data_flat = data.view(data.size(0), -1)
            
            reconstruction = model(data_flat)
            loss = model.loss_function(reconstruction, data_flat)
            test_loss += loss.item()
    
    return test_loss / len(test_loader.dataset)


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
    model = SimpleAutoencoder(
        input_dim=784,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim
    )
    model = model.to(device)
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Latent dimension: {args.latent_dim}")
    print(f"Hidden dimension: {args.hidden_dim}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    best_test_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Test
        test_loss = test_epoch(model, test_loader, device)
        
        # Print statistics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss:  {test_loss:.4f}")
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': test_loss,
            }, args.checkpoint_path)
            print(f"Saved checkpoint to {args.checkpoint_path}")
    
    # Load best model and visualize
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\nGenerating visualizations...")
    visualize_reconstruction(model, test_loader, num_images=10, device=device, conditional=False)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Autoencoder on MNIST')
    
    # Model arguments
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    
    # Checkpoint
    parser.add_argument('--checkpoint-path', type=str, default='checkpoints/autoencoder_model.pt',
                        help='Path to save checkpoint')
    
    args = parser.parse_args()
    
    # Create checkpoint directory
    import os
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    main(args)
