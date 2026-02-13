"""
Generate and visualize samples from trained VAE models
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import sys
sys.path.append('..')
from models.vae import VAE
from models.conv_vae import ConvVAE
from models.conditional_vae import ConditionalVAE
from models.conv_cvae import ConvConditionalVAE
from models.beta_vae import BetaVAE, ConvBetaVAE
from utils.visualization import (
    visualize_reconstruction,
    visualize_samples,
    visualize_latent_traversal,
    visualize_interpolation,
    plot_latent_space
)


def load_model(model_type, checkpoint_path, device, **model_kwargs):
    """Load trained model from checkpoint"""
    # Create model
    if model_type == 'vae':
        model = VAE(**model_kwargs)
    elif model_type == 'conv_vae':
        model = ConvVAE(**model_kwargs)
    elif model_type == 'cvae':
        model = ConditionalVAE(**model_kwargs)
    elif model_type == 'conv_cvae':
        model = ConvConditionalVAE(**model_kwargs)
    elif model_type == 'beta_vae':
        model = BetaVAE(**model_kwargs)
    elif model_type == 'conv_beta_vae':
        model = ConvBetaVAE(**model_kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded {model_type} from {checkpoint_path}")
    print(f"Training epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Training loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    return model


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    # Model kwargs
    model_kwargs = {
        'latent_dim': args.latent_dim,
    }
    
    if args.model_type in ['vae', 'beta_vae']:
        model_kwargs['input_dim'] = 784
        model_kwargs['hidden_dim'] = 256
    elif args.model_type in ['cvae']:
        model_kwargs['input_dim'] = 784
        model_kwargs['hidden_dim'] = 256
        model_kwargs['num_classes'] = 10
    elif args.model_type in ['conv_vae', 'conv_beta_vae']:
        model_kwargs['img_channels'] = 1
        model_kwargs['img_size'] = 28
    elif args.model_type in ['conv_cvae']:
        model_kwargs['img_channels'] = 1
        model_kwargs['img_size'] = 28
        model_kwargs['num_classes'] = 10
    
    if args.model_type in ['beta_vae', 'conv_beta_vae']:
        model_kwargs['beta'] = 4.0
    
    # Load model
    model = load_model(args.model_type, args.checkpoint_path, device, **model_kwargs)
    
    # Determine if conditional
    is_conditional = 'cvae' in args.model_type
    
    # Generate visualizations
    print("\n=== Generating Visualizations ===")
    
    if args.reconstruction:
        print("\n1. Reconstruction visualization...")
        visualize_reconstruction(
            model, test_loader,
            num_images=args.num_samples,
            device=device,
            conditional=is_conditional
        )
    
    if args.samples:
        print("\n2. Random sample generation...")
        if is_conditional:
            # Generate samples for each class
            for class_label in range(10):
                print(f"   Generating samples for class {class_label}...")
                visualize_samples(
                    model,
                    args.latent_dim,
                    num_samples=args.num_samples,
                    device=device,
                    class_label=class_label
                )
        else:
            visualize_samples(
                model,
                args.latent_dim,
                num_samples=args.num_samples,
                device=device
            )
    
    if args.interpolation and not is_conditional:
        print("\n3. Latent space interpolation...")
        visualize_interpolation(
            model,
            test_loader,
            device=device,
            num_steps=args.num_steps
        )
    
    if args.traversal and hasattr(model, 'traverse_latent_dimension'):
        print("\n4. Latent dimension traversals...")
        num_dims = min(args.num_traversals, args.latent_dim)
        for dim_idx in range(num_dims):
            print(f"   Traversing dimension {dim_idx}...")
            visualize_latent_traversal(
                model,
                dim_idx=dim_idx,
                num_steps=args.num_steps,
                range_limit=3.0,
                device=device
            )
    
    if args.latent_space and args.latent_dim == 2:
        print("\n5. Latent space visualization (2D only)...")
        plot_latent_space(
            model,
            test_loader,
            device=device,
            num_batches=20
        )
    
    print("\n=== All visualizations complete! ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate samples from trained VAE')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, required=True,
                        choices=['vae', 'conv_vae', 'cvae', 'conv_cvae', 'beta_vae', 'conv_beta_vae'],
                        help='Type of VAE model')
    parser.add_argument('--checkpoint-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--latent-dim', type=int, default=32,
                        help='Latent dimension (must match training)')
    
    # Generation arguments
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples to generate')
    parser.add_argument('--num-steps', type=int, default=10,
                        help='Number of steps for interpolation/traversal')
    parser.add_argument('--num-traversals', type=int, default=5,
                        help='Number of latent dimensions to traverse')
    
    # What to generate
    parser.add_argument('--reconstruction', action='store_true',
                        help='Visualize reconstructions')
    parser.add_argument('--samples', action='store_true',
                        help='Generate random samples')
    parser.add_argument('--interpolation', action='store_true',
                        help='Visualize latent space interpolation')
    parser.add_argument('--traversal', action='store_true',
                        help='Visualize latent dimension traversals')
    parser.add_argument('--latent-space', action='store_true',
                        help='Visualize 2D latent space (only for 2D latent dim)')
    parser.add_argument('--all', action='store_true',
                        help='Generate all visualizations')
    
    args = parser.parse_args()
    
    # If --all is specified, enable all visualizations
    if args.all:
        args.reconstruction = True
        args.samples = True
        args.interpolation = True
        args.traversal = True
        args.latent_space = True
    
    # If no visualization is specified, default to reconstruction and samples
    if not any([args.reconstruction, args.samples, args.interpolation, 
                args.traversal, args.latent_space]):
        args.reconstruction = True
        args.samples = True
    
    main(args)
