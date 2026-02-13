"""
Example Training Script

Simple script to train the MNIST diffusion model with default settings.
Adjust hyperparameters as needed for experimentation.
"""

from mnist_diffusion import MNISTDiffusion


def main():
    # Configuration
    config = {
        'timesteps': 1000,       # Number of diffusion steps
        'batch_size': 128,       # Batch size for training
        'learning_rate': 2e-4,   # Learning rate
        'epochs': 100,           # Number of training epochs
        'save_interval': 10,     # Save samples every N epochs
    }
    
    print("Training Configuration:")
    print("-" * 40)
    for key, value in config.items():
        print(f"{key:20s}: {value}")
    print("-" * 40)
    print()
    
    # Initialize model
    diffusion = MNISTDiffusion(
        timesteps=config['timesteps'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )
    
    # Train
    diffusion.train(
        epochs=config['epochs'],
        save_interval=config['save_interval']
    )
    
    # Save final model
    diffusion.save_checkpoint('mnist_diffusion_model.pt')
    
    print("\nTraining complete! Model saved to 'mnist_diffusion_model.pt'")
    print("Check the 'samples' folder for generated images during training.")


if __name__ == "__main__":
    main()
