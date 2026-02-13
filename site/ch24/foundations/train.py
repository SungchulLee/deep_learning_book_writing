"""
Training Script for Autoregressive Time Series Models

This script demonstrates how to:
1. Generate synthetic time series data
2. Prepare it for AR model training
3. Train both linear and neural AR models
4. Evaluate and visualize results
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our custom modules
from ar_model import ARModel, NeuralARModel
from data import generate_sine_wave, generate_ar_process, create_sequences, train_test_split_temporal


def train_model(model: nn.Module, 
                X_train: torch.Tensor, 
                y_train: torch.Tensor,
                X_test: torch.Tensor,
                y_test: torch.Tensor,
                n_epochs: int = 100,
                learning_rate: float = 0.01,
                verbose: bool = True) -> dict:
    """
    Train an AR model using Mean Squared Error loss.
    
    Args:
        model: The AR model to train (ARModel or NeuralARModel)
        X_train: Training input sequences
        y_train: Training target values
        X_test: Test input sequences
        y_test: Test target values
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        verbose: Whether to show progress bar
        
    Returns:
        Dictionary containing training history (losses)
    """
    
    # Loss function: Mean Squared Error
    # MSE = (1/n) * Σ(predicted - actual)²
    criterion = nn.MSELoss()
    
    # Optimizer: Adam (adaptive learning rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Store losses for plotting
    train_losses = []
    test_losses = []
    
    # Training loop
    iterator = tqdm(range(n_epochs), desc="Training") if verbose else range(n_epochs)
    
    for epoch in iterator:
        # ==================== Training Phase ====================
        model.train()  # Set model to training mode
        
        # Forward pass: compute predictions
        train_predictions = model(X_train)
        
        # Compute loss: how far are predictions from true values?
        train_loss = criterion(train_predictions, y_train)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()  # Clear old gradients
        train_loss.backward()  # Compute new gradients
        
        # Update model parameters
        optimizer.step()
        
        # ==================== Evaluation Phase ====================
        model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():  # Don't compute gradients for evaluation
            test_predictions = model(X_test)
            test_loss = criterion(test_predictions, y_test)
        
        # Store losses
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        
        # Update progress bar
        if verbose and epoch % 10 == 0:
            iterator.set_postfix({
                'train_loss': f'{train_loss.item():.4f}',
                'test_loss': f'{test_loss.item():.4f}'
            })
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses
    }


def visualize_results(data: np.ndarray,
                     model: nn.Module,
                     X_test: torch.Tensor,
                     y_test: torch.Tensor,
                     sequence_length: int,
                     train_size: int,
                     n_forecast: int = 50,
                     title: str = "AR Model Results"):
    """
    Create comprehensive visualization of model performance.
    
    Args:
        data: Original time series data
        model: Trained AR model
        X_test: Test input sequences
        y_test: Test target values
        sequence_length: Length of input sequences
        train_size: Number of training samples
        n_forecast: Number of steps to forecast into future
        title: Plot title
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # ==================== Plot 1: Predictions vs Actual ====================
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test).numpy().flatten()
    
    actual_values = y_test.numpy().flatten()
    
    axes[0].plot(actual_values, label='Actual', linewidth=2, alpha=0.7)
    axes[0].plot(test_predictions, label='Predicted', linewidth=2, alpha=0.7)
    axes[0].set_title(f"{title}: Test Set Predictions")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Value")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ==================== Plot 2: Future Forecasting ====================
    # Take the last sequence from test set as starting point
    initial_sequence = X_test[-1]
    
    # Generate future predictions
    future_predictions = model.predict_sequence(initial_sequence, n_steps=n_forecast)
    
    # Plot original data and forecast
    forecast_start = len(data) - n_forecast
    
    axes[1].plot(range(len(data)), data, label='Historical Data', 
                linewidth=2, alpha=0.7, color='blue')
    axes[1].plot(range(forecast_start, forecast_start + n_forecast), 
                future_predictions, label='Forecast', 
                linewidth=2, alpha=0.7, color='red', linestyle='--')
    
    # Add vertical line to separate history and forecast
    axes[1].axvline(x=forecast_start, color='gray', linestyle=':', 
                   linewidth=2, label='Forecast Start')
    
    axes[1].set_title(f"{title}: Future Forecasting")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Value")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """
    Main training pipeline
    """
    print("=" * 70)
    print("Autoregressive Time Series Model Training")
    print("=" * 70)
    
    # ==================== Hyperparameters ====================
    SEQUENCE_LENGTH = 10  # AR(10) - use 10 past values
    N_SAMPLES = 1000      # Generate 1000 time points
    TRAIN_RATIO = 0.8     # 80% train, 20% test
    N_EPOCHS = 200        # Training epochs
    LEARNING_RATE = 0.01  # Learning rate
    
    print(f"\nHyperparameters:")
    print(f"  Sequence Length (AR order): {SEQUENCE_LENGTH}")
    print(f"  Number of samples: {N_SAMPLES}")
    print(f"  Train/Test split: {TRAIN_RATIO:.0%}/{1-TRAIN_RATIO:.0%}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    # ==================== Generate Data ====================
    print(f"\n{'='*70}")
    print("Step 1: Generating synthetic time series data...")
    print(f"{'='*70}")
    
    # You can choose which type of data to use:
    # Option 1: Sine wave with noise
    data = generate_sine_wave(n_samples=N_SAMPLES, frequency=0.05, noise_std=0.2)
    data_name = "Sine Wave"
    
    # Option 2: True AR process (uncomment to use this instead)
    # data = generate_ar_process(n_samples=N_SAMPLES, 
    #                            coefficients=[0.7, -0.3, 0.1], 
    #                            noise_std=0.3)
    # data_name = "AR(3) Process"
    
    print(f"✓ Generated {len(data)} data points ({data_name})")
    
    # ==================== Prepare Data ====================
    print(f"\n{'='*70}")
    print("Step 2: Preparing sequences for training...")
    print(f"{'='*70}")
    
    X, y = create_sequences(data, sequence_length=SEQUENCE_LENGTH)
    print(f"✓ Created {len(X)} sequences")
    print(f"  Input shape: {X.shape}")
    print(f"  Output shape: {y.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split_temporal(X, y, TRAIN_RATIO)
    print(f"✓ Split into train/test:")
    print(f"  Train: {len(X_train)} sequences")
    print(f"  Test: {len(X_test)} sequences")
    
    # ==================== Train Linear AR Model ====================
    print(f"\n{'='*70}")
    print("Step 3a: Training Linear AR Model...")
    print(f"{'='*70}")
    
    linear_model = ARModel(order=SEQUENCE_LENGTH)
    
    # Count parameters
    n_params = sum(p.numel() for p in linear_model.parameters())
    print(f"Model has {n_params} parameters")
    
    linear_history = train_model(
        linear_model, X_train, y_train, X_test, y_test,
        n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Final train loss: {linear_history['train_losses'][-1]:.4f}")
    print(f"  Final test loss: {linear_history['test_losses'][-1]:.4f}")
    
    # Show learned coefficients
    coeffs = linear_model.get_coefficients()
    print(f"\nLearned AR coefficients:")
    for i, coef in enumerate(coeffs['coefficients']):
        print(f"  φ_{i+1} = {coef:.4f}")
    print(f"  Constant c = {coeffs['constant']:.4f}")
    
    # ==================== Train Neural AR Model ====================
    print(f"\n{'='*70}")
    print("Step 3b: Training Neural AR Model...")
    print(f"{'='*70}")
    
    neural_model = NeuralARModel(order=SEQUENCE_LENGTH, hidden_size=64)
    
    n_params = sum(p.numel() for p in neural_model.parameters())
    print(f"Model has {n_params} parameters")
    
    neural_history = train_model(
        neural_model, X_train, y_train, X_test, y_test,
        n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE
    )
    
    print(f"\n✓ Training complete!")
    print(f"  Final train loss: {neural_history['train_losses'][-1]:.4f}")
    print(f"  Final test loss: {neural_history['test_losses'][-1]:.4f}")
    
    # ==================== Visualize Results ====================
    print(f"\n{'='*70}")
    print("Step 4: Creating visualizations...")
    print(f"{'='*70}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(linear_history['train_losses'], label='Train', alpha=0.7)
    plt.plot(linear_history['test_losses'], label='Test', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Linear AR Model: Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(neural_history['train_losses'], label='Train', alpha=0.7)
    plt.plot(neural_history['test_losses'], label='Test', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Neural AR Model: Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150)
    print("✓ Saved training_curves.png")
    
    # Plot predictions for linear model
    fig = visualize_results(
        data, linear_model, X_test, y_test, 
        SEQUENCE_LENGTH, len(X_train),
        n_forecast=100, title="Linear AR Model"
    )
    plt.savefig('linear_ar_results.png', dpi=150)
    print("✓ Saved linear_ar_results.png")
    
    # Plot predictions for neural model
    fig = visualize_results(
        data, neural_model, X_test, y_test,
        SEQUENCE_LENGTH, len(X_train),
        n_forecast=100, title="Neural AR Model"
    )
    plt.savefig('neural_ar_results.png', dpi=150)
    print("✓ Saved neural_ar_results.png")
    
    # ==================== Summary ====================
    print(f"\n{'='*70}")
    print("Training Complete! Summary:")
    print(f"{'='*70}")
    print(f"\nLinear AR Model:")
    print(f"  Test MSE: {linear_history['test_losses'][-1]:.4f}")
    print(f"\nNeural AR Model:")
    print(f"  Test MSE: {neural_history['test_losses'][-1]:.4f}")
    
    if neural_history['test_losses'][-1] < linear_history['test_losses'][-1]:
        print(f"\n✓ Neural model performed better (lower test loss)")
    else:
        print(f"\n✓ Linear model performed better (lower test loss)")
    
    print(f"\nCheck the generated PNG files for visualizations!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
