"""
Autoregressive Model for Time Series

This module implements a simple AR(p) model using PyTorch.
The model learns to predict the next value in a time series based on p previous values.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class ARModel(nn.Module):
    """
    Autoregressive Model AR(p) for Time Series Prediction.
    
    This is a simple linear autoregressive model that predicts:
    X_t = c + φ₁*X_{t-1} + φ₂*X_{t-2} + ... + φₚ*X_{t-p}
    
    In neural network terms, this is just a linear layer with no activation function.
    
    Architecture:
        Input: [batch_size, sequence_length] - past p values
        Output: [batch_size, 1] - predicted next value
    """
    
    def __init__(self, order: int):
        """
        Initialize the AR model.
        
        Args:
            order: The order p of the AR model (how many past values to use)
        """
        super(ARModel, self).__init__()
        
        self.order = order
        
        # Linear layer: learns the coefficients φ₁, φ₂, ..., φₚ and constant c
        # Input dimension: order (p past values)
        # Output dimension: 1 (next value)
        # bias=True means we learn the constant term c
        self.linear = nn.Linear(order, 1, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict next value given past values.
        
        Args:
            x: Input tensor of shape [batch_size, order]
               Contains the past 'order' values for each sample
               
        Returns:
            Predicted next value, shape [batch_size, 1]
        """
        # Simple linear transformation
        # output = w₁*x₁ + w₂*x₂ + ... + wₚ*xₚ + b
        return self.linear(x)
    
    def get_coefficients(self) -> dict:
        """
        Extract the learned AR coefficients.
        
        Returns:
            Dictionary with:
                - 'coefficients': The learned φ values [φ₁, φ₂, ..., φₚ]
                - 'constant': The learned constant term c
        """
        # Get weights and bias from the linear layer
        weights = self.linear.weight.data.cpu().numpy().flatten()
        bias = self.linear.bias.data.cpu().numpy()[0]
        
        return {
            'coefficients': weights,
            'constant': bias
        }
    
    def predict_sequence(self, 
                        initial_sequence: torch.Tensor, 
                        n_steps: int) -> np.ndarray:
        """
        Generate future predictions autoregressively.
        
        This function predicts multiple steps into the future by:
        1. Using known past values to predict next value
        2. Adding prediction to the sequence
        3. Using most recent values (including prediction) for next prediction
        4. Repeating steps 2-3
        
        Args:
            initial_sequence: Tensor of shape [order] containing starting values
            n_steps: Number of steps to predict into the future
            
        Returns:
            numpy array of length n_steps containing predictions
        """
        self.eval()  # Set to evaluation mode
        
        # Store predictions
        predictions = []
        
        # Current sequence (sliding window)
        current_seq = initial_sequence.clone().unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():  # No gradient computation needed
            for _ in range(n_steps):
                # Predict next value
                pred = self.forward(current_seq)
                predictions.append(pred.item())
                
                # Update sequence: remove oldest, add newest prediction
                # Shift left and add new prediction at the end
                current_seq = torch.cat([current_seq[:, 1:], pred], dim=1)
        
        return np.array(predictions)


class NeuralARModel(nn.Module):
    """
    Neural Autoregressive Model - a nonlinear extension of AR(p).
    
    Instead of a simple linear model, this uses a small neural network
    to capture nonlinear relationships in the time series.
    
    Architecture:
        Input -> Hidden Layer (ReLU) -> Hidden Layer (ReLU) -> Output
    """
    
    def __init__(self, 
                 order: int, 
                 hidden_size: int = 64):
        """
        Initialize the neural AR model.
        
        Args:
            order: Number of past values to use as input
            hidden_size: Number of neurons in hidden layers
        """
        super(NeuralARModel, self).__init__()
        
        self.order = order
        
        # Multi-layer neural network
        self.network = nn.Sequential(
            # First hidden layer
            nn.Linear(order, hidden_size),
            nn.ReLU(),
            
            # Second hidden layer
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            
            # Output layer
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through neural network.
        
        Args:
            x: Input tensor of shape [batch_size, order]
            
        Returns:
            Predicted next value, shape [batch_size, 1]
        """
        return self.network(x)
    
    def predict_sequence(self, 
                        initial_sequence: torch.Tensor, 
                        n_steps: int) -> np.ndarray:
        """
        Generate future predictions autoregressively.
        
        Same as ARModel.predict_sequence() but uses neural network.
        
        Args:
            initial_sequence: Starting values
            n_steps: Number of steps to forecast
            
        Returns:
            Array of predictions
        """
        self.eval()
        predictions = []
        current_seq = initial_sequence.clone().unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(n_steps):
                pred = self.forward(current_seq)
                predictions.append(pred.item())
                current_seq = torch.cat([current_seq[:, 1:], pred], dim=1)
        
        return np.array(predictions)


if __name__ == "__main__":
    """
    Demo: Test the AR models with dummy data
    """
    
    # Create dummy data
    batch_size = 32
    order = 5  # AR(5) model
    
    # Random input: 5 past values for each sample in batch
    X = torch.randn(batch_size, order)
    
    print("=" * 60)
    print("Testing Linear AR Model")
    print("=" * 60)
    
    # Initialize linear AR model
    model = ARModel(order=order)
    
    # Forward pass
    predictions = model(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {predictions.shape}")
    
    # Show learned coefficients
    coeffs = model.get_coefficients()
    print(f"\nLearned coefficients: {coeffs['coefficients']}")
    print(f"Learned constant: {coeffs['constant']:.4f}")
    
    # Test sequence prediction
    initial_seq = torch.randn(order)
    future_preds = model.predict_sequence(initial_seq, n_steps=10)
    print(f"\nGenerated {len(future_preds)} future predictions")
    
    print("\n" + "=" * 60)
    print("Testing Neural AR Model")
    print("=" * 60)
    
    # Initialize neural AR model
    neural_model = NeuralARModel(order=order, hidden_size=32)
    
    # Forward pass
    predictions = neural_model(X)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {predictions.shape}")
    
    # Count parameters
    n_params = sum(p.numel() for p in neural_model.parameters())
    print(f"Number of parameters: {n_params}")
    
    print("\n✓ Both models working correctly!")
