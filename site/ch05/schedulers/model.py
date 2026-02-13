"""
Model Module for Learning Rate Scheduler Demo
==============================================

This module defines the neural network architecture used in the scheduler
demonstration. We use a simple Multi-Layer Perceptron (MLP) to keep the
focus on learning rate scheduling rather than model complexity.

The model is intentionally simple to:
- Train quickly for demonstration purposes
- Make learning rate effects clearly visible
- Avoid distractions from architectural complexity
- Allow running on CPU without performance issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ============================================================================
# TINY MLP MODEL
# ============================================================================
class TinyMLP(nn.Module):
    """
    A simple Multi-Layer Perceptron for classification.
    
    Architecture:
        Input Layer → Hidden Layer (ReLU) → Dropout → Output Layer
    
    This is a basic feedforward neural network with:
    - One hidden layer with ReLU activation
    - Optional dropout for regularization
    - Linear output layer (logits for classification)
    
    The simplicity makes it ideal for observing learning rate effects:
    - Fast training allows testing many scheduler configurations
    - Small size means CPU training is feasible
    - Clear loss curves show scheduler impact
    
    Attributes:
        fc1 (nn.Linear): First linear layer (input → hidden)
        fc2 (nn.Linear): Second linear layer (hidden → output)
        dropout (nn.Dropout): Dropout layer for regularization
        input_dim (int): Number of input features
        hidden_dim (int): Number of hidden layer neurons
        num_classes (int): Number of output classes
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 100,
        num_classes: int = 10,
        dropout_prob: float = 0.2
    ):
        """
        Initialize the TinyMLP model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dim (int): Number of neurons in hidden layer
                              Larger = more capacity but slower training
            num_classes (int): Number of output classes for classification
            dropout_prob (float): Dropout probability (0-1)
                                  0 = no dropout, higher = more regularization
        
        Example:
            >>> model = TinyMLP(input_dim=20, hidden_dim=100, num_classes=10)
            >>> print(model)
            >>> x = torch.randn(32, 20)  # Batch of 32 samples
            >>> output = model(x)
            >>> print(output.shape)  # torch.Size([32, 10])
        """
        super(TinyMLP, self).__init__()
        
        # Store dimensions for reference
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # ====================================================================
        # LAYER DEFINITIONS
        # ====================================================================
        
        # First linear layer: input_dim → hidden_dim
        # This layer learns to extract features from input
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # Dropout layer for regularization
        # Randomly sets neurons to zero during training to prevent overfitting
        # Not applied during evaluation (model.eval())
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Second linear layer: hidden_dim → num_classes
        # This layer produces logits (raw scores) for each class
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # ====================================================================
        # WEIGHT INITIALIZATION
        # ====================================================================
        # Initialize weights using Xavier/Glorot initialization
        # This helps with training stability and convergence
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize network weights using Xavier uniform initialization.
        
        Xavier initialization sets weights to values that help maintain
        variance across layers, preventing vanishing/exploding gradients.
        
        This is especially important for:
        - Deep networks
        - Networks with many layers
        - Ensuring stable training from the start
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                
                # Initialize biases to zero
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        This defines how input data flows through the network to produce output.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
                         These are raw scores (not probabilities)
        
        The forward pass:
        1. Linear transformation: x @ W1 + b1
        2. ReLU activation: max(0, x)
        3. Dropout: randomly zero out neurons (training only)
        4. Linear transformation: x @ W2 + b2
        5. Return logits (no softmax here, applied in loss function)
        
        Example:
            >>> model = TinyMLP(input_dim=20, hidden_dim=100, num_classes=10)
            >>> x = torch.randn(32, 20)
            >>> logits = model(x)
            >>> print(logits.shape)  # torch.Size([32, 10])
            >>> 
            >>> # Get probabilities
            >>> probs = F.softmax(logits, dim=1)
            >>> print(probs.sum(dim=1))  # All close to 1.0
            >>> 
            >>> # Get predictions
            >>> predictions = logits.argmax(dim=1)
            >>> print(predictions.shape)  # torch.Size([32])
        """
        
        # ====================================================================
        # LAYER 1: INPUT → HIDDEN
        # ====================================================================
        # Linear transformation followed by ReLU activation
        # ReLU(x) = max(0, x) introduces non-linearity
        # Without non-linearity, the network would be equivalent to a single layer
        x = F.relu(self.fc1(x))
        
        # ====================================================================
        # DROPOUT
        # ====================================================================
        # Apply dropout for regularization (only during training)
        # This helps prevent overfitting by randomly dropping neurons
        x = self.dropout(x)
        
        # ====================================================================
        # LAYER 2: HIDDEN → OUTPUT
        # ====================================================================
        # Final linear layer produces logits (raw scores)
        # No activation here because:
        # - CrossEntropyLoss applies softmax internally
        # - Keeping logits separate from softmax is more numerically stable
        x = self.fc2(x)
        
        return x
    
    def get_num_parameters(self) -> int:
        """
        Calculate total number of trainable parameters.
        
        Returns:
            int: Total number of trainable parameters
        
        Formula:
            Layer 1: (input_dim + 1) * hidden_dim  (weights + biases)
            Layer 2: (hidden_dim + 1) * num_classes
            Total: sum of above
        
        Example:
            >>> model = TinyMLP(input_dim=20, hidden_dim=100, num_classes=10)
            >>> print(f"Parameters: {model.get_num_parameters():,}")
            Parameters: 3,110
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __repr__(self) -> str:
        """
        String representation of the model.
        
        Returns:
            str: Formatted string describing the model architecture
        """
        num_params = self.get_num_parameters()
        return (
            f"TinyMLP(\n"
            f"  (fc1): Linear(in_features={self.input_dim}, "
            f"out_features={self.hidden_dim})\n"
            f"  (relu): ReLU()\n"
            f"  (dropout): Dropout(p={self.dropout.p})\n"
            f"  (fc2): Linear(in_features={self.hidden_dim}, "
            f"out_features={self.num_classes})\n"
            f")\n"
            f"Total parameters: {num_params:,}"
        )


# ============================================================================
# ALTERNATIVE: DEEPER MLP (Optional)
# ============================================================================
class DeepMLP(nn.Module):
    """
    A deeper Multi-Layer Perceptron with multiple hidden layers.
    
    Architecture:
        Input → Hidden1 → Hidden2 → Hidden3 → Output
        Each hidden layer has ReLU and Dropout
    
    This can be used for more complex problems or to observe
    scheduler effects on deeper networks.
    
    Note: For the demo, TinyMLP is recommended for speed.
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dims: list = [100, 50, 25],
        num_classes: int = 10,
        dropout_prob: float = 0.3
    ):
        """
        Initialize the DeepMLP model.
        
        Args:
            input_dim (int): Number of input features
            hidden_dims (list): List of hidden layer sizes
            num_classes (int): Number of output classes
            dropout_prob (float): Dropout probability
        """
        super(DeepMLP, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        
        # Build layers dynamically
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # Create sequential model
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the deeper network."""
        return self.network(x)
    
    def get_num_parameters(self) -> int:
        """Calculate total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# MODULE TEST
# ============================================================================
if __name__ == '__main__':
    """
    Test the model module.
    
    Run this file directly to test the model:
        python -m scheduler.model
    """
    print("Testing model module...\n")
    
    # ========================================================================
    # TEST TINYMLP
    # ========================================================================
    print("="*70)
    print("Testing TinyMLP")
    print("="*70)
    
    # Create model
    model = TinyMLP(input_dim=20, hidden_dim=100, num_classes=10)
    print(f"\n{model}\n")
    
    # Create sample input
    batch_size = 32
    x = torch.randn(batch_size, 20)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    print(f"Output dtype: {logits.dtype}")
    
    # Test softmax (for probabilities)
    probs = F.softmax(logits, dim=1)
    print(f"\nProbabilities shape: {probs.shape}")
    print(f"Probabilities sum: {probs.sum(dim=1)[0].item():.6f} (should be ~1.0)")
    
    # Test predictions
    predictions = logits.argmax(dim=1)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Sample predictions: {predictions[:10].tolist()}")
    
    # ========================================================================
    # TEST DEEPMLP
    # ========================================================================
    print("\n" + "="*70)
    print("Testing DeepMLP")
    print("="*70)
    
    deep_model = DeepMLP(
        input_dim=20,
        hidden_dims=[100, 50, 25],
        num_classes=10
    )
    print(f"\nTotal parameters: {deep_model.get_num_parameters():,}\n")
    
    # Forward pass
    logits = deep_model(x)
    print(f"Output shape: {logits.shape}")
    
    # ========================================================================
    # TEST GRADIENT FLOW
    # ========================================================================
    print("\n" + "="*70)
    print("Testing Gradient Flow")
    print("="*70)
    
    # Create dummy loss and backpropagate
    loss = logits.mean()
    loss.backward()
    
    # Check if gradients exist
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"\nGradients computed: {has_gradients}")
    
    if has_gradients:
        # Print gradient statistics
        for name, param in model.named_parameters():
            if param.grad is not None:
                print(f"{name:20s} grad mean: {param.grad.mean().item():10.6f}, "
                      f"grad std: {param.grad.std().item():10.6f}")
    
    print("\nModel module test completed successfully!")
