"""
Data Generation Utilities for Time Series AR Models

This module contains functions to generate synthetic time series data
for training and testing autoregressive models.
"""

import numpy as np
import torch
from typing import Tuple


def generate_sine_wave(n_samples: int = 1000, 
                       frequency: float = 0.1, 
                       noise_std: float = 0.1,
                       seed: int = 42) -> np.ndarray:
    """
    Generate a sine wave with additive Gaussian noise.
    
    This is a simple time series that demonstrates periodic behavior.
    AR models should be able to learn and predict this pattern.
    
    Args:
        n_samples: Number of time points to generate
        frequency: Frequency of the sine wave (higher = faster oscillation)
        noise_std: Standard deviation of Gaussian noise to add
        seed: Random seed for reproducibility
        
    Returns:
        numpy array of shape (n_samples,) containing the time series
    """
    np.random.seed(seed)
    
    # Generate time points
    t = np.arange(n_samples)
    
    # Generate clean sine wave
    signal = np.sin(2 * np.pi * frequency * t)
    
    # Add Gaussian noise
    noise = np.random.normal(0, noise_std, n_samples)
    
    return signal + noise


def generate_ar_process(n_samples: int = 1000,
                       coefficients: list = [0.6, -0.3],
                       noise_std: float = 0.5,
                       seed: int = 42) -> np.ndarray:
    """
    Generate a true AR(p) process with specified coefficients.
    
    This generates data from a known AR model:
    X_t = φ₁*X_{t-1} + φ₂*X_{t-2} + ... + φₚ*X_{t-p} + ε_t
    
    where ε_t ~ N(0, noise_std²)
    
    Args:
        n_samples: Number of time points to generate
        coefficients: List of AR coefficients [φ₁, φ₂, ..., φₚ]
        noise_std: Standard deviation of the noise term
        seed: Random seed for reproducibility
        
    Returns:
        numpy array of shape (n_samples,) containing the time series
        
    Example:
        # Generate AR(2) process: X_t = 0.6*X_{t-1} - 0.3*X_{t-2} + ε_t
        data = generate_ar_process(1000, coefficients=[0.6, -0.3])
    """
    np.random.seed(seed)
    
    p = len(coefficients)  # Order of AR process
    series = np.zeros(n_samples)
    
    # Initialize first p values with small random numbers
    series[:p] = np.random.normal(0, 0.1, p)
    
    # Generate the rest of the series using AR equation
    for t in range(p, n_samples):
        # Compute autoregressive term: sum of (coefficient * past value)
        ar_term = sum(coefficients[i] * series[t - i - 1] for i in range(p))
        
        # Add noise
        noise = np.random.normal(0, noise_std)
        
        # Combine
        series[t] = ar_term + noise
    
    return series


def create_sequences(data: np.ndarray, 
                     sequence_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create input-output pairs for training an AR model.
    
    Converts a time series into sequences for supervised learning.
    For AR models, we want to predict X_t given X_{t-p}, ..., X_{t-1}
    
    Args:
        data: 1D numpy array of time series values
        sequence_length: Number of past values to use (this is 'p' in AR(p))
        
    Returns:
        Tuple of (X, y) where:
            X: Tensor of shape (n_sequences, sequence_length) - input sequences
            y: Tensor of shape (n_sequences, 1) - target values
            
    Example:
        If data = [1, 2, 3, 4, 5] and sequence_length = 2:
        X = [[1, 2], [2, 3], [3, 4]]
        y = [[3], [4], [5]]
    """
    X, y = [], []
    
    # Slide a window across the time series
    for i in range(len(data) - sequence_length):
        # Input: past 'sequence_length' values
        X.append(data[i:i + sequence_length])
        
        # Output: the next value
        y.append(data[i + sequence_length])
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(np.array(X))
    y = torch.FloatTensor(np.array(y)).unsqueeze(1)  # Add dimension for consistency
    
    return X, y


def train_test_split_temporal(X: torch.Tensor, 
                              y: torch.Tensor, 
                              train_ratio: float = 0.8) -> Tuple[torch.Tensor, ...]:
    """
    Split time series data into train and test sets.
    
    IMPORTANT: For time series, we must NOT shuffle the data!
    We split chronologically - train on earlier data, test on later data.
    This simulates real forecasting scenarios.
    
    Args:
        X: Input sequences tensor
        y: Target values tensor
        train_ratio: Fraction of data to use for training (rest is test)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    n_samples = len(X)
    split_idx = int(n_samples * train_ratio)
    
    # Split chronologically (no shuffling!)
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    """
    Demo: Generate and visualize sample data
    """
    import matplotlib.pyplot as plt
    
    # Generate sine wave
    sine_data = generate_sine_wave(n_samples=200, frequency=0.05, noise_std=0.2)
    
    # Generate true AR process
    ar_data = generate_ar_process(n_samples=200, 
                                   coefficients=[0.7, -0.2], 
                                   noise_std=0.3)
    
    # Create sequences
    X, y = create_sequences(ar_data, sequence_length=5)
    print(f"Created {len(X)} training sequences")
    print(f"Input shape: {X.shape}, Output shape: {y.shape}")
    
    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    axes[0].plot(sine_data)
    axes[0].set_title("Sine Wave with Noise")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")
    axes[0].grid(True)
    
    axes[1].plot(ar_data)
    axes[1].set_title("AR(2) Process: X_t = 0.7*X_{t-1} - 0.2*X_{t-2} + noise")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Value")
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig("sample_time_series.png", dpi=150)
    print("Saved visualization to sample_time_series.png")
