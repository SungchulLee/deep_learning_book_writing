#!/usr/bin/env python3
"""
Script 07: Regression with Activation Functions
DIFFICULTY: ⭐⭐⭐ Medium | TIME: 8 min | PREREQ: Scripts 01-06
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

class RegressionNetwork(nn.Module):
    """Regression network: predict continuous values"""
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # NO activation on output!
        return x

def generate_regression_data(n_samples=500):
    """Generate nonlinear regression data: y = sin(x) + noise"""
    X = np.linspace(-3*np.pi, 3*np.pi, n_samples).reshape(-1, 1)
    y = np.sin(X) + 0.2 * np.random.randn(n_samples, 1)
    
    X_train = torch.FloatTensor(X)
    y_train = torch.FloatTensor(y)
    
    return X_train, y_train

def train_regression(model, X, y, epochs=300, lr=0.01):
    """Train regression model"""
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X)
        loss = criterion(predictions, y)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if (epoch + 1) % 60 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.6f}")
    
    return losses

def main():
    print("\n" + "█" * 70)
    print("   Script 07: Regression with Activations")
    print("█" * 70)
    
    print("\n[1/3] Generating regression data (sin wave + noise)...")
    X_train, y_train = generate_regression_data()
    
    print("\n[2/3] Training regression network...")
    model = RegressionNetwork()
    losses = train_regression(model, X_train, y_train, epochs=300)
    print(f"✅ Final Loss: {losses[-1]:.6f}")
    
    print("\n[3/3] Visualizing results...")
    
    # Predictions
    model.eval()
    with torch.no_grad():
        X_test = torch.linspace(-3*np.pi, 3*np.pi, 1000).unsqueeze(1)
        y_pred = model(X_test)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    ax1.plot(losses, linewidth=2, color='blue')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE Loss')
    ax1.set_title('Training Loss'); ax1.grid(True, alpha=0.3)
    
    # Predictions vs actual
    ax2.scatter(X_train.numpy(), y_train.numpy(), 
                alpha=0.5, s=10, label='Training Data', color='blue')
    ax2.plot(X_test.numpy(), y_pred.numpy(), 
             linewidth=2.5, color='red', label='Model Prediction')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y')
    ax2.set_title('Regression: sin(x) + noise')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print("\n" + "=" * 70)
    print("KEY POINTS FOR REGRESSION:")
    print("=" * 70)
    print("✅ Output layer: NO activation (or use bounds if needed)")
    print("✅ Hidden layers: Use ReLU, Leaky ReLU, etc.")
    print("✅ Loss: MSELoss (Mean Squared Error)")
    print("✅ Task: Predict continuous values")
    print("\n💡 When to add output activation:")
    print("   • Bounded output (0,1): Use sigmoid")
    print("   • Bounded output (-1,1): Use tanh")
    print("   • Positive output: Use ReLU or softplus")
    print("   • Unbounded: No activation")
    
    print("\n✅ Next: Run '08_custom_activation.py'")
    plt.show()

if __name__ == "__main__":
    main()
