#!/usr/bin/env python3
"""
Script 06: Multiclass Classification with Activation Functions
DIFFICULTY: ⭐⭐⭐ Medium-Hard | TIME: 10 min | PREREQ: Scripts 01-05
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

class MulticlassClassifier(nn.Module):
    """
    Multiclass classification network.
    Output: num_classes logits (NO softmax)
    Use with CrossEntropyLoss
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # Return logits (NO softmax!)
        return x

def generate_multiclass_data(n_samples=600, n_classes=3):
    """Generate multiclass dataset"""
    X, y = make_blobs(n_samples=n_samples, centers=n_classes, 
                      n_features=2, cluster_std=1.0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)  # Long tensor for CrossEntropyLoss
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)
    
    return X_train, y_train, X_test, y_test

def train_multiclass(model, X_train, y_train, X_test, y_test, epochs=200, lr=0.01):
    """Train multiclass model"""
    criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax + NLLLoss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        logits = model(X_train)
        loss = criterion(logits, y_train)  # y_train: class indices
        
        loss.backward()
        optimizer.step()
        
        # Accuracy
        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            train_acc = (predictions == y_train).float().mean()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_loss = criterion(test_logits, y_test)
            test_predictions = test_logits.argmax(dim=1)
            test_acc = (test_predictions == y_test).float().mean()
        
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        train_accs.append(train_acc.item())
        test_accs.append(test_acc.item())
        
        if (epoch + 1) % 40 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc.item():.4f} | Test Acc: {test_acc.item():.4f}")
    
    return train_losses, test_losses, train_accs, test_accs

def plot_multiclass_boundary(model, X, y, num_classes=3):
    """Visualize multiclass decision boundaries"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    model.eval()
    with torch.no_grad():
        mesh_input = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        logits = model(mesh_input)
        predictions = logits.argmax(dim=1)
        Z = predictions.numpy().reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=num_classes-1, alpha=0.5, cmap='viridis')
    scatter = plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), 
                          c=y.numpy(), cmap='viridis',
                          edgecolors='k', s=50, alpha=0.9)
    plt.colorbar(scatter, label='Class')
    plt.title('Multiclass Decision Boundaries', fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=11)
    plt.ylabel('Feature 2', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

def main():
    print("\n" + "█" * 70)
    print("   Script 06: Multiclass Classification")
    print("█" * 70)
    
    print("\n[1/4] Generating multiclass dataset...")
    X_train, y_train, X_test, y_test = generate_multiclass_data()
    print(f"   Classes: {torch.unique(y_train).tolist()}")
    
    print("\n[2/4] Creating model...")
    model = MulticlassClassifier(input_size=2, hidden_size=32, num_classes=3)
    print(f"   Architecture: 2 → 32 → 32 → 3")
    
    print("\n[3/4] Training...")
    train_losses, test_losses, train_accs, test_accs = train_multiclass(
        model, X_train, y_train, X_test, y_test, epochs=200
    )
    print(f"\n✅ Final Test Accuracy: {test_accs[-1]:.4f}")
    
    print("\n[4/4] Visualizing results...")
    
    # Training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs_range, train_losses, label='Train', linewidth=2)
    ax1.plot(epochs_range, test_losses, label='Test', linewidth=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, train_accs, label='Train', linewidth=2)
    ax2.plot(epochs_range, test_accs, label='Test', linewidth=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Decision boundaries
    plot_multiclass_boundary(model, X_train, y_train)
    
    print("\n" + "=" * 70)
    print("KEY POINTS FOR MULTICLASS CLASSIFICATION:")
    print("=" * 70)
    print("✅ Output: num_classes neurons (return logits, NO softmax)")
    print("✅ Loss: CrossEntropyLoss (internally applies LogSoftmax)")
    print("✅ Targets: Long tensor with class indices [0, 1, 2, ...]")
    print("✅ Prediction: logits.argmax(dim=1)")
    print("❌ DON'T apply softmax before CrossEntropyLoss!")
    
    print("\n✅ Next: Run '07_regression_with_activations.py'")
    plt.show()

if __name__ == "__main__":
    main()
