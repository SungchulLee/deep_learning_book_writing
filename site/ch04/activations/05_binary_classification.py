#!/usr/bin/env python3
"""
==============================================================================
Script 05: Binary Classification with Activation Functions
==============================================================================

DIFFICULTY: ⭐⭐⭐ Medium
ESTIMATED TIME: 5 minutes
PREREQUISITES: Scripts 01-04

LEARNING OBJECTIVES:
--------------------
1. Build a complete binary classification pipeline
2. Understand proper use of sigmoid and BCEWithLogitsLoss
3. Implement training loop with evaluation
4. Visualize decision boundaries
5. Compare different hidden layer activations

KEY CONCEPTS:
-------------
- Binary classification: 2 classes (0 or 1)
- Output layer: Return LOGITS (not probabilities)
- Loss function: BCEWithLogitsLoss (combines sigmoid + BCE)
- Hidden layers: Use ReLU, Leaky ReLU, or modern activations
- Evaluation: Accuracy, decision boundaries

RUN THIS SCRIPT:
----------------
    python 05_binary_classification.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==============================================================================
# SECTION 1: Generate Binary Classification Dataset
# ==============================================================================

def section1_generate_data():
    """
    Create a synthetic 2D dataset for binary classification.
    Using "moons" dataset - two interleaving crescents.
    """
    print("=" * 70)
    print("SECTION 1: Generating Binary Classification Dataset")
    print("=" * 70)
    
    # Generate moons dataset
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)  # Shape: (N, 1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    
    print(f"\nDataset created:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples:  {len(X_test)}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: 2 (binary)")
    
    # Visualize dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train[:, 0].numpy(), X_train[:, 1].numpy(), 
                c=y_train.squeeze().numpy(), cmap='coolwarm',
                alpha=0.7, edgecolors='k', s=50)
    plt.title('Binary Classification Dataset (Moons)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=11)
    plt.ylabel('Feature 2', fontsize=11)
    plt.colorbar(label='Class')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    print("\n✅ Dataset visualization created")
    
    return X_train, y_train, X_test, y_test, plt.gcf()


# ==============================================================================
# SECTION 2: Define Binary Classification Model
# ==============================================================================

class BinaryClassifier(nn.Module):
    """
    Binary classification neural network.
    
    Architecture:
        Input (2) → Hidden (16) → Hidden (16) → Output (1)
    
    Key points:
    - Hidden layers: Use ReLU or other activation
    - Output layer: NO activation (return logits)
    - Use with BCEWithLogitsLoss
    """
    def __init__(self, input_size=2, hidden_size=16, activation_type='relu'):
        super().__init__()
        # Layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Binary: single output
        
        # Choose activation function
        if activation_type == 'relu':
            self.activation = nn.ReLU()
        elif activation_type == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation_type == 'gelu':
            self.activation = nn.GELU()
        elif activation_type == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()  # Default
        
        self.activation_name = activation_type
    
    def forward(self, x):
        """
        Forward pass.
        Returns: LOGITS (not probabilities)
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # NO activation here! Return logits
        return x


def section2_define_model():
    """
    Demonstrate the model architecture.
    """
    print("\n" + "=" * 70)
    print("SECTION 2: Binary Classification Model Architecture")
    print("=" * 70)
    
    model = BinaryClassifier(input_size=2, hidden_size=16, activation_type='relu')
    
    print("\nModel Architecture:")
    print(model)
    
    print("\n🔑 KEY DESIGN DECISIONS:")
    print("=" * 70)
    print("1. Hidden Layer Activation: ReLU (or Leaky ReLU, GELU, etc.)")
    print("   → Introduces non-linearity for complex decision boundaries")
    print()
    print("2. Output Layer Activation: NONE (return logits)")
    print("   → BCEWithLogitsLoss will apply sigmoid internally")
    print("   → More numerically stable than sigmoid + BCELoss")
    print()
    print("3. Output Size: 1 neuron")
    print("   → Binary classification: single value")
    print("   → After sigmoid: prob(class=1)")
    print("   → 1 - prob(class=1) = prob(class=0)")
    
    # Demonstrate forward pass
    sample_input = torch.randn(1, 2)
    logits = model(sample_input)
    probs = torch.sigmoid(logits)  # Convert to probabilities
    
    print("\n📊 Example Forward Pass:")
    print(f"   Input shape:  {sample_input.shape}")
    print(f"   Logits:       {logits.item():.4f}")
    print(f"   Probability:  {probs.item():.4f}")
    print(f"   Prediction:   {(probs > 0.5).item()} (threshold=0.5)")
    
    return model


# ==============================================================================
# SECTION 3: Training Function
# ==============================================================================

def train_model(model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01):
    """
    Train the binary classification model.
    
    Args:
        model: PyTorch model
        X_train, y_train: Training data
        X_test, y_test: Test data
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        train_losses, test_losses, train_accs, test_accs
    """
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + BCE
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        # Forward pass (returns logits)
        logits = model(X_train)
        loss = criterion(logits, y_train)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate training accuracy
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            train_acc = (predictions == y_train).float().mean()
        
        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test)
            test_loss = criterion(test_logits, y_test)
            test_probs = torch.sigmoid(test_logits)
            test_predictions = (test_probs > 0.5).float()
            test_acc = (test_predictions == y_test).float().mean()
        
        # Store metrics
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        train_accs.append(train_acc.item())
        test_accs.append(test_acc.item())
        
        # Print progress
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc.item():.4f} | "
                  f"Test Acc: {test_acc.item():.4f}")
    
    return train_losses, test_losses, train_accs, test_accs


def section3_train():
    """
    Train the model and visualize training progress.
    """
    print("\n" + "=" * 70)
    print("SECTION 3: Training the Model")
    print("=" * 70)
    
    # Get data
    X_train, y_train, X_test, y_test, _ = section1_generate_data()
    plt.close()  # Close the data visualization for now
    
    # Create model
    model = BinaryClassifier(activation_type='relu')
    
    print("\n🏃 Training started...")
    print("-" * 70)
    
    # Train
    train_losses, test_losses, train_accs, test_accs = train_model(
        model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01
    )
    
    print("-" * 70)
    print(f"✅ Training completed!")
    print(f"   Final Train Accuracy: {train_accs[-1]:.4f}")
    print(f"   Final Test Accuracy:  {test_accs[-1]:.4f}")
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, label='Train Loss', linewidth=2)
    ax1.plot(epochs_range, test_losses, label='Test Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Training and Test Loss', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs_range, train_accs, label='Train Accuracy', linewidth=2)
    ax2.plot(epochs_range, test_accs, label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Accuracy', fontsize=11)
    ax2.set_title('Training and Test Accuracy', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    return model, X_train, y_train, X_test, y_test, fig


# ==============================================================================
# SECTION 4: Visualize Decision Boundary
# ==============================================================================

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plot the decision boundary of the trained model.
    """
    # Create mesh
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predict on mesh
    model.eval()
    with torch.no_grad():
        mesh_input = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        logits = model(mesh_input)
        probs = torch.sigmoid(logits)
        Z = probs.numpy().reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.7)
    plt.colorbar(label='Probability of Class 1')
    
    # Plot data points
    plt.scatter(X[:, 0].numpy(), X[:, 1].numpy(), 
                c=y.squeeze().numpy(), cmap='RdYlBu',
                edgecolors='k', s=50, alpha=0.9)
    
    # Decision boundary (prob = 0.5)
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Feature 1', fontsize=11)
    plt.ylabel('Feature 2', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()


def section4_visualize_boundary():
    """
    Train model and visualize decision boundary.
    """
    print("\n" + "=" * 70)
    print("SECTION 4: Visualizing Decision Boundary")
    print("=" * 70)
    
    # Get trained model and data
    model, X_train, y_train, X_test, y_test, _ = section3_train()
    plt.close()
    
    # Plot decision boundary
    plot_decision_boundary(model, X_train, y_train, 
                          f"Decision Boundary ({model.activation_name.upper()})")
    
    print("\n✅ Decision boundary visualization created")
    print("   → Blue regions: Model predicts class 0")
    print("   → Red regions: Model predicts class 1")
    print("   → Black line: Decision boundary (prob = 0.5)")
    
    return plt.gcf()


# ==============================================================================
# SECTION 5: Compare Different Activations
# ==============================================================================

def section5_compare_activations():
    """
    Compare binary classification with different hidden layer activations.
    """
    print("\n" + "=" * 70)
    print("SECTION 5: Comparing Different Activations")
    print("=" * 70)
    
    # Get data
    X_train, y_train, X_test, y_test, _ = section1_generate_data()
    plt.close()
    
    # Activations to compare
    activations = ['relu', 'leaky_relu', 'gelu', 'silu']
    
    # Train models
    results = {}
    for activation in activations:
        print(f"\n🔹 Training with {activation.upper()}...")
        model = BinaryClassifier(activation_type=activation)
        _, _, train_accs, test_accs = train_model(
            model, X_train, y_train, X_test, y_test, epochs=100, lr=0.01
        )
        results[activation] = {
            'model': model,
            'train_acc': train_accs[-1],
            'test_acc': test_accs[-1]
        }
        print(f"   Final Test Accuracy: {test_accs[-1]:.4f}")
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS:")
    print("=" * 70)
    for activation, res in results.items():
        print(f"{activation.upper():15s} - Test Accuracy: {res['test_acc']:.4f}")
    
    # Find best activation
    best_activation = max(results.items(), key=lambda x: x[1]['test_acc'])
    print(f"\n🏆 Best performing: {best_activation[0].upper()} "
          f"(Acc: {best_activation[1]['test_acc']:.4f})")
    
    print("\n💡 Note: Results may vary between runs due to randomness.")
    print("   For this simple task, all activations should work well!")


# ==============================================================================
# SECTION 6: Best Practices Summary
# ==============================================================================

def section6_best_practices():
    """
    Summary of best practices for binary classification.
    """
    print("\n" + "=" * 70)
    print("SECTION 6: Best Practices for Binary Classification 🎓")
    print("=" * 70)
    
    practices = [
        "\n1. OUTPUT LAYER:",
        "   ✅ Return LOGITS (no sigmoid activation)",
        "   ✅ Use nn.BCEWithLogitsLoss()",
        "   ❌ DON''T use sigmoid + nn.BCELoss()",
        "   → BCEWithLogitsLoss is more numerically stable",
        
        "\n2. HIDDEN LAYERS:",
        "   ✅ Use ReLU as default",
        "   ✅ Try Leaky ReLU if you see dead neurons",
        "   ✅ Try GELU/SiLU for potentially better performance",
        "   → Choice of hidden activation affects learning",
        
        "\n3. PREDICTIONS:",
        "   • Training: Use logits directly with loss",
        "   • Inference: Convert logits to probabilities",
        "     probs = torch.sigmoid(logits)",
        "   • Classification: Apply threshold (usually 0.5)",
        "     predictions = (probs > 0.5).float()",
        
        "\n4. ARCHITECTURE:",
        "   • Binary classification: Single output neuron",
        "   • Target format: Shape (N, 1), values in {0, 1}",
        "   • Typical: 2-3 hidden layers for simple tasks",
        
        "\n5. EVALUATION:",
        "   • Accuracy: (predictions == targets).mean()",
        "   • Also consider: Precision, Recall, F1-score",
        "   • Visualize: Decision boundaries, confusion matrix"
    ]
    
    for practice in practices:
        print(practice)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Run all sections.
    """
    print("\n" + "█" * 70)
    print("   PYTORCH ACTIVATION FUNCTIONS TUTORIAL")
    print("   Script 05: Binary Classification")
    print("█" * 70)
    
    fig1 = section1_generate_data()[4]
    section2_define_model()
    fig2 = section3_train()[5]
    fig3 = section4_visualize_boundary()
    section5_compare_activations()
    section6_best_practices()
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("Run ''06_multiclass_classification.py'' for multi-class problems!")
    print("=" * 70)
    
    print("\n✅ Script completed successfully!")
    print("\n📊 Showing plots... (Close windows to continue)")
    
    plt.show()


if __name__ == "__main__":
    main()
