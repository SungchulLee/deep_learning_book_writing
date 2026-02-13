"""
===============================================================================
LEVEL 2: Building Your First Softmax Classifier
===============================================================================
Difficulty: Beginner-Intermediate
Prerequisites: Level 1, basic PyTorch
Learning Goals:
  - Build a simple neural network for multi-class classification
  - Understand the training loop structure
  - Visualize decision boundaries
  - Evaluate model performance

Time to complete: 30-45 minutes
===============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.model_selection import train_test_split

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("LEVEL 2: BUILDING YOUR FIRST SOFTMAX CLASSIFIER")
print("=" * 80)


# =============================================================================
# PART 1: Generate Synthetic Dataset
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: Creating Synthetic Data")
print("=" * 80)

def create_dataset(dataset_type='blobs', n_samples=1000):
    """
    Create synthetic 2D datasets for classification.
    
    Args:
        dataset_type (str): 'blobs', 'moons', or 'circles'
        n_samples (int): Number of samples to generate
    
    Returns:
        X (np.array): Features of shape (n_samples, 2)
        y (np.array): Labels of shape (n_samples,)
    """
    if dataset_type == 'blobs':
        # Well-separated clusters (easiest)
        X, y = make_blobs(n_samples=n_samples, centers=3, n_features=2,
                         center_box=(-5, 5), random_state=42)
    elif dataset_type == 'moons':
        # Two interleaving half circles (medium difficulty)
        X, y = make_moons(n_samples=n_samples, noise=0.2, random_state=42)
    elif dataset_type == 'circles':
        # Concentric circles (harder - needs non-linearity)
        X, y = make_circles(n_samples=n_samples, noise=0.1, factor=0.5,
                           random_state=42)
    else:
        raise ValueError("dataset_type must be 'blobs', 'moons', or 'circles'")
    
    return X, y


# Create the dataset
X, y = create_dataset('blobs', n_samples=1000)

print(f"Dataset shape: X = {X.shape}, y = {y.shape}")
print(f"Number of classes: {len(np.unique(y))}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Feature range: X_min = {X.min():.2f}, X_max = {X.max():.2f}")


# Visualize the dataset
def plot_data(X, y, title="Dataset Visualization"):
    """Plot 2D data points colored by class."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                         alpha=0.6, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    return plt


# Uncomment to show plot
# plot_data(X, y, "Original Dataset")
# plt.show()


# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")


# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

print(f"\nTensor shapes:")
print(f"  X_train: {X_train_tensor.shape}")
print(f"  y_train: {y_train_tensor.shape}")


# =============================================================================
# PART 2: Define the Neural Network Model
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: Building the Neural Network")
print("=" * 80)

class SoftmaxClassifier(nn.Module):
    """
    Simple feedforward neural network for multi-class classification.
    
    Architecture:
        Input Layer (2 features)
          ↓
        Hidden Layer (64 neurons) + ReLU
          ↓
        Hidden Layer (32 neurons) + ReLU
          ↓
        Output Layer (3 classes) → logits
    
    Note: We don't apply softmax in forward() because CrossEntropyLoss
          handles it internally for numerical stability.
    """
    
    def __init__(self, input_size=2, hidden_size1=64, hidden_size2=32, num_classes=3):
        """
        Initialize the network layers.
        
        Args:
            input_size (int): Number of input features
            hidden_size1 (int): Number of neurons in first hidden layer
            hidden_size2 (int): Number of neurons in second hidden layer
            num_classes (int): Number of output classes
        """
        super(SoftmaxClassifier, self).__init__()
        
        # Define layers
        self.fc1 = nn.Linear(input_size, hidden_size1)      # First hidden layer
        self.relu1 = nn.ReLU()                              # Activation function
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)    # Second hidden layer
        self.relu2 = nn.ReLU()                              # Activation function
        self.fc3 = nn.Linear(hidden_size2, num_classes)     # Output layer (logits)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        out = self.fc1(x)      # Apply first linear transformation
        out = self.relu1(out)  # Apply ReLU activation
        out = self.fc2(out)    # Apply second linear transformation
        out = self.relu2(out)  # Apply ReLU activation
        out = self.fc3(out)    # Get final logits (no activation!)
        return out


# Create the model
model = SoftmaxClassifier(input_size=2, hidden_size1=64, hidden_size2=32, num_classes=3)

print("Model Architecture:")
print(model)
print("\n" + "-" * 80)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# =============================================================================
# PART 3: Set Up Training Components
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: Setting Up Training")
print("=" * 80)

# Loss function
criterion = nn.CrossEntropyLoss()
print("Loss function: CrossEntropyLoss")
print("  - Combines softmax + log + negative log likelihood")
print("  - Takes logits (raw scores) as input")
print("  - Takes class indices as targets")

# Optimizer
learning_rate = 0.01
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f"\nOptimizer: Adam")
print(f"  - Learning rate: {learning_rate}")
print("  - Adaptive learning rate for each parameter")
print("  - Good default choice for most problems")

# Number of training epochs
num_epochs = 100
print(f"\nTraining for {num_epochs} epochs")


# =============================================================================
# PART 4: Training Loop
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: Training the Model")
print("=" * 80)

def train_model(model, X_train, y_train, X_test, y_test, 
                criterion, optimizer, num_epochs=100, verbose=True):
    """
    Train the model and track metrics.
    
    Args:
        model: PyTorch model to train
        X_train, y_train: Training data
        X_test, y_test: Test data
        criterion: Loss function
        optimizer: Optimization algorithm
        num_epochs: Number of training iterations
        verbose: Whether to print progress
    
    Returns:
        dict: Training history (losses and accuracies)
    """
    # Store training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_acc': [],
        'test_acc': []
    }
    
    for epoch in range(num_epochs):
        # =====================================================================
        # Training Phase
        # =====================================================================
        model.train()  # Set model to training mode
        
        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear previous gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        train_correct = (predicted == y_train).sum().item()
        train_acc = train_correct / len(y_train)
        
        # =====================================================================
        # Evaluation Phase
        # =====================================================================
        model.eval()  # Set model to evaluation mode
        
        with torch.no_grad():  # Disable gradient computation
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test)
            
            # Calculate test accuracy
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_correct = (test_predicted == y_test).sum().item()
            test_acc = test_correct / len(y_test)
        
        # Store metrics
        history['train_loss'].append(loss.item())
        history['test_loss'].append(test_loss.item())
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"Train Loss: {loss.item():.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Loss: {test_loss.item():.4f} | "
                  f"Test Acc: {test_acc:.4f}")
    
    return history


# Train the model
print("Starting training...\n")
history = train_model(
    model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor,
    criterion, optimizer, num_epochs=num_epochs, verbose=True
)

print("\n✅ Training complete!")


# =============================================================================
# PART 5: Visualize Training Progress
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: Analyzing Training Results")
print("=" * 80)

def plot_training_history(history):
    """Plot training and test losses and accuracies."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot losses
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['test_loss'], label='Test Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss over Time')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot accuracies
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['test_acc'], label='Test Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# Uncomment to show plots
# plot_training_history(history)
# plt.show()

# Print final metrics
final_train_acc = history['train_acc'][-1]
final_test_acc = history['test_acc'][-1]
final_train_loss = history['train_loss'][-1]
final_test_loss = history['test_loss'][-1]

print(f"\nFinal Results:")
print(f"  Train Accuracy: {final_train_acc:.2%}")
print(f"  Test Accuracy:  {final_test_acc:.2%}")
print(f"  Train Loss:     {final_train_loss:.4f}")
print(f"  Test Loss:      {final_test_loss:.4f}")


# =============================================================================
# PART 6: Visualize Decision Boundaries
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: Visualizing Decision Boundaries")
print("=" * 80)

def plot_decision_boundaries(model, X, y, title="Decision Boundaries"):
    """
    Visualize the decision boundaries learned by the model.
    
    Args:
        model: Trained PyTorch model
        X (np.array): Input features
        y (np.array): True labels
        title (str): Plot title
    """
    # Create a mesh grid
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict for each point in the mesh
    model.eval()
    with torch.no_grad():
        mesh_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
        Z = model(mesh_tensor)
        Z = torch.argmax(Z, dim=1).numpy()
    
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                         edgecolors='black', linewidth=1, s=50)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    return plt


# Uncomment to show plot
# plot_decision_boundaries(model, X_test, y_test, 
#                         "Decision Boundaries on Test Set")
# plt.show()


# =============================================================================
# PART 7: Making Predictions
# =============================================================================
print("\n" + "=" * 80)
print("PART 7: Making Predictions on New Data")
print("=" * 80)

def predict_with_probabilities(model, X_new):
    """
    Make predictions and show probabilities for each class.
    
    Args:
        model: Trained model
        X_new (torch.Tensor): New input samples
    
    Returns:
        predictions (torch.Tensor): Predicted class indices
        probabilities (torch.Tensor): Probabilities for each class
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_new)
        probabilities = torch.softmax(logits, dim=1)  # Convert logits to probs
        predictions = torch.argmax(probabilities, dim=1)
    return predictions, probabilities


# Make predictions on a few test samples
n_samples_to_show = 5
X_sample = X_test_tensor[:n_samples_to_show]
y_sample = y_test_tensor[:n_samples_to_show]

predictions, probabilities = predict_with_probabilities(model, X_sample)

print(f"\nPredictions on {n_samples_to_show} test samples:")
print("-" * 80)
for i in range(n_samples_to_show):
    print(f"Sample {i+1}:")
    print(f"  Features: [{X_sample[i, 0].item():.2f}, {X_sample[i, 1].item():.2f}]")
    print(f"  True class: {y_sample[i].item()}")
    print(f"  Predicted class: {predictions[i].item()}")
    print(f"  Probabilities: {probabilities[i].numpy()}")
    correct = "✓" if predictions[i].item() == y_sample[i].item() else "✗"
    print(f"  Correct? {correct}")
    print()


# =============================================================================
# PART 8: Model Evaluation
# =============================================================================
print("\n" + "=" * 80)
print("PART 8: Detailed Model Evaluation")
print("=" * 80)

from sklearn.metrics import classification_report, confusion_matrix

# Get all predictions on test set
model.eval()
with torch.no_grad():
    all_outputs = model(X_test_tensor)
    all_predictions = torch.argmax(all_outputs, dim=1).numpy()

# Classification report
print("\nClassification Report:")
print("-" * 80)
print(classification_report(y_test, all_predictions))

# Confusion matrix
cm = confusion_matrix(y_test, all_predictions)
print("\nConfusion Matrix:")
print("-" * 80)
print(cm)
print("\nRow = True class, Column = Predicted class")


# =============================================================================
# PART 9: Saving and Loading the Model
# =============================================================================
print("\n" + "=" * 80)
print("PART 9: Saving and Loading the Model")
print("=" * 80)

# Save the model
model_path = '/home/claude/softmax_regression_tutorial/level_02_model.pth'
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to: {model_path}")

# Load the model
loaded_model = SoftmaxClassifier(input_size=2, hidden_size1=64, 
                                 hidden_size2=32, num_classes=3)
loaded_model.load_state_dict(torch.load(model_path))
loaded_model.eval()
print(f"✅ Model loaded successfully")

# Verify loaded model works
with torch.no_grad():
    test_output = loaded_model(X_test_tensor[:5])
    print(f"\nTest prediction from loaded model: {torch.argmax(test_output, dim=1).numpy()}")


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY - What You Learned")
print("=" * 80)

print("""
✅ Created synthetic datasets for classification
✅ Built a multi-layer neural network with PyTorch
✅ Implemented a complete training loop
✅ Tracked training and test metrics
✅ Visualized decision boundaries
✅ Made predictions with probability estimates
✅ Evaluated model performance with metrics
✅ Saved and loaded trained models

Key Training Loop Components:
------------------------------
1. Forward pass: model(X) → logits
2. Compute loss: criterion(logits, y)
3. Clear gradients: optimizer.zero_grad()
4. Backward pass: loss.backward()
5. Update weights: optimizer.step()

Next Steps:
-----------
→ Level 3: Train on real datasets (MNIST, Fashion-MNIST)
→ Level 4: Implement from scratch (custom training)
→ Level 5: Advanced techniques (regularization, data augmentation)

🎉 Great job! You've built and trained your first classifier!
""")
