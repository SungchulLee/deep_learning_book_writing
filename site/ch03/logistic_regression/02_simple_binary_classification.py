"""
================================================================================
02_simple_binary_classification.py - Your First Logistic Regression Model
================================================================================

LEARNING OBJECTIVES:
- Understand binary classification problems
- Implement logistic regression from scratch using PyTorch
- Learn the sigmoid function and its properties
- Train a model using gradient descent
- Evaluate model performance

PREREQUISITES:
- Completed 01_introduction.py
- Understanding of linear models (y = mx + b)
- Basic probability concepts

TIME TO COMPLETE: ~45 minutes

DIFFICULTY: ⭐⭐☆☆☆ (Easy)
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

print("="*80)
print("PART 1: UNDERSTANDING THE PROBLEM")
print("="*80)

# ============================================================================
# 1.1: Binary Classification
# ============================================================================
print("\n1.1: What is Binary Classification?")
print("-" * 40)

print("""
Binary Classification: Predict one of two classes (0 or 1)

Examples:
  - Email: Spam (1) or Not Spam (0)
  - Medical: Disease (1) or Healthy (0)  
  - Customer: Will Buy (1) or Won't Buy (0)
  - Image: Cat (1) or Dog (0)

In this tutorial:
  - We'll create synthetic data with 2 features (x1, x2)
  - Each sample belongs to class 0 or class 1
  - Goal: Learn to predict the class from features
""")

# ============================================================================
# 1.2: Generate Synthetic Data
# ============================================================================
print("\n1.2: Generating Dataset")
print("-" * 40)

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate a simple 2D binary classification dataset
# n_samples: number of data points
# n_features: number of input features (we'll use 2 for easy visualization)
# n_classes: 2 (binary classification)
# n_clusters_per_class: how "separated" the classes are
X, y = make_classification(
    n_samples=200,           # Total number of examples
    n_features=2,            # 2D data (x1, x2) for easy plotting
    n_redundant=0,           # No redundant features
    n_informative=2,         # Both features are informative
    n_clusters_per_class=1,  # Single cluster per class
    random_state=42,
    flip_y=0.1              # Add 10% noise (some mislabeled examples)
)

print(f"Dataset shape: X={X.shape}, y={y.shape}")
print(f"X (features): {X.shape[0]} samples × {X.shape[1]} features")
print(f"y (labels): {y.shape[0]} labels")
print(f"Class distribution: Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")

# Split into train and test sets
# Train: 80%, Test: 20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nAfter split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Convert to PyTorch tensors
# Important: Convert to float32 (PyTorch default)
X_train = torch.FloatTensor(X_train)  # Shape: (160, 2)
X_test = torch.FloatTensor(X_test)    # Shape: (40, 2)
y_train = torch.FloatTensor(y_train).reshape(-1, 1)  # Shape: (160, 1)
y_test = torch.FloatTensor(y_test).reshape(-1, 1)    # Shape: (40, 1)

print(f"\nTensor shapes:")
print(f"X_train: {X_train.shape} (160 samples, 2 features)")
print(f"y_train: {y_train.shape} (160 labels, reshaped to column vector)")
print(f"X_test: {X_test.shape}")
print(f"y_test: {y_test.shape}")


print("\n" + "="*80)
print("PART 2: THE LOGISTIC REGRESSION MODEL")
print("="*80)

# ============================================================================
# 2.1: Understanding the Model
# ============================================================================
print("\n2.1: Model Architecture")
print("-" * 40)

print("""
Logistic Regression Model:

Step 1: Linear Combination
    z = w1*x1 + w2*x2 + b
    where w1, w2 are weights, b is bias

Step 2: Sigmoid Activation
    probability = sigmoid(z) = 1 / (1 + e^(-z))
    
Properties of Sigmoid:
    - Maps any value to range (0, 1)
    - sigmoid(0) = 0.5 (decision boundary)
    - sigmoid(large positive) ≈ 1
    - sigmoid(large negative) ≈ 0
    
Step 3: Classification
    If probability >= 0.5: predict class 1
    If probability < 0.5: predict class 0
""")

# ============================================================================
# 2.2: Implementing the Model
# ============================================================================
print("\n2.2: Implementing in PyTorch")
print("-" * 40)

class LogisticRegression(nn.Module):
    """
    Simple Logistic Regression Model
    
    Architecture:
        Input (n_features) → Linear Layer → Sigmoid → Output (probability)
    
    Parameters:
        n_features (int): Number of input features
    """
    
    def __init__(self, n_features):
        super(LogisticRegression, self).__init__()
        
        # Linear layer: y = xW^T + b
        # in_features: number of input features (2 in our case)
        # out_features: number of outputs (1 for binary classification)
        self.linear = nn.Linear(n_features, 1)
        
        # The linear layer creates:
        # - self.linear.weight: shape (1, n_features) - the weights
        # - self.linear.bias: shape (1,) - the bias term
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, n_features)
            
        Returns:
            probability: Output tensor of shape (batch_size, 1)
                        Values in range (0, 1)
        """
        # Step 1: Linear transformation
        # x shape: (batch_size, n_features)
        # output shape: (batch_size, 1)
        z = self.linear(x)  # z = w*x + b
        
        # Step 2: Apply sigmoid activation
        # Sigmoid maps z to probability in range (0, 1)
        probability = torch.sigmoid(z)
        
        return probability

# Create model instance
n_features = 2  # We have 2 features (x1, x2)
model = LogisticRegression(n_features)

print("Model created!")
print(f"Model structure:\n{model}")
print(f"\nInitial weights: {model.linear.weight.data}")
print(f"Initial bias: {model.linear.bias.data}")


print("\n" + "="*80)
print("PART 3: LOSS FUNCTION AND OPTIMIZER")
print("="*80)

# ============================================================================
# 3.1: Binary Cross-Entropy Loss
# ============================================================================
print("\n3.1: Understanding the Loss Function")
print("-" * 40)

print("""
Binary Cross-Entropy (BCE) Loss:

For a single example:
    If actual label y = 1:
        loss = -log(predicted_probability)
        → Model is punished if it predicts low probability for class 1
    
    If actual label y = 0:
        loss = -log(1 - predicted_probability)
        → Model is punished if it predicts high probability for class 1

Full formula:
    loss = -[y*log(p) + (1-y)*log(1-p)]

Properties:
    - Always positive
    - Smaller is better
    - Heavily penalizes confident wrong predictions
""")

# Create loss function
# BCELoss: Binary Cross Entropy Loss
# Expects: predictions and targets both as probabilities in [0, 1]
criterion = nn.BCELoss()

print("Loss function: Binary Cross-Entropy (BCE)")

# ============================================================================
# 3.2: Optimizer
# ============================================================================
print("\n3.2: Choosing an Optimizer")
print("-" * 40)

print("""
Optimizer: Algorithm that updates model parameters (weights and biases)

Common optimizers:
    - SGD (Stochastic Gradient Descent): Basic but reliable
    - Adam: Adaptive learning rate, usually converges faster
    - RMSprop: Good for RNNs
    
We'll use SGD for simplicity and understanding.
""")

learning_rate = 0.1  # How big each update step is
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

print(f"Optimizer: SGD with learning rate = {learning_rate}")


print("\n" + "="*80)
print("PART 4: TRAINING THE MODEL")
print("="*80)

# ============================================================================
# 4.1: Training Loop
# ============================================================================
print("\n4.1: Training Loop")
print("-" * 40)

num_epochs = 1000  # Number of times to see the entire dataset
print_every = 100  # Print progress every N epochs

# Lists to store training history
train_losses = []
train_accuracies = []

print(f"Starting training for {num_epochs} epochs...")
print("-" * 40)

for epoch in range(num_epochs):
    # ====================
    # TRAINING STEP
    # ====================
    
    # 1. Forward Pass: Compute predictions
    # X_train shape: (160, 2)
    # predictions shape: (160, 1)
    predictions = model(X_train)  # Get model's predicted probabilities
    
    # 2. Compute Loss
    # Compare predictions with true labels
    loss = criterion(predictions, y_train)
    
    # 3. Backward Pass: Compute gradients
    optimizer.zero_grad()  # Clear old gradients (important!)
    loss.backward()        # Compute new gradients
    
    # 4. Update Parameters
    optimizer.step()       # Update weights and bias using gradients
    
    # ====================
    # TRACKING PROGRESS
    # ====================
    
    # Compute training accuracy
    with torch.no_grad():  # Don't compute gradients for evaluation
        # Convert probabilities to class predictions (0 or 1)
        # If probability >= 0.5, predict 1, else predict 0
        predicted_classes = (predictions >= 0.5).float()
        
        # Calculate accuracy: percentage of correct predictions
        correct = (predicted_classes == y_train).sum()
        accuracy = (correct / y_train.shape[0]).item()
    
    # Store history
    train_losses.append(loss.item())
    train_accuracies.append(accuracy)
    
    # Print progress
    if (epoch + 1) % print_every == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {loss.item():.4f} "
              f"Accuracy: {accuracy:.4f}")

print("\nTraining completed!")
print(f"Final loss: {train_losses[-1]:.4f}")
print(f"Final training accuracy: {train_accuracies[-1]:.4f}")


print("\n" + "="*80)
print("PART 5: EVALUATING THE MODEL")
print("="*80)

# ============================================================================
# 5.1: Test Set Evaluation
# ============================================================================
print("\n5.1: Performance on Test Set")
print("-" * 40)

# Evaluate on test set (unseen data)
model.eval()  # Set model to evaluation mode

with torch.no_grad():  # Don't compute gradients during evaluation
    # Get predictions on test set
    test_predictions = model(X_test)
    
    # Convert probabilities to class predictions
    test_predicted_classes = (test_predictions >= 0.5).float()
    
    # Calculate test accuracy
    test_correct = (test_predicted_classes == y_test).sum()
    test_accuracy = (test_correct / y_test.shape[0]).item()
    
    # Calculate test loss
    test_loss = criterion(test_predictions, y_test)

print(f"Test Loss: {test_loss.item():.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Correct predictions: {int(test_correct)}/{len(y_test)}")


print("\n" + "="*80)
print("PART 6: VISUALIZATION")
print("="*80)

# ============================================================================
# 6.1: Training History
# ============================================================================
print("\n6.1: Creating Visualizations...")

fig = plt.figure(figsize=(15, 10))

# Plot 1: Loss curve
plt.subplot(2, 3, 1)
plt.plot(train_losses, 'b-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Plot 2: Accuracy curve
plt.subplot(2, 3, 2)
plt.plot(train_accuracies, 'g-', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Training Accuracy Over Time', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.ylim([0, 1])

# Plot 3: Data distribution
plt.subplot(2, 3, 3)
X_train_np = X_train.numpy()
y_train_np = y_train.numpy().flatten()
plt.scatter(X_train_np[y_train_np==0, 0], X_train_np[y_train_np==0, 1], 
           c='blue', label='Class 0', alpha=0.6, edgecolors='k')
plt.scatter(X_train_np[y_train_np==1, 0], X_train_np[y_train_np==1, 1], 
           c='red', label='Class 1', alpha=0.6, edgecolors='k')
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Training Data Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 4: Decision boundary
plt.subplot(2, 3, 4)
# Create a mesh to plot decision boundary
x_min, x_max = X_train_np[:, 0].min() - 1, X_train_np[:, 0].max() + 1
y_min, y_max = X_train_np[:, 1].min() - 1, X_train_np[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict on mesh
with torch.no_grad():
    Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape).numpy()

plt.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
plt.colorbar(label='Probability')
plt.scatter(X_train_np[y_train_np==0, 0], X_train_np[y_train_np==0, 1], 
           c='blue', label='Class 0', edgecolors='k', s=50)
plt.scatter(X_train_np[y_train_np==1, 0], X_train_np[y_train_np==1, 1], 
           c='red', label='Class 1', edgecolors='k', s=50)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.title('Decision Boundary', fontsize=14, fontweight='bold')
plt.legend()

# Plot 5: Model parameters
plt.subplot(2, 3, 5)
weights = model.linear.weight.data.numpy().flatten()
bias = model.linear.bias.data.numpy()[0]
params_text = f"Learned Parameters:\n\n"
params_text += f"Weight 1 (w1): {weights[0]:.3f}\n"
params_text += f"Weight 2 (w2): {weights[1]:.3f}\n"
params_text += f"Bias (b): {bias:.3f}\n\n"
params_text += f"Decision boundary:\n"
params_text += f"{weights[0]:.3f}*x1 + {weights[1]:.3f}*x2 + {bias:.3f} = 0"
plt.text(0.1, 0.5, params_text, fontsize=12, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.axis('off')
plt.title('Model Parameters', fontsize=14, fontweight='bold')

# Plot 6: Performance summary
plt.subplot(2, 3, 6)
summary_text = f"Performance Summary\n\n"
summary_text += f"Training:\n"
summary_text += f"  Loss: {train_losses[-1]:.4f}\n"
summary_text += f"  Accuracy: {train_accuracies[-1]:.4f}\n\n"
summary_text += f"Testing:\n"
summary_text += f"  Loss: {test_loss.item():.4f}\n"
summary_text += f"  Accuracy: {test_accuracy:.4f}\n\n"
summary_text += f"Dataset:\n"
summary_text += f"  Training samples: {len(X_train)}\n"
summary_text += f"  Test samples: {len(X_test)}\n"
summary_text += f"  Features: {n_features}\n\n"
summary_text += f"Training:\n"
summary_text += f"  Epochs: {num_epochs}\n"
summary_text += f"  Learning rate: {learning_rate}"

plt.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
plt.axis('off')
plt.title('Summary', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_logistic_regression_tutorial/01_basics/simple_classification_results.png',
            dpi=150, bbox_inches='tight')
print("Visualization saved as: simple_classification_results.png")


print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)

print("""
1. LOGISTIC REGRESSION
   - Linear model + Sigmoid activation
   - Outputs probability between 0 and 1
   - Threshold at 0.5 for binary classification

2. TRAINING PROCESS
   - Forward pass: compute predictions
   - Calculate loss: measure error
   - Backward pass: compute gradients
   - Update parameters: improve model

3. IMPORTANT CONCEPTS
   - Always call optimizer.zero_grad() before backward()
   - Use model.eval() during evaluation
   - Use torch.no_grad() when not training

4. EVALUATION
   - Train set: what model learned from
   - Test set: how well it generalizes
   - Accuracy: percentage of correct predictions
""")


print("\n" + "="*80)
print("EXERCISES")
print("="*80)

print("""
1. EASY: Change learning_rate to 0.01 and 1.0
   - How does it affect convergence?
   - Which learns faster?

2. MEDIUM: Change num_epochs to 100 and 5000
   - Does more training always help?
   - Look for overfitting signs

3. MEDIUM: Try different train/test splits
   - Change test_size to 0.1 and 0.5
   - How does it affect results?

4. HARD: Implement a function to predict new data:
   def predict(model, x1, x2):
       # Your code here
       pass
   
   Test with: predict(model, 0.5, 0.5)

5. HARD: Add more features to the dataset:
   - Use n_features=4 or 10
   - Modify the model accordingly
   - Compare performance
""")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
Great job! You've built your first logistic regression model!

Next tutorial: 03_with_sklearn_data.py
- Work with real-world datasets
- Learn data preprocessing techniques
- Handle different data types

Ready? Run: python 03_with_sklearn_data.py
""")
print("="*80)
