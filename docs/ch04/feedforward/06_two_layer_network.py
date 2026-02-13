"""
================================================================================
02_two_layer_network.py - Your First Multi-Layer Neural Network
================================================================================

This example builds a 2-layer neural network to solve a NON-LINEAR problem.
This demonstrates why we need multiple layers and activation functions!

PROBLEM: Learn the XOR function (cannot be solved by a single neuron)
    XOR Truth Table:
    Input1  Input2  Output
      0       0       0
      0       1       1
      1       0       1
      1       1       0

KEY CONCEPT: Non-linearity
    - A single neuron can only learn linear decision boundaries
    - Adding hidden layers + activation functions enables learning complex patterns
    
ARCHITECTURE:
    Input (2) → Hidden Layer (4 neurons with ReLU) → Output (1 with Sigmoid)

LEARNING OBJECTIVES:
    1. Understand why we need hidden layers
    2. Learn the role of activation functions
    3. Build custom forward/backward passes
    4. Visualize decision boundaries

DIFFICULTY: ⭐⭐☆☆☆ (Beginner)
TIME: 20-30 minutes
================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ================================================================================
# PART 1: CREATE XOR DATASET
# ================================================================================
print("=" * 80)
print("STEP 1: Creating XOR Dataset")
print("=" * 80)

# Set seed for reproducibility
torch.manual_seed(42)

# XOR is a simple but important problem:
# It requires non-linear decision boundary (can't be solved by single layer)
X = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])

y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])

print("XOR Truth Table:")
print("-" * 40)
for i in range(len(X)):
    print(f"Input: [{X[i, 0]:.1f}, {X[i, 1]:.1f}] → Output: {y[i, 0]:.1f}")
print("-" * 40)

# ================================================================================
# PART 2: DEFINE TWO-LAYER NEURAL NETWORK
# ================================================================================
print("\n" + "=" * 80)
print("STEP 2: Building the Neural Network")
print("=" * 80)

class TwoLayerNet(nn.Module):
    """
    A simple 2-layer neural network.
    
    Architecture:
        Layer 1: Input (2) → Hidden (4) with ReLU activation
        Layer 2: Hidden (4) → Output (1) with Sigmoid activation
    
    Why this architecture?
        - 2 inputs: x1 and x2 (XOR has 2 inputs)
        - 4 hidden neurons: Enough to learn XOR (actually 2-3 would work)
        - 1 output: Binary classification (0 or 1)
        - ReLU: Introduces non-linearity (essential for learning XOR)
        - Sigmoid: Squashes output to (0, 1) range for probability
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize network layers.
        
        Args:
            input_size: Number of input features (2 for XOR)
            hidden_size: Number of neurons in hidden layer
            output_size: Number of output neurons (1 for binary classification)
        """
        super(TwoLayerNet, self).__init__()
        
        # Layer 1: Input → Hidden
        # Creates weight matrix of shape (hidden_size, input_size)
        # and bias vector of shape (hidden_size,)
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Activation function for hidden layer
        # ReLU(x) = max(0, x) - simple but effective!
        # Introduces non-linearity needed to solve XOR
        self.relu = nn.ReLU()
        
        # Layer 2: Hidden → Output
        # Creates weight matrix of shape (output_size, hidden_size)
        # and bias vector of shape (output_size,)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # Sigmoid activation for output
        # Sigmoid(x) = 1 / (1 + e^(-x))
        # Maps any value to range (0, 1) - perfect for probabilities!
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        This method defines how data flows through the network.
        PyTorch automatically builds the computational graph for backprop!
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Layer 1: Linear transformation
        # z1 = W1 * x + b1
        hidden = self.fc1(x)  # Shape: (batch_size, hidden_size)
        
        # Apply ReLU activation
        # a1 = ReLU(z1) = max(0, z1)
        # Without this, network would just be a linear transformation!
        hidden = self.relu(hidden)
        
        # Layer 2: Linear transformation
        # z2 = W2 * a1 + b2
        output = self.fc2(hidden)  # Shape: (batch_size, output_size)
        
        # Apply Sigmoid activation
        # y = Sigmoid(z2)
        # Converts to probability-like output
        output = self.sigmoid(output)
        
        return output
    
    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Create the network
input_size = 2      # XOR has 2 inputs
hidden_size = 4     # 4 hidden neurons (can experiment with this!)
output_size = 1     # Binary output (0 or 1)

model = TwoLayerNet(input_size, hidden_size, output_size)

print(f"Network Architecture:")
print(f"  Input layer:  {input_size} neurons")
print(f"  Hidden layer: {hidden_size} neurons (with ReLU)")
print(f"  Output layer: {output_size} neuron (with Sigmoid)")
print(f"\nTotal parameters: {model.count_parameters()}")
print(f"  Layer 1: {input_size * hidden_size + hidden_size} params (weights + biases)")
print(f"  Layer 2: {hidden_size * output_size + output_size} params (weights + biases)")

# ================================================================================
# PART 3: DEFINE TRAINING COMPONENTS
# ================================================================================
print("\n" + "=" * 80)
print("STEP 3: Setting Up Training")
print("=" * 80)

# Binary Cross-Entropy Loss
# Perfect for binary classification problems
# BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
# Penalizes confident wrong predictions heavily
criterion = nn.BCELoss()  # BCELoss expects sigmoid output

# Adam optimizer - more sophisticated than SGD
# Adapts learning rate for each parameter
# Generally works better than SGD without much tuning
learning_rate = 0.1
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(f"Loss function: Binary Cross-Entropy")
print(f"Optimizer: Adam")
print(f"Learning rate: {learning_rate}")

# ================================================================================
# PART 4: TRAINING LOOP
# ================================================================================
print("\n" + "=" * 80)
print("STEP 4: Training the Network")
print("=" * 80)

num_epochs = 5000
loss_history = []
print_interval = 500  # Print every 500 epochs

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X)
    loss = criterion(outputs, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()         # Compute new gradients
    optimizer.step()        # Update weights
    
    # Record loss
    loss_history.append(loss.item())
    
    # Print progress
    if (epoch + 1) % print_interval == 0:
        # Calculate accuracy
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y).float().mean() * 100
        print(f"Epoch [{epoch+1:4d}/{num_epochs}] | "
              f"Loss: {loss.item():.6f} | "
              f"Accuracy: {accuracy:.1f}%")

# ================================================================================
# PART 5: EVALUATE RESULTS
# ================================================================================
print("\n" + "=" * 80)
print("STEP 5: Final Evaluation")
print("=" * 80)

# Make predictions
model.eval()  # Set to evaluation mode (affects dropout/batchnorm if present)
with torch.no_grad():
    predictions = model(X)
    predicted_labels = (predictions > 0.5).float()

print("\nFinal Predictions:")
print("-" * 60)
print("Input 1 | Input 2 | True Output | Predicted | Probability")
print("-" * 60)
for i in range(len(X)):
    print(f"  {X[i, 0]:.1f}   |   {X[i, 1]:.1f}   |     {y[i, 0]:.1f}     "
          f"|    {predicted_labels[i, 0]:.1f}    |    {predictions[i, 0]:.4f}")
print("-" * 60)

# Calculate final accuracy
accuracy = (predicted_labels == y).float().mean() * 100
print(f"\nFinal Accuracy: {accuracy:.1f}%")

# ================================================================================
# PART 6: VISUALIZATIONS
# ================================================================================
print("\n" + "=" * 80)
print("STEP 6: Visualizing Results")
print("=" * 80)

fig = plt.figure(figsize=(15, 5))

# ----------------------------------------
# Plot 1: Training Loss
# ----------------------------------------
ax1 = plt.subplot(1, 3, 1)
ax1.plot(loss_history, linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss (BCE)', fontsize=12)
ax1.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# ----------------------------------------
# Plot 2: Decision Boundary
# ----------------------------------------
ax2 = plt.subplot(1, 3, 2)

# Create a mesh grid for visualization
x_min, x_max = -0.5, 1.5
y_min, y_max = -0.5, 1.5
h = 0.01  # Step size in mesh

xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Make predictions on mesh
mesh_input = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
with torch.no_grad():
    Z = model(mesh_input).numpy()
Z = Z.reshape(xx.shape)

# Plot decision boundary
contour = ax2.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
plt.colorbar(contour, ax=ax2, label='Prediction Probability')

# Plot data points
X_np = X.numpy()
y_np = y.numpy()
ax2.scatter(X_np[y_np.squeeze() == 0, 0], X_np[y_np.squeeze() == 0, 1],
           c='blue', s=200, edgecolors='black', linewidths=2, 
           marker='o', label='Class 0', zorder=10)
ax2.scatter(X_np[y_np.squeeze() == 1, 0], X_np[y_np.squeeze() == 1, 1],
           c='red', s=200, edgecolors='black', linewidths=2, 
           marker='s', label='Class 1', zorder=10)

ax2.set_xlim(x_min, x_max)
ax2.set_ylim(y_min, y_max)
ax2.set_xlabel('Input 1', fontsize=12)
ax2.set_ylabel('Input 2', fontsize=12)
ax2.set_title('Decision Boundary (XOR)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# ----------------------------------------
# Plot 3: Network Diagram
# ----------------------------------------
ax3 = plt.subplot(1, 3, 3)
ax3.axis('off')

# This is a simple text representation
network_text = """
Network Architecture:

Input Layer          Hidden Layer        Output Layer
(2 neurons)          (4 neurons)         (1 neuron)

    x₁ ─────────────┐
                     ├──→ h₁ ─┐
    x₂ ─────────────┤         ├──→ y
                     ├──→ h₂ ─┤
                     ├──→ h₃ ─┤
                     └──→ h₄ ─┘
                     
         ReLU                Sigmoid
    
Parameters:
  Layer 1: 2×4 weights + 4 biases = 12
  Layer 2: 4×1 weights + 1 bias = 5
  Total: 17 trainable parameters

Why it works:
✓ Multiple layers allow non-linear boundaries
✓ ReLU introduces non-linearity
✓ Enough neurons to represent XOR logic
"""

ax3.text(0.1, 0.5, network_text, fontsize=10, family='monospace',
        verticalalignment='center', transform=ax3.transAxes)

plt.tight_layout()
plt.savefig('02_two_layer_network_results.png', dpi=150, bbox_inches='tight')
print("Plot saved as '02_two_layer_network_results.png'")
plt.show()

# ================================================================================
# KEY TAKEAWAYS
# ================================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. Single layers can only learn LINEAR decision boundaries
   - Cannot solve XOR (requires non-linear boundary)
   
2. Hidden layers + activation functions enable NON-LINEAR learning
   - ReLU: Simple but effective: max(0, x)
   - Creates complex decision boundaries
   
3. Network architecture matters:
   - Too few neurons: Can't learn complex patterns
   - Too many neurons: Overfitting, slow training
   
4. Binary classification setup:
   - Sigmoid output: Maps to (0, 1) probability
   - BCE Loss: Appropriate for binary problems
   
5. The universal approximation theorem:
   - Even a 2-layer network can approximate any continuous function!
   - (with enough hidden neurons)

AMAZING: With just 17 parameters, we solved a non-linear problem!
""")

# ================================================================================
# EXERCISES FOR STUDENTS
# ================================================================================
print("=" * 80)
print("EXERCISES TO TRY")
print("=" * 80)
print("""
1. Try hidden_size = 2 or 3 - does it still work?
2. Replace ReLU with nn.Tanh() - compare convergence speed
3. Use SGD instead of Adam - what learning rate works?
4. Remove the ReLU activation - can it still learn XOR? (No!)
5. Visualize what each hidden neuron learns (weight visualization)
6. Create a 3-layer network (input → hidden1 → hidden2 → output)
""")
