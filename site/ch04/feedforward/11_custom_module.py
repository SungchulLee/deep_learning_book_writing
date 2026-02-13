"""
==================================================================
05_custom_module.py - Creating Custom nn.Module Classes
==================================================================

Learn to build flexible, reusable network components by subclassing
nn.Module. This is the preferred way for complex architectures!

WHY CUSTOM nn.MODULE:
    ✓ Full control over forward pass
    ✓ Multiple inputs/outputs
    ✓ Conditional execution
    ✓ Custom parameter initialization
    ✓ Hooks and debugging support

LEARNING OBJECTIVES:
    1. Proper nn.Module subclassing
    2. Parameter registration
    3. Forward method implementation
    4. Advanced features (hooks, custom init)
    5. Best practices and patterns

DIFFICULTY: ⭐⭐⭐☆☆
TIME: 25-30 minutes
==================================================================
"""

import torch
import torch.nn as nn
import torch.nn.init as init

print("="*70)
print("PART 1: Basic Custom Module")
print("="*70)

class BasicNet(nn.Module):
    """
    Basic custom network showing fundamental nn.Module structure.
    
    REQUIRED METHODS:
        - __init__: Define layers and parameters
        - forward: Define computation flow
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        # ALWAYS call parent constructor first!
        super(BasicNet, self).__init__()
        
        # Define layers as attributes
        # PyTorch automatically tracks these as parameters
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
        # You can also store non-parameter attributes
        self.input_size = input_size
        
    def forward(self, x):
        """
        Forward pass - defines how data flows through network.
        PyTorch builds computation graph automatically!
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

model = BasicNet(784, 128, 10)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

print("\n"+"="*70)
print("PART 2: Module with Custom Initialization")
print("="*70)

class InitializedNet(nn.Module):
    """Network with custom weight initialization."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super(InitializedNet, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
        # Custom initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier uniform for layers followed by ReLU
                init.xavier_uniform_(m.weight)
                # Initialize biases to small positive value
                init.constant_(m.bias, 0.01)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = InitializedNet(784, 128, 10)
print("Custom initialization applied!")
print(f"First layer bias sample: {model.fc1.bias[:5]}")

print("\n"+"="*70)
print("PART 3: Module with Multiple Paths")
print("="*70)

class MultiPathNet(nn.Module):
    """
    Network with multiple computational paths.
    Demonstrates why custom modules are necessary!
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super(MultiPathNet, self).__init__()
        
        # Path 1: Deep path
        self.deep_path = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Path 2: Shallow path (skip connection)
        self.shallow_path = nn.Linear(input_size, hidden_size)
        
        # Combine paths
        self.output = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Compute both paths
        deep = self.deep_path(x)
        shallow = self.shallow_path(x)
        
        # Combine with residual connection
        combined = deep + shallow  # Element-wise addition
        
        # Final output
        return self.output(combined)

model = MultiPathNet(784, 128, 10)
print("Multi-path network with skip connections created!")

print("\n"+"="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
1. Always call super().__init__() first in __init__
2. Define layers as instance attributes (self.xxx)
3. Implement forward() to define computation
4. PyTorch automatically:
   - Tracks parameters
   - Builds computation graph
   - Enables backpropagation
5. Custom modules enable:
   - Skip connections
   - Multiple inputs/outputs
   - Conditional logic
   - Custom initialization
""")

print("\n"+"="*70)
print("EXERCISES")
print("="*70)
print("""
1. Add dropout to BasicNet
2. Create a ResNet-style block with skip connections
3. Implement custom weight initialization schemes
4. Build a network that uses different paths for different inputs
5. Add layer normalization to the network
""")
