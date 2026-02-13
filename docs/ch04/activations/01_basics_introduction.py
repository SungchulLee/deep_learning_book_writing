#!/usr/bin/env python3
"""
==============================================================================
Script 01: Introduction to Activation Functions
==============================================================================

DIFFICULTY: ⭐ Easy (Beginner)
ESTIMATED TIME: 5 minutes
PREREQUISITES: None - Start here!

LEARNING OBJECTIVES:
--------------------
1. Understand what activation functions are
2. Learn why neural networks need non-linear activation functions
3. See the difference between linear and non-linear transformations
4. Understand the role of activations in neural networks

KEY CONCEPTS:
-------------
- Activation functions introduce non-linearity
- Without activations, deep networks collapse to single linear layer
- Different activations have different properties
- Activation functions are applied element-wise

WHAT YOU'LL SEE:
----------------
- Simple examples with individual neurons
- Comparison of linear vs. non-linear outputs
- Basic activation function behavior

RUN THIS SCRIPT:
----------------
    python 01_basics_introduction.py

EXPECTED OUTPUT:
----------------
- Text output showing how activations transform values
- Demonstration of why we need non-linearity
"""

import torch
import torch.nn as nn

# ==============================================================================
# SECTION 1: What is an Activation Function?
# ==============================================================================

def section1_what_is_activation():
    """
    An activation function is a mathematical function applied to the output
    of a neuron. It determines whether a neuron should be "activated" or not.
    
    Think of it like: input → weights → sum → ACTIVATION → output
    """
    print("=" * 70)
    print("SECTION 1: What is an Activation Function?")
    print("=" * 70)
    
    # Let's start with a simple input value
    x = torch.tensor([2.0])
    print(f"\n1. Input value: {x.item()}")
    
    # Without activation (just linear transformation: output = input)
    linear_output = x
    print(f"   Without activation: {linear_output.item()}")
    
    # With ReLU activation (output = max(0, input))
    relu_output = torch.relu(x)
    print(f"   With ReLU activation: {relu_output.item()}")
    
    # Let's try a negative value
    x_negative = torch.tensor([-2.0])
    print(f"\n2. Input value: {x_negative.item()}")
    print(f"   Without activation: {x_negative.item()}")
    print(f"   With ReLU activation: {torch.relu(x_negative).item()}")
    print(f"   → ReLU blocks negative values!")
    
    # Sigmoid activation squashes values to (0, 1)
    x_large = torch.tensor([100.0])
    print(f"\n3. Input value: {x_large.item()}")
    print(f"   Without activation: {x_large.item()}")
    print(f"   With Sigmoid activation: {torch.sigmoid(x_large).item():.6f}")
    print(f"   → Sigmoid squashes large values close to 1!")


# ==============================================================================
# SECTION 2: Why Do We Need Activation Functions?
# ==============================================================================

def section2_why_we_need_activations():
    """
    WITHOUT activation functions:
    - Multiple layers collapse into a single linear transformation
    - Cannot learn complex, non-linear patterns
    - Network becomes equivalent to single-layer perceptron
    
    WITH activation functions:
    - Can approximate any continuous function (Universal Approximation Theorem)
    - Can learn complex decision boundaries
    - Can solve non-linear problems
    """
    print("\n" + "=" * 70)
    print("SECTION 2: Why We Need Non-Linear Activation Functions")
    print("=" * 70)
    
    # Simple input
    x = torch.tensor([[1.0, 2.0]])  # shape: (1, 2)
    
    print("\nExample: Stacking linear layers WITHOUT activations")
    print("-" * 70)
    
    # Two linear layers WITHOUT activation
    # Layer 1: multiply by 2
    w1 = torch.tensor([[2.0], [2.0]])  # shape: (2, 1)
    layer1_output = x @ w1  # matrix multiplication
    print(f"Input: {x[0].tolist()}")
    print(f"After Layer 1: {layer1_output[0].item()}")
    
    # Layer 2: multiply by 3
    w2 = torch.tensor([[3.0]])  # shape: (1, 1)
    layer2_output = layer1_output @ w2
    print(f"After Layer 2: {layer2_output[0].item()}")
    
    # This is equivalent to a single layer: (x @ w1) @ w2 = x @ (w1 @ w2)
    combined_weight = w1 @ w2
    direct_output = x @ combined_weight
    print(f"Direct computation (x @ (w1 @ w2)): {direct_output[0].item()}")
    print("→ Same result! Multiple linear layers = One linear layer")
    
    print("\n" + "-" * 70)
    print("Example: Adding ReLU activation")
    print("-" * 70)
    
    # With ReLU activation between layers
    layer1_output_relu = torch.relu(x @ w1)
    print(f"After Layer 1 + ReLU: {layer1_output_relu[0].item()}")
    
    layer2_output_relu = layer1_output_relu @ w2
    print(f"After Layer 2: {layer2_output_relu[0].item()}")
    
    # This is NOT equivalent to a single linear layer!
    print("\n→ With activation, we get non-linear transformation!")
    print("→ This allows the network to learn complex patterns")


# ==============================================================================
# SECTION 3: Common Activation Functions Preview
# ==============================================================================

def section3_common_activations():
    """
    Preview of the most common activation functions.
    We'll explore each in detail in later scripts.
    """
    print("\n" + "=" * 70)
    print("SECTION 3: Preview of Common Activation Functions")
    print("=" * 70)
    
    # Test values: negative, zero, positive
    test_values = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    print("\nInput values:", test_values.tolist())
    print("-" * 70)
    
    # 1. ReLU (Rectified Linear Unit)
    relu_output = torch.relu(test_values)
    print(f"ReLU:     {relu_output.tolist()}")
    print("   → Blocks negative values, keeps positive as-is")
    
    # 2. Sigmoid
    sigmoid_output = torch.sigmoid(test_values)
    print(f"\nSigmoid:  {[f'{v:.3f}' for v in sigmoid_output.tolist()]}")
    print("   → Squashes all values to range (0, 1)")
    
    # 3. Tanh (Hyperbolic Tangent)
    tanh_output = torch.tanh(test_values)
    print(f"\nTanh:     {[f'{v:.3f}' for v in tanh_output.tolist()]}")
    print("   → Squashes all values to range (-1, 1)")
    
    # 4. Leaky ReLU
    leaky_relu_output = torch.nn.functional.leaky_relu(test_values, negative_slope=0.1)
    print(f"\nLeaky ReLU: {[f'{v:.3f}' for v in leaky_relu_output.tolist()]}")
    print("   → Like ReLU, but allows small negative values")


# ==============================================================================
# SECTION 4: Activation Functions in Neural Networks
# ==============================================================================

def section4_activations_in_networks():
    """
    Demonstrate how activation functions are used in a simple neural network.
    """
    print("\n" + "=" * 70)
    print("SECTION 4: Activation Functions in Neural Networks")
    print("=" * 70)
    
    # Simple 2-layer neural network
    class SimpleNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # Two linear layers
            self.layer1 = nn.Linear(2, 3)  # 2 inputs → 3 hidden neurons
            self.layer2 = nn.Linear(3, 1)  # 3 hidden → 1 output
            # ReLU activation
            self.activation = nn.ReLU()
        
        def forward(self, x):
            # Layer 1 + activation
            x = self.layer1(x)
            print(f"   After layer 1 (before activation): {x[0].detach().tolist()}")
            x = self.activation(x)
            print(f"   After ReLU activation: {x[0].detach().tolist()}")
            
            # Layer 2 (output layer, usually no activation for regression)
            x = self.layer2(x)
            print(f"   After layer 2 (output): {x[0].detach().item():.4f}")
            return x
    
    # Create network and sample input
    net = SimpleNetwork()
    sample_input = torch.tensor([[1.0, 2.0]])
    
    print("\nForward pass through the network:")
    print("-" * 70)
    print(f"Input: {sample_input[0].tolist()}")
    
    # Forward pass (with automatic printing)
    output = net(sample_input)
    
    print("\n→ Notice how ReLU zeros out any negative values from layer 1!")


# ==============================================================================
# SECTION 5: Key Takeaways
# ==============================================================================

def section5_key_takeaways():
    """
    Summary of what we learned in this script.
    """
    print("\n" + "=" * 70)
    print("SECTION 5: Key Takeaways 🎓")
    print("=" * 70)
    
    takeaways = [
        "1. Activation functions introduce NON-LINEARITY to neural networks",
        "2. Without activations, deep networks collapse to a single linear layer",
        "3. Activations are applied ELEMENT-WISE (each value independently)",
        "4. Different activations have different properties:",
        "   - ReLU: Blocks negative values",
        "   - Sigmoid: Squashes to (0, 1)",
        "   - Tanh: Squashes to (-1, 1)",
        "5. Activations are typically applied AFTER linear layers",
        "6. Choice of activation affects network behavior and training",
    ]
    
    for takeaway in takeaways:
        print(f"\n{takeaway}")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("Run '02_functional_vs_module.py' to learn PyTorch's two activation APIs")
    print("=" * 70)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Run all sections of this tutorial script.
    """
    print("\n" + "█" * 70)
    print("   PYTORCH ACTIVATION FUNCTIONS TUTORIAL")
    print("   Script 01: Introduction to Activation Functions")
    print("█" * 70)
    
    # Run all sections
    section1_what_is_activation()
    section2_why_we_need_activations()
    section3_common_activations()
    section4_activations_in_networks()
    section5_key_takeaways()
    
    print("\n✅ Script completed successfully!\n")


if __name__ == "__main__":
    main()
