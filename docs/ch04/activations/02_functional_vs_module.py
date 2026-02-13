#!/usr/bin/env python3
"""
==============================================================================
Script 02: Functional API vs Module API for Activation Functions
==============================================================================

DIFFICULTY: ⭐ Easy
ESTIMATED TIME: 5-7 minutes
PREREQUISITES: Script 01

LEARNING OBJECTIVES:
--------------------
1. Understand PyTorch's two ways to use activation functions
2. Learn when to use Functional API vs Module API
3. Master the differences and best practices
4. See practical examples of both approaches

KEY CONCEPTS:
-------------
- Functional API: torch.relu(), F.relu(), torch.sigmoid()
  → Stateless functions, called directly
- Module API: nn.ReLU(), nn.Sigmoid()
  → Stateful objects, can be part of nn.Sequential
- Both produce identical outputs
- Choice is often a matter of style and use case

WHAT YOU'LL SEE:
----------------
- Side-by-side comparisons of both APIs
- When each approach is more convenient
- Integration with nn.Sequential and custom modules

RUN THIS SCRIPT:
----------------
    python 02_functional_vs_module.py

EXPECTED OUTPUT:
----------------
- Demonstrations of both API styles
- Comparison outputs showing equivalence
- Best practice guidelines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F  # Functional API lives here

# ==============================================================================
# SECTION 1: Functional API Introduction
# ==============================================================================

def section1_functional_api():
    """
    Functional API: Direct function calls without creating objects.
    Found in: torch.* or torch.nn.functional.*
    
    Advantages:
    - Concise and direct
    - No need to instantiate objects
    - Good for one-off operations
    """
    print("=" * 70)
    print("SECTION 1: Functional API")
    print("=" * 70)
    
    # Sample input tensor
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    print(f"\nInput: {x.tolist()}")
    print("-" * 70)
    
    # Functional API examples
    print("\n1. Using torch.relu() - directly from torch module:")
    output1 = torch.relu(x)
    print(f"   Output: {output1.tolist()}")
    
    print("\n2. Using F.relu() - from torch.nn.functional:")
    output2 = F.relu(x)
    print(f"   Output: {output2.tolist()}")
    print(f"   → Same result! F is shorthand for torch.nn.functional")
    
    print("\n3. Other functional activations:")
    print(f"   Sigmoid: {torch.sigmoid(x).tolist()}")
    print(f"   Tanh:    {torch.tanh(x).tolist()}")
    print(f"   Leaky ReLU: {F.leaky_relu(x, negative_slope=0.1).tolist()}")
    
    print("\n" + "-" * 70)
    print("KEY POINT: Functional API = Direct function calls")
    print("           No object creation, stateless operations")
    print("-" * 70)


# ==============================================================================
# SECTION 2: Module API Introduction
# ==============================================================================

def section2_module_api():
    """
    Module API: Create activation objects that are callable.
    Found in: torch.nn.*
    
    Advantages:
    - Can be stored as class attributes
    - Works seamlessly with nn.Sequential
    - Can have learnable parameters (e.g., PReLU)
    """
    print("\n" + "=" * 70)
    print("SECTION 2: Module API")
    print("=" * 70)
    
    # Sample input tensor
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    print(f"\nInput: {x.tolist()}")
    print("-" * 70)
    
    # Create activation modules (objects)
    print("\n1. Creating activation modules:")
    relu = nn.ReLU()
    sigmoid = nn.Sigmoid()
    tanh = nn.Tanh()
    leaky_relu = nn.LeakyReLU(negative_slope=0.1)
    
    print(f"   relu object: {relu}")
    print(f"   sigmoid object: {sigmoid}")
    
    # Apply activations by calling the objects
    print("\n2. Applying activations (call the objects):")
    print(f"   ReLU:       {relu(x).tolist()}")
    print(f"   Sigmoid:    {[f'{v:.3f}' for v in sigmoid(x).tolist()]}")
    print(f"   Tanh:       {[f'{v:.3f}' for v in tanh(x).tolist()]}")
    print(f"   Leaky ReLU: {leaky_relu(x).tolist()}")
    
    print("\n" + "-" * 70)
    print("KEY POINT: Module API = Create objects, then call them")
    print("           Can be reused, stored, and integrated with nn.Sequential")
    print("-" * 70)


# ==============================================================================
# SECTION 3: Side-by-Side Comparison
# ==============================================================================

def section3_side_by_side_comparison():
    """
    Direct comparison showing both APIs produce identical results.
    """
    print("\n" + "=" * 70)
    print("SECTION 3: Side-by-Side Comparison (Identical Results)")
    print("=" * 70)
    
    x = torch.tensor([-3.0, -1.5, 0.0, 1.5, 3.0])
    
    print(f"\nInput: {x.tolist()}\n")
    
    # ReLU comparison
    functional_relu = torch.relu(x)
    module_relu = nn.ReLU()(x)
    print(f"Functional ReLU: {functional_relu.tolist()}")
    print(f"Module ReLU:     {module_relu.tolist()}")
    print(f"Are they equal? {torch.equal(functional_relu, module_relu)}")
    
    print()
    
    # Sigmoid comparison
    functional_sigmoid = torch.sigmoid(x)
    module_sigmoid = nn.Sigmoid()(x)
    print(f"Functional Sigmoid: {[f'{v:.4f}' for v in functional_sigmoid.tolist()]}")
    print(f"Module Sigmoid:     {[f'{v:.4f}' for v in module_sigmoid.tolist()]}")
    print(f"Are they equal? {torch.equal(functional_sigmoid, module_sigmoid)}")
    
    print()
    
    # Leaky ReLU comparison
    functional_leaky = F.leaky_relu(x, negative_slope=0.2)
    module_leaky = nn.LeakyReLU(negative_slope=0.2)(x)
    print(f"Functional Leaky ReLU: {functional_leaky.tolist()}")
    print(f"Module Leaky ReLU:     {module_leaky.tolist()}")
    print(f"Are they equal? {torch.equal(functional_leaky, module_leaky)}")
    
    print("\n" + "-" * 70)
    print("CONCLUSION: Both APIs produce IDENTICAL outputs!")
    print("-" * 70)


# ==============================================================================
# SECTION 4: Use Cases - When to Use Each
# ==============================================================================

def section4_when_to_use_each():
    """
    Practical examples showing when each API style is more convenient.
    """
    print("\n" + "=" * 70)
    print("SECTION 4: When to Use Each API")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # USE CASE 1: Module API with nn.Sequential
    # -------------------------------------------------------------------------
    print("\n✅ USE CASE 1: Module API is great with nn.Sequential")
    print("-" * 70)
    
    # Easy to build layer-by-layer with Module API
    model_sequential = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),           # Module API - object
        nn.Linear(20, 30),
        nn.ReLU(),           # Module API - object
        nn.Linear(30, 1)
    )
    
    print("Sequential model (using Module API):")
    print(model_sequential)
    
    # -------------------------------------------------------------------------
    # USE CASE 2: Functional API in forward() method
    # -------------------------------------------------------------------------
    print("\n✅ USE CASE 2: Functional API is concise in forward()")
    print("-" * 70)
    
    class CustomNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # Only store layers, not activations
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 30)
            self.fc3 = nn.Linear(30, 1)
        
        def forward(self, x):
            # Apply activations inline using Functional API
            x = F.relu(self.fc1(x))  # Concise!
            x = F.relu(self.fc2(x))  # Concise!
            x = self.fc3(x)
            return x
    
    model_functional = CustomNetwork()
    print("Custom model (using Functional API in forward):")
    print(model_functional)
    
    # -------------------------------------------------------------------------
    # USE CASE 3: Module API when activation needs to be stored
    # -------------------------------------------------------------------------
    print("\n✅ USE CASE 3: Module API when you need to store activation")
    print("-" * 70)
    
    class NetworkWithStoredActivation(nn.Module):
        def __init__(self, activation_type='relu'):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 1)
            
            # Store activation as an attribute (easy to change later)
            if activation_type == 'relu':
                self.activation = nn.ReLU()
            elif activation_type == 'tanh':
                self.activation = nn.Tanh()
            else:
                self.activation = nn.Sigmoid()
        
        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.fc2(x)
            return x
    
    model_with_relu = NetworkWithStoredActivation(activation_type='relu')
    model_with_tanh = NetworkWithStoredActivation(activation_type='tanh')
    
    print("Network with ReLU:")
    print(f"  Activation: {model_with_relu.activation}")
    print("\nNetwork with Tanh:")
    print(f"  Activation: {model_with_tanh.activation}")
    
    # -------------------------------------------------------------------------
    # USE CASE 4: Functional API for one-off operations
    # -------------------------------------------------------------------------
    print("\n✅ USE CASE 4: Functional API for quick one-off operations")
    print("-" * 70)
    
    # Just need to apply activation once? Use functional!
    x = torch.randn(5)
    activated = torch.relu(x)  # Quick and simple!
    print(f"Quick ReLU application: {activated.tolist()}")


# ==============================================================================
# SECTION 5: Softmax Dimension - Important Detail
# ==============================================================================

def section5_softmax_dimension():
    """
    Special attention to Softmax, which requires specifying a dimension.
    """
    print("\n" + "=" * 70)
    print("SECTION 5: Special Case - Softmax (Dimension Matters!)")
    print("=" * 70)
    
    # Batch of class logits: (batch_size=3, num_classes=4)
    logits = torch.tensor([
        [1.0, 2.0, 3.0, 4.0],
        [4.0, 3.0, 2.0, 1.0],
        [2.0, 2.0, 2.0, 2.0]
    ])
    
    print(f"\nLogits shape: {logits.shape} (batch_size=3, num_classes=4)")
    print(f"Logits:\n{logits}")
    
    # Apply softmax over the class dimension (dim=1)
    print("\n1. Functional API - must specify dim:")
    probs_functional = F.softmax(logits, dim=1)  # dim=1 means across classes
    print(f"   Probabilities:\n{probs_functional}")
    print(f"   Row sums (should be 1.0): {probs_functional.sum(dim=1).tolist()}")
    
    print("\n2. Module API - specify dim in constructor:")
    softmax_module = nn.Softmax(dim=1)  # Create with dim parameter
    probs_module = softmax_module(logits)
    print(f"   Probabilities:\n{probs_module}")
    print(f"   Row sums (should be 1.0): {probs_module.sum(dim=1).tolist()}")
    
    print("\n" + "-" * 70)
    print("IMPORTANT: For classification with shape (N, C):")
    print("           Use dim=1 to get probability distribution over classes")
    print("-" * 70)


# ==============================================================================
# SECTION 6: Best Practices Summary
# ==============================================================================

def section6_best_practices():
    """
    Summary of best practices for choosing between APIs.
    """
    print("\n" + "=" * 70)
    print("SECTION 6: Best Practices 🎓")
    print("=" * 70)
    
    practices = [
        "\n✅ USE MODULE API WHEN:",
        "   • Building with nn.Sequential",
        "   • Need to store activation as class attribute",
        "   • Activation has learnable parameters (PReLU)",
        "   • Want to easily swap activations",
        
        "\n✅ USE FUNCTIONAL API WHEN:",
        "   • Writing custom forward() methods",
        "   • Need quick one-off activation",
        "   • Want more concise code",
        "   • Don't need to store activation object",
        
        "\n⚠️  REMEMBER:",
        "   • Both produce identical outputs",
        "   • Can mix both styles in same model",
        "   • Softmax requires 'dim' parameter",
        "   • Module API objects are callable (act like functions)",
    ]
    
    for practice in practices:
        print(practice)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("Run '03_visualizing_activations.py' to see activation curves!")
    print("=" * 70)


# ==============================================================================
# SECTION 7: Practical Example - Both Styles in Action
# ==============================================================================

def section7_practical_example():
    """
    Complete example showing both styles working together.
    """
    print("\n" + "=" * 70)
    print("SECTION 7: Practical Example - Mixed Styles")
    print("=" * 70)
    
    class HybridNetwork(nn.Module):
        """
        Network that uses BOTH Module and Functional APIs.
        This is perfectly fine and common in practice!
        """
        def __init__(self):
            super().__init__()
            # Layers
            self.fc1 = nn.Linear(5, 10)
            self.fc2 = nn.Linear(10, 20)
            self.fc3 = nn.Linear(20, 1)
            
            # Store one activation as module (for demonstration)
            self.activation_module = nn.ReLU()
        
        def forward(self, x):
            # Use stored module
            x = self.activation_module(self.fc1(x))
            
            # Use functional API inline
            x = F.relu(self.fc2(x))
            
            # Output layer (no activation for regression)
            x = self.fc3(x)
            return x
    
    model = HybridNetwork()
    sample_input = torch.randn(3, 5)  # 3 samples, 5 features
    
    print("\nHybrid model (mixing both styles):")
    print(model)
    
    print("\nForward pass:")
    output = model(sample_input)
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")
    
    print("\n→ Both styles work together seamlessly!")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Run all sections of this tutorial script.
    """
    print("\n" + "█" * 70)
    print("   PYTORCH ACTIVATION FUNCTIONS TUTORIAL")
    print("   Script 02: Functional API vs Module API")
    print("█" * 70)
    
    # Run all sections
    section1_functional_api()
    section2_module_api()
    section3_side_by_side_comparison()
    section4_when_to_use_each()
    section5_softmax_dimension()
    section6_best_practices()
    section7_practical_example()
    
    print("\n✅ Script completed successfully!\n")


if __name__ == "__main__":
    main()
