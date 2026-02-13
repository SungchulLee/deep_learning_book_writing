"""
Module 64.02: TorchScript Basics
=================================

This module introduces TorchScript, PyTorch's way to create serializable and optimizable
models that can run independently from Python. TorchScript is essential for production
deployment, especially in C++ environments or mobile devices.

Learning Objectives:
-------------------
1. Understand what TorchScript is and why it's useful
2. Learn the difference between tracing and scripting
3. Convert PyTorch models to TorchScript
4. Save and load TorchScript models
5. Optimize models with TorchScript

Time Estimate: 60 minutes
Difficulty: Beginner
Prerequisites: Module 64.01 (Basic Model Saving)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import time
from typing import Tuple, List


# ============================================================================
# PART 1: Simple Models for TorchScript Conversion
# ============================================================================

class SimpleModel(nn.Module):
    """
    A simple model that's easy to convert to TorchScript.
    
    TorchScript works best with:
    - Simple control flow
    - Standard PyTorch operations
    - Type-annotated forward methods
    """
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        """
        Initialize a simple feedforward network.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden neurons
            num_classes: Number of output classes
        """
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with type annotations (required for TorchScript).
        
        Type annotations tell TorchScript what types to expect,
        which enables better optimization and error checking.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ModelWithControlFlow(nn.Module):
    """
    Model with control flow (if statements).
    
    This demonstrates how TorchScript handles conditional logic.
    Control flow must be scriptable (compatible with TorchScript's constraints).
    """
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super(ModelWithControlFlow, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x: torch.Tensor, training: bool = False) -> torch.Tensor:
        """
        Forward pass with conditional logic.
        
        Note: TorchScript can handle if statements, but they must be
        based on constants or simple comparisons, not on tensor values.
        
        Args:
            x: Input tensor
            training: Whether in training mode (affects dropout)
            
        Returns:
            Output tensor
        """
        x = F.relu(self.fc1(x))
        
        # Conditional logic - scriptable because 'training' is a bool
        if training:
            x = self.dropout(x)
            
        x = self.fc2(x)
        return x


class ModelWithLoop(nn.Module):
    """
    Model with loop operations.
    
    Demonstrates how TorchScript handles loops, which is important
    for RNNs and other sequential models.
    """
    
    def __init__(self, hidden_size: int = 128):
        super(ModelWithLoop, self).__init__()
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(10, hidden_size)
        self.output = nn.Linear(hidden_size, 10)
        
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Process sequence with a loop.
        
        Args:
            inputs: Tensor of shape (batch_size, sequence_length, 10)
            
        Returns:
            Output tensor of shape (batch_size, 10)
        """
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)
        
        # Initialize hidden state
        hidden = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        
        # Loop through sequence
        # TorchScript can optimize this loop
        for i in range(seq_len):
            hidden = self.rnn_cell(inputs[:, i, :], hidden)
            
        # Return output from final hidden state
        return self.output(hidden)


# ============================================================================
# PART 2: TorchScript Method 1 - Tracing
# ============================================================================

def demonstrate_tracing():
    """
    Tracing: Records operations performed on example inputs.
    
    How it works:
    1. Run model on example input
    2. Record all operations performed
    3. Create a static graph of these operations
    
    Advantages:
    - Simple and automatic
    - Works with complex models
    
    Disadvantages:
    - Cannot handle data-dependent control flow
    - Captures only one execution path
    - Requires representative example inputs
    """
    print(f"\n{'='*70}")
    print("METHOD 1: TRACING")
    print(f"{'='*70}")
    
    # Create model
    model = SimpleModel()
    model.eval()  # Set to evaluation mode
    
    # Create example input
    # This input determines what operations get recorded
    example_input = torch.randn(1, 784)
    
    print("\n1. Original PyTorch Model")
    print(f"   Input shape: {example_input.shape}")
    with torch.no_grad():
        output = model(example_input)
    print(f"   Output shape: {output.shape}")
    print(f"   Output sample: {output[0, :3]}")
    
    # Trace the model
    print("\n2. Tracing the Model")
    print("   torch.jit.trace() records all operations...")
    traced_model = torch.jit.trace(model, example_input)
    print("   ✓ Model traced successfully")
    
    # Test traced model
    print("\n3. Testing Traced Model")
    with torch.no_grad():
        traced_output = traced_model(example_input)
    print(f"   Output shape: {traced_output.shape}")
    print(f"   Output sample: {traced_output[0, :3]}")
    
    # Verify outputs match
    max_diff = torch.max(torch.abs(output - traced_output)).item()
    print(f"   Max difference from original: {max_diff:.10f}")
    print("   ✓ Outputs match (traced model is equivalent)")
    
    # Save traced model
    Path("models").mkdir(exist_ok=True)
    save_path = "models/traced_model.pt"
    traced_model.save(save_path)
    print(f"\n4. Saved traced model to: {save_path}")
    
    # View the traced graph
    print("\n5. Traced Graph Structure (partial):")
    print("   " + str(traced_model.graph)[:500] + "...")
    
    return traced_model


def demonstrate_tracing_limitations():
    """
    Demonstrate limitations of tracing with control flow.
    
    Key point: Tracing only captures ONE execution path.
    If your model has if/else or loops that depend on input data,
    tracing will only capture the path taken during tracing.
    """
    print(f"\n{'='*70}")
    print("TRACING LIMITATIONS: CONTROL FLOW")
    print(f"{'='*70}")
    
    model = ModelWithControlFlow()
    model.eval()
    
    example_input = torch.randn(1, 784)
    
    # Trace with training=False
    print("\n1. Tracing with training=False")
    traced_model = torch.jit.trace(
        model, 
        (example_input, False),  # Example inputs as tuple
        check_trace=True  # Verify trace is correct
    )
    print("   ✓ Model traced")
    
    # Test with training=False (should work)
    print("\n2. Testing traced model with training=False")
    output1 = traced_model(example_input, False)
    print(f"   Output: {output1[0, :3]}")
    
    # Test with training=True (will use same path as training=False!)
    print("\n3. Testing traced model with training=True")
    output2 = traced_model(example_input, True)
    print(f"   Output: {output2[0, :3]}")
    
    # Compare
    if torch.allclose(output1, output2):
        print("   ⚠️  WARNING: Outputs are identical!")
        print("   This is because tracing captured only the training=False path")
        print("   Solution: Use torch.jit.script() for models with control flow")


# ============================================================================
# PART 3: TorchScript Method 2 - Scripting
# ============================================================================

def demonstrate_scripting():
    """
    Scripting: Analyzes Python source code to generate TorchScript.
    
    How it works:
    1. Parse Python source code
    2. Compile to TorchScript IR (Intermediate Representation)
    3. Support data-dependent control flow
    
    Advantages:
    - Handles control flow correctly
    - Captures all execution paths
    - Type checking at compilation
    
    Disadvantages:
    - Requires Python subset compatibility
    - May need code modifications
    - Less flexible with dynamic operations
    """
    print(f"\n{'='*70}")
    print("METHOD 2: SCRIPTING")
    print(f"{'='*70}")
    
    # Create model with control flow
    model = ModelWithControlFlow()
    model.eval()
    
    print("\n1. Original Model with Control Flow")
    example_input = torch.randn(1, 784)
    
    with torch.no_grad():
        output_train_false = model(example_input, training=False)
        output_train_true = model(example_input, training=True)
    
    print(f"   Output (training=False): {output_train_false[0, :3]}")
    print(f"   Output (training=True): {output_train_true[0, :3]}")
    print(f"   Outputs differ: {not torch.allclose(output_train_false, output_train_true)}")
    
    # Script the model
    print("\n2. Scripting the Model")
    print("   torch.jit.script() analyzes Python code...")
    scripted_model = torch.jit.script(model)
    print("   ✓ Model scripted successfully")
    
    # Test scripted model with both conditions
    print("\n3. Testing Scripted Model")
    with torch.no_grad():
        scripted_output_false = scripted_model(example_input, False)
        scripted_output_true = scripted_model(example_input, True)
    
    print(f"   Output (training=False): {scripted_output_false[0, :3]}")
    print(f"   Output (training=True): {scripted_output_true[0, :3]}")
    print(f"   Outputs differ: {not torch.allclose(scripted_output_false, scripted_output_true)}")
    print("   ✓ Both execution paths work correctly!")
    
    # Save scripted model
    save_path = "models/scripted_model.pt"
    scripted_model.save(save_path)
    print(f"\n4. Saved scripted model to: {save_path}")
    
    # View scripted code
    print("\n5. Scripted Code (partial):")
    print("   " + str(scripted_model.code)[:500] + "...")
    
    return scripted_model


def demonstrate_scripting_loops():
    """
    Demonstrate scripting with loops.
    
    Loops are particularly important for RNNs and sequential models.
    Scripting can optimize loops that tracing cannot.
    """
    print(f"\n{'='*70}")
    print("SCRIPTING WITH LOOPS")
    print(f"{'='*70}")
    
    model = ModelWithLoop(hidden_size=64)
    model.eval()
    
    # Create sequence input
    batch_size = 2
    seq_len = 5
    input_size = 10
    example_input = torch.randn(batch_size, seq_len, input_size)
    
    print(f"\n1. Processing sequence with {seq_len} timesteps")
    print(f"   Input shape: {example_input.shape}")
    
    # Script the model
    print("\n2. Scripting model with loop")
    scripted_model = torch.jit.script(model)
    print("   ✓ Loop-based model scripted successfully")
    
    # Test
    with torch.no_grad():
        output = model(example_input)
        scripted_output = scripted_model(example_input)
    
    print(f"\n3. Testing")
    print(f"   Original output: {output[0, :3]}")
    print(f"   Scripted output: {scripted_output[0, :3]}")
    print(f"   Max difference: {torch.max(torch.abs(output - scripted_output)).item():.10f}")
    print("   ✓ Outputs match")


# ============================================================================
# PART 4: Loading and Using TorchScript Models
# ============================================================================

def demonstrate_loading_torchscript():
    """
    Loading TorchScript models is simple and doesn't require
    the original model class definition.
    
    This is a key advantage: You can load the model anywhere
    (Python, C++, mobile) without the original source code.
    """
    print(f"\n{'='*70}")
    print("LOADING TORCHSCRIPT MODELS")
    print(f"{'='*70}")
    
    # Load traced model
    print("\n1. Loading Traced Model")
    traced_loaded = torch.jit.load("models/traced_model.pt")
    print("   ✓ Traced model loaded")
    print(f"   Model type: {type(traced_loaded)}")
    
    # Load scripted model
    print("\n2. Loading Scripted Model")
    scripted_loaded = torch.jit.load("models/scripted_model.pt")
    print("   ✓ Scripted model loaded")
    print(f"   Model type: {type(scripted_loaded)}")
    
    # Use loaded models for inference
    print("\n3. Running Inference")
    example_input = torch.randn(1, 784)
    
    with torch.no_grad():
        traced_output = traced_loaded(example_input)
        scripted_output = scripted_loaded(example_input, False)
    
    print(f"   Traced model output: {traced_output[0, :3]}")
    print(f"   Scripted model output: {scripted_output[0, :3]}")
    print("   ✓ Inference successful")
    
    # No need for original class definitions!
    print("\n4. Key Advantage: No Original Code Needed")
    print("   ✓ Loaded models work without SimpleModel or ModelWithControlFlow")
    print("   ✓ Can deploy to C++ or mobile without Python")
    print("   ✓ Self-contained serialization")


# ============================================================================
# PART 5: Performance Comparison
# ============================================================================

def benchmark_torchscript_performance():
    """
    Compare performance between regular PyTorch and TorchScript.
    
    TorchScript can be faster because:
    1. Optimized graph execution
    2. Reduced Python overhead
    3. Fusion of operations
    4. Better memory management
    """
    print(f"\n{'='*70}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    
    # Create models
    model_pytorch = SimpleModel()
    model_pytorch.eval()
    
    model_scripted = torch.jit.script(model_pytorch)
    
    # Create test data
    batch_size = 32
    test_input = torch.randn(batch_size, 784)
    num_iterations = 1000
    
    # Warm up (JIT compilation)
    print("\n1. Warming up models (JIT compilation)...")
    with torch.no_grad():
        for _ in range(10):
            _ = model_pytorch(test_input)
            _ = model_scripted(test_input)
    print("   ✓ Warm-up complete")
    
    # Benchmark PyTorch model
    print(f"\n2. Benchmarking PyTorch model ({num_iterations} iterations)")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model_pytorch(test_input)
    pytorch_time = time.time() - start_time
    print(f"   Time: {pytorch_time:.4f} seconds")
    print(f"   Throughput: {num_iterations / pytorch_time:.2f} batches/second")
    
    # Benchmark TorchScript model
    print(f"\n3. Benchmarking TorchScript model ({num_iterations} iterations)")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model_scripted(test_input)
    torchscript_time = time.time() - start_time
    print(f"   Time: {torchscript_time:.4f} seconds")
    print(f"   Throughput: {num_iterations / torchscript_time:.2f} batches/second")
    
    # Compare
    speedup = pytorch_time / torchscript_time
    print(f"\n4. Performance Comparison")
    print(f"   Speedup: {speedup:.2f}x")
    if speedup > 1:
        print(f"   ✓ TorchScript is {speedup:.2f}x faster")
    else:
        print(f"   Note: Speedup varies by model complexity and hardware")


# ============================================================================
# PART 6: Decision Guide: Tracing vs Scripting
# ============================================================================

def print_decision_guide():
    """
    Print a decision guide for choosing between tracing and scripting.
    """
    print(f"\n{'='*70}")
    print("DECISION GUIDE: TRACING VS SCRIPTING")
    print(f"{'='*70}")
    
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                    WHEN TO USE TRACING                             ║
╠════════════════════════════════════════════════════════════════════╣
║ ✓ Model has no control flow (if/else, loops based on data)        ║
║ ✓ Simple feedforward networks (CNNs, simple MLPs)                 ║
║ ✓ Pre-trained models you don't want to modify                     ║
║ ✓ Quick prototyping                                               ║
║ ✓ When you have representative example inputs                     ║
║                                                                    ║
║ Example: torch.jit.trace(model, example_input)                    ║
╚════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════╗
║                    WHEN TO USE SCRIPTING                           ║
╠════════════════════════════════════════════════════════════════════╣
║ ✓ Model has data-dependent control flow                           ║
║ ✓ Contains if/else statements based on tensor values              ║
║ ✓ Has loops with variable iteration counts                        ║
║ ✓ RNNs, transformers, or other sequential models                  ║
║ ✓ Need to guarantee all paths work correctly                      ║
║                                                                    ║
║ Example: torch.jit.script(model)                                  ║
╚════════════════════════════════════════════════════════════════════╝

╔════════════════════════════════════════════════════════════════════╗
║                        COMPARISON TABLE                            ║
╠═══════════════════╦══════════════════════╦═══════════════════════╣
║ Feature           ║ Tracing              ║ Scripting             ║
╠═══════════════════╬══════════════════════╬═══════════════════════╣
║ Ease of use       ║ Very Easy            ║ Moderate              ║
║ Control flow      ║ Limited              ║ Full support          ║
║ Dynamic loops     ║ No                   ║ Yes                   ║
║ Code analysis     ║ No                   ║ Yes                   ║
║ Type checking     ║ No                   ║ Yes                   ║
║ Flexibility       ║ High                 ║ Moderate              ║
║ Debugging         ║ Harder               ║ Easier                ║
╚═══════════════════╩══════════════════════╩═══════════════════════╝

RECOMMENDATION:
- Start with TRACING for simple models
- Use SCRIPTING when you encounter control flow issues
- Test both if unsure (scripting is safer but may need code changes)
    """)


# ============================================================================
# PART 7: Main Demonstration
# ============================================================================

def main():
    """
    Main demonstration of all TorchScript concepts.
    """
    print("\n" + "="*70)
    print("MODULE 64.02: TORCHSCRIPT BASICS")
    print("="*70)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    # Demonstration 1: Tracing
    traced_model = demonstrate_tracing()
    
    # Demonstration 2: Tracing limitations
    demonstrate_tracing_limitations()
    
    # Demonstration 3: Scripting
    scripted_model = demonstrate_scripting()
    
    # Demonstration 4: Scripting with loops
    demonstrate_scripting_loops()
    
    # Demonstration 5: Loading TorchScript models
    demonstrate_loading_torchscript()
    
    # Demonstration 6: Performance comparison
    benchmark_torchscript_performance()
    
    # Print decision guide
    print_decision_guide()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. TorchScript creates Python-independent models:
       ✓ Can run without Python runtime
       ✓ Deploy to C++, mobile, embedded systems
       ✓ Self-contained serialization
       
    2. Two methods for creating TorchScript:
       ✓ TRACING: Record operations on example inputs
       ✓ SCRIPTING: Compile Python source code
       
    3. Tracing is easier but has limitations:
       ✓ Cannot handle data-dependent control flow
       ✓ Only captures one execution path
       ✓ Best for simple, feedforward models
       
    4. Scripting handles complex models:
       ✓ Full control flow support (if/else, loops)
       ✓ Type checking at compile time
       ✓ Better for RNNs, transformers
       
    5. Performance benefits:
       ✓ Reduced Python overhead
       ✓ Optimized graph execution
       ✓ Operation fusion
       
    6. Production advantages:
       ✓ No dependency on original model code
       ✓ Cross-language deployment
       ✓ Consistent behavior across platforms
    """)
    
    print("\n✅ Module 64.02 completed successfully!")
    print("Next: Module 64.03 - ONNX Export")


if __name__ == "__main__":
    main()
