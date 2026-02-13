"""
Module 64.03: ONNX Export
=========================

This module covers exporting PyTorch models to ONNX (Open Neural Network Exchange) format.
ONNX enables model interoperability across different frameworks and deployment platforms.

Learning Objectives:
-------------------
1. Understand ONNX format and its benefits
2. Export PyTorch models to ONNX
3. Verify ONNX model correctness
4. Run inference with ONNX Runtime
5. Optimize ONNX models for production

Time Estimate: 60 minutes
Difficulty: Beginner
Prerequisites: Module 64.01, 64.02
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import time
from typing import List, Dict


# ============================================================================
# PART 1: Sample Models for ONNX Export
# ============================================================================

class ConvNet(nn.Module):
    """
    A simple convolutional network for image classification.
    Demonstrates ONNX export for CNN architectures.
    """
    
    def __init__(self, num_classes: int = 10):
        """
        Initialize CNN with standard layers.
        
        Args:
            num_classes: Number of output classes
        """
        super(ConvNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))  # -> (batch, 32, 14, 14)
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))  # -> (batch, 64, 7, 7)
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# ============================================================================
# PART 2: Basic ONNX Export
# ============================================================================

def export_to_onnx_basic(model: nn.Module, example_input: torch.Tensor,
                        export_path: str, opset_version: int = 11):
    """
    Basic ONNX export with essential parameters.
    
    ONNX (Open Neural Network Exchange) is an open format for representing
    deep learning models. It allows you to:
    - Move models between frameworks (PyTorch -> TensorFlow, etc.)
    - Deploy on optimized inference engines (ONNX Runtime, TensorRT)
    - Run on various platforms (cloud, edge, mobile)
    
    Args:
        model: PyTorch model to export
        example_input: Example input tensor (defines input shape)
        export_path: Where to save the .onnx file
        opset_version: ONNX operator set version (higher = more features)
        
    Key Parameters Explained:
    - example_input: Used to trace the model (like torch.jit.trace)
    - opset_version: ONNX version (11 is widely supported, 13+ has new ops)
    - input_names/output_names: Descriptive names for model inputs/outputs
    - dynamic_axes: Allow variable-sized inputs (batch size, sequence length)
    """
    print(f"\n{'='*70}")
    print("BASIC ONNX EXPORT")
    print(f"{'='*70}")
    
    # Set model to evaluation mode (important!)
    model.eval()
    
    print("\n1. Model Information")
    print(f"   Model type: {type(model).__name__}")
    print(f"   Example input shape: {example_input.shape}")
    
    # Export to ONNX
    print("\n2. Exporting to ONNX...")
    torch.onnx.export(
        model,                      # Model to export
        example_input,              # Example input for tracing
        export_path,                # Output file path
        export_params=True,         # Store trained weights
        opset_version=opset_version, # ONNX version
        do_constant_folding=True,   # Optimize by folding constants
        input_names=['input'],      # Model input names
        output_names=['output'],    # Model output names
        # Dynamic axes allow variable batch size
        dynamic_axes={
            'input': {0: 'batch_size'},   # First dim is dynamic
            'output': {0: 'batch_size'}
        }
    )
    
    file_size = Path(export_path).stat().st_size / 1024  # KB
    print(f"   ✓ Export successful!")
    print(f"   File: {export_path}")
    print(f"   Size: {file_size:.2f} KB")
    print(f"   Opset version: {opset_version}")


def verify_onnx_model(onnx_path: str):
    """
    Verify that the exported ONNX model is valid.
    
    This performs several checks:
    1. File can be loaded
    2. Model graph is valid
    3. All operators are supported
    4. No structural errors
    
    Always verify your ONNX exports before deployment!
    
    Args:
        onnx_path: Path to the ONNX model file
    """
    print(f"\n{'='*70}")
    print("ONNX MODEL VERIFICATION")
    print(f"{'='*70}")
    
    # Load ONNX model
    print("\n1. Loading ONNX model...")
    onnx_model = onnx.load(onnx_path)
    print("   ✓ Model loaded successfully")
    
    # Check model validity
    print("\n2. Checking model validity...")
    try:
        onnx.checker.check_model(onnx_model)
        print("   ✓ Model is valid!")
    except Exception as e:
        print(f"   ✗ Model validation failed: {e}")
        return
    
    # Print model information
    print("\n3. Model Information:")
    graph = onnx_model.graph
    
    # Input information
    print(f"\n   Inputs:")
    for input_tensor in graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' 
                for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"     - {input_tensor.name}: {shape}")
    
    # Output information
    print(f"\n   Outputs:")
    for output_tensor in graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic'
                for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"     - {output_tensor.name}: {shape}")
    
    # Operator count
    op_types = {}
    for node in graph.node:
        op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
    
    print(f"\n   Operations ({len(graph.node)} total):")
    for op_type, count in sorted(op_types.items()):
        print(f"     - {op_type}: {count}")


# ============================================================================
# PART 3: ONNX Runtime Inference
# ============================================================================

def run_onnx_inference(onnx_path: str, input_data: np.ndarray) -> np.ndarray:
    """
    Run inference using ONNX Runtime.
    
    ONNX Runtime is a high-performance inference engine that:
    - Executes ONNX models efficiently
    - Supports multiple hardware backends (CPU, GPU, TensorRT)
    - Often faster than PyTorch for inference
    - Used in production by many companies
    
    Args:
        onnx_path: Path to ONNX model
        input_data: Input as numpy array
        
    Returns:
        Model output as numpy array
    """
    print(f"\n{'='*70}")
    print("ONNX RUNTIME INFERENCE")
    print(f"{'='*70}")
    
    # Create inference session
    print("\n1. Creating ONNX Runtime session...")
    # ExecutionProviders determine what hardware to use
    # Options: 'CPUExecutionProvider', 'CUDAExecutionProvider', etc.
    session = ort.InferenceSession(
        onnx_path,
        providers=['CPUExecutionProvider']
    )
    print(f"   ✓ Session created")
    print(f"   Provider: {session.get_providers()}")
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"\n2. Model I/O Information:")
    print(f"   Input name: {input_name}")
    print(f"   Input shape: {session.get_inputs()[0].shape}")
    print(f"   Output name: {output_name}")
    print(f"   Output shape: {session.get_outputs()[0].shape}")
    
    # Run inference
    print(f"\n3. Running inference...")
    print(f"   Input data shape: {input_data.shape}")
    
    outputs = session.run(
        [output_name],              # Output names to compute
        {input_name: input_data}    # Input dictionary
    )
    
    output = outputs[0]
    print(f"   Output shape: {output.shape}")
    print(f"   Output sample: {output[0, :5]}")
    print("   ✓ Inference successful")
    
    return output


def compare_pytorch_onnx_outputs(model: nn.Module, onnx_path: str,
                                input_tensor: torch.Tensor):
    """
    Compare outputs from PyTorch and ONNX Runtime to verify correctness.
    
    This is a critical validation step: ensure that the ONNX model
    produces the same results as the original PyTorch model.
    Small numerical differences (< 1e-5) are acceptable due to
    different implementations of operations.
    
    Args:
        model: Original PyTorch model
        onnx_path: Path to exported ONNX model
        input_tensor: Test input
    """
    print(f"\n{'='*70}")
    print("COMPARING PYTORCH vs ONNX OUTPUTS")
    print(f"{'='*70}")
    
    model.eval()
    
    # PyTorch inference
    print("\n1. Running PyTorch inference...")
    with torch.no_grad():
        pytorch_output = model(input_tensor)
    print(f"   Output shape: {pytorch_output.shape}")
    print(f"   Output sample: {pytorch_output[0, :5]}")
    
    # ONNX inference
    print("\n2. Running ONNX Runtime inference...")
    input_numpy = input_tensor.numpy()
    onnx_output = run_onnx_inference(onnx_path, input_numpy)
    
    # Compare outputs
    print("\n3. Comparing outputs...")
    pytorch_np = pytorch_output.numpy()
    
    # Compute differences
    abs_diff = np.abs(pytorch_np - onnx_output)
    max_diff = np.max(abs_diff)
    mean_diff = np.mean(abs_diff)
    
    print(f"   Max absolute difference: {max_diff:.2e}")
    print(f"   Mean absolute difference: {mean_diff:.2e}")
    
    # Check if outputs match (allowing for numerical precision)
    tolerance = 1e-5
    if max_diff < tolerance:
        print(f"   ✓ Outputs match within tolerance ({tolerance})")
        print("   ✓ ONNX export is correct!")
    else:
        print(f"   ⚠️  Outputs differ by more than tolerance ({tolerance})")
        print("   Check model export settings or operations used")


# ============================================================================
# PART 4: Advanced Export Options
# ============================================================================

def export_with_dynamic_axes(model: nn.Module, export_path: str):
    """
    Export model with multiple dynamic axes.
    
    Dynamic axes allow the model to accept variable-sized inputs,
    which is essential for:
    - Variable batch sizes
    - Variable sequence lengths (NLP, time series)
    - Variable image sizes (some vision models)
    
    Without dynamic axes, the model accepts only fixed-size inputs.
    """
    print(f"\n{'='*70}")
    print("EXPORT WITH DYNAMIC AXES")
    print(f"{'='*70}")
    
    model.eval()
    
    # Create example input
    batch_size = 4
    channels = 1
    height = 28
    width = 28
    example_input = torch.randn(batch_size, channels, height, width)
    
    print("\n1. Exporting with multiple dynamic axes...")
    print(f"   Example input shape: {example_input.shape}")
    
    torch.onnx.export(
        model,
        example_input,
        export_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        # Define which dimensions are dynamic
        dynamic_axes={
            'input': {
                0: 'batch_size',      # Batch dimension is dynamic
                2: 'height',          # Height can vary (for some models)
                3: 'width'            # Width can vary
            },
            'output': {
                0: 'batch_size'       # Output batch matches input
            }
        }
    )
    print("   ✓ Export complete with dynamic axes")
    
    # Test with different batch sizes
    print("\n2. Testing with different batch sizes...")
    session = ort.InferenceSession(export_path, providers=['CPUExecutionProvider'])
    
    for test_batch_size in [1, 4, 8]:
        test_input = torch.randn(test_batch_size, channels, height, width).numpy()
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: test_input})[0]
        print(f"   Batch size {test_batch_size}: Input {test_input.shape} → Output {output.shape} ✓")


def export_with_metadata(model: nn.Module, export_path: str, metadata: Dict):
    """
    Export model with custom metadata.
    
    Metadata helps track:
    - Model version
    - Training date
    - Accuracy metrics
    - Author information
    - License
    
    This is useful for model management and deployment tracking.
    """
    print(f"\n{'='*70}")
    print("EXPORT WITH METADATA")
    print(f"{'='*70}")
    
    model.eval()
    example_input = torch.randn(1, 1, 28, 28)
    
    print("\n1. Model metadata:")
    for key, value in metadata.items():
        print(f"   {key}: {value}")
    
    print("\n2. Exporting with metadata...")
    torch.onnx.export(
        model,
        example_input,
        export_path,
        export_params=True,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    
    # Add metadata to ONNX model
    onnx_model = onnx.load(export_path)
    
    # Add metadata as model properties
    for key, value in metadata.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = str(value)
    
    # Save with metadata
    onnx.save(onnx_model, export_path)
    print("   ✓ Model saved with metadata")
    
    # Verify metadata
    print("\n3. Verifying metadata...")
    loaded_model = onnx.load(export_path)
    for prop in loaded_model.metadata_props:
        print(f"   {prop.key}: {prop.value}")


# ============================================================================
# PART 5: Performance Benchmarking
# ============================================================================

def benchmark_onnx_vs_pytorch(model: nn.Module, onnx_path: str,
                             input_shape: tuple, num_iterations: int = 1000):
    """
    Benchmark inference speed: PyTorch vs ONNX Runtime.
    
    ONNX Runtime is often faster because:
    - Optimized operation implementations
    - Graph-level optimizations
    - Reduced Python overhead
    - Hardware-specific optimizations
    """
    print(f"\n{'='*70}")
    print("PERFORMANCE BENCHMARK: PYTORCH vs ONNX")
    print(f"{'='*70}")
    
    model.eval()
    
    # Prepare inputs
    torch_input = torch.randn(*input_shape)
    numpy_input = torch_input.numpy()
    
    # Create ONNX session
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Warmup (JIT compilation, cache warming)
    print(f"\n1. Warming up (100 iterations)...")
    with torch.no_grad():
        for _ in range(100):
            _ = model(torch_input)
            _ = session.run(None, {input_name: numpy_input})
    print("   ✓ Warmup complete")
    
    # Benchmark PyTorch
    print(f"\n2. Benchmarking PyTorch ({num_iterations} iterations)...")
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(torch_input)
    pytorch_time = time.time() - start
    pytorch_throughput = num_iterations / pytorch_time
    print(f"   Time: {pytorch_time:.4f}s")
    print(f"   Throughput: {pytorch_throughput:.2f} inferences/second")
    
    # Benchmark ONNX Runtime
    print(f"\n3. Benchmarking ONNX Runtime ({num_iterations} iterations)...")
    start = time.time()
    for _ in range(num_iterations):
        _ = session.run(None, {input_name: numpy_input})
    onnx_time = time.time() - start
    onnx_throughput = num_iterations / onnx_time
    print(f"   Time: {onnx_time:.4f}s")
    print(f"   Throughput: {onnx_throughput:.2f} inferences/second")
    
    # Compare
    speedup = pytorch_time / onnx_time
    print(f"\n4. Comparison:")
    print(f"   Speedup: {speedup:.2f}x")
    if speedup > 1:
        print(f"   ✓ ONNX Runtime is {speedup:.2f}x faster")
    else:
        print(f"   PyTorch is {1/speedup:.2f}x faster")
    print(f"   (Results vary by model, hardware, and optimization settings)")


# ============================================================================
# PART 6: Main Demonstration
# ============================================================================

def main():
    """
    Main demonstration of ONNX export functionality.
    """
    print("\n" + "="*70)
    print("MODULE 64.03: ONNX EXPORT")
    print("="*70)
    
    # Create directories
    Path("models").mkdir(exist_ok=True)
    
    # Create and initialize model
    print("\n1. Creating model...")
    model = ConvNet(num_classes=10)
    example_input = torch.randn(1, 1, 28, 28)
    print("   ✓ Model created")
    
    # Basic export
    onnx_basic_path = "models/convnet_basic.onnx"
    export_to_onnx_basic(model, example_input, onnx_basic_path)
    
    # Verify model
    verify_onnx_model(onnx_basic_path)
    
    # ONNX Runtime inference
    input_numpy = example_input.numpy()
    output = run_onnx_inference(onnx_basic_path, input_numpy)
    
    # Compare PyTorch vs ONNX
    compare_pytorch_onnx_outputs(model, onnx_basic_path, example_input)
    
    # Dynamic axes
    onnx_dynamic_path = "models/convnet_dynamic.onnx"
    export_with_dynamic_axes(model, onnx_dynamic_path)
    
    # Export with metadata
    onnx_metadata_path = "models/convnet_metadata.onnx"
    metadata = {
        'model_name': 'ConvNet',
        'version': '1.0.0',
        'author': 'Deep Learning Course',
        'date': '2024-01-15',
        'accuracy': '98.5%',
        'framework': 'PyTorch'
    }
    export_with_metadata(model, onnx_metadata_path, metadata)
    
    # Performance benchmark
    benchmark_onnx_vs_pytorch(
        model,
        onnx_basic_path,
        input_shape=(1, 1, 28, 28),
        num_iterations=1000
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. ONNX enables cross-framework model deployment:
       ✓ Framework-agnostic format
       ✓ Deploy PyTorch models anywhere
       ✓ Interoperability (PyTorch ↔ TensorFlow ↔ etc.)
       
    2. Export process is straightforward:
       ✓ torch.onnx.export() with example input
       ✓ Specify input/output names
       ✓ Define dynamic axes for flexibility
       
    3. Always verify exported models:
       ✓ Use onnx.checker to validate
       ✓ Compare outputs with original model
       ✓ Test with different input sizes
       
    4. ONNX Runtime for production inference:
       ✓ Often faster than PyTorch
       ✓ Multiple hardware backends
       ✓ Wide deployment support
       
    5. Advanced features:
       ✓ Dynamic axes for variable inputs
       ✓ Metadata for model tracking
       ✓ Multiple opset versions
       
    6. Production benefits:
       ✓ Standardized model format
       ✓ Optimized inference engines
       ✓ Cross-platform deployment
    """)
    
    print("\n✅ Module 64.03 completed successfully!")
    print("Next: Module 64.04 - Flask API Deployment")


if __name__ == "__main__":
    main()
