"""
Module 64.10: Inference Optimization
====================================

This module covers advanced techniques for optimizing model inference performance.
Students will learn quantization, pruning, mixed precision, and other optimization
methods to achieve production-level performance.

Learning Objectives:
-------------------
1. Apply dynamic and static quantization
2. Implement mixed precision inference
3. Optimize CUDA operations
4. Benchmark optimization techniques
5. Choose appropriate optimizations for deployment

Time Estimate: 120 minutes
Difficulty: Advanced
Prerequisites: Module 64.01-64.03
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from typing import Dict, Tuple, Any
from pathlib import Path


# ============================================================================
# PART 1: Reference Model for Optimization
# ============================================================================

class OptimizationTestModel(nn.Module):
    """
    Medium-sized CNN for optimization testing.
    
    This model is large enough to show meaningful speedups
    from optimization techniques.
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        # Conv block 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        # Conv block 3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 7x7 -> 3x3
        # Flatten and FC
        x = x.view(-1, 256 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================================================
# PART 2: Dynamic Quantization
# ============================================================================

def apply_dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Apply dynamic quantization to model.
    
    Dynamic quantization:
    - Quantizes weights ahead of time (int8)
    - Quantizes activations dynamically during inference
    - Good for models with varied input sizes (NLP, RNNs)
    - Reduces model size ~4x
    - Can improve inference speed 2-4x on CPU
    
    Works best on:
    - Linear layers
    - LSTM/GRU layers
    - Embedding layers
    
    Args:
        model: PyTorch model to quantize
        
    Returns:
        Quantized model
    """
    print("\n" + "="*70)
    print("DYNAMIC QUANTIZATION")
    print("="*70)
    
    # Get model size before quantization
    size_before = get_model_size_mb(model)
    print(f"\nOriginal model size: {size_before:.2f} MB")
    
    # Apply dynamic quantization
    print("Applying dynamic quantization to Linear layers...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Quantize these layer types
        dtype=torch.qint8  # Use 8-bit integers
    )
    
    # Get model size after quantization
    size_after = get_model_size_mb(quantized_model)
    print(f"Quantized model size: {size_after:.2f} MB")
    print(f"Size reduction: {(1 - size_after/size_before) * 100:.1f}%")
    
    return quantized_model


# ============================================================================
# PART 3: Static Quantization (Quantization-Aware)
# ============================================================================

def apply_static_quantization(model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
    """
    Apply static quantization with calibration.
    
    Static quantization:
    - Quantizes weights and activations (both int8)
    - Requires calibration data to determine scale/zero-point
    - More aggressive than dynamic quantization
    - Better speedups (2-4x) but requires more setup
    - Best for fixed input sizes (vision models)
    
    Process:
    1. Prepare model (insert observers)
    2. Calibrate (run representative data through model)
    3. Convert to quantized version
    
    Args:
        model: Model to quantize
        calibration_data: Representative data for calibration
        
    Returns:
        Quantized model
    """
    print("\n" + "="*70)
    print("STATIC QUANTIZATION")
    print("="*70)
    
    model.eval()
    
    # Step 1: Specify quantization configuration
    print("\n1. Setting quantization config...")
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Step 2: Prepare model (insert observers)
    print("2. Preparing model (inserting observers)...")
    model_prepared = torch.quantization.prepare(model)
    
    # Step 3: Calibrate with representative data
    print(f"3. Calibrating with {calibration_data.shape[0]} samples...")
    with torch.no_grad():
        for i in range(min(100, calibration_data.shape[0])):
            _ = model_prepared(calibration_data[i:i+1])
    
    # Step 4: Convert to quantized version
    print("4. Converting to quantized model...")
    quantized_model = torch.quantization.convert(model_prepared)
    
    size_before = get_model_size_mb(model)
    size_after = get_model_size_mb(quantized_model)
    
    print(f"\nOriginal model size: {size_before:.2f} MB")
    print(f"Quantized model size: {size_after:.2f} MB")
    print(f"Size reduction: {(1 - size_after/size_before) * 100:.1f}%")
    
    return quantized_model


# ============================================================================
# PART 4: Mixed Precision (FP16) Inference
# ============================================================================

def demonstrate_mixed_precision():
    """
    Demonstrate mixed precision (FP16) inference.
    
    Mixed precision:
    - Use FP16 (half precision) instead of FP32 (full precision)
    - Reduces memory usage by 2x
    - Can speed up inference on modern GPUs (Tensor Cores)
    - Minimal accuracy loss for most models
    
    Automatic Mixed Precision (AMP):
    - PyTorch automatically manages precision
    - Keeps some operations in FP32 for numerical stability
    - Easy to use with torch.cuda.amp
    """
    print("\n" + "="*70)
    print("MIXED PRECISION (FP16) INFERENCE")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - mixed precision works best on GPU")
        return
    
    model = OptimizationTestModel()
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    
    # Create test input
    test_input = torch.randn(32, 1, 28, 28, device=device)
    
    # FP32 inference
    print("\n1. FP32 (Full Precision) Inference")
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(test_input)
    fp32_time = time.time() - start
    print(f"   Time: {fp32_time:.4f}s")
    
    # FP16 inference with AMP
    print("\n2. FP16 (Half Precision) Inference with AMP")
    start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast():  # Enable automatic mixed precision
            for _ in range(100):
                _ = model(test_input)
    fp16_time = time.time() - start
    print(f"   Time: {fp16_time:.4f}s")
    
    speedup = fp32_time / fp16_time
    print(f"\n3. Speedup: {speedup:.2f}x")
    
    # Memory comparison
    model_fp32 = model
    model_fp16 = model.half()  # Convert to FP16
    
    size_fp32 = get_model_size_mb(model_fp32)
    size_fp16 = get_model_size_mb(model_fp16)
    
    print(f"\n4. Memory Usage:")
    print(f"   FP32 model: {size_fp32:.2f} MB")
    print(f"   FP16 model: {size_fp16:.2f} MB")
    print(f"   Memory reduction: {(1 - size_fp16/size_fp32) * 100:.1f}%")


# ============================================================================
# PART 5: CUDA Optimizations
# ============================================================================

def optimize_for_cuda(model: nn.Module):
    """
    Apply CUDA-specific optimizations.
    
    Optimizations:
    1. cudnn.benchmark - Auto-tune convolution algorithms
    2. TorchScript - JIT compilation for GPU
    3. Batch size tuning
    4. Tensor Cores (mixed precision)
    """
    print("\n" + "="*70)
    print("CUDA OPTIMIZATIONS")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available")
        return
    
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    
    # Optimization 1: Enable cudnn benchmarking
    print("\n1. Enabling cuDNN auto-tuner...")
    print("   (Automatically selects fastest convolution algorithm)")
    torch.backends.cudnn.benchmark = True
    
    # Optimization 2: JIT compile with TorchScript
    print("\n2. Compiling with TorchScript...")
    example_input = torch.randn(1, 1, 28, 28, device=device)
    scripted_model = torch.jit.trace(model, example_input)
    
    # Benchmark
    test_input = torch.randn(32, 1, 28, 28, device=device)
    
    print("\n3. Benchmarking (100 iterations)...")
    
    # Original model
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(test_input)
    torch.cuda.synchronize()
    original_time = time.time() - start
    
    # Optimized model
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = scripted_model(test_input)
    torch.cuda.synchronize()
    optimized_time = time.time() - start
    
    print(f"   Original: {original_time:.4f}s")
    print(f"   Optimized: {optimized_time:.4f}s")
    print(f"   Speedup: {original_time / optimized_time:.2f}x")


# ============================================================================
# PART 6: Comprehensive Benchmarking
# ============================================================================

def benchmark_all_optimizations():
    """
    Comprehensive benchmark of all optimization techniques.
    
    Compares:
    - Baseline (FP32)
    - Dynamic quantization
    - Mixed precision (FP16)
    - TorchScript
    - Combined optimizations
    """
    print("\n" + "="*70)
    print("COMPREHENSIVE OPTIMIZATION BENCHMARK")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Create baseline model
    model_baseline = OptimizationTestModel()
    model_baseline.to(device)
    model_baseline.eval()
    
    # Test data
    batch_size = 32
    test_input = torch.randn(batch_size, 1, 28, 28, device=device)
    calibration_data = torch.randn(100, 1, 28, 28)
    
    results = {}
    
    # 1. Baseline
    print("\n1. Baseline (FP32)...")
    results['baseline'] = benchmark_model(model_baseline, test_input, "Baseline")
    
    # 2. Dynamic Quantization
    print("\n2. Dynamic Quantization...")
    model_dynamic_quant = apply_dynamic_quantization(OptimizationTestModel())
    # Note: Quantized models work on CPU
    results['dynamic_quant'] = benchmark_model(
        model_dynamic_quant,
        test_input.cpu(),
        "Dynamic Quant"
    )
    
    # 3. Mixed Precision (if CUDA available)
    if torch.cuda.is_available():
        print("\n3. Mixed Precision (FP16)...")
        results['fp16'] = benchmark_model_fp16(model_baseline, test_input, "FP16")
    
    # 4. TorchScript
    print("\n4. TorchScript...")
    model_scripted = torch.jit.trace(model_baseline, test_input[:1])
    results['torchscript'] = benchmark_model(model_scripted, test_input, "TorchScript")
    
    # Print comparison table
    print("\n" + "="*70)
    print("OPTIMIZATION COMPARISON TABLE")
    print("="*70)
    print(f"{'Method':<20} {'Time (ms)':<12} {'Throughput':<15} {'Speedup':<10} {'Size (MB)':<10}")
    print("-" * 70)
    
    baseline_time = results['baseline']['time_ms']
    for name, result in results.items():
        speedup = baseline_time / result['time_ms']
        print(f"{name:<20} {result['time_ms']:<12.2f} {result['throughput']:<15.1f} "
              f"{speedup:<10.2f}x {result['size_mb']:<10.2f}")


def benchmark_model(model: nn.Module, input_tensor: torch.Tensor,
                   name: str, num_iterations: int = 100) -> Dict[str, float]:
    """
    Benchmark a model.
    
    Returns metrics: latency, throughput, model size
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Benchmark
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(input_tensor)
    
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    total_time = time.time() - start
    time_ms = (total_time / num_iterations) * 1000
    throughput = (input_tensor.shape[0] * num_iterations) / total_time
    size_mb = get_model_size_mb(model)
    
    return {
        'time_ms': time_ms,
        'throughput': throughput,
        'size_mb': size_mb
    }


def benchmark_model_fp16(model: nn.Module, input_tensor: torch.Tensor,
                        name: str, num_iterations: int = 100) -> Dict[str, float]:
    """Benchmark model with FP16."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for _ in range(10):
                _ = model(input_tensor)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for _ in range(num_iterations):
                _ = model(input_tensor)
    torch.cuda.synchronize()
    
    total_time = time.time() - start
    time_ms = (total_time / num_iterations) * 1000
    throughput = (input_tensor.shape[0] * num_iterations) / total_time
    size_mb = get_model_size_mb(model.half())
    
    return {
        'time_ms': time_ms,
        'throughput': throughput,
        'size_mb': size_mb
    }


# ============================================================================
# PART 7: Utility Functions
# ============================================================================

def get_model_size_mb(model: nn.Module) -> float:
    """
    Calculate model size in MB.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / (1024 * 1024)
    return size_mb


def print_optimization_guide():
    """Print guide for choosing optimization techniques."""
    print("\n" + "="*70)
    print("OPTIMIZATION SELECTION GUIDE")
    print("="*70)
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                     DEPLOYMENT SCENARIOS                           ║
╠════════════════════════════════════════════════════════════════════╣
║                                                                    ║
║ 1. CLOUD DEPLOYMENT (GPU available)                               ║
║    Best: Mixed Precision (FP16) + TorchScript                     ║
║    - 2-3x speedup on modern GPUs                                  ║
║    - Minimal accuracy loss                                        ║
║    - Easy to implement                                            ║
║                                                                    ║
║ 2. CLOUD DEPLOYMENT (CPU only)                                    ║
║    Best: Dynamic Quantization + TorchScript                       ║
║    - 2-4x speedup on CPU                                          ║
║    - Significant size reduction                                   ║
║    - Good for cost optimization                                   ║
║                                                                    ║
║ 3. EDGE DEPLOYMENT (Mobile, IoT)                                  ║
║    Best: Static Quantization + Model Pruning                      ║
║    - Maximum size reduction (4-10x)                               ║
║    - Lower memory usage                                           ║
║    - Trade-off: some accuracy loss                                ║
║                                                                    ║
║ 4. REAL-TIME INFERENCE (Low latency required)                     ║
║    Best: TorchScript + CUDA optimizations                         ║
║    - Minimize latency overhead                                    ║
║    - Predictable performance                                      ║
║    - Use batch size 1 optimizations                               ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝

OPTIMIZATION TECHNIQUES COMPARISON:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Technique         | Speedup | Size   | Accuracy | Difficulty | Best For
                  |         | Saving | Impact   |            |
─────────────────┼─────────┼────────┼──────────┼────────────┼──────────
TorchScript      | 1.2-2x  | -      | None     | Easy       | All
Mixed Precision  | 1.5-3x  | 50%    | Minimal  | Easy       | GPU
Dynamic Quant    | 2-4x    | 75%    | Minimal  | Easy       | CPU
Static Quant     | 2-4x    | 75%    | Low      | Medium     | Edge
Model Pruning    | 1.5-3x  | 50-90% | Medium   | Hard       | Edge
Knowledge Dist.  | Varies  | Varies | Low      | Hard       | Custom

RECOMMENDATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Start with TorchScript - it's free performance
2. Add Mixed Precision for GPU deployment
3. Use Quantization for CPU/Edge deployment
4. Benchmark on target hardware
5. Monitor accuracy degradation
6. Profile to find bottlenecks
7. Iterate and optimize
    """)


# ============================================================================
# PART 8: Main Demonstration
# ============================================================================

def main():
    """
    Main demonstration of inference optimizations.
    """
    print("\n" + "="*70)
    print("MODULE 64.10: INFERENCE OPTIMIZATION")
    print("="*70)
    
    # Create model
    model = OptimizationTestModel()
    
    # Demonstrate each optimization
    print("\n" + "="*70)
    print("DEMONSTRATING INDIVIDUAL OPTIMIZATIONS")
    print("="*70)
    
    # Dynamic quantization
    model_quant = apply_dynamic_quantization(model)
    
    # Mixed precision (if GPU available)
    if torch.cuda.is_available():
        demonstrate_mixed_precision()
        optimize_for_cuda(model)
    
    # Comprehensive benchmark
    benchmark_all_optimizations()
    
    # Print guide
    print_optimization_guide()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY: KEY TAKEAWAYS")
    print("="*70)
    print("""
    1. Choose optimizations based on deployment target:
       ✓ GPU: Mixed precision + TorchScript
       ✓ CPU: Dynamic quantization + TorchScript  
       ✓ Edge: Static quantization + pruning
       
    2. Quantization reduces model size significantly:
       ✓ 4x size reduction (FP32 → INT8)
       ✓ 2-4x inference speedup on CPU
       ✓ Minimal accuracy loss for most models
       
    3. Mixed precision (FP16) best for modern GPUs:
       ✓ 2x memory reduction
       ✓ 1.5-3x speedup on Tensor Core GPUs
       ✓ Easy to implement with AMP
       
    4. Always benchmark on target hardware:
       ✓ Results vary significantly by model
       ✓ Different hardware has different bottlenecks
       ✓ Measure latency, throughput, and memory
       
    5. Consider accuracy-performance tradeoffs:
       ✓ Some optimizations reduce accuracy
       ✓ Test on validation set
       ✓ A/B test in production
    """)
    
    print("\n✅ Module 64.10 completed successfully!")
    print("Next: Module 64.11 - Monitoring and Logging")


if __name__ == "__main__":
    main()
