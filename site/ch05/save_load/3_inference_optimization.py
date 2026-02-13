"""
Inference Optimization Techniques

This module demonstrates:
- Batch processing and dynamic batching
- Model optimization (pruning, distillation)
- TorchScript compilation
- CUDA optimization
- Profiling and benchmarking
- Caching strategies
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import time
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import threading
from dataclasses import dataclass


class OptimizedModel(nn.Module):
    """Example model for optimization demonstrations"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# =============================================================================
# 1. BATCH PROCESSING
# =============================================================================

class DynamicBatcher:
    """
    Dynamic batching - Accumulate requests and process in batches
    
    Benefits:
    - Better GPU utilization
    - Reduced overhead
    - Higher throughput
    """
    
    def __init__(
        self,
        model: nn.Module,
        max_batch_size: int = 32,
        timeout_ms: int = 100
    ):
        self.model = model
        self.max_batch_size = max_batch_size
        self.timeout_ms = timeout_ms
        self.queue = deque()
        self.lock = threading.Lock()
        
    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """Add request to queue and wait for batched processing"""
        # In production, this would use async/await or threading
        # Simplified for demonstration
        
        with self.lock:
            self.queue.append(input_data)
        
        # Wait for batch or timeout
        if len(self.queue) >= self.max_batch_size:
            return self._process_batch()
        
        return self._single_predict(input_data)
    
    def _process_batch(self) -> torch.Tensor:
        """Process accumulated batch"""
        with self.lock:
            batch = list(self.queue)
            self.queue.clear()
        
        if not batch:
            return None
        
        # Stack into single batch
        batch_tensor = torch.stack(batch)
        
        # Process batch
        with torch.no_grad():
            outputs = self.model(batch_tensor)
        
        return outputs
    
    def _single_predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """Fallback for single prediction"""
        with torch.no_grad():
            return self.model(input_data.unsqueeze(0))


def benchmark_batching(model: nn.Module, input_size: Tuple, num_samples: int = 100):
    """Compare single vs batched inference"""
    print("\n=== Batching Benchmark ===")
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Generate test data
    test_data = [torch.randn(*input_size).to(device) for _ in range(num_samples)]
    
    # Single inference
    start_time = time.time()
    with torch.no_grad():
        for data in test_data:
            _ = model(data.unsqueeze(0))
    single_time = time.time() - start_time
    
    # Batched inference (batch_size=16)
    batch_size = 16
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            batch = torch.stack(test_data[i:i+batch_size])
            _ = model(batch)
    batched_time = time.time() - start_time
    
    print(f"Single inference time: {single_time:.3f}s")
    print(f"Batched inference time: {batched_time:.3f}s")
    print(f"Speedup: {single_time/batched_time:.2f}x")
    print(f"Throughput (single): {num_samples/single_time:.1f} samples/sec")
    print(f"Throughput (batched): {num_samples/batched_time:.1f} samples/sec")


# =============================================================================
# 2. MODEL PRUNING
# =============================================================================

def prune_model(model: nn.Module, amount: float = 0.3):
    """
    Prune model weights - Remove least important weights
    
    Args:
        model: PyTorch model
        amount: Fraction of weights to prune (0-1)
    
    Returns:
        Pruned model
    """
    print(f"\n=== Model Pruning (amount={amount}) ===")
    
    # Global pruning across all conv and linear layers
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    
    # Apply global unstructured pruning
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Count pruned parameters
    total_params = 0
    pruned_params = 0
    for module, param_name in parameters_to_prune:
        param = getattr(module, param_name)
        total_params += param.nelement()
        pruned_params += (param == 0).sum().item()
    
    print(f"Total parameters: {total_params:,}")
    print(f"Pruned parameters: {pruned_params:,} ({pruned_params/total_params*100:.1f}%)")
    
    # Make pruning permanent
    for module, param_name in parameters_to_prune:
        prune.remove(module, param_name)
    
    return model


def structured_pruning(model: nn.Module, amount: float = 0.3):
    """
    Structured pruning - Remove entire channels/neurons
    
    Advantages:
    - Actual speedup without special hardware
    - Reduces model size more effectively
    """
    print(f"\n=== Structured Pruning (amount={amount}) ===")
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # Prune entire output channels
            prune.ln_structured(
                module,
                name='weight',
                amount=amount,
                n=2,
                dim=0  # Prune output channels
            )
            prune.remove(module, 'weight')
    
    return model


# =============================================================================
# 3. TORCHSCRIPT COMPILATION
# =============================================================================

def torchscript_optimization(model: nn.Module, example_input: torch.Tensor):
    """
    TorchScript compilation for optimization
    
    Benefits:
    - Removes Python overhead
    - Enables graph optimizations
    - Better for production deployment
    """
    print("\n=== TorchScript Optimization ===")
    
    model.eval()
    
    # Method 1: Tracing
    print("1. Tracing...")
    traced_model = torch.jit.trace(model, example_input)
    
    # Method 2: Scripting (better for control flow)
    print("2. Scripting...")
    scripted_model = torch.jit.script(model)
    
    # Optimize
    print("3. Applying optimizations...")
    optimized_traced = torch.jit.optimize_for_inference(traced_model)
    
    # Benchmark
    num_runs = 100
    
    # Original model
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(example_input)
    original_time = time.time() - start
    
    # TorchScript model
    start = time.time()
    for _ in range(num_runs):
        _ = optimized_traced(example_input)
    torchscript_time = time.time() - start
    
    print(f"\nOriginal model: {original_time*1000:.2f}ms")
    print(f"TorchScript model: {torchscript_time*1000:.2f}ms")
    print(f"Speedup: {original_time/torchscript_time:.2f}x")
    
    return optimized_traced


# =============================================================================
# 4. CUDA OPTIMIZATION
# =============================================================================

def cuda_optimization_tips():
    """CUDA optimization best practices"""
    print("""
    === CUDA Optimization Tips ===
    
    1. TENSOR CORES (Mixed Precision):
       - Use torch.cuda.amp for automatic mixed precision
       - FP16 for compute, FP32 for stability
       - 2-3x speedup on modern GPUs
    
    2. MEMORY MANAGEMENT:
       - Use pin_memory=True for DataLoader
       - Preallocate tensors when possible
       - Clear cache: torch.cuda.empty_cache()
    
    3. ASYNC OPERATIONS:
       - Use non_blocking=True for H2D/D2H transfers
       - Overlap compute and data transfer
       - Use CUDA streams for parallelism
    
    4. KERNEL FUSION:
       - Fuse operations to reduce kernel launches
       - Use JIT compilation
       - Consider custom CUDA kernels for bottlenecks
    
    5. BATCH SIZE:
       - Use largest batch size that fits in memory
       - Powers of 2 often optimal
       - Monitor GPU utilization
    """)


class MixedPrecisionInference:
    """Automatic Mixed Precision inference wrapper"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler()
    
    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """Run inference with mixed precision"""
        input_data = input_data.to(self.device)
        
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = self.model(input_data)
        
        return output


def benchmark_mixed_precision(model: nn.Module, input_size: Tuple):
    """Benchmark FP32 vs FP16 inference"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping mixed precision benchmark")
        return
    
    print("\n=== Mixed Precision Benchmark ===")
    
    device = torch.device('cuda')
    model = model.to(device)
    test_input = torch.randn(*input_size).to(device)
    
    # FP32
    model.eval()
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(test_input)
    torch.cuda.synchronize()
    fp32_time = time.time() - start
    
    # FP16
    start = time.time()
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            for _ in range(100):
                _ = model(test_input)
    torch.cuda.synchronize()
    fp16_time = time.time() - start
    
    print(f"FP32 inference: {fp32_time*1000:.2f}ms")
    print(f"FP16 inference: {fp16_time*1000:.2f}ms")
    print(f"Speedup: {fp32_time/fp16_time:.2f}x")


# =============================================================================
# 5. PROFILING & BENCHMARKING
# =============================================================================

def profile_model(model: nn.Module, input_data: torch.Tensor):
    """
    Profile model to identify bottlenecks
    """
    print("\n=== Model Profiling ===")
    
    model.eval()
    
    # PyTorch profiler
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            _ = model(input_data)
    
    # Print results
    print("\nTop 10 operations by CPU time:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    if torch.cuda.is_available():
        print("\nTop 10 operations by CUDA time:")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


def benchmark_comprehensive(
    model: nn.Module,
    input_size: Tuple,
    batch_sizes: List[int] = [1, 8, 16, 32],
    num_runs: int = 100
):
    """Comprehensive benchmarking across different configurations"""
    print("\n=== Comprehensive Benchmark ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    
    results = {}
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, *input_size).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(test_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed = time.time() - start
        
        results[batch_size] = {
            'time': elapsed,
            'throughput': (batch_size * num_runs) / elapsed,
            'latency': (elapsed / num_runs) * 1000  # ms
        }
    
    # Print results
    print(f"\n{'Batch Size':<12} {'Latency (ms)':<15} {'Throughput (samples/s)':<25}")
    print("-" * 52)
    for batch_size, metrics in results.items():
        print(f"{batch_size:<12} {metrics['latency']:<15.2f} {metrics['throughput']:<25.1f}")
    
    return results


# =============================================================================
# 6. CACHING STRATEGIES
# =============================================================================

class KVCache:
    """
    Key-Value Cache for autoregressive models (e.g., transformers)
    
    Reduces computation by caching previous key-value pairs
    """
    
    def __init__(self, max_length: int = 1024):
        self.max_length = max_length
        self.cache = {}
    
    def get(self, layer_idx: int):
        """Get cached KV for a layer"""
        return self.cache.get(layer_idx)
    
    def update(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor):
        """Update cache for a layer"""
        if layer_idx not in self.cache:
            self.cache[layer_idx] = {'key': key, 'value': value}
        else:
            # Concatenate with previous cache
            self.cache[layer_idx]['key'] = torch.cat(
                [self.cache[layer_idx]['key'], key], dim=-2
            )
            self.cache[layer_idx]['value'] = torch.cat(
                [self.cache[layer_idx]['value'], value], dim=-2
            )
            
            # Truncate if exceeds max length
            if self.cache[layer_idx]['key'].size(-2) > self.max_length:
                self.cache[layer_idx]['key'] = self.cache[layer_idx]['key'][..., -self.max_length:, :]
                self.cache[layer_idx]['value'] = self.cache[layer_idx]['value'][..., -self.max_length:, :]
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()


# =============================================================================
# MAIN DEMO
# =============================================================================

def demo_all_optimizations():
    """Demonstrate all optimization techniques"""
    print("="*70)
    print("INFERENCE OPTIMIZATION DEMO")
    print("="*70)
    
    # Create model
    model = OptimizedModel()
    input_size = (3, 32, 32)
    example_input = torch.randn(1, *input_size)
    
    # 1. Batching
    benchmark_batching(model, input_size, num_samples=100)
    
    # 2. Pruning
    pruned_model = prune_model(model, amount=0.3)
    
    # 3. TorchScript
    torchscript_model = torchscript_optimization(model, example_input)
    
    # 4. CUDA tips
    cuda_optimization_tips()
    
    # 5. Profiling
    profile_model(model, example_input)
    
    # 6. Comprehensive benchmark
    benchmark_comprehensive(model, input_size)
    
    print("\n" + "="*70)
    print("Demo completed!")


if __name__ == "__main__":
    demo_all_optimizations()
