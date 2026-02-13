"""Tutorial 23: GPU Operations - CUDA and device management"""
import torch

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Checking CUDA Availability")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU")
    
    header("2. Device Selection")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Selected device: {device}")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Specific GPU
        print(f"Specific GPU: {device}")
    
    header("3. Moving Tensors to GPU")
    x = torch.randn(3, 3)
    print(f"CPU tensor device: {x.device}")
    if torch.cuda.is_available():
        x_gpu = x.to(device)
        print(f"GPU tensor device: {x_gpu.device}")
        x_gpu_alt = x.cuda()  # Alternative method
        print(f"Using .cuda(): {x_gpu_alt.device}")
    
    header("4. Creating Tensors Directly on GPU")
    if torch.cuda.is_available():
        x_gpu = torch.randn(3, 3, device=device)
        print(f"Created on GPU: {x_gpu.device}")
        print("This is more efficient than creating on CPU then moving!")
    else:
        print("Create with device parameter: torch.randn(3, 3, device=device)")
    
    header("5. Moving Models to GPU")
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    print(f"Model on CPU")
    if torch.cuda.is_available():
        model = model.to(device)
        print(f"Model moved to GPU")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.device}")
    
    header("6. GPU Memory Management")
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Cached: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        x = torch.randn(1000, 1000, device=device)
        print(f"After allocation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        del x
        torch.cuda.empty_cache()
        print(f"After clearing cache: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    header("7. Performance Comparison")
    print("CPU vs GPU performance (if GPU available):")
    size = 1000
    x = torch.randn(size, size)
    y = torch.randn(size, size)
    
    import time
    start = time.time()
    z_cpu = x @ y
    cpu_time = time.time() - start
    print(f"CPU matrix multiplication: {cpu_time:.4f}s")
    
    if torch.cuda.is_available():
        x_gpu = x.to(device)
        y_gpu = y.to(device)
        torch.cuda.synchronize()  # Wait for GPU
        start = time.time()
        z_gpu = x_gpu @ y_gpu
        torch.cuda.synchronize()
        gpu_time = time.time() - start
        print(f"GPU matrix multiplication: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    header("8. Best Practices")
    print("""
    GPU Best Practices:
    
    1. Create tensors directly on GPU when possible
    2. Minimize CPU ↔ GPU data transfers
    3. Use larger batch sizes to saturate GPU
    4. Clear cache when running out of memory
    5. Use torch.cuda.synchronize() for accurate timing
    6. Use mixed precision (see tutorial 24)
    7. Monitor memory with torch.cuda.memory_allocated()
    """)

if __name__ == "__main__":
    main()
