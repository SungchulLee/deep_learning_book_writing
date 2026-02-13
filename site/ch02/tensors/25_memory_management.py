"""Tutorial 25: Memory Management - Optimizing memory usage"""
import torch
import torch.nn as nn

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Understanding Memory Usage")
    x = torch.randn(1000, 1000)
    memory_bytes = x.element_size() * x.nelement()
    memory_mb = memory_bytes / (1024 ** 2)
    print(f"Tensor shape: {x.shape}")
    print(f"Element size: {x.element_size()} bytes")
    print(f"Number of elements: {x.nelement()}")
    print(f"Total memory: {memory_mb:.2f} MB")
    
    header("2. In-place Operations Save Memory")
    x = torch.randn(1000, 1000)
    print(f"Initial memory ID: {id(x)}")
    y = x + 1  # Creates new tensor
    print(f"After x + 1 (new tensor): {id(y)}")
    x.add_(1)  # In-place operation
    print(f"After x.add_(1) (same tensor): {id(x)}")
    print("In-place operations modify tensor without creating a copy!")
    
    header("3. Detaching from Computation Graph")
    x = torch.randn(100, 100, requires_grad=True)
    y = x ** 2
    print(f"y requires_grad: {y.requires_grad}")
    print(f"y has grad_fn: {y.grad_fn is not None}")
    
    z = y.detach()
    print(f"\nAfter detach:")
    print(f"z requires_grad: {z.requires_grad}")
    print(f"z has grad_fn: {z.grad_fn is not None}")
    print("Detach removes from computation graph, saves memory!")
    
    header("4. Using torch.no_grad()")
    model = nn.Linear(1000, 1000)
    x = torch.randn(100, 1000)
    
    print("During inference, use no_grad to save memory:")
    with torch.no_grad():
        output = model(x)
    print(f"Output requires_grad: {output.requires_grad}")
    print("No gradients computed or stored!")
    
    header("5. Gradient Accumulation")
    print("""
    Instead of:
        batch_size = 128  # Might run out of memory!
        
    Use gradient accumulation:
        batch_size = 32
        accumulation_steps = 4  # Effective batch size = 128
        
    for i, (x, y) in enumerate(dataloader):
        output = model(x)
        loss = criterion(output, y) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    """)
    
    header("6. Checkpoint Activations")
    print("""
    For very deep networks, use gradient checkpointing:
    
    from torch.utils.checkpoint import checkpoint
    
    class DeepModel(nn.Module):
        def forward(self, x):
            # Trade computation for memory
            x = checkpoint(self.layer1, x)
            x = checkpoint(self.layer2, x)
            x = checkpoint(self.layer3, x)
            return x
    
    Trades 30% more compute for 10x less memory!
    """)
    
    header("7. Empty Cache (GPU)")
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated()/1e6:.2f} MB")
        x = torch.randn(1000, 1000, device='cuda')
        print(f"After allocation: {torch.cuda.memory_allocated()/1e6:.2f} MB")
        del x
        torch.cuda.empty_cache()
        print(f"After cache clear: {torch.cuda.memory_allocated()/1e6:.2f} MB")
    else:
        print("torch.cuda.empty_cache() - Releases cached GPU memory")
    
    header("8. Memory Profiling")
    print("""
    Use PyTorch's memory profiler:
    
    from torch.profiler import profile, ProfilerActivity
    
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 profile_memory=True) as prof:
        model(input)
    
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage"))
    """)
    
    header("9. Best Practices Summary")
    print("""
    Memory Optimization Tips:
    
    1. Use torch.no_grad() during inference
    2. Call .detach() when you don't need gradients
    3. Use in-place operations (_) when safe
    4. Accumulate gradients for larger effective batch sizes
    5. Use mixed precision training (see tutorial 24)
    6. Delete large tensors when done: del x
    7. Clear GPU cache: torch.cuda.empty_cache()
    8. Use gradient checkpointing for deep networks
    9. Profile memory usage to find bottlenecks
    10. Consider smaller batch sizes or model size
    """)

if __name__ == "__main__":
    main()
