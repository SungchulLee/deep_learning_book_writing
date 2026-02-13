"""Tutorial 24: Mixed Precision Training - Faster training with less memory"""
import torch
import torch.nn as nn

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. What is Mixed Precision?")
    print("""
    Mixed Precision Training:
    - Uses float16 (half precision) for forward/backward passes
    - Uses float32 (full precision) for parameter updates
    - Benefits: 2-3x speedup, 50% memory reduction
    - Requires: GPU with Tensor Cores (V100, A100, RTX 20/30 series)
    """)
    
    header("2. Float16 vs Float32")
    x_32 = torch.tensor([1.0], dtype=torch.float32)
    x_16 = torch.tensor([1.0], dtype=torch.float16)
    print(f"Float32: {x_32.dtype}, size: {x_32.element_size()} bytes")
    print(f"Float16: {x_16.dtype}, size: {x_16.element_size()} bytes")
    print(f"Memory savings: {100 * (1 - x_16.element_size()/x_32.element_size()):.0f}%")
    
    header("3. Automatic Mixed Precision (AMP)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(100, 200),
        nn.ReLU(),
        nn.Linear(200, 10)
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler()  # Prevents underflow
    
    print("Using autocast for mixed precision:")
    x = torch.randn(32, 100, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    
    # Training step with AMP
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():  # Automatic mixed precision
        output = model(x)
        loss = nn.functional.cross_entropy(output, y)
    
    scaler.scale(loss).backward()  # Scale loss to prevent underflow
    scaler.step(optimizer)
    scaler.update()
    
    print(f"Loss computed in mixed precision: {loss.item():.4f}")
    
    header("4. Manual Mixed Precision")
    model_fp16 = model.half()  # Convert to float16
    print("Model converted to float16:")
    for name, param in model_fp16.named_parameters():
        print(f"  {name}: {param.dtype}")
    
    header("5. When to Use Mixed Precision")
    print("""
    Use Mixed Precision when:
    ✓ Training large models
    ✓ Working with limited GPU memory
    ✓ Have compatible GPU (Tensor Cores)
    ✓ Batch size bottleneck
    
    Be careful with:
    ✗ Very small gradients (use GradScaler)
    ✗ Custom operations (may not support fp16)
    ✗ Numerical stability issues
    """)
    
    header("6. Complete Training Example")
    print("""
    # Standard training
    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(epochs):
        for x, y in dataloader:
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(x)
                loss = criterion(output, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    """)

if __name__ == "__main__":
    main()
