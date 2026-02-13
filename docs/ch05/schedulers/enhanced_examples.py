"""
Enhanced Learning Rate Scheduler Examples
==========================================

This script demonstrates how to use both PyTorch built-in schedulers
and custom warmup/cyclical implementations together.

This combines:
- PyTorch's native schedulers (StepLR, MultiStepLR, etc.)
- Custom warmup implementations (Linear, Exponential, Cosine)
- Custom cyclical schedulers (with advanced features)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import PyTorch built-in schedulers
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR,
    OneCycleLR as PyTorchOneCycleLR, CyclicLR as PyTorchCyclicLR
)

# Import custom schedulers
from scheduler.custom import (
    LinearWarmup, ExponentialWarmup, CosineWarmup, WarmupWithDecay,
    CyclicLR, OneCycleLR, CosineAnnealingWarmRestarts
)


# ============================================================================
# SIMPLE MODEL FOR DEMONSTRATION
# ============================================================================
class SimpleNet(nn.Module):
    """Simple MLP for demonstration purposes."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ============================================================================
# DATA GENERATION
# ============================================================================
def create_dummy_data(num_samples=1000, input_size=784, num_classes=10, batch_size=32):
    """Create dummy dataset for demonstration."""
    X = torch.randn(num_samples, input_size)
    y = torch.randint(0, num_classes, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# ============================================================================
# EXAMPLE 1: PYTORCH STEPLR + CUSTOM LINEAR WARMUP
# ============================================================================
def example_warmup_then_step(model, dataloader, epochs=10):
    """
    Combines custom linear warmup with PyTorch StepLR.
    First 100 steps: warmup from 0 to 1e-3
    After warmup: use StepLR to decay every 30 epochs
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Custom Linear Warmup + PyTorch StepLR")
    print("="*70)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Warmup scheduler
    warmup_steps = 100
    base_lr = 1e-3
    warmup_scheduler = LinearWarmup(base_lr, warmup_steps)
    
    # Step scheduler (will activate after warmup)
    step_scheduler = StepLR(optimizer, step_size=3, gamma=0.5)
    
    step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Apply warmup schedule during initial steps
            if step < warmup_steps:
                current_lr = warmup_scheduler.get_lr(step)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
            
            # Training step
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
        
        # After warmup, use StepLR at epoch level
        if step >= warmup_steps:
            step_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1:2d} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f}")


# ============================================================================
# EXAMPLE 2: CUSTOM WARMUP WITH COSINE DECAY
# ============================================================================
def example_warmup_cosine_decay(model, dataloader, epochs=10):
    """
    Uses custom WarmupWithDecay which combines warmup and cosine annealing
    in a single scheduler - popular in transformer training.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Custom Warmup + Cosine Decay (Transformer-style)")
    print("="*70)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    warmup_steps = 100
    total_steps = len(dataloader) * epochs
    base_lr = 5e-4
    scheduler = WarmupWithDecay(base_lr, warmup_steps, total_steps, min_lr=1e-6)
    
    step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            current_lr = scheduler.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
        
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_lr(step-1)
        print(f"Epoch {epoch+1:2d} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f}")


# ============================================================================
# EXAMPLE 3: CUSTOM CYCLICAL LR (TRIANGULAR2)
# ============================================================================
def example_custom_cyclical(model, dataloader, epochs=10):
    """
    Uses custom CyclicLR implementation with triangular2 mode.
    Amplitude halves with each cycle for stable convergence.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Custom Cyclical LR (Triangular2 Mode)")
    print("="*70)
    
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Cycle every 2 epochs
    step_size = len(dataloader) * 2
    scheduler = CyclicLR(
        base_lr=1e-4,
        max_lr=1e-2,
        step_size=step_size,
        mode='triangular2'
    )
    
    step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            current_lr = scheduler.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
        
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_lr(step-1)
        print(f"Epoch {epoch+1:2d} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f}")


# ============================================================================
# EXAMPLE 4: CUSTOM 1CYCLE POLICY
# ============================================================================
def example_custom_1cycle(model, dataloader, epochs=5):
    """
    Uses custom OneCycleLR implementation - great for fast convergence.
    Increases LR for 30% of training, then decreases for remaining 70%.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Custom 1cycle Policy (Super-Convergence)")
    print("="*70)
    
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    total_steps = len(dataloader) * epochs
    scheduler = OneCycleLR(
        max_lr=1e-2,
        total_steps=total_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            current_lr = scheduler.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
        
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_lr(step-1)
        print(f"Epoch {epoch+1:2d} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f}")


# ============================================================================
# EXAMPLE 5: SGDR WITH WARM RESTARTS
# ============================================================================
def example_sgdr_warm_restarts(model, dataloader, epochs=10):
    """
    Uses custom SGDR (Stochastic Gradient Descent with Warm Restarts).
    Periodically restarts the learning rate to help escape local minima.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: SGDR with Warm Restarts")
    print("="*70)
    
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # First restart after 2 epochs, then double each time
    t_0 = len(dataloader) * 2
    scheduler = CosineAnnealingWarmRestarts(
        max_lr=1e-2,
        min_lr=1e-5,
        t_0=t_0,
        t_mult=2
    )
    
    step = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            current_lr = scheduler.get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
        
        avg_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_lr(step-1)
        print(f"Epoch {epoch+1:2d} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f}")


# ============================================================================
# EXAMPLE 6: COMPARING PYTORCH VS CUSTOM 1CYCLE
# ============================================================================
def example_compare_implementations(model1, model2, dataloader, epochs=5):
    """
    Side-by-side comparison of PyTorch's built-in OneCycleLR
    vs our custom implementation to show they produce similar results.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: PyTorch OneCycleLR vs Custom OneCycleLR")
    print("="*70)
    
    # Setup for PyTorch version
    optimizer1 = optim.SGD(model1.parameters(), lr=1e-4, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    torch_scheduler = PyTorchOneCycleLR(
        optimizer1,
        max_lr=1e-2,
        steps_per_epoch=len(dataloader),
        epochs=epochs,
        pct_start=0.3
    )
    
    # Setup for custom version
    optimizer2 = optim.SGD(model2.parameters(), lr=1e-4, momentum=0.9)
    total_steps = len(dataloader) * epochs
    custom_scheduler = OneCycleLR(
        max_lr=1e-2,
        total_steps=total_steps,
        pct_start=0.3
    )
    
    print("\n{:^15} | {:^20} | {:^20}".format("Epoch", "PyTorch LR", "Custom LR"))
    print("-" * 60)
    
    step = 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(dataloader):
            # PyTorch version
            optimizer1.zero_grad()
            output1 = model1(data)
            loss1 = criterion(output1, target)
            loss1.backward()
            optimizer1.step()
            torch_scheduler.step()
            
            # Custom version
            custom_lr = custom_scheduler.get_lr(step)
            for param_group in optimizer2.param_groups:
                param_group['lr'] = custom_lr
            
            optimizer2.zero_grad()
            output2 = model2(data)
            loss2 = criterion(output2, target)
            loss2.backward()
            optimizer2.step()
            
            step += 1
        
        torch_lr = optimizer1.param_groups[0]['lr']
        custom_lr = custom_scheduler.get_lr(step-1)
        print(f"{epoch+1:^15} | {torch_lr:^20.6f} | {custom_lr:^20.6f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ENHANCED LEARNING RATE SCHEDULER EXAMPLES")
    print("Combining PyTorch Built-in and Custom Implementations")
    print("="*70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create dataset
    dataloader = create_dummy_data(num_samples=1000, batch_size=32)
    
    # Run all examples
    print("\nRunning 6 comprehensive examples...")
    
    # Example 1
    model = SimpleNet()
    example_warmup_then_step(model, dataloader, epochs=10)
    
    # Example 2
    model = SimpleNet()
    example_warmup_cosine_decay(model, dataloader, epochs=10)
    
    # Example 3
    model = SimpleNet()
    example_custom_cyclical(model, dataloader, epochs=10)
    
    # Example 4
    model = SimpleNet()
    example_custom_1cycle(model, dataloader, epochs=5)
    
    # Example 5
    model = SimpleNet()
    example_sgdr_warm_restarts(model, dataloader, epochs=10)
    
    # Example 6 - Comparison
    model1 = SimpleNet()
    model2 = SimpleNet()
    # Copy weights so both models start identically
    model2.load_state_dict(model1.state_dict())
    example_compare_implementations(model1, model2, dataloader, epochs=5)
    
    print("\n" + "="*70)
    print("ALL EXAMPLES COMPLETED!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Custom warmup can be combined with PyTorch schedulers")
    print("2. WarmupWithDecay is great for transformer-style training")
    print("3. Custom cyclical schedulers offer more flexibility")
    print("4. SGDR helps escape local minima with periodic restarts")
    print("5. Both PyTorch and custom implementations work well")
    print("="*70 + "\n")
