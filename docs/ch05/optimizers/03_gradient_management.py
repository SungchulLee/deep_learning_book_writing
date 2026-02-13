"""
================================================================================
INTERMEDIATE 03: Gradient Manipulation and Clipping
================================================================================

WHAT YOU'LL LEARN:
- Understanding gradient explosion and vanishing
- Gradient clipping techniques
- Gradient accumulation for large batch sizes
- Monitoring gradients during training
- Best practices for gradient management

PREREQUISITES:
- Complete beginner tutorials
- Understand backpropagation basics

TIME TO COMPLETE: ~20 minutes
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 80)
print("GRADIENT MANIPULATION AND CLIPPING")
print("=" * 80)

# ============================================================================
# SECTION 1: Understanding Gradient Problems
# ============================================================================
print("\n" + "-" * 80)
print("GRADIENT PROBLEMS IN DEEP LEARNING")
print("-" * 80)

print("""
TWO MAJOR GRADIENT PROBLEMS:

1. GRADIENT EXPLOSION:
   • Gradients become very large (>1000)
   • Causes unstable training
   • Weights update by huge amounts
   • Loss becomes NaN or Inf
   • Common in RNNs and very deep networks

2. GRADIENT VANISHING:
   • Gradients become very small (<0.001)
   • Early layers don't learn
   • Training is extremely slow
   • Common in deep networks without proper activation/normalization

SOLUTIONS:
   • Gradient clipping (for explosion)
   • Better architectures (ResNet, BatchNorm)
   • Better activations (ReLU instead of sigmoid)
   • Careful initialization
""")

# ============================================================================
# SECTION 2: Simulating Gradient Explosion
# ============================================================================
print("\n" + "-" * 80)
print("SIMULATING GRADIENT EXPLOSION")
print("-" * 80)

class ProblematicModel(nn.Module):
    """
    A model prone to gradient explosion
    (Poor initialization, no normalization)
    """
    def __init__(self):
        super(ProblematicModel, self).__init__()
        # Large initial weights → potential explosion
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        
        # Initialize with large weights (bad practice!)
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.normal_(layer.weight, mean=0, std=2.0)  # Too large!
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create problematic model
bad_model = ProblematicModel()
optimizer = optim.SGD(bad_model.parameters(), lr=0.1)  # High LR makes it worse
criterion = nn.MSELoss()

print("Training model WITHOUT gradient clipping:\n")

# Train for a few steps
for step in range(5):
    # Generate random data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    # Forward pass
    outputs = bad_model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradient norms
    total_norm = 0
    for p in bad_model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    print(f"Step {step+1}: Loss = {loss.item():.4f}, Gradient Norm = {total_norm:.4f}")
    
    # Update (might explode!)
    optimizer.step()
    
    # Check if loss became NaN
    if torch.isnan(loss):
        print("\n⚠️  Training collapsed! Loss became NaN due to gradient explosion")
        break

# ============================================================================
# SECTION 3: Gradient Clipping by Norm
# ============================================================================
print("\n" + "-" * 80)
print("GRADIENT CLIPPING BY NORM")
print("-" * 80)

print("""
GRADIENT CLIPPING BY NORM:
  • Rescale gradients if their norm exceeds a threshold
  • Formula: g_clipped = (max_norm / norm(g)) × g  if norm(g) > max_norm
  • Preserves gradient direction
  • Most common clipping method
  
torch.nn.utils.clip_grad_norm_(parameters, max_norm)
""")

# Create new model with same architecture
model = ProblematicModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)
max_norm = 1.0  # Clip gradients to max norm of 1.0

print(f"Training model WITH gradient clipping (max_norm={max_norm}):\n")

for step in range(5):
    # Generate random data
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Compute gradient norm BEFORE clipping
    total_norm_before = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_before += param_norm.item() ** 2
    total_norm_before = total_norm_before ** 0.5
    
    # GRADIENT CLIPPING
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    
    # Compute gradient norm AFTER clipping
    total_norm_after = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** 0.5
    
    print(f"Step {step+1}: Loss = {loss.item():.4f}, "
          f"Grad Norm: {total_norm_before:.4f} → {total_norm_after:.4f}")
    
    # Update parameters
    optimizer.step()

print("\n✓ Training is stable! Gradients are clipped to max_norm")

# ============================================================================
# SECTION 4: Gradient Clipping by Value
# ============================================================================
print("\n" + "-" * 80)
print("GRADIENT CLIPPING BY VALUE")
print("-" * 80)

print("""
GRADIENT CLIPPING BY VALUE:
  • Clip each gradient element to [-clip_value, clip_value]
  • Simpler but can distort gradient direction
  • Less common than clipping by norm
  
torch.nn.utils.clip_grad_value_(parameters, clip_value)
""")

# Create new model
model_value_clip = ProblematicModel()
optimizer_value = optim.SGD(model_value_clip.parameters(), lr=0.1)
clip_value = 0.5

print(f"Training with VALUE clipping (clip_value={clip_value}):\n")

for step in range(3):
    inputs = torch.randn(32, 10)
    targets = torch.randn(32, 1)
    
    outputs = model_value_clip(inputs)
    loss = criterion(outputs, targets)
    
    optimizer_value.zero_grad()
    loss.backward()
    
    # Show gradient statistics before clipping
    max_grad = max(p.grad.abs().max().item() 
                   for p in model_value_clip.parameters() if p.grad is not None)
    
    # GRADIENT CLIPPING BY VALUE
    torch.nn.utils.clip_grad_value_(model_value_clip.parameters(), clip_value)
    
    # Show gradient statistics after clipping
    max_grad_after = max(p.grad.abs().max().item() 
                         for p in model_value_clip.parameters() if p.grad is not None)
    
    print(f"Step {step+1}: Max gradient: {max_grad:.4f} → {max_grad_after:.4f}")
    
    optimizer_value.step()

# ============================================================================
# SECTION 5: Gradient Accumulation
# ============================================================================
print("\n" + "-" * 80)
print("GRADIENT ACCUMULATION")
print("-" * 80)

print("""
GRADIENT ACCUMULATION:
  • Simulate large batch sizes with limited memory
  • Accumulate gradients over multiple forward passes
  • Update parameters after N accumulation steps
  • Effective batch size = batch_size × accumulation_steps
  
USE CASES:
  • GPU memory is limited
  • Want to train with large effective batch sizes
  • Training very large models
""")

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

model_accum = SimpleModel()
optimizer_accum = optim.SGD(model_accum.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Configuration
batch_size = 8
accumulation_steps = 4  # Effective batch size = 8 × 4 = 32

print(f"Batch size: {batch_size}")
print(f"Accumulation steps: {accumulation_steps}")
print(f"Effective batch size: {batch_size * accumulation_steps}\n")

print("Training with gradient accumulation:")

for epoch in range(2):
    print(f"\nEpoch {epoch + 1}:")
    
    # Simulate 4 mini-batches
    for step in range(4):
        # Generate small batch
        inputs = torch.randn(batch_size, 10)
        targets = torch.randn(batch_size, 1)
        
        # Forward pass
        outputs = model_accum(inputs)
        loss = criterion(outputs, targets)
        
        # Normalize loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass (accumulate gradients)
        loss.backward()
        
        print(f"  Step {step+1}: Loss = {loss.item() * accumulation_steps:.4f}")
        
        # Update parameters every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            optimizer_accum.step()
            optimizer_accum.zero_grad()
            print("  → Parameters updated and gradients cleared")

print("\n✓ Gradient accumulation allows training with larger effective batch size")

# ============================================================================
# SECTION 6: Monitoring Gradients
# ============================================================================
print("\n" + "-" * 80)
print("MONITORING GRADIENTS DURING TRAINING")
print("-" * 80)

def compute_gradient_stats(model):
    """
    Compute statistics about gradients in the model
    """
    stats = {
        'max': 0.0,
        'min': float('inf'),
        'mean': 0.0,
        'norm': 0.0,
        'num_params': 0
    }
    
    total_sum = 0.0
    total_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            
            # Update statistics
            stats['max'] = max(stats['max'], grad.abs().max().item())
            stats['min'] = min(stats['min'], grad.abs().min().item())
            total_sum += grad.sum().item()
            total_count += grad.numel()
            
            # Compute norm contribution
            stats['norm'] += grad.norm().item() ** 2
            stats['num_params'] += 1
    
    if total_count > 0:
        stats['mean'] = total_sum / total_count
        stats['norm'] = stats['norm'] ** 0.5
    
    return stats

# Demo gradient monitoring
model_monitor = SimpleModel()
optimizer_monitor = optim.SGD(model_monitor.parameters(), lr=0.01)

print("Example gradient statistics:\n")

for step in range(3):
    inputs = torch.randn(16, 10)
    targets = torch.randn(16, 1)
    
    outputs = model_monitor(inputs)
    loss = criterion(outputs, targets)
    
    optimizer_monitor.zero_grad()
    loss.backward()
    
    # Compute and display gradient statistics
    stats = compute_gradient_stats(model_monitor)
    
    print(f"Step {step+1}:")
    print(f"  Gradient norm: {stats['norm']:.6f}")
    print(f"  Max gradient: {stats['max']:.6f}")
    print(f"  Min gradient: {stats['min']:.6f}")
    print(f"  Mean gradient: {stats['mean']:.6f}\n")
    
    optimizer_monitor.step()

# ============================================================================
# SECTION 7: Best Practices
# ============================================================================
print("\n" + "-" * 80)
print("BEST PRACTICES FOR GRADIENT MANAGEMENT")
print("-" * 80)

print("""
✓ ALWAYS DO:

1. MONITOR GRADIENTS:
   • Log gradient norms during training
   • Watch for explosion (>10) or vanishing (<0.0001)
   • Use tensorboard or wandb for visualization

2. USE GRADIENT CLIPPING:
   • Especially for RNNs and transformers
   • Typical max_norm: 0.5 - 5.0
   • Start with 1.0 and adjust

3. PROPER INITIALIZATION:
   • Use Xavier/Kaiming initialization
   • Avoid large initial weights
   • PyTorch does this by default

4. BATCH NORMALIZATION:
   • Helps stabilize gradients
   • Reduces need for aggressive clipping

5. APPROPRIATE LEARNING RATE:
   • Too high → explosion
   • Too low → vanishing
   • Use learning rate schedulers

✗ AVOID:

1. IGNORING GRADIENT PROBLEMS:
   • If loss becomes NaN, investigate gradients
   • Don't just restart training

2. VERY DEEP NETWORKS WITHOUT SKIP CONNECTIONS:
   • Use ResNet-style architectures
   • Add skip connections

3. SIGMOID/TANH IN DEEP NETWORKS:
   • Causes vanishing gradients
   • Use ReLU or variants

4. ACCUMULATING GRADIENTS UNINTENTIONALLY:
   • Always call optimizer.zero_grad()
   • Before loss.backward()
""")

# ============================================================================
# SECTION 8: Complete Training Loop with Gradient Management
# ============================================================================
print("\n" + "-" * 80)
print("COMPLETE EXAMPLE: Training Loop with Gradient Management")
print("-" * 80)

print("""
def train_with_gradient_management(model, train_loader, epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    max_norm = 1.0  # Gradient clipping threshold
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (optional but recommended)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            # Monitor gradients (every N batches)
            if batch_idx % 100 == 0:
                grad_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norm += p.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                
                print(f'Epoch {epoch}, Batch {batch_idx}: '
                      f'Loss={loss.item():.4f}, '
                      f'Grad Norm={grad_norm:.4f}')
            
            # Parameter update
            optimizer.step()
""")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. GRADIENT PROBLEMS:
   • Explosion: Gradients too large → unstable training
   • Vanishing: Gradients too small → slow/no learning

2. GRADIENT CLIPPING:
   • By norm: torch.nn.utils.clip_grad_norm_(params, max_norm)
     → Preserves direction, most common
   • By value: torch.nn.utils.clip_grad_value_(params, clip_value)
     → Simpler but can distort direction

3. GRADIENT ACCUMULATION:
   • Simulate large batch sizes
   • Accumulate over multiple forward passes
   • Useful when GPU memory is limited

4. MONITORING:
   • Always track gradient norms
   • Warning signs: norm > 10 or norm < 0.0001
   • Log and visualize during training

5. IMPLEMENTATION:
   • Clip after backward(), before step()
   • Typical max_norm: 0.5 - 5.0 (start with 1.0)
   • Combine with proper initialization and architecture

6. WHEN TO USE CLIPPING:
   • RNNs/LSTMs: Almost always
   • Transformers: Very common
   • CNNs: Sometimes, especially if very deep
   • Simple networks: Rarely needed

NEXT STEPS:
→ Add gradient monitoring to your training loops
→ Experiment with different clipping thresholds
→ Use gradient accumulation for larger effective batch sizes
→ Study advanced techniques (gradient centralization, etc.)
""")
print("=" * 80)
