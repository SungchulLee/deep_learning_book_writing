"""Tutorial 22: Optimization Basics - Optimizers and training"""
import torch
import torch.nn as nn
import torch.optim as optim

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. SGD Optimizer")
    params = [torch.tensor([2.0], requires_grad=True)]
    optimizer = optim.SGD(params, lr=0.1)
    print(f"Initial param: {params[0]}")
    for i in range(3):
        optimizer.zero_grad()
        loss = (params[0] - 1.0) ** 2
        loss.backward()
        print(f"Step {i+1}: grad={params[0].grad.item():.4f}, loss={loss.item():.4f}")
        optimizer.step()
        print(f"  Updated param: {params[0].item():.4f}")
    
    header("2. Adam Optimizer")
    model = nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Adam uses adaptive learning rates")
    print(f"Learning rate: {optimizer.defaults['lr']}")
    print(f"Betas: {optimizer.defaults['betas']}")
    
    header("3. Learning Rate Scheduler")
    optimizer = optim.SGD([torch.randn(1, requires_grad=True)], lr=1.0)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    for epoch in range(10):
        optimizer.step()
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: LR = {current_lr:.6f}")
    
    header("4. Gradient Clipping")
    model = nn.Linear(5, 1)
    x = torch.randn(10, 5)
    y = torch.randn(10, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    
    total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
    print(f"Gradient norm before clipping: {total_norm_before:.4f}")
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    total_norm_after = sum(p.grad.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
    print(f"Gradient norm after clipping: {total_norm_after:.4f}")
    
    header("5. Complete Training Example")
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X_train = torch.randn(100, 10)
    y_train = torch.randn(100, 1)
    
    print("Training for 5 epochs...")
    for epoch in range(5):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/5, Loss: {loss.item():.4f}")
    
    header("6. Comparing Optimizers")
    print("""
    Common optimizers:
    
    SGD: Simple, requires tuning, good generalization
    Adam: Adaptive, works well out-of-box, may overfit
    RMSprop: Good for RNNs
    AdaGrad: Good for sparse gradients
    
    Rule of thumb: Start with Adam, switch to SGD for final tuning
    """)

if __name__ == "__main__":
    main()
