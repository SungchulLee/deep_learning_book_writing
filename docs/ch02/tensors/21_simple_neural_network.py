"""Tutorial 21: Simple Neural Network - Building from scratch"""
import torch
import torch.nn as nn

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

def main():
    header("1. Simple Linear Layer")
    input_size, output_size = 3, 2
    layer = nn.Linear(input_size, output_size)
    print(f"Layer: {layer}")
    print(f"Weight shape: {layer.weight.shape}")  # (2, 3)
    print(f"Bias shape: {layer.bias.shape}")  # (2,)
    x = torch.randn(1, 3)  # Batch of 1 sample
    output = layer(x)
    print(f"\nInput: {x.shape}")
    print(f"Output: {output.shape}")
    
    header("2. Multi-Layer Network")
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 10)
            self.fc3 = nn.Linear(10, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    model = SimpleNet()
    print(model)
    x = torch.randn(5, 10)  # 5 samples, 10 features
    output = model(x)
    print(f"\nInput: {x.shape}")
    print(f"Output: {output.shape}")
    
    header("3. Model Parameters")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    
    header("4. Forward Pass")
    x = torch.randn(3, 10)
    with torch.no_grad():
        output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")
    
    header("5. Loss Function")
    predictions = torch.tensor([[2.5], [3.0], [1.5]], requires_grad=True)
    targets = torch.tensor([[2.0], [3.0], [2.0]])
    mse_loss = nn.MSELoss()
    loss = mse_loss(predictions, targets)
    print(f"Predictions:\n{predictions}")
    print(f"Targets:\n{targets}")
    print(f"MSE Loss: {loss.item():.4f}")
    
    header("6. Backward Pass")
    loss.backward()
    print(f"Gradient of predictions:\n{predictions.grad}")
    
    header("7. Simple Training Loop")
    model = SimpleNet()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    X = torch.randn(100, 10)  # 100 samples
    y = torch.randn(100, 1)   # 100 targets
    
    for epoch in range(3):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        if epoch % 1 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    header("8. Model Evaluation")
    model.eval()  # Set to evaluation mode
    with torch.no_grad():
        test_input = torch.randn(5, 10)
        test_output = model(test_input)
    print(f"Test output shape: {test_output.shape}")
    print("Model in eval mode - no gradients computed!")

if __name__ == "__main__":
    main()
