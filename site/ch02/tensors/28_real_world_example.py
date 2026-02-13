"""Tutorial 28: Real-World Example - Complete MNIST digit classifier"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

def header(title): print(f"\n{'='*70}\n{title}\n{'='*70}")

# Synthetic MNIST-like dataset (replace with real MNIST in practice)
class SyntheticMNIST(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 1, 28, 28)
        self.labels = torch.randint(0, 10, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class ConvNet(nn.Module):
    """Convolutional Neural Network for digit classification."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    header("Complete MNIST Classifier Pipeline")
    
    # 1. Setup
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Data preparation
    print("\n1. Preparing Data...")
    dataset = SyntheticMNIST(size=1000)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # 3. Model creation
    print("\n2. Creating Model...")
    model = ConvNet().to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # 4. Loss and optimizer
    print("\n3. Setting up Training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    # 5. Training loop
    print("\n4. Training...")
    num_epochs = 5
    best_val_acc = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  → New best model! Acc: {best_val_acc:.2f}%")
    
    # 6. Final evaluation
    print("\n5. Final Evaluation...")
    final_loss, final_acc = validate(model, val_loader, criterion, device)
    print(f"Final Validation Accuracy: {final_acc:.2f}%")
    
    # 7. Inference example
    print("\n6. Inference Example...")
    model.eval()
    with torch.no_grad():
        sample_data, sample_label = dataset[0]
        sample_data = sample_data.unsqueeze(0).to(device)
        output = model(sample_data)
        _, predicted = torch.max(output, 1)
        print(f"True label: {sample_label.item()}")
        print(f"Predicted: {predicted.item()}")
        print(f"Confidence: {torch.softmax(output, 1).max().item():.4f}")
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print("""
    Next Steps:
    1. Try with real MNIST dataset (torchvision.datasets.MNIST)
    2. Experiment with different architectures
    3. Add data augmentation
    4. Try different optimizers and learning rates
    5. Implement early stopping
    6. Add logging with tensorboard
    7. Deploy the model for production use
    """)

if __name__ == "__main__":
    main()
