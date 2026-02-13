"""
INTERMEDIATE LEVEL: Quantization-Aware Training (QAT)

This script demonstrates quantization-aware training, where we simulate
quantization during training so the model learns weights robust to quantization.

Topics Covered:
- Fake quantization during training
- Quantization-aware training process
- Comparison with post-training quantization
- Per-channel vs per-tensor quantization

Mathematical Background:
- Forward pass: Use quantized weights
- Backward pass: Use straight-through estimator
  ∂L/∂w_float ≈ ∂L/∂w_quant (pretend quantization is identity)
- STE allows gradients to flow through non-differentiable quantization

Prerequisites:
- Module 01: Quantization Basics
- Understanding of backpropagation
- Familiarity with training loops
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

from utils import (
    evaluate_accuracy,
    compare_model_sizes,
    compare_accuracies,
    seed_everything
)


class SimpleResNet(nn.Module):
    """ResNet-like model for CIFAR-10."""
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Initial conv
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Residual block 1
        identity = x
        x = self.relu(self.bn2(self.conv2(x)))
        x = x + identity
        
        # Residual block 2 (with downsampling)
        x = self.relu(self.bn3(self.conv3(x)))
        identity = x
        x = self.relu(self.bn4(self.conv4(x)))
        x = x + identity
        
        # Classification head
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def get_cifar10_dataloaders(batch_size=128):
    """Load CIFAR-10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_with_qat(model, train_loader, test_loader, epochs=20, lr=0.01, device='cpu'):
    """
    Train model with quantization-aware training.
    
    QAT Process:
    1. Insert fake quantization modules
    2. Train with quantization simulation
    3. Gradients flow through straight-through estimator
    4. Convert to actual quantized model after training
    """
    model = model.to(device)
    
    # Prepare model for QAT
    model.train()
    model.qconfig = quant.get_default_qat_qconfig('fbgemm')
    quant.prepare_qat(model, inplace=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\nTraining with Quantization-Aware Training...")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Freeze batch norm after warmup
        if epoch > epochs // 2:
            model.apply(torch.quantization.disable_observer)
        if epoch > epochs * 3 // 4:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        scheduler.step()
        
        if epoch % 5 == 0:
            test_acc = evaluate_accuracy(model, test_loader, device)
            print(f"Epoch [{epoch+1}/{epochs}] Test Acc: {test_acc*100:.2f}%")
    
    # Convert to quantized model
    model.eval()
    model_quantized = quant.convert(model, inplace=False)
    
    return model_quantized


def main():
    """Main function for QAT demonstration."""
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("QUANTIZATION-AWARE TRAINING DEMONSTRATION")
    print("="*60)
    
    # Load data
    train_loader, test_loader = get_cifar10_dataloaders()
    
    # Create and train model with QAT
    model = SimpleResNet(num_classes=10)
    model_qat = train_with_qat(model, train_loader, test_loader, epochs=20, device=device)
    
    # Evaluate
    qat_acc = evaluate_accuracy(model_qat, test_loader, device)
    print(f"\nQAT Model Accuracy: {qat_acc*100:.2f}%")
    
    print("\nQAT typically achieves:")
    print("- Better accuracy than PTQ (1-2% improvement)")
    print("- Same size reduction (4x for INT8)")
    print("- Requires retraining (longer time)")
    print("- Best for models sensitive to quantization")


if __name__ == "__main__":
    main()
