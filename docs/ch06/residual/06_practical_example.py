"""
Practical Example: Training ResNet on CIFAR-10
==============================================
Complete example of training a ResNet model on a real dataset.
Includes data loading, training loop, evaluation, and best practices.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import time
import os


# Import ResNet from our implementation
# Note: In practice, run this from the residual_connections directory
# or adjust the import path accordingly
try:
    from residual_connections_02_resnet_implementation import resnet18, resnet34, resnet50
except ImportError:
    # Alternative: create minimal ResNet here or import from torchvision
    print("Note: Could not import from 02_resnet_implementation.py")
    print("Make sure to run from the correct directory or adjust imports")
    import sys
    sys.exit(1)


def get_cifar10_dataloaders(batch_size=128, num_workers=2):
    """
    Load and prepare CIFAR-10 dataset with appropriate augmentation
    """
    print("Loading CIFAR-10 dataset...")
    
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # No augmentation for validation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Download and load training data
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # Download and load test data
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    print(f"Training samples: {len(trainset)}")
    print(f"Test samples: {len(testset)}")
    print(f"Classes: {classes}")
    
    return trainloader, testloader, classes


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train for one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Print progress
        if batch_idx % 100 == 0:
            print(f'  Batch [{batch_idx}/{len(dataloader)}] '
                  f'Loss: {loss.item():.3f} | Acc: {100.*correct/total:.2f}%')
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on test set
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def train_resnet_cifar10(
    model_name='resnet18',
    num_epochs=100,
    batch_size=128,
    learning_rate=0.1,
    weight_decay=5e-4,
    device=None
):
    """
    Complete training pipeline for ResNet on CIFAR-10
    
    Args:
        model_name: 'resnet18', 'resnet34', or 'resnet50'
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        weight_decay: L2 regularization strength
        device: Device to train on (None for auto-detect)
    """
    print("=" * 80)
    print(f"Training {model_name.upper()} on CIFAR-10")
    print("=" * 80)
    
    # Setup device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load data
    trainloader, testloader, classes = get_cifar10_dataloaders(batch_size)
    
    # Create model
    print(f"\nInitializing {model_name}...")
    if model_name == 'resnet18':
        model = resnet18(num_classes=10)
    elif model_name == 'resnet34':
        model = resnet34(num_classes=10)
    elif model_name == 'resnet50':
        model = resnet50(num_classes=10)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                         momentum=0.9, weight_decay=weight_decay)
    
    # Learning rate scheduler (cosine annealing)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': []
    }
    
    best_acc = 0
    start_time = time.time()
    
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, trainloader, criterion, optimizer, device, epoch)
        
        # Evaluate
        test_loss, test_acc = evaluate(model, testloader, criterion, device)
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, f'{model_name}_cifar10_best.pth')
            print(f"  ✓ New best model saved! (Test Acc: {best_acc:.2f}%)")
        
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
        print(f"  Time: {epoch_time:.2f}s")
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total training time: {total_time/60:.2f} minutes")
    print(f"Best test accuracy: {best_acc:.2f}%")
    print(f"Final test accuracy: {history['test_acc'][-1]:.2f}%")
    
    return model, history


def evaluate_per_class(model, dataloader, classes, device):
    """
    Evaluate per-class accuracy
    """
    print("\n" + "=" * 80)
    print("Per-Class Accuracy Analysis")
    print("=" * 80)
    
    model.eval()
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for i in range(len(targets)):
                label = targets[i].item()
                class_correct[label] += (predicted[i] == targets[i]).item()
                class_total[label] += 1
    
    print(f"\n{'Class':15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    for i, class_name in enumerate(classes):
        acc = 100 * class_correct[i] / class_total[i]
        print(f"{class_name:15} {class_correct[i]:8d} {class_total[i]:8d} {acc:9.2f}%")
    
    overall_acc = 100 * sum(class_correct) / sum(class_total)
    print("-" * 50)
    print(f"{'Overall':15} {sum(class_correct):8d} {sum(class_total):8d} {overall_acc:9.2f}%")
    print("=" * 80)


def quick_demo(epochs=2):
    """
    Quick demo with just a few epochs for testing
    """
    print("\n" + "=" * 80)
    print("QUICK DEMO: Training ResNet-18 on CIFAR-10")
    print("=" * 80)
    print("\nNote: This is a quick demo with only 2 epochs.")
    print("For real training, use 100+ epochs to achieve ~93% accuracy.")
    
    model, history = train_resnet_cifar10(
        model_name='resnet18',
        num_epochs=epochs,
        batch_size=128,
        learning_rate=0.1
    )
    
    return model, history


if __name__ == "__main__":
    # Quick demo mode
    print("\n" + "=" * 80)
    print("PRACTICAL EXAMPLE: ResNet on CIFAR-10")
    print("=" * 80)
    
    print("\nThis script demonstrates:")
    print("1. Loading and preprocessing CIFAR-10 dataset")
    print("2. Training ResNet with proper hyperparameters")
    print("3. Using learning rate scheduling")
    print("4. Evaluating model performance")
    print("5. Saving best model checkpoints")
    
    print("\n" + "=" * 80)
    print("Running quick demo (2 epochs)...")
    print("=" * 80)
    
    # Run quick demo
    model, history = quick_demo(epochs=2)
    
    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)
    
    print("\nTo train a full model (100 epochs, ~93% accuracy):")
    print("  python 06_practical_example.py --full-training")
    print("\nExpected results after full training:")
    print("  - ResNet-18: ~93-94% test accuracy")
    print("  - ResNet-34: ~94-95% test accuracy")
    print("  - ResNet-50: ~94-95% test accuracy")
    
    print("\n" + "=" * 80 + "\n")
