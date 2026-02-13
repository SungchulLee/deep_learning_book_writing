"""
09_hogwild_training.py
=======================
Multi-Process Parallel Training (Hogwild!)

This advanced tutorial demonstrates lock-free parallel SGD training
using multiple CPU processes. Each process trains on the shared model
simultaneously - a technique called "Hogwild!"

What you'll learn:
- Multi-process training in PyTorch
- Shared memory for model parameters
- Lock-free parallel SGD
- Speedup analysis
- When parallel training helps

Difficulty: Advanced
Estimated Time: 2-3 hours

Prerequisites:
- Multi-core CPU (benefits increase with more cores)
- Understanding of training loops
- Familiarity with Python multiprocessing

Author: PyTorch CNN Tutorial
Date: November 2025

Reference: Hogwild!: A Lock-Free Approach to Parallelizing SGD (Recht et al., 2011)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import argparse
import time

print("=" * 70)
print("Hogwild! Multi-Process Training")
print("=" * 70)

# =============================================================================
# Configuration
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Hogwild Training')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num-processes', type=int, default=2,
                        help='Number of parallel training processes')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist', 'cifar10'])
    args = parser.parse_args()
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return args

# =============================================================================
# Model Definition
# =============================================================================

class CNN(nn.Module):
    """Basic CNN that works for all datasets"""
    def __init__(self, in_channels=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# =============================================================================
# Data Loading
# =============================================================================

def load_data(args):
    """Load specified dataset"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if args.dataset != 'cifar10' 
        else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if args.dataset == 'mnist':
        train_dataset = datasets.MNIST(root='./data', train=True, 
                                       download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False, 
                                      download=True, transform=transform)
    elif args.dataset == 'fashion-mnist':
        train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                              download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                             download=True, transform=transform)
    else:  # cifar10
        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    
    return train_dataset, test_dataset

# =============================================================================
# Training Function for Each Process
# =============================================================================

def train_process(rank, model, args, train_dataset):
    """
    Training function run by each process
    
    Args:
        rank: Process ID (0, 1, 2, ...)
        model: Shared model (all processes update same model!)
        args: Training arguments
        train_dataset: Training dataset
    """
    print(f"[Process {rank}] Starting training...")
    
    # Each process gets its own data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Each process has its own optimizer, but updates shared model
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(args.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()  # Updates shared model parameters!
            
            if batch_idx % 100 == 0:
                print(f'[Process {rank}] Epoch {epoch+1}, '
                      f'Batch {batch_idx}, Loss: {loss.item():.4f}')
    
    print(f"[Process {rank}] Training complete!")

# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate(model, test_loader, device):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100.0 * correct / total

# =============================================================================
# Main Function
# =============================================================================

def main():
    args = parse_args()
    
    print(f"\nConfiguration:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num processes: {args.num_processes}")
    print(f"  Device: {args.device}")
    
    # Load data
    print("\nLoading dataset...")
    train_dataset, test_dataset = load_data(args)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size)
    
    # Create model
    in_channels = 3 if args.dataset == 'cifar10' else 1
    model = CNN(in_channels=in_channels).to(args.device)
    
    # CRITICAL: Share model memory across processes
    model.share_memory()
    print("\nModel memory shared across processes")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluate before training
    print("\nBefore training:")
    acc_before = evaluate(model, test_loader, args.device)
    print(f"Accuracy: {acc_before:.2f}%")
    
    # Launch parallel training processes
    print(f"\nLaunching {args.num_processes} parallel training processes...")
    print("Each process trains on the SAME shared model!")
    print("This is Hogwild! - lock-free parallel SGD\n")
    
    start_time = time.time()
    
    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train_process, args=(rank, model, args, train_dataset))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    training_time = time.time() - start_time
    
    # Evaluate after training
    print("\nAfter training:")
    acc_after = evaluate(model, test_loader, args.device)
    print(f"Accuracy: {acc_after:.2f}%")
    
    # =============================================================================
    # Results Analysis
    # =============================================================================
    
    print(f"\n{'='*70}")
    print("Results")
    print(f"{'='*70}")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Accuracy improvement: {acc_before:.2f}% → {acc_after:.2f}%")
    print(f"Number of processes: {args.num_processes}")
    
    print(f"""

How Hogwild! Works:
-------------------
1. Model parameters are in shared memory
2. Multiple processes train simultaneously
3. Each updates the SAME model (no locks!)
4. Some gradient updates may conflict
5. But overall it converges faster

Key Benefits:
-------------
✓ Near-linear speedup with more cores
✓ No synchronization overhead
✓ Simple to implement
✓ Works well in practice

When to Use:
------------
✓ Training on CPU with multiple cores
✓ Sparse gradient updates
✓ When communication is expensive
✗ Not for distributed training (use DDP instead)

Theory:
-------
Hogwild! works because:
• SGD is robust to noise
• Conflicting updates are rare with sparse gradients
• Benefits outweigh occasional conflicts

Try Different Configurations:
-----------------------------
python 09_hogwild_training.py --num-processes 4 --epochs 5
python 09_hogwild_training.py --dataset fashion-mnist --num-processes 2
python 09_hogwild_training.py --dataset cifar10 --num-processes 4 --epochs 10

Challenge:
----------
Compare training time with 1, 2, 4, and 8 processes!
Plot speedup vs number of processes.

Reference Paper:
----------------
Recht, B., Re, C., Wright, S., & Niu, F. (2011).
Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent.
""")
    
    print("=" * 70)
    print("Tutorial Complete! ✓")
    print("=" * 70)

if __name__ == '__main__':
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
