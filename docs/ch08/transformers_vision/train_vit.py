"""
Training script for Vision Transformer
Demonstrates how to train ViT and compare with CNN approaches
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import time
from typing import Tuple, Dict
from vit_model import create_vit_tiny, create_vit_small, create_vit_base


class Trainer:
    """Training utility for Vision Transformer models"""
    
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   criterion: nn.Module,
                   optimizer: optim.Optimizer,
                   epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    def validate(self, 
                val_loader: DataLoader,
                criterion: nn.Module) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="[Validation]")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total
        }
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             n_epochs: int = 10,
             learning_rate: float = 3e-4,
             weight_decay: float = 0.1) -> Dict[str, list]:
        """Full training loop"""
        
        # Setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Cosine annealing scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs
        )
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        print(f"\n{'='*60}")
        print(f"Training Vision Transformer")
        print(f"Device: {self.device}")
        print(f"Epochs: {n_epochs}")
        print(f"Learning Rate: {learning_rate}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, n_epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(
                train_loader, criterion, optimizer, epoch
            )
            
            # Validate
            val_metrics = self.validate(val_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Record history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Print summary
            epoch_time = time.time() - start_time
            print(f"\nEpoch {epoch}/{n_epochs} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | "
                  f"Train Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  Time: {epoch_time:.2f}s | LR: {scheduler.get_last_lr()[0]:.6f}")
            print(f"{'-'*60}\n")
        
        return history


def get_data_loaders(data_dir: str = "./data",
                    batch_size: int = 128,
                    num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """
    Prepare data loaders for training.
    Uses standard augmentation techniques.
    """
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 as an example
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform
    )
    
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=val_transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def main():
    """Main training function"""
    
    # Configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    n_epochs = 50
    learning_rate = 3e-4
    
    print("Loading data...")
    train_loader, val_loader = get_data_loaders(
        batch_size=batch_size,
        num_workers=4
    )
    
    print("Creating model...")
    # Use ViT-Tiny for faster training
    model = create_vit_tiny(n_classes=10)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    trainer = Trainer(model, device=device)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        learning_rate=learning_rate
    )
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history
    }, 'vit_checkpoint.pth')
    
    print("\nTraining complete! Model saved to 'vit_checkpoint.pth'")


if __name__ == "__main__":
    main()
