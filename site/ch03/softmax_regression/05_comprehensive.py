"""
===============================================================================
LEVEL 5: Comprehensive Multi-Dataset Classification Project
===============================================================================
Difficulty: Advanced
Prerequisites: Level 1-4
Learning Goals:
  - Work with multiple real datasets
  - Build a flexible model factory
  - Implement comprehensive experiment tracking
  - Create reusable training pipelines
  - Generate detailed performance reports
  - Compare architectures systematically

Time to complete: 90-120 minutes
===============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from collections import defaultdict

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("LEVEL 5: COMPREHENSIVE MULTI-DATASET CLASSIFICATION")
print("=" * 80)


# =============================================================================
# PART 1: Dataset Manager
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: Building a Dataset Manager")
print("=" * 80)

class DatasetManager:
    """
    Centralized dataset loading and preprocessing.
    Supports multiple datasets with consistent interface.
    """
    
    def __init__(self, dataset_name, batch_size=128, val_split=0.1):
        """
        Initialize dataset manager.
        
        Args:
            dataset_name: 'mnist', 'fashion_mnist', or 'cifar10'
            batch_size: Batch size for data loaders
            val_split: Fraction of training data for validation
        """
        self.dataset_name = dataset_name.lower()
        self.batch_size = batch_size
        self.val_split = val_split
        
        # Load dataset
        self.train_loader, self.val_loader, self.test_loader = self._load_dataset()
        self.num_classes = self._get_num_classes()
        self.input_shape = self._get_input_shape()
        
    def _load_dataset(self):
        """Load and prepare the specified dataset."""
        if self.dataset_name == 'mnist':
            return self._load_mnist()
        elif self.dataset_name == 'fashion_mnist':
            return self._load_fashion_mnist()
        elif self.dataset_name == 'cifar10':
            return self._load_cifar10()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _load_mnist(self):
        """Load MNIST dataset."""
        print(f"Loading MNIST...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = datasets.MNIST(root='./data', train=True,
                                      download=True, transform=transform)
        test_dataset = datasets.MNIST(root='./data', train=False,
                                     download=True, transform=transform)
        
        return self._create_loaders(train_dataset, test_dataset)
    
    def _load_fashion_mnist(self):
        """Load Fashion-MNIST dataset."""
        print(f"Loading Fashion-MNIST...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        train_dataset = datasets.FashionMNIST(root='./data', train=True,
                                             download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False,
                                            download=True, transform=transform)
        
        return self._create_loaders(train_dataset, test_dataset)
    
    def _load_cifar10(self):
        """Load CIFAR-10 dataset."""
        print(f"Loading CIFAR-10...")
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                               (0.2023, 0.1994, 0.2010))
        ])
        
        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
        
        return self._create_loaders(train_dataset, test_dataset)
    
    def _create_loaders(self, train_dataset, test_dataset):
        """Create train, validation, and test loaders."""
        # Split training data
        val_size = int(len(train_dataset) * self.val_split)
        train_size = len(train_dataset) - val_size
        
        train_subset, val_subset = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loader = DataLoader(train_subset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=2)
        val_loader = DataLoader(val_subset, batch_size=self.batch_size,
                               shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size,
                                shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    def _get_num_classes(self):
        """Get number of classes in dataset."""
        if self.dataset_name in ['mnist', 'fashion_mnist', 'cifar10']:
            return 10
        return None
    
    def _get_input_shape(self):
        """Get input shape of dataset."""
        # Get a sample batch
        sample_batch = next(iter(self.train_loader))[0]
        return sample_batch.shape[1:]  # Remove batch dimension
    
    def get_info(self):
        """Return dataset information."""
        return {
            'name': self.dataset_name,
            'num_classes': self.num_classes,
            'input_shape': self.input_shape,
            'train_samples': len(self.train_loader.dataset),
            'val_samples': len(self.val_loader.dataset),
            'test_samples': len(self.test_loader.dataset),
            'batch_size': self.batch_size
        }


# =============================================================================
# PART 2: Model Factory
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: Building a Model Factory")
print("=" * 80)

class ModelFactory:
    """
    Factory for creating different model architectures.
    """
    
    @staticmethod
    def create_model(model_type, input_shape, num_classes, **kwargs):
        """
        Create a model based on type.
        
        Args:
            model_type: 'simple', 'medium', or 'deep'
            input_shape: Input tensor shape (C, H, W)
            num_classes: Number of output classes
            **kwargs: Additional model parameters
        
        Returns:
            PyTorch model
        """
        if model_type == 'simple':
            return ModelFactory._simple_model(input_shape, num_classes, **kwargs)
        elif model_type == 'medium':
            return ModelFactory._medium_model(input_shape, num_classes, **kwargs)
        elif model_type == 'deep':
            return ModelFactory._deep_model(input_shape, num_classes, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _simple_model(input_shape, num_classes, dropout=0.2):
        """Simple 2-layer fully connected network."""
        input_size = int(np.prod(input_shape))
        
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(input_size, 128)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout)
                self.fc2 = nn.Linear(128, num_classes)
            
            def forward(self, x):
                x = self.flatten(x)
                x = self.fc1(x)
                x = self.relu1(x)
                x = self.dropout1(x)
                x = self.fc2(x)
                return x
        
        return SimpleModel()
    
    @staticmethod
    def _medium_model(input_shape, num_classes, dropout=0.3):
        """Medium 3-layer fully connected network with batch norm."""
        input_size = int(np.prod(input_shape))
        
        class MediumModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(input_size, 256)
                self.bn1 = nn.BatchNorm1d(256)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout)
                
                self.fc2 = nn.Linear(256, 128)
                self.bn2 = nn.BatchNorm1d(128)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout)
                
                self.fc3 = nn.Linear(128, num_classes)
            
            def forward(self, x):
                x = self.flatten(x)
                x = self.fc1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.dropout1(x)
                
                x = self.fc2(x)
                x = self.bn2(x)
                x = self.relu2(x)
                x = self.dropout2(x)
                
                x = self.fc3(x)
                return x
        
        return MediumModel()
    
    @staticmethod
    def _deep_model(input_shape, num_classes, dropout=0.4):
        """Deep 4-layer fully connected network."""
        input_size = int(np.prod(input_shape))
        
        class DeepModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                
                self.fc1 = nn.Linear(input_size, 512)
                self.bn1 = nn.BatchNorm1d(512)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout)
                
                self.fc2 = nn.Linear(512, 256)
                self.bn2 = nn.BatchNorm1d(256)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout)
                
                self.fc3 = nn.Linear(256, 128)
                self.bn3 = nn.BatchNorm1d(128)
                self.relu3 = nn.ReLU()
                self.dropout3 = nn.Dropout(dropout)
                
                self.fc4 = nn.Linear(128, num_classes)
            
            def forward(self, x):
                x = self.flatten(x)
                
                x = self.fc1(x)
                x = self.bn1(x)
                x = self.relu1(x)
                x = self.dropout1(x)
                
                x = self.fc2(x)
                x = self.bn2(x)
                x = self.relu2(x)
                x = self.dropout2(x)
                
                x = self.fc3(x)
                x = self.bn3(x)
                x = self.relu3(x)
                x = self.dropout3(x)
                
                x = self.fc4(x)
                return x
        
        return DeepModel()


# =============================================================================
# PART 3: Comprehensive Trainer
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: Building a Comprehensive Trainer")
print("=" * 80)

class Trainer:
    """
    Comprehensive training pipeline with experiment tracking.
    """
    
    def __init__(self, model, train_loader, val_loader, test_loader,
                 criterion, optimizer, device, experiment_name="experiment"):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_loader, val_loader, test_loader: Data loaders
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on
            experiment_name: Name for saving results
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.experiment_name = experiment_name
        
        # Initialize tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        self.best_val_acc = 0.0
        self.best_model_state = None
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        avg_loss = running_loss / len(self.train_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = running_loss / len(self.val_loader)
        accuracy = correct / total
        return avg_loss, accuracy
    
    def test(self):
        """Test the model."""
        self.model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = correct / total
        return accuracy, np.array(all_predictions), np.array(all_targets)
    
    def train(self, num_epochs, scheduler=None, early_stopping_patience=None):
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            scheduler: Learning rate scheduler (optional)
            early_stopping_patience: Patience for early stopping (optional)
        
        Returns:
            Dictionary with training results
        """
        print(f"\n{'='*80}")
        print(f"Training: {self.experiment_name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Track metrics
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Track learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}]")
                print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
                print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")
                print(f"  LR: {current_lr:.6f}")
            
            # Early stopping
            if early_stopping_patience and patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        # Test final model
        test_acc, predictions, targets = self.test()
        
        training_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"Training Complete!")
        print(f"  Time: {training_time:.2f}s")
        print(f"  Best Val Acc: {self.best_val_acc:.4f}")
        print(f"  Test Acc: {test_acc:.4f}")
        print(f"{'='*80}\n")
        
        return {
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'test_acc': test_acc,
            'predictions': predictions,
            'targets': targets,
            'training_time': training_time,
            'experiment_name': self.experiment_name
        }


# =============================================================================
# PART 4: Experiment Runner
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: Running Comprehensive Experiments")
print("=" * 80)

class ExperimentRunner:
    """
    Run and compare multiple experiments.
    """
    
    def __init__(self):
        self.results = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def run_experiment(self, dataset_name, model_type, num_epochs=20,
                      lr=0.001, use_scheduler=False):
        """
        Run a single experiment.
        
        Args:
            dataset_name: Name of dataset
            model_type: Type of model
            num_epochs: Number of training epochs
            lr: Learning rate
            use_scheduler: Whether to use LR scheduling
        """
        experiment_name = f"{dataset_name}_{model_type}_lr{lr}"
        print(f"\n{'='*80}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*80}")
        
        # Load dataset
        dm = DatasetManager(dataset_name, batch_size=128)
        info = dm.get_info()
        print(f"\nDataset: {info['name']}")
        print(f"  Classes: {info['num_classes']}")
        print(f"  Input shape: {info['input_shape']}")
        print(f"  Train samples: {info['train_samples']}")
        
        # Create model
        model = ModelFactory.create_model(
            model_type,
            info['input_shape'],
            info['num_classes']
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: {model_type}")
        print(f"  Total parameters: {total_params:,}")
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        scheduler = None
        if use_scheduler:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        # Train
        trainer = Trainer(
            model, dm.train_loader, dm.val_loader, dm.test_loader,
            criterion, optimizer, self.device, experiment_name
        )
        
        result = trainer.train(
            num_epochs=num_epochs,
            scheduler=scheduler,
            early_stopping_patience=10
        )
        
        # Store results
        result['dataset'] = dataset_name
        result['model_type'] = model_type
        result['num_params'] = total_params
        result['learning_rate'] = lr
        result['use_scheduler'] = use_scheduler
        
        self.results.append(result)
        
        return result
    
    def generate_report(self):
        """Generate a comprehensive comparison report."""
        print(f"\n{'='*80}")
        print("EXPERIMENT COMPARISON REPORT")
        print(f"{'='*80}\n")
        
        # Sort by test accuracy
        sorted_results = sorted(self.results, key=lambda x: x['test_acc'], reverse=True)
        
        print(f"{'Rank':<6} {'Experiment':<35} {'Val Acc':<10} {'Test Acc':<10} {'Time(s)':<10}")
        print("-" * 80)
        
        for i, result in enumerate(sorted_results, 1):
            name = result['experiment_name']
            val_acc = result['best_val_acc']
            test_acc = result['test_acc']
            time_taken = result['training_time']
            
            print(f"{i:<6} {name:<35} {val_acc:.4f}{'':>4} {test_acc:.4f}{'':>4} {time_taken:.2f}")
        
        # Best per dataset
        print(f"\n{'='*80}")
        print("BEST MODEL PER DATASET")
        print(f"{'='*80}\n")
        
        datasets = set(r['dataset'] for r in self.results)
        for dataset in datasets:
            dataset_results = [r for r in self.results if r['dataset'] == dataset]
            best = max(dataset_results, key=lambda x: x['test_acc'])
            
            print(f"{dataset}:")
            print(f"  Best model: {best['model_type']}")
            print(f"  Test accuracy: {best['test_acc']:.4f}")
            print(f"  Parameters: {best['num_params']:,}")
            print()
        
        return sorted_results


# =============================================================================
# PART 5: Run Experiments
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: Running Multiple Experiments")
print("=" * 80)

# Create experiment runner
runner = ExperimentRunner()

# Define experiments to run (limited for demonstration)
experiments = [
    # Dataset, Model Type, Epochs, LR, Use Scheduler
    ('mnist', 'simple', 15, 0.001, False),
    ('mnist', 'medium', 15, 0.001, True),
    ('fashion_mnist', 'simple', 15, 0.001, False),
    ('fashion_mnist', 'medium', 15, 0.001, True),
]

print("\nRunning experiments...")
print("Note: Running limited experiments for demonstration.")
print("For full comparison, uncomment additional experiments below.\n")

# Run experiments
for dataset, model_type, epochs, lr, use_sched in experiments:
    try:
        runner.run_experiment(dataset, model_type, epochs, lr, use_sched)
    except Exception as e:
        print(f"Error in experiment: {e}")
        continue

# For more comprehensive testing, uncomment:
# experiments_full = [
#     ('mnist', 'simple', 20, 0.001, False),
#     ('mnist', 'medium', 20, 0.001, True),
#     ('mnist', 'deep', 20, 0.001, True),
#     ('fashion_mnist', 'simple', 20, 0.001, False),
#     ('fashion_mnist', 'medium', 20, 0.001, True),
#     ('fashion_mnist', 'deep', 20, 0.001, True),
#     ('cifar10', 'simple', 30, 0.001, False),
#     ('cifar10', 'medium', 30, 0.001, True),
#     ('cifar10', 'deep', 30, 0.001, True),
# ]

# Generate report
if runner.results:
    final_report = runner.generate_report()


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY - What You Accomplished")
print("=" * 80)

print("""
✅ Built a comprehensive dataset manager supporting multiple datasets
✅ Created a model factory for different architectures
✅ Implemented a full-featured training pipeline
✅ Tracked experiments with detailed metrics
✅ Compared multiple models and datasets systematically
✅ Generated performance comparison reports

Project Structure:
------------------
1. DatasetManager
   - Unified interface for multiple datasets
   - Automatic train/val/test splitting
   - Configurable batch sizes and preprocessing

2. ModelFactory
   - Simple, Medium, Deep architectures
   - Flexible parameter configuration
   - Easy to extend with new models

3. Trainer
   - Complete training loop
   - Learning rate scheduling
   - Early stopping
   - Best model tracking
   - Comprehensive history

4. ExperimentRunner
   - Run multiple experiments
   - Automatic comparison
   - Report generation
   - Results tracking

Key Takeaways:
--------------
• Modular design enables easy experimentation
• Systematic comparison reveals best approaches
• Experiment tracking is crucial for reproducibility
• Different datasets need different architectures
• Hyperparameters significantly impact performance

Professional Tips:
------------------
1. Always track experiments systematically
2. Use validation set for model selection
3. Report results on held-out test set
4. Compare models fairly (same data, device, seeds)
5. Document all hyperparameters and configurations

Next Steps:
-----------
→ Extend with more datasets (CIFAR-100, ImageNet, custom)
→ Add convolutional architectures
→ Implement more advanced techniques (mixup, cutout)
→ Create visualizations and tensorboard logging
→ Deploy best models for inference

🎉 Congratulations! You've completed the entire tutorial series!

You now have the skills to:
• Understand softmax regression from theory to practice
• Build and train deep learning classifiers
• Apply advanced techniques and best practices
• Run systematic experiments and comparisons
• Create production-ready training pipelines

Keep learning and experimenting! 🚀
""")
