"""
Module 33.7: Comprehensive Model Comparison (ADVANCED)

This module provides systematic comparison of all major CNN architectures covered in this course.
Students will learn how to evaluate models across multiple dimensions: accuracy, efficiency,
computational cost, and practical deployment considerations.

Key Concepts:
1. Multi-dimensional model evaluation
2. Accuracy vs efficiency trade-offs
3. FLOPs and parameter counting
4. Inference time measurement
5. Model selection for different scenarios

Learning Objectives:
- Understand trade-offs between different architectures
- Learn to select models based on deployment constraints
- Analyze computational complexity
- Interpret performance metrics holistically
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import pandas as pd
from collections import OrderedDict

# Import models from previous modules
# (In practice, these would be imported from separate files)


# ==============================================================================
# PART 1: MODEL COMPLEXITY ANALYSIS
# ==============================================================================

def count_parameters(model):
    """
    Count total and trainable parameters in a model.
    
    Parameters are the weights and biases that are learned during training.
    More parameters generally mean more capacity but also more memory and computation.
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def count_flops(model, input_size=(1, 3, 32, 32)):
    """
    Estimate FLOPs (Floating Point Operations) for a single forward pass.
    
    FLOPs measure computational cost. Lower FLOPs mean faster inference
    and lower energy consumption.
    
    This is a simplified estimation focusing on conv and linear layers.
    
    Args:
        model: PyTorch model
        input_size: Input tensor shape (batch, channels, height, width)
        
    Returns:
        int: Estimated FLOPs
    """
    def conv_flops(layer, input_shape):
        """Calculate FLOPs for convolutional layer."""
        batch, in_c, in_h, in_w = input_shape
        out_c = layer.out_channels
        kernel_ops = layer.kernel_size[0] * layer.kernel_size[1] * in_c
        out_h = (in_h + 2 * layer.padding[0] - layer.kernel_size[0]) // layer.stride[0] + 1
        out_w = (in_w + 2 * layer.padding[1] - layer.kernel_size[1]) // layer.stride[1] + 1
        
        flops = batch * out_c * out_h * out_w * kernel_ops
        return flops, (batch, out_c, out_h, out_w)
    
    def linear_flops(layer, input_shape):
        """Calculate FLOPs for linear layer."""
        batch = input_shape[0]
        in_features = layer.in_features
        out_features = layer.out_features
        flops = batch * in_features * out_features
        return flops, (batch, out_features)
    
    model.eval()
    total_flops = 0
    current_shape = input_size
    
    def count_layer(layer):
        nonlocal total_flops, current_shape
        if isinstance(layer, nn.Conv2d):
            flops, current_shape = conv_flops(layer, current_shape)
            total_flops += flops
        elif isinstance(layer, nn.Linear):
            if len(current_shape) > 2:
                # Flatten for linear layer
                current_shape = (current_shape[0], np.prod(current_shape[1:]))
            flops, current_shape = linear_flops(layer, current_shape)
            total_flops += flops
        elif isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            # Approximate shape change for pooling
            if isinstance(layer, nn.AdaptiveAvgPool2d):
                current_shape = (current_shape[0], current_shape[1], 
                               layer.output_size[0], layer.output_size[1])
            else:
                h_out = (current_shape[2] - layer.kernel_size) // layer.stride + 1
                w_out = (current_shape[3] - layer.kernel_size) // layer.stride + 1
                current_shape = (current_shape[0], current_shape[1], h_out, w_out)
    
    # Recursively count FLOPs
    for module in model.modules():
        count_layer(module)
    
    return total_flops


def measure_inference_time(model, input_size=(1, 3, 32, 32), 
                          device='cpu', num_iterations=100):
    """
    Measure average inference time.
    
    Actual inference time depends on hardware, batch size, and implementation.
    This provides a relative comparison between models.
    
    Args:
        model: PyTorch model
        input_size: Input tensor shape
        device: Device to run on ('cpu' or 'cuda')
        num_iterations: Number of forward passes to average
        
    Returns:
        float: Average inference time in milliseconds
    """
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Measure time
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations * 1000  # Convert to ms
    return avg_time


def get_model_size_mb(model):
    """
    Calculate model size in megabytes.
    
    This is the amount of storage needed to save the model to disk
    and the RAM needed to load it.
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Model size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    size_mb = (param_size + buffer_size) / (1024 ** 2)
    return size_mb


# ==============================================================================
# PART 2: TRAINING COMPARISON
# ==============================================================================

class ModelTrainer:
    """
    Unified trainer for comparing multiple models.
    
    Provides consistent training setup and metrics collection for fair comparison.
    """
    
    def __init__(self, model, model_name, device='cuda'):
        """
        Args:
            model: PyTorch model
            model_name: String identifier for the model
            device: Device to train on
        """
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epoch_time': []
        }
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch and return metrics."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        for inputs, targets in tqdm(train_loader, desc=f'{self.model_name} Training'):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy, epoch_time
    
    def evaluate(self, test_loader, criterion):
        """Evaluate on test set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = running_loss / len(test_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, test_loader, num_epochs, 
             learning_rate=0.1, weight_decay=5e-4):
        """
        Train model for specified number of epochs.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization factor
            
        Returns:
            dict: Training history
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate,
                            momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        
        print(f'\n{"="*60}')
        print(f'Training {self.model_name}')
        print(f'{"="*60}')
        
        best_acc = 0.0
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Train
            train_loss, train_acc, epoch_time = self.train_epoch(
                train_loader, criterion, optimizer
            )
            
            # Evaluate
            test_loss, test_acc = self.evaluate(test_loader, criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['epoch_time'].append(epoch_time)
            
            # Print results
            print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%')
            print(f'Epoch Time: {epoch_time:.2f}s')
            
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), f'{self.model_name}_best.pth')
                print(f'✓ Best model saved: {best_acc:.2f}%')
        
        print(f'\n{self.model_name} training completed!')
        print(f'Best test accuracy: {best_acc:.2f}%')
        
        return self.history


# ==============================================================================
# PART 3: COMPREHENSIVE COMPARISON
# ==============================================================================

class ModelComparison:
    """
    Comprehensive comparison framework for CNN architectures.
    
    Compares models across:
    - Accuracy (train and test)
    - Parameters and model size
    - FLOPs (computational cost)
    - Inference time
    - Training time
    """
    
    def __init__(self):
        """Initialize comparison framework."""
        self.models = OrderedDict()
        self.results = {}
    
    def add_model(self, model, name):
        """
        Add a model to comparison.
        
        Args:
            model: PyTorch model
            name: String identifier
        """
        self.models[name] = model
        print(f'Added model: {name}')
    
    def analyze_complexity(self, input_size=(1, 3, 32, 32)):
        """
        Analyze computational complexity of all models.
        
        Args:
            input_size: Input tensor shape
            
        Returns:
            pd.DataFrame: Complexity metrics for each model
        """
        complexity_data = []
        
        print('\nAnalyzing model complexity...')
        for name, model in self.models.items():
            # Count parameters
            total_params, trainable_params = count_parameters(model)
            
            # Estimate FLOPs
            flops = count_flops(model, input_size)
            
            # Measure model size
            size_mb = get_model_size_mb(model)
            
            complexity_data.append({
                'Model': name,
                'Parameters (M)': total_params / 1e6,
                'Trainable (M)': trainable_params / 1e6,
                'FLOPs (M)': flops / 1e6,
                'Size (MB)': size_mb
            })
            
            print(f'{name:20s}: {total_params/1e6:6.2f}M params, '
                  f'{flops/1e6:8.1f}M FLOPs, {size_mb:6.2f} MB')
        
        df = pd.DataFrame(complexity_data)
        return df
    
    def measure_inference_speed(self, input_size=(1, 3, 32, 32), 
                               device='cpu', num_iterations=100):
        """
        Measure inference time for all models.
        
        Args:
            input_size: Input tensor shape
            device: Device to benchmark on
            num_iterations: Number of forward passes
            
        Returns:
            pd.DataFrame: Inference times
        """
        speed_data = []
        
        print(f'\nMeasuring inference speed on {device}...')
        for name, model in self.models.items():
            avg_time = measure_inference_time(
                model, input_size, device, num_iterations
            )
            speed_data.append({
                'Model': name,
                'Inference Time (ms)': avg_time,
                'FPS': 1000 / avg_time
            })
            print(f'{name:20s}: {avg_time:6.2f} ms/image ({1000/avg_time:6.1f} FPS)')
        
        df = pd.DataFrame(speed_data)
        return df
    
    def compare_training(self, train_loader, test_loader, num_epochs=50, device='cuda'):
        """
        Train and compare all models.
        
        Args:
            train_loader: Training data loader
            test_loader: Test data loader
            num_epochs: Number of epochs to train
            device: Device to train on
            
        Returns:
            dict: Training results for each model
        """
        training_results = {}
        
        for name, model in self.models.items():
            trainer = ModelTrainer(model, name, device)
            history = trainer.train(train_loader, test_loader, num_epochs)
            training_results[name] = {
                'history': history,
                'best_acc': max(history['test_acc']),
                'final_acc': history['test_acc'][-1],
                'total_time': sum(history['epoch_time'])
            }
        
        self.results = training_results
        return training_results
    
    def generate_comparison_report(self, save_path='model_comparison_report.png'):
        """
        Generate comprehensive comparison visualization.
        
        Creates multi-panel figure showing:
        - Accuracy comparison
        - Parameter vs accuracy scatter
        - FLOPs vs accuracy scatter  
        - Training curves
        """
        if not self.results:
            print('No training results available. Run compare_training() first.')
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Accuracy comparison bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        models = list(self.results.keys())
        best_accs = [self.results[m]['best_acc'] for m in models]
        colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
        ax1.barh(models, best_accs, color=colors)
        ax1.set_xlabel('Test Accuracy (%)')
        ax1.set_title('Best Test Accuracy Comparison')
        ax1.set_xlim([min(best_accs)-2, 100])
        for i, v in enumerate(best_accs):
            ax1.text(v+0.5, i, f'{v:.2f}%', va='center')
        
        # 2. Training time comparison
        ax2 = fig.add_subplot(gs[0, 1])
        train_times = [self.results[m]['total_time']/3600 for m in models]
        ax2.barh(models, train_times, color=colors)
        ax2.set_xlabel('Total Training Time (hours)')
        ax2.set_title('Training Time Comparison')
        for i, v in enumerate(train_times):
            ax2.text(v+0.05, i, f'{v:.2f}h', va='center')
        
        # 3. Accuracy vs Parameters
        ax3 = fig.add_subplot(gs[1, 0])
        # Would need complexity data here - placeholder
        ax3.set_xlabel('Parameters (Millions)')
        ax3.set_ylabel('Test Accuracy (%)')
        ax3.set_title('Accuracy vs Model Size')
        ax3.grid(True, alpha=0.3)
        
        # 4. Training curves
        ax4 = fig.add_subplot(gs[1, 1])
        for name, results in self.results.items():
            ax4.plot(results['history']['test_acc'], label=name)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Test Accuracy (%)')
        ax4.set_title('Training Curves')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Loss curves
        ax5 = fig.add_subplot(gs[2, :])
        for name, results in self.results.items():
            ax5.plot(results['history']['train_loss'], '--', alpha=0.7, label=f'{name} (train)')
            ax5.plot(results['history']['test_loss'], '-', label=f'{name} (test)')
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Loss')
        ax5.set_title('Loss Curves')
        ax5.legend(ncol=2)
        ax5.grid(True, alpha=0.3)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'\nComparison report saved to {save_path}')
        plt.close()
    
    def print_summary_table(self):
        """Print summary comparison table."""
        if not self.results:
            print('No results available.')
            return
        
        print('\n' + '='*80)
        print('MODEL COMPARISON SUMMARY')
        print('='*80)
        
        headers = ['Model', 'Best Acc (%)', 'Final Acc (%)', 'Train Time (h)']
        print(f'{headers[0]:20s} {headers[1]:15s} {headers[2]:15s} {headers[3]:15s}')
        print('-'*80)
        
        for name, results in self.results.items():
            best = results['best_acc']
            final = results['final_acc']
            time_h = results['total_time'] / 3600
            print(f'{name:20s} {best:15.2f} {final:15.2f} {time_h:15.2f}')
        
        print('='*80)


# ==============================================================================
# PART 4: MAIN COMPARISON SCRIPT
# ==============================================================================

def main():
    """
    Main comparison script.
    
    Compares ResNet-18, VGG-16, GoogLeNet on CIFAR-10.
    """
    print('='*80)
    print('COMPREHENSIVE MODEL COMPARISON')
    print('='*80)
    
    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nUsing device: {device}')
    
    # Load data
    print('\nLoading CIFAR-10 dataset...')
    # (Data loading code here - same as previous modules)
    
    # Create comparison framework
    comparison = ModelComparison()
    
    # Add models (would use imports in practice)
    print('\nInitializing models...')
    # comparison.add_model(resnet18(num_classes=10), 'ResNet-18')
    # comparison.add_model(vgg16(num_classes=10, batch_norm=True), 'VGG-16-BN')
    # comparison.add_model(GoogLeNet(num_classes=10), 'GoogLeNet')
    
    # Analyze complexity
    print('\n' + '='*80)
    print('COMPLEXITY ANALYSIS')
    print('='*80)
    complexity_df = comparison.analyze_complexity()
    print('\n', complexity_df.to_string(index=False))
    
    # Measure inference speed
    print('\n' + '='*80)
    print('INFERENCE SPEED ANALYSIS')
    print('='*80)
    speed_df = comparison.measure_inference_speed(device='cpu')
    print('\n', speed_df.to_string(index=False))
    
    # Compare training
    print('\n' + '='*80)
    print('TRAINING COMPARISON')
    print('='*80)
    # results = comparison.compare_training(train_loader, test_loader, num_epochs=50, device=device)
    
    # Generate report
    # comparison.generate_comparison_report()
    # comparison.print_summary_table()


if __name__ == '__main__':
    main()


# ==============================================================================
# EXERCISES
# ==============================================================================

"""
ADVANCED EXERCISES:

1. Complete Comparison Study:
   - Implement and train all 6 architectures (ResNet, VGG, Inception, EfficientNet, MobileNet, DenseNet)
   - Record all metrics: accuracy, parameters, FLOPs, inference time, training time
   - Create comprehensive comparison report with visualizations

2. Trade-off Analysis:
   - Plot Pareto frontier: accuracy vs parameters
   - Identify models that are Pareto optimal
   - Discuss which models are best for different scenarios

3. Deployment Scenario Analysis:
   Create model recommendations for:
   - Mobile app (inference on phone)
   - Edge device (Raspberry Pi)
   - Cloud server (unlimited resources)
   - Real-time system (latency critical)
   - Batch processing (throughput critical)

4. Scaling Study:
   - Train models on different dataset sizes (10%, 50%, 100% of CIFAR-10)
   - Analyze which architectures are most data-efficient
   - Plot learning curves

5. Architecture Search:
   - Design a custom architecture that achieves best accuracy-efficiency trade-off
   - Justify each design choice based on insights from comparison study
   - Compare with existing architectures

EXPECTED INSIGHTS:
- No single "best" architecture - depends on constraints
- ResNet: Good balance of accuracy and efficiency
- VGG: Simple but parameter-heavy
- Inception: Parameter-efficient but complex
- EfficientNet: SOTA efficiency
- MobileNet: Best for mobile/edge
- DenseNet: Feature reuse, memory-intensive during training
"""
