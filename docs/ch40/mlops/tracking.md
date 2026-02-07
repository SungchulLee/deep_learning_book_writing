# Experiment Tracking

Experiment tracking is the systematic recording and organization of machine learning experiments to enable reproducibility, comparison, and informed decision-making. This section provides a comprehensive framework for understanding when and how to use different tracking approaches, along with best practices for organizing experiments at scale.

## Motivation

Machine learning development is inherently iterative. A typical project involves:

- Hundreds of training runs with varying hyperparameters
- Multiple model architectures under evaluation  
- Different data preprocessing pipelines
- Various regularization strategies

Without systematic tracking, valuable insights are lost, successful experiments cannot be reproduced, and teams waste effort re-running experiments. Experiment tracking transforms chaotic exploration into organized scientific inquiry.

## What to Track

### Essential Elements

Every experiment should record:

**Configuration**
```python
experiment_config = {
    # Model architecture
    'model_type': 'transformer',
    'hidden_size': 512,
    'num_layers': 6,
    'num_heads': 8,
    
    # Training hyperparameters
    'learning_rate': 1e-4,
    'batch_size': 32,
    'num_epochs': 100,
    'optimizer': 'adamw',
    'weight_decay': 0.01,
    
    # Data configuration
    'dataset': 'cifar10',
    'train_split': 0.8,
    'augmentation': ['random_crop', 'horizontal_flip'],
    
    # Regularization
    'dropout': 0.1,
    'label_smoothing': 0.1,
    
    # Compute environment
    'device': 'cuda',
    'num_gpus': 1,
    'precision': 'fp16'
}
```

**Metrics**
```python
# Training metrics (logged frequently)
training_metrics = {
    'train_loss': 0.234,
    'train_accuracy': 0.891,
    'gradient_norm': 1.23,
    'learning_rate': 1e-4  # If using scheduler
}

# Validation metrics (logged per epoch)
validation_metrics = {
    'val_loss': 0.312,
    'val_accuracy': 0.867,
    'val_f1': 0.854,
    'val_precision': 0.871,
    'val_recall': 0.838
}

# Test metrics (logged once at end)
test_metrics = {
    'test_loss': 0.298,
    'test_accuracy': 0.873,
    'inference_time_ms': 12.4
}
```

**Artifacts**
```python
artifacts = {
    # Model checkpoints
    'best_model': 'checkpoints/best_model.pth',
    'final_model': 'checkpoints/final_model.pth',
    
    # Visualizations
    'loss_curves': 'plots/loss_curves.png',
    'confusion_matrix': 'plots/confusion_matrix.png',
    'sample_predictions': 'plots/predictions.png',
    
    # Analysis files
    'classification_report': 'reports/classification_report.txt',
    'error_analysis': 'reports/error_analysis.csv',
    
    # Configuration
    'config_file': 'configs/experiment_config.yaml',
    'model_architecture': 'configs/model_architecture.txt'
}
```

**Environment Information**
```python
import torch
import platform
import sys

environment = {
    'python_version': sys.version,
    'pytorch_version': torch.__version__,
    'cuda_version': torch.version.cuda,
    'cudnn_version': torch.backends.cudnn.version(),
    'os': platform.system(),
    'hostname': platform.node(),
    'git_commit': get_git_commit_hash(),
    'requirements': read_requirements_file()
}
```

## Tracking Frequency

Different metrics require different logging frequencies:

### High Frequency (Every N Batches)

```python
LOG_INTERVAL = 100  # batches

for batch_idx, (data, target) in enumerate(train_loader):
    loss = train_step(model, data, target, optimizer)
    
    if batch_idx % LOG_INTERVAL == 0:
        tracker.log({
            'batch_loss': loss.item(),
            'learning_rate': optimizer.param_groups[0]['lr'],
            'gradient_norm': compute_gradient_norm(model)
        }, step=global_step)
```

### Medium Frequency (Every Epoch)

```python
for epoch in range(num_epochs):
    train_metrics = train_one_epoch(model, train_loader)
    val_metrics = evaluate(model, val_loader)
    
    tracker.log({
        'epoch': epoch,
        'train_loss': train_metrics['loss'],
        'train_accuracy': train_metrics['accuracy'],
        'val_loss': val_metrics['loss'],
        'val_accuracy': val_metrics['accuracy']
    }, step=epoch)
    
    # Log weight histograms (expensive, so per-epoch)
    for name, param in model.named_parameters():
        tracker.log_histogram(f'weights/{name}', param, epoch)
```

### Low Frequency (Once Per Experiment)

```python
# At experiment start
tracker.log_config(config)
tracker.log_environment(environment)
tracker.log_model_architecture(model)

# At experiment end
tracker.log_artifact('best_model.pth')
tracker.log_figure('confusion_matrix', confusion_matrix_fig)
tracker.log_table('error_analysis', error_df)
```

## Experiment Organization

### Naming Conventions

Establish consistent naming patterns:

```python
from datetime import datetime

def generate_experiment_name(config):
    """Generate descriptive experiment name."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Include key differentiating factors
    name_parts = [
        config['model_type'],
        f"lr{config['learning_rate']}",
        f"bs{config['batch_size']}",
        f"h{config['hidden_size']}",
        timestamp
    ]
    
    return '_'.join(name_parts)

# Example: "transformer_lr0.0001_bs32_h512_20240115_143022"
```

### Project Structure

```
project/
├── configs/
│   ├── base_config.yaml
│   ├── model_configs/
│   │   ├── small.yaml
│   │   ├── medium.yaml
│   │   └── large.yaml
│   └── experiment_configs/
│       ├── baseline.yaml
│       ├── ablation_dropout.yaml
│       └── ablation_lr.yaml
├── experiments/
│   ├── exp001_baseline/
│   │   ├── config.yaml
│   │   ├── logs/
│   │   ├── checkpoints/
│   │   └── results/
│   └── exp002_larger_model/
│       └── ...
├── src/
│   ├── models/
│   ├── data/
│   ├── training/
│   └── evaluation/
└── scripts/
    ├── train.py
    ├── evaluate.py
    └── analyze_experiments.py
```

### Tagging and Grouping

Use tags to enable filtering and comparison:

```python
# Experiment tags
tags = {
    'model_family': 'transformer',
    'task': 'classification',
    'dataset': 'imagenet',
    'experiment_type': 'ablation',
    'ablation_target': 'learning_rate',
    'team': 'vision',
    'priority': 'high'
}

tracker.set_tags(tags)
```

### Run Groups

Group related experiments:

```python
# Hyperparameter sweep
with tracker.start_group("lr_sweep_20240115"):
    for lr in [1e-5, 1e-4, 1e-3, 1e-2]:
        with tracker.start_run(name=f"lr_{lr}"):
            train_with_lr(lr)

# Architecture comparison
with tracker.start_group("architecture_comparison"):
    for arch in ['resnet18', 'resnet50', 'efficientnet']:
        with tracker.start_run(name=arch):
            train_architecture(arch)
```

## Comparison and Analysis

### Metric Comparison

```python
def compare_experiments(experiment_ids, metrics):
    """Compare metrics across experiments."""
    results = []
    
    for exp_id in experiment_ids:
        run = tracker.get_run(exp_id)
        result = {
            'experiment_id': exp_id,
            'name': run.name,
            **{m: run.metrics[m] for m in metrics}
        }
        results.append(result)
    
    df = pd.DataFrame(results)
    return df.sort_values(metrics[0], ascending=False)

# Usage
comparison = compare_experiments(
    experiment_ids=['exp001', 'exp002', 'exp003'],
    metrics=['val_accuracy', 'val_loss', 'train_time']
)
print(comparison)
```

### Statistical Analysis

```python
import numpy as np
from scipy import stats

def analyze_experiment_variance(experiment_ids, metric):
    """Analyze variance across repeated experiments."""
    values = []
    
    for exp_id in experiment_ids:
        run = tracker.get_run(exp_id)
        values.append(run.metrics[metric])
    
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'confidence_interval_95': stats.t.interval(
            0.95, len(values)-1, loc=np.mean(values), 
            scale=stats.sem(values)
        )
    }
```

### Visualization

```python
import matplotlib.pyplot as plt

def plot_experiment_comparison(experiments, metric):
    """Plot metric across experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training curves
    ax1 = axes[0]
    for exp in experiments:
        history = tracker.get_metric_history(exp['id'], metric)
        ax1.plot(history['step'], history['value'], label=exp['name'])
    ax1.set_xlabel('Step')
    ax1.set_ylabel(metric)
    ax1.set_title(f'{metric} over Training')
    ax1.legend()
    
    # Final values bar chart
    ax2 = axes[1]
    names = [exp['name'] for exp in experiments]
    values = [tracker.get_run(exp['id']).metrics[metric] for exp in experiments]
    ax2.bar(names, values)
    ax2.set_ylabel(metric)
    ax2.set_title(f'Final {metric}')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig
```

## Tool Selection Guide

### Decision Framework

| Criterion | TensorBoard | W&B | MLflow |
|-----------|-------------|-----|--------|
| Quick local experiments | ✓✓✓ | ✓ | ✓ |
| Team collaboration | ✗ | ✓✓✓ | ✓✓ |
| Model deployment | ✗ | ✓ | ✓✓✓ |
| Hyperparameter sweeps | ✗ | ✓✓✓ | ✓✓ |
| Self-hosted requirement | ✓✓✓ | ✗ | ✓✓✓ |
| Cloud-first workflow | ✗ | ✓✓✓ | ✓ |
| Zero setup | ✓✓✓ | ✓ | ✓ |

### Feature Comparison Matrix

| Feature | TensorBoard | W&B | MLflow |
|---------|-------------|-----|--------|
| **Setup Complexity** | Low | Medium | Medium |
| **Local Storage** | ✓ | ✓ | ✓ |
| **Cloud Storage** | ✗ | ✓ | ✓ (optional) |
| **Real-time Logging** | ✓ | ✓ | ✓ |
| **Metric Comparison** | ✓ | ✓ | ✓ |
| **Hyperparameter Sweeps** | ✗ | ✓ | ✓ |
| **Model Registry** | ✗ | ✓ | ✓ |
| **Model Serving** | ✗ | Limited | ✓ |
| **Collaboration** | Manual | Built-in | Self-hosted |
| **Artifact Versioning** | ✗ | ✓ | ✓ |
| **Code Versioning** | ✗ | ✓ | ✓ (Projects) |
| **Cost** | Free | Freemium | Free |
| **Offline Mode** | Native | ✓ | Native |

### Use Case Recommendations

**Research/Academic**

- Primary: TensorBoard (simple, local)
- For collaboration: W&B (free for academics)

**Startup/Small Team**

- Primary: W&B (quick setup, good free tier)
- Alternative: MLflow (self-hosted control)

**Enterprise**

- Primary: MLflow (open-source, self-hosted)
- Alternative: W&B Enterprise

**Production ML**

- Primary: MLflow (model registry, serving)
- Supplement with: TensorBoard (debugging)

## Unified Tracking Interface

Create an abstraction layer to switch between tools:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os

class ExperimentTracker(ABC):
    """Abstract base class for experiment tracking."""
    
    @abstractmethod
    def start_run(self, name: str, config: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        pass
    
    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        pass
    
    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        pass
    
    @abstractmethod
    def end_run(self) -> None:
        pass


class TensorBoardTracker(ExperimentTracker):
    """TensorBoard implementation."""
    
    def __init__(self, log_dir: str = 'runs'):
        self.log_dir = log_dir
        self.writer = None
    
    def start_run(self, name: str, config: Dict[str, Any]) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(f'{self.log_dir}/{name}')
        
        # Log config as text
        config_str = '\n'.join(f'{k}: {v}' for k, v in config.items())
        self.writer.add_text('config', config_str)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(name, value, step)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        params_str = '\n'.join(f'{k}: {v}' for k, v in params.items())
        self.writer.add_text('params', params_str)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        import shutil
        dest = os.path.join(self.writer.log_dir, artifact_path or os.path.basename(local_path))
        shutil.copy(local_path, dest)
    
    def end_run(self) -> None:
        if self.writer:
            self.writer.close()


class WandBTracker(ExperimentTracker):
    """Weights & Biases implementation."""
    
    def __init__(self, project: str, entity: Optional[str] = None):
        self.project = project
        self.entity = entity
    
    def start_run(self, name: str, config: Dict[str, Any]) -> None:
        import wandb
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=config
        )
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        import wandb
        if step is not None:
            metrics['_step'] = step
        wandb.log(metrics)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        import wandb
        wandb.config.update(params)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        import wandb
        artifact = wandb.Artifact(
            name=artifact_path or os.path.basename(local_path),
            type='file'
        )
        artifact.add_file(local_path)
        wandb.log_artifact(artifact)
    
    def end_run(self) -> None:
        import wandb
        wandb.finish()


class MLflowTracker(ExperimentTracker):
    """MLflow implementation."""
    
    def __init__(self, experiment_name: str, tracking_uri: Optional[str] = None):
        import mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
    
    def start_run(self, name: str, config: Dict[str, Any]) -> None:
        import mlflow
        mlflow.start_run(run_name=name)
        mlflow.log_params(config)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        import mlflow
        mlflow.log_metrics(metrics, step=step)
    
    def log_params(self, params: Dict[str, Any]) -> None:
        import mlflow
        mlflow.log_params(params)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        import mlflow
        mlflow.log_artifact(local_path, artifact_path)
    
    def end_run(self) -> None:
        import mlflow
        mlflow.end_run()


def get_tracker(tracker_type: str, **kwargs) -> ExperimentTracker:
    """Factory function to create appropriate tracker."""
    trackers = {
        'tensorboard': TensorBoardTracker,
        'wandb': WandBTracker,
        'mlflow': MLflowTracker
    }
    
    if tracker_type not in trackers:
        raise ValueError(f"Unknown tracker: {tracker_type}")
    
    return trackers[tracker_type](**kwargs)
```

### Usage Example

```python
# Configuration-driven tracker selection
config = {
    'tracker': 'wandb',  # or 'tensorboard', 'mlflow'
    'tracker_config': {
        'project': 'my-project',
        'entity': 'my-team'
    }
}

# Create tracker
tracker = get_tracker(config['tracker'], **config['tracker_config'])

# Use unified interface
experiment_config = {'learning_rate': 0.001, 'batch_size': 32}
tracker.start_run('experiment_001', experiment_config)

for epoch in range(10):
    loss = train_epoch(model, train_loader)
    tracker.log_metrics({'train_loss': loss}, step=epoch)

tracker.log_artifact('model.pth')
tracker.end_run()
```

## Best Practices Summary

### Do's

1. **Track everything reproducible**: All hyperparameters, random seeds, environment details
2. **Use meaningful names**: Descriptive experiment and run names
3. **Tag extensively**: Enable filtering and grouping
4. **Version data and code**: Link experiments to specific versions
5. **Document decisions**: Add notes explaining experiment rationale
6. **Set baselines**: Always have a reference point for comparison
7. **Automate tracking**: Integrate into training pipelines, not as afterthought

### Don'ts

1. **Don't over-log**: Logging too frequently creates bloat without value
2. **Don't ignore failures**: Failed experiments contain valuable information
3. **Don't skip validation metrics**: Training metrics alone are insufficient
4. **Don't hardcode paths**: Use configuration for flexibility
5. **Don't forget cleanup**: Archive or delete old experiments periodically
6. **Don't mix experiments**: Keep different projects in separate namespaces

### Checklist

Before starting an experiment:

- [ ] Configuration file created and version controlled
- [ ] Experiment name follows naming convention
- [ ] Appropriate tags defined
- [ ] Baseline experiment identified for comparison
- [ ] Data version documented
- [ ] Code committed to version control

During training:

- [ ] Metrics logged at appropriate frequency
- [ ] Checkpoints saved periodically
- [ ] Validation performed at regular intervals
- [ ] Learning rate and other dynamics tracked

After training:

- [ ] Final metrics logged
- [ ] Best model saved
- [ ] Visualizations generated
- [ ] Error analysis completed
- [ ] Documentation updated

## Summary

Effective experiment tracking transforms machine learning development from ad-hoc exploration into systematic scientific inquiry. Key principles:

1. **Comprehensiveness**: Track all factors that affect reproducibility
2. **Organization**: Use consistent naming, tagging, and grouping
3. **Analysis**: Compare experiments quantitatively and visualize results
4. **Tool selection**: Choose tools based on team needs and project requirements
5. **Automation**: Integrate tracking into workflows, not as manual overhead

The investment in proper experiment tracking pays dividends through improved reproducibility, faster iteration, and better collaboration across teams and time.


---

# TensorBoard

TensorBoard is TensorFlow's visualization toolkit that provides a suite of tools for visualizing machine learning experiments. Despite its origins in TensorFlow, TensorBoard integrates seamlessly with PyTorch through the `torch.utils.tensorboard` module, making it the de facto standard for experiment visualization in deep learning research and production.

## Motivation and Purpose

Training deep neural networks involves tracking numerous quantities: loss values, accuracy metrics, gradient magnitudes, weight distributions, and model architectures. Without proper visualization tools, understanding training dynamics becomes nearly impossible. TensorBoard addresses this by providing real-time visualization of training metrics, enabling practitioners to diagnose problems, compare experiments, and communicate results effectively.

The key advantages of TensorBoard include:

- **Real-time monitoring**: Watch training progress as it happens
- **Interactive exploration**: Zoom, pan, and filter visualizations
- **Experiment comparison**: Overlay multiple runs on the same plots
- **Rich media logging**: Support for scalars, images, histograms, graphs, and more
- **Lightweight integration**: Minimal code changes required

## Installation and Setup

TensorBoard requires the `tensorboard` package alongside PyTorch:

```bash
pip install tensorboard
```

For the complete environment with PyTorch:

```bash
pip install torch torchvision tensorboard matplotlib
```

Verify the installation:

```python
import torch
from torch.utils.tensorboard import SummaryWriter
print(f"PyTorch version: {torch.__version__}")
print("TensorBoard integration available")
```

## Core Concepts

### The SummaryWriter

The `SummaryWriter` class is the primary interface between PyTorch and TensorBoard. It creates log files that TensorBoard reads and visualizes.

```python
from torch.utils.tensorboard import SummaryWriter

# Create a writer with default log directory (runs/CURRENT_DATETIME_HOSTNAME)
writer = SummaryWriter()

# Create a writer with custom log directory
writer = SummaryWriter('runs/experiment_1')

# Create a writer with comment suffix
writer = SummaryWriter(comment='_lr_0.001_batch_64')
```

The writer creates a directory structure containing event files that store all logged data:

```
runs/
├── experiment_1/
│   └── events.out.tfevents.1234567890.hostname
├── experiment_2/
│   └── events.out.tfevents.1234567891.hostname
└── experiment_3/
    └── events.out.tfevents.1234567892.hostname
```

### Global Step

Most logging methods accept a `global_step` parameter that represents the x-axis value in TensorBoard plots. This is typically the iteration number or epoch number:

```python
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Training step
        loss = train_step(model, data, target)
        
        # Global step combines epoch and batch information
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('Loss/train', loss, global_step)
```

## Logging Scalars

Scalar logging is the most common TensorBoard operation, used for tracking metrics like loss, accuracy, and learning rate.

### Basic Scalar Logging

```python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/scalar_example')

# Simulating a training loop
for step in range(1000):
    # Simulated loss values
    train_loss = 1.0 / (step + 1) + 0.1 * torch.rand(1).item()
    val_loss = 1.2 / (step + 1) + 0.15 * torch.rand(1).item()
    
    writer.add_scalar('Loss/train', train_loss, step)
    writer.add_scalar('Loss/validation', val_loss, step)

writer.close()
```

### Grouping Scalars

TensorBoard organizes scalars by their tag names. Using forward slashes creates hierarchical groupings:

```python
# These appear under the "Loss" group in TensorBoard
writer.add_scalar('Loss/train', train_loss, step)
writer.add_scalar('Loss/validation', val_loss, step)
writer.add_scalar('Loss/test', test_loss, step)

# These appear under the "Accuracy" group
writer.add_scalar('Accuracy/train', train_acc, step)
writer.add_scalar('Accuracy/validation', val_acc, step)

# These appear under the "Metrics" group
writer.add_scalar('Metrics/learning_rate', lr, step)
writer.add_scalar('Metrics/gradient_norm', grad_norm, step)
```

### Multiple Scalars in One Call

For logging multiple related scalars simultaneously:

```python
writer.add_scalars('Loss', {
    'train': train_loss,
    'validation': val_loss,
    'test': test_loss
}, step)
```

This creates overlapping curves in a single plot, useful for direct comparison.

## Logging Images

TensorBoard can display images, which is invaluable for tasks like image classification, generation, and segmentation.

### Single Images

```python
import torchvision

# Load sample data
transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
image, label = dataset[0]

# Log single image (expects [C, H, W] format)
writer.add_image('Sample/single', image, 0)
```

### Image Grids

For visualizing multiple images:

```python
from torch.utils.data import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)
images, labels = next(iter(loader))

# Create grid of images
img_grid = torchvision.utils.make_grid(images, nrow=8, normalize=True)
writer.add_image('Sample/batch_grid', img_grid, 0)
```

### Images During Training

Log generated images or predictions during training:

```python
def log_predictions(model, data_loader, writer, epoch):
    model.eval()
    images, labels = next(iter(data_loader))
    
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs, 1)
    
    # Create annotated grid (would require custom visualization)
    img_grid = torchvision.utils.make_grid(images[:16], nrow=4)
    writer.add_image('Predictions', img_grid, epoch)
    model.train()
```

## Logging Histograms

Histograms visualize the distribution of tensors over time, essential for monitoring weight and gradient distributions.

### Weight Histograms

```python
def log_weights(model, writer, epoch):
    for name, param in model.named_parameters():
        writer.add_histogram(f'Weights/{name}', param, epoch)
        
        if param.grad is not None:
            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
```

### Activation Histograms

Monitor layer activations to detect issues like vanishing or exploding activations:

```python
class ModelWithLogging(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.activations = {}
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        self.activations['fc1'] = x.detach()
        x = self.fc2(x)
        self.activations['fc2'] = x.detach()
        return x

def log_activations(model, writer, step):
    for name, activation in model.activations.items():
        writer.add_histogram(f'Activations/{name}', activation, step)
```

## Logging Model Graphs

Visualize the computational graph of your model:

```python
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size=784, hidden_size=500, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNet()
dummy_input = torch.randn(1, 784)

writer.add_graph(model, dummy_input)
writer.close()
```

The graph visualization shows tensor shapes, operations, and data flow through the network.

## Logging Precision-Recall Curves

For classification tasks, precision-recall curves provide insight into model performance across different thresholds:

```python
import torch.nn.functional as F

def log_pr_curves(model, test_loader, writer, epoch, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28).to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu())
            all_labels.append(labels)
    
    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    
    # Log PR curve for each class
    num_classes = all_probs.shape[1]
    for class_idx in range(num_classes):
        binary_labels = (all_labels == class_idx)
        class_probs = all_probs[:, class_idx]
        
        writer.add_pr_curve(
            f'PR_Curve/class_{class_idx}',
            binary_labels,
            class_probs,
            global_step=epoch
        )
    
    model.train()
```

## Logging Hyperparameters

TensorBoard's HPARAMS plugin enables systematic hyperparameter comparison:

```python
def train_with_hparams(hparams):
    """Train model with given hyperparameters and log results."""
    
    # Unpack hyperparameters
    lr = hparams['learning_rate']
    batch_size = hparams['batch_size']
    hidden_size = hparams['hidden_size']
    
    # Create unique log directory
    log_dir = f'runs/hparam_lr{lr}_bs{batch_size}_hs{hidden_size}'
    writer = SummaryWriter(log_dir)
    
    # Initialize model and train
    model = NeuralNet(hidden_size=hidden_size)
    # ... training code ...
    
    # Log final metrics
    final_accuracy = evaluate(model, test_loader)
    final_loss = compute_loss(model, test_loader)
    
    # Log hyperparameters with their associated metrics
    writer.add_hparams(
        hparam_dict={
            'learning_rate': lr,
            'batch_size': batch_size,
            'hidden_size': hidden_size,
        },
        metric_dict={
            'accuracy': final_accuracy,
            'loss': final_loss,
        }
    )
    
    writer.close()
    return final_accuracy

# Run hyperparameter search
for lr in [0.001, 0.01, 0.1]:
    for batch_size in [32, 64, 128]:
        for hidden_size in [128, 256, 512]:
            train_with_hparams({
                'learning_rate': lr,
                'batch_size': batch_size,
                'hidden_size': hidden_size
            })
```

## Complete Training Example

The following example demonstrates comprehensive TensorBoard integration:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
config = {
    'input_size': 784,
    'hidden_size': 500,
    'num_classes': 10,
    'num_epochs': 5,
    'batch_size': 64,
    'learning_rate': 0.001,
    'log_interval': 100
}

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def train():
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/mnist_training')
    
    # Load data
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Log sample images
    examples = next(iter(train_loader))
    img_grid = torchvision.utils.make_grid(examples[0][:32])
    writer.add_image('MNIST_samples', img_grid)
    
    # Initialize model
    model = NeuralNet(
        config['input_size'],
        config['hidden_size'],
        config['num_classes']
    ).to(device)
    
    # Log model graph
    dummy_input = torch.randn(1, config['input_size']).to(device)
    writer.add_graph(model, dummy_input)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    global_step = 0
    for epoch in range(config['num_epochs']):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.reshape(-1, config['input_size']).to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()
            
            # Log at intervals
            if (batch_idx + 1) % config['log_interval'] == 0:
                avg_loss = running_loss / config['log_interval']
                accuracy = running_correct / running_total
                
                writer.add_scalar('Training/Loss', avg_loss, global_step)
                writer.add_scalar('Training/Accuracy', accuracy, global_step)
                
                print(f'Epoch [{epoch+1}/{config["num_epochs"]}], '
                      f'Step [{batch_idx+1}/{len(train_loader)}], '
                      f'Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
                
                running_loss = 0.0
                running_correct = 0
                running_total = 0
            
            global_step += 1
        
        # Log weight histograms at end of each epoch
        for name, param in model.named_parameters():
            writer.add_histogram(f'Weights/{name}', param, epoch)
            if param.grad is not None:
                writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.reshape(-1, config['input_size']).to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                probs = F.softmax(outputs, dim=1)
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())
        
        test_loss /= len(test_loader)
        test_accuracy = correct / total
        
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_accuracy, epoch)
        
        # Log PR curves
        all_probs = torch.cat(all_probs)
        all_labels = torch.cat(all_labels)
        
        for i in range(config['num_classes']):
            binary_labels = (all_labels == i)
            writer.add_pr_curve(f'PR_Curve/class_{i}', binary_labels, all_probs[:, i], epoch)
        
        print(f'Epoch [{epoch+1}/{config["num_epochs"]}] - '
              f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    
    # Log hyperparameters with final metrics
    writer.add_hparams(
        {k: v for k, v in config.items() if isinstance(v, (int, float, str))},
        {'final_accuracy': test_accuracy, 'final_loss': test_loss}
    )
    
    writer.close()
    print(f'\nTraining complete. View results: tensorboard --logdir=runs')

if __name__ == '__main__':
    train()
```

## Launching TensorBoard

After running training scripts that log to TensorBoard:

```bash
# Basic usage
tensorboard --logdir=runs

# Specify port
tensorboard --logdir=runs --port=6007

# Allow remote access
tensorboard --logdir=runs --host=0.0.0.0

# Compare multiple experiment directories
tensorboard --logdir=experiment1:./runs/exp1,experiment2:./runs/exp2
```

Access TensorBoard at `http://localhost:6006` (or the specified port).

## TensorBoard Interface Guide

### Scalars Tab

The Scalars tab displays line plots of logged scalar values. Key features:

- **Smoothing slider**: Applies exponential smoothing to reduce noise
- **Run selector**: Toggle visibility of different runs
- **Download**: Export data as CSV or plots as images
- **Relative time**: Toggle between wall time and step

### Images Tab

Displays logged images organized by tag and step. Navigate through different steps using the slider.

### Graphs Tab

Shows the computational graph of logged models. Click on nodes to see tensor shapes and operation details.

### Distributions and Histograms Tabs

Visualize tensor distributions over time. Distributions show a condensed view while Histograms show detailed 3D visualizations.

### HPARAMS Tab

Displays hyperparameter search results in a parallel coordinates plot and table view, enabling identification of optimal configurations.

### PR Curves Tab

Shows precision-recall curves for classification tasks, with threshold selection capabilities.

## Best Practices

### Logging Frequency

Avoid logging too frequently, which creates large log files and slows down training:

```python
# Good: Log every N steps
if step % 100 == 0:
    writer.add_scalar('loss', loss, step)

# Better: Log epoch-level summaries for histograms
if batch_idx == len(train_loader) - 1:  # End of epoch
    for name, param in model.named_parameters():
        writer.add_histogram(f'weights/{name}', param, epoch)
```

### Organizing Experiments

Use descriptive log directories:

```python
from datetime import datetime

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
experiment_name = f'lr{lr}_bs{batch_size}_{timestamp}'
writer = SummaryWriter(f'runs/{experiment_name}')
```

### Flushing Writers

Ensure data is written to disk:

```python
# Flush periodically during long training runs
if step % 1000 == 0:
    writer.flush()

# Always close the writer when done
writer.close()
```

### Cleaning Up Old Logs

TensorBoard logs accumulate. Periodically clean old experiments:

```bash
# Remove all logs
rm -rf runs/*

# Keep only recent experiments
find runs -type d -mtime +7 -exec rm -rf {} +
```

## Common Issues and Solutions

### TensorBoard Not Showing Data

1. Ensure training script completed without errors
2. Wait a few seconds for log synchronization
3. Refresh the browser
4. Check that the `--logdir` path matches where logs are written

### Large Log Files

Reduce logging frequency or be selective about what to log:

```python
# Only log heavy items (histograms, images) occasionally
if epoch % 5 == 0:
    writer.add_histogram('weights', model.fc1.weight, epoch)
```

### Port Already in Use

```bash
# Kill existing TensorBoard processes
pkill -f tensorboard

# Use a different port
tensorboard --logdir=runs --port=6007
```

### Remote Access

For accessing TensorBoard running on a remote server:

```bash
# On the server
tensorboard --logdir=runs --host=0.0.0.0 --port=6006

# On local machine (SSH tunnel)
ssh -L 6006:localhost:6006 user@server

# Then access http://localhost:6006 locally
```

## Integration with Configuration Management

Combine TensorBoard with configuration classes for reproducible experiments:

```python
from dataclasses import dataclass, asdict

@dataclass
class ExperimentConfig:
    learning_rate: float = 0.001
    batch_size: int = 64
    hidden_size: int = 500
    num_epochs: int = 10
    experiment_name: str = 'default'
    
    @property
    def log_dir(self):
        return f'runs/{self.experiment_name}_lr{self.learning_rate}_bs{self.batch_size}'

def train_experiment(config: ExperimentConfig):
    writer = SummaryWriter(config.log_dir)
    
    # Log configuration as text
    config_str = '\n'.join(f'{k}: {v}' for k, v in asdict(config).items())
    writer.add_text('Configuration', config_str)
    
    # Training code...
    
    writer.close()
```

## Summary

TensorBoard provides essential visualization capabilities for deep learning experiments. Key takeaways:

1. **SummaryWriter** is the central interface for logging
2. **Scalars** track metrics over time (loss, accuracy, learning rate)
3. **Images** visualize data samples and model outputs
4. **Histograms** monitor weight and gradient distributions
5. **Graphs** display model architecture
6. **PR Curves** analyze classification performance
7. **HPARAMS** enable systematic hyperparameter comparison

Effective use of TensorBoard transforms debugging from guesswork into systematic analysis, accelerating the development of high-performing models.


---

# Weights & Biases

Weights & Biases (W&B, or wandb) is a machine learning experiment tracking platform that provides comprehensive tools for logging, visualization, and collaboration. While TensorBoard excels at local visualization, W&B offers cloud-based storage, team collaboration features, and advanced capabilities like hyperparameter sweep orchestration and model versioning.

## Motivation

Modern machine learning development involves numerous experiments across different machines, team members, and time periods. Key challenges include:

- **Reproducibility**: Tracking exact configurations that produced specific results
- **Collaboration**: Sharing experiments with team members
- **Scalability**: Managing thousands of experiments across multiple projects
- **Automation**: Running systematic hyperparameter searches
- **Artifact management**: Versioning datasets and models

W&B addresses these challenges through a unified platform that integrates seamlessly with PyTorch workflows.

## Installation and Setup

Install the wandb package:

```bash
pip install wandb
```

Create a free account at [wandb.ai](https://wandb.ai) and authenticate:

```bash
wandb login
```

This stores an API key locally for future use. Alternatively, set the environment variable:

```bash
export WANDB_API_KEY=your_api_key_here
```

## Core Concepts

### Projects and Runs

W&B organizes experiments hierarchically:

- **Entity**: Your username or team name
- **Project**: A collection of related experiments (e.g., "mnist-classification")
- **Run**: A single training execution with its own configuration and metrics

```python
import wandb

# Initialize a new run
run = wandb.init(
    project="mnist-classification",
    entity="your-username",  # Optional if using default entity
    name="baseline-model",   # Optional human-readable name
    notes="First experiment with basic MLP",
    tags=["baseline", "mlp"]
)
```

### Configuration Tracking

W&B automatically tracks hyperparameters and configuration:

```python
config = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'hidden_size': 500,
    'num_epochs': 10,
    'optimizer': 'adam',
    'architecture': 'mlp'
}

run = wandb.init(project="mnist", config=config)

# Access config values (supports dot notation)
print(wandb.config.learning_rate)
print(wandb.config['batch_size'])

# Update config during training
wandb.config.update({'dropout': 0.5})
```

## Basic Logging

### Metrics

Log scalar metrics with `wandb.log()`:

```python
import wandb

wandb.init(project="example")

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    val_loss, val_acc = evaluate(model, val_loader)
    
    # Log multiple metrics at once
    wandb.log({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'learning_rate': optimizer.param_groups[0]['lr']
    })
```

### Step Control

W&B automatically increments a global step counter. For explicit control:

```python
# Log with explicit step
wandb.log({'loss': loss}, step=global_step)

# Commit determines when to increment the step
wandb.log({'train_loss': train_loss}, commit=False)
wandb.log({'train_acc': train_acc}, commit=True)  # Increments step
```

### Grouped Metrics

Create metric groups using forward slashes:

```python
wandb.log({
    'train/loss': train_loss,
    'train/accuracy': train_acc,
    'val/loss': val_loss,
    'val/accuracy': val_acc,
    'gradients/norm': grad_norm,
    'gradients/max': grad_max
})
```

## Media Logging

### Images

```python
import wandb
import torch
import torchvision

# Log individual images
image = train_dataset[0][0]  # [C, H, W] tensor
wandb.log({'sample_image': wandb.Image(image, caption='Sample digit')})

# Log batch of images
images, labels = next(iter(train_loader))
wandb.log({
    'batch_samples': [
        wandb.Image(img, caption=f'Label: {label}')
        for img, label in zip(images[:8], labels[:8])
    ]
})

# Log image with overlays (bounding boxes, masks)
wandb.log({
    'predictions': wandb.Image(
        image,
        boxes={
            'predictions': {
                'box_data': [
                    {'position': {'minX': 0.1, 'minY': 0.2, 'maxX': 0.5, 'maxY': 0.8},
                     'class_id': 1, 'scores': {'confidence': 0.95}}
                ],
                'class_labels': {1: 'digit'}
            }
        }
    )
})
```

### Histograms

```python
# Log weight distributions
for name, param in model.named_parameters():
    wandb.log({f'weights/{name}': wandb.Histogram(param.detach().cpu())})

# Log gradient distributions
for name, param in model.named_parameters():
    if param.grad is not None:
        wandb.log({f'gradients/{name}': wandb.Histogram(param.grad.detach().cpu())})
```

### Tables

W&B Tables enable rich data exploration:

```python
# Create a table for predictions
columns = ['image', 'prediction', 'ground_truth', 'confidence']
table = wandb.Table(columns=columns)

for images, labels in test_loader:
    outputs = model(images)
    probs = torch.softmax(outputs, dim=1)
    preds = outputs.argmax(dim=1)
    confidences = probs.max(dim=1).values
    
    for img, pred, label, conf in zip(images, preds, labels, confidences):
        table.add_data(
            wandb.Image(img),
            pred.item(),
            label.item(),
            conf.item()
        )
        
        if len(table.data) >= 100:  # Limit table size
            break
    if len(table.data) >= 100:
        break

wandb.log({'predictions_table': table})
```

### Plots

```python
import matplotlib.pyplot as plt

# Log matplotlib figures
fig, ax = plt.subplots()
ax.plot(train_losses, label='Train')
ax.plot(val_losses, label='Validation')
ax.legend()
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
wandb.log({'loss_curves': wandb.Image(fig)})
plt.close(fig)

# Log confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt='d', ax=ax)
wandb.log({'confusion_matrix': wandb.Image(fig)})
plt.close(fig)
```

### Custom Charts

W&B supports custom Vega-Lite charts:

```python
# Log data for custom visualization
data = [[x, y] for x, y in zip(epochs, accuracies)]
table = wandb.Table(data=data, columns=['epoch', 'accuracy'])

wandb.log({
    'custom_chart': wandb.plot.line(
        table, 'epoch', 'accuracy', title='Accuracy over Time'
    )
})

# Precision-Recall curve
wandb.log({
    'pr_curve': wandb.plot.pr_curve(
        ground_truth=all_labels,
        predictions=all_probs,
        labels=list(range(10))
    )
})

# ROC curve
wandb.log({
    'roc_curve': wandb.plot.roc_curve(
        ground_truth=all_labels,
        predictions=all_probs,
        labels=list(range(10))
    )
})
```

## Model Checkpointing and Artifacts

### Saving Models

```python
# Save model checkpoint
checkpoint_path = 'model_checkpoint.pth'
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, checkpoint_path)

# Log as artifact
artifact = wandb.Artifact(
    name='model-checkpoint',
    type='model',
    description='Best model checkpoint',
    metadata={'epoch': epoch, 'val_accuracy': val_acc}
)
artifact.add_file(checkpoint_path)
wandb.log_artifact(artifact)
```

### Loading Models

```python
# Download artifact from previous run
run = wandb.init(project="mnist")
artifact = run.use_artifact('model-checkpoint:latest')
artifact_dir = artifact.download()

# Load checkpoint
checkpoint = torch.load(f'{artifact_dir}/model_checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Dataset Versioning

```python
# Create dataset artifact
dataset_artifact = wandb.Artifact(
    name='mnist-processed',
    type='dataset',
    description='Preprocessed MNIST dataset'
)
dataset_artifact.add_dir('data/processed/')
wandb.log_artifact(dataset_artifact)

# Use dataset in training
dataset_artifact = run.use_artifact('mnist-processed:v2')
dataset_dir = dataset_artifact.download()
```

## Hyperparameter Sweeps

W&B Sweeps automate hyperparameter search:

### Defining a Sweep

```python
sweep_config = {
    'method': 'bayes',  # 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        },
        'batch_size': {
            'values': [32, 64, 128, 256]
        },
        'hidden_size': {
            'distribution': 'int_uniform',
            'min': 128,
            'max': 1024
        },
        'dropout': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.5
        },
        'optimizer': {
            'values': ['adam', 'sgd', 'rmsprop']
        }
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 3
    }
}

sweep_id = wandb.sweep(sweep_config, project="mnist-sweep")
```

### Training Function for Sweeps

```python
def train_sweep():
    # Initialize run (config comes from sweep)
    run = wandb.init()
    config = wandb.config
    
    # Build model with sweep parameters
    model = NeuralNet(
        input_size=784,
        hidden_size=config.hidden_size,
        num_classes=10,
        dropout=config.dropout
    ).to(device)
    
    # Select optimizer
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    
    # Data loaders with sweep batch size
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    # Training loop
    for epoch in range(10):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        wandb.log({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epoch': epoch
        })
    
    run.finish()

# Run the sweep
wandb.agent(sweep_id, function=train_sweep, count=50)
```

### Sweep Methods

**Grid Search**: Exhaustive search over all parameter combinations
```python
'method': 'grid'
```

**Random Search**: Random sampling from parameter distributions
```python
'method': 'random'
```

**Bayesian Optimization**: Uses Gaussian Process to model the objective function
```python
'method': 'bayes'
```

## Complete Training Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import wandb

def train():
    # Initialize W&B
    config = {
        'learning_rate': 0.001,
        'batch_size': 64,
        'hidden_size': 500,
        'num_epochs': 10,
        'dropout': 0.2
    }
    
    run = wandb.init(
        project='mnist-classification',
        config=config,
        name='mlp-baseline',
        tags=['mlp', 'baseline']
    )
    config = wandb.config  # Use wandb config for sweep compatibility
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loading
    transform = transforms.ToTensor()
    train_dataset = torchvision.datasets.MNIST(
        './data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        './data', train=False, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # Log sample images
    examples = next(iter(train_loader))
    wandb.log({
        'training_samples': [
            wandb.Image(img, caption=f'Label: {label}')
            for img, label in zip(examples[0][:16], examples[1][:16])
        ]
    })
    
    # Model
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes, dropout):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_size, num_classes)
        
        def forward(self, x):
            x = x.view(-1, 784)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    model = MLP(784, config.hidden_size, 10, config.dropout).to(device)
    
    # Watch model (logs gradients and parameters)
    wandb.watch(model, log='all', log_freq=100)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    best_accuracy = 0
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                wandb.log({
                    'batch_loss': loss.item(),
                    'batch': epoch * len(train_loader) + batch_idx
                })
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Evaluation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                probs = F.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = test_correct / test_total
        
        # Log epoch metrics
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/accuracy': train_acc,
            'test/loss': test_loss,
            'test/accuracy': test_acc
        })
        
        # Log weight histograms
        for name, param in model.named_parameters():
            wandb.log({f'weights/{name}': wandb.Histogram(param.detach().cpu())})
        
        print(f'Epoch {epoch+1}/{config.num_epochs} - '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
        
        # Save best model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            
            torch.save(model.state_dict(), 'best_model.pth')
            artifact = wandb.Artifact(
                name='best-model',
                type='model',
                description=f'Best model with {test_acc:.4f} accuracy'
            )
            artifact.add_file('best_model.pth')
            run.log_artifact(artifact)
    
    # Log final confusion matrix
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    wandb.log({'confusion_matrix': wandb.Image(fig)})
    plt.close(fig)
    
    # Log PR curves
    wandb.log({
        'pr_curve': wandb.plot.pr_curve(
            y_true=all_labels,
            y_probas=all_probs,
            labels=[str(i) for i in range(10)]
        )
    })
    
    # Summary metrics
    wandb.run.summary['best_accuracy'] = best_accuracy
    wandb.run.summary['final_train_loss'] = train_loss
    wandb.run.summary['final_test_loss'] = test_loss
    
    run.finish()

if __name__ == '__main__':
    train()
```

## Advanced Features

### Watching Models

Automatically log gradients and parameter distributions:

```python
# Log gradients and parameters
wandb.watch(model, log='all', log_freq=100)

# Options for log parameter:
# 'gradients': Log gradients only
# 'parameters': Log parameters only
# 'all': Log both
# None: Log nothing
```

### Alerts

Set up alerts for training issues:

```python
# Alert when validation loss increases
if val_loss > prev_val_loss * 1.1:
    wandb.alert(
        title='Validation Loss Spike',
        text=f'Validation loss increased from {prev_val_loss:.4f} to {val_loss:.4f}',
        level=wandb.AlertLevel.WARN
    )

# Alert on training completion
wandb.alert(
    title='Training Complete',
    text=f'Final accuracy: {best_accuracy:.4f}',
    level=wandb.AlertLevel.INFO
)
```

### Reports

Create interactive reports programmatically:

```python
import wandb

# Create a report
report = wandb.Report(
    project='mnist-classification',
    title='MNIST Experiment Results',
    description='Analysis of MLP architectures on MNIST'
)

# Reports are typically created through the web UI
# but can be shared via URLs
```

### Offline Mode

Train without internet connection:

```python
import os
os.environ['WANDB_MODE'] = 'offline'

# Or in code
wandb.init(mode='offline')

# Later, sync offline runs
# wandb sync ./wandb/offline-run-*
```

### Resuming Runs

Resume interrupted training:

```python
# Save run ID when starting
run = wandb.init(project='mnist', resume='allow')
run_id = run.id

# Later, resume with the same ID
run = wandb.init(project='mnist', id=run_id, resume='must')
```

## Best Practices

### Project Organization

```python
# Use consistent naming conventions
wandb.init(
    project='project-name',           # Lowercase with hyphens
    name='experiment-description',    # Human-readable
    tags=['tag1', 'tag2'],           # For filtering
    group='experiment-group',         # Group related runs
    job_type='train'                  # 'train', 'eval', 'sweep'
)
```

### Configuration Management

```python
# Use config files for reproducibility
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

wandb.init(config=config)
```

### Efficient Logging

```python
# Batch logging for efficiency
metrics = {}
for batch_idx, (data, target) in enumerate(train_loader):
    loss = train_step(model, data, target)
    metrics[f'batch_{batch_idx}_loss'] = loss

# Log all at once (less overhead)
wandb.log(metrics)
```

### Team Collaboration

```python
# Use teams for shared projects
wandb.init(
    entity='team-name',
    project='shared-project'
)

# Add notes and descriptions for teammates
wandb.run.notes = 'Experiment with new architecture'
wandb.run.save()
```

## W&B vs TensorBoard

| Feature | W&B | TensorBoard |
|---------|-----|-------------|
| Storage | Cloud (persistent) | Local (ephemeral) |
| Collaboration | Built-in sharing | Manual file sharing |
| Hyperparameter Sweeps | Integrated | External tools needed |
| Artifacts | Versioned datasets/models | Not included |
| Cost | Free tier + paid | Free |
| Setup | Account required | No account needed |
| Offline Mode | Supported | Native |

Use TensorBoard for quick local experiments; use W&B for serious projects requiring collaboration, experiment management, and reproducibility.

## Summary

Weights & Biases provides a comprehensive platform for experiment tracking that goes beyond basic metric visualization:

1. **Centralized logging**: All experiments stored in the cloud
2. **Rich media support**: Images, tables, plots, and custom visualizations
3. **Artifact management**: Version control for models and datasets
4. **Hyperparameter sweeps**: Automated search with Bayesian optimization
5. **Team collaboration**: Shared projects and reports
6. **Integration**: Works seamlessly with PyTorch and other frameworks

For production ML workflows, W&B offers the infrastructure needed to manage experiments at scale while maintaining reproducibility and enabling collaboration across teams.
