# Model Comparison: Systematic Architecture Evaluation

## Overview

Selecting the right architecture requires understanding trade-offs across multiple dimensions: accuracy, parameters, FLOPs, inference time, and training requirements. This section provides frameworks for systematic model comparison.

## Learning Objectives

1. Evaluate models across accuracy, efficiency, and speed
2. Measure FLOPs and parameters programmatically
3. Create meaningful comparison visualizations
4. Select models for different deployment scenarios

## Evaluation Metrics

### Accuracy Metrics

```python
# Top-1: Correct if highest prediction matches label
# Top-5: Correct if label in top 5 predictions

def evaluate_accuracy(model, dataloader, device):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Top-1
            _, pred = outputs.max(1)
            top1_correct += pred.eq(labels).sum().item()
            
            # Top-5
            _, pred5 = outputs.topk(5, 1, True, True)
            top5_correct += pred5.eq(labels.view(-1, 1)).sum().item()
            
            total += labels.size(0)
    
    return {
        'top1': 100 * top1_correct / total,
        'top5': 100 * top5_correct / total
    }
```

### Parameter Count

```python
def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}

def format_params(params):
    """Format as readable string."""
    if params >= 1e9:
        return f"{params/1e9:.1f}B"
    elif params >= 1e6:
        return f"{params/1e6:.1f}M"
    elif params >= 1e3:
        return f"{params/1e3:.1f}K"
    return str(params)
```

### FLOPs Estimation

```python
def estimate_flops(model, input_size=(1, 3, 224, 224)):
    """
    Estimate FLOPs for forward pass.
    
    For conv: H_out × W_out × K² × C_in × C_out
    For linear: in_features × out_features
    """
    total_flops = 0
    
    def conv_flops_hook(module, input, output):
        nonlocal total_flops
        batch, out_c, out_h, out_w = output.shape
        in_c = module.in_channels
        k_h, k_w = module.kernel_size
        
        # Multiply-adds per output element
        flops_per_elem = k_h * k_w * in_c / module.groups
        total_flops += batch * out_c * out_h * out_w * flops_per_elem
    
    def linear_flops_hook(module, input, output):
        nonlocal total_flops
        batch = input[0].shape[0]
        total_flops += batch * module.in_features * module.out_features
    
    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_flops_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_flops_hook))
    
    model.eval()
    with torch.no_grad():
        model(torch.randn(input_size))
    
    for hook in hooks:
        hook.remove()
    
    return total_flops
```

### Inference Time

```python
def measure_inference_time(model, input_size, device='cuda', warmup=10, iterations=100):
    """Measure average inference time in milliseconds."""
    model = model.to(device).eval()
    x = torch.randn(input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    # Synchronize GPU
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # Measure
    start = time.time()
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(x)
            if device == 'cuda':
                torch.cuda.synchronize()
    
    elapsed = (time.time() - start) / iterations * 1000  # ms
    return elapsed
```

## Comparison Framework

### Model Registry

```python
class ModelComparison:
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def register(self, name, model_fn, input_size=224):
        """Register a model for comparison."""
        self.models[name] = {
            'fn': model_fn,
            'input_size': input_size
        }
    
    def analyze(self, device='cuda'):
        """Analyze all registered models."""
        for name, config in self.models.items():
            model = config['fn']()
            size = config['input_size']
            
            self.results[name] = {
                'params': count_parameters(model)['total'],
                'flops': estimate_flops(model, (1, 3, size, size)),
                'latency_cpu': measure_inference_time(model, (1, 3, size, size), 'cpu'),
                'latency_gpu': measure_inference_time(model, (1, 3, size, size), 'cuda') if device == 'cuda' else None
            }
        
        return self.results
    
    def summary_table(self):
        """Print comparison table."""
        print(f"{'Model':<20} {'Params':<10} {'FLOPs':<10} {'CPU(ms)':<10} {'GPU(ms)':<10}")
        print("-" * 60)
        
        for name, r in self.results.items():
            print(f"{name:<20} {format_params(r['params']):<10} {format_params(r['flops']):<10} "
                  f"{r['latency_cpu']:.1f}ms{'':<5} {r['latency_gpu']:.1f}ms" if r['latency_gpu'] else "")
```

### Usage Example

```python
# Register models
comparison = ModelComparison()
comparison.register('ResNet-18', lambda: resnet18(num_classes=1000))
comparison.register('ResNet-50', lambda: resnet50(num_classes=1000))
comparison.register('VGG-16', lambda: vgg16(num_classes=1000, batch_norm=True))
comparison.register('MobileNetV2', lambda: mobilenet_v2(num_classes=1000))
comparison.register('EfficientNet-B0', lambda: efficientnet_b0(num_classes=1000))
comparison.register('DenseNet-121', lambda: densenet121(num_classes=1000))

# Analyze
results = comparison.analyze()
comparison.summary_table()
```

## Benchmark Results

### ImageNet Performance

| Model | Top-1 | Top-5 | Params | FLOPs | GPU ms |
|-------|-------|-------|--------|-------|--------|
| VGG-16 | 71.3% | 90.1% | 138M | 15.5B | 4.2 |
| ResNet-18 | 69.8% | 89.1% | 11.7M | 1.8B | 1.1 |
| ResNet-50 | 76.1% | 92.9% | 25.6M | 4.1B | 2.3 |
| ResNet-152 | 78.3% | 94.2% | 60.2M | 11.6B | 5.8 |
| DenseNet-121 | 75.0% | 92.2% | 8.0M | 2.9B | 2.8 |
| MobileNetV2 | 72.0% | 90.3% | 3.5M | 300M | 0.9 |
| EfficientNet-B0 | 77.3% | 93.5% | 5.3M | 390M | 1.5 |
| EfficientNet-B7 | 84.4% | 97.1% | 66M | 37B | 15.2 |

### Accuracy vs Efficiency

```
Accuracy vs Parameters:

  85% ─┤                                    ○ EfficientNet-B7
      │                          
  82% ─┤             ○ EfficientNet-B3
      │                  
  79% ─┤    ○ EffNet-B0  ○ ResNet-152
      │○ ResNet-50
  76% ─┤    ○ DenseNet-121
      │
  73% ─┤○ MobileNetV2
      │                           ○ VGG-16
  70% ─┼─────┼─────┼─────┼─────┼─────┼───── Params
           5M   20M   50M  100M  150M

EfficientNet dominates the Pareto frontier!
```

## Deployment Scenarios

### Model Selection Guide

| Scenario | Recommended | Reasoning |
|----------|-------------|-----------|
| **Mobile App** | MobileNetV2/V3 | Lowest latency, small memory |
| **Edge Device** | EfficientNet-B0 | Best accuracy/efficiency |
| **Cloud (throughput)** | ResNet-50 | Well-optimized, high throughput |
| **Cloud (accuracy)** | EfficientNet-B7 | Maximum accuracy |
| **Limited Data** | Transfer + ResNet-50 | Robust pretrained features |
| **Real-time Video** | MobileNetV2 | Consistent low latency |
| **Research Baseline** | ResNet-50 | Standard benchmark |

### Hardware Considerations

```python
def recommend_model(constraints):
    """
    Recommend model based on constraints.
    
    Args:
        constraints: dict with 'latency_budget', 'memory_budget', 'min_accuracy'
    """
    candidates = [
        ('MobileNetV2', {'acc': 72.0, 'lat': 0.9, 'mem': 14}),
        ('EfficientNet-B0', {'acc': 77.3, 'lat': 1.5, 'mem': 21}),
        ('ResNet-50', {'acc': 76.1, 'lat': 2.3, 'mem': 98}),
        ('EfficientNet-B3', {'acc': 81.7, 'lat': 3.2, 'mem': 48}),
    ]
    
    valid = []
    for name, specs in candidates:
        if (specs['lat'] <= constraints.get('latency_budget', float('inf')) and
            specs['mem'] <= constraints.get('memory_budget', float('inf')) and
            specs['acc'] >= constraints.get('min_accuracy', 0)):
            valid.append((name, specs))
    
    if not valid:
        return "No model meets all constraints"
    
    # Return highest accuracy among valid
    return max(valid, key=lambda x: x[1]['acc'])[0]
```

## Training Comparison

```python
class TrainingComparison:
    """Compare training characteristics across models."""
    
    def compare_training(self, models, train_loader, val_loader, epochs=50):
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            history = {'train_loss': [], 'val_acc': [], 'epoch_time': []}
            
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
            criterion = nn.CrossEntropyLoss()
            
            for epoch in range(epochs):
                start = time.time()
                
                # Train
                model.train()
                train_loss = 0
                for x, y in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(x.cuda()), y.cuda())
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validate
                val_acc = evaluate_accuracy(model, val_loader, 'cuda')['top1']
                
                epoch_time = time.time() - start
                
                history['train_loss'].append(train_loss / len(train_loader))
                history['val_acc'].append(val_acc)
                history['epoch_time'].append(epoch_time)
                
                scheduler.step()
            
            results[name] = {
                'history': history,
                'best_acc': max(history['val_acc']),
                'total_time': sum(history['epoch_time'])
            }
        
        return results
```

## Visualization

```python
def plot_comparison(results, metric='accuracy'):
    """Generate comparison plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy comparison
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]
    axes[0, 0].barh(names, accuracies)
    axes[0, 0].set_xlabel('Top-1 Accuracy (%)')
    axes[0, 0].set_title('Accuracy Comparison')
    
    # 2. Parameters comparison
    params = [results[n]['params'] / 1e6 for n in names]
    axes[0, 1].barh(names, params)
    axes[0, 1].set_xlabel('Parameters (M)')
    axes[0, 1].set_title('Model Size')
    
    # 3. Accuracy vs FLOPs scatter
    for name in names:
        axes[1, 0].scatter(
            results[name]['flops'] / 1e9,
            results[name]['accuracy'],
            s=100, label=name
        )
    axes[1, 0].set_xlabel('FLOPs (B)')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].legend()
    axes[1, 0].set_title('Accuracy vs Compute')
    
    # 4. Latency comparison
    latencies = [results[n]['latency_gpu'] for n in names]
    axes[1, 1].barh(names, latencies)
    axes[1, 1].set_xlabel('Latency (ms)')
    axes[1, 1].set_title('Inference Speed (GPU)')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150)
```

## Exercises

1. **Complete Benchmark**: Evaluate all architectures on CIFAR-10
2. **Pareto Analysis**: Identify models on accuracy-efficiency frontier
3. **Deployment Study**: Recommend models for mobile, edge, cloud
4. **Custom Architecture**: Design model optimized for your constraints

## Key Takeaways

1. **No single best model**: Selection depends on constraints
2. **EfficientNet** dominates accuracy/efficiency Pareto frontier
3. **MobileNet** best for mobile/edge with strict latency
4. **ResNet-50** reliable baseline for cloud deployment
5. **Always measure on target hardware** - theoretical FLOPs ≠ actual speed

---

**Previous**: [DenseNet](densenet.md) | **Next**: [Transfer Learning](transfer_learning.md)
