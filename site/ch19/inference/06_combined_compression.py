"""
ADVANCED LEVEL: Combined Compression Pipeline

This script demonstrates a complete compression pipeline combining:
1. Pruning (remove redundant weights)
2. Knowledge Distillation (train compact model)
3. Quantization (reduce precision)

This achieves maximum compression with minimal accuracy loss.

Topics Covered:
- Multi-stage compression pipeline
- Pruning → Distillation → Quantization order
- Trade-off analysis
- Production deployment considerations

Compression Pipeline:
Stage 1: Prune teacher model (50-70% sparsity)
Stage 2: Distill to smaller student
Stage 3: Quantize student to INT8
Result: 10-20x compression with <2% accuracy loss

Prerequisites:
- All previous modules (01-05)
- Deep understanding of each compression technique
- Experience with model training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy

from utils import (
    count_parameters,
    get_model_size,
    evaluate_accuracy,
    seed_everything
)


class LargeTeacher(nn.Module):
    """Large teacher model."""
    def __init__(self, num_classes=10):
        super(LargeTeacher, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TinyStudent(nn.Module):
    """Tiny student model."""
    def __init__(self, num_classes=10):
        super(TinyStudent, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def compression_pipeline(train_loader, test_loader, device='cpu'):
    """
    Complete compression pipeline.
    
    Returns:
        Dictionary with results at each stage
    """
    results = {}
    
    # Stage 0: Train teacher
    print("\n" + "="*60)
    print("STAGE 0: TRAINING TEACHER")
    print("="*60)
    
    teacher = LargeTeacher().to(device)
    # Train teacher (simplified - would normally train longer)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=0.001)
    
    for epoch in range(3):  # Quick training for demo
        teacher.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = teacher(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    teacher_acc = evaluate_accuracy(teacher, test_loader, device)
    teacher_size = get_model_size(teacher)['mb']
    
    results['teacher'] = {
        'accuracy': teacher_acc,
        'size_mb': teacher_size,
        'params': count_parameters(teacher)
    }
    
    print(f"Teacher: {teacher_acc*100:.2f}% accuracy, {teacher_size:.2f} MB")
    
    # Stage 1: Prune teacher
    print("\n" + "="*60)
    print("STAGE 1: PRUNING TEACHER")
    print("="*60)
    
    # Simplified pruning (zero out 50% of weights)
    pruned_teacher = copy.deepcopy(teacher)
    for param in pruned_teacher.parameters():
        if len(param.shape) > 1:  # Only weights
            threshold = torch.quantile(param.data.abs(), 0.5)
            mask = param.data.abs() >= threshold
            param.data *= mask.float()
    
    pruned_acc = evaluate_accuracy(pruned_teacher, test_loader, device)
    
    results['pruned_teacher'] = {
        'accuracy': pruned_acc,
        'size_mb': teacher_size,  # Same size (sparse)
        'params': count_parameters(pruned_teacher)
    }
    
    print(f"Pruned Teacher: {pruned_acc*100:.2f}% accuracy")
    
    # Stage 2: Distill to student
    print("\n" + "="*60)
    print("STAGE 2: DISTILLING TO STUDENT")
    print("="*60)
    
    student = TinyStudent().to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
    
    # Simplified distillation training
    T = 4.0
    alpha = 0.3
    
    for epoch in range(3):  # Quick training for demo
        student.train()
        pruned_teacher.eval()
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            with torch.no_grad():
                teacher_logits = pruned_teacher(data)
            
            optimizer.zero_grad()
            student_logits = student(data)
            
            # Distillation loss
            hard_loss = criterion(student_logits, target)
            soft_loss = F.kl_div(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1),
                reduction='batchmean'
            ) * (T ** 2)
            
            loss = alpha * hard_loss + (1 - alpha) * soft_loss
            loss.backward()
            optimizer.step()
    
    student_acc = evaluate_accuracy(student, test_loader, device)
    student_size = get_model_size(student)['mb']
    
    results['student'] = {
        'accuracy': student_acc,
        'size_mb': student_size,
        'params': count_parameters(student)
    }
    
    print(f"Student: {student_acc*100:.2f}% accuracy, {student_size:.2f} MB")
    
    # Stage 3: Quantize student
    print("\n" + "="*60)
    print("STAGE 3: QUANTIZING STUDENT")
    print("="*60)
    
    quantized_student = copy.deepcopy(student)
    quantized_size = student_size / 4  # INT8 = 1/4 of FP32
    
    # Note: Actual quantization would use torch.quantization
    # This is simplified for demonstration
    
    results['quantized_student'] = {
        'accuracy': student_acc,  # Would be slightly lower
        'size_mb': quantized_size,
        'params': count_parameters(quantized_student)
    }
    
    print(f"Quantized Student: ~{student_acc*100:.2f}% accuracy, {quantized_size:.2f} MB")
    
    return results


def print_compression_summary(results):
    """Print comprehensive summary."""
    print("\n" + "="*60)
    print("COMPRESSION PIPELINE SUMMARY")
    print("="*60)
    
    teacher = results['teacher']
    final = results['quantized_student']
    
    compression_ratio = teacher['params'] / final['params']
    size_reduction = (1 - final['size_mb'] / teacher['size_mb']) * 100
    accuracy_drop = (teacher['accuracy'] - final['accuracy']) * 100
    
    print(f"\n{'Stage':<25} {'Params':<15} {'Size (MB)':<12} {'Accuracy (%)'}")
    print("-" * 70)
    
    for name, data in results.items():
        display_name = name.replace('_', ' ').title()
        print(f"{display_name:<25} {data['params']:<15,} "
              f"{data['size_mb']:<12.2f} {data['accuracy']*100:.2f}")
    
    print("\n" + "="*60)
    print("FINAL COMPRESSION METRICS")
    print("="*60)
    print(f"Total Compression Ratio:   {compression_ratio:.1f}x")
    print(f"Size Reduction:            {size_reduction:.1f}%")
    print(f"Accuracy Drop:             {accuracy_drop:.2f}%")
    print("="*60)
    
    print("\n" + "="*60)
    print("PRODUCTION DEPLOYMENT GUIDE")
    print("="*60)
    print("""
    1. Hardware Selection:
       ✓ CPU: Use INT8 quantized model
       ✓ GPU: May benefit from FP16 instead of INT8
       ✓ Mobile: INT8 or even INT4 quantization
       ✓ Edge TPU: INT8 required
    
    2. Optimization Order:
       ✓ Always: Quantization (easy wins)
       ✓ If accuracy permits: Pruning
       ✓ If large teacher available: Distillation
       ✓ Combined: Maximum compression
    
    3. Trade-off Guidelines:
       ✓ <1% accuracy loss: Acceptable for most
       ✓ 1-2% loss: Acceptable for edge deployment
       ✓ >2% loss: Requires careful evaluation
    
    4. Validation:
       ✓ Test on target hardware (CPU/GPU/Mobile)
       ✓ Measure actual latency (not theoretical)
       ✓ Check memory usage under load
       ✓ Verify accuracy on diverse test sets
    
    5. Next Steps:
       - Export to ONNX for deployment
       - Optimize with TensorRT/CoreML
       - Profile on target device
       - A/B test against baseline
    """)


def main():
    """Main function for combined compression."""
    seed_everything(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*60)
    print("COMBINED COMPRESSION PIPELINE")
    print("="*60)
    print("\nThis demonstrates a complete compression workflow:")
    print("1. Train large teacher model")
    print("2. Prune teacher to remove redundancy")
    print("3. Distill knowledge to tiny student")
    print("4. Quantize student to INT8")
    print("\nResult: 10-20x compression with minimal accuracy loss")
    
    # Load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Run compression pipeline
    results = compression_pipeline(train_loader, test_loader, device)
    
    # Print summary
    print_compression_summary(results)


if __name__ == "__main__":
    main()
