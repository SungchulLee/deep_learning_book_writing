"""
BEGINNER LEVEL: Knowledge Distillation

This script demonstrates how to transfer knowledge from a large "teacher" model
to a smaller "student" model, achieving better performance than training the
student from scratch.

Topics Covered:
- Teacher-student learning framework
- Soft targets and temperature scaling
- Hard vs soft label training
- Distillation loss components

Mathematical Background:
- Soft targets: p_i = softmax(z_i / T) where T is temperature
- Student loss: L = α×L_hard + (1-α)×L_soft
- L_hard = CrossEntropy(student_output, true_labels)
- L_soft = KL_divergence(student_output/T, teacher_output/T) × T²

Intuition:
Teacher's soft predictions contain "dark knowledge" about class similarities
that hard labels don't provide. For example, a teacher might output:
  Hard: [0, 0, 1, 0, 0]  (cat)
  Soft: [0.01, 0.02, 0.90, 0.05, 0.02]  (cat, but similar to tiger/leopard)

Prerequisites:
- Module 20: Feedforward Networks
- Module 23: CNNs
- Understanding of softmax and cross-entropy
- Familiarity with probability distributions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import numpy as np
import matplotlib.pyplot as plt

# Import our utility functions
from utils import (
    count_parameters,
    get_model_size,
    compare_model_sizes,
    evaluate_accuracy,
    compare_accuracies,
    seed_everything
)


# ============================================================================
# STEP 1: DEFINE TEACHER AND STUDENT MODELS
# ============================================================================

class TeacherCNN(nn.Module):
    """
    A large teacher model with high capacity.
    
    Architecture (deep and wide):
    - Conv1: 1 → 64 channels
    - Conv2: 64 → 128 channels
    - Conv3: 128 → 256 channels
    - FC: 256*3*3 → 512 → 256 → 10
    
    Total parameters: ~1.8M
    This model is intentionally large to learn rich representations.
    """
    def __init__(self, num_classes=10):
        super(TeacherCNN, self).__init__()
        
        # Large convolutional layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Large fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        # Convolutional blocks with batch norm
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28→14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14→7
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 7→3
        
        # Flatten and fully connected
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class StudentCNN(nn.Module):
    """
    A small student model with limited capacity.
    
    Architecture (shallow and narrow):
    - Conv1: 1 → 16 channels
    - Conv2: 16 → 32 channels
    - FC: 32*7*7 → 64 → 10
    
    Total parameters: ~54k (33x smaller than teacher!)
    This model would normally underperform but benefits from distillation.
    """
    def __init__(self, num_classes=10):
        super(StudentCNN, self).__init__()
        
        # Small convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Small fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 28→14
        x = self.pool(F.relu(self.conv2(x)))  # 14→7
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


# ============================================================================
# STEP 2: LOAD DATA
# ============================================================================

def get_mnist_dataloaders(batch_size=128):
    """Load MNIST dataset."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ============================================================================
# STEP 3: TRAIN TEACHER MODEL
# ============================================================================

def train_teacher(model, train_loader, test_loader, epochs=10, lr=0.001, device='cpu'):
    """
    Train the teacher model to high accuracy.
    
    Goal: Teacher should be as accurate as possible to provide
    high-quality knowledge to the student.
    
    Args:
        model: Teacher model
        train_loader: Training data
        test_loader: Test data
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained teacher model
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print("\nTraining Teacher Model...")
    print("-" * 60)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
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
        
        train_acc = correct / total
        test_acc = evaluate_accuracy(model, test_loader, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc*100:.2f}%, "
              f"Test Acc: {test_acc*100:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'teacher_mnist.pth')
    
    print(f"\nBest Teacher Accuracy: {best_acc*100:.2f}%")
    print("-" * 60)
    
    # Load best model
    model.load_state_dict(torch.load('teacher_mnist.pth', map_location=device))
    return model


# ============================================================================
# STEP 4: DISTILLATION LOSS FUNCTION
# ============================================================================

class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation.
    
    Loss = α × L_hard + (1-α) × L_soft
    
    where:
    - L_hard: Standard cross-entropy with true labels
    - L_soft: KL divergence between teacher and student soft predictions
    - α: Weight balancing the two losses (typically 0.1 to 0.5)
    - T: Temperature for softening predictions (typically 3 to 20)
    
    Temperature scaling:
    - Higher T → softer (more uniform) distribution
    - T=1 → standard softmax
    - T→∞ → uniform distribution
    
    Mathematical derivation:
    Given logits z, soft probabilities are:
        p_i = exp(z_i/T) / Σ_j exp(z_j/T)
    
    As T increases:
        p_i ≈ 1/C + (z_i - z̄)/(CT) + O(1/T²)
    This reveals differences between logits even for small probabilities.
    """
    def __init__(self, temperature=4.0, alpha=0.3):
        """
        Args:
            temperature: Temperature for softening (T)
            alpha: Weight for hard loss (α)
                  Soft loss weight is (1-α)
        """
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.hard_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        """
        Calculate distillation loss.
        
        Args:
            student_logits: Raw outputs from student (before softmax)
            teacher_logits: Raw outputs from teacher (before softmax)
            labels: Ground truth labels
            
        Returns:
            Combined distillation loss
        """
        # Hard loss: Standard cross-entropy with true labels
        hard_loss = self.hard_loss(student_logits, labels)
        
        # Soft loss: KL divergence between softened predictions
        # Apply temperature scaling
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        
        # Important: Scale soft loss by T² to compensate for temperature
        # This ensures gradients have appropriate magnitude
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Combine losses
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return total_loss, hard_loss, soft_loss


# ============================================================================
# STEP 5: TRAIN STUDENT WITH DISTILLATION
# ============================================================================

def train_student_with_distillation(student, teacher, train_loader, test_loader,
                                   epochs=15, lr=0.001, temperature=4.0, 
                                   alpha=0.3, device='cpu'):
    """
    Train student model using knowledge distillation.
    
    Process:
    1. Teacher generates soft targets for each batch
    2. Student tries to match both soft targets and hard labels
    3. Distillation loss guides student to learn teacher's knowledge
    
    Args:
        student: Student model to train
        teacher: Trained teacher model (frozen)
        train_loader: Training data
        test_loader: Test data
        epochs: Number of training epochs
        lr: Learning rate
        temperature: Temperature for softening (T)
        alpha: Weight for hard loss (α)
        device: Device to train on
        
    Returns:
        Trained student model
    """
    student = student.to(device)
    teacher = teacher.to(device)
    teacher.eval()  # Teacher is frozen (no gradient updates)
    
    criterion = DistillationLoss(temperature=temperature, alpha=alpha)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    
    print(f"\nTraining Student with Distillation...")
    print(f"Temperature (T): {temperature}")
    print(f"Alpha (α): {alpha} [hard loss weight]")
    print(f"1-Alpha: {1-alpha} [soft loss weight]")
    print("-" * 60)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0.0
        total_hard = 0.0
        total_soft = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            # Get teacher predictions (no gradient)
            with torch.no_grad():
                teacher_logits = teacher(data)
            
            # Get student predictions
            optimizer.zero_grad()
            student_logits = student(data)
            
            # Calculate distillation loss
            loss, hard_loss, soft_loss = criterion(student_logits, teacher_logits, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            total_hard += hard_loss.item()
            total_soft += soft_loss.item()
            
            _, predicted = student_logits.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        train_acc = correct / total
        test_acc = evaluate_accuracy(student, test_loader, device)
        
        avg_loss = total_loss / len(train_loader)
        avg_hard = total_hard / len(train_loader)
        avg_soft = total_soft / len(train_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {avg_loss:.4f} (Hard: {avg_hard:.4f}, Soft: {avg_soft:.4f}), "
              f"Train Acc: {train_acc*100:.2f}%, "
              f"Test Acc: {test_acc*100:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), 'student_distilled_mnist.pth')
    
    print(f"\nBest Student Accuracy (Distilled): {best_acc*100:.2f}%")
    print("-" * 60)
    
    # Load best model
    student.load_state_dict(torch.load('student_distilled_mnist.pth', map_location=device))
    return student


def train_student_baseline(student, train_loader, test_loader, 
                          epochs=15, lr=0.001, device='cpu'):
    """
    Train student model WITHOUT distillation (baseline comparison).
    
    This trains the student using only hard labels, providing
    a baseline to measure distillation's benefit.
    
    Args:
        student: Student model
        train_loader: Training data
        test_loader: Test data
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        
    Returns:
        Trained student model
    """
    student = student.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    
    print("\nTraining Student WITHOUT Distillation (Baseline)...")
    print("-" * 60)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        student.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = student(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        train_acc = correct / total
        test_acc = evaluate_accuracy(student, test_loader, device)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Acc: {train_acc*100:.2f}%, "
              f"Test Acc: {test_acc*100:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(student.state_dict(), 'student_baseline_mnist.pth')
    
    print(f"\nBest Student Accuracy (Baseline): {best_acc*100:.2f}%")
    print("-" * 60)
    
    student.load_state_dict(torch.load('student_baseline_mnist.pth', map_location=device))
    return student


# ============================================================================
# STEP 6: VISUALIZE SOFT TARGETS
# ============================================================================

def visualize_soft_targets(teacher, test_loader, device='cpu', num_samples=5):
    """
    Visualize how teacher's soft targets differ from hard labels.
    
    This demonstrates the "dark knowledge" that teachers provide:
    - Hard labels: One-hot vectors
    - Soft targets: Probability distributions showing class relationships
    
    Args:
        teacher: Trained teacher model
        test_loader: Test data
        device: Device to run on
        num_samples: Number of samples to visualize
    """
    teacher = teacher.to(device)
    teacher.eval()
    
    # Get one batch
    data, targets = next(iter(test_loader))
    data, targets = data.to(device), targets.to(device)
    
    # Get predictions
    with torch.no_grad():
        logits = teacher(data)
        probs = F.softmax(logits, dim=1)
    
    # Plot
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 3*num_samples))
    
    for idx in range(num_samples):
        # Show image
        ax_img = axes[idx, 0] if num_samples > 1 else axes[0]
        img = data[idx].cpu().squeeze()
        ax_img.imshow(img, cmap='gray')
        ax_img.set_title(f'True Label: {targets[idx].item()}')
        ax_img.axis('off')
        
        # Show probability distribution
        ax_prob = axes[idx, 1] if num_samples > 1 else axes[1]
        probs_np = probs[idx].cpu().numpy()
        bars = ax_prob.bar(range(10), probs_np, color='steelblue', edgecolor='black')
        
        # Highlight true class
        true_class = targets[idx].item()
        bars[true_class].set_color('green')
        
        ax_prob.set_xlabel('Class')
        ax_prob.set_ylabel('Probability')
        ax_prob.set_title('Teacher Soft Predictions')
        ax_prob.set_xticks(range(10))
        ax_prob.set_ylim([0, 1])
        ax_prob.grid(True, alpha=0.3, axis='y')
        
        # Add text showing top-3 predictions
        top3_probs, top3_classes = torch.topk(probs[idx], 3)
        text = "Top 3:\n"
        for prob, cls in zip(top3_probs, top3_classes):
            text += f"  {cls.item()}: {prob.item():.3f}\n"
        ax_prob.text(0.98, 0.98, text, transform=ax_prob.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=9, family='monospace')
    
    plt.tight_layout()
    plt.savefig('soft_targets_visualization.png', dpi=150, bbox_inches='tight')
    print("\nSoft targets visualization saved to: soft_targets_visualization.png")
    plt.show()


# ============================================================================
# STEP 7: TEMPERATURE ANALYSIS
# ============================================================================

def analyze_temperature_effect(teacher, train_loader, test_loader, device='cpu'):
    """
    Experiment with different temperature values.
    
    This shows how temperature affects:
    - Softness of the distribution
    - Student performance
    - Training dynamics
    
    Args:
        teacher: Trained teacher
        train_loader: Training data
        test_loader: Test data
        device: Device to run on
    """
    temperatures = [1, 2, 4, 8, 16]
    results = {'temperature': [], 'accuracy': []}
    
    print("\n" + "="*60)
    print("TEMPERATURE EXPERIMENT")
    print("="*60)
    
    for temp in temperatures:
        print(f"\n--- Testing Temperature T = {temp} ---")
        
        # Create fresh student
        student = StudentCNN()
        
        # Train with this temperature
        trained_student = train_student_with_distillation(
            student, teacher, train_loader, test_loader,
            epochs=10, temperature=temp, alpha=0.3, device=device
        )
        
        # Evaluate
        acc = evaluate_accuracy(trained_student, test_loader, device)
        
        results['temperature'].append(temp)
        results['accuracy'].append(acc)
        
        print(f"Final accuracy with T={temp}: {acc*100:.2f}%")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(results['temperature'], [acc*100 for acc in results['accuracy']],
             marker='o', linewidth=2, markersize=8)
    plt.xlabel('Temperature (T)', fontsize=12)
    plt.ylabel('Student Test Accuracy (%)', fontsize=12)
    plt.title('Effect of Temperature on Distillation', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('temperature_analysis.png', dpi=150, bbox_inches='tight')
    print("\nTemperature analysis saved to: temperature_analysis.png")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function demonstrating knowledge distillation.
    """
    seed_everything(42)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load data
    train_loader, test_loader = get_mnist_dataloaders(batch_size=128)
    
    # ========================================================================
    # PART A: TRAIN TEACHER MODEL
    # ========================================================================
    
    print("="*60)
    print("PART A: TEACHER MODEL")
    print("="*60)
    
    teacher = TeacherCNN()
    print(f"\nTeacher parameters: {count_parameters(teacher):,}")
    print(f"Teacher size: {get_model_size(teacher)['mb']:.2f} MB")
    
    # Train or load teacher
    try:
        teacher.load_state_dict(torch.load('teacher_mnist.pth', map_location=device))
        print("\nLoaded pre-trained teacher")
        teacher_acc = evaluate_accuracy(teacher, test_loader, device)
        print(f"Teacher accuracy: {teacher_acc*100:.2f}%")
    except:
        teacher = train_teacher(teacher, train_loader, test_loader, epochs=10, device=device)
    
    # Visualize soft targets
    visualize_soft_targets(teacher, test_loader, device, num_samples=5)
    
    # ========================================================================
    # PART B: TRAIN STUDENT WITHOUT DISTILLATION
    # ========================================================================
    
    print("\n" + "="*60)
    print("PART B: STUDENT BASELINE (NO DISTILLATION)")
    print("="*60)
    
    student_baseline = StudentCNN()
    print(f"\nStudent parameters: {count_parameters(student_baseline):,}")
    print(f"Student size: {get_model_size(student_baseline)['mb']:.2f} MB")
    
    compression_ratio = count_parameters(teacher) / count_parameters(student_baseline)
    print(f"Compression ratio: {compression_ratio:.1f}x")
    
    # Train baseline student
    try:
        student_baseline.load_state_dict(torch.load('student_baseline_mnist.pth', map_location=device))
        print("\nLoaded pre-trained baseline student")
    except:
        student_baseline = train_student_baseline(
            student_baseline, train_loader, test_loader, epochs=15, device=device
        )
    
    baseline_acc = evaluate_accuracy(student_baseline, test_loader, device)
    print(f"\nBaseline Student Accuracy: {baseline_acc*100:.2f}%")
    
    # ========================================================================
    # PART C: TRAIN STUDENT WITH DISTILLATION
    # ========================================================================
    
    print("\n" + "="*60)
    print("PART C: STUDENT WITH DISTILLATION")
    print("="*60)
    
    student_distilled = StudentCNN()
    
    # Train with distillation
    try:
        student_distilled.load_state_dict(torch.load('student_distilled_mnist.pth', map_location=device))
        print("\nLoaded pre-trained distilled student")
    except:
        student_distilled = train_student_with_distillation(
            student_distilled, teacher, train_loader, test_loader,
            epochs=15, temperature=4.0, alpha=0.3, device=device
        )
    
    distilled_acc = evaluate_accuracy(student_distilled, test_loader, device)
    print(f"\nDistilled Student Accuracy: {distilled_acc*100:.2f}%")
    
    # ========================================================================
    # PART D: COMPARISON
    # ========================================================================
    
    print("\n" + "="*60)
    print("COMPREHENSIVE COMPARISON")
    print("="*60)
    
    teacher_acc = evaluate_accuracy(teacher, test_loader, device)
    
    print(f"\n{'Model':<25} {'Parameters':<15} {'Size (MB)':<12} {'Accuracy (%)'}")
    print("-" * 70)
    print(f"{'Teacher (Large)':<25} {count_parameters(teacher):<15,} "
          f"{get_model_size(teacher)['mb']:<12.2f} {teacher_acc*100:.2f}")
    print(f"{'Student (Baseline)':<25} {count_parameters(student_baseline):<15,} "
          f"{get_model_size(student_baseline)['mb']:<12.2f} {baseline_acc*100:.2f}")
    print(f"{'Student (Distilled)':<25} {count_parameters(student_distilled):<15,} "
          f"{get_model_size(student_distilled)['mb']:<12.2f} {distilled_acc*100:.2f}")
    
    improvement = (distilled_acc - baseline_acc) * 100
    gap_to_teacher = (teacher_acc - distilled_acc) * 100
    
    print("\n" + "="*60)
    print("KEY RESULTS")
    print("="*60)
    print(f"Distillation Improvement:  +{improvement:.2f}%")
    print(f"Gap to Teacher:            {gap_to_teacher:.2f}%")
    print(f"Model Size Reduction:      {compression_ratio:.1f}x")
    print("="*60)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*60)
    print("SUMMARY: KEY TAKEAWAYS")
    print("="*60)
    print("""
    1. Knowledge Distillation Benefits:
       ✓ Student learns from teacher's soft predictions
       ✓ Captures "dark knowledge" about class relationships
       ✓ Typically 2-5% accuracy improvement over baseline
       ✓ Achieves better compression-accuracy trade-off
    
    2. Temperature (T):
       ✓ Higher T → softer distributions → more information
       ✓ Typical range: 3-20
       ✓ T=1 reduces to standard training
       ✓ Too high T can hurt performance
    
    3. Alpha (α):
       ✓ Balances hard vs soft loss
       ✓ α=0.3-0.5 works well in practice
       ✓ Higher α → more focus on true labels
       ✓ Lower α → more focus on teacher
    
    4. When to Use:
       - Deploying on resource-constrained devices
       - Need smaller models without accuracy loss
       - Have a well-trained teacher available
       - Want faster inference
    
    5. Advanced Techniques (Future modules):
       - Feature-based distillation
       - Self-distillation
       - Multi-teacher distillation
       - Online distillation
    """)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
