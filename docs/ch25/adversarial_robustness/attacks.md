# Adversarial Attack Fundamentals

Adversarial attacks manipulate machine learning models by introducing carefully crafted perturbations to input data. These perturbations, often imperceptible to humans, can cause models to make incorrect predictions with high confidence.

## Introduction to Adversarial Attacks

### What Are Adversarial Examples?

Adversarial examples are inputs to machine learning models that have been intentionally designed to cause the model to make a mistake. In the context of image classification:

$$x_{adv} = x + \delta$$

where:
- $x$ is the original input
- $\delta$ is a small perturbation
- $x_{adv}$ is the adversarial example

The perturbation $\delta$ is constrained to be small (often measured by $L_p$ norms) so that $x_{adv}$ remains visually similar to $x$.

### Why Do Adversarial Examples Exist?

Several hypotheses explain the existence of adversarial examples:

1. **Linear Hypothesis (Goodfellow et al., 2015)**: Deep networks are essentially linear in high-dimensional spaces. Small perturbations accumulate across dimensions.

2. **Non-Robust Features**: Models may rely on features that are predictive but not robust to perturbations.

3. **Decision Boundary Geometry**: Complex, high-dimensional decision boundaries may have regions where small input changes cross the boundary.

## Threat Models

### Attack Scenarios

| Scenario | Attacker Knowledge | Example |
|----------|-------------------|---------|
| **White-box** | Full model access (architecture, weights, gradients) | Gradient-based attacks |
| **Black-box** | Query access only (inputs → outputs) | Transfer attacks, query attacks |
| **Gray-box** | Partial knowledge (architecture but not weights) | Architecture-specific attacks |

### Attack Goals

1. **Untargeted Attacks**: Cause misclassification to any incorrect class

$$\text{Find } \delta: f(x + \delta) \neq y_{true}$$

2. **Targeted Attacks**: Cause misclassification to a specific target class

$$\text{Find } \delta: f(x + \delta) = y_{target}$$

## Perturbation Constraints

Perturbations are typically constrained by $L_p$ norms:

### $L_\infty$ Norm (Maximum Perturbation)

$$\|\delta\|_\infty = \max_i |\delta_i| \leq \epsilon$$

- Bounds the maximum change to any single pixel
- Common constraint: $\epsilon = 8/255$ for images in $[0, 1]$

### $L_2$ Norm (Euclidean Distance)

$$\|\delta\|_2 = \sqrt{\sum_i \delta_i^2} \leq \epsilon$$

- Bounds the total magnitude of perturbation
- Allows larger changes to individual pixels if others are small

### $L_0$ Norm (Sparse Perturbations)

$$\|\delta\|_0 = |\{i : \delta_i \neq 0\}| \leq k$$

- Bounds the number of modified pixels
- Used for sparse adversarial patches

## Attack Types in GAN Context

### Attacks on GANs

GANs face unique adversarial vulnerabilities:

1. **Poisoning Attacks**: Inject malicious samples into training data
2. **Evasion Attacks**: Craft inputs that fool the discriminator
3. **Inference Attacks**: Extract information about training data
4. **Model Extraction**: Steal the generator's learned distribution

### GAN-Generated Adversarial Examples

GANs can be used to generate adversarial examples:

```python
import torch
import torch.nn as nn

class AdversarialGenerator(nn.Module):
    """Generate adversarial perturbations using a GAN-like architecture."""
    
    def __init__(self, input_channels=3, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon
        
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_channels, 4, 2, 1),
            nn.Tanh()  # Output in [-1, 1], scale by epsilon
        )
    
    def forward(self, x):
        """Generate adversarial perturbation for input x."""
        features = self.encoder(x)
        perturbation = self.decoder(features)
        
        # Scale perturbation to epsilon ball
        perturbation = self.epsilon * perturbation
        
        # Create adversarial example
        x_adv = torch.clamp(x + perturbation, 0, 1)
        
        return x_adv, perturbation
```

## Attack Success Metrics

### Classification Metrics

```python
def evaluate_attack(model, x_clean, x_adv, y_true, y_target=None):
    """
    Evaluate adversarial attack success.
    
    Args:
        model: Target classifier
        x_clean: Clean images
        x_adv: Adversarial images
        y_true: True labels
        y_target: Target labels (for targeted attacks)
    
    Returns:
        Dictionary of attack metrics
    """
    model.eval()
    
    with torch.no_grad():
        # Clean accuracy
        pred_clean = model(x_clean).argmax(dim=1)
        clean_acc = (pred_clean == y_true).float().mean().item()
        
        # Adversarial predictions
        pred_adv = model(x_adv).argmax(dim=1)
        adv_acc = (pred_adv == y_true).float().mean().item()
        
        # Attack success rate (untargeted)
        fooling_rate = (pred_adv != y_true).float().mean().item()
        
        metrics = {
            'clean_accuracy': clean_acc,
            'adversarial_accuracy': adv_acc,
            'fooling_rate': fooling_rate,
            'accuracy_drop': clean_acc - adv_acc,
        }
        
        # Targeted attack success
        if y_target is not None:
            target_success = (pred_adv == y_target).float().mean().item()
            metrics['target_success_rate'] = target_success
    
    return metrics
```

### Perturbation Metrics

```python
def perturbation_metrics(x_clean, x_adv):
    """
    Compute perturbation statistics.
    
    Args:
        x_clean: Clean images (batch)
        x_adv: Adversarial images (batch)
    
    Returns:
        Dictionary of perturbation metrics
    """
    delta = x_adv - x_clean
    
    # Per-sample norms
    l_inf = delta.abs().view(delta.size(0), -1).max(dim=1)[0]
    l_2 = delta.view(delta.size(0), -1).norm(p=2, dim=1)
    l_0 = (delta.abs() > 1e-6).view(delta.size(0), -1).sum(dim=1).float()
    
    return {
        'l_inf_mean': l_inf.mean().item(),
        'l_inf_max': l_inf.max().item(),
        'l_2_mean': l_2.mean().item(),
        'l_2_max': l_2.max().item(),
        'l_0_mean': l_0.mean().item(),
        'l_0_max': l_0.max().item(),
    }
```

## Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def visualize_adversarial_example(x_clean, x_adv, y_true, y_pred_clean, y_pred_adv,
                                   class_names=None, amplify=10):
    """
    Visualize clean image, perturbation, and adversarial image.
    
    Args:
        x_clean: Clean image tensor (C, H, W)
        x_adv: Adversarial image tensor (C, H, W)
        y_true: True label
        y_pred_clean: Prediction on clean image
        y_pred_adv: Prediction on adversarial image
        class_names: Optional list of class names
        amplify: Factor to amplify perturbation for visibility
    """
    # Compute perturbation
    delta = x_adv - x_clean
    
    # Convert to displayable format
    clean_img = x_clean.cpu().numpy().transpose(1, 2, 0)
    adv_img = x_adv.cpu().numpy().transpose(1, 2, 0)
    
    # Amplify perturbation for visibility
    pert_img = delta.cpu().numpy().transpose(1, 2, 0)
    pert_display = 0.5 + amplify * pert_img  # Center at 0.5
    pert_display = np.clip(pert_display, 0, 1)
    
    # Handle grayscale
    if clean_img.shape[-1] == 1:
        clean_img = clean_img.squeeze(-1)
        adv_img = adv_img.squeeze(-1)
        pert_display = pert_display.squeeze(-1)
    
    # Get class names
    if class_names is None:
        true_name = str(y_true)
        clean_name = str(y_pred_clean)
        adv_name = str(y_pred_adv)
    else:
        true_name = class_names[y_true]
        clean_name = class_names[y_pred_clean]
        adv_name = class_names[y_pred_adv]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    cmap = 'gray' if clean_img.ndim == 2 else None
    
    axes[0].imshow(np.clip(clean_img, 0, 1), cmap=cmap)
    axes[0].set_title(f'Clean Image\nTrue: {true_name}\nPred: {clean_name}')
    axes[0].axis('off')
    
    axes[1].imshow(pert_display, cmap=cmap)
    axes[1].set_title(f'Perturbation (×{amplify})\n'
                      f'L∞: {delta.abs().max():.4f}\n'
                      f'L2: {delta.norm():.4f}')
    axes[1].axis('off')
    
    axes[2].imshow(np.clip(adv_img, 0, 1), cmap=cmap)
    axes[2].set_title(f'Adversarial Image\nPred: {adv_name}')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
```

## Common Neural Network for Testing

```python
class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification, used as attack target."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)           # (B, 32, 28, 28)
        x = self.relu(x)
        x = self.maxpool(x)         # (B, 32, 14, 14)
        x = self.conv2(x)           # (B, 64, 14, 14)
        x = self.relu(x)
        x = self.maxpool(x)         # (B, 64, 7, 7)
        x = self.flatten(x)         # (B, 64*7*7)
        x = self.fc1(x)             # (B, 128)
        x = self.relu(x)
        x = self.fc2(x)             # (B, 10)
        return x


def train_target_model(model, train_loader, epochs=5, lr=0.001, device='cpu'):
    """Train the target model for adversarial attack experiments."""
    model.to(device)
    model.train()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Loss: {total_loss/len(train_loader):.4f}, '
              f'Acc: {100*correct/total:.2f}%')
    
    return model
```

## Summary

| Concept | Description |
|---------|-------------|
| **Adversarial Example** | Input with small perturbation causing misclassification |
| **White-box Attack** | Full model access, can compute gradients |
| **Black-box Attack** | Query-only access |
| **Untargeted Attack** | Cause any misclassification |
| **Targeted Attack** | Cause specific misclassification |
| **$L_\infty$ Constraint** | Bound maximum pixel change |
| **$L_2$ Constraint** | Bound total perturbation magnitude |
| **Fooling Rate** | Fraction of successful attacks |

Understanding these fundamentals is essential for both attacking and defending neural networks, including GANs where the discriminator can be targeted by adversarial examples.

---

# Fast Gradient Sign Method (FGSM)

The Fast Gradient Sign Method (FGSM), introduced by Goodfellow et al. in 2015, is a simple yet effective white-box adversarial attack that uses the gradient of the loss with respect to the input.

## Mathematical Foundation

### The FGSM Attack

FGSM creates adversarial examples by taking a single step in the direction of the gradient:

$$x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(\theta, x, y))$$

where:
- $x$ is the original input
- $\epsilon$ is the perturbation magnitude (attack strength)
- $\nabla_x \mathcal{L}$ is the gradient of the loss with respect to the input
- $\text{sign}(\cdot)$ takes the sign of each gradient component

### Why Sign?

Using the sign function ensures the perturbation satisfies the $L_\infty$ constraint:

$$\|\delta\|_\infty = \epsilon$$

Each pixel is perturbed by exactly $\pm\epsilon$, maximizing the perturbation within the constraint.

### Linear Approximation Interpretation

FGSM is based on a first-order Taylor expansion of the loss:

$$\mathcal{L}(x + \delta) \approx \mathcal{L}(x) + \delta^T \nabla_x \mathcal{L}(x)$$

To maximize the loss increase under $\|\delta\|_\infty \leq \epsilon$:

$$\delta^* = \arg\max_{\|\delta\|_\infty \leq \epsilon} \delta^T \nabla_x \mathcal{L}(x) = \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}(x))$$

## Implementation

### Basic FGSM Attack

```python
import torch
import torch.nn as nn

def fgsm_attack(model, images, labels, epsilon, criterion=None):
    """
    Perform FGSM attack on a batch of images.
    
    Args:
        model: Target classifier
        images: Input images (requires_grad should be True)
        labels: True labels
        epsilon: Perturbation magnitude
        criterion: Loss function (default: CrossEntropyLoss)
    
    Returns:
        adversarial_images: Perturbed images
        perturbation: The applied perturbation
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Ensure images require gradients
    images = images.clone().detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Backward pass to get gradients
    model.zero_grad()
    loss.backward()
    
    # Get gradient sign
    grad_sign = images.grad.data.sign()
    
    # Create perturbation
    perturbation = epsilon * grad_sign
    
    # Create adversarial images
    adversarial_images = images + perturbation
    
    # Clamp to valid range [0, 1]
    adversarial_images = torch.clamp(adversarial_images, 0, 1)
    
    return adversarial_images.detach(), perturbation.detach()


class FGSMAttack:
    """FGSM Attack wrapper class with additional functionality."""
    
    def __init__(self, model, epsilon=0.1, criterion=None, device='cpu'):
        """
        Args:
            model: Target model to attack
            epsilon: Perturbation bound
            criterion: Loss function
            device: Device to run attack on
        """
        self.model = model
        self.epsilon = epsilon
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.device = device
    
    def attack(self, images, labels):
        """
        Generate adversarial examples.
        
        Args:
            images: Clean images
            labels: True labels
        
        Returns:
            Adversarial images
        """
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        self.model.eval()
        
        # Enable gradient computation for images
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        
        # Compute gradients
        self.model.zero_grad()
        loss.backward()
        
        # FGSM step
        grad_sign = images.grad.data.sign()
        perturbed_images = images + self.epsilon * grad_sign
        perturbed_images = torch.clamp(perturbed_images, 0, 1)
        
        return perturbed_images.detach()
    
    def evaluate(self, dataloader, max_batches=None):
        """
        Evaluate attack success on a dataset.
        
        Args:
            dataloader: Test dataloader
            max_batches: Maximum batches to evaluate
        
        Returns:
            Dictionary with attack metrics
        """
        self.model.eval()
        
        clean_correct = 0
        adv_correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Clean predictions
            with torch.no_grad():
                clean_outputs = self.model(images)
                clean_pred = clean_outputs.argmax(dim=1)
                clean_correct += (clean_pred == labels).sum().item()
            
            # Adversarial examples
            adv_images = self.attack(images, labels)
            
            # Adversarial predictions
            with torch.no_grad():
                adv_outputs = self.model(adv_images)
                adv_pred = adv_outputs.argmax(dim=1)
                adv_correct += (adv_pred == labels).sum().item()
            
            total += labels.size(0)
        
        return {
            'clean_accuracy': clean_correct / total,
            'adversarial_accuracy': adv_correct / total,
            'attack_success_rate': (clean_correct - adv_correct) / clean_correct,
            'total_samples': total
        }
```

### Untargeted FGSM

The standard FGSM attack is untargeted—it simply maximizes the loss:

```python
def untargeted_fgsm(model, images, labels, epsilon):
    """
    Untargeted FGSM: maximize loss to cause any misclassification.
    
    The perturbation moves in the direction that increases the loss,
    pushing the prediction away from the true class.
    """
    criterion = nn.CrossEntropyLoss()
    
    images = images.clone().detach().requires_grad_(True)
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    model.zero_grad()
    loss.backward()
    
    # Move in direction of increasing loss (positive gradient direction)
    perturbation = epsilon * images.grad.data.sign()
    
    adv_images = torch.clamp(images + perturbation, 0, 1)
    
    return adv_images.detach()
```

### Targeted FGSM

Targeted FGSM aims to cause misclassification to a specific target class:

```python
def targeted_fgsm(model, images, target_labels, epsilon):
    """
    Targeted FGSM: minimize loss for target class.
    
    The perturbation moves in the direction that decreases the loss
    with respect to the target class, pulling predictions toward it.
    """
    criterion = nn.CrossEntropyLoss()
    
    images = images.clone().detach().requires_grad_(True)
    
    outputs = model(images)
    loss = criterion(outputs, target_labels)
    
    model.zero_grad()
    loss.backward()
    
    # Move in direction of DECREASING loss (negative gradient direction)
    # This pushes the prediction TOWARD the target class
    perturbation = -epsilon * images.grad.data.sign()
    
    adv_images = torch.clamp(images + perturbation, 0, 1)
    
    return adv_images.detach()


def run_targeted_attack(model, dataloader, target_class, epsilon, device='cpu'):
    """
    Run targeted FGSM attack to misclassify all images as target_class.
    
    Args:
        model: Target classifier
        dataloader: Test data
        target_class: Class to misclassify images as
        epsilon: Perturbation magnitude
        device: Device to run on
    
    Returns:
        Attack results dictionary
    """
    model.eval()
    model.to(device)
    
    success = 0
    total = 0
    
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # Create target labels
        target_labels = torch.full((batch_size,), target_class, device=device)
        
        # Skip images already classified as target
        with torch.no_grad():
            pred = model(images).argmax(dim=1)
            
        # Generate adversarial examples
        adv_images = targeted_fgsm(model, images, target_labels, epsilon)
        
        # Check success
        with torch.no_grad():
            adv_pred = model(adv_images).argmax(dim=1)
            success += (adv_pred == target_class).sum().item()
        
        total += batch_size
    
    return {
        'target_class': target_class,
        'success_rate': success / total,
        'total_samples': total
    }
```

## Complete Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt


# Define a simple CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_model(model, train_loader, epochs=5):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} complete")
    
    return model


def visualize_fgsm_attack(model, image, label, epsilons, class_names):
    """Visualize FGSM attack at different epsilon values."""
    
    fig, axes = plt.subplots(1, len(epsilons) + 1, figsize=(3 * (len(epsilons) + 1), 3))
    
    # Original image
    axes[0].imshow(image.squeeze().cpu(), cmap='gray')
    with torch.no_grad():
        pred = model(image.unsqueeze(0)).argmax().item()
    axes[0].set_title(f'Original\nPred: {class_names[pred]}')
    axes[0].axis('off')
    
    # Adversarial images at different epsilons
    for i, eps in enumerate(epsilons):
        adv_image, _ = fgsm_attack(
            model, 
            image.unsqueeze(0), 
            torch.tensor([label]), 
            eps
        )
        
        with torch.no_grad():
            adv_pred = model(adv_image).argmax().item()
        
        axes[i+1].imshow(adv_image.squeeze().cpu(), cmap='gray')
        axes[i+1].set_title(f'ε={eps}\nPred: {class_names[adv_pred]}')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    # Setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_data = torchvision.datasets.MNIST('./data', train=True, 
                                            download=True, transform=transform)
    test_data = torchvision.datasets.MNIST('./data', train=False, 
                                           download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    
    # Train model
    model = SimpleCNN()
    model = train_model(model, train_loader, epochs=5)
    model.eval()
    
    # Create FGSM attacker
    attacker = FGSMAttack(model, epsilon=0.1)
    
    # Evaluate attack
    results = attacker.evaluate(test_loader, max_batches=100)
    
    print("\nFGSM Attack Results:")
    print(f"Clean Accuracy: {results['clean_accuracy']:.2%}")
    print(f"Adversarial Accuracy: {results['adversarial_accuracy']:.2%}")
    print(f"Attack Success Rate: {results['attack_success_rate']:.2%}")
    
    # Visualize
    class_names = [str(i) for i in range(10)]
    image, label = next(iter(test_loader))
    epsilons = [0.05, 0.1, 0.15, 0.2, 0.3]
    visualize_fgsm_attack(model, image.squeeze(0), label.item(), epsilons, class_names)


if __name__ == "__main__":
    main()
```

## Epsilon Selection

The choice of $\epsilon$ controls the trade-off between attack success and perturbation visibility:

| Epsilon | Effect |
|---------|--------|
| Small (0.01-0.05) | Subtle changes, lower success rate |
| Medium (0.1-0.2) | Visible changes, high success rate |
| Large (0.3+) | Very visible, nearly 100% success |

### Common Epsilon Values

- **MNIST**: $\epsilon = 0.3$ (images in [0, 1])
- **CIFAR-10**: $\epsilon = 8/255 \approx 0.031$
- **ImageNet**: $\epsilon = 4/255$ or $8/255$

## Limitations of FGSM

1. **Single Step**: FGSM uses only one gradient step; iterative methods (PGD) are stronger
2. **Linear Approximation**: Assumes loss is approximately linear, which may not hold
3. **Gradient Masking**: Models trained against FGSM may learn to hide gradients
4. **Transferability**: FGSM examples may not transfer well to other models

## Summary

| Aspect | FGSM |
|--------|------|
| **Type** | White-box, gradient-based |
| **Constraint** | $L_\infty$ |
| **Steps** | Single |
| **Complexity** | O(1) backward pass |
| **Strength** | Fast, simple |
| **Weakness** | Suboptimal perturbation |

FGSM remains important as a baseline attack and for understanding adversarial robustness. Its simplicity makes it useful for quick evaluations and adversarial training.

---

# Targeted Adversarial Attacks

Targeted adversarial attacks aim to cause misclassification to a specific target class rather than just any incorrect class. These attacks are more challenging but also more dangerous in real-world scenarios.

## Targeted vs Untargeted Attacks

### Untargeted Attack

$$\text{Find } \delta: f(x + \delta) \neq y_{true}$$

Goal: Cause any misclassification.

### Targeted Attack

$$\text{Find } \delta: f(x + \delta) = y_{target}$$

Goal: Cause misclassification to a specific target class.

### Practical Differences

| Aspect | Untargeted | Targeted |
|--------|------------|----------|
| Difficulty | Easier | Harder |
| Control | Low | High |
| Real-world danger | Moderate | High |
| Example | Image not recognized | Face recognized as authorized user |

## Targeted FGSM

The simplest targeted attack modifies FGSM to minimize loss toward the target:

```python
import torch
import torch.nn as nn

def targeted_fgsm(model, images, target_labels, epsilon):
    """
    Targeted FGSM attack.
    
    Instead of maximizing loss (pushing away from true class),
    we minimize loss (pulling toward target class).
    
    Args:
        model: Target classifier
        images: Input images
        target_labels: Desired target class
        epsilon: Perturbation magnitude
    
    Returns:
        Adversarial images
    """
    criterion = nn.CrossEntropyLoss()
    
    # Clone and enable gradients
    images = images.clone().detach().requires_grad_(True)
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, target_labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Move in NEGATIVE gradient direction (minimize loss toward target)
    perturbation = -epsilon * images.grad.data.sign()
    
    # Create adversarial examples
    adv_images = torch.clamp(images + perturbation, 0, 1)
    
    return adv_images.detach()
```

## Iterative Targeted Attacks

### Basic Iterative Method (BIM) - Targeted

```python
def targeted_bim(model, images, target_labels, epsilon, alpha, num_iter):
    """
    Targeted Basic Iterative Method (Iterative FGSM).
    
    Takes multiple small steps toward the target class.
    
    Args:
        model: Target classifier
        images: Input images
        target_labels: Desired target class
        epsilon: Maximum total perturbation (L_inf bound)
        alpha: Step size per iteration
        num_iter: Number of iterations
    
    Returns:
        Adversarial images
    """
    criterion = nn.CrossEntropyLoss()
    
    # Start from original images
    adv_images = images.clone().detach()
    
    for i in range(num_iter):
        adv_images.requires_grad_(True)
        
        outputs = model(adv_images)
        loss = criterion(outputs, target_labels)
        
        model.zero_grad()
        loss.backward()
        
        # Small step toward target
        adv_images = adv_images - alpha * adv_images.grad.sign()
        
        # Project back to epsilon ball around original
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + perturbation, 0, 1).detach()
    
    return adv_images
```

### Projected Gradient Descent (PGD) - Targeted

```python
def targeted_pgd(model, images, target_labels, epsilon, alpha, num_iter, 
                 random_start=True):
    """
    Targeted PGD attack with random initialization.
    
    Args:
        model: Target classifier
        images: Input images
        target_labels: Desired target class
        epsilon: Maximum perturbation (L_inf)
        alpha: Step size
        num_iter: Number of iterations
        random_start: Whether to start from random point in epsilon ball
    
    Returns:
        Adversarial images
    """
    criterion = nn.CrossEntropyLoss()
    
    # Random initialization within epsilon ball
    if random_start:
        adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
        adv_images = torch.clamp(adv_images, 0, 1).detach()
    else:
        adv_images = images.clone().detach()
    
    for i in range(num_iter):
        adv_images.requires_grad_(True)
        
        outputs = model(adv_images)
        loss = criterion(outputs, target_labels)
        
        model.zero_grad()
        loss.backward()
        
        # Gradient descent (minimize loss toward target)
        adv_images = adv_images - alpha * adv_images.grad.sign()
        
        # Project to epsilon ball
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + perturbation, 0, 1).detach()
    
    return adv_images


class TargetedPGDAttack:
    """Targeted PGD attack wrapper."""
    
    def __init__(self, model, epsilon=0.3, alpha=0.01, num_iter=40, 
                 random_start=True, device='cpu'):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.random_start = random_start
        self.device = device
    
    def attack(self, images, target_labels):
        """Generate adversarial examples targeting specific class."""
        images = images.to(self.device)
        target_labels = target_labels.to(self.device)
        
        self.model.eval()
        
        return targeted_pgd(
            self.model, images, target_labels,
            self.epsilon, self.alpha, self.num_iter, self.random_start
        )
    
    def evaluate(self, dataloader, target_class, max_samples=1000):
        """
        Evaluate targeted attack success.
        
        Args:
            dataloader: Test data
            target_class: Class to target
            max_samples: Maximum samples to evaluate
        
        Returns:
            Attack metrics
        """
        self.model.eval()
        
        total = 0
        successful_attacks = 0
        already_target = 0
        
        for images, labels in dataloader:
            if total >= max_samples:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            batch_size = images.size(0)
            
            # Create target labels
            targets = torch.full((batch_size,), target_class, 
                                device=self.device, dtype=torch.long)
            
            # Check original predictions
            with torch.no_grad():
                original_pred = self.model(images).argmax(dim=1)
            
            # Count images already classified as target
            already_target += (original_pred == target_class).sum().item()
            
            # Generate adversarial examples
            adv_images = self.attack(images, targets)
            
            # Check adversarial predictions
            with torch.no_grad():
                adv_pred = self.model(adv_images).argmax(dim=1)
            
            successful_attacks += (adv_pred == target_class).sum().item()
            total += batch_size
        
        return {
            'target_class': target_class,
            'success_rate': successful_attacks / total,
            'already_target_rate': already_target / total,
            'total_samples': total
        }
```

## Carlini & Wagner (C&W) Targeted Attack

The C&W attack is one of the strongest targeted attacks, using optimization to find minimal perturbations:

```python
def cw_targeted_attack(model, images, target_labels, c=1.0, kappa=0, 
                        num_iter=1000, lr=0.01, device='cpu'):
    """
    Carlini & Wagner L2 targeted attack.
    
    Minimizes: ||δ||_2 + c * max(Z(x+δ)_t - max_{j≠t} Z(x+δ)_j, -κ)
    
    where Z is the logits, t is target class, κ is confidence margin.
    
    Args:
        model: Target classifier
        images: Input images
        target_labels: Target class
        c: Confidence parameter
        kappa: Confidence margin
        num_iter: Optimization iterations
        lr: Learning rate
        device: Device
    
    Returns:
        Adversarial images
    """
    images = images.to(device)
    target_labels = target_labels.to(device)
    
    # Initialize perturbation as tanh-scaled variable
    # Using tanh ensures bounded output
    w = torch.zeros_like(images, requires_grad=True, device=device)
    
    optimizer = torch.optim.Adam([w], lr=lr)
    
    best_adv = images.clone()
    best_l2 = float('inf') * torch.ones(images.size(0), device=device)
    
    for step in range(num_iter):
        # Transform w to valid image range using tanh
        adv_images = 0.5 * (torch.tanh(w) + 1)  # Maps to [0, 1]
        
        # L2 distance
        l2_dist = ((adv_images - images) ** 2).view(images.size(0), -1).sum(dim=1)
        
        # Get logits
        logits = model(adv_images)
        
        # C&W loss: maximize logit for target class
        # f(x') = max(max{Z(x')_j : j ≠ t} - Z(x')_t, -κ)
        target_logits = logits.gather(1, target_labels.view(-1, 1)).squeeze()
        
        # Max logit excluding target
        other_logits = logits.clone()
        other_logits.scatter_(1, target_labels.view(-1, 1), -float('inf'))
        max_other_logits = other_logits.max(dim=1)[0]
        
        # f function
        f_loss = torch.clamp(max_other_logits - target_logits + kappa, min=0)
        
        # Total loss
        loss = l2_dist.sum() + c * f_loss.sum()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track best adversarial examples (successful with minimum L2)
        with torch.no_grad():
            pred = logits.argmax(dim=1)
            is_successful = pred == target_labels
            is_better = l2_dist < best_l2
            should_update = is_successful & is_better
            
            best_l2 = torch.where(should_update, l2_dist, best_l2)
            for i in range(images.size(0)):
                if should_update[i]:
                    best_adv[i] = adv_images[i]
    
    return best_adv.detach()
```

## Target Selection Strategies

### Random Target

```python
def random_target(true_labels, num_classes):
    """Select random target different from true class."""
    targets = torch.randint(0, num_classes, true_labels.shape)
    # Ensure target is different from true label
    same_mask = targets == true_labels
    targets[same_mask] = (targets[same_mask] + 1) % num_classes
    return targets
```

### Least Likely Target

```python
def least_likely_target(model, images):
    """Select the class model is least confident about."""
    with torch.no_grad():
        logits = model(images)
        return logits.argmin(dim=1)
```

### Most Confusing Target

```python
def most_confusing_target(model, images, true_labels):
    """Select the class model is second most confident about."""
    with torch.no_grad():
        logits = model(images)
        # Set true class logit to -inf
        logits.scatter_(1, true_labels.view(-1, 1), -float('inf'))
        return logits.argmax(dim=1)
```

## Complete Example

```python
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt

# CNN Definition (same as before)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def visualize_targeted_attack(model, image, true_label, target_label, 
                               attack_fn, attack_params, class_names):
    """Visualize a targeted attack."""
    
    # Original prediction
    model.eval()
    with torch.no_grad():
        original_pred = model(image.unsqueeze(0)).argmax().item()
    
    # Generate adversarial example
    target_tensor = torch.tensor([target_label])
    adv_image = attack_fn(model, image.unsqueeze(0), target_tensor, **attack_params)
    
    # Adversarial prediction
    with torch.no_grad():
        adv_pred = model(adv_image).argmax().item()
    
    # Compute perturbation
    perturbation = adv_image.squeeze(0) - image
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(image.squeeze().cpu(), cmap='gray')
    axes[0].set_title(f'Original\nTrue: {class_names[true_label]}\n'
                      f'Pred: {class_names[original_pred]}')
    axes[0].axis('off')
    
    # Amplify perturbation for visibility
    pert_display = 0.5 + 10 * perturbation.squeeze().cpu()
    axes[1].imshow(pert_display.clamp(0, 1), cmap='gray')
    axes[1].set_title(f'Perturbation (×10)\n'
                      f'L∞: {perturbation.abs().max():.4f}\n'
                      f'L2: {perturbation.norm():.4f}')
    axes[1].axis('off')
    
    axes[2].imshow(adv_image.squeeze().cpu(), cmap='gray')
    success = "✓" if adv_pred == target_label else "✗"
    axes[2].set_title(f'Adversarial {success}\n'
                      f'Target: {class_names[target_label]}\n'
                      f'Pred: {class_names[adv_pred]}')
    axes[2].axis('off')
    
    plt.suptitle('Targeted Adversarial Attack', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return adv_pred == target_label


def main():
    # Load data and model
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    test_data = torchvision.datasets.MNIST('./data', train=False, 
                                           download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    
    # Load pre-trained model (assume training code from previous section)
    model = SimpleCNN()
    # model.load_state_dict(torch.load('mnist_cnn.pth'))
    model.eval()
    
    class_names = [str(i) for i in range(10)]
    
    # Find a sample to attack
    for image, label in test_loader:
        if label.item() != 3:  # Find a non-3 image
            break
    
    # Attack: try to make it classify as "3"
    target_label = 3
    
    print("Testing Targeted Attacks:")
    print("="*50)
    
    # Targeted FGSM
    print("\n1. Targeted FGSM (ε=0.3):")
    success = visualize_targeted_attack(
        model, image.squeeze(0), label.item(), target_label,
        targeted_fgsm, {'epsilon': 0.3}, class_names
    )
    print(f"   Success: {success}")
    
    # Targeted PGD
    print("\n2. Targeted PGD (ε=0.3, 40 iterations):")
    success = visualize_targeted_attack(
        model, image.squeeze(0), label.item(), target_label,
        lambda m, x, t, **kw: targeted_pgd(m, x, t, epsilon=0.3, alpha=0.01, num_iter=40),
        {}, class_names
    )
    print(f"   Success: {success}")


if __name__ == "__main__":
    main()
```

## Summary

| Attack | Steps | Optimization | Best For |
|--------|-------|--------------|----------|
| Targeted FGSM | 1 | None | Quick attacks |
| Targeted BIM | Multiple | None | Iterative refinement |
| Targeted PGD | Multiple | Projected GD | Strong attacks |
| C&W | Many | Adam | Minimal perturbations |

Targeted attacks are more challenging but enable precise control over model misbehavior. They're particularly dangerous in security-critical applications like facial recognition and autonomous systems.
