# Defense Mechanisms Against Adversarial Attacks

This section covers techniques to defend neural networks against adversarial attacks, with special emphasis on GAN-related defenses.

## Defense Categories

### Overview

| Category | Approach | Example |
|----------|----------|---------|
| **Adversarial Training** | Train on adversarial examples | PGD-AT, TRADES |
| **Input Preprocessing** | Transform inputs before classification | Denoising, quantization |
| **Certified Defenses** | Provide provable robustness guarantees | Randomized smoothing |
| **Detection** | Detect adversarial inputs | Statistical tests |
| **Ensemble Methods** | Combine multiple models | Diverse ensemble |

## Adversarial Training

### Basic Adversarial Training

Train the model on a mixture of clean and adversarial examples:

```python
import torch
import torch.nn as nn
import torch.optim as optim

def generate_pgd_attack(model, images, labels, epsilon, alpha, num_iter):
    """Generate PGD adversarial examples."""
    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, 0, 1).detach()
    
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(num_iter):
        adv_images.requires_grad_(True)
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        adv_images = adv_images + alpha * adv_images.grad.sign()
        perturbation = torch.clamp(adv_images - images, -epsilon, epsilon)
        adv_images = torch.clamp(images + perturbation, 0, 1).detach()
    
    return adv_images


def adversarial_training(model, train_loader, epochs, epsilon=0.3, 
                         alpha=0.01, num_iter=10, device='cpu'):
    """
    Train model with adversarial examples.
    
    Args:
        model: Neural network
        train_loader: Training data
        epochs: Training epochs
        epsilon: Perturbation bound
        alpha: PGD step size
        num_iter: PGD iterations
        device: Device
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Generate adversarial examples
            model.eval()
            adv_images = generate_pgd_attack(
                model, images, labels, epsilon, alpha, num_iter
            )
            model.train()
            
            # Train on both clean and adversarial
            optimizer.zero_grad()
            
            # Clean loss
            clean_outputs = model(images)
            clean_loss = criterion(clean_outputs, labels)
            
            # Adversarial loss
            adv_outputs = model(adv_images)
            adv_loss = criterion(adv_outputs, labels)
            
            # Combined loss
            loss = 0.5 * clean_loss + 0.5 * adv_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = adv_outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        print(f'Epoch {epoch+1}/{epochs}: '
              f'Loss: {total_loss/len(train_loader):.4f}, '
              f'Adv Acc: {100*correct/total:.2f}%')
    
    return model
```

### TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)

TRADES balances clean accuracy and adversarial robustness:

```python
def trades_loss(model, x_natural, y, optimizer, step_size=0.003, 
                epsilon=0.031, perturb_steps=10, beta=6.0):
    """
    TRADES loss function.
    
    Loss = CE(f(x), y) + β * KL(f(x) || f(x'))
    
    Args:
        model: Neural network
        x_natural: Clean inputs
        y: Labels
        optimizer: Optimizer
        step_size: PGD step size
        epsilon: Perturbation bound
        perturb_steps: Number of PGD steps
        beta: Regularization parameter
    
    Returns:
        Total loss
    """
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    
    model.eval()
    batch_size = len(x_natural)
    
    # Generate adversarial examples
    x_adv = x_natural.detach() + 0.001 * torch.randn_like(x_natural)
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_(True)
        
        with torch.enable_grad():
            # KL divergence between clean and adversarial predictions
            loss_kl = criterion_kl(
                torch.log_softmax(model(x_adv), dim=1),
                torch.softmax(model(x_natural), dim=1)
            )
        
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()
    
    x_adv = x_adv.detach()
    
    optimizer.zero_grad()
    
    # Natural loss
    logits_natural = model(x_natural)
    loss_natural = criterion_ce(logits_natural, y)
    
    # Robust loss (KL between clean and adversarial)
    logits_adv = model(x_adv)
    loss_robust = criterion_kl(
        torch.log_softmax(logits_adv, dim=1),
        torch.softmax(logits_natural, dim=1)
    )
    
    # Combined loss
    loss = loss_natural + beta * loss_robust
    
    return loss
```

## Input Preprocessing Defenses

### JPEG Compression

```python
import io
from PIL import Image
import torchvision.transforms as transforms

def jpeg_compression_defense(images, quality=75):
    """
    Apply JPEG compression to remove adversarial perturbations.
    
    Args:
        images: Input tensor (B, C, H, W) in [0, 1]
        quality: JPEG quality (1-100)
    
    Returns:
        Compressed images
    """
    defended_images = []
    
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()
    
    for img in images:
        # Convert to PIL Image
        pil_img = to_pil(img)
        
        # Apply JPEG compression
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed_img = Image.open(buffer)
        
        # Convert back to tensor
        tensor_img = to_tensor(compressed_img)
        defended_images.append(tensor_img)
    
    return torch.stack(defended_images)
```

### Gaussian Noise Defense

```python
def gaussian_noise_defense(images, std=0.05):
    """
    Add Gaussian noise to disrupt adversarial patterns.
    
    Args:
        images: Input tensor
        std: Standard deviation of noise
    
    Returns:
        Noisy images
    """
    noise = torch.randn_like(images) * std
    defended_images = torch.clamp(images + noise, 0, 1)
    return defended_images
```

### Feature Squeezing

```python
def bit_depth_reduction(images, bits=5):
    """
    Reduce bit depth to remove fine perturbations.
    
    Args:
        images: Input tensor in [0, 1]
        bits: Number of bits to keep
    
    Returns:
        Quantized images
    """
    levels = 2 ** bits
    return torch.round(images * (levels - 1)) / (levels - 1)


def spatial_smoothing(images, kernel_size=3):
    """
    Apply spatial smoothing filter.
    
    Args:
        images: Input tensor (B, C, H, W)
        kernel_size: Smoothing kernel size
    
    Returns:
        Smoothed images
    """
    # Create averaging kernel
    kernel = torch.ones(1, 1, kernel_size, kernel_size) / (kernel_size ** 2)
    kernel = kernel.to(images.device)
    
    # Apply per channel
    smoothed = []
    for c in range(images.size(1)):
        channel = images[:, c:c+1, :, :]
        smoothed_channel = torch.nn.functional.conv2d(
            channel, kernel, padding=kernel_size//2
        )
        smoothed.append(smoothed_channel)
    
    return torch.cat(smoothed, dim=1)


class FeatureSqueezing:
    """Feature squeezing defense combining multiple techniques."""
    
    def __init__(self, bit_depth=5, median_filter=True, kernel_size=3):
        self.bit_depth = bit_depth
        self.median_filter = median_filter
        self.kernel_size = kernel_size
    
    def __call__(self, images):
        # Bit depth reduction
        defended = bit_depth_reduction(images, self.bit_depth)
        
        # Spatial smoothing
        if self.median_filter:
            defended = spatial_smoothing(defended, self.kernel_size)
        
        return defended
```

## Detection-Based Defenses

### Statistical Detection

```python
import numpy as np
from scipy import stats

class AdversarialDetector:
    """Detect adversarial examples using statistical methods."""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.clean_stats = None
    
    def collect_clean_statistics(self, dataloader, num_batches=100):
        """Collect statistics from clean data."""
        self.model.eval()
        
        confidences = []
        entropies = []
        
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                
                # Maximum confidence
                max_conf = probs.max(dim=1)[0]
                confidences.extend(max_conf.cpu().numpy())
                
                # Entropy
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                entropies.extend(entropy.cpu().numpy())
        
        self.clean_stats = {
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'entropy_mean': np.mean(entropies),
            'entropy_std': np.std(entropies),
        }
        
        return self.clean_stats
    
    def detect(self, images, threshold=2.0):
        """
        Detect adversarial examples.
        
        Args:
            images: Input images
            threshold: Z-score threshold for detection
        
        Returns:
            Boolean mask (True = detected as adversarial)
        """
        if self.clean_stats is None:
            raise ValueError("Must collect clean statistics first")
        
        self.model.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1)
            
            max_conf = probs.max(dim=1)[0].cpu().numpy()
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).cpu().numpy()
        
        # Z-scores
        conf_z = np.abs(max_conf - self.clean_stats['confidence_mean']) / self.clean_stats['confidence_std']
        ent_z = np.abs(entropy - self.clean_stats['entropy_mean']) / self.clean_stats['entropy_std']
        
        # Detect if either z-score exceeds threshold
        is_adversarial = (conf_z > threshold) | (ent_z > threshold)
        
        return is_adversarial
```

### Feature-Based Detection

```python
class FeatureBasedDetector:
    """Detect adversarial examples using intermediate features."""
    
    def __init__(self, model, layer_name, device='cpu'):
        self.model = model
        self.layer_name = layer_name
        self.device = device
        self.features = None
        self.clean_features = None
        
        # Register hook to capture features
        self._register_hook()
    
    def _register_hook(self):
        def hook_fn(module, input, output):
            self.features = output.detach()
        
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                module.register_forward_hook(hook_fn)
                break
    
    def collect_clean_features(self, dataloader, num_batches=100):
        """Collect feature statistics from clean data."""
        self.model.eval()
        
        all_features = []
        
        with torch.no_grad():
            for i, (images, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                
                images = images.to(self.device)
                _ = self.model(images)
                
                # Flatten features
                flat_features = self.features.view(self.features.size(0), -1)
                all_features.append(flat_features.cpu())
        
        all_features = torch.cat(all_features, dim=0)
        
        self.clean_features = {
            'mean': all_features.mean(dim=0),
            'std': all_features.std(dim=0),
        }
    
    def detect(self, images, threshold=3.0):
        """Detect adversarial examples based on feature deviation."""
        self.model.eval()
        
        with torch.no_grad():
            images = images.to(self.device)
            _ = self.model(images)
            
            flat_features = self.features.view(self.features.size(0), -1).cpu()
        
        # Compute z-scores
        z_scores = (flat_features - self.clean_features['mean']) / (self.clean_features['std'] + 1e-10)
        max_z = z_scores.abs().max(dim=1)[0]
        
        return max_z > threshold
```

## Differential Privacy Defense

```python
def add_differential_privacy_noise(model, epsilon=1.0, delta=1e-5):
    """
    Add noise to model gradients for differential privacy.
    
    Args:
        model: Neural network
        epsilon: Privacy budget
        delta: Failure probability
    
    Returns:
        Noise scale used
    """
    # Simplified DP noise addition
    sensitivity = 1.0  # Assumed gradient sensitivity
    noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * noise_scale
                param.grad.add_(noise)
    
    return noise_scale
```

## Ensemble Defense

```python
class EnsembleDefense:
    """Defense using ensemble of diverse models."""
    
    def __init__(self, models, device='cpu'):
        """
        Args:
            models: List of trained models
            device: Device
        """
        self.models = models
        self.device = device
        
        for model in self.models:
            model.to(device)
            model.eval()
    
    def predict(self, images, voting='soft'):
        """
        Make predictions using ensemble.
        
        Args:
            images: Input images
            voting: 'soft' (average probabilities) or 'hard' (majority vote)
        
        Returns:
            Predictions
        """
        images = images.to(self.device)
        
        all_outputs = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(images)
                all_outputs.append(outputs)
        
        if voting == 'soft':
            # Average softmax probabilities
            probs = [torch.softmax(out, dim=1) for out in all_outputs]
            avg_probs = torch.stack(probs).mean(dim=0)
            predictions = avg_probs.argmax(dim=1)
        else:
            # Majority voting
            votes = torch.stack([out.argmax(dim=1) for out in all_outputs])
            predictions = torch.mode(votes, dim=0)[0]
        
        return predictions
    
    def get_disagreement(self, images):
        """
        Compute disagreement among ensemble members.
        
        High disagreement may indicate adversarial input.
        """
        images = images.to(self.device)
        
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(images)
                predictions.append(outputs.argmax(dim=1))
        
        predictions = torch.stack(predictions)  # (num_models, batch_size)
        
        # Count unique predictions per sample
        disagreement = []
        for i in range(predictions.size(1)):
            unique_preds = predictions[:, i].unique().size(0)
            disagreement.append(unique_preds / len(self.models))
        
        return torch.tensor(disagreement)
```

## Summary

| Defense | Type | Strength | Weakness |
|---------|------|----------|----------|
| Adversarial Training | Training | Strong empirical | Computationally expensive |
| TRADES | Training | Balances accuracy/robustness | Requires tuning β |
| JPEG Compression | Preprocessing | Simple | Reduces image quality |
| Feature Squeezing | Preprocessing | Low overhead | Limited effectiveness |
| Detection | Runtime | Doesn't modify model | Can be evaded |
| Ensemble | Architecture | Robust to transfer | Multiple model overhead |

No single defense provides complete protection. Practical systems often combine multiple techniques for defense in depth.
