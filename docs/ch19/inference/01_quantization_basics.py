"""
BEGINNER LEVEL: Post-Training Quantization (PTQ)

This script demonstrates the basics of quantizing a pre-trained neural network
to reduce its memory footprint and potentially speed up inference.

Topics Covered:
- What is quantization and why it matters
- FP32 → INT8 quantization
- Dynamic vs Static quantization
- Measuring accuracy and size trade-offs

Mathematical Background:
- Quantization maps: x_float → x_int8 via: x_int = round((x_float - zero_point) / scale)
- Dequantization: x_float_approx = scale * x_int + zero_point
- Quantization error: ε = x_float - x_float_approx

Prerequisites:
- Understanding of neural networks
- Basic PyTorch knowledge
- Familiarity with MNIST/CIFAR datasets
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import copy
import numpy as np
import matplotlib.pyplot as plt

# Import our utility functions
from utils import (
    get_model_size, 
    compare_model_sizes,
    evaluate_accuracy,
    compare_accuracies,
    measure_inference_time,
    seed_everything
)


# ============================================================================
# STEP 1: DEFINE A SIMPLE CNN MODEL
# ============================================================================

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network for image classification.
    
    Architecture:
    - Conv1: 1 → 32 channels (3x3 kernel)
    - ReLU + MaxPool
    - Conv2: 32 → 64 channels (3x3 kernel)  
    - ReLU + MaxPool
    - Fully Connected: 64*7*7 → 128 → 10
    
    This is a toy model suitable for MNIST or Fashion-MNIST.
    """
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After 2 pooling layers, 28x28 becomes 7x7
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
        # Dropout for regularization (only during training)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = self.conv1(x)        # (B, 1, 28, 28) → (B, 32, 28, 28)
        x = self.relu(x)
        x = self.pool(x)         # (B, 32, 28, 28) → (B, 32, 14, 14)
        
        # Second convolutional block
        x = self.conv2(x)        # (B, 32, 14, 14) → (B, 64, 14, 14)
        x = self.relu(x)
        x = self.pool(x)         # (B, 64, 14, 14) → (B, 64, 7, 7)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (B, 64, 7, 7) → (B, 64*7*7)
        
        # Fully connected layers
        x = self.fc1(x)          # (B, 3136) → (B, 128)
        x = self.relu(x)
        x = self.dropout(x)      # Apply dropout
        x = self.fc2(x)          # (B, 128) → (B, 10)
        
        return x


# ============================================================================
# STEP 2: PREPARE DATASET
# ============================================================================

def get_mnist_dataloaders(batch_size=64):
    """
    Load MNIST dataset and create train/test dataloaders.
    
    Args:
        batch_size: Batch size for dataloaders
        
    Returns:
        train_loader, test_loader
    """
    # Normalization parameters for MNIST
    # These are precomputed: mean=0.1307, std=0.3081
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Download and load training data
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Download and load test data
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, test_loader


# ============================================================================
# STEP 3: TRAIN THE MODEL (OR LOAD PRE-TRAINED)
# ============================================================================

def train_model(model, train_loader, test_loader, epochs=5, device='cpu'):
    """
    Train the model from scratch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        test_loader: Test data loader  
        epochs: Number of training epochs
        device: Device to train on ('cpu' or 'cuda')
        
    Returns:
        Trained model
        
    Note:
        For quantization, we need a well-trained baseline model.
        Poor accuracy before quantization will only get worse!
    """
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("\nTraining the model...")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], "
                      f"Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {running_loss/(batch_idx+1):.4f}, "
                      f"Acc: {100.*correct/total:.2f}%")
        
        # Evaluate on test set
        test_acc = evaluate_accuracy(model, test_loader, device)
        print(f"Epoch [{epoch+1}/{epochs}] Test Accuracy: {test_acc*100:.2f}%\n")
    
    print("Training completed!")
    print("-" * 60)
    return model


# ============================================================================
# STEP 4: POST-TRAINING QUANTIZATION
# ============================================================================

def quantize_model_dynamic(model):
    """
    Apply dynamic quantization to the model.
    
    Dynamic Quantization:
    - Quantizes ONLY weights to INT8 before inference
    - Activations remain in FP32 during computation
    - Quantization happens dynamically at runtime
    - Best for LSTM, RNN, or models where activations dominate compute
    
    Mathematical process:
    For each weight tensor W:
        1. Compute: scale = (W_max - W_min) / 255
        2. Compute: zero_point = round(-W_min / scale)
        3. Quantize: W_int8 = round(W / scale + zero_point)
    
    Args:
        model: FP32 model to quantize
        
    Returns:
        Dynamically quantized model
    """
    # Specify which layers to quantize
    # Linear and Conv2d are good candidates
    quantized_model = quant.quantize_dynamic(
        model,
        # Which layer types to quantize
        qconfig_spec={nn.Linear, nn.Conv2d},
        # Target quantization dtype
        dtype=torch.qint8
    )
    
    return quantized_model


def quantize_model_static(model, train_loader, device='cpu'):
    """
    Apply static quantization to the model.
    
    Static Quantization (More aggressive):
    - Quantizes BOTH weights AND activations to INT8
    - Requires calibration on representative data
    - All computations in INT8 (if hardware supports)
    - Better compression and speed, but needs more care
    
    Process:
    1. Prepare model: Insert observer modules
    2. Calibrate: Run data through model to collect statistics
    3. Convert: Replace FP32 ops with INT8 ops
    
    Args:
        model: FP32 model to quantize
        train_loader: Data loader for calibration
        device: Device for calibration
        
    Returns:
        Statically quantized model
    """
    # Create a copy to avoid modifying the original
    model_to_quantize = copy.deepcopy(model)
    model_to_quantize.eval()
    
    # Step 1: Prepare the model
    # This inserts observers to collect statistics about activations
    model_to_quantize.qconfig = quant.get_default_qconfig('x86')  # or 'qnnpack' for mobile
    quant.prepare(model_to_quantize, inplace=True)
    
    # Step 2: Calibrate
    # Run representative data through the model
    # The observers will record min/max values of activations
    print("Calibrating quantized model (collecting statistics)...")
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(train_loader):
            if batch_idx >= 100:  # Use 100 batches for calibration
                break
            data = data.to(device)
            model_to_quantize(data)
    
    # Step 3: Convert to quantized model
    # This replaces FP32 operations with INT8 operations
    quantized_model = quant.convert(model_to_quantize, inplace=True)
    
    return quantized_model


# ============================================================================
# STEP 5: ANALYZE QUANTIZATION EFFECTS
# ============================================================================

def analyze_weight_distribution(fp32_model, quantized_model, layer_name='conv1'):
    """
    Visualize how quantization affects weight distribution.
    
    This helps understand:
    - What precision loss looks like
    - Whether important weights are preserved
    - The discretization effect of quantization
    
    Args:
        fp32_model: Original FP32 model
        quantized_model: Quantized model
        layer_name: Name of layer to visualize
    """
    # Extract weights from FP32 model
    fp32_weights = None
    for name, param in fp32_model.named_parameters():
        if layer_name in name and 'weight' in name:
            fp32_weights = param.data.cpu().numpy().flatten()
            break
    
    # For quantized model, we need to dequantize to compare
    # Note: This is tricky because quantized weights are stored differently
    # For visualization, we'll compare distributions
    
    if fp32_weights is not None:
        plt.figure(figsize=(14, 5))
        
        # FP32 distribution
        plt.subplot(1, 2, 1)
        plt.hist(fp32_weights, bins=100, alpha=0.7, edgecolor='black')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.title(f'{layer_name} - FP32 Weights')
        plt.grid(True, alpha=0.3)
        
        # Statistics
        plt.subplot(1, 2, 2)
        stats_text = f"""
        FP32 Statistics:
        Mean: {np.mean(fp32_weights):.6f}
        Std:  {np.std(fp32_weights):.6f}
        Min:  {np.min(fp32_weights):.6f}
        Max:  {np.max(fp32_weights):.6f}
        
        INT8 Range: [-128, 127]
        Quantization bins: 256
        
        Expected precision loss:
        ≈ (max-min) / 256 per value
        """
        plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes,
                fontsize=11, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('weight_distribution_comparison.png', dpi=150, bbox_inches='tight')
        print("Weight distribution saved to: weight_distribution_comparison.png")
        plt.show()


def demonstrate_quantization_error(model, test_loader, device='cpu'):
    """
    Demonstrate the quantization error on actual predictions.
    
    This shows:
    - How output logits change after quantization
    - Whether prediction confidence decreases
    - If quantization causes misclassifications
    
    Args:
        model: Original FP32 model
        test_loader: Test data loader
        device: Device to run on
    """
    model.eval()
    
    # Get one batch
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    
    # FP32 predictions
    with torch.no_grad():
        fp32_output = model(data)
        fp32_probs = torch.softmax(fp32_output, dim=1)
        fp32_pred = fp32_output.argmax(dim=1)
    
    # Quantize model
    quantized_model = quantize_model_dynamic(copy.deepcopy(model))
    quantized_model.eval()
    
    # INT8 predictions
    with torch.no_grad():
        int8_output = quantized_model(data)
        int8_probs = torch.softmax(int8_output, dim=1)
        int8_pred = int8_output.argmax(dim=1)
    
    # Analyze differences
    logit_diff = torch.abs(fp32_output - int8_output).mean().item()
    prob_diff = torch.abs(fp32_probs - int8_probs).mean().item()
    agreement = (fp32_pred == int8_pred).float().mean().item()
    
    print("\n" + "="*60)
    print("QUANTIZATION ERROR ANALYSIS")
    print("="*60)
    print(f"Average logit difference:       {logit_diff:.6f}")
    print(f"Average probability difference: {prob_diff:.6f}")
    print(f"Prediction agreement:           {agreement*100:.2f}%")
    print("="*60 + "\n")
    
    # Show some example differences
    print("Sample predictions comparison (first 5):")
    print("-" * 60)
    for i in range(min(5, len(target))):
        print(f"Sample {i+1}:")
        print(f"  True label: {target[i].item()}")
        print(f"  FP32 pred:  {fp32_pred[i].item()} (conf: {fp32_probs[i].max():.4f})")
        print(f"  INT8 pred:  {int8_pred[i].item()} (conf: {int8_probs[i].max():.4f})")
        print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function demonstrating post-training quantization.
    """
    # Set random seed for reproducibility
    seed_everything(42)
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # ========================================================================
    # PART A: PREPARE MODEL AND DATA
    # ========================================================================
    
    print("="*60)
    print("PART A: MODEL PREPARATION")
    print("="*60 + "\n")
    
    # Load data
    train_loader, test_loader = get_mnist_dataloaders(batch_size=64)
    
    # Create model
    model = SimpleCNN(num_classes=10)
    
    # Either train from scratch or load pre-trained
    # For speed in this demo, we'll train briefly (or load if available)
    try:
        # Try to load pre-trained model
        model.load_state_dict(torch.load('simple_cnn_mnist.pth', map_location=device))
        print("Loaded pre-trained model from: simple_cnn_mnist.pth")
    except:
        # Train from scratch
        print("No pre-trained model found. Training from scratch...")
        model = train_model(model, train_loader, test_loader, epochs=3, device=device)
        # Save for future use
        torch.save(model.state_dict(), 'simple_cnn_mnist.pth')
        print("Model saved to: simple_cnn_mnist.pth")
    
    # Evaluate baseline accuracy
    print("\nEvaluating baseline FP32 model...")
    baseline_acc = evaluate_accuracy(model, test_loader, device, max_batches=None)
    print(f"Baseline FP32 Accuracy: {baseline_acc*100:.2f}%")
    
    # ========================================================================
    # PART B: DYNAMIC QUANTIZATION
    # ========================================================================
    
    print("\n" + "="*60)
    print("PART B: DYNAMIC QUANTIZATION")
    print("="*60 + "\n")
    
    # Quantize the model
    print("Applying dynamic quantization...")
    dynamic_quantized_model = quantize_model_dynamic(copy.deepcopy(model))
    
    # Compare sizes
    compare_model_sizes(
        model, 
        dynamic_quantized_model,
        original_dtype=torch.float32,
        compressed_dtype=torch.qint8
    )
    
    # Compare accuracies
    compare_accuracies(model, dynamic_quantized_model, test_loader, device)
    
    # Analyze quantization error
    demonstrate_quantization_error(model, test_loader, device)
    
    # ========================================================================
    # PART C: STATIC QUANTIZATION
    # ========================================================================
    
    print("\n" + "="*60)
    print("PART C: STATIC QUANTIZATION")
    print("="*60 + "\n")
    
    # Note: Static quantization can be tricky and may not work for all models
    # It requires the model to be in eval mode and use specific operations
    print("Static quantization requires calibration...")
    print("(This quantizes both weights AND activations)\n")
    
    try:
        static_quantized_model = quantize_model_static(copy.deepcopy(model), train_loader, device)
        
        # Compare sizes
        compare_model_sizes(
            model,
            static_quantized_model,
            original_dtype=torch.float32,
            compressed_dtype=torch.qint8
        )
        
        # Compare accuracies
        compare_accuracies(model, static_quantized_model, test_loader, device)
        
    except Exception as e:
        print(f"Static quantization failed: {e}")
        print("This is common and depends on model architecture.")
        print("Some operations may not support quantization.")
    
    # ========================================================================
    # PART D: VISUALIZATIONS
    # ========================================================================
    
    print("\n" + "="*60)
    print("PART D: VISUALIZATIONS")
    print("="*60 + "\n")
    
    # Analyze weight distributions
    analyze_weight_distribution(model, dynamic_quantized_model, layer_name='conv1')
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*60)
    print("SUMMARY: KEY TAKEAWAYS")
    print("="*60)
    print("""
    1. Dynamic Quantization:
       ✓ Weights → INT8, Activations → FP32
       ✓ Easy to apply, no calibration needed
       ✓ ~4x size reduction
       ✓ Minimal accuracy loss (typically <0.5%)
       ✓ Best for: RNNs, LSTMs, attention models
    
    2. Static Quantization:
       ✓ Weights → INT8, Activations → INT8
       ✓ Requires calibration data
       ✓ ~4x size reduction + potential speed gains
       ✓ May lose more accuracy (1-2%)
       ✓ Best for: CNNs with hardware INT8 support
    
    3. When to Use Quantization:
       - Deploying to mobile/edge devices
       - Reducing cloud inference costs
       - Limited memory environments
       - Need for faster inference
    
    4. Important Considerations:
       - Always measure actual speedup (hardware dependent)
       - First/last layers are most sensitive
       - Batch normalization can complicate quantization
       - Consider QAT if accuracy drop is too large
    """)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
