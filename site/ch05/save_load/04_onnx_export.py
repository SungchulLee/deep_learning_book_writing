#!/usr/bin/env python3
"""
============================================================
Exporting PyTorch Models to ONNX Format
============================================================

ONNX (Open Neural Network Exchange) is an open format for
representing deep learning models. It enables model portability
across different frameworks and deployment platforms.

Topics:
- Why export to ONNX
- Basic ONNX export
- Dynamic input shapes
- Model optimization
- Verification and validation
"""

import torch
import torch.nn as nn
import torch.onnx

try:
    import onnx
    import onnxruntime
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    print("WARNING: onnx and onnxruntime not installed")
    print("Install with: pip install onnx onnxruntime")

import numpy as np
import os

print("=" * 70)
print("ONNX EXPORT TUTORIAL")
print("=" * 70)

# Define example model
class ConvNet(nn.Module):
    """
    Simple CNN for MNIST-style classification.
    Demonstrates various layer types for ONNX export.
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Create and prepare model
model = ConvNet()
model.eval()  # IMPORTANT: Set to eval mode before export

print("\nModel created and set to eval mode")

# Create dummy input
batch_size = 1
dummy_input = torch.randn(batch_size, 1, 28, 28)

# Define output path
onnx_path = "convnet_model.onnx"

print(f"\nExporting model to ONNX format...")
print(f"Input shape: {dummy_input.shape}")

# Export the model
torch.onnx.export(
    model,                          # Model to export
    dummy_input,                    # Example input tensor
    onnx_path,                      # Output file path
    export_params=True,             # Store trained parameters
    opset_version=11,               # ONNX version
    do_constant_folding=True,       # Optimize constant folding
    input_names=['input'],          # Input tensor name
    output_names=['output'],        # Output tensor name
    dynamic_axes={                  # Dynamic input/output shapes
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Model exported successfully to '{onnx_path}'")
file_size = os.path.getsize(onnx_path) / 1024
print(f"File size: {file_size:.2f} KB")

if HAS_ONNX:
    # Verify ONNX model
    print("\n" + "=" * 70)
    print("VERIFYING ONNX MODEL")
    print("=" * 70)
    
    onnx_model = onnx.load(onnx_path)
    
    try:
        onnx.checker.check_model(onnx_model)
        print("\nONNX model is valid")
    except Exception as e:
        print(f"\nONNX model validation failed: {e}")
    
    # Run inference with ONNX Runtime
    print("\n" + "=" * 70)
    print("RUNNING INFERENCE WITH ONNX RUNTIME")
    print("=" * 70)
    
    ort_session = onnxruntime.InferenceSession(onnx_path)
    input_data = dummy_input.numpy()
    
    print("\nRunning inference...")
    ort_inputs = {ort_session.get_inputs()[0].name: input_data}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"Inference complete")
    print(f"Output shape: {ort_outputs[0].shape}")
    
    # Compare with PyTorch
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()
    
    max_diff = np.abs(torch_output - ort_outputs[0]).max()
    print(f"\nMax difference: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("Outputs match!")

# Cleanup
if os.path.exists(onnx_path):
    os.remove(onnx_path)
    print(f"\nCleaned up '{onnx_path}'")

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)
