"""
Model Quantization - Static and Dynamic Quantization

This module demonstrates:
- Dynamic quantization (weights only)
- Static quantization (weights + activations)
- Quantization-aware training (QAT)
- Post-training quantization (PTQ)
- Comparing model sizes and performance
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import numpy as np
import time
import os
from typing import Callable, Tuple
from copy import deepcopy


class CNNModel(nn.Module):
    """Example CNN model for quantization"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class LSTMModel(nn.Module):
    """Example LSTM model for quantization"""
    def __init__(self, input_size=10, hidden_size=20, num_layers=2, num_classes=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def dynamic_quantization_demo(model: nn.Module, example_input: torch.Tensor):
    """
    Dynamic Quantization - Quantizes weights at model load time
    
    Best for:
    - RNNs, LSTMs, Transformers
    - Models with variable input sizes
    - Quick deployment without calibration
    
    Args:
        model: PyTorch model
        example_input: Example input for testing
    """
    print("\n=== Dynamic Quantization ===")
    
    # Quantize model (only weights are quantized)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},  # Layers to quantize
        dtype=torch.qint8
    )
    
    # Compare sizes
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")
    
    # Compare inference time
    original_time = measure_inference_time(model, example_input)
    quantized_time = measure_inference_time(quantized_model, example_input)
    
    print(f"Original inference time: {original_time*1000:.2f} ms")
    print(f"Quantized inference time: {quantized_time*1000:.2f} ms")
    print(f"Speedup: {original_time/quantized_time:.2f}x")
    
    return quantized_model


def static_quantization_demo(
    model: nn.Module,
    calibration_loader,
    example_input: torch.Tensor
):
    """
    Static Quantization - Quantizes both weights and activations
    
    Best for:
    - CNNs
    - Models with fixed input sizes
    - Maximum performance gain
    
    Requires calibration data to determine activation ranges
    
    Args:
        model: PyTorch model
        calibration_loader: DataLoader with calibration data
        example_input: Example input for testing
    """
    print("\n=== Static Quantization ===")
    
    # Set quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for quantization
    model_prepared = torch.quantization.prepare(model, inplace=False)
    
    # Calibrate with representative dataset
    print("Calibrating model...")
    model_prepared.eval()
    with torch.no_grad():
        for data, _ in calibration_loader:
            model_prepared(data)
    
    # Convert to quantized model
    quantized_model = torch.quantization.convert(model_prepared, inplace=False)
    
    # Compare sizes
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    
    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")
    
    # Compare inference time
    original_time = measure_inference_time(model, example_input)
    quantized_time = measure_inference_time(quantized_model, example_input)
    
    print(f"Original inference time: {original_time*1000:.2f} ms")
    print(f"Quantized inference time: {quantized_time*1000:.2f} ms")
    print(f"Speedup: {original_time/quantized_time:.2f}x")
    
    return quantized_model


class QATModel(nn.Module):
    """Model prepared for Quantization-Aware Training"""
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dequant(x)
        return x
    
    def fuse_model(self):
        """Fuse conv+bn+relu for better quantization"""
        torch.quantization.fuse_modules(
            self,
            [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']],
            inplace=True
        )


def quantization_aware_training_demo(model: QATModel, train_loader, num_epochs=3):
    """
    Quantization-Aware Training - Train with quantization in mind
    
    Best for:
    - Models that lose significant accuracy with PTQ
    - When you have training resources
    - Maximum accuracy retention
    
    Args:
        model: QAT-prepared model
        train_loader: Training data loader
        num_epochs: Number of training epochs
    """
    print("\n=== Quantization-Aware Training ===")
    
    # Fuse layers
    model.fuse_model()
    
    # Set QAT config
    model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    
    # Prepare for QAT
    model_prepared = torch.quantization.prepare_qat(model, inplace=False)
    
    # Training loop
    optimizer = torch.optim.Adam(model_prepared.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training for {num_epochs} epochs...")
    model_prepared.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model_prepared(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 10 == 9:
                print(f"[Epoch {epoch+1}, Batch {i+1}] Loss: {running_loss/10:.3f}")
                running_loss = 0.0
    
    # Convert to quantized model
    model_prepared.eval()
    quantized_model = torch.quantization.convert(model_prepared, inplace=False)
    
    print("QAT completed!")
    return quantized_model


def onnx_quantization_demo(onnx_model_path: str, output_path: str, calibration_data_reader=None):
    """
    ONNX Quantization - Quantize ONNX models
    
    Args:
        onnx_model_path: Path to ONNX model
        output_path: Path to save quantized model
        calibration_data_reader: Optional calibration data reader
    """
    from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType
    
    print("\n=== ONNX Quantization ===")
    
    # Dynamic quantization (no calibration needed)
    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=output_path,
        weight_type=QuantType.QInt8
    )
    
    # For static quantization, you would use:
    # quantize_static(
    #     model_input=onnx_model_path,
    #     model_output=output_path,
    #     calibration_data_reader=calibration_data_reader
    # )
    
    # Compare sizes
    original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
    quantized_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"Original ONNX model size: {original_size:.2f} MB")
    print(f"Quantized ONNX model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {original_size/quantized_size:.2f}x")


def get_model_size(model: nn.Module) -> float:
    """
    Get model size in MB
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in MB
    """
    torch.save(model.state_dict(), "temp_model.pth")
    size = os.path.getsize("temp_model.pth") / (1024 * 1024)
    os.remove("temp_model.pth")
    return size


def measure_inference_time(model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 100) -> float:
    """
    Measure average inference time
    
    Args:
        model: PyTorch model
        input_tensor: Input tensor
        num_runs: Number of runs for averaging
    
    Returns:
        Average inference time in seconds
    """
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Measure
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end_time = time.time()
    
    return (end_time - start_time) / num_runs


def compare_model_outputs(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_input: torch.Tensor
) -> Tuple[float, float]:
    """
    Compare outputs between original and quantized models
    
    Args:
        original_model: Original model
        quantized_model: Quantized model
        test_input: Test input
    
    Returns:
        Tuple of (mean_absolute_error, max_absolute_error)
    """
    original_model.eval()
    quantized_model.eval()
    
    with torch.no_grad():
        original_output = original_model(test_input)
        quantized_output = quantized_model(test_input)
    
    mae = torch.mean(torch.abs(original_output - quantized_output)).item()
    max_error = torch.max(torch.abs(original_output - quantized_output)).item()
    
    print(f"\n=== Output Comparison ===")
    print(f"Mean Absolute Error: {mae:.6f}")
    print(f"Max Absolute Error: {max_error:.6f}")
    
    return mae, max_error


def quantization_best_practices():
    """Print quantization best practices"""
    print("""
    === Quantization Best Practices ===
    
    1. CHOOSE THE RIGHT METHOD:
       - Dynamic: LSTMs, Transformers, quick deployment
       - Static: CNNs, maximum performance
       - QAT: When accuracy loss is unacceptable
    
    2. CALIBRATION (for static quantization):
       - Use representative data
       - 100-1000 samples typically sufficient
       - Should cover input distribution
    
    3. ACCURACY VALIDATION:
       - Always validate on full test set
       - Compare metrics, not just output values
       - Acceptable accuracy drop: 1-2%
    
    4. LAYER SELECTION:
       - Not all layers benefit equally
       - Linear/Conv layers: high benefit
       - Attention layers: moderate benefit
       - Embedding layers: usually skip
    
    5. DEPLOYMENT CONSIDERATIONS:
       - Check hardware support (INT8, INT4)
       - Verify runtime compatibility
       - Benchmark on target device
    
    6. COMMON ISSUES:
       - Accuracy drop > 5%: Try QAT
       - Activation outliers: Use per-channel quantization
       - Saturation: Adjust quantization ranges
    """)


def demo_all_quantization_methods():
    """Demonstrate all quantization methods"""
    print("=== Quantization Methods Demo ===\n")
    
    # 1. Dynamic Quantization (LSTM)
    print("\n" + "="*50)
    print("1. DYNAMIC QUANTIZATION (LSTM)")
    print("="*50)
    lstm_model = LSTMModel()
    lstm_input = torch.randn(8, 20, 10)  # (batch, seq_len, features)
    dynamic_quantized = dynamic_quantization_demo(lstm_model, lstm_input)
    
    # 2. Create dummy calibration data for static quantization
    print("\n" + "="*50)
    print("2. STATIC QUANTIZATION (CNN)")
    print("="*50)
    
    # Create synthetic calibration data
    calibration_data = [(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))) for _ in range(10)]
    calibration_loader = calibration_data  # Simplified for demo
    
    cnn_model = CNNModel()
    cnn_input = torch.randn(1, 3, 32, 32)
    
    # Note: Full static quantization requires proper setup
    print("Static quantization requires proper model preparation.")
    print("See code for full implementation details.")
    
    # 3. Show best practices
    quantization_best_practices()


if __name__ == "__main__":
    demo_all_quantization_methods()
