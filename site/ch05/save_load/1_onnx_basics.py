"""
ONNX Basics - Model Conversion and Inference

This module demonstrates:
- Converting PyTorch/TensorFlow models to ONNX format
- Loading and running ONNX models
- Checking model validity
- Optimizing ONNX models
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Tuple


class SimpleModel(nn.Module):
    """Example PyTorch model for demonstration"""
    def __init__(self, input_size: int = 10, hidden_size: int = 20, output_size: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def convert_pytorch_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    onnx_path: str,
    input_names: List[str] = ['input'],
    output_names: List[str] = ['output'],
    dynamic_axes: Dict = None
) -> None:
    """
    Convert PyTorch model to ONNX format
    
    Args:
        model: PyTorch model
        dummy_input: Example input tensor for tracing
        onnx_path: Path to save ONNX model
        input_names: Names for input nodes
        output_names: Names for output nodes
        dynamic_axes: Dynamic axes for variable input shapes
    """
    model.eval()
    
    # Dynamic axes allow variable batch sizes and sequence lengths
    if dynamic_axes is None:
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,  # Use appropriate ONNX opset version
        do_constant_folding=True,  # Optimize by folding constants
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    print(f"Model exported to {onnx_path}")


def verify_onnx_model(onnx_path: str) -> bool:
    """
    Verify the ONNX model is valid
    
    Args:
        onnx_path: Path to ONNX model
    
    Returns:
        True if model is valid
    """
    try:
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("✓ ONNX model is valid")
        return True
    except Exception as e:
        print(f"✗ ONNX model validation failed: {e}")
        return False


def print_onnx_model_info(onnx_path: str) -> None:
    """Print information about ONNX model"""
    model = onnx.load(onnx_path)
    
    print("\n=== Model Information ===")
    print(f"IR Version: {model.ir_version}")
    print(f"Producer: {model.producer_name}")
    print(f"Opset Version: {model.opset_import[0].version}")
    
    print("\n=== Inputs ===")
    for input_tensor in model.graph.input:
        print(f"Name: {input_tensor.name}")
        print(f"Shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")
        print(f"Type: {input_tensor.type.tensor_type.elem_type}")
    
    print("\n=== Outputs ===")
    for output_tensor in model.graph.output:
        print(f"Name: {output_tensor.name}")
        print(f"Shape: {[d.dim_value for d in output_tensor.type.tensor_type.shape.dim]}")


class ONNXInferenceSession:
    """Wrapper for ONNX Runtime inference"""
    
    def __init__(self, onnx_path: str, providers: List[str] = None):
        """
        Initialize ONNX Runtime session
        
        Args:
            onnx_path: Path to ONNX model
            providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"Loaded model with providers: {self.session.get_providers()}")
    
    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference
        
        Args:
            input_data: Input numpy array
        
        Returns:
            Model output
        """
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_data}
        )
        return outputs[0]
    
    def get_model_metadata(self) -> Dict:
        """Get model metadata"""
        return {
            'inputs': [(i.name, i.shape, i.type) for i in self.session.get_inputs()],
            'outputs': [(o.name, o.shape, o.type) for o in self.session.get_outputs()],
            'providers': self.session.get_providers()
        }


def compare_pytorch_onnx_outputs(
    pytorch_model: nn.Module,
    onnx_path: str,
    test_input: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> Tuple[bool, float]:
    """
    Compare outputs between PyTorch and ONNX models
    
    Args:
        pytorch_model: Original PyTorch model
        onnx_path: Path to ONNX model
        test_input: Test input tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
    
    Returns:
        Tuple of (match, max_difference)
    """
    # PyTorch inference
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()
    
    # ONNX inference
    onnx_session = ONNXInferenceSession(onnx_path)
    onnx_output = onnx_session.predict(test_input.numpy())
    
    # Compare
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    match = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)
    
    print(f"\n=== Output Comparison ===")
    print(f"Max difference: {max_diff}")
    print(f"Outputs match (rtol={rtol}, atol={atol}): {match}")
    
    return match, max_diff


def optimize_onnx_model(input_path: str, output_path: str) -> None:
    """
    Optimize ONNX model using ONNX optimizer
    
    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model
    """
    import onnx
    from onnx import optimizer
    
    # Load model
    model = onnx.load(input_path)
    
    # Apply optimizations
    passes = [
        'eliminate_identity',
        'eliminate_nop_transpose',
        'eliminate_nop_pad',
        'eliminate_unused_initializer',
        'fuse_consecutive_transposes',
        'fuse_transpose_into_gemm',
        'fuse_bn_into_conv',
    ]
    
    optimized_model = optimizer.optimize(model, passes)
    
    # Save optimized model
    onnx.save(optimized_model, output_path)
    print(f"Optimized model saved to {output_path}")


def demo_onnx_workflow():
    """Demonstrate complete ONNX workflow"""
    print("=== ONNX Workflow Demo ===\n")
    
    # 1. Create PyTorch model
    print("1. Creating PyTorch model...")
    model = SimpleModel(input_size=10, hidden_size=20, output_size=5)
    
    # 2. Convert to ONNX
    print("\n2. Converting to ONNX...")
    dummy_input = torch.randn(1, 10)
    onnx_path = "simple_model.onnx"
    convert_pytorch_to_onnx(model, dummy_input, onnx_path)
    
    # 3. Verify model
    print("\n3. Verifying ONNX model...")
    verify_onnx_model(onnx_path)
    
    # 4. Print model info
    print_onnx_model_info(onnx_path)
    
    # 5. Run inference
    print("\n4. Running ONNX inference...")
    onnx_session = ONNXInferenceSession(onnx_path)
    test_input = torch.randn(5, 10)  # Batch of 5
    output = onnx_session.predict(test_input.numpy())
    print(f"Output shape: {output.shape}")
    
    # 6. Compare outputs
    print("\n5. Comparing PyTorch vs ONNX outputs...")
    compare_pytorch_onnx_outputs(model, onnx_path, test_input)
    
    # 7. Optimize model
    print("\n6. Optimizing ONNX model...")
    optimized_path = "simple_model_optimized.onnx"
    optimize_onnx_model(onnx_path, optimized_path)


if __name__ == "__main__":
    demo_onnx_workflow()
