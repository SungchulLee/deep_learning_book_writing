"""
Utility Functions for Model Deployment

This module provides helper functions and utilities for:
- Model conversion
- Performance monitoring
- Deployment validation
- Common operations
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
import time
import psutil
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import hashlib


@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    model_name: str
    version: str
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    quantization: bool = False
    batch_size: int = 1
    device: str = 'cpu'
    precision: str = 'fp32'  # fp32, fp16, int8


class ModelValidator:
    """Validate models before deployment"""
    
    @staticmethod
    def validate_pytorch_model(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_classes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Validate PyTorch model
        
        Returns dict with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Test forward pass
            dummy_input = torch.randn(1, *input_shape)
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            results['info']['output_shape'] = list(output.shape)
            results['info']['num_parameters'] = sum(p.numel() for p in model.parameters())
            
            # Check output shape
            if num_classes and output.shape[-1] != num_classes:
                results['warnings'].append(
                    f"Output shape {output.shape} doesn't match num_classes {num_classes}"
                )
            
            # Check for NaN/Inf
            if torch.isnan(output).any():
                results['errors'].append("Model output contains NaN values")
                results['valid'] = False
            
            if torch.isinf(output).any():
                results['errors'].append("Model output contains Inf values")
                results['valid'] = False
                
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Forward pass failed: {str(e)}")
        
        return results
    
    @staticmethod
    def validate_onnx_model(
        onnx_path: str,
        input_shape: Tuple[int, ...]
    ) -> Dict[str, Any]:
        """Validate ONNX model"""
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            # Load and check model
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)
            
            # Test inference
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            dummy_input = np.random.randn(1, *input_shape).astype(np.float32)
            
            output = session.run(None, {input_name: dummy_input})
            
            results['info']['output_shape'] = list(output[0].shape)
            results['info']['providers'] = session.get_providers()
            
            # Check for NaN/Inf
            if np.isnan(output[0]).any():
                results['errors'].append("ONNX output contains NaN values")
                results['valid'] = False
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"ONNX validation failed: {str(e)}")
        
        return results


class PerformanceMonitor:
    """Monitor model performance in production"""
    
    def __init__(self):
        self.metrics = {
            'latencies': [],
            'throughputs': [],
            'memory_usage': [],
            'errors': 0,
            'total_requests': 0
        }
    
    def record_inference(
        self,
        latency_ms: float,
        batch_size: int = 1,
        success: bool = True
    ):
        """Record single inference metrics"""
        self.metrics['latencies'].append(latency_ms)
        self.metrics['throughputs'].append(batch_size / (latency_ms / 1000))
        self.metrics['total_requests'] += 1
        
        if not success:
            self.metrics['errors'] += 1
        
        # Memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        self.metrics['memory_usage'].append(memory_mb)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.metrics['latencies']:
            return {}
        
        latencies = np.array(self.metrics['latencies'])
        throughputs = np.array(self.metrics['throughputs'])
        memory = np.array(self.metrics['memory_usage'])
        
        return {
            'latency': {
                'mean': float(np.mean(latencies)),
                'median': float(np.median(latencies)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99)),
                'min': float(np.min(latencies)),
                'max': float(np.max(latencies))
            },
            'throughput': {
                'mean': float(np.mean(throughputs)),
                'max': float(np.max(throughputs))
            },
            'memory_mb': {
                'mean': float(np.mean(memory)),
                'peak': float(np.max(memory))
            },
            'reliability': {
                'total_requests': self.metrics['total_requests'],
                'errors': self.metrics['errors'],
                'error_rate': self.metrics['errors'] / max(self.metrics['total_requests'], 1)
            }
        }
    
    def print_report(self):
        """Print formatted performance report"""
        stats = self.get_statistics()
        
        if not stats:
            print("No data collected yet")
            return
        
        print("\n" + "="*70)
        print("PERFORMANCE REPORT")
        print("="*70)
        
        print(f"\n{'LATENCY (ms)':<20}")
        print(f"  Mean:     {stats['latency']['mean']:>10.2f}")
        print(f"  Median:   {stats['latency']['median']:>10.2f}")
        print(f"  P95:      {stats['latency']['p95']:>10.2f}")
        print(f"  P99:      {stats['latency']['p99']:>10.2f}")
        
        print(f"\n{'THROUGHPUT (samples/s)':<20}")
        print(f"  Mean:     {stats['throughput']['mean']:>10.1f}")
        print(f"  Peak:     {stats['throughput']['max']:>10.1f}")
        
        print(f"\n{'MEMORY (MB)':<20}")
        print(f"  Average:  {stats['memory_mb']['mean']:>10.1f}")
        print(f"  Peak:     {stats['memory_mb']['peak']:>10.1f}")
        
        print(f"\n{'RELIABILITY':<20}")
        print(f"  Requests: {stats['reliability']['total_requests']:>10}")
        print(f"  Errors:   {stats['reliability']['errors']:>10}")
        print(f"  Error %:  {stats['reliability']['error_rate']*100:>10.2f}")


class ModelRegistry:
    """Simple model registry for tracking deployed models"""
    
    def __init__(self, registry_path: str = "./model_registry.json"):
        self.registry_path = Path(registry_path)
        self.models = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load registry from disk"""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save registry to disk"""
        with open(self.registry_path, 'w') as f:
            json.dump(self.models, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        version: str,
        model_path: str,
        config: DeploymentConfig,
        metrics: Optional[Dict] = None
    ):
        """Register a new model"""
        model_id = f"{model_name}:{version}"
        
        # Calculate file hash for integrity
        file_hash = self._calculate_file_hash(model_path)
        
        self.models[model_id] = {
            'name': model_name,
            'version': version,
            'path': model_path,
            'config': asdict(config),
            'metrics': metrics or {},
            'file_hash': file_hash,
            'registered_at': time.time()
        }
        
        self._save_registry()
        print(f"✓ Registered model: {model_id}")
    
    def get_model(self, model_name: str, version: str = 'latest') -> Optional[Dict]:
        """Get model info from registry"""
        if version == 'latest':
            # Find latest version
            versions = [k for k in self.models.keys() if k.startswith(model_name + ':')]
            if not versions:
                return None
            version = max(versions).split(':')[1]
        
        model_id = f"{model_name}:{version}"
        return self.models.get(model_id)
    
    def list_models(self) -> List[Dict]:
        """List all registered models"""
        return list(self.models.values())
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def compare_models(
    model1_path: str,
    model2_path: str,
    test_inputs: List[np.ndarray],
    model1_type: str = 'onnx',
    model2_type: str = 'onnx'
) -> Dict[str, Any]:
    """
    Compare outputs from two models
    
    Useful for validating conversions/optimizations
    """
    print("\n=== Model Comparison ===")
    
    # Load models
    if model1_type == 'onnx':
        session1 = ort.InferenceSession(model1_path)
        input_name1 = session1.get_inputs()[0].name
    
    if model2_type == 'onnx':
        session2 = ort.InferenceSession(model2_path)
        input_name2 = session2.get_inputs()[0].name
    
    differences = []
    max_diff = 0.0
    
    for i, test_input in enumerate(test_inputs):
        # Run inference
        out1 = session1.run(None, {input_name1: test_input})[0]
        out2 = session2.run(None, {input_name2: test_input})[0]
        
        # Calculate difference
        diff = np.abs(out1 - out2)
        differences.append(np.mean(diff))
        max_diff = max(max_diff, np.max(diff))
    
    avg_diff = np.mean(differences)
    
    results = {
        'average_difference': float(avg_diff),
        'max_difference': float(max_diff),
        'outputs_match': max_diff < 1e-5,
        'num_comparisons': len(test_inputs)
    }
    
    print(f"Average difference: {avg_diff:.6f}")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Outputs match: {'✓' if results['outputs_match'] else '✗'}")
    
    return results


def create_model_card(
    model_name: str,
    description: str,
    metrics: Dict[str, Any],
    output_path: str
):
    """
    Create a model card with documentation
    
    Following https://arxiv.org/abs/1810.03993
    """
    card = {
        'model_details': {
            'name': model_name,
            'description': description,
            'version': '1.0',
            'date': time.strftime('%Y-%m-%d')
        },
        'model_performance': metrics,
        'intended_use': {
            'primary_uses': 'TODO: Describe primary use cases',
            'out_of_scope': 'TODO: Describe out-of-scope uses'
        },
        'training_data': {
            'description': 'TODO: Describe training data',
            'preprocessing': 'TODO: Describe preprocessing'
        },
        'evaluation_data': {
            'description': 'TODO: Describe evaluation data',
            'metrics': 'TODO: List evaluation metrics'
        },
        'ethical_considerations': {
            'risks': 'TODO: Describe potential risks',
            'mitigations': 'TODO: Describe mitigations'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(card, f, indent=2)
    
    print(f"✓ Model card saved to {output_path}")


def export_for_production(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    output_dir: str,
    model_name: str = "model",
    quantize: bool = True,
    optimize: bool = True
):
    """
    Complete export pipeline for production
    
    Creates all necessary artifacts:
    - Original PyTorch model
    - ONNX model
    - Quantized ONNX model (if requested)
    - Model configuration
    - Validation report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"EXPORTING MODEL: {model_name}")
    print(f"{'='*70}")
    
    # 1. Save PyTorch model
    print("\n1. Saving PyTorch model...")
    torch_path = output_dir / f"{model_name}.pth"
    torch.save(model.state_dict(), torch_path)
    print(f"   ✓ Saved to {torch_path}")
    
    # 2. Export to ONNX
    print("\n2. Exporting to ONNX...")
    onnx_path = output_dir / f"{model_name}.onnx"
    dummy_input = torch.randn(1, *input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"   ✓ Saved to {onnx_path}")
    
    # 3. Quantize if requested
    if quantize:
        print("\n3. Quantizing ONNX model...")
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        quantized_path = output_dir / f"{model_name}_quantized.onnx"
        quantize_dynamic(
            str(onnx_path),
            str(quantized_path),
            weight_type=QuantType.QInt8
        )
        print(f"   ✓ Saved to {quantized_path}")
    
    # 4. Validate
    print("\n4. Validating models...")
    validator = ModelValidator()
    
    pytorch_validation = validator.validate_pytorch_model(model, input_shape)
    onnx_validation = validator.validate_onnx_model(str(onnx_path), input_shape)
    
    # 5. Save config
    print("\n5. Saving configuration...")
    config = {
        'model_name': model_name,
        'input_shape': list(input_shape),
        'pytorch_model': str(torch_path),
        'onnx_model': str(onnx_path),
        'quantized': quantize,
        'validation': {
            'pytorch': pytorch_validation,
            'onnx': onnx_validation
        }
    }
    
    config_path = output_dir / f"{model_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"   ✓ Saved to {config_path}")
    
    print(f"\n{'='*70}")
    print("EXPORT COMPLETE!")
    print(f"{'='*70}")
    print(f"\nFiles created in {output_dir}:")
    for file in output_dir.iterdir():
        size_mb = file.stat().st_size / (1024*1024)
        print(f"  - {file.name:<40} {size_mb:>8.2f} MB")


if __name__ == "__main__":
    print("Model Deployment Utilities")
    print("="*70)
    print("\nAvailable utilities:")
    print("  - ModelValidator: Validate PyTorch and ONNX models")
    print("  - PerformanceMonitor: Track inference metrics")
    print("  - ModelRegistry: Manage deployed models")
    print("  - compare_models(): Compare model outputs")
    print("  - export_for_production(): Complete export pipeline")
    print("\nImport this module to use these utilities in your code.")
