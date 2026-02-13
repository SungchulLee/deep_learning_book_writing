"""
End-to-End Model Deployment Pipeline

This module demonstrates a complete deployment workflow:
1. Train a model
2. Optimize (pruning, quantization)
3. Convert to ONNX
4. Deploy with optimized inference
5. Monitor and benchmark
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import onnx
import onnxruntime as ort
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ModelMetrics:
    """Store model performance metrics"""
    accuracy: float
    latency_ms: float
    throughput: float
    model_size_mb: float
    memory_usage_mb: float = 0.0


class ImageClassifier(nn.Module):
    """Example image classifier"""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class DeploymentPipeline:
    """Complete deployment pipeline"""
    
    def __init__(self, model: nn.Module, output_dir: str = "./deployment"):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.metrics_history = {
            'original': None,
            'quantized': None,
            'pruned': None,
            'onnx': None,
            'optimized': None
        }
    
    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        lr: float = 0.001
    ) -> nn.Module:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            lr: Learning rate
        
        Returns:
            Trained model
        """
        print("\n" + "="*70)
        print("STEP 1: TRAINING MODEL")
        print("="*70)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        best_val_acc = 0.0
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100. * train_correct / train_total
            
            # Validation
            val_acc = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.3f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
        
        # Load best model
        self.model.load_state_dict(torch.load(self.output_dir / "best_model.pth"))
        
        # Benchmark
        self.metrics_history['original'] = self.benchmark_model(self.model, val_loader)
        print(f"\n✓ Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.model
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """Evaluate model accuracy"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return 100. * correct / total
    
    def apply_quantization(self, val_loader: DataLoader) -> nn.Module:
        """
        Apply dynamic quantization
        
        Args:
            val_loader: Validation data for accuracy testing
        
        Returns:
            Quantized model
        """
        print("\n" + "="*70)
        print("STEP 2: APPLYING QUANTIZATION")
        print("="*70)
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        # Save
        torch.save(quantized_model.state_dict(), self.output_dir / "quantized_model.pth")
        
        # Benchmark
        self.metrics_history['quantized'] = self.benchmark_model(quantized_model, val_loader)
        
        print(f"\n✓ Quantization complete!")
        self._compare_metrics('original', 'quantized')
        
        return quantized_model
    
    def apply_pruning(self, amount: float = 0.3, val_loader: DataLoader = None) -> nn.Module:
        """
        Apply model pruning
        
        Args:
            amount: Fraction of weights to prune
            val_loader: Validation data for accuracy testing
        
        Returns:
            Pruned model
        """
        print("\n" + "="*70)
        print(f"STEP 3: APPLYING PRUNING (amount={amount})")
        print("="*70)
        
        import torch.nn.utils.prune as prune
        
        # Global pruning
        parameters_to_prune = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        
        # Make permanent
        for module, param in parameters_to_prune:
            prune.remove(module, param)
        
        # Save
        torch.save(self.model.state_dict(), self.output_dir / "pruned_model.pth")
        
        # Benchmark
        if val_loader:
            self.metrics_history['pruned'] = self.benchmark_model(self.model, val_loader)
            print(f"\n✓ Pruning complete!")
            self._compare_metrics('original', 'pruned')
        
        return self.model
    
    def convert_to_onnx(
        self,
        input_shape: Tuple[int, ...],
        opset_version: int = 14
    ) -> str:
        """
        Convert model to ONNX format
        
        Args:
            input_shape: Input tensor shape (without batch dimension)
            opset_version: ONNX opset version
        
        Returns:
            Path to ONNX model
        """
        print("\n" + "="*70)
        print("STEP 4: CONVERTING TO ONNX")
        print("="*70)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device).eval()
        
        dummy_input = torch.randn(1, *input_shape).to(device)
        onnx_path = str(self.output_dir / "model.onnx")
        
        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✓ ONNX model saved to {onnx_path}")
        print(f"  Model size: {Path(onnx_path).stat().st_size / (1024*1024):.2f} MB")
        
        return onnx_path
    
    def optimize_onnx(self, onnx_path: str) -> str:
        """
        Optimize ONNX model
        
        Args:
            onnx_path: Path to ONNX model
        
        Returns:
            Path to optimized ONNX model
        """
        print("\n" + "="*70)
        print("STEP 5: OPTIMIZING ONNX MODEL")
        print("="*70)
        
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        optimized_path = str(self.output_dir / "model_optimized.onnx")
        
        # Quantize ONNX model
        quantize_dynamic(
            model_input=onnx_path,
            model_output=optimized_path,
            weight_type=QuantType.QInt8
        )
        
        print(f"✓ Optimized ONNX model saved to {optimized_path}")
        
        # Compare sizes
        original_size = Path(onnx_path).stat().st_size / (1024*1024)
        optimized_size = Path(optimized_path).stat().st_size / (1024*1024)
        
        print(f"  Original size: {original_size:.2f} MB")
        print(f"  Optimized size: {optimized_size:.2f} MB")
        print(f"  Compression ratio: {original_size/optimized_size:.2f}x")
        
        return optimized_path
    
    def benchmark_model(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        num_runs: int = 100
    ) -> ModelMetrics:
        """
        Comprehensive model benchmarking
        
        Args:
            model: Model to benchmark
            data_loader: Data loader for testing
            num_runs: Number of runs for latency measurement
        
        Returns:
            Model metrics
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()
        
        # Accuracy
        accuracy = self.evaluate(data_loader)
        
        # Latency (single sample)
        test_input = torch.randn(1, 3, 32, 32).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Measure
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(test_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        latency_ms = (time.time() - start) / num_runs * 1000
        
        # Throughput (batch processing)
        batch_input = torch.randn(32, 3, 32, 32).to(device)
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(batch_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        throughput = (32 * num_runs) / (time.time() - start)
        
        # Model size
        temp_path = self.output_dir / "temp_model.pth"
        torch.save(model.state_dict(), temp_path)
        model_size_mb = temp_path.stat().st_size / (1024*1024)
        temp_path.unlink()
        
        return ModelMetrics(
            accuracy=accuracy,
            latency_ms=latency_ms,
            throughput=throughput,
            model_size_mb=model_size_mb
        )
    
    def _compare_metrics(self, baseline: str, optimized: str):
        """Compare metrics between two model versions"""
        base = self.metrics_history[baseline]
        opt = self.metrics_history[optimized]
        
        print(f"\n{'Metric':<20} {'Baseline':<15} {'Optimized':<15} {'Change':<15}")
        print("-" * 65)
        print(f"{'Accuracy (%)':<20} {base.accuracy:<15.2f} {opt.accuracy:<15.2f} "
              f"{opt.accuracy - base.accuracy:+.2f}%")
        print(f"{'Latency (ms)':<20} {base.latency_ms:<15.2f} {opt.latency_ms:<15.2f} "
              f"{(opt.latency_ms/base.latency_ms - 1)*100:+.1f}%")
        print(f"{'Throughput (s/s)':<20} {base.throughput:<15.1f} {opt.throughput:<15.1f} "
              f"{(opt.throughput/base.throughput - 1)*100:+.1f}%")
        print(f"{'Model Size (MB)':<20} {base.model_size_mb:<15.2f} {opt.model_size_mb:<15.2f} "
              f"{(opt.model_size_mb/base.model_size_mb - 1)*100:+.1f}%")
    
    def save_deployment_report(self):
        """Save comprehensive deployment report"""
        report = {
            'metrics': {k: asdict(v) if v else None for k, v in self.metrics_history.items()},
            'recommendations': self.generate_recommendations()
        }
        
        report_path = self.output_dir / "deployment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Deployment report saved to {report_path}")
    
    def generate_recommendations(self) -> Dict[str, str]:
        """Generate deployment recommendations"""
        recommendations = {}
        
        if self.metrics_history['quantized']:
            base = self.metrics_history['original']
            quant = self.metrics_history['quantized']
            
            acc_drop = base.accuracy - quant.accuracy
            size_reduction = (1 - quant.model_size_mb/base.model_size_mb) * 100
            
            if acc_drop < 1.0 and size_reduction > 50:
                recommendations['quantization'] = "✓ Highly recommended - minimal accuracy loss with significant size reduction"
            elif acc_drop < 2.0:
                recommendations['quantization'] = "○ Recommended - acceptable accuracy trade-off"
            else:
                recommendations['quantization'] = "✗ Not recommended - significant accuracy loss"
        
        return recommendations


def create_synthetic_dataset(num_samples: int = 1000) -> Tuple[DataLoader, DataLoader]:
    """Create synthetic dataset for demonstration"""
    # Training data
    X_train = torch.randn(num_samples, 3, 32, 32)
    y_train = torch.randint(0, 10, (num_samples,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Validation data
    X_val = torch.randn(200, 3, 32, 32)
    y_val = torch.randint(0, 10, (200,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


def main():
    """Run complete deployment pipeline"""
    print("="*70)
    print("COMPLETE MODEL DEPLOYMENT PIPELINE")
    print("="*70)
    
    # Create model
    model = ImageClassifier(num_classes=10)
    
    # Create synthetic dataset
    train_loader, val_loader = create_synthetic_dataset()
    
    # Initialize pipeline
    pipeline = DeploymentPipeline(model, output_dir="./deployment_output")
    
    # Run pipeline
    try:
        # 1. Train
        pipeline.train_model(train_loader, val_loader, num_epochs=5)
        
        # 2. Quantize
        quantized_model = pipeline.apply_quantization(val_loader)
        
        # 3. Convert to ONNX
        onnx_path = pipeline.convert_to_onnx(input_shape=(3, 32, 32))
        
        # 4. Optimize ONNX
        optimized_onnx = pipeline.optimize_onnx(onnx_path)
        
        # 5. Save report
        pipeline.save_deployment_report()
        
        print("\n" + "="*70)
        print("✓ DEPLOYMENT PIPELINE COMPLETE!")
        print("="*70)
        print(f"\nOutput directory: {pipeline.output_dir}")
        print("\nGenerated files:")
        for file in pipeline.output_dir.iterdir():
            print(f"  - {file.name}")
        
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
