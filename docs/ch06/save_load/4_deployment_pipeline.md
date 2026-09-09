# 배포 파이프라인

완전한 배포 파이프라인은 학습된 모델을 최적화(가지치기, 양자화), 형식 변환(ONNX), 성능 측정을 거쳐 실전에 배포한다. 이 모듈은 학습, 양자화, ONNX 내보내기, 최적화, 그리고 정확도·지연·처리량 지표를 담은 배포 보고서 생성까지 이어지는 작업 흐름을 보인다.

## 1. 코드

```python
"""
처음부터 끝까지의 모델 배포 파이프라인

이 모듈은 완전한 배포 작업 흐름을 보인다:
1. 모델을 학습시킨다
2. 최적화한다 (가지치기, 양자화)
3. ONNX로 바꾼다
4. 최적화된 추론으로 배포한다
5. 감시하고 성능을 잰다
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

# ========================================================================
# 메인
# ========================================================================


@dataclass
class ModelMetrics:
    """모델의 성능 지표를 담는다"""
    accuracy: float
    latency_ms: float
    throughput: float
    model_size_mb: float
    memory_usage_mb: float = 0.0


class ImageClassifier(nn.Module):
    """예시 이미지 분류기"""
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
    """완전한 배포 파이프라인"""
    
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
        모델을 학습시킨다
        
        인수:
            train_loader: 학습 데이터 로더
            val_loader: 검증 데이터 로더
            num_epochs: 학습 에포크 수
            lr: 학습률
        
        반환값:
            학습된 모델
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
            # 학습
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
            
            # 검증
            val_acc = self.evaluate(val_loader)
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.3f}, "
                  f"Train Acc: {train_acc:.2f}%, "
                  f"Val Acc: {val_acc:.2f}%")
            
            # 가장 좋은 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.output_dir / "best_model.pth")
        
        # 가장 좋은 모델을 불러온다
        self.model.load_state_dict(torch.load(self.output_dir / "best_model.pth"))
        
        # 성능 측정
        self.metrics_history['original'] = self.benchmark_model(self.model, val_loader)
        print(f"\n✓ Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        
        return self.model
    
    def evaluate(self, data_loader: DataLoader) -> float:
        """모델의 정확도를 평가한다"""
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
        동적 양자화를 적용한다
        
        인수:
            val_loader: 정확도 시험용 검증 데이터
        
        반환값:
            양자화한 모델
        """
        print("\n" + "="*70)
        print("STEP 2: APPLYING QUANTIZATION")
        print("="*70)
        
        # 동적 양자화 적용
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8
        )
        
        # 저장
        torch.save(quantized_model.state_dict(), self.output_dir / "quantized_model.pth")
        
        # 성능 측정
        self.metrics_history['quantized'] = self.benchmark_model(quantized_model, val_loader)
        
        print(f"\n✓ Quantization complete!")
        self._compare_metrics('original', 'quantized')
        
        return quantized_model
    
    def apply_pruning(self, amount: float = 0.3, val_loader: DataLoader = None) -> nn.Module:
        """
        모델 가지치기를 적용한다
        
        인수:
            amount: 잘라 낼 가중치의 비율
            val_loader: 정확도 시험용 검증 데이터
        
        반환값:
            가지친 모델
        """
        print("\n" + "="*70)
        print(f"STEP 3: APPLYING PRUNING (amount={amount})")
        print("="*70)
        
        import torch.nn.utils.prune as prune
        
        # 전역 가지치기
        parameters_to_prune = []
        for module in self.model.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
        
        # 영구히 적용
        for module, param in parameters_to_prune:
            prune.remove(module, param)
        
        # 저장
        torch.save(self.model.state_dict(), self.output_dir / "pruned_model.pth")
        
        # 성능 측정
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
        모델을 ONNX 형식으로 바꾼다
        
        인수:
            input_shape: 입력 텐서의 모양 (배치 차원 제외)
            opset_version: ONNX opset 판본
        
        반환값:
            ONNX 모델의 경로
        """
        print("\n" + "="*70)
        print("STEP 4: CONVERTING TO ONNX")
        print("="*70)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device).eval()
        
        dummy_input = torch.randn(1, *input_shape).to(device)
        onnx_path = str(self.output_dir / "model.onnx")
        
        # 내보내기
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
        
        # 확인
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"✓ ONNX model saved to {onnx_path}")
        print(f"  Model size: {Path(onnx_path).stat().st_size / (1024*1024):.2f} MB")
        
        return onnx_path
    
    def optimize_onnx(self, onnx_path: str) -> str:
        """
        ONNX 모델을 최적화한다
        
        인수:
            onnx_path: ONNX 모델의 경로
        
        반환값:
            최적화한 ONNX 모델의 경로
        """
        print("\n" + "="*70)
        print("STEP 5: OPTIMIZING ONNX MODEL")
        print("="*70)
        
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        optimized_path = str(self.output_dir / "model_optimized.onnx")
        
        # ONNX 모델 양자화
        quantize_dynamic(
            model_input=onnx_path,
            model_output=optimized_path,
            weight_type=QuantType.QInt8
        )
        
        print(f"✓ Optimized ONNX model saved to {optimized_path}")
        
        # 크기 비교
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
        모델의 성능을 두루 측정한다
        
        인수:
            model: 성능을 잴 모델
            data_loader: 시험용 데이터 로더
            num_runs: 지연을 재기 위한 실행 횟수
        
        반환값:
            모델의 지표
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device).eval()
        
        # 정확도
        accuracy = self.evaluate(data_loader)
        
        # 지연 (표본 하나)
        test_input = torch.randn(1, 3, 32, 32).to(device)
        
        # 워밍업
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 측정
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(test_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        latency_ms = (time.time() - start) / num_runs * 1000
        
        # 처리량 (배치 처리)
        batch_input = torch.randn(32, 3, 32, 32).to(device)
        start = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(batch_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        throughput = (32 * num_runs) / (time.time() - start)
        
        # 모델의 크기
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
        """두 모델 판본의 지표를 견준다"""
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
        """종합 배포 보고서를 저장한다"""
        report = {
            'metrics': {k: asdict(v) if v else None for k, v in self.metrics_history.items()},
            'recommendations': self.generate_recommendations()
        }
        
        report_path = self.output_dir / "deployment_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n✓ Deployment report saved to {report_path}")
    
    def generate_recommendations(self) -> Dict[str, str]:
        """배포 권고를 만든다"""
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
    """시연을 위한 합성 데이터셋을 만든다"""
    # 학습 데이터
    X_train = torch.randn(num_samples, 3, 32, 32)
    y_train = torch.randint(0, 10, (num_samples,))
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 검증 데이터
    X_val = torch.randn(200, 3, 32, 32)
    y_val = torch.randint(0, 10, (200,))
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader


def main():
    """완전한 배포 파이프라인을 실행한다"""
    print("="*70)
    print("COMPLETE MODEL DEPLOYMENT PIPELINE")
    print("="*70)
    
    # 모델 만들기
    model = ImageClassifier(num_classes=10)
    
    # 합성 데이터셋 만들기
    train_loader, val_loader = create_synthetic_dataset()
    
    # 파이프라인 초기화
    pipeline = DeploymentPipeline(model, output_dir="./deployment_output")
    
    # 파이프라인 실행
    try:
        # 1. 학습
        pipeline.train_model(train_loader, val_loader, num_epochs=5)
        
        # 2. 양자화
        quantized_model = pipeline.apply_quantization(val_loader)
        
        # 3. ONNX로 변환
        onnx_path = pipeline.convert_to_onnx(input_shape=(3, 32, 32))
        
        # 4. ONNX 최적화
        optimized_onnx = pipeline.optimize_onnx(onnx_path)
        
        # 5. 보고서 저장
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
```

## 2. 논의

배포 파이프라인은 체계적인 순서를 따른다. 학습과 검증, 양자화 적용, ONNX 변환, ONNX 모델 최적화, 종합 보고서 생성이다. 각 단계마다 성능을 측정하여 정확도, 지연, 처리량, 모델 크기에 미치는 영향을 기록한다.

성능 측정 방법은 JIT 컴파일과 캐시의 영향을 빼려고 워밍업을 먼저 돌린 뒤 지연을 재고, 안정성을 위해 100번의 평균을 낸다. 처리량은 배치 처리로 재어 모델이 데이터를 처리할 수 있는 최대 속도를 잡아낸다.

배포 보고서에는 지표와 함께 권고도 담기며, 정확도 손실이 1% 아래이고 크기가 50% 넘게 줄면 양자화를 자동으로 권한다. 이러한 자동화는 정확도와 효율 사이의 절충을 두고 팀이 근거 있는 판단을 내리도록 돕는다.

## 연습문제

**연습문제 1.**
코드를 따라가며 쓰인 주요 자료 구조를 찾아라. 각각에 대해 자료형, (해당한다면) 모양, 파이프라인에서의 구실을 적어라.

??? success "연습문제 1 풀이"
    코드를 꼼꼼히 읽으며 변수 대입마다 살펴본다. 텐서는 `.shape`과 `.dtype`을 확인하고, 클래스는 `__init__`의 매개변수와 `forward`/`__call__`의 서명을 확인한다. 이름, 자료형, 모양, 구실을 열로 하는 표에 정리한다.

---


**연습문제 2.**
오류 처리와 입력 검증을 넣도록 코드를 고쳐라. 이 코드를 실전에 쓸 수 있게 하려면 어떤 검사를 더하겠는가?

??? success "연습문제 2 풀이"
    입력에 자료형 검사(`isinstance`), 모양 검증(`assert tensor.dim() == expected`), 값 범위 검사(예: 확률이 [0,1] 안인지)를 넣고, 입출력 연산은 try-except로 감싼다. 빈 배치나 NaN 같은 경계 상황에는 경고를 남긴다. 매개변수와 반환값의 자료형을 적은 독스트링을 붙인다.

---


**연습문제 3.**
직접 고른 새로운 쓰임새를 지원하도록 코드를 확장하라. 무엇을 왜 바꿀지 설명하라.

??? success "연습문제 3 풀이"
    알맞은 확장을 하나 고른다(예: 다른 데이터셋, 지표 추가, 새 모델 변형). 필요한 변경을 설명한다. 새 임포트, 클래스 정의 수정, 초매개변수 갱신, 새로운 시각화나 기록 등이다. 핵심 변경을 구현하고 간단한 시험으로 올바름을 확인한다.

## 정리하며

**다룬 것** — 배포 파이프라인

배포 파이프라인은 체계적인 순서를 따른다.

핵심 클래스는 `ModelMetrics`, `ImageClassifier`, `DeploymentPipeline`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
