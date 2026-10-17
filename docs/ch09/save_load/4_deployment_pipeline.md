# 배포 파이프라인

학습, 모델 변환, 간단한 성능 측정과 보고서를 연결하는 교육용 배포 파이프라인이다. 각 단계의 실제 대상 모델과 평가 방식에 남아 있는 한계를 함께 확인한다.

## 1. 코드

??? note "코드 (528줄)"

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
                lr: 러닝레이트

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
                amount: 잘라 낼 웨이트의 비율
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

예제의 입력과 레이블은 무작위로 만든 합성 데이터이다. 따라서 정확도는 실제 이미지 분류 성능의 증거가 아니다. main은 학습 뒤 동적 양자화와 ONNX 변환을 각각 수행하며 가지치기 함수는 호출하지 않는다. optimize_onnx는 일반적인 그래프 최적화가 아니라 ONNX 동적 양자화를 적용한다.

benchmark_model은 인수로 받은 model의 지연·크기를 측정하지만 정확도는 self.evaluate를 호출해 self.model에서 계산한다. 따라서 양자화 모델에 기록된 정확도를 그 모델의 실제 정확도로 해석할 수 없다. 또한 CUDA가 있으면 모델을 일괄적으로 CUDA로 옮기므로 CPU 양자화 커널을 사용하는 경로와 충돌할 수 있다. 이 상태의 자동 추천은 실제 배포 결정을 뒷받침하는 검증 결과가 아니다.

지연은 워밍업 뒤 반복 실행의 평균이며 서비스의 전처리·대기열·전송 시간과 꼬리 지연을 포함하지 않는다. 배치 32의 처리량도 해당 입력과 장치에서의 측정값으로 최대 처리량을 뜻하지 않는다. 모델별로 올바른 평가 함수와 장치 경로를 연결하고 독립 평가셋, 지연 백분위수와 자원 사용량을 추가해야 한다. best_model 파일을 저장해도 이후 단계가 그 파일을 다시 불러오는지는 별도로 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
양자화 모델의 정확도 지표가 잘못 연결된 위치를 설명하라.

</div>

??? success "풀이"
    benchmark_model의 model 인수 대신 self.evaluate가 self.model을 평가한다. 따라서 다른 모델의 지연과 원래 모델의 정확도가 한 지표 객체에 섞인다. 대상 모델을 평가 함수에 명시적으로 전달해야 한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
파이프라인의 모든 변환 단계가 main에서 실행되는가?

</div>

??? success "풀이"
    아니다. 가지치기 함수는 정의되지만 main에서 호출되지 않는다. ONNX 최적화라는 이름의 단계는 동적 양자화이며 각 단계가 같은 변환 모델을 순차적으로 넘기는 것으로 가정해서도 안 된다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
자동 추천의 정확도 로스 1% 기준은 어떻게 해석해야 하는가?

</div>

??? success "풀이"
    코드의 정확도가 퍼센트 단위이면 두 값의 차이는 퍼센트포인트이다. 업무에 맞게 정한 허용 기준일 뿐 보편적인 배포 기준이 아니다. 먼저 해당 모델의 정확도를 올바르게 측정해야 한다.

## 정리하며

이 파이프라인은 단계 연결의 예시이며 완성된 배포 검증 체계가 아니다. 대상 모델과 지표를 정확히 연결하고 장치 지원, 평가 데이터와 실제 서비스의 성능 요구를 확인해야 한다.
