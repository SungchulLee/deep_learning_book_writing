# 양자화

양자화는 웨이트와 연산에 낮은 정밀도 표현을 적용하는 방법이다. 동적·정적 양자화와 양자화 인지 학습(QAT)의 준비 과정, 지원 연산과 평가 기준을 비교한다.

## 1. 코드

??? note "코드 (429줄)"

    ```python
    """
    모델 양자화 - 정적 양자화와 동적 양자화

    이 모듈은 다음을 보인다:
    - 동적 양자화 (웨이트만)
    - 정적 양자화 (웨이트와 활성화)
    - 양자화 인지 학습 (QAT)
    - 학습 뒤 양자화 (PTQ)
    - 모델의 크기와 성능 견주기
    """

    import torch
    import torch.nn as nn
    import torch.quantization as quantization
    import numpy as np
    import time
    import os
    from typing import Callable, Tuple
    from copy import deepcopy

    # ========================================================================
    # 메인
    # ========================================================================


    class CNNModel(nn.Module):
        """양자화를 위한 예시 CNN 모델"""
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
        """양자화를 위한 예시 LSTM 모델"""
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
        동적 양자화 - 모델을 불러올 때 웨이트를 양자화한다

        가장 알맞은 곳:
        - RNN, LSTM, 트랜스포머
        - 입력의 크기가 변하는 모델
        - 보정 없이 빠르게 배포할 때

        인수:
            model: PyTorch 모델
            example_input: 시험용 예시 입력
        """
        print("\n=== Dynamic Quantization ===")

        # 모델 양자화 (웨이트만 양자화된다)
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.LSTM},  # 양자화할 레이어
            dtype=torch.qint8
        )

        # 크기 비교
        original_size = get_model_size(model)
        quantized_size = get_model_size(quantized_model)

        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {original_size/quantized_size:.2f}x")

        # 추론 시간 비교
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
        정적 양자화 - 웨이트와 활성화를 모두 양자화한다

        가장 알맞은 곳:
        - CNN
        - 입력의 크기가 고정된 모델
        - 성능을 최대로 끌어올릴 때

        활성화의 범위를 정하려면 보정 데이터가 필요하다

        인수:
            model: PyTorch 모델
            calibration_loader: 보정 데이터를 담은 DataLoader
            example_input: 시험용 예시 입력
        """
        print("\n=== Static Quantization ===")

        # 양자화 설정 지정
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # 양자화를 위해 모델 준비
        model_prepared = torch.quantization.prepare(model, inplace=False)

        # 대표성 있는 데이터셋으로 보정
        print("Calibrating model...")
        model_prepared.eval()
        with torch.no_grad():
            for data, _ in calibration_loader:
                model_prepared(data)

        # 양자화된 모델로 바꾸기
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)

        # 크기 비교
        original_size = get_model_size(model)
        quantized_size = get_model_size(quantized_model)

        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {original_size/quantized_size:.2f}x")

        # 추론 시간 비교
        original_time = measure_inference_time(model, example_input)
        quantized_time = measure_inference_time(quantized_model, example_input)

        print(f"Original inference time: {original_time*1000:.2f} ms")
        print(f"Quantized inference time: {quantized_time*1000:.2f} ms")
        print(f"Speedup: {original_time/quantized_time:.2f}x")

        return quantized_model


    class QATModel(nn.Module):
        """양자화 인지 학습을 위해 준비된 모델"""
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
            """양자화를 잘하려고 conv+bn+relu를 융합한다"""
            torch.quantization.fuse_modules(
                self,
                [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2', 'relu2']],
                inplace=True
            )


    def quantization_aware_training_demo(model: QATModel, train_loader, num_epochs=3):
        """
        양자화 인지 학습 - 양자화를 염두에 두고 학습한다

        가장 알맞은 곳:
        - 학습 뒤 양자화로 정확도를 크게 잃는 모델
        - 학습 자원이 있을 때
        - 정확도를 최대한 지켜야 할 때

        인수:
            model: 양자화 인지 학습을 위해 준비된 모델
            train_loader: 학습 데이터 로더
            num_epochs: 학습 에포크 수
        """
        print("\n=== Quantization-Aware Training ===")

        # 레이어 융합
        model.fuse_model()

        # 양자화 인지 학습 설정 지정
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        # 양자화 인지 학습 준비
        model_prepared = torch.quantization.prepare_qat(model, inplace=False)

        # 학습 루프
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

        # 양자화된 모델로 바꾸기
        model_prepared.eval()
        quantized_model = torch.quantization.convert(model_prepared, inplace=False)

        print("QAT completed!")
        return quantized_model


    def onnx_quantization_demo(onnx_model_path: str, output_path: str, calibration_data_reader=None):
        """
        ONNX 양자화 - ONNX 모델을 양자화한다

        인수:
            onnx_model_path: ONNX 모델의 경로
            output_path: 양자화한 모델을 저장할 경로
            calibration_data_reader: 보정 데이터 읽개 (선택 사항)
        """
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType

        print("\n=== ONNX Quantization ===")

        # 동적 양자화 (보정이 필요 없다)
        quantize_dynamic(
            model_input=onnx_model_path,
            model_output=output_path,
            weight_type=QuantType.QInt8
        )

        # 정적 양자화라면 다음을 쓴다:
        # quantize_static(
        #     model_input=onnx_model_path,
        #     model_output=output_path,
        #     calibration_data_reader=calibration_data_reader
        # )

        # 크기 비교
        original_size = os.path.getsize(onnx_model_path) / (1024 * 1024)
        quantized_size = os.path.getsize(output_path) / (1024 * 1024)

        print(f"Original ONNX model size: {original_size:.2f} MB")
        print(f"Quantized ONNX model size: {quantized_size:.2f} MB")
        print(f"Compression ratio: {original_size/quantized_size:.2f}x")


    def get_model_size(model: nn.Module) -> float:
        """
        모델의 크기를 MB 단위로 얻는다

        인수:
            model: PyTorch 모델

        반환값:
            MB 단위의 모델 크기
        """
        torch.save(model.state_dict(), "temp_model.pth")
        size = os.path.getsize("temp_model.pth") / (1024 * 1024)
        os.remove("temp_model.pth")
        return size


    def measure_inference_time(model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 100) -> float:
        """
        평균 추론 시간을 잰다

        인수:
            model: PyTorch 모델
            input_tensor: 입력 텐서
            num_runs: 평균을 낼 실행 횟수

        반환값:
            초 단위의 평균 추론 시간
        """
        model.eval()

        # 워밍업
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_tensor)

        # 측정
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
        원래 모델과 양자화한 모델의 출력을 견준다

        인수:
            original_model: 원래 모델
            quantized_model: 양자화한 모델
            test_input: 시험 입력

        반환값:
            (평균 절대 오차, 최대 절대 오차)의 튜플
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
        """양자화의 좋은 관행을 출력한다"""
        print("""
        === 양자화 모범 사례 ===

        1. 알맞은 방법을 골라라:
           - 동적: LSTM, 트랜스포머, 빠른 배포
           - 정적: CNN, 최고 성능
           - 양자화 인지 학습: 정확도 로스를 받아들일 수 없을 때

        2. 보정(정적 양자화용):
           - 대표성 있는 데이터를 쓴다
           - 보통 100~1000개 표본이면 넉넉하다
           - 입력 분포를 고루 덮어야 한다

        3. 정확도 검증:
           - 늘 전체 시험 집합에서 검증하라
           - 출력값만이 아니라 지표를 견주어라
           - 받아들일 만한 정확도 하락: 1~2%

        4. 레이어 고르기:
           - 모든 레이어가 똑같이 이득을 보지는 않는다
           - 선형 레이어와 컨볼루션 레이어: 이득이 크다
           - 어텐션 레이어: 이득이 보통이다
           - 임베딩 레이어: 보통 건너뛴다

        5. 배포할 때 살필 점:
           - 하드웨어가 받쳐 주는지 확인하라(INT8, INT4)
           - 런타임 호환성을 확인하라
           - 대상 장치에서 성능을 재어 보라

        6. 흔한 문제:
           - 정확도 하락이 5%를 넘으면: 양자화 인지 학습을 써 보라
           - 활성화 이상치: 채널별 양자화를 쓰라
           - 포화: 양자화 구간을 조정하라
        """)


    def demo_all_quantization_methods():
        """모든 양자화 방법을 보인다"""
        print("=== Quantization Methods Demo ===\n")

        # 1. 동적 양자화 (LSTM)
        print("\n" + "="*50)
        print("1. DYNAMIC QUANTIZATION (LSTM)")
        print("="*50)
        lstm_model = LSTMModel()
        lstm_input = torch.randn(8, 20, 10)  # (배치, seq_len, 피처)
        dynamic_quantized = dynamic_quantization_demo(lstm_model, lstm_input)

        # 2. 정적 양자화를 위한 임시 보정 데이터 만들기
        print("\n" + "="*50)
        print("2. STATIC QUANTIZATION (CNN)")
        print("="*50)

        # 합성 보정 데이터 만들기
        calibration_data = [(torch.randn(8, 3, 32, 32), torch.randint(0, 10, (8,))) for _ in range(10)]
        calibration_loader = calibration_data  # 시연을 위해 간략화함

        cnn_model = CNNModel()
        cnn_input = torch.randn(1, 3, 32, 32)

        # 참고: 온전한 정적 양자화에는 제대로 된 준비가 필요하다
        print("Static quantization requires proper model preparation.")
        print("See code for full implementation details.")

        # 3. 좋은 관행 보이기
        quantization_best_practices()


    if __name__ == "__main__":
        demo_all_quantization_methods()
    ```


## 2. 논의

동적 양자화는 지원하는 웨이트를 미리 양자화하고 활성화의 양자화 범위를 실행 중 결정한다. 모듈 입출력이 부동소수점이라는 이유로 내부 활성화까지 항상 FP32라고 설명하면 부정확하다. 이 예제는 Linear와 LSTM 등 지정한 모듈을 변환하며 모든 레이어를 자동으로 INT8로 바꾸지 않는다.

정적 양자화는 대표 데이터로 활성화 범위를 보정하고 변환한다. eager 방식에서는 적절한 모듈 융합과 양자화·역양자화 경계가 필요하다. 이 코드의 일반 CNN에 prepare와 convert만 적용하는 함수는 임의의 모델을 즉시 실행 가능한 양자화 모델로 만드는 완성 경로가 아니다. main도 정적 양자화가 추가 준비를 요구함을 알리고 실제 호출하지 않는다. QAT는 가짜 양자화로 오차를 학습에 반영하지만 정확도 로스의 상한이나 정적 방식보다 나은 결과를 보장하지 않는다.

난수 입력에서의 출력 차이는 과제 정확도 평가를 대신하지 않는다. 파일 크기 비율과 속도도 양자화 비율, 직렬화 부가 정보와 하드웨어 커널에 따라 다르다. 현재 개발은 torchao로 옮겨 가고 있으며 이 페이지는 기존 torch.quantization API의 예제이다. [공식 양자화 안내](https://docs.pytorch.org/docs/stable/quantization.html)에서 설치 버전과 목표 백엔드를 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
동적 양자화가 활성화를 항상 FP32로 계산한다는 설명이 왜 부정확한가?

</div>

??? success "연습문제 1 풀이"
    모듈의 바깥 입력·출력과 내부 연산을 구분해야 한다. 활성화 범위를 런타임에 계산해 지원하는 양자화 연산을 수행할 수 있으며, 웨이트 저장만 낮은 정밀도라는 설명으로는 충분하지 않다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
정적 양자화에서 난수 데이터를 보정에 사용하는 한계는 무엇인가?

</div>

??? success "연습문제 2 풀이"
    실제 입력 분포와 다르면 활성화 범위를 잘못 추정해 클리핑이나 해상도 로스가 커질 수 있다. 실제 전처리를 적용한 대표 데이터를 쓰고 별도 평가셋에서 정확도를 비교해야 한다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
INT8 웨이트 파일이 FP32 파일의 정확히 4분의 1이 되지 않는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"
    모든 텐서가 양자화되지는 않으며 스케일·제로포인트, 버퍼와 직렬화 메타데이터가 추가된다. 실제 파일 크기와 런타임 메모리는 별도로 측정해야 한다.

## 정리하며

양자화는 변환 자체보다 지원 연산·보정 데이터·실행 장치와 과제 성능을 함께 평가해야 한다. 압축률, 속도 향상과 정확도 유지에 고정된 보장값은 없다.
