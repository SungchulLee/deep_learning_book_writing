# 양자화

모델 양자화는 가중치와 활성화를 FP32 대신 INT8 같은 낮은 정밀도 자료형으로 나타내어 모델의 크기와 추론 지연을 줄인다. 동적 양자화가 가장 간단하고, 정적 양자화는 보정 데이터를 써서 속도를 가장 크게 높이며, 양자화 인지 학습은 민감한 모델의 정확도를 지켜 준다.

## 코드

```python
"""
모델 양자화 - 정적 양자화와 동적 양자화

이 모듈은 다음을 보인다:
- 동적 양자화 (가중치만)
- 정적 양자화 (가중치와 활성화)
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
    동적 양자화 - 모델을 불러올 때 가중치를 양자화한다
    
    가장 알맞은 곳:
    - RNN, LSTM, 트랜스포머
    - 입력의 크기가 변하는 모델
    - 보정 없이 빠르게 배포할 때
    
    인수:
        model: PyTorch 모델
        example_input: 시험용 예시 입력
    """
    print("\n=== Dynamic Quantization ===")
    
    # 모델 양자화 (가중치만 양자화된다)
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.LSTM},  # 양자화할 층
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
    정적 양자화 - 가중치와 활성화를 모두 양자화한다
    
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
    
    # 층 융합
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
       - 양자화 인지 학습: 정확도 손실을 받아들일 수 없을 때
    
    2. 보정(정적 양자화용):
       - 대표성 있는 데이터를 쓴다
       - 보통 100~1000개 표본이면 넉넉하다
       - 입력 분포를 고루 덮어야 한다
    
    3. 정확도 검증:
       - 늘 전체 시험 집합에서 검증하라
       - 출력값만이 아니라 지표를 견주어라
       - 받아들일 만한 정확도 하락: 1~2%
    
    4. 층 고르기:
       - 모든 층이 똑같이 이득을 보지는 않는다
       - 선형층과 합성곱층: 이득이 크다
       - 어텐션 층: 이득이 보통이다
       - 임베딩 층: 보통 건너뛴다
    
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
    lstm_input = torch.randn(8, 20, 10)  # (배치, seq_len, 특징)
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
    demo_all_quantization_methods()```

## 논의

동적 양자화는 모델을 불러올 때 가중치를 FP32에서 INT8로 바꾸되 활성화는 FP32로 남긴다. 보정 데이터가 필요 없고, 행렬 곱이 주를 이루는 모델(Linear, LSTM 층)에 특히 효과적이어서 정확도 손실을 거의 없이 크기를 보통 2~4배 줄인다.

정적 양자화는 가중치와 활성화를 모두 양자화하므로 층마다 활성화 값의 범위를 정하기 위한 보정 데이터셋이 필요하다. 속도를 가장 많이 높이지만(CPU에서 흔히 2~4배) 준비할 것이 많고, 보정 데이터가 대표성이 없으면 정확도가 떨어질 수 있다.

양자화 인지 학습(QAT)은 학습 중에 가짜 양자화 연산을 끼워 넣어 모델이 양자화의 영향에 견고해지도록 배우게 한다. QAT는 보통 원래 모델과 1% 안의 정확도를 지켜 주므로, 정적 양자화로 인한 정확도 손실을 받아들일 수 없을 때 가장 좋은 선택이다.

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

