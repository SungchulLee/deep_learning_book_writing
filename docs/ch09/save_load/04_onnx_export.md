# ONNX 내보내기

ONNX는 모델의 계산 그래프와 파라메터를 표현하는 교환 형식이다. 이 예제는 평가 모드의 컨볼루션 모델을 내보내고, 형식 검사와 ONNX Runtime 출력 비교를 구분해 보여 준다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
============================================================
PyTorch 모델을 ONNX 형식으로 내보내기
============================================================

ONNX(Open Neural Network Exchange)는 딥러닝 모델을 나타내는 열린 형식이다.
여러 프레임워크와 배포 플랫폼 사이에서 모델을 옮겨 쓸 수 있게 해 준다.


주제:
- 왜 ONNX로 내보내는가
- 기본적인 ONNX 내보내기
- 동적 입력 모양
- 모델 최적화
- 확인과 검증
"""

import torch
import torch.nn as nn
import torch.onnx

# ========================================================================
# 메인
# ========================================================================

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

# 예시 모델 정의
class ConvNet(nn.Module):
    """
    MNIST 방식 분류를 위한 간단한 CNN.
    ONNX 내보내기를 위해 여러 종류의 레이어를 보인다.
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


# 모델을 만들어 준비
model = ConvNet()
model.eval()  # 중요: 내보내기 전에 평가 모드로 둔다

print("\nModel created and set to eval mode")

# 임시 입력 만들기
batch_size = 1
dummy_input = torch.randn(batch_size, 1, 28, 28)

# 출력 경로 정하기
onnx_path = "convnet_model.onnx"

print(f"\nExporting model to ONNX format...")
print(f"Input shape: {dummy_input.shape}")

# 모델 내보내기
torch.onnx.export(
    model,                          # 내보낼 모델
    dummy_input,                    # 예시 입력 텐서
    onnx_path,                      # 출력 파일 경로
    export_params=True,             # 학습된 파라메터 저장
    opset_version=11,               # ONNX 판본
    do_constant_folding=True,       # 상수 접기 최적화
    input_names=['input'],          # 입력 텐서의 이름
    output_names=['output'],        # 출력 텐서의 이름
    dynamic_axes={                  # 동적 입력/출력 모양
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)

print(f"Model exported successfully to '{onnx_path}'")
file_size = os.path.getsize(onnx_path) / 1024
print(f"File size: {file_size:.2f} KB")

if HAS_ONNX:
    # ONNX 모델 검증
    print("\n" + "=" * 70)
    print("VERIFYING ONNX MODEL")
    print("=" * 70)
    
    onnx_model = onnx.load(onnx_path)
    
    try:
        onnx.checker.check_model(onnx_model)
        print("\nONNX model is valid")
    except Exception as e:
        print(f"\nONNX model validation failed: {e}")
    
    # ONNX Runtime으로 추론
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
    
    # PyTorch와 비교
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()
    
    max_diff = np.abs(torch_output - ort_outputs[0]).max()
    print(f"\nMax difference: {max_diff:.6f}")
    
    if max_diff < 1e-5:
        print("Outputs match!")

# 뒷정리
if os.path.exists(onnx_path):
    os.remove(onnx_path)
    print(f"\nCleaned up '{onnx_path}'")

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)


if __name__ == "__main__":
    pass
```

## 2. 논의

코드는 opset 11과 dynamic_axes를 사용하는 기존 내보내기 방식의 예제이다. ONNX 변환이 언제나 단순한 실행 추적만 사용하는 것은 아니다. 현재 PyTorch 문서는 torch.export 기반 dynamo=True 경로를 기본으로 안내하며, 이 경로에서는 dynamic_shapes를 사용한다. 코드에 dynamo가 명시되지 않아 설치 버전의 기본값에 영향을 받는다. [공식 내보내기 문서](https://docs.pytorch.org/docs/stable/onnx_export.html)를 기준으로 환경과 지원되는 opset을 확인한다.

동적 축은 이름을 붙인 차원의 크기가 런타임에 달라질 수 있음을 표현한다. 그래프 내부의 고정 reshape나 모델 자체의 입력 제약까지 없애지는 않는다. 이 코드에서는 배치 축만 동적으로 지정하므로 이미지 채널·공간 크기까지 임의로 바꿀 수 있다는 뜻이 아니다.

onnx.checker는 모델의 형식적 일관성을 검사하며 원래 모델과의 의미적 동등성을 증명하지 않는다. 같은 입력의 최대 절대 오차 비교도 그 입력에서의 점검일 뿐이다. 허용 오차는 출력 스케일과 자료형에 따라 절대·상대 기준으로 정하고 여러 배치 크기와 대표 입력을 비교해야 한다. ONNX 의존성의 유무를 확인한 뒤에도 export 자체가 의존성을 요구하므로 설치 누락 시 모든 단계가 자동으로 생략되는 구조는 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
배치 축만 동적으로 지정하면 이미지 높이와 너비도 바꿀 수 있는가?

</div>

??? success "연습문제 1 풀이"
    아니다. 해당 축의 동적 지정과 모델 연산의 지원이 별도로 필요하다. 고정된 완전 연결 레이어 입력 차원 같은 제약도 만족해야 한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
onnx.checker의 성공이 PyTorch 출력과 같다는 뜻인가?

</div>

??? success "연습문제 2 풀이"
    아니다. 형식 검사는 구조와 연산 규약을 확인한다. 수치 출력은 별도로 비교해야 하며 여러 입력에서의 점검도 모든 입력의 동등성을 증명하지는 않는다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
최대 절대 오차 1e-5를 모든 모델의 합격 기준으로 사용할 수 있는가?

</div>

??? success "연습문제 3 풀이"
    출력의 스케일과 자료형에 따라 의미가 달라진다. 절대·상대 허용 오차를 과제에 맞게 정하고 대표 입력, 배치 크기, 실행 제공자에서 수치 차이와 예측 영향을 함께 확인한다.

## 정리하며

ONNX 내보내기는 exporter, opset과 런타임의 조합에 의존한다. 동적 축의 범위와 실제 모델 제약을 구별하고, 형식 검사와 수치 비교를 별도로 수행하는 과정을 이해한다.
