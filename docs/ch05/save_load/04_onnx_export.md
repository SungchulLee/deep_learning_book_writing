# ONNX 내보내기

ONNX(Open Neural Network Exchange)는 딥러닝 모델을 나타내는 열린 형식으로, 여러 프레임워크와 배포 플랫폼 사이를 오갈 수 있게 해 준다. PyTorch 모델을 ONNX로 내보내려면 임시 입력으로 모델을 추적하고, 배치 크기가 변할 수 있는 동적 축을 지정하며, ONNX Runtime으로 출력이 일치하는지 확인한다.

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
    ONNX 내보내기를 위해 여러 종류의 층을 보인다.
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
    export_params=True,             # 학습된 매개변수 저장
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

ONNX 내보내기는 추적을 쓴다. PyTorch가 임시 입력으로 순전파를 돌리며 수행된 연산을 기록하고 이를 ONNX 연산자로 옮긴다. `opset_version` 매개변수는 어떤 ONNX 연산을 쓸 수 있는지를 정한다. 버전이 높을수록 연산자가 많지만 실행 환경의 지원 폭은 좁을 수 있다.

동적 축을 쓰면 내보낸 모델이 크기가 변하는 입력을 받을 수 있다. `dynamic_axes`을 지정하지 않으면 모델은 내보낼 때 쓴 입력 모양에 맞추어 굳는다. `{'input': {0: 'batch_size'}}`으로 두면 추론 시점에 아무 배치 크기나 쓸 수 있다.

검증은 같은 입력에 대해 원래 PyTorch 모델과 ONNX Runtime 추론 세션의 출력을 견주어 본다. 최대 차이가 $10^{-5}$ 아래이면 변환이 수치 정확도를 지켰다고 볼 수 있다. 차이가 더 크면 내보내는 동안 지원되지 않아 근사된 연산이 있을 수 있다.

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

**다룬 것** — ONNX 내보내기

ONNX 내보내기는 추적을 쓴다.

핵심 클래스는 `ConvNet`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
