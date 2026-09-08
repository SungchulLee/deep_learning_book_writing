# TorchScript 내보내기

TorchScript는 실전 배포를 위해 직렬화하고 최적화할 수 있는 모델 표현을 만든다. 추적은 순전파 동안의 연산을 기록하고, 스크립팅은 파이썬 소스 코드를 분석하여 제어 흐름까지 다룬다. TorchScript 모델은 C++ 환경에서도 불러올 수 있고 연산자 융합이나 상수 접기 같은 최적화의 덕을 본다.

## 1. 코드

```python
#!/usr/bin/env python3
"""
============================================================
TorchScript: 실전 배포를 위한 최적화된 모델 내보내기
============================================================

TorchScript는 실전 배포를 위해 직렬화하고 최적화할 수 있는 모델을
만드는 PyTorch의 방법이다.

주제:
- 추적과 스크립팅
- TorchScript로 변환하기
- 모델 최적화
- 모바일 배포 준비
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================

print("=" * 70)
print("TORCHSCRIPT EXPORT TUTORIAL")
print("=" * 70)

# 모델 정의
class SimpleModel(nn.Module):
    """간단한 순방향 신경망"""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ModelWithControlFlow(nn.Module):
    """제어 흐름이 있는 모델 - 스크립팅이 필요하다"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        if x.sum() > 0:
            x = self.fc(x)
        else:
            x = x * 2
        return x


# 방법 1: 추적
print("\n" + "=" * 70)
print("METHOD 1: TRACING")
print("=" * 70)

model = SimpleModel()
model.eval()

example_input = torch.randn(1, 10)

# 모델 추적
traced_model = torch.jit.trace(model, example_input)
print("\nModel successfully traced")

# 추적한 모델 저장
traced_path = "traced_model.pt"
traced_model.save(traced_path)
print(f"Traced model saved to '{traced_path}'")

# 추적한 모델 시험
test_input = torch.randn(2, 10)

with torch.no_grad():
    original_output = model(test_input)
    traced_output = traced_model(test_input)

diff = torch.abs(original_output - traced_output).max().item()
print(f"Max difference: {diff:.6f}")

if diff < 1e-6:
    print("Traced model produces identical results")

# 뒷정리
import os
if os.path.exists(traced_path):
    os.remove(traced_path)


# 방법 2: 스크립팅
print("\n" + "=" * 70)
print("METHOD 2: SCRIPTING")
print("=" * 70)

# 간단한 모델 스크립팅
simple_model = SimpleModel()
simple_model.eval()

scripted_simple = torch.jit.script(simple_model)
print("\nSimple model successfully scripted")

scripted_path = "scripted_simple.pt"
scripted_simple.save(scripted_path)
print(f"Scripted model saved to '{scripted_path}'")

# 뒷정리
if os.path.exists(scripted_path):
    os.remove(scripted_path)

# 제어 흐름이 있는 모델 스크립팅
print("\nScripting model with control flow...")
control_flow_model = ModelWithControlFlow()
control_flow_model.eval()

scripted_control = torch.jit.script(control_flow_model)
print("Control flow model successfully scripted")

# 여러 입력으로 시험
pos_input = torch.ones(1, 10)
neg_input = -torch.ones(1, 10)

with torch.no_grad():
    pos_result = scripted_control(pos_input)
    neg_result = scripted_control(neg_input)

print("Control flow preserved correctly")


# 최적화
print("\n" + "=" * 70)
print("TORCHSCRIPT OPTIMIZATIONS")
print("=" * 70)

model = SimpleModel()
model.eval()

scripted_model = torch.jit.script(model)

print("\nApplying optimizations...")

# 모델 얼리기
frozen_model = torch.jit.freeze(scripted_model)
print("Model frozen")

# 추론을 위해 최적화
optimized_model = torch.jit.optimize_for_inference(frozen_model)
print("Optimized for inference")

print("\nOptimizations applied:")
print("- Constant propagation")
print("- Dead code elimination")
print("- Operator fusion")
print("- Memory layout optimization")

print("\n" + "=" * 70)
print("TUTORIAL COMPLETE")
print("=" * 70)


if __name__ == "__main__":
    pass
```

## 2. 논의

추적과 스크립팅은 TorchScript로 변환하는 서로 보완적인 두 방법이다. 추적은 실행 경로 하나만 기록하므로 간단하지만 데이터에 따라 달라지는 제어 흐름(텐서 값에 따른 if 문)을 담지 못한다. 스크립팅은 파이썬 소스 코드를 분석하여 제어 흐름을 다룰 수 있지만, 코드가 파이썬의 제한된 부분집합만 써야 한다.

얼린 모델은 상수 전파와 죽은 코드 제거를 적용하여 매개변수를 현재 값으로 바꾸고 쓰이지 않는 가지를 없앤다. 그 결과 원래의 파이썬 클래스 정의에 더는 기대지 않는 자립적인 모델이 된다.

`torch.jit.optimize_for_inference`은 연산자 융합(여러 연산을 하나의 커널로 합치기)과 메모리 배치 최적화 같은 추가 최적화를 적용한다. 그 결과 모델은 대체로 원래 PyTorch 모델보다 빠르며, 특히 파이썬 디스패치의 부담이 큰 배포 상황에서 그렇다.

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

**다룬 것** — TorchScript 내보내기

추적과 스크립팅은 TorchScript로 변환하는 서로 보완적인 두 방법이다.

핵심 클래스는 `SimpleModel`, `ModelWithControlFlow`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
