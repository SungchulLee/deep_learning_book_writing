# TorchScript 내보내기

TorchScript의 추적과 스크립팅, 평가 모델의 동결과 추론 최적화를 살펴보는 기존 배포 예제이다. 현재 공식 문서에서는 TorchScript를 deprecated로 표시하며 새 내보내기에는 torch.export를 안내한다. 이 페이지의 코드는 기존 방식을 이해하기 위한 자료로 읽는다.

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
    """간단한 순방향 뉴럴 넷트웍"""
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

**출력:**

```
======================================================================
TORCHSCRIPT EXPORT TUTORIAL
======================================================================

======================================================================
METHOD 1: TRACING
======================================================================

Model successfully traced
Traced model saved to 'traced_model.pt'
Max difference: 0.000000
Traced model produces identical results

======================================================================
METHOD 2: SCRIPTING
======================================================================

Simple model successfully scripted
Scripted model saved to 'scripted_simple.pt'

Scripting model with control flow...
Control flow model successfully scripted
Control flow preserved correctly

======================================================================
TORCHSCRIPT OPTIMIZATIONS
======================================================================

Applying optimizations...
Model frozen
Optimized for inference

Optimizations applied:
- Constant propagation
- Dead code elimination
- Operator fusion
- Memory layout optimization

======================================================================
TUTORIAL COMPLETE
======================================================================
```

## 2. 논의

추적은 예시 입력에서 실행된 텐서 연산을 기록한다. 데이터 값에 따라 달라지는 파이썬 분기는 해당 경로로 굳어질 수 있다. 스크립팅은 지원되는 파이썬 부분집합을 분석해 제어 흐름을 표현한다. ModelWithControlFlow는 입력 합이 양수인지에 따라 선형 레이어 또는 두 배 연산을 선택하므로 이런 차이를 설명하는 예이다.

freeze는 평가 모드의 ScriptModule에서 변경되지 않는 속성 등을 그래프에 반영해 최적화할 수 있게 한다. 파이썬 원본 클래스에 의존하지 않는 저장·로딩은 TorchScript 표현의 특성이며 freeze만의 효과는 아니다. 필요한 속성이나 사용자 정의 연산과 런타임 지원은 별도로 고려한다.

optimize_for_inference는 적용 가능한 그래프와 실행 환경에서 추론 최적화를 시도한다. 모든 모델에 모든 최적화가 적용되거나 실행이 빨라진다는 보장은 없다. 이 코드는 시간을 측정하지 않으므로 출력 문구만으로 속도 개선을 입증하지 않는다. 양수·음수 입력으로 분기를 호출하는 부분도 원본 출력과의 명시적 비교는 하지 않는다. 현재 상태는 [공식 배포 안내](https://docs.pytorch.org/tutorials/advanced/cpp_export.html)에서 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
ModelWithControlFlow를 양수 입력으로 추적하면 어떤 위험이 있는가?

</div>

??? success "연습문제 1 풀이"
    실행된 선형 레이어 경로만 기록되어 음수 합 입력에서도 원래 파이썬 모델과 다른 결과를 낼 수 있다. 입력 값에 따른 분기를 표현할 수 있는 변환 방식을 사용해야 한다.

---


<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
freeze 전에 eval()이 필요한 이유는 무엇인가?

</div>

??? success "연습문제 2 풀이"
    동결은 평가 동작을 기준으로 변경되지 않는 상태를 그래프에 반영하는 과정이다. 드롭아웃이나 BatchNorm의 학습 동작을 그대로 유지하며 훈련하는 용도로 사용하는 과정이 아니다.

---


<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
추론 최적화의 속도 개선을 평가하려면 어떤 조건을 통제해야 하는가?

</div>

??? success "연습문제 3 풀이"
    동일한 장치·자료형·입력 크기에서 워밍업 후 반복 측정하고 출력 차이도 확인한다. GPU라면 비동기 실행을 고려한 동기화가 필요하다. 이 페이지에는 그런 성능 측정이 포함되어 있지 않다.

## 정리하며

TorchScript 예제에서는 추적의 경로 제한과 스크립팅의 지원 범위를 구분한다. 변환·동결·최적화의 성공 문구와 의미적 동등성·속도 검증은 서로 다른 확인 사항이다.
