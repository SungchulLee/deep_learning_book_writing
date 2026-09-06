# 보기 4

보기 4: 앞선 물체 알아내기 재주. 실전에 쓸 수 있는 알아내기 재주를 보여 준다.

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 물체 알아내기를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 코드

```python
"""
보기 4: 앞선 물체 알아내기 재주
================================================

실전에 쓸 수 있는 알아내기 재주를 보여 준다.
"""

import torch
import numpy as np
import time
from pathlib import Path

# ========================================================================
# 메인
# ========================================================================

try:
    from ultralytics import YOLO
except:
    print("Install: pip install ultralytics")
    exit()

print("="*70)
print("ADVANCED OBJECT DETECTION TECHNIQUES")
print("="*70)

# 모델을 불러온다
model = YOLO('yolov8n.pt')

# 재주 1: 여러 잣수 미룸
print("\n1. Multi-Scale Inference")
print("-"*70)
print("Testing at multiple scales improves accuracy")

from PIL import Image
test_img = Image.new('RGB', (640, 640), (200, 200, 200))

scales = [0.8, 1.0, 1.2]
for scale in scales:
    size = int(640 * scale)
    resized = test_img.resize((size, size))
    results = model(resized, verbose=False)
    print(f"Scale {scale}: {len(results[0].boxes)} detections")

# 재주 2: 모델 내보내기
print("\n2. Model Export to ONNX")
print("-"*70)
print("Exporting model to ONNX for production deployment...")

try:
    model.export(format='onnx', dynamic=False, simplify=True)
    print("✓ Model exported to ONNX successfully")
    print("  File: yolov8n.onnx")
except Exception as e:
    print(f"Export failed: {e}")

# 재주 3: 묶음 단위 다루기
print("\n3. Batch Processing")
print("-"*70)

batch_size = 4
images = [test_img] * batch_size

start = time.time()
results = model(images, verbose=False)
batch_time = time.time() - start

print(f"Processed {batch_size} images in {batch_time*1000:.2f} ms")
print(f"Per-image: {batch_time*1000/batch_size:.2f} ms")

# 재주 4: 믿음도 눈금 맞추기
print("\n4. Confidence Thresholds")
print("-"*70)

thresholds = [0.25, 0.5, 0.75]
for conf in thresholds:
    results = model(test_img, conf=conf, verbose=False)
    print(f"Confidence {conf}: {len(results[0].boxes)} detections")

# 요약
print("\n" + "="*70)
print("ADVANCED TECHNIQUES COMPLETE!")
print("="*70)
print("\nTechniques Covered:")
print("✓ Multi-scale inference for accuracy")
print("✓ ONNX export for deployment")
print("✓ Batch processing for efficiency")
print("✓ Confidence calibration")
print("\nProduction Ready!")
print("="*70)


if __name__ == "__main__":
    pass```

## 논의

여기 짠 것은 깔끔하고 읽기 좋은 PyTorch 코드로 물체 알아내기의 핵심 개념을 보여 준다. 단원별로 나뉜 짜임 덕분에 낱낱의 조각을 익히고 다른 일이나 자료 뭉치에 맞게 고치기 쉽다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 컴퓨터 비전 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
코드를 끝까지 읽고 핵심 꾸밈 결정을 가려내어라. 구체적인 짜기 고름 세 가지를 들고 저마다 왜 물체 알아내기에 알맞은지 설명하여라.

??? success "연습문제 1 풀이"
    설계 결정은 구현마다 다르지만 흔히 다음이 포함된다. (1) 활성화 함수의 선택 — ReLU 계열은 포화되지 않는 경사를 주어 학습을 빠르게 한다. (2) 정규화 전략 — 배치 정규화는 내부 공변량 이동을 줄여 학습을 안정시킨다. (3) 잔차 연결 — 있을 경우 건너뛰는 경로를 제공하여 깊은 신경망에서도 경사가 흐르게 한다. 각 선택은 표현력, 계산 비용, 학습 안정성 사이의 절충을 반영한다.

---

**연습문제 2.**
입력이 기대하는 모양과 자료형을 갖는지 확인하도록 주 함수나 클래스에 입력 검증을 추가하라. 잘못된 입력에는 유익한 오류 메시지를 내라.

??? success "연습문제 2 풀이"
    `forward` 메서드(또는 해당 함수)의 첫머리에 다음과 같은 검사를 추가한다. `assert x.dim() == expected_dims, f'Expected {expected_dims}D input, got {x.dim()}D'`와 `assert x.dtype == torch.float32, f'Expected float32, got {x.dtype}'`. 모양을 검증할 때는 중요한 차원을 확인한다. `B, C, H, W = x.shape; assert C == self.expected_channels`. 유익한 오류 메시지는 디버깅 속도를 크게 높이고 코드를 재사용하기에도 더 견고하게 만든다.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
Example 4의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_example 4():
        model = Example 4(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (넘침을 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 경사의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.
