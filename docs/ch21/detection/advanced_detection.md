# 객체 탐지 추론 설정과 ONNX 내보내기

YOLOv8n에서 입력 이미지 크기 변경, ONNX 내보내기, 배치 추론과 신뢰도 임계값을 살펴보는 예제다. 입력은 물체가 없는 단색 이미지이며 정확도 평가나 배포 시스템의 검증을 수행하지 않는다. 기존 출력의 ‘Production Ready’는 그러한 검증을 대신하지 않는다.

## 1. 코드와 기존 출력

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

# 재주 3: 배치 단위 다루기
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
    pass
```

**기존 출력:**

```
Install: pip install ultralytics
```

## 2. 이미지 resize와 다중 스케일 결합의 차이

코드는 같은 회색 이미지를 512·640·768 크기로 바꾸어 각각 모델에 전달하고 탐지 수를 출력한다. 여러 스케일에서 얻은 상자를 원본 좌표로 모아 NMS·융합하는 다중 스케일 앙상블은 아니다. 고수준 모델 호출은 자체 resize·letterbox를 적용하므로 입력 PIL 이미지 크기를 바꿨다고 실제 뉴럴 넷트웍 입력 해상도도 같은 비율로 바뀌었다고 단정할 수 없다. `imgsz`와 전처리된 텐서 크기를 구분해야 한다.

단색 이미지에는 정답 객체가 없고 AP도 계산하지 않는다. 탐지 수만으로 해상도 증가가 정확도를 개선했다고 말할 수 없다. 실제 다중 스케일 평가에는 좌표 역변환·결과 융합·동일 평가 집합과 추가 계산비용의 비교가 필요하다.

## 3. ONNX 내보내기의 의미

export는 모델을 다른 런타임에서 사용할 수 있는 형식으로 저장하는 단계다. `dynamic=False`는 내보내기에서 정한 입력 크기를 고정하는 설정이며, 임의 크기를 모두 받는 모델로 해석하지 않는다. `simplify=True`의 처리와 의존성은 설치 환경에 달렸다. 이 예제는 예외를 출력한 뒤 계속 진행하므로 뒤의 완료 메시지만으로 내보내기 성공을 판정할 수 없다.

성공적으로 파일을 만들더라도 실제 ONNX 런타임에서 실행하거나 PyTorch 결과와 비교한 것은 아니다. 전처리·출력 디코딩·NMS·좌표 역변환, 목표 장치의 정확도·지연시간을 따로 확인해야 한다. 관련 옵션은 [공식 export 설명](https://docs.ultralytics.com/modes/export/)을 기준으로 설치 버전과 맞춘다.

## 4. 배치 시간과 신뢰도 임계값

배치 목록은 같은 PIL 객체를 네 번 참조한다. 모델 입력 네 장의 처리 형식은 보여 주지만 서로 다른 장면의 성능 비교는 아니다. 전체 호출 시간/4는 이 배치의 이미지당 평균 비용이며 단일 요청의 응답 지연시간이 아니다. 한 번의 time.time 측정이고 명시적인 GPU 동기화나 반복 분포도 없어 안정된 처리량으로 일반화하지 않는다.

conf=0.25·0.5·0.75는 유지할 예측의 점수 기준을 바꾼다. 정답과 예측 확률의 일치도를 학습·측정하지 않으므로 **confidence calibration이 아니라 임계값 변경**이다. 진짜 calibration 평가는 정답을 가진 별도 데이터와 탐지 매칭·점수 정의를 필요로 한다. 높은 임계값이 모든 상황에서 더 좋은 운영점을 보장하지 않는다.

## 5. 예제의 실행 범위

웨이트 로드·추론·export는 최상위에서 수행되고 main 조건문은 pass뿐이다. import해도 실행되며 파일 저장·웨이트 다운로드가 생길 수 있다. 이 절은 설정의 의미와 한계를 읽는 자료다. 배포 준비의 근거가 없는 고정 완료 문구나 실행 시간을 실제 검증 결과로 인용하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
현재 세 크기의 추론을 다중 스케일 앙상블이라고 할 수 있는가?

</div>

??? success "풀이"

    각 호출의 탐지 수만 출력하고 원본 좌표로 모아 결합하지 않으므로 아니다. 라이브러리 전처리가 이미지를 다시 조정할 수도 있어 실제 네트워크 입력 해상도도 별도로 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
dynamic=False로 export한 모델의 입력과 결과를 검증할 때 필요한 조건을 설명하라.

</div>

??? success "풀이"

    고정 입력 크기·배치·자료형과 전처리 계약을 확인한다. 런타임의 출력 형식·좌표·점수·NMS 포함 여부를 맞춰 원 모델과 비교해야 한다. 파일 생성 성공만으로 추론 결과나 배포 지연시간이 검증되는 것은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
신뢰도 임계값 변경과 calibration의 차이를 설명하라.

</div>

??? success "풀이"

    임계값 변경은 이미 계산된 점수로 후보를 선택하는 규칙이다. calibration은 점수와 실제 정답 빈도의 관계를 평가하거나 보정하는 문제다. 이 단색 이미지에서 탐지 수를 비교하는 코드는 정답 기반 calibration을 수행하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 기존 시험 조각이 이 고수준 추론·export 예제의 확인 방법으로 적절한가?

</div>

??? success "풀이"

    공백 식별자·정의되지 않은 Example 4와 변수 때문에 실행 가능한 시험이 아니다. 추론 Results에 고정 shape·backward를 요구하는 것도 맞지 않는다. 빈 탐지·입력 크기·배치 결과 대응, export 실패 시 파일 상태, 실제 런타임 출력의 좌표·점수 계약을 확인하는 명세가 적절하다. 기존 조각은 자리 표시자로 읽는다.

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


## 정리하며

resize별 독립 추론은 다중 스케일 결과 융합과 다르고 점수 임계값 변경은 calibration과 다르다. ONNX 파일 생성·배치 호출은 배포 과정의 일부이며 목표 런타임의 정확도·입출력·성능 검증을 대신하지 않는다.
