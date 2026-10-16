# 합성 데이터셋으로 YOLO 학습 구성하기

원·사각형·삼각형을 그린 합성 이미지와 YOLO 형식 레이블을 만들고, 사전 학습된 YOLOv8n을 파인튜닝하는 예제다. 목적은 이미지·레이블·클래스 목록·학습 설정을 연결하는 방법을 이해하는 것이다. 코드와 기존 출력은 학습 절차의 설명 자료이며 여기서 다시 실행하거나 그림을 생성할 필요는 없다.

## 1. 데이터 생성과 학습 코드

```python
"""
보기 3: 맞춤 물체 알아내기 익히기
============================================

인공 자료로 맞춤 물체에 대해 YOLO를 익힌다.
온전한 익히기 물길을 보여 준다.
"""

import os
import yaml
import numpy as np
from PIL import Image, ImageDraw
from pathlib import Path

# ========================================================================
# 메인
# ========================================================================

try:
    from ultralytics import YOLO
except:
    print("Install ultralytics: pip install ultralytics")
    exit()

print("="*70)
print("CUSTOM OBJECT DETECTION TRAINING")
print("="*70)

# 합성 데이터셋 만들기
def create_custom_dataset(root='custom_dataset', num_images=100):
    """YOLO 꼴의 인공 자료 뭉치를 만든다."""
    
    # 디렉터리를 만든다
    for split in ['train', 'val']:
        Path(f'{root}/images/{split}').mkdir(parents=True, exist_ok=True)
        Path(f'{root}/labels/{split}').mkdir(parents=True, exist_ok=True)
    
    classes = ['circle', 'square', 'triangle']
    
    for split, count in [('train', num_images), ('val', num_images//5)]:
        for i in range(count):
            # 그림 만들기
            img = Image.new('RGB', (640, 640), (240, 240, 240))
            draw = ImageDraw.Draw(img)
            
            # 물체 만들기
            labels = []
            for _ in range(np.random.randint(1, 4)):
                cls = np.random.randint(0, len(classes))
                x = np.random.randint(100, 540)
                y = np.random.randint(100, 540)
                s = np.random.randint(40, 80)
                
                if cls == 0:  # 동그라미
                    draw.ellipse([x-s, y-s, x+s, y+s], fill=(255,100,100))
                elif cls == 1:  # 정사각형
                    draw.rectangle([x-s, y-s, x+s, y+s], fill=(100,255,100))
                else:  # 세모
                    draw.polygon([(x, y-s), (x-s, y+s), (x+s, y+s)], fill=(100,100,255))
                
                # YOLO 꼴: 클래스 x_가운데 y_가운데 너비 높이(고르게 맞춘 값)
                x_center, y_center = x/640, y/640
                width, height = (2*s)/640, (2*s)/640
                labels.append(f"{cls} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            # 그림과 이름표 갈무리
            img.save(f'{root}/images/{split}/img_{i:04d}.jpg')
            with open(f'{root}/labels/{split}/img_{i:04d}.txt', 'w') as f:
                f.write('\n'.join(labels))
    
    # data.yaml 만들기
    data_yaml = {
        'path': os.path.abspath(root),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    with open(f'{root}/data.yaml', 'w') as f:
        yaml.dump(data_yaml, f)
    
    return f'{root}/data.yaml'

# 데이터셋 생성
print("Creating synthetic custom dataset...")
data_yaml = create_custom_dataset()
print(f"Dataset created: {data_yaml}")
print("  - 100 training images")
print("  - 20 validation images")
print("  - 3 classes: circle, square, triangle\n")

# 모델을 불러온다
print("Loading YOLOv8n model...")
model = YOLO('yolov8n.pt')

# 학습
print("\nTraining model...")
print("(This will take a few minutes)")
results = model.train(
    data=data_yaml,
    epochs=10,
    imgsz=640,
    batch=16,
    name='custom_detection',
    verbose=False
)

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nModel saved to: runs/detect/custom_detection/weights/best.pt")
print("\nTo use trained model:")
print("  model = YOLO('runs/detect/custom_detection/weights/best.pt')")
print("  results = model('image.jpg')")
print("="*70)


if __name__ == "__main__":
    pass
```

**기존 출력:**

```
Install ultralytics: pip install ultralytics
```

## 2. 이미지와 객체 레이블의 대응

기본값은 학습 이미지 100개와 검증 이미지 20개다. 각 640×640 이미지에 객체 1~3개를 그리고, 클래스 0은 원·1은 사각형·2는 삼각형으로 대응시킨다. 각 레이블 행은 `class cx cy w h`이며 중심과 크기를 640으로 나눈 정규화 값이다. 예를 들어 x=320,y=160,s=64이면 중심은 (0.5,0.25), 너비·높이는 각각 0.2다.

삼각형도 윤곽을 따라가는 마스크가 아니라 이를 감싸는 직사각형을 정답으로 사용한다. 중심·반크기 범위가 정해져 모든 객체는 이미지 안에 들어가지만 서로 겹칠 수 있다. 나중에 그린 도형이 앞 도형을 가려도 먼저 기록한 상자는 그대로 남으므로, 가림이 심한 표본의 주석 의미를 별도로 생각해야 한다.

원은 빨강, 사각형은 초록, 삼각형은 파랑으로 고정되어 클래스와 색상이 완전히 연결된다. 모델이 모양보다 색상을 이용해도 좋은 검증 점수를 얻을 수 있는 쉬운 설정이다. 흰색에 가까운 단색 배경·고정 해상도·한정된 크기의 결과를 일반 사진의 탐지 성능으로 해석하지 않는다.

## 3. 데이터 YAML과 분할

폴더는 images/train·images/val과 같은 이름의 labels 하위 폴더로 대응시킨다. `data.yaml`의 names 순서와 0부터 시작하는 클래스 ID, nc=3이 일치해야 한다. 이 파일은 데이터 경로와 분할을 지정할 뿐 객체 상자를 대신 저장하지 않는다. 형식은 [공식 데이터 설명](https://docs.ultralytics.com/datasets/detect/)을 참고한다.

학습·검증 이미지는 같은 합성 생성 규칙에서 별도로 만들어지며 독립 시험 분할은 없다. 난수 시드를 고정하지 않아 생성 결과도 실행마다 달라질 수 있다. 기존 폴더의 파일을 정리하지 않으므로 이미지 수를 줄여 다시 생성하면 이전 파일이 남아 실제 데이터 수가 출력 설명과 달라질 수 있다.

실제 자료로 확장할 때는 관련 영상·촬영 대상·연속 프레임이 분할 사이에 섞이지 않게 하고, 다양한 배경·색상·크기·객체 없음·가림 조건에서 평가한다. 합성 검증으로 튜닝한 결과와 새로운 실제 도메인의 일반화를 구분한다.

## 4. 사전 학습 모델과 실행 범위

`YOLO('yolov8n.pt')`에서 시작하므로 무작위 초기화한 탐지기를 처음부터 학습하는 예가 아니다. train 호출은 새 클래스 집합을 가진 데이터에 파인튜닝하는 라이브러리 경로를 사용한다. 10 epoch·batch16이라는 설정만으로 수렴·정확도·실행 시간이 보장되지 않는다. 실제 시간은 장치와 설치 환경에 달렸다.

코드는 완료 메시지에 고정된 best.pt 경로를 출력하지만 같은 실행 이름이 이미 있으면 라이브러리가 저장 디렉터리에 번호를 붙일 수 있다. 실제 실행의 저장 경로·선택된 체크포인트를 확인해야 하며 메시지만으로 그 경로에 원하는 웨이트가 존재한다고 단정하지 않는다.

데이터 생성·모델 로드·학습은 최상위에서 실행되고 마지막 main 조건문은 pass뿐이다. 따라서 import해도 폴더·파일 생성과 학습이 시작되는 구조다. 이 예제에는 자체 nn.Module 클래스나 뉴럴 넷트웍 내부 구조를 구현한 부분이 없으므로 그 구성 요소를 직접 설계했다고 설명하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
세 클래스의 색을 고정한 데이터에서 높은 검증 점수만으로 모양 인식을 입증하기 어려운 이유를 설명하라.

</div>

??? success "연습문제 1 풀이"

    색이 클래스와 완전히 연결되어 색상만으로 예측하는 지름길이 가능하다. 색을 독립적으로 바꾸고 배경·크기·가림을 다양화한 평가와 실제 자료의 별도 시험이 있어야 모양과 일반화에 대한 주장을 확인할 수 있다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
레이블 한 행과 데이터 YAML에서 확인할 계약을 설명하라.

</div>

??? success "연습문제 2 풀이"

    클래스 ID는 names 순서의 0~2이고 네 좌표는 유한한 정규화 cxcywh다. 너비·높이는 양수이고 이미지·레이블 파일 이름이 대응해야 한다. nc와 names 길이, 분할 경로의 존재·독립성도 확인한다. YOLO의 외부 결과 객체에 B×C×H×W assert를 무조건 적용하는 검사는 맞지 않는다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
재생성한 데이터 수와 출력된 모델 경로가 실제와 달라질 수 있는 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"

    기존 이미지·레이블을 지우지 않아 더 작은 수로 재생성하면 이전 파일이 남는다. 학습 디렉터리 이름도 중복 시 자동 변경될 수 있다. 고정 메시지 대신 실제 데이터 목록과 실행이 기록한 저장 경로를 기준으로 해석해야 한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 기존 시험 조각의 한계를 설명하고 이 예제에 맞는 확인 항목을 제안하라.

</div>

??? success "연습문제 4 풀이"

    공백이 들어간 식별자와 정의되지 않은 Example 3·입력 변수 때문에 실행 가능한 시험이 아니다. 이 예제에는 자체 PyTorch 모델 forward가 없다. 이미지와 레이블 수·파일 대응·클래스 범위·좌표의 유효성·분할 격리·다시 생성할 때 남는 파일·실제 저장 경로를 확인하는 명세가 적절하다. 아래 조각은 기존 자리 표시자로만 읽는다.

    ```python
    def test_example 3():
        model = Example 3(...)
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

사용자 데이터 학습에는 이미지와 레이블의 대응·정규화 좌표·클래스 인덱스·분할과 저장 경로가 맞아야 한다. 고정된 색상 도형의 합성 실험은 전체 학습 구성을 설명하지만 실제 이미지의 탐지 성능을 입증하지 않는다.
