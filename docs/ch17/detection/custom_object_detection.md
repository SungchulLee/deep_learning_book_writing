# 보기 3

보기 3: 맞춤 물체 알아내기 익히기. 인공 자료로 맞춤 물체에 대해 YOLO를 익힌다.

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 물체 알아내기를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 코드

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
                
                # YOLO 꼴: 갈래 x_가운데 y_가운데 너비 높이(고르게 맞춘 값)
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
Example 3의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
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
    경사의 흐름을 시험하는 것은 그 구조가 처음부터 끝까지 이어지는 학습을 지원하는지 확인하는 데 특히 중요하다.
