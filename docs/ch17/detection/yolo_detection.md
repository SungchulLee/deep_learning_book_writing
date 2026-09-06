# 보기 2

보기 2: YOLO 물체 알아내기. 이 각본은 미리 익힌 YOLOv8을 물체 알아내기에 쓰는 법을 보여 준다.

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 물체 알아내기를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 코드

```python
"""
보기 2: YOLO 물체 알아내기
=================================

이 각본은 미리 익힌 YOLOv8을 물체 알아내기에 쓰는 법을 보여 준다.
YOLO(한 번만 본다)는 빠르고 정확한 한 단계 알아내개이며
실시간 쓰임새에 안성맞춤이다.

핵심 개념:
- 미리 익힌 YOLO 모델 읽어 들이기
- 그림에 미룸 돌리기
- YOLO의 어림 이해하기
- 그려 보기와 읽어 내기
- 모델 견줌(크기별)

지은이: PyTorch Object Detection Tutorial
날짜: 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import time
import os

# ultralytics(YOLOv8) 들여오기 시도
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("⚠ ultralytics not installed. Installing...")
    print("Run: pip install ultralytics")
    YOLO_AVAILABLE = False

# 난수 씨앗 고정
np.random.seed(42)
torch.manual_seed(42)

print("="*70)
print("YOLO OBJECT DETECTION")
print("="*70)
print("\nThis example demonstrates:")
print("1. Loading pre-trained YOLOv8 models")
print("2. Running object detection on images")
print("3. Understanding YOLO outputs")
print("4. Comparing different model sizes")
print("5. Real-time performance analysis\n")

# 기기 살피기
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cpu':
    print("⚠ GPU not available. Inference will be slower.\n")
else:
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}\n")

# ============================================================================
# 1단계: 인공 시험 그림 만들기
# ============================================================================
"""
보여 주려고 단순한 물체가 있는 인공 그림을 만든다.
실전에서는 실제 그림을 쓴다.
"""

def create_test_image(size=(640, 640), num_objects=3):
    """시험용 인공 그림을 만든다."""
    img = Image.new('RGB', size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # 미리 정해 둔 물체 갈래
    object_types = [
        {'shape': 'rectangle', 'color': (255, 100, 100), 'label': 'red_box'},
        {'shape': 'ellipse', 'color': (100, 100, 255), 'label': 'blue_circle'},
        {'shape': 'rectangle', 'color': (100, 255, 100), 'label': 'green_box'},
    ]
    
    objects_drawn = []
    
    for i in range(min(num_objects, len(object_types))):
        obj_type = object_types[i]
        
        # 마구잡이 자리와 크기
        x = np.random.randint(50, size[0] - 150)
        y = np.random.randint(50, size[1] - 150)
        w = np.random.randint(80, 150)
        h = np.random.randint(80, 150)
        
        bbox = [x, y, x + w, y + h]
        
        if obj_type['shape'] == 'rectangle':
            draw.rectangle(bbox, fill=obj_type['color'], outline=(0, 0, 0), width=3)
        else:
            draw.ellipse(bbox, fill=obj_type['color'], outline=(0, 0, 0), width=3)
        
        objects_drawn.append({
            'bbox': bbox,
            'label': obj_type['label'],
            'shape': obj_type['shape']
        })
    
    return img, objects_drawn


print("Step 1: Creating Test Images")
print("-" * 70)

# 시험 그림 여럿 만들기
test_images = []
for i in range(3):
    img, objects = create_test_image()
    test_images.append((img, objects))
    print(f"Created test image {i+1} with {len(objects)} objects")

print()

# ============================================================================
# 2단계: 미리 익힌 YOLO 모델 읽어 들이기
# ============================================================================
"""
YOLOv8에는 여러 크기가 있다:
- yolov8n(나노): 가장 빠르고 작다(매개변수 320만)
- yolov8s(소형): 균형(매개변수 1120만)
- yolov8m(중형): 좋은 정확도(매개변수 2590만)
- yolov8l(대형): 더 나은 정확도(매개변수 4370만)
- yolov8x(초대형): 가장 좋은 정확도(매개변수 6820만)

빠르기를 위해 yolov8n으로 시작한다.
"""

if not YOLO_AVAILABLE:
    print("Skipping YOLO demo - ultralytics not installed")
    print("Install with: pip install ultralytics")
    exit()

print("Step 2: Loading Pre-trained YOLOv8 Model")
print("-" * 70)

# YOLOv8 나노 모델 읽어 들이기(처음 돌릴 때 내려받는다)
print("Loading YOLOv8n (nano) model...")
print("(First run will download ~6MB model weights)")

model = YOLO('yolov8n.pt')  # 없으면 저절로 내려받는다

print(f"✓ Model loaded successfully")
print(f"Model type: {type(model)}")
print(f"Model device: {next(model.model.parameters()).device}")

# 쓸 수 있으면 GPU로 옮기기
if device == 'cuda':
    model.to(device)
    print(f"✓ Model moved to GPU\n")
else:
    print()

# ============================================================================
# 3단계: YOLO의 어림 이해하기
# ============================================================================
"""
YOLO는 알아냄마다 다음을 내놓는다:
- 두름 상자: [x_최소, y_최소, x_최대, y_최대]
- 믿음도: 0과 1 사이의 실수
- 갈래: 정수 갈래 번호
- 갈래 이름: 글자 이름표

결과 개체에는 다음이 들어 있다:
- boxes: 모든 두름 상자
- names: 갈래 이름 대응(사전)
- conf: 믿음도 점수
- cls: 갈래 번호
"""

print("Step 3: Running Detection and Understanding Output")
print("-" * 70)

# 첫 시험 그림에 알아내기 돌리기
test_img = test_images[0][0]

print("Running inference...")
start_time = time.time()
results = model(test_img, verbose=False)
inference_time = (time.time() - start_time) * 1000

print(f"Inference time: {inference_time:.2f} ms")
print(f"\nResults type: {type(results)}")
print(f"Number of result objects: {len(results)}")

# 첫 결과 뽑아내기
result = results[0]

print(f"\nDetections found: {len(result.boxes)}")
if len(result.boxes) > 0:
    print("\nDetailed output structure:")
    print(f"  result.boxes: {type(result.boxes)}")
    print(f"  result.boxes.data: Shape {result.boxes.data.shape}")
    print(f"  Each detection: [x1, y1, x2, y2, confidence, class_id]")
    
    # 첫 알아냄의 세부 보이기
    if len(result.boxes) > 0:
        first_det = result.boxes[0]
        print(f"\nFirst detection:")
        print(f"  Box coordinates: {first_det.xyxy[0].cpu().numpy()}")
        print(f"  Confidence: {first_det.conf[0].cpu().numpy():.3f}")
        print(f"  Class ID: {int(first_det.cls[0].cpu().numpy())}")
        print(f"  Class name: {result.names[int(first_det.cls[0].cpu().numpy())]}")

print()

# ============================================================================
# 4단계: COCO 갈래 훑어보기
# ============================================================================
"""
YOLOv8은 갈래 80개의 COCO 자료 뭉치로 미리 익혀져 있다.
어떤 갈래를 알아낼 수 있는지 보자.
"""

print("Step 4: COCO Dataset Classes")
print("-" * 70)

# 갈래 이름 얻기
class_names = result.names
print(f"Total classes: {len(class_names)}")
print(f"\nSample of COCO classes:")

# 흔한 갈래 몇 개 보이기
common_classes = [0, 1, 2, 3, 15, 16, 17, 18]  # person, bicycle, car 등
for class_id in common_classes:
    print(f"  ID {class_id:2d}: {class_names[class_id]}")

print(f"\nFull class list:")
# 모든 갈래를 칸으로 찍기
class_list = [f"{i:2d}:{name}" for i, name in class_names.items()]
for i in range(0, len(class_list), 4):
    print("  " + "  ".join(class_list[i:i+4]))

print()

# ============================================================================
# 5단계: 믿음도 문턱값을 둔 알아내기
# ============================================================================
"""
매개변수로 알아내기의 몸짓을 다스릴 수 있다:
- conf: 믿음도 문턱값(붙박이 0.25)
- iou: NMS의 겹침 비 문턱값(붙박이 0.45)
- classes: 알아낼 특정 갈래
"""

print("Step 5: Detection with Different Thresholds")
print("-" * 70)

# 믿음도 문턱값을 달리해 시험
thresholds = [0.25, 0.5, 0.7]

for conf_thresh in thresholds:
    results = model(test_img, conf=conf_thresh, verbose=False)
    num_detections = len(results[0].boxes)
    print(f"Confidence threshold {conf_thresh:.2f}: {num_detections} detections")

print()

# ============================================================================
# 6단계: 알아냄 그려 보기
# ============================================================================
"""
YOLO는 그려 보기를 갖추고 있지만 여기서는 맞춤
그려 보기를 만들어 결과를 더 잘 이해한다.
"""

def visualize_yolo_results(image, result, title="YOLO Detections", save_path=None):
    """
    YOLO의 알아내기 결과를 그려 본다.
    
    인수:
        image: PIL 그림 또는 numpy 배열
        result: YOLO 결과 개체
        title: 그림의 제목
        save_path: 그림을 저장할 경로
    """
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image)
    
    # 알아냄 얻기
    boxes = result.boxes
    
    # 갈래마다 빛깔 정하기
    colors = plt.cm.tab20(np.linspace(0, 1, len(result.names)))
    
    for box in boxes:
        # 상자 자리표 뽑아내기
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = int(box.cls[0].cpu().numpy())
        class_name = result.names[cls]
        
        # 이 갈래의 빛깔 얻기
        color = colors[cls % len(colors)]
        
        # 네모 그리기
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # 이름표 더하기
        label_text = f'{class_name}: {conf:.2f}'
        ax.text(
            x1, y1 - 5,
            label_text,
            color='white',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
        )
    
    ax.set_title(f"{title} ({len(boxes)} detections)", fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.close()


print("Step 6: Creating Visualizations")
print("-" * 70)

# 알아내고 그려 보기
results = model(test_img, conf=0.25, verbose=False)
visualize_yolo_results(
    test_img,
    results[0],
    title="YOLOv8 Detections",
    save_path="yolo_detections.png"
)

print()

# ============================================================================
# 7단계: 모델 크기 견주기
# ============================================================================
"""
YOLOv8 모델 크기를 견준다:
- 빠르기(미룸 시간)
- 정확도(알아낸 물체)
- 모델 크기(매개변수)
"""

print("Step 7: Comparing YOLO Model Sizes")
print("-" * 70)

# 쓸 수 있는 모델(빠르게 하려면 주석 처리)
model_sizes = ['yolov8n.pt']  # 보임용으로 나노만
# 여럿을 견주려면 주석 해제:
# model_sizes = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt']

comparison_results = []

for model_name in model_sizes:
    print(f"\nTesting {model_name}...")
    
    # 모델을 불러온다
    test_model = YOLO(model_name)
    if device == 'cuda':
        test_model.to(device)
    
    # 몸풀기 돌리기
    _ = test_model(test_img, verbose=False)
    
    # 시간을 잰 돌리기
    times = []
    for _ in range(5):
        start = time.time()
        results = test_model(test_img, verbose=False)
        times.append((time.time() - start) * 1000)
    
    avg_time = np.mean(times)
    num_detections = len(results[0].boxes)
    
    comparison_results.append({
        'model': model_name,
        'time_ms': avg_time,
        'detections': num_detections
    })
    
    print(f"  Average inference time: {avg_time:.2f} ms")
    print(f"  Detections: {num_detections}")

print("\nComparison Summary:")
print("-" * 70)
print(f"{'Model':<15} {'Time (ms)':<15} {'Detections':<15}")
print("-" * 70)
for res in comparison_results:
    print(f"{res['model']:<15} {res['time_ms']:<15.2f} {res['detections']:<15}")

print()

# ============================================================================
# 8단계: 묶음 단위 다루기
# ============================================================================
"""
YOLO는 그림 여럿을 묶음으로 효율적으로 다룰 수 있다.
"""

print("Step 8: Batch Processing Multiple Images")
print("-" * 70)

# 시험 그림 모두 다루기
batch_images = [img for img, _ in test_images]

print(f"Processing {len(batch_images)} images in batch...")
start_time = time.time()
batch_results = model(batch_images, verbose=False)
batch_time = (time.time() - start_time) * 1000

print(f"Batch inference time: {batch_time:.2f} ms")
print(f"Average per image: {batch_time / len(batch_images):.2f} ms")

for i, result in enumerate(batch_results):
    print(f"  Image {i+1}: {len(result.boxes)} detections")

print()

# ============================================================================
# 9단계: 앞선 기능
# ============================================================================
"""
YOLOv8은 여러 앞선 기능을 받쳐 준다:
- 갈래 거르기
- 관심 자리(ROI)
- 맞춤 믿음도/겹침 비 문턱값
- 미룸 동안의 그림 불리기
"""

print("Step 9: Advanced YOLO Features")
print("-" * 70)

# 기능 1: 특정 갈래만 알아내기
print("Feature 1: Detect specific classes only")
print("  Detecting only persons, cars, and dogs (classes 0, 2, 15)...")
results = model(test_img, classes=[0, 2, 15], verbose=False)
print(f"  Detections: {len(results[0].boxes)}")

# 기능 2: 맞춤 문턱값
print("\nFeature 2: Custom confidence and IoU thresholds")
results_strict = model(test_img, conf=0.7, iou=0.3, verbose=False)
print(f"  High confidence (0.7): {len(results_strict[0].boxes)} detections")

# 기능 3: 어림의 세부 얻기
print("\nFeature 3: Accessing detailed predictions")
results = model(test_img, verbose=False)
if len(results[0].boxes) > 0:
    print("  Available attributes:")
    print(f"    - boxes.xyxy: Bounding boxes (x1, y1, x2, y2)")
    print(f"    - boxes.xywh: Bounding boxes (x_center, y_center, width, height)")
    print(f"    - boxes.conf: Confidence scores")
    print(f"    - boxes.cls: Class indices")
    print(f"    - names: Class name mapping")

print()

# ============================================================================
# 10단계: 성능 살피기
# ============================================================================
"""
YOLO의 성능 성질을 살핀다.
"""

print("Step 10: Performance Analysis")
print("-" * 70)

# 초당 틀 수 재기
num_runs = 10
times = []

for i in range(num_runs):
    start = time.time()
    _ = model(test_img, verbose=False)
    times.append(time.time() - start)

avg_time = np.mean(times) * 1000
std_time = np.std(times) * 1000
fps = 1000 / avg_time

print(f"Performance metrics ({num_runs} runs):")
print(f"  Average inference time: {avg_time:.2f} ± {std_time:.2f} ms")
print(f"  Estimated FPS: {fps:.1f}")
print(f"  Throughput: {fps * 3600:.0f} images/hour")

print(f"\nModel Information:")
print(f"  Device: {device}")
print(f"  Model: YOLOv8n")
print(f"  Input size: 640×640")

print()

# ============================================================================
# 요약
# ============================================================================

print("="*70)
print("YOLO OBJECT DETECTION - COMPLETE!")
print("="*70)
print("\nKey Takeaways:")
print("1. YOLO is a fast single-stage detector")
print("   - One forward pass for all detections")
print("   - Grid-based predictions")
print("2. Pre-trained on COCO (80 classes)")
print("   - Ready to use out-of-the-box")
print("   - Can detect common objects")
print("3. Multiple model sizes available")
print("   - Nano (fastest) to Extra Large (most accurate)")
print("   - Trade-off between speed and accuracy")
print("4. Easy-to-use API")
print("   - Simple inference: model(image)")
print("   - Built-in visualization")
print("   - Configurable thresholds")
print("5. Real-time capable")
print(f"   - {fps:.1f} FPS on {device}")
print("   - Suitable for video processing")
print("\nYOLO Advantages:")
print("  ✓ Fast inference (real-time)")
print("  ✓ High accuracy")
print("  ✓ Easy to use")
print("  ✓ Active community")
print("  ✓ Pre-trained models")
print("\nNext Steps:")
print("  - Try on your own images")
print("  - Test different model sizes")
print("  - Experiment with thresholds")
print("  - Move to Example 3 for custom training!")
print("="*70)


if __name__ == "__main__":
    pass
```

## 논의

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

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
Example 2의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_example 2():
        model = Example 2(...)
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
