# 보기 1

보기 1: 기본 물체 알아내기 개념. 이 각본은 물체 알아내기의 근본 개념을 맨바닥부터 짠다:

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 물체 알아내기를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 1. 코드

```python
"""
보기 1: 기본 물체 알아내기 개념
===========================================

이 각본은 물체 알아내기의 근본 개념을 맨바닥부터 짠다:
- 두름 상자 나타내기
- 겹침 비(IoU)
- 최대가 아닌 것 누르기(NMS)
- 믿음도 문턱값 두기
- 시각화

복잡한 얼개를 쓰기에 앞서 이 개념을 아는 것이 결정적이다.

지은이: PyTorch Object Detection Tutorial
날짜: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw
import random

# 재현성을 위해 난수 씨앗 고정
np.random.seed(42)
random.seed(42)

print("="*70)
print("BASIC OBJECT DETECTION CONCEPTS")
print("="*70)
print("\nThis example teaches fundamental detection concepts:")
print("1. Bounding boxes and coordinate systems")
print("2. Intersection over Union (IoU)")
print("3. Non-Maximum Suppression (NMS)")
print("4. Confidence scores and thresholding\n")

# ============================================================================
# 1단계: 두름 상자 나타내기
# ============================================================================
"""
두름 상자는 물체를 감싸는 네모이다.
[x_최소, y_최소, x_최대, y_최대] 꼴을 쓴다

여기서 각 기호는 다음과 같다.
- (x_최소, y_최소)는 왼쪽 위 모서리
- (x_최대, y_최대)는 오른쪽 아래 모서리
"""

def convert_bbox_format(bbox, from_format='xyxy', to_format='xywh'):
    """
    두름 상자를 다른 꼴로 바꾼다.
    
    꼴:
    - 'xyxy': [x_최소, y_최소, x_최대, y_최대]
    - 'xywh': [x_최소, y_최소, 너비, 높이]
    - 'cxcywh': [x_가운데, y_가운데, 너비, 높이](YOLO 꼴)
    
    인수:
        bbox: 원래 꼴의 두름 상자
        from_format: 원래 꼴
        to_format: 목표 꼴
    
    반환값:
        목표 꼴의 두름 상자
    """
    x1, y1, x2, y2 = bbox if from_format == 'xyxy' else [0, 0, 0, 0]
    
    if from_format == 'xywh':
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
    elif from_format == 'cxcywh':
        cx, cy, w, h = bbox
        x1 = cx - w/2
        y1 = cy - h/2
        x2 = cx + w/2
        y2 = cy + h/2
    
    # 목표 꼴로 바꾸기
    if to_format == 'xyxy':
        return [x1, y1, x2, y2]
    elif to_format == 'xywh':
        return [x1, y1, x2-x1, y2-y1]
    elif to_format == 'cxcywh':
        return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]


# 두름 상자 보기
print("Step 1: Bounding Box Formats")
print("-" * 70)

example_box_xyxy = [100, 150, 300, 450]  # [x_min, y_min, x_max, y_max]
print(f"Original format (xyxy): {example_box_xyxy}")
print(f"  → Top-left: ({example_box_xyxy[0]}, {example_box_xyxy[1]})")
print(f"  → Bottom-right: ({example_box_xyxy[2]}, {example_box_xyxy[3]})")

example_box_xywh = convert_bbox_format(example_box_xyxy, 'xyxy', 'xywh')
print(f"\nConverted to xywh: {example_box_xywh}")
print(f"  → Position: ({example_box_xywh[0]}, {example_box_xywh[1]})")
print(f"  → Size: {example_box_xywh[2]} × {example_box_xywh[3]}")

example_box_cxcywh = convert_bbox_format(example_box_xyxy, 'xyxy', 'cxcywh')
print(f"\nConverted to cxcywh (YOLO): {example_box_cxcywh}")
print(f"  → Center: ({example_box_cxcywh[0]}, {example_box_cxcywh[1]})")
print(f"  → Size: {example_box_cxcywh[2]} × {example_box_cxcywh[3]}\n")

# ============================================================================
# 2단계: 겹침 비(IoU)
# ============================================================================
"""
겹침 비는 두 두름 상자 사이의 겹침을 잰다.
교집합 넓이와 합집합 넓이의 비이다.

겹침 비 = 겹친 넓이 / 합집합 넓이
    = 넓이(A ∩ B) / 넓이(A ∪ B)

겹침 비는 0(안 겹침)부터 1(완전히 맞음)까지이다.
"""

def calculate_iou(box1, box2):
    """
    두 상자 사이의 겹침 비를 셈한다.
    
    인수:
        box1: [x_최소, y_최소, x_최대, y_최대]
        box2: [x_최소, y_최소, x_최대, y_최대]
    
    반환값:
        iou: 0과 1 사이의 실수
    """
    # 자리표 뽑아내기
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 교집합 넓이 셈하기
    # 교집합 네모의 자리표
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # 교집합의 너비와 높이 셈하기
    # 안 겹치는 상자를 다루려 max(0, ...) 쓰기
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    
    intersection_area = inter_width * inter_height
    
    # 합집합 넓이 셈하기
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - intersection_area
    
    # 겹침 비 셈하기
    # 0으로 나누지 않도록 작은 엡실론 더하기
    iou = intersection_area / (union_area + 1e-6)
    
    return iou


print("\nStep 2: Intersection over Union (IoU)")
print("-" * 70)

# 겹침 비 셈하기 보기
box_a = [100, 100, 200, 200]  # 100x100 상자
box_b = [150, 150, 250, 250]  # 100x100 상자, 일부 겹침

iou = calculate_iou(box_a, box_b)
print(f"Box A: {box_a}")
print(f"Box B: {box_b}")
print(f"IoU: {iou:.4f}")

# 더 많은 보기
box_c = [100, 100, 200, 200]  # box_a와 같음
iou_perfect = calculate_iou(box_a, box_c)
print(f"\nPerfect match IoU: {iou_perfect:.4f} (boxes are identical)")

box_d = [300, 300, 400, 400]  # 겹치지 않음
iou_zero = calculate_iou(box_a, box_d)
print(f"No overlap IoU: {iou_zero:.4f} (boxes don't overlap)")

box_e = [120, 120, 180, 180]  # box_a 안의 box_e
iou_inside = calculate_iou(box_a, box_e)
print(f"One inside another IoU: {iou_inside:.4f}\n")

# ============================================================================
# 3단계: 최대가 아닌 것 누르기(NMS)
# ============================================================================
"""
NMS는 같은 물체를 거듭 알아낸 것을 없앤다.

알고리즘:
1. 모든 상자를 믿음도 점수로 정렬(높은 것부터)
2. 믿음도가 가장 높은 상자 고르기
3. 고른 상자와의 겹침 비가 문턱값을 넘는 상자 모두 없애기
4. 상자가 남지 않을 때까지 되풀이

가장 좋은 알아냄을 남기고 겹친 것을 없앤다.
"""

def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
    """
    거듭 알아낸 것을 없애려 최대가 아닌 것 누르기를 쓴다.
    
    인수:
        boxes: 두름 상자의 목록 [[x_최소, y_최소, x_최대, y_최대], ...]
        scores: 상자마다의 믿음도 점수 목록
        iou_threshold: 누르기용 겹침 비 문턱값(붙박이 0.5)
    
    반환값:
        keep_indices: 남길 상자의 번호
    """
    # numpy 배열로 바꾸기
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # 상자를 점수로 정렬(높은 것부터)
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # 점수가 가장 높은 상자 고르기
        current_idx = sorted_indices[0]
        keep_indices.append(current_idx)
        
        if len(sorted_indices) == 1:
            break
        
        # 지금 상자 얻기
        current_box = boxes[current_idx]
        
        # 남은 모든 상자와의 겹침 비 셈하기
        remaining_indices = sorted_indices[1:]
        remaining_boxes = boxes[remaining_indices]
        
        ious = np.array([calculate_iou(current_box, box) for box in remaining_boxes])
        
        # 겹침 비가 문턱값 아래인 상자만 남기기
        keep_mask = ious < iou_threshold
        sorted_indices = remaining_indices[keep_mask]
    
    return keep_indices


print("Step 3: Non-Maximum Suppression (NMS)")
print("-" * 70)

# 보기: 같은 물체를 겹쳐 여러 번 알아냄
detections = [
    [100, 100, 200, 200],  # 알아냄 1
    [105, 105, 205, 205],  # 알아냄 2(1과 비슷)
    [110, 95, 210, 195],   # 알아냄 3(1과 비슷)
    [300, 300, 400, 400],  # 알아냄 4(다른 물체)
]

confidence_scores = [0.95, 0.88, 0.82, 0.90]

print(f"Before NMS: {len(detections)} detections")
for i, (box, score) in enumerate(zip(detections, confidence_scores)):
    print(f"  Detection {i+1}: {box}, confidence: {score:.2f}")

# NMS 쓰기
keep_indices = non_maximum_suppression(detections, confidence_scores, iou_threshold=0.5)

print(f"\nAfter NMS: {len(keep_indices)} detections kept")
for idx in keep_indices:
    print(f"  Detection {idx+1}: {detections[idx]}, confidence: {confidence_scores[idx]:.2f}")

print(f"\nRemoved {len(detections) - len(keep_indices)} duplicate detections\n")

# ============================================================================
# 4단계: 믿음도 문턱값 두기
# ============================================================================
"""
모든 알아냄이 믿을 만하지는 않다. 믿음도가 낮은 알아냄은
NMS를 쓰기 앞서 걸러낸다.

믿음도 문턱값이 정밀도와 재현율의 맞바꿈을 다스린다:
- 높은 문턱값(보기로 0.7): 정밀도 높고 재현율 낮음(물체 일부를 놓친다)
- 낮은 문턱값(보기로 0.3): 정밀도 낮고 재현율 높음(헛양성이 는다)
"""

def filter_by_confidence(boxes, scores, classes, conf_threshold=0.5):
    """
    믿음도 문턱값으로 알아냄을 거른다.
    
    인수:
        boxes: 두름 상자의 목록
        scores: 믿음도 점수의 목록
        classes: 갈래 이름표의 목록
        conf_threshold: 남길 최소 믿음도
    
    반환값:
        거른 상자, 점수, 갈래
    """
    keep_mask = np.array(scores) >= conf_threshold
    
    filtered_boxes = [box for i, box in enumerate(boxes) if keep_mask[i]]
    filtered_scores = [score for i, score in enumerate(scores) if keep_mask[i]]
    filtered_classes = [cls for i, cls in enumerate(classes) if keep_mask[i]]
    
    return filtered_boxes, filtered_scores, filtered_classes


print("Step 4: Confidence Thresholding")
print("-" * 70)

# 믿음도가 여러 가지인 알아냄 보기
all_detections = [
    ([100, 100, 200, 200], 0.95, 'dog'),    # 높은 믿음도
    ([150, 150, 250, 250], 0.75, 'dog'),    # 가운데 믿음도
    ([300, 100, 400, 200], 0.45, 'cat'),    # 낮은 믿음도
    ([350, 300, 450, 400], 0.25, 'car'),    # 아주 낮은 믿음도
]

print("All detections:")
for box, score, cls in all_detections:
    print(f"  {cls}: confidence={score:.2f}, box={box}")

# 믿음도 문턱값 쓰기
conf_threshold = 0.5
boxes = [det[0] for det in all_detections]
scores = [det[1] for det in all_detections]
classes = [det[2] for det in all_detections]

filtered_boxes, filtered_scores, filtered_classes = filter_by_confidence(
    boxes, scores, classes, conf_threshold
)

print(f"\nAfter confidence threshold ({conf_threshold}):")
for box, score, cls in zip(filtered_boxes, filtered_scores, filtered_classes):
    print(f"  {cls}: confidence={score:.2f}, box={box}")

print(f"\nFiltered out {len(all_detections) - len(filtered_boxes)} low-confidence detections\n")

# ============================================================================
# 5단계: 온전한 알아내기 물길
# ============================================================================
"""
모두 모아 보면:
1. 모델에서 어림 얻기(상자, 점수, 갈래)
2. 믿음도 문턱값으로 거르기
3. 갈래마다 NMS 쓰기
4. 마지막 알아냄 돌려주기
"""

def detection_pipeline(boxes, scores, classes, conf_threshold=0.5, nms_threshold=0.5):
    """
    온전한 물체 알아내기 뒷손질 물길.
    
    인수:
        boxes: 두름 상자의 목록
        scores: 믿음도 점수의 목록
        classes: 갈래 이름표의 목록
        conf_threshold: 믿음도 문턱값
        nms_threshold: NMS 겹침 비 문턱값
    
    반환값:
        거르기와 NMS를 거친 마지막 상자, 점수, 갈래
    """
    # 1단계: 믿음도로 거르기
    boxes, scores, classes = filter_by_confidence(boxes, scores, classes, conf_threshold)
    
    if len(boxes) == 0:
        return [], [], []
    
    # 2단계: 갈래마다 NMS 쓰기
    # 알아냄을 갈래별로 묶기
    unique_classes = set(classes)
    final_boxes = []
    final_scores = []
    final_classes = []
    
    for cls in unique_classes:
        # 이 갈래의 알아냄 얻기
        class_mask = [c == cls for c in classes]
        class_boxes = [boxes[i] for i in range(len(boxes)) if class_mask[i]]
        class_scores = [scores[i] for i in range(len(scores)) if class_mask[i]]
        
        # NMS 쓰기
        keep_indices = non_maximum_suppression(class_boxes, class_scores, nms_threshold)
        
        # 마지막 결과에 더하기
        for idx in keep_indices:
            final_boxes.append(class_boxes[idx])
            final_scores.append(class_scores[idx])
            final_classes.append(cls)
    
    return final_boxes, final_scores, final_classes


print("Step 5: Complete Detection Pipeline")
print("-" * 70)

# 모델 어림 흉내내기(물체마다 알아냄 여럿)
raw_predictions = [
    ([100, 100, 200, 200], 0.95, 'dog'),
    ([105, 105, 205, 205], 0.88, 'dog'),    # dog의 겹침
    ([110, 95, 210, 195], 0.82, 'dog'),     # 또 다른 겹침
    ([300, 300, 400, 400], 0.90, 'cat'),
    ([305, 305, 405, 405], 0.75, 'cat'),    # cat의 겹침
    ([500, 100, 600, 200], 0.45, 'car'),    # 낮은 믿음도
    ([700, 300, 800, 400], 0.30, 'person'), # 아주 낮은 믿음도
]

print(f"Raw model predictions: {len(raw_predictions)} detections")
for box, score, cls in raw_predictions:
    print(f"  {cls}: {score:.2f}")

# 물길 돌리기
boxes = [pred[0] for pred in raw_predictions]
scores = [pred[1] for pred in raw_predictions]
classes = [pred[2] for pred in raw_predictions]

final_boxes, final_scores, final_classes = detection_pipeline(
    boxes, scores, classes,
    conf_threshold=0.5,
    nms_threshold=0.5
)

print(f"\nFinal detections: {len(final_boxes)} objects")
for box, score, cls in zip(final_boxes, final_scores, final_classes):
    print(f"  {cls}: {score:.2f} at {box}")

print(f"\nPipeline removed {len(raw_predictions) - len(final_boxes)} detections")
print(f"  - Confidence filtering: {sum(1 for s in scores if s < 0.5)} detections")
print(f"  - NMS: {len(raw_predictions) - sum(1 for s in scores if s < 0.5) - len(final_boxes)} duplicates\n")

# ============================================================================
# 6단계: 그려 보기
# ============================================================================
"""
알아냄을 그려 보는 것은 이해와 벌레잡기에 결정적이다.
물체가 있는 인공 그림을 만들고 두름 상자를 그린다.
"""

def create_synthetic_image(size=(600, 600)):
    """빛깔 있는 네모를 물체로 삼은 인공 그림을 만든다."""
    img = Image.new('RGB', size, color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # "물체" 몇 개 그리기(빛깔 있는 네모)
    objects = [
        ([100, 100, 200, 200], (255, 100, 100), 'red_square'),
        ([300, 300, 400, 400], (100, 100, 255), 'blue_square'),
        ([450, 100, 550, 180], (100, 255, 100), 'green_rect'),
    ]
    
    for box, color, _ in objects:
        draw.rectangle(box, fill=color, outline=(0, 0, 0), width=2)
    
    return img, objects


def visualize_detections(image, boxes, labels, scores, title="Object Detections"):
    """
    그림 위에 두름 상자를 그려 본다.
    
    인수:
        image: PIL 그림 또는 numpy 배열
        boxes: [x_최소, y_최소, x_최대, y_최대]의 목록
        labels: 갈래 이름표의 목록
        scores: 믿음도 점수의 목록
        title: 그림의 제목
    """
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)
    
    # 갈래마다 빛깔 정하기
    colors = {
        'red_square': 'red',
        'blue_square': 'blue',
        'green_rect': 'green',
        'dog': 'yellow',
        'cat': 'cyan',
        'car': 'magenta',
    }
    
    for box, label, score in zip(boxes, labels, scores):
        x_min, y_min, x_max, y_max = box
        width = x_max - x_min
        height = y_max - y_min
        
        # 이 갈래의 빛깔 얻기
        color = colors.get(label, 'white')
        
        # 네모 그리기
        rect = patches.Rectangle(
            (x_min, y_min), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # 믿음도와 함께 이름표 더하기
        label_text = f'{label}: {score:.2f}'
        ax.text(
            x_min, y_min - 5,
            label_text,
            color='white',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.7)
        )
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('basic_detection_results.png', dpi=150, bbox_inches='tight')
    print(f"Visualization saved as 'basic_detection_results.png'")
    plt.close()


print("Step 6: Visualization")
print("-" * 70)

# 인공 그림 만들기
image, ground_truth = create_synthetic_image()

# 알아냄 흉내내기(겹친 것과 헛양성 포함)
detections = [
    ([100, 100, 200, 200], 0.95, 'red_square'),
    ([105, 105, 205, 205], 0.85, 'red_square'),  # 겹침
    ([300, 300, 400, 400], 0.92, 'blue_square'),
    ([450, 100, 550, 180], 0.88, 'green_rect'),
    ([200, 400, 280, 500], 0.35, 'red_square'),  # 헛양성
]

print("Creating visualization...")
boxes = [det[0] for det in detections]
scores = [det[1] for det in detections]
classes = [det[2] for det in detections]

# 알아내기 물길 돌리기
final_boxes, final_scores, final_classes = detection_pipeline(
    boxes, scores, classes,
    conf_threshold=0.5,
    nms_threshold=0.5
)

visualize_detections(image, final_boxes, final_classes, final_scores,
                    title="Object Detection Results (After NMS)")

# ============================================================================
# 요약
# ============================================================================

print("\n" + "="*70)
print("BASIC OBJECT DETECTION CONCEPTS - COMPLETE!")
print("="*70)
print("\nKey Takeaways:")
print("1. Bounding Box: 4 numbers define object location")
print("   - Multiple formats: xyxy, xywh, cxcywh")
print("2. IoU: Measures overlap between boxes (0-1 range)")
print("   - Used for evaluation and NMS")
print("3. NMS: Removes duplicate detections")
print("   - Keeps highest confidence, removes high IoU overlaps")
print("4. Confidence: Filters unreliable detections")
print("   - Trade-off between precision and recall")
print("5. Pipeline: Confidence filter → NMS → Final detections")
print("\nCore Metrics:")
print(f"  - Typical confidence threshold: 0.5")
print(f"  - Typical NMS IoU threshold: 0.5")
print(f"  - IoU > 0.5 considered 'good' detection")
print("\nYou now understand the foundations of object detection!")
print("Next: Example 2 - Learn YOLO architecture")
print("="*70)


if __name__ == "__main__":
    pass
```

## 2. 논의

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
Example 1의 짜기를 확인하는 두루 살피는 시험 함수를 쓰라. 빈 들임, 원소가 하나인 들임, 아주 큰 들임, 값이 극단인 들임(0, 아주 큰 수)을 비롯한 모서리 경우를 시험하여라.

??? success "연습문제 4 풀이"
    경계 조건을 두루 시험하는 함수를 만든다.
    ```python
    def test_example 1():
        model = Example 1(...)
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

## 정리하며

**다룬 것** — 보기 1

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다.

앞의 연습문제 4개로 스스로 따져 볼 수 있다.
