# 객체 탐지 기초: 좌표·IoU·후처리

객체 탐지 결과는 보통 경계 상자·클래스·점수로 표현한다. 이 예제는 NumPy·PIL로 좌표 변환, IoU, 신뢰도 필터와 클래스별 NMS를 설명한다. 뉴럴 네트워크를 트레인하거나 이미지에서 객체를 실제로 추론하는 코드가 아니다. 상자·점수·클래스는 설명을 위해 직접 지정한 값이다.

## 1. 예제 코드와 출력

??? note "코드"

    ```python
    """
    예제 1: 기본 객체 탐지 개념
    ===========================================

    이 스크립트는 객체 탐지의 기본 개념을 밑바닥부터 구현한다:
    - 경계 상자 표현
    - IoU(Intersection over Union)
    - 비최대 억제(NMS)
    - 신뢰도 임계값 적용
    - 시각화

    복잡한 구조를 쓰기에 앞서 이 개념을 이해하는 것이 매우 중요하다.

    저자: PyTorch Object Detection Tutorial
    날짜: 2025
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from PIL import Image, ImageDraw
    import random

    # 재현성을 위해 난수 시드 고정
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
    # 1단계: 경계 상자 표현
    # ============================================================================
    """
    경계 상자는 객체를 감싸는 직사각형이다.
    [x_최소, y_최소, x_최대, y_최대] 형태를 쓴다

    여기서 각 기호는 다음과 같다.
    - (x_최소, y_최소)는 왼쪽 위 모서리
    - (x_최대, y_최대)는 오른쪽 아래 모서리
    """

    def convert_bbox_format(bbox, from_format='xyxy', to_format='xywh'):
        """
        경계 상자를 다른 형태로 바꾼다.

        형태:
        - 'xyxy': [x_최소, y_최소, x_최대, y_최대]
        - 'xywh': [x_최소, y_최소, 너비, 높이]
        - 'cxcywh': [x_중심, y_중심, 너비, 높이](YOLO 형태)

        인수:
            bbox: 원래 형태의 경계 상자
            from_format: 원래 형태
            to_format: 목표 형태

        반환값:
            목표 형태의 경계 상자
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

        # 목표 형태로 바꾸기
        if to_format == 'xyxy':
            return [x1, y1, x2, y2]
        elif to_format == 'xywh':
            return [x1, y1, x2-x1, y2-y1]
        elif to_format == 'cxcywh':
            return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]


    # 경계 상자 예제
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
    # 2단계: IoU(Intersection over Union)
    # ============================================================================
    """
    IoU는 두 경계 상자가 겹치는 정도를 잰다.
    교집합 넓이와 합집합 넓이의 비이다.

    IoU = 교집합 넓이 / 합집합 넓이
        = 넓이(A ∩ B) / 넓이(A ∪ B)

    IoU는 0(안 겹침)부터 1(완전히 일치)까지이다.
    """

    def calculate_iou(box1, box2):
        """
        두 상자 사이의 IoU를 계산한다.

        인수:
            box1: [x_최소, y_최소, x_최대, y_최대]
            box2: [x_최소, y_최소, x_최대, y_최대]

        반환값:
            iou: 0과 1 사이의 실수
        """
        # 좌표 추출
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        # 교집합 넓이 계산하기
        # 교집합 직사각형의 좌표
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        # 교집합의 너비와 높이 계산하기
        # 안 겹치는 상자를 다루려 max(0, ...) 쓰기
        inter_width = max(0, inter_x_max - inter_x_min)
        inter_height = max(0, inter_y_max - inter_y_min)

        intersection_area = inter_width * inter_height

        # 합집합 넓이 계산하기
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - intersection_area

        # IoU 계산하기
        # 0으로 나누지 않도록 작은 엡실론 더하기
        iou = intersection_area / (union_area + 1e-6)

        return iou


    print("\nStep 2: Intersection over Union (IoU)")
    print("-" * 70)

    # IoU 계산 예제
    box_a = [100, 100, 200, 200]  # 100x100 상자
    box_b = [150, 150, 250, 250]  # 100x100 상자, 일부 겹침

    iou = calculate_iou(box_a, box_b)
    print(f"Box A: {box_a}")
    print(f"Box B: {box_b}")
    print(f"IoU: {iou:.4f}")

    # 추가 예제
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
    # 3단계: 비최대 억제(NMS)
    # ============================================================================
    """
    NMS는 같은 객체에 대한 중복 탐지를 제거한다.

    알고리즘:
    1. 모든 상자를 신뢰도 점수로 정렬(높은 것부터)
    2. 신뢰도가 가장 높은 상자 고르기
    3. 고른 상자와의 IoU가 임계값을 넘는 상자 모두 없애기
    4. 상자가 남지 않을 때까지 반복

    가장 좋은 탐지를 남기고 중복을 제거한다.
    """

    def non_maximum_suppression(boxes, scores, iou_threshold=0.5):
        """
        중복 탐지를 제거하려 비최대 억제를 적용한다.

        인수:
            boxes: 경계 상자의 목록 [[x_최소, y_최소, x_최대, y_최대], ...]
            scores: 상자마다의 신뢰도 점수 목록
            iou_threshold: 억제용 IoU 임계값(기본값 0.5)

        반환값:
            keep_indices: 남길 상자의 인덱스
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

            # 현재 상자 얻기
            current_box = boxes[current_idx]

            # 남은 모든 상자와의 IoU 계산하기
            remaining_indices = sorted_indices[1:]
            remaining_boxes = boxes[remaining_indices]

            ious = np.array([calculate_iou(current_box, box) for box in remaining_boxes])

            # IoU가 임계값 아래인 상자만 남기기
            keep_mask = ious < iou_threshold
            sorted_indices = remaining_indices[keep_mask]

        return keep_indices


    print("Step 3: Non-Maximum Suppression (NMS)")
    print("-" * 70)

    # 예: 같은 객체를 겹쳐 여러 번 탐지
    detections = [
        [100, 100, 200, 200],  # 탐지 1
        [105, 105, 205, 205],  # 탐지 2(1과 비슷)
        [110, 95, 210, 195],   # 탐지 3(1과 비슷)
        [300, 300, 400, 400],  # 탐지 4(다른 객체)
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
    # 4단계: 신뢰도 임계값 적용
    # ============================================================================
    """
    모든 탐지가 믿을 만하지는 않다. 신뢰도가 낮은 탐지는
    NMS를 쓰기 앞서 걸러낸다.

    신뢰도 임계값이 정밀도와 재현율의 트레이드오프를 조절한다:
    - 높은 임계값(예: 0.7): 정밀도 높고 재현율 낮음(객체 일부를 놓친다)
    - 낮은 임계값(예: 0.3): 정밀도 낮고 재현율 높음(거짓 양성이 는다)
    """

    def filter_by_confidence(boxes, scores, classes, conf_threshold=0.5):
        """
        신뢰도 임계값으로 탐지를 거른다.

        인수:
            boxes: 경계 상자의 목록
            scores: 신뢰도 점수의 목록
            classes: 클래스 라벨의 목록
            conf_threshold: 남길 최소 신뢰도

        반환값:
            거른 상자, 점수, 클래스
        """
        keep_mask = np.array(scores) >= conf_threshold

        filtered_boxes = [box for i, box in enumerate(boxes) if keep_mask[i]]
        filtered_scores = [score for i, score in enumerate(scores) if keep_mask[i]]
        filtered_classes = [cls for i, cls in enumerate(classes) if keep_mask[i]]

        return filtered_boxes, filtered_scores, filtered_classes


    print("Step 4: Confidence Thresholding")
    print("-" * 70)

    # 신뢰도가 여러 가지인 탐지 예제
    all_detections = [
        ([100, 100, 200, 200], 0.95, 'dog'),    # 높은 신뢰도
        ([150, 150, 250, 250], 0.75, 'dog'),    # 중간 신뢰도
        ([300, 100, 400, 200], 0.45, 'cat'),    # 낮은 신뢰도
        ([350, 300, 450, 400], 0.25, 'car'),    # 아주 낮은 신뢰도
    ]

    print("All detections:")
    for box, score, cls in all_detections:
        print(f"  {cls}: confidence={score:.2f}, box={box}")

    # 신뢰도 임계값 쓰기
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
    # 5단계: 전체 탐지 파이프라인
    # ============================================================================
    """
    모두 모아 보면:
    1. 모델에서 예측 얻기(상자, 점수, 클래스)
    2. 신뢰도 임계값으로 거르기
    3. 클래스마다 NMS 쓰기
    4. 최종 탐지 반환
    """

    def detection_pipeline(boxes, scores, classes, conf_threshold=0.5, nms_threshold=0.5):
        """
        객체 탐지의 전체 후처리 파이프라인.

        인수:
            boxes: 경계 상자의 목록
            scores: 신뢰도 점수의 목록
            classes: 클래스 라벨의 목록
            conf_threshold: 신뢰도 임계값
            nms_threshold: NMS IoU 임계값

        반환값:
            거르기와 NMS를 거친 최종 상자, 점수, 클래스
        """
        # 1단계: 신뢰도로 거르기
        boxes, scores, classes = filter_by_confidence(boxes, scores, classes, conf_threshold)

        if len(boxes) == 0:
            return [], [], []

        # 2단계: 클래스마다 NMS 쓰기
        # 탐지를 클래스별로 묶기.
        # sorted 를 씌우는 까닭: 집합을 그냥 돌면 순서가 파이썬이 그때그때 잡는
        # 해시에 달려 있어, 같은 시드로 돌려도 아래 찍히는 순서가 바뀐다
        unique_classes = sorted(set(classes))
        final_boxes = []
        final_scores = []
        final_classes = []

        for cls in unique_classes:
            # 이 클래스의 탐지 얻기
            class_mask = [c == cls for c in classes]
            class_boxes = [boxes[i] for i in range(len(boxes)) if class_mask[i]]
            class_scores = [scores[i] for i in range(len(scores)) if class_mask[i]]

            # NMS 쓰기
            keep_indices = non_maximum_suppression(class_boxes, class_scores, nms_threshold)

            # 최종 결과에 더하기
            for idx in keep_indices:
                final_boxes.append(class_boxes[idx])
                final_scores.append(class_scores[idx])
                final_classes.append(cls)

        return final_boxes, final_scores, final_classes


    print("Step 5: Complete Detection Pipeline")
    print("-" * 70)

    # 모델 예측 모사(객체마다 탐지 여럿)
    raw_predictions = [
        ([100, 100, 200, 200], 0.95, 'dog'),
        ([105, 105, 205, 205], 0.88, 'dog'),    # dog의 겹침
        ([110, 95, 210, 195], 0.82, 'dog'),     # 또 다른 겹침
        ([300, 300, 400, 400], 0.90, 'cat'),
        ([305, 305, 405, 405], 0.75, 'cat'),    # cat의 겹침
        ([500, 100, 600, 200], 0.45, 'car'),    # 낮은 신뢰도
        ([700, 300, 800, 400], 0.30, 'person'), # 아주 낮은 신뢰도
    ]

    print(f"Raw model predictions: {len(raw_predictions)} detections")
    for box, score, cls in raw_predictions:
        print(f"  {cls}: {score:.2f}")

    # 파이프라인 실행
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
    # 6단계: 시각화
    # ============================================================================
    """
    탐지를 시각화하는 것은 이해와 디버깅에 매우 중요하다.
    객체가 있는 합성 이미지를 만들고 경계 상자를 그린다.
    """

    def create_synthetic_image(size=(600, 600)):
        """색칠한 사각형을 객체로 삼은 합성 이미지를 만든다."""
        img = Image.new('RGB', size, color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        # "객체" 몇 개 그리기(색칠한 사각형)
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
        이미지 위에 경계 상자를 시각화한다.

        인수:
            image: PIL 이미지 또는 numpy 배열
            boxes: [x_최소, y_최소, x_최대, y_최대]의 목록
            labels: 클래스 라벨의 목록
            scores: 신뢰도 점수의 목록
            title: 그림의 제목
        """
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image)

        # 클래스마다 색 정하기
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

            # 이 클래스의 색 얻기
            color = colors.get(label, 'white')

            # 사각형 그리기
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)

            # 신뢰도와 함께 라벨 더하기
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

    # 합성 이미지 만들기
    image, ground_truth = create_synthetic_image()

    # 탐지 모사(겹친 것과 거짓 양성 포함)
    detections = [
        ([100, 100, 200, 200], 0.95, 'red_square'),
        ([105, 105, 205, 205], 0.85, 'red_square'),  # 겹침
        ([300, 300, 400, 400], 0.92, 'blue_square'),
        ([450, 100, 550, 180], 0.88, 'green_rect'),
        ([200, 400, 280, 500], 0.35, 'red_square'),  # 거짓 양성
    ]

    print("Creating visualization...")
    boxes = [det[0] for det in detections]
    scores = [det[1] for det in detections]
    classes = [det[2] for det in detections]

    # 탐지 파이프라인 실행
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

??? note "전체 출력"

    ```
    ======================================================================
    BASIC OBJECT DETECTION CONCEPTS
    ======================================================================

    This example teaches fundamental detection concepts:
    1. Bounding boxes and coordinate systems
    2. Intersection over Union (IoU)
    3. Non-Maximum Suppression (NMS)
    4. Confidence scores and thresholding

    Step 1: Bounding Box Formats
    ----------------------------------------------------------------------
    Original format (xyxy): [100, 150, 300, 450]
      → Top-left: (100, 150)
      → Bottom-right: (300, 450)

    Converted to xywh: [100, 150, 200, 300]
      → Position: (100, 150)
      → Size: 200 × 300

    Converted to cxcywh (YOLO): [200.0, 300.0, 200, 300]
      → Center: (200.0, 300.0)
      → Size: 200 × 300


    Step 2: Intersection over Union (IoU)
    ----------------------------------------------------------------------
    Box A: [100, 100, 200, 200]
    Box B: [150, 150, 250, 250]
    IoU: 0.1429

    Perfect match IoU: 1.0000 (boxes are identical)
    No overlap IoU: 0.0000 (boxes don't overlap)
    One inside another IoU: 0.3600

    Step 3: Non-Maximum Suppression (NMS)
    ----------------------------------------------------------------------
    Before NMS: 4 detections
      Detection 1: [100, 100, 200, 200], confidence: 0.95
      Detection 2: [105, 105, 205, 205], confidence: 0.88
      Detection 3: [110, 95, 210, 195], confidence: 0.82
      Detection 4: [300, 300, 400, 400], confidence: 0.90

    After NMS: 2 detections kept
      Detection 1: [100, 100, 200, 200], confidence: 0.95
      Detection 4: [300, 300, 400, 400], confidence: 0.90

    Removed 2 duplicate detections

    Step 4: Confidence Thresholding
    ----------------------------------------------------------------------
    All detections:
      dog: confidence=0.95, box=[100, 100, 200, 200]
      dog: confidence=0.75, box=[150, 150, 250, 250]
      cat: confidence=0.45, box=[300, 100, 400, 200]
      car: confidence=0.25, box=[350, 300, 450, 400]

    After confidence threshold (0.5):
      dog: confidence=0.95, box=[100, 100, 200, 200]
      dog: confidence=0.75, box=[150, 150, 250, 250]

    Filtered out 2 low-confidence detections

    Step 5: Complete Detection Pipeline
    ----------------------------------------------------------------------
    Raw model predictions: 7 detections
      dog: 0.95
      dog: 0.88
      dog: 0.82
      cat: 0.90
      cat: 0.75
      car: 0.45
      person: 0.30

    Final detections: 2 objects
      cat: 0.90 at [300, 300, 400, 400]
      dog: 0.95 at [100, 100, 200, 200]

    Pipeline removed 5 detections
      - Confidence filtering: 2 detections
      - NMS: 3 duplicates

    Step 6: Visualization
    ----------------------------------------------------------------------
    Creating visualization...
    Visualization saved as 'basic_detection_results.png'

    ======================================================================
    BASIC OBJECT DETECTION CONCEPTS - COMPLETE!
    ======================================================================

    Key Takeaways:
    1. Bounding Box: 4 numbers define object location
       - Multiple formats: xyxy, xywh, cxcywh
    2. IoU: Measures overlap between boxes (0-1 range)
       - Used for evaluation and NMS
    3. NMS: Removes duplicate detections
       - Keeps highest confidence, removes high IoU overlaps
    4. Confidence: Filters unreliable detections
       - Trade-off between precision and recall
    5. Pipeline: Confidence filter → NMS → Final detections

    Core Metrics:
      - Typical confidence threshold: 0.5
      - Typical NMS IoU threshold: 0.5
      - IoU > 0.5 considered 'good' detection

    You now understand the foundations of object detection!
    Next: Example 2 - Learn YOLO architecture
    ======================================================================
    ```

## 2. 세 좌표 표현의 관계

xyxy는 (왼쪽,위,오른쪽,아래), xywh는 (왼쪽,위,너비,높이), cxcywh는 (중심 x,중심 y,너비,높이)다. 같은 상자라도 숫자의 의미가 다르므로 형식과 단위를 먼저 확인한다. [100,150,300,450]은 xywh로 [100,150,200,300], cxcywh로 [200,300,200,300]이다.

`convert_bbox_format`은 이 세 표현 사이를 변환한다. 여기의 cxcywh는 **픽셀 좌표**이며 YOLO 데이터셋 라벨에 흔히 쓰는 이미지 크기로 정규화한 cxcywh와 단위가 다르다. 이미지 너비 W·높이 H로 중심과 크기를 각각 나누어야 정규화 좌표가 된다.

잘못된 from_format을 주면 현재 코드는 좌표를 0으로 시작하고, 잘못된 to_format은 반환 분기가 없어 None이 된다. 유효 형식·길이 4·유한값과 양의 너비·높이를 검증하는 기능은 없다. 연속 좌표 면적과 PIL의 래스터 사각형 경계 관례도 구분해 읽는다.

## 3. IoU는 겹침 비율이다

IoU=교집합 면적/합집합 면적이다. 예제의 100×100 상자 두 개가 50×50만큼 겹치면 2500/(10000+10000−2500)=1/7이다. 한 상자가 다른 상자 안에 있으면 작은 면적/큰 면적이고 두 상자가 접하기만 해도 IoU는 0이다. 상자가 같으면 수학적으로 1이지만 이 구현은 분모에 ε를 더해 1보다 조금 작은 값을 만들 수 있다.

IoU가 0.5보다 높다는 것만으로 정답 탐지라는 판정을 완성하지 못한다. 평가에는 정답 클래스·일대일 매칭·중복 예측·평가 임계값이 필요하다. 여기에는 그러한 AP 계산이나 데이터셋 평가가 없다. 코드 출력의 ‘good detection’은 설명용 표현으로만 읽는다.

## 4. 신뢰도 필터와 클래스별 NMS

`filter_by_confidence`는 점수가 임계값 **이상**인 후보를 유지한다. 점수는 직접 만든 값이므로 실제 정답 확률에 맞게 보정(calibration)되었다는 의미는 없다. 임계값을 높이면 유지 후보 집합은 줄지만, 매칭과 점수 순위에 따라 정밀도가 항상 증가한다고 보장할 수는 없다. 선택 기준은 검증 데이터와 실제 오류 비용에 맞춘다.

`non_maximum_suppression`은 최고 점수 후보를 남기고 그것과 IoU가 임계값 **미만**인 후보만 다음 반복으로 넘긴다. 따라서 같은 값도 제거한다. 가장 높은 점수는 가장 정확한 상자와 다를 수 있고, 겹친 서로 다른 객체가 억제될 수도 있다. 클래스·이미지별로 적용하는 정책을 정해야 한다.

`detection_pipeline`은 신뢰도 필터 뒤 클래스별 NMS를 수행하며, 필터 후 남은 후보가 없으면 빈 목록 세 개를 반환한다. 결과는 클래스 이름 정렬 순서로 모으므로 전체 점수 내림차순이라고 가정하면 안 된다. 입력 세 목록의 길이·순서가 같아야 한다. 이 함수에는 뉴럴 네트워크 예측·좌표 디코딩·크기 조정의 역변환은 없다.

## 5. 시각화와 코드 범위

합성 이미지의 사각형과 그 주변에 직접 지정한 예측 상자를 그려 후처리 효과를 보여 준다. 저장된 그림이 실제 탐지 성능이나 트레인 수렴을 증명하지 않는다. 이 코드는 트레인 가능한 모델·역전파를 사용하지 않으며 NumPy의 선택 연산에 뉴럴 네트워크의 그래디언트 검사를 적용할 이유도 없다.

대부분의 예제 호출·출력·그림 저장은 파일 최상위에 있고 마지막 `if __name__ == '__main__': pass`가 이를 감싸지 않는다. 따라서 모듈로 import만 해도 예제와 파일 저장이 수행되는 구조다. 위 출력이 전체 결과이므로 결과를 보려고 코드를 다시 실행할 필요는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
xyxy [100,150,300,450]을 다른 두 형식으로 바꾸고 정규화에 추가로 필요한 정보를 설명하라.

</div>

??? success "풀이"

    xywh는 [100,150,200,300], cxcywh는 [200,300,200,300]이다. 정규화에는 이미지 너비 W·높이 H가 필요하다. x·너비는 W, y·높이는 H로 나누며 변환 함수 자체는 이 나눗셈을 하지 않는다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
각 함수의 입력 계약에 포함할 검증 항목을 설명하라.

</div>

??? success "풀이"

    상자는 N×4의 유한한 좌표와 양의 너비·높이, 점수·클래스는 길이 N이며 같은 후보 순서여야 한다. 좌표 형식과 이미지 단위도 일치시킨다. 부동소수점 입력을 무조건 float32로 제한할 필요는 없으며 빈 입력의 반환 계약도 명시한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
같은 IoU 임계값을 평가와 NMS에서 사용할 때 의미가 같은가?

</div>

??? success "풀이"

    평가는 예측·정답의 매칭 품질을 판단하고 NMS는 예측 후보끼리 비교해 중복을 줄인다. 대상과 목적이 다르므로 같은 수치를 써도 같은 판정은 아니다. 신뢰도 임계값은 점수를 비교하는 또 다른 기준이다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 시험 조각이 이 NumPy 예제에 맞지 않는 이유와 적절한 정적 확인 사례를 설명하라.

</div>

??? success "풀이"

    공백이 있는 test_example 1과 Example 1은 유효한 Python 식별자가 아니며 실제로 정의된 모델도 없다. 본문에는 PyTorch 트레인 모델이 없어 .backward() 그래디언트 검사가 맞지 않는다. 동일 상자·비겹침·포함 관계의 IoU, 좌표 변환의 왕복, 빈 후보·동점 점수·임계값 경계·클래스별 격리 등을 명세와 비교하는 것이 알맞다. 아래 코드는 실행 가능한 완성 시험이 아니라 자리 표시자다.

    ```python
    def test_example 1():
        model = Example 1(...)
        # 보통의 입력
        assert model(normal_input).shape == expected_shape
        # 원소가 하나인 배치
        assert model(single_input).shape == (1, ...)
        # 큰 값 (오버플로를 확인한다)
        out = model(torch.ones(...) * 1000)
        assert torch.isfinite(out).all()
        # 그래디언트의 흐름
        out = model(normal_input)
        out.sum().backward()
        for p in model.parameters():
            assert p.grad is not None
    ```


## 정리하며

탐지 후처리에서는 좌표 표현과 단위, 점수 필터와 IoU 기준, 클래스별 선택과 반환 순서를 명확히 해야 한다. 이 합성 예제는 기하·선택 규칙을 설명하며 트레인된 탐지기나 데이터셋 성능 평가를 구현하지 않는다.
