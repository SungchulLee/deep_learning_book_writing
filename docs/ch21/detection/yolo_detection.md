# 사전 학습된 YOLO의 추론 결과 읽기

사전 학습된 YOLOv8n을 불러 합성 이미지에서 추론하고 상자·점수·클래스를 읽는 예제다. 모델 내부를 새로 구현하거나 학습하지 않는다. 합성 도형의 레이블은 COCO 객체 클래스와 다르므로 이 자료만으로 일반 객체 탐지 정확도를 평가할 수 없다.

## 1. 추론·시각화 코드와 기존 출력

??? note "코드"

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
    # 8단계: 배치 단위 다루기
    # ============================================================================
    """
    YOLO는 그림 여럿을 배치로 효율적으로 다룰 수 있다.
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

**기존 출력:**

```
⚠ ultralytics not installed. Installing...
Run: pip install ultralytics
======================================================================
YOLO OBJECT DETECTION
======================================================================

This example demonstrates:
1. Loading pre-trained YOLOv8 models
2. Running object detection on images
3. Understanding YOLO outputs
4. Comparing different model sizes
5. Real-time performance analysis

Using device: cpu
⚠ GPU not available. Inference will be slower.

Step 1: Creating Test Images
----------------------------------------------------------------------
Created test image 1 with 3 objects
Created test image 2 with 3 objects
Created test image 3 with 3 objects

Skipping YOLO demo - ultralytics not installed
Install with: pip install ultralytics
```

## 2. 입력 이미지와 클래스의 의미

각 이미지에는 빨간 사각형·파란 타원·초록 사각형을 그린다. 그림을 그릴 때 기록한 red_box·blue_circle·green_box는 COCO 가중치가 학습한 클래스 이름이 아니다. 객체가 탐지되지 않아도 이 모델이 그 도형 클래스를 학습하지 못했다는 실험으로 해석하지 않는다. 반대로 도형을 다른 COCO 객체로 탐지한 수를 정확도로 삼을 수도 없다.

`create_test_image`의 좌표 난수 범위는 너비·높이가 충분히 큰 이미지를 가정한다. 임의의 작은 크기를 넣으면 난수 범위가 유효하지 않을 수 있다. 입력은 이 예제에서 RGB PIL 이미지이며 NumPy·Tensor 입력의 채널 순서·범위는 별도의 API 계약을 따른다. torchvision 입력과 모든 영상 라이브러리의 BGR/RGB 관례가 같다고 가정하지 않는다.

## 3. Results와 Boxes 객체

한 이미지의 추론 결과도 Results의 목록으로 반환되므로 먼저 results[0]을 선택한다. [Results API](https://docs.ultralytics.com/reference/engine/results/)에서 boxes.xyxy는 원본 이미지 기준 모서리 좌표, boxes.xywh는 **중심 x,y·너비·높이**다. 이 xywh를 왼쪽 위 x,y·너비·높이로 사용하는 다른 함수와 혼동하지 않는다. 정규화 표현 xyxyn·xywhn과도 구분한다.

점수·클래스는 result.conf·result.cls가 아니라 result.boxes.conf·result.boxes.cls에 있다. names는 클래스 ID에서 이름으로 가는 매핑이다. 일반 탐지의 boxes.data는 N×6이지만 tracking ID가 포함되는 경로 등에서는 열 구성이 달라질 수 있으므로 모든 Results의 data를 고정된 여섯 열로 취급하지 않는다. 탐지가 없으면 0개 상자를 정상적으로 처리한다.

코드의 ‘persons, cars, and dogs’ 메시지와 `[0,2,15]`는 일치하지 않는다. [COCO 클래스 매핑](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml)에서 15는 cat이고 dog는 16이다. 클래스 번호의 의미는 모델의 names를 기준으로 읽는다. 점수는 예측 신뢰도이며 실제 정답일 빈도를 자동으로 보장하는 보정된 확률은 아니다.

## 4. 신뢰도·NMS와 시각화

conf는 낮은 점수 후보를 거르는 기준이고 iou는 NMS에서 겹친 후보를 정리하는 기준이다. 높은 conf는 후보를 줄이는 방향이며 낮은 NMS IoU는 더 작은 겹침에서도 억제하도록 한다. 둘을 동시에 바꾼 결과를 신뢰도 하나의 효과라고 해석하면 안 된다. 기본값은 설치 버전에 따라 확인하고, 코드 주석의 값을 모든 버전의 기본값으로 외우지 않는다.

그림은 선택된 예측의 위치·클래스·점수를 보여 준다. 데이터셋 정답과 일대일 매칭한 TP·FP·FN이나 AP는 계산하지 않는다. 도형을 그릴 때 기록한 objects 목록을 성능 평가에 사용하지도 않는다. 이 예제의 시각화·탐지 개수는 결과 구조를 이해하는 자료다.

## 5. 모델 크기와 시간 측정의 한계

모델 크기 비교 부분의 `model_sizes`에는 yolov8n.pt 하나만 있다. 다른 크기의 결과나 크기별 정확도 순위를 실제로 비교한 실험은 아니다. 검출 상자 수가 많다는 것도 정확도가 높다는 뜻이 아니다. 크기별 비교에는 동일 평가 데이터와 AP 등의 지표가 필요하다.

최초 호출 시간에는 초기 준비·전처리·후처리 등이 섞일 수 있다. 반복 부분은 한 차례 warm-up 뒤 다섯 번 또는 뒤의 열 번을 측정하지만 time.time의 벽시계 측정이며 GPU 비동기 실행을 직접 동기화하지 않는다. API 내부의 동기화 여부까지 확인하지 않고 순수 GPU 커널 시간이라고 부를 수 없다. 평균±표준편차는 이 반복값의 요약이지 신뢰구간이 아니다.

세 이미지 배치의 전체 시간/3은 해당 배치의 이미지당 평균 비용이다. 한 이미지가 도착해서 결과를 받기까지의 지연시간과 같지 않다. 1000/ms를 FPS, 다시 3600을 곱한 값을 시간당 처리량으로 출력하지만 실제 영상의 읽기·대기·전송·표시와 장시간 변동까지 포함한 처리량 보장은 아니다. CUDA가 없으면 CPU를 고르며 Apple MPS 등 다른 장치를 자동 선택하지 않는다.

## 6. 예제 파일의 실행 범위

설치되지 않았을 때 ‘Installing’ 메시지를 내지만 실제 설치 명령은 실행하지 않는다. 가중치 로드에는 다운로드가 필요할 수 있다. 예제 호출과 그림 저장은 최상위에 있어 import만 해도 수행되고, 마지막 main 조건문의 pass가 이를 막지 않는다. 읽기·검토 작업에서는 기존 출력과 코드를 설명하면 충분하며 재실행할 필요는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
합성 도형에서 탐지 개수가 많으면 더 정확한 모델인가?

</div>

??? success "연습문제 1 풀이"

    아니다. 도형 이름은 COCO 클래스와 다르고 정답 매칭·오탐·미탐 지표를 계산하지 않는다. model_sizes도 nano 하나뿐이므로 크기별 정확도 실험이 아니다. 평가 데이터와 클래스 정의를 맞춰 AP 등을 비교해야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
boxes.xywh의 앞 두 값과 클래스 [0,2,15]의 의미를 설명하라.

</div>

??? success "연습문제 2 풀이"

    xywh의 앞 두 값은 중심 좌표다. COCO 기본 가중치에서 [0,2,15]는 person·car·cat이며 dog는 16이다. 모델이 달라지면 names를 기준으로 확인한다. 점수·클래스·상자 배열의 행 순서도 같아야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
처음 한 번의 추론 시간과 세 이미지 배치의 시간/3을 단일 이미지 지연시간으로 비교할 때 문제를 설명하라.

</div>

??? success "연습문제 3 풀이"

    첫 호출에는 초기화 비용이 있고 배치 시간/3은 배치 처리의 이미지당 비용이다. 서로 다른 측정 대상이다. warm-up·반복 수·GPU 동기화·입력 준비·NMS 포함 여부와 장치를 맞추고 지연시간과 처리량을 따로 보고한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 기존 시험 조각이 이 추론 API에 적절한지 검토하라.

</div>

??? success "연습문제 4 풀이"

    공백 식별자·미정의 Example 2 때문에 실행 가능한 시험이 아니다. YOLO의 고수준 추론 결과는 고정 모양의 학습 텐서가 아니며 여기에 out.sum().backward()를 바로 적용할 수 없다. 빈 탐지 결과·좌표 형식·클래스 매핑·입력 수와 Results 수·여러 이미지의 대응을 확인하는 명세가 적절하다. 아래 조각은 자리 표시자로 읽는다.

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


## 정리하며

사전 학습 모델의 Results를 읽을 때 좌표 표현·클래스 매핑·점수의 의미를 확인한다. 이 합성 예제는 API를 설명하며 탐지 개수나 짧은 반복 시간으로 정확도·실시간 처리 능력을 입증하지 않는다.
