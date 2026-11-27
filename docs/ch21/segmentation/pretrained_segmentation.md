# 사전 트레인 모델을 이용한 이미지 분할

이 예제는 ImageNet 사전 트레인 ResNet-50 인코더를 사용하는 DeepLabV3+를 PASCAL VOC2012 시맨틱 분할에 파인튜닝한다. 사전 트레인된 분할 모델을 그대로 평가하는 것과 달리 디코더와 분할 헤드를 새 과제에서 트레인한다. 데이터 전처리·무시 라벨·차등 러닝레이트와 지표의 집계 방식을 함께 확인한다.

## 1. 코드

??? note "코드 (617줄)"

    ```python
    """
    보기 2: 사전 트레인된 인코더로 하는 분할 전이학습
    ========================================================================

    이 스크립트는 사전 트레인된 인코더(ResNet, VGG 등)를
    시맨틱 분할에 쓰는 법을 보여 준다. segmentation_models_pytorch 라이브러리를 쓰는데
    이는 ImageNet에서 사전 트레인된 백본을 갖춘 여러 구조를 제공한다.

    핵심 개념:
    - 분할을 위한 전이학습
    - 사전 트레인된 인코더(ResNet, EfficientNet)
    - DeepLabV3+ 구조
    - 다중 클래스 분류(클래스 21개)
    - 실제 데이터셋 다루기
    - 고급 평가 지표(평균 IoU)

    지은이: PyTorch Semantic Segmentation Tutorial
    날짜: 2025
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
    from torchvision import datasets
    import segmentation_models_pytorch as smp
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from tqdm import tqdm

    # 난수 시드를 설정한다
    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================================
    # 1단계: 장치 설정
    # ============================================================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    if not torch.cuda.is_available():
        print("⚠ WARNING: CUDA not available. Training will be slow.")
        print("Consider using Google Colab or a GPU instance.\n")

    # ============================================================================
    # 2단계: 데이터셋 갖추기
    # ============================================================================
    """
    분할의 표준 벤치마크인 PASCAL VOC 2012 데이터셋을 쓴다.

    PASCAL VOC에는 다음이 들어 있다:
    - 트레인 이미지 1,464장
    - 검증 이미지 1,449장
    - 클래스 21개(물체 클래스 20개 + 배경)

    클래스: background, aeroplane, bicycle, bird, boat, bottle, bus, car,
             고양이, 의자, 소, 식탁, 개, 말, 오토바이, 사람,
             화분, 양, 소파, 기차, 텔레비전과 모니터
    """

    # 시각화용 VOC 컬러맵(클래스마다의 RGB 값)
    VOC_COLORMAP = [
        [0, 0, 0],        # 배경
        [128, 0, 0],      # aeroplane
        [0, 128, 0],      # bicycle
        [128, 128, 0],    # bird
        [0, 0, 128],      # boat
        [128, 0, 128],    # bottle
        [0, 128, 128],    # bus
        [128, 128, 128],  # car
        [64, 0, 0],       # cat
        [192, 0, 0],      # chair
        [64, 128, 0],     # cow
        [192, 128, 0],    # dining table
        [64, 0, 128],     # dog
        [192, 0, 128],    # horse
        [64, 128, 128],   # motorbike
        [192, 128, 128],  # person
        [0, 64, 0],       # potted plant
        [128, 64, 0],     # sheep
        [0, 192, 0],      # sofa
        [128, 192, 0],    # train
        [0, 64, 128],     # tv/monitor
    ]

    print("Preparing PASCAL VOC 2012 dataset...")
    print("(First run will download ~2GB of data)\n")

    # 이미지 변환
    # 분할에서는 이미지와 마스크를 함께 변환해야 한다
    IMG_SIZE = 512  # DeepLab은 512×512에서 잘 된다

    # 변환 정의
    image_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 통계량
                            std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor()
    ])


    class VOCSegmentation(datasets.VOCSegmentation):
        """
        이미지와 마스크 모두에 변환을 적용하는 사용자 정의 VOC 데이터셋 클래스.
        """
        def __getitem__(self, index):
            img, mask = super().__getitem__(index)

            # 이미지 변환 적용
            img = image_transform(img)

            # 마스크 변환 적용
            mask = mask_transform(mask)
            mask = mask.squeeze(0).long()  # 채널 차원을 없애고 long으로 바꾸기

            # VOC는 255를 무시 번호로 쓰며 그대로 둔다
            return img, mask


    # 데이터셋 불러오기
    train_dataset = VOCSegmentation(
        root='./data',
        year='2012',
        image_set='train',
        download=True
    )

    val_dataset = VOCSegmentation(
        root='./data',
        year='2012',
        image_set='val',
        download=True
    )

    # 데이터 로더 생성
    BATCH_SIZE = 4  # 분할은 메모리를 많이 쓴다

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Dataset loaded:")
    print(f"  Training images: {len(train_dataset)}")
    print(f"  Validation images: {len(val_dataset)}")
    print(f"  Number of classes: 21")
    print(f"  Image size: {IMG_SIZE}×{IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}\n")

    # ============================================================================
    # 3단계: 사전 트레인된 인코더로 모델 고르기
    # ============================================================================
    """
    segmentation_models_pytorch(smp)는 사전 트레인된 인코더를 갖춘 여러 구조를 제공한다.
    여기서는 ResNet-50 인코더를 쓴 DeepLabV3+를 쓴다.

    구조 선택:
    - Unet: 스킵 연결을 갖춘 고전 U-넷
    - UnetPlusPlus: 중첩된 스킵 연결을 갖춘 개선된 U-넷
    - DeepLabV3+: 아트러스(팽창) 컨볼루션과 ASPP를 쓴 최신 구조
    - FPN: 피처 피라미드 네트워크
    - PSPNet: 피라미드 장면 파싱 네트워크

    인코더 선택:
    - resnet18, resnet34, resnet50, resnet101
    - efficientnet-b0 to b7
    - mobilenet_v2
    - vgg16, vgg19
    """

    print("Creating DeepLabV3+ model with pre-trained ResNet-50 encoder...")

    # 사전 트레인된 인코더로 모델 만들기
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",           # 인코더 고르기(resnet50, efficientnet-b0 등)
        encoder_weights="imagenet",         # ImageNet에서 사전 트레인된 웨이트 쓰기
        in_channels=3,                      # 입력 채널(RGB)
        classes=21,                         # 출력 클래스(VOC는 21개)
    )

    model = model.to(device)

    # 모델 요약
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: DeepLabV3+ with ResNet-50")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nModel Architecture:")
    print("  Encoder: ResNet-50 (pre-trained on ImageNet)")
    print("  ASPP: Atrous Spatial Pyramid Pooling")
    print("  Decoder: Upsampling + refinement")
    print("  Output: 21-class segmentation")

    # ============================================================================
    # 4단계: 로스 함수
    # ============================================================================
    """
    다중 클래스 분류에는 다음을 갖춘 CrossEntropyLoss를 쓴다:
    - ignore_index=255: VOC는 경계/불확실 픽셀에 255를 쓴다
    - 불균형한 클래스를 위한 선택적 클래스 웨이트

    다른 로스:
    - 다이스 로스: 불균형한 클래스에 더 낫다
    - 포컬 로스: 어려운 예에 초점을 둔다
    - 결합: 교차 엔트로피 + 다이스
    """

    # 다중 클래스 분류를 위한 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # VOC는 경계에 255를 쓴다

    # 대안: smp의 다이스 로스 쓰기
    # criterion = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)

    # 또는 결합 로스
    # criterion = smp.losses.DiceLoss(mode='multiclass') + nn.CrossEntropyLoss()

    print("\nLoss function: CrossEntropyLoss (ignore boundary pixels)")

    # ============================================================================
    # 5단계: 레이어마다 러닝레이트가 다른 옵티마이저
    # ============================================================================
    """
    전이학습에는 러닝레이트를 달리 쓴다:
    - 인코더(사전 트레인): 작은 러닝레이트(0.0001) — 조심스레 파인튜닝한다
    - 디코더(무작위 초기화): 큰 러닝레이트(0.001) — 더 많이 배워야 한다

    이를 "레이어마다 다른 러닝레이트(discriminative learning rate)"라 하며
    분할의 전이학습을 잘하는 데 결정적이다.
    """

    # 인코더와 디코더의 파라메터 나누기
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    # 러닝레이트가 다른 옵티마이저 만들기
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': 0.0001},   # 사전 트레인된 인코더에는 작은 러닝레이트
        {'params': decoder_params, 'lr': 0.001},    # 디코더에는 더 큰 러닝레이트
    ])

    # 러닝레이트 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',      # 평균 IoU를 가장 크게
        factor=0.5,
        patience=3,
        verbose=True
    )

    print("\nOptimizer: Adam with discriminative learning rates")
    print(f"  Encoder LR: 0.0001 (pre-trained, fine-tune)")
    print(f"  Decoder LR: 0.001 (random init, learn)")

    # ============================================================================
    # 6단계: 평가 지표
    # ============================================================================
    """
    다중 클래스 분류에는 다음을 쓴다:
    1. mIoU(평균 IoU): 모든 클래스에 걸친 IoU의 평균
    2. 클래스별 IoU: 클래스마다의 IoU
    3. 픽셀 정확도: 전체 픽셀 분류 정확도

    mIoU는 분할 벤치마크의 표준 지표다.
    """

    def calculate_miou(pred, target, num_classes=21, ignore_index=255):
        """
        다중 클래스 분류의 평균 IoU를 계산한다.

        인수:
            pred: 형태가 (batch, num_classes, H, W)인 예측
            target: 형태가 (batch, H, W)인 참값
            num_classes: 클래스의 개수
            ignore_index: 무시할 번호(경계)

        반환값:
            miou: 모든 클래스에 걸친 평균 IoU
            iou_per_class: 클래스마다의 IoU
        """
        with torch.no_grad():
            # 예측한 클래스 얻기
            pred_class = torch.argmax(pred, dim=1)  # (batch, H, W)

            # IoU 저장 리스트 초기화
            iou_per_class = []

            # 클래스마다 IoU 계산
            for cls in range(num_classes):
                # 지금 클래스의 이진 마스크 만들기
                pred_mask = (pred_class == cls)
                target_mask = (target == cls)

                # 경계 픽셀 무시하기
                valid_mask = (target != ignore_index)
                pred_mask = pred_mask & valid_mask
                target_mask = target_mask & valid_mask

                # 교집합과 합집합 계산
                intersection = (pred_mask & target_mask).float().sum()
                union = (pred_mask | target_mask).float().sum()

                # IoU 계산(0으로 나누기 피하기)
                if union == 0:
                    iou = float('nan')  # 그 클래스가 없음
                else:
                    iou = (intersection / union).item()

                iou_per_class.append(iou)

            # 평균 IoU 계산(없는 클래스의 NaN은 무시)
            valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
            miou = np.mean(valid_ious) if valid_ious else 0.0

            return miou, iou_per_class


    # ============================================================================
    # 7단계: 트레인 함수
    # ============================================================================

    def train_one_epoch(model, loader, criterion, optimizer, device):
        """모델을 한 에포크 트레인한다."""
        model.train()
        running_loss = 0.0
        running_miou = 0.0

        pbar = tqdm(loader, desc='Training')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            # 순전파
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 역전파
            loss.backward()
            optimizer.step()

            # 평균 IoU 계산
            miou, _ = calculate_miou(outputs, masks)

            # 통계
            running_loss += loss.item()
            running_miou += miou

            # 진행 막대를 고친다
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mIoU': f'{miou:.4f}'
            })

        epoch_loss = running_loss / len(loader)
        epoch_miou = running_miou / len(loader)

        return epoch_loss, epoch_miou


    # ============================================================================
    # 8단계: 검증 함수
    # ============================================================================

    def validate(model, loader, criterion, device):
        """모델을 검증한다."""
        model.eval()
        running_loss = 0.0
        running_miou = 0.0
        all_ious = [[] for _ in range(21)]  # 클래스별 IoU 저장

        with torch.no_grad():
            pbar = tqdm(loader, desc='Validation')
            for images, masks in pbar:
                images = images.to(device)
                masks = masks.to(device)

                # 순전파
                outputs = model(images)
                loss = criterion(outputs, masks)

                # 지표를 계산한다
                miou, iou_per_class = calculate_miou(outputs, masks)

                # 클래스별 IoU 저장
                for cls, iou in enumerate(iou_per_class):
                    if not np.isnan(iou):
                        all_ious[cls].append(iou)

                # 통계
                running_loss += loss.item()
                running_miou += miou

                # 진행 막대를 고친다
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mIoU': f'{miou:.4f}'
                })

        avg_loss = running_loss / len(loader)
        avg_miou = running_miou / len(loader)

        # 클래스마다 평균 IoU 계산
        avg_iou_per_class = [np.mean(ious) if ious else 0.0 for ious in all_ious]

        return avg_loss, avg_miou, avg_iou_per_class


    # ============================================================================
    # 9단계: 시각화 함수
    # ============================================================================

    def visualize_predictions(model, dataset, device, num_samples=3):
        """모델의 예측을 시각화한다."""
        model.eval()

        # VOC 클래스 이름
        class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                       'dining table', 'dog', 'horse', 'motorbike', 'person',
                       'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

        for i in range(num_samples):
            # 표본 하나 얻기
            image, mask = dataset[np.random.randint(len(dataset))]
            image_input = image.unsqueeze(0).to(device)

            # 예측한다
            with torch.no_grad():
                output = model(image_input)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            # 시각화하려고 이미지의 정규화 되돌리기
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)

            mask_np = mask.cpu().numpy()
            mask_np[mask_np == 255] = 0  # 시각화하려고 경계를 배경으로 두기

            # 마스크를 색으로 바꾸기
            def mask_to_color(mask, colormap):
                h, w = mask.shape
                color_mask = np.zeros((h, w, 3), dtype=np.uint8)
                for i, color in enumerate(colormap):
                    color_mask[mask == i] = color
                return color_mask

            gt_color = mask_to_color(mask_np, VOC_COLORMAP)
            pred_color = mask_to_color(pred_mask, VOC_COLORMAP)

            # 그림
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(gt_color)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_color)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig('pretrained_segmentation_results.png', dpi=150, bbox_inches='tight')
        print("\nVisualization saved as 'pretrained_segmentation_results.png'")
        plt.close()


    # ============================================================================
    # 10단계: 트레인 루프
    # ============================================================================

    NUM_EPOCHS = 30

    print(f"\n{'='*70}")
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print(f"{'='*70}\n")

    best_miou = 0.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 70)

        # 트레인
        train_loss, train_miou = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 검증
        val_loss, val_miou, val_iou_per_class = validate(
            model, val_loader, criterion, device
        )

        # 결과 출력
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, mIoU: {train_miou:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}")

        # 러닝레이트 스케줄링
        scheduler.step(val_miou)

        # 최고 성능 모델 저장
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), 'best_pretrained_segmentation.pth')
            print(f"  ✓ New best mIoU! Model saved.")

        # 5에포크마다 클래스별 IoU 찍기
        if (epoch + 1) % 5 == 0:
            print("\n  Per-class IoU:")
            class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                          'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                          'dining table', 'dog', 'horse', 'motorbike', 'person',
                          'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']
            for cls, (name, iou) in enumerate(zip(class_names, val_iou_per_class)):
                if iou > 0:
                    print(f"    {name:>15s}: {iou:.4f}")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best validation mIoU: {best_miou:.4f}")
    print(f"{'='*70}\n")

    # ============================================================================
    # 11단계: 마지막 평가
    # ============================================================================

    # 가장 좋은 모델을 불러온다
    model.load_state_dict(torch.load('best_pretrained_segmentation.pth'))

    print("Final Validation Evaluation:")
    print("="*70)
    val_loss, val_miou, val_iou_per_class = validate(
        model, val_loader, criterion, device
    )
    print(f"\nFinal Validation mIoU: {val_miou:.4f}")

    # 클래스별 자세한 결과 찍기
    print("\nPer-class IoU:")
    print("-" * 70)
    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'dining table', 'dog', 'horse', 'motorbike', 'person',
                   'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    for name, iou in zip(class_names, val_iou_per_class):
        if iou > 0:
            print(f"{name:>15s}: {iou:.4f}")

    # 예측 시각화
    print("\nGenerating visualizations...")
    visualize_predictions(model, val_dataset, device, num_samples=5)

    # ============================================================================
    # 요약
    # ============================================================================

    print("\n" + "="*70)
    print("TRANSFER LEARNING SEGMENTATION COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Used pre-trained ResNet-50 encoder from ImageNet")
    print("2. Applied DeepLabV3+ architecture with ASPP")
    print("3. Used discriminative learning rates (encoder vs decoder)")
    print("4. Achieved strong performance on PASCAL VOC (21 classes)")
    print("5. Used mIoU as the primary evaluation metric")
    print("\nAdvantages of Pre-trained Encoders:")
    print("✓ Faster training convergence")
    print("✓ Better performance with limited data")
    print("✓ Leverages knowledge from millions of ImageNet images")
    print("✓ Industry-standard approach")
    print("\nNext Steps:")
    print("- Try Example 3 for medical image segmentation")
    print("- Experiment with different encoders (EfficientNet, VGG)")
    print("- Try other architectures (U-Net, FPN)")
    print("- Fine-tune hyperparameters")
    print("="*70)


    if __name__ == "__main__":
        pass
    ```


## 2. 논의

### VOC의 픽셀 라벨과 전처리

VOC2012의 기본 segmentation train·val은 각각 1,464장과 1,449장이다. 배경 0과 객체 클래스 1~20을 사용하고 255는 void로 제외한다. 정답의 255를 일반 배경으로 트레인시키지 않는다. 공식 데이터의 크기·라벨 규약은 [VOC 통계](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/dbstats.html)와 [개발 문서](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html)를 참고한다.

이미지는 512×512로 resize하고 ImageNet 평균·표준편차로 정규화한다. 마스크는 최근접 보간과 PILToTensor로 정수 클래스 ID를 보존한 뒤 long H×W로 반환한다. 마스크에 이미지처럼 /255 정규화를 적용하면 클래스가 훼손된다. 현재는 두 경로 모두 결정적인 같은 크기 resize라 좌표가 대응하지만, 무작위 crop·회전을 추가하려면 같은 난수를 이미지·마스크에 공유해야 한다. 종횡비를 고정 크기로 변형하는 효과도 평가의 일부다.

### 사전 트레인 범위와 러닝레이트

`encoder_weights='imagenet'`는 인코더의 사전 트레인 웨이트를 지정한다. 이 설정만으로 VOC에 이미 트레인된 디코더·21클래스 헤드가 제공되는 것은 아니다. 기본 엑티베이션 없는 모델의 출력은 N×21×H×W 로짓이며 CE는 N×H×W의 클래스 라벨과 연결한다. 처음 실행하면 데이터와 웨이트를 내려받을 수 있다. 정의는 [사용 라이브러리의 모델 설명](https://smp.readthedocs.io/en/v0.3.0/_modules/segmentation_models_pytorch/decoders/deeplabv3/model.html)을 해당 설치 버전과 대조한다.

코드는 이름에 encoder가 포함된 파라메터의 러닝레이트를 0.0001, 나머지를 0.001로 나눈다. 이는 인코더를 동결한 것이 아니며 `model.train()`은 BN의 이동 통계도 갱신한다. 차등 러닝레이트는 전이학습의 한 선택이며 항상 단일 러닝레이트보다 낫거나 이런 비율이 필수인 것은 아니다. 파라메터 이름에 의존하는 그룹 구분이 실제 의도한 모든 파라메터를 포함하는지도 확인해야 한다.

### mIoU 집계의 실제 의미

`calculate_miou`는 한 배치에서 클래스별 교집합·합집합을 합하고, 합집합이 0인 클래스를 제외해 평균한다. 255인 정답 픽셀은 예측·정답 양쪽에서 제외한다. 이 배치 mIoU를 epoch에서 다시 단순 평균하므로 데이터 전체의 누적 혼동행렬에서 구한 공식 방식의 mIoU와 일반적으로 다르다.

클래스별 IoU도 해당 클래스가 유효했던 배치의 비율을 평균한다. 배치마다 클래스 집합과 합집합 크기가 달라질 수 있어 최종 per-class 평균과 전체 평균의 관계도 단순하지 않다. 정확한 데이터 단위 지표를 원하면 클래스별 TP·FP·FN을 전체 유효 픽셀에 누적한 뒤 비율을 구해야 한다. 마지막 작은 배치와 모두 void인 배치도 규칙을 명시한다.

출력은 IoU>0인 클래스만 인쇄하므로 0점 클래스가 표시에서 빠진다. 생략된 클래스가 존재하지 않거나 성능이 양호하다는 뜻은 아니다. 시각화에서 void를 0으로 색칠한 것은 표시 선택이며 평가의 void 마스크를 배경 정답으로 바꿔도 된다는 의미가 아니다.

### 검증 선택과 체크포인트

검증 mIoU로 scheduler와 최선 웨이트를 선택하고, 마지막에도 같은 검증셋을 평가한다. 이는 선택에 사용하지 않은 독립 시험 성능이 아니다. `best_miou=0`에서 엄격한 개선만 저장하므로 새 체크포인트가 없을 때 고정 파일 로드가 실패하거나 이전 실행의 파일을 읽을 수 있다. 기본 루프에는 조기 종료가 없으며 연습의 예제는 별도 골격이다.

연습의 `model.state_dict().copy()`는 얕은 사전 복사라 텐서 저장소가 후속 트레인과 공유될 수 있다. 최선 시점의 값을 실제로 보존하려면 깊은 복사·텐서 복제 또는 저장 파일처럼 독립적인 snapshot이 필요하다. 루프 마지막에도 최선 상태를 복원하는 규칙과 개선이 없는 초기·NaN 지표의 처리도 정해야 한다.

### 실행과 결과의 범위

긴 코드에는 데이터 다운로드·worker 생성·트레인이 최상위에 있다. 끝의 main guard 안에 pass가 있다고 import의 부작용이 사라지지는 않으며, 프로세스 spawn 환경에서는 보호 범위가 중요하다. 설치한 torchvision·SMP·scheduler API의 호환성도 필요하다.

코드가 끝에 찍는 요약 문자열의 ‘강한 성능’이나 전이학습의 이점은 이 쪽에서 확인한 실험 결과가 아니다. 사전 트레인이 도움을 줄 수 있지만 새 데이터·정규화·모델 용량에 따라 달라진다. 예측 그림 몇 장은 전체 지표와 실패 사례 분석을 대신하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 코드에서 인코더를 동결했는가? 그래디언트 초기화를 생략하면 어떤 계산이 되는가?

</div>

??? success "풀이"
    동결하지 않았다. 인코더 파라메터도 작은 러닝레이트로 Adam에 등록되고 BN 통계도 트레인 모드에서 변한다. zero_grad를 생략하면 이전 그래디언트가 누적되어 의도한 배치별 갱신과 달라진다. 변화하는 파라메터와 Adam 상태가 있어 단순히 러닝레이트에 단계 수를 곱한 것과 동일하지는 않다. 의도적인 그래디언트 누적은 로스 배율과 초기화 시점을 따로 설계한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 두 러닝레이트 그룹을 단일 Adam 러닝레이트로 바꾼 비교에서 통제할 조건을 설명하라.

</div>

??? success "풀이"
    현재 코드도 Adam이며 인코더 0.0001·나머지 0.001의 두 그룹을 사용한다. 모두 0.001로 바꾸는 것은 옵티마이저 종류뿐 아니라 인코더의 배율을 바꾸는 실험이다. 같은 사전 트레인 초기값·데이터 순서·트레인 예산을 맞추고 러닝레이트 후보와 검증 선택을 구분해 비교해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
배치 mIoU의 평균을 VOC 데이터 전체 mIoU와 동일하게 볼 수 있는가?

</div>

??? success "풀이"
    아니다. 배치마다 합집합 크기와 유효 클래스가 다르고 마지막 배치 크기도 다를 수 있다. 전체 클래스별 교집합·합집합을 누적한 뒤 IoU와 클래스 평균을 구하는 값과 다르다. void 처리·배경 포함·빈 클래스와 0점 클래스의 표시 규칙도 명시해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 골격이 최선 웨이트를 정확히 보존·복원하는지 검토하라.

</div>

??? success "풀이"

    state_dict().copy()는 텐서까지 독립 복사하지 않으므로 후속 트레인이 저장된 값에도 반영될 수 있다. deepcopy·clone 또는 파일 snapshot이 필요하다. 또한 evaluate와 num_epochs 등은 외부 정의이고, patience로 break하지 않고 최대 epoch로 끝나면 복원 코드도 실행되지 않는다. 모든 종료 경로에서 유효한 best_state를 확인해 복원하고 독립 시험 평가는 별도로 남겨야 한다.

    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 트레인 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```


## 정리하며

사전 트레인 인코더의 파인튜닝은 라벨·전처리·러닝레이트를 새 과제에 맞추는 과정이다. 배치 mIoU와 전체 데이터 지표, 검증 선택과 독립 시험을 구분하고, 최선 체크포인트를 실제로 독립 보존하는지 확인한다.
