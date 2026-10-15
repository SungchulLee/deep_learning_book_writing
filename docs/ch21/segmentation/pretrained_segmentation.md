# 사전 학습 모델을 이용한 이미지 분할

이 예제는 ImageNet 사전 학습 ResNet-50 인코더를 사용하는 DeepLabV3+를 PASCAL VOC2012 시맨틱 분할에 파인튜닝한다. 사전 학습된 분할 모델을 그대로 평가하는 것과 달리 decoder와 분할 head를 새 과제에서 학습한다. 데이터 전처리·무시 레이블·차등 학습률과 지표의 집계 방식을 함께 확인한다.

## 1. 코드

??? note "코드 (617줄)"

    ```python
    """
    보기 2: 미리 익힌 인코더로 나누기에 옮겨 배우기
    ========================================================================

    이 각본은 미리 익힌 인코더(ResNet, VGG 등)를 쓰는 법을 보여 준다
    뜻 나누기에 쓰는 법을 보여 준다. segmentation_models_pytorch 라이브러리를 쓰는데
    이는 ImageNet에서 미리 익힌 등뼈를 갖춘 여러 구조를 준다.

    핵심 개념:
    - 나누기를 위한 옮겨 배우기
    - 미리 익힌 인코더(ResNet, EfficientNet)
    - DeepLabV3+ 구조
    - 여러 갈래 나누기(갈래 21개)
    - 실제 자료 뭉치 다루기
    - 앞선 값매김 잣대(평균 겹침 비)

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

    # 난수 씨앗을 설정한다
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
    # 2단계: 자료 뭉치 갖추기
    # ============================================================================
    """
    나누기의 표준 잣대인 PASCAL VOC 2012 자료 뭉치를 쓴다.

    PASCAL VOC에는 다음이 들어 있다:
    - 익힘 그림 1,464장
    - 검증 그림 1,449장
    - 갈래 21개(물체 갈래 20개 + 뒷바탕)

    갈래: background, aeroplane, bicycle, bird, boat, bottle, bus, car,
             고양이, 의자, 소, 식탁, 개, 말, 오토바이, 사람,
             화분, 양, 소파, 기차, 텔레비전과 모니터
    """

    # 그려 보기용 VOC 빛깔판(갈래마다의 RGB 값)
    VOC_COLORMAP = [
        [0, 0, 0],        # 뒷바탕
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

    # 그림 바꾸기
    # 나누기에서는 그림과 마스크를 함께 바꿔야 한다
    IMG_SIZE = 512  # DeepLab은 512×512에서 잘 된다

    # 바꾸기 정하기
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
        그림과 마스크 모두에 바꾸기를 쓰는 맞춤 VOC 자료 뭉치 클래스.
        """
        def __getitem__(self, index):
            img, mask = super().__getitem__(index)

            # 그림 바꾸기 쓰기
            img = image_transform(img)

            # 마스크 바꾸기 쓰기
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
    BATCH_SIZE = 4  # 나누기는 기억 공간을 많이 쓴다

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
    # 3단계: 미리 익힌 인코더로 모델 고르기
    # ============================================================================
    """
    segmentation_models_pytorch(smp)는 미리 익힌 인코더를 갖춘 여러 구조를 준다
    인코더를 쓴다. 여기서는 ResNet-50 인코더를 쓴 DeepLabV3+를 쓴다.

    구조 고름:
    - Unet: 건너뛰는 이음을 갖춘 고전 U-넷
    - UnetPlusPlus: 겹겹이 든 건너뛰는 이음을 갖춘 나아진 U-넷
    - DeepLabV3+: 구멍 뚫린 누비기와 ASPP를 쓴 가장 앞선 것
    - FPN: 특징 피라미드 그물
    - PSPNet: 피라미드 장면 뜯어 읽기 그물

    인코더 고름:
    - resnet18, resnet34, resnet50, resnet101
    - efficientnet-b0 to b7
    - mobilenet_v2
    - vgg16, vgg19
    """

    print("Creating DeepLabV3+ model with pre-trained ResNet-50 encoder...")

    # 미리 익힌 인코더로 모델 만들기
    model = smp.DeepLabV3Plus(
        encoder_name="resnet50",           # 인코더 고르기(resnet50, efficientnet-b0 등)
        encoder_weights="imagenet",         # ImageNet에서 미리 익힌 무게 쓰기
        in_channels=3,                      # 들임 채널(RGB)
        classes=21,                         # 내놓는 갈래(VOC는 21개)
    )

    model = model.to(device)

    # 모델 간추림
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
    # 4단계: 손실 함수
    # ============================================================================
    """
    여러 갈래 나누기에는 다음을 갖춘 CrossEntropyLoss를 쓴다:
    - ignore_index=255: VOC는 테두리/불확실 화소에 255를 쓴다
    - 치우친 갈래를 위한, 있어도 되는 갈래 무게

    다른 손실:
    - 다이스 손실: 치우친 갈래에 더 낫다
    - 초점 손실: 어려운 보기에 초점을 둔다
    - 아우름: 엇갈린 엔트로피 + 다이스
    """

    # 여러 갈래 나누기를 위한 CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # VOC는 테두리에 255를 쓴다

    # 대안: smp의 다이스 손실 쓰기
    # criterion = smp.losses.DiceLoss(mode='multiclass', ignore_index=255)

    # 또는 아우른 손실
    # criterion = smp.losses.DiceLoss(mode='multiclass') + nn.CrossEntropyLoss()

    print("\nLoss function: CrossEntropyLoss (ignore boundary pixels)")

    # ============================================================================
    # 5단계: 층마다 배움 비율이 다른 가장 좋게 하개
    # ============================================================================
    """
    옮겨 배우기에는 배움 비율을 달리 쓴다:
    - 인코더(미리 익힘): 작은 배움 비율(0.0001) — 조심스레 곱게 다듬는다
    - 디코더(마구잡이 첫자리매김): 큰 배움 비율(0.001) — 더 많이 배워야 한다

    이를 "층마다 다른 배움 비율"이라 하며
    나누기에서 옮겨 배우기를 잘하는 데 결정적이다.
    """

    # 인코더와 디코더의 매개변수 나누기
    encoder_params = []
    decoder_params = []

    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    # 배움 비율이 다른 가장 좋게 하개 만들기
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': 0.0001},   # 미리 익힌 인코더에는 작은 배움 비율
        {'params': decoder_params, 'lr': 0.001},    # 디코더에는 더 큰 배움 비율
    ])

    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',      # 평균 겹침 비를 가장 크게
        factor=0.5,
        patience=3,
        verbose=True
    )

    print("\nOptimizer: Adam with discriminative learning rates")
    print(f"  Encoder LR: 0.0001 (pre-trained, fine-tune)")
    print(f"  Decoder LR: 0.001 (random init, learn)")

    # ============================================================================
    # 6단계: 값매김 잣대
    # ============================================================================
    """
    여러 갈래 나누기에는 다음을 쓴다:
    1. mIoU(평균 겹침 비): 모든 갈래에 걸친 겹침 비의 평균
    2. 갈래별 겹침 비: 갈래마다의 겹침 비
    3. 화소 정확도: 전체 화소 갈래 매기기 정확도

    평균 겹침 비는 나누기 잣대 시험의 표준 잣대이다.
    """

    def calculate_miou(pred, target, num_classes=21, ignore_index=255):
        """
        여러 갈래 나누기의 평균 겹침 비를 셈한다.

        인수:
            pred: 꼴이 (batch, num_classes, H, W)인 어림
            target: 꼴이 (batch, H, W)인 참값
            num_classes: 갈래의 개수
            ignore_index: 무시할 번호(테두리)

        반환값:
            miou: 모든 갈래에 걸친 평균 겹침 비
            iou_per_class: 갈래마다의 겹침 비
        """
        with torch.no_grad():
            # 어림한 갈래 얻기
            pred_class = torch.argmax(pred, dim=1)  # (batch, H, W)

            # 겹침 비 갈무리 첫자리매김
            iou_per_class = []

            # 갈래마다 겹침 비 셈하기
            for cls in range(num_classes):
                # 지금 갈래의 두 갈래 마스크 만들기
                pred_mask = (pred_class == cls)
                target_mask = (target == cls)

                # 테두리 화소 무시하기
                valid_mask = (target != ignore_index)
                pred_mask = pred_mask & valid_mask
                target_mask = target_mask & valid_mask

                # 교집합과 합집합 계산
                intersection = (pred_mask & target_mask).float().sum()
                union = (pred_mask | target_mask).float().sum()

                # 겹침 비 셈하기(0으로 나누기 피하기)
                if union == 0:
                    iou = float('nan')  # 그 갈래가 없음
                else:
                    iou = (intersection / union).item()

                iou_per_class.append(iou)

            # 평균 겹침 비 셈하기(없는 갈래의 NaN은 무시)
            valid_ious = [iou for iou in iou_per_class if not np.isnan(iou)]
            miou = np.mean(valid_ious) if valid_ious else 0.0

            return miou, iou_per_class


    # ============================================================================
    # 7단계: 학습 함수
    # ============================================================================

    def train_one_epoch(model, loader, criterion, optimizer, device):
        """모델을 한 세대 학습한다."""
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

            # 평균 겹침 비 셈하기
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
        all_ious = [[] for _ in range(21)]  # 갈래별 겹침 비 갈무리

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

                # 갈래별 겹침 비 갈무리
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

        # 갈래마다 평균 겹침 비 셈하기
        avg_iou_per_class = [np.mean(ious) if ious else 0.0 for ious in all_ious]

        return avg_loss, avg_miou, avg_iou_per_class


    # ============================================================================
    # 9단계: 그려 보기 함수
    # ============================================================================

    def visualize_predictions(model, dataset, device, num_samples=3):
        """모델의 어림을 그려 본다."""
        model.eval()

        # VOC 갈래 이름
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

            # 그려 보려고 그림의 고르게 맞추기 되돌리기
            image_np = image.cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            image_np = std * image_np + mean
            image_np = np.clip(image_np, 0, 1)

            mask_np = mask.cpu().numpy()
            mask_np[mask_np == 255] = 0  # 그려 보려고 테두리를 뒷바탕으로 두기

            # 마스크를 빛깔로 바꾸기
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
    # 10단계: 익히기 되풀이
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

        # 학습
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

        # 학습률 스케줄링
        scheduler.step(val_miou)

        # 최고 성능 모델 저장
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), 'best_pretrained_segmentation.pth')
            print(f"  ✓ New best mIoU! Model saved.")

        # 5세대마다 갈래별 겹침 비 찍기
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
    # 11단계: 마지막 값매김
    # ============================================================================

    # 가장 좋은 모델을 불러온다
    model.load_state_dict(torch.load('best_pretrained_segmentation.pth'))

    print("Final Validation Evaluation:")
    print("="*70)
    val_loss, val_miou, val_iou_per_class = validate(
        model, val_loader, criterion, device
    )
    print(f"\nFinal Validation mIoU: {val_miou:.4f}")

    # 갈래별 자세한 결과 찍기
    print("\nPer-class IoU:")
    print("-" * 70)
    class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                   'dining table', 'dog', 'horse', 'motorbike', 'person',
                   'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

    for name, iou in zip(class_names, val_iou_per_class):
        if iou > 0:
            print(f"{name:>15s}: {iou:.4f}")

    # 어림 그려 보기
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

### VOC의 픽셀 레이블과 전처리

VOC2012의 기본 segmentation train·val은 각각1,464장과1,449장이다. 배경0과 객체 클래스1~20을 사용하고255는 void로 제외한다. 정답의255를 일반 배경으로 학습시키지 않는다. 공식 데이터의 크기·레이블 규약은 [VOC 통계](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/dbstats.html)와 [개발 문서](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/htmldoc/index.html)를 참고한다.

이미지는512×512로 resize하고 ImageNet 평균·표준편차로 정규화한다. 마스크는 최근접 보간과 PILToTensor로 정수 클래스 ID를 보존한 뒤 long H×W로 반환한다. 마스크에 이미지처럼 /255 정규화를 적용하면 클래스가 훼손된다. 현재는 두 경로 모두 결정적인 같은 크기 resize라 좌표가 대응하지만, 무작위 crop·회전을 추가하려면 같은 난수를 이미지·마스크에 공유해야 한다. 종횡비를 고정 크기로 변형하는 효과도 평가의 일부다.

### 사전 학습 범위와 학습률

`encoder_weights='imagenet'`는 인코더의 사전 학습 가중치를 지정한다. 이 설정만으로 VOC에 이미 학습된 decoder·21클래스 head가 제공되는 것은 아니다. 기본 활성화 없는 모델의 출력은 N×21×H×W 로짓이며 CE는 N×H×W의 클래스 레이블과 연결한다. 필요 시 데이터와 가중치 다운로드가 발생할 수 있으나 이번 편집에서는 실행하지 않았다. 정의는 [사용 라이브러리의 모델 설명](https://smp.readthedocs.io/en/v0.3.0/_modules/segmentation_models_pytorch/decoders/deeplabv3/model.html)을 해당 설치 버전과 대조한다.

코드는 이름에 encoder가 포함된 파라미터의 학습률을0.0001, 나머지를0.001로 나눈다. 이는 인코더를 동결한 것이 아니며 `model.train()`은 BN의 이동 통계도 갱신한다. 차등 학습률은 전이학습의 한 선택이며 항상 단일 학습률보다 낫거나 이런 비율이 필수인 것은 아니다. 파라미터 이름에 의존하는 그룹 구분이 실제 의도한 모든 파라미터를 포함하는지도 확인해야 한다.

### mIoU 집계의 실제 의미

`calculate_miou`는 한 배치에서 클래스별 교집합·합집합을 합하고, 합집합0인 클래스를 제외해 평균한다.255의 정답 픽셀은 예측·정답 양쪽에서 제외한다. 이 배치 mIoU를 epoch에서 다시 단순 평균하므로 데이터 전체의 누적 혼동행렬에서 구한 공식 방식의 mIoU와 일반적으로 다르다.

클래스별 IoU도 해당 클래스가 유효했던 배치의 비율을 평균한다. 배치마다 클래스 집합과 합집합 크기가 달라질 수 있어 최종 per-class 평균과 전체 평균의 관계도 단순하지 않다. 정확한 데이터 단위 지표를 원하면 클래스별 TP·FP·FN을 전체 유효 픽셀에 누적한 뒤 비율을 구해야 한다. 마지막 작은 배치와 모두 void인 배치도 규칙을 명시한다.

출력은 IoU>0인 클래스만 인쇄하므로0점 클래스가 표시에서 빠진다. 생략된 클래스가 존재하지 않거나 성능이 양호하다는 뜻은 아니다. 시각화에서 void를0으로 색칠한 것은 표시 선택이며 평가의 void 마스크를 배경 정답으로 바꿔도 된다는 의미가 아니다.

### 검증 선택과 체크포인트

검증 mIoU로 scheduler와 최선 가중치를 선택하고, 마지막에도 같은 검증셋을 평가한다. 이는 선택에 사용하지 않은 독립 테스트 성능이 아니다. `best_miou=0`에서 엄격한 개선만 저장하므로 새 체크포인트가 없을 때 고정 파일 로드가 실패하거나 이전 실행의 파일을 읽을 수 있다. 기본 루프에는 조기 종료가 없으며 연습의 예제는 별도 골격이다.

연습의 `model.state_dict().copy()`는 얕은 사전 복사라 텐서 저장소가 후속 학습과 공유될 수 있다. 최선 시점의 값을 실제로 보존하려면 깊은 복사·텐서 복제 또는 저장 파일처럼 독립적인 snapshot이 필요하다. 루프 마지막에도 최선 상태를 복원하는 규칙과 개선이 없는 초기·NaN 지표의 처리도 정해야 한다.

### 실행과 결과의 범위

긴 코드에는 데이터 다운로드·worker 생성·학습이 최상위에 있다. 끝의 main guard 안에 pass가 있다고 import의 부작용이 사라지지는 않으며, 프로세스 spawn 환경에서는 보호 범위가 중요하다. 설치한 torchvision·SMP·scheduler API의 호환성도 필요하다.

기존 출력 문자열의 ‘강한 성능’이나 전이학습의 이점은 이 편집에서 확인한 실험 결과가 아니다. 사전 학습이 도움을 줄 수 있지만 새 데이터·정규화·모델 용량에 따라 달라진다. 예측 그림 몇 장은 전체 지표와 실패 사례 분석을 대신하지 않는다. 코드·출력·그림은 그대로 보존했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
이 코드에서 encoder를 동결했는가? 경사 초기화를 생략하면 어떤 계산이 되는가?

</div>

??? success "연습문제 1 풀이"
    동결하지 않았다. encoder 파라미터도 작은 학습률로 Adam에 등록되고 BN 통계도 학습 모드에서 변한다. zero_grad를 생략하면 이전 경사가 누적되어 의도한 배치별 갱신과 달라진다. 변화하는 파라미터와 Adam 상태가 있어 단순히 학습률에 단계 수를 곱한 것과 동일하지는 않다. 의도적인 경사 누적은 손실 배율과 초기화 시점을 따로 설계한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
현재 두 학습률 그룹을 단일 Adam 학습률로 바꾼 비교에서 통제할 조건을 설명하라.

</div>

??? success "연습문제 2 풀이"
    기존 코드도 Adam이며 encoder0.0001·나머지0.001의 두 그룹을 사용한다. 모두0.001로 바꾸는 것은 옵티마이저 종류뿐 아니라 encoder의 배율을 바꾸는 실험이다. 같은 사전 학습 초기값·데이터 순서·학습 예산을 맞추고 학습률 후보와 검증 선택을 구분해 비교해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
배치 mIoU의 평균을 VOC 데이터 전체 mIoU와 동일하게 볼 수 있는가?

</div>

??? success "연습문제 3 풀이"
    아니다. 배치마다 합집합 크기와 유효 클래스가 다르고 마지막 배치 크기도 다를 수 있다. 전체 클래스별 교집합·합집합을 누적한 뒤 IoU와 클래스 평균을 구하는 값과 다르다. void 처리·배경 포함·빈 클래스와0점 클래스의 표시 규칙도 명시해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 조기 종료 골격이 최선 가중치를 정확히 보존·복원하는지 검토하라.

</div>

??? success "연습문제 4 풀이"

    state_dict().copy()는 텐서까지 독립 복사하지 않으므로 후속 학습이 저장된 값에도 반영될 수 있다. deepcopy·clone 또는 파일 snapshot이 필요하다. 또한 evaluate와 num_epochs 등은 외부 정의이고, patience로 break하지 않고 최대 epoch로 끝나면 복원 코드도 실행되지 않는다. 모든 종료 경로에서 유효한 best_state를 확인해 복원하고 독립 테스트 평가는 별도로 남겨야 한다. 기존 골격은 유지했다.

    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
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

사전 학습 인코더의 파인튜닝은 라벨·전처리·학습률을 새 과제에 맞추는 과정이다. 배치 mIoU와 전체 데이터 지표, 검증 선택과 독립 테스트를 구분하고, 최선 체크포인트를 실제로 독립 보존하는지 확인한다.
