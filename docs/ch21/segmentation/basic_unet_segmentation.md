# U-Net 이미지 분할 기초

이 예제는 합성 도형 이미지에서 배경과 전경을 구분하는 U-Net 학습 흐름을 담는다. 데이터 생성, 두 클래스 로짓, 픽셀별 교차 엔트로피, 검증·체크포인트·시각화를 연결한다. 간단한 색상 규칙을 갖는 교육용 문제이며 실제 장면이나 의료 영상의 성능을 검증하는 실험은 아니다.

## 1. 코드

??? note "코드 (627줄)"

    ```python
    """
    보기 1: 기본 U-넷 뜻 나누기
    =============================================

    이 각본은 온전한 U-넷 구조를 맨바닥부터 짠다
    두 클래스 뜻 나누기를 한다. 단순한 꼴로 된 인공 자료 뭉치를 쓴다.

    핵심 개념:
    - U-넷 구조(건너뛰는 이음을 갖춘 인코더-디코더)
    - 두 분류(클래스 2개)
    - 화소마다의 엇갈린 엔트로피 로스
    - 겹침 비(IoU) 잣대
    - 나누기를 위한 자료 불리기

    지은이: PyTorch Semantic Segmentation Tutorial
    날짜: 2025
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    import time
    import os

    # 재현성을 위한 난수 시드 설정
    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================================
    # 1단계: 장치 설정
    # ============================================================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ============================================================================
    # 2단계: 인공 자료 뭉치 만들기
    # ============================================================================
    """
    배우려는 목적이므로 단순한 꼴로 된 인공 자료 뭉치를 만든다.
    그림마다 동그라미, 네모, 세모가 들어 있고 이를
    뒷바탕에서 갈라내야 한다.

    실제 쓰임새에서는 진짜 그림과 마스크를 읽어 들인다.
    """

    class SyntheticShapesDataset(Dataset):
        """
        나누기를 위해 단순한 꼴로 된 인공 그림을 만든다.

        표본마다 다음으로 이루어진다:
        - 들임 그림: 꼴이 그려진 RGB 그림
        - 마스크: 1=꼴, 0=뒷바탕인 두 클래스 마스크
        """

        def __init__(self, num_samples=1000, image_size=256, transform=None):
            """
            인수:
                num_samples: 만들 인공 그림의 개수
                image_size: 정사각 그림의 크기(image_size × image_size)
                transform: 자료 불리기(없어도 됨)
            """
            self.num_samples = num_samples
            self.image_size = image_size
            self.transform = transform

            # 한결같도록 표본을 미리 모두 만들기
            self.samples = []
            print(f"Generating {num_samples} synthetic images...")
            for i in range(num_samples):
                img, mask = self._generate_sample()
                self.samples.append((img, mask))
                if (i + 1) % 200 == 0:
                    print(f"  Generated {i + 1}/{num_samples} images")
            print("Dataset generation complete!\n")

        def _generate_sample(self):
            """그림-마스크 짝 하나를 만든다."""
            # 빈 그림 만들기(마구잡이 뒷바탕 빛깔)
            bg_color = tuple(np.random.randint(0, 100, 3).tolist())
            img = Image.new('RGB', (self.image_size, self.image_size), bg_color)
            mask = Image.new('L', (self.image_size, self.image_size), 0)

            draw_img = ImageDraw.Draw(img)
            draw_mask = ImageDraw.Draw(mask)

            # 마구잡이 꼴 빛깔(뒷바탕과 다르게)
            shape_color = tuple(np.random.randint(150, 255, 3).tolist())

            # 꼴 클래스를 마구잡이로 고르기
            shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])

            # 마구잡이 자리와 크기
            center_x = np.random.randint(64, self.image_size - 64)
            center_y = np.random.randint(64, self.image_size - 64)
            size = np.random.randint(30, 70)

            if shape_type == 'circle':
                # 동그라미 그리기
                bbox = [center_x - size, center_y - size, 
                       center_x + size, center_y + size]
                draw_img.ellipse(bbox, fill=shape_color)
                draw_mask.ellipse(bbox, fill=255)  # 255 = 마스크의 앞바탕

            elif shape_type == 'rectangle':
                # 네모 그리기
                bbox = [center_x - size, center_y - size, 
                       center_x + size, center_y + size]
                draw_img.rectangle(bbox, fill=shape_color)
                draw_mask.rectangle(bbox, fill=255)

            else:  # 세모
                # 세모 그리기
                points = [
                    (center_x, center_y - size),
                    (center_x - size, center_y + size),
                    (center_x + size, center_y + size)
                ]
                draw_img.polygon(points, fill=shape_color)
                draw_mask.polygon(points, fill=255)

            return img, mask

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            """
            자료 뭉치에서 표본 하나를 얻는다.

            반환값:
                image: 꼴이 (3, H, W)이고 [0, 1]로 고르게 맞춘 텐서
                mask: 꼴이 (1, H, W)이고 값이 {0, 1}인 텐서
            """
            img, mask = self.samples[idx]

            # PIL 그림을 numpy 배열로 바꾸기
            img_array = np.array(img).astype(np.float32) / 255.0  # [0, 1]로 고르게 맞추기
            mask_array = np.array(mask).astype(np.float32) / 255.0  # [0, 1]로 바꾸기

            # PyTorch 텐서로 변환
            # 그림: (H, W, C) -> (C, H, W)
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            # 마스크: (H, W) -> (1, H, W)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

            # 주어졌으면 불리기 쓰기
            if self.transform:
                # 단순하게 하려고 기본 바꾸기만 쓴다
                # 실전에서는 마스크를 제대로 불리려면 albumentations를 쓴다
                img_tensor, mask_tensor = self.transform(img_tensor, mask_tensor)

            return img_tensor, mask_tensor


    # 단순한 불리기 함수
    def simple_augment(img, mask):
        """그림과 마스크 모두에 마구잡이 가로 뒤집기를 쓴다."""
        if np.random.random() > 0.5:
            img = torch.flip(img, dims=[2])  # 너비 뒤집기
            mask = torch.flip(mask, dims=[2])
        return img, mask


    # 데이터셋들을 만든다
    print("Creating datasets...")
    train_dataset = SyntheticShapesDataset(num_samples=800, image_size=256)
    val_dataset = SyntheticShapesDataset(num_samples=100, image_size=256)
    test_dataset = SyntheticShapesDataset(num_samples=100, image_size=256)

    # 데이터 로더 생성
    BATCH_SIZE = 8
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # ============================================================================
    # 3단계: U-넷 구조 짜기
    # ============================================================================
    """
    U-넷은 다음으로 이루어진다:
    1. 인코더(오그라드는 길): 맥락을 담아낸다
    2. 병목: 가장 깊은 피처
    3. 디코더(부풀어 오르는 길): 정밀한 자리 잡기를 가능하게 한다
    4. 건너뛰는 이음: 인코더 피처를 디코더에 이어 붙인다
    """

    class DoubleConv(nn.Module):
        """
        겹 컨볼루션 덩이: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
        이것이 U-넷의 기본 벽돌이다.
        """
        def __init__(self, in_channels, out_channels):
            super(DoubleConv, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)


    class UNet(nn.Module):
        """
        뜻 나누기를 위한 U-넷 구조.

        구조:
            들임(3채널) -> 인코더(건너뛰는 이음 포함) -> 병목 ->
            디코더(건너뛰는 이음 포함) -> 내놓음(num_classes 채널)

        인수:
            in_channels: 들임 채널의 개수(RGB는 3)
            num_classes: 내놓는 클래스의 개수(두 클래스는 2: 뒷바탕, 앞바탕)
        """

        def __init__(self, in_channels=3, num_classes=2):
            super(UNet, self).__init__()

            # 인코더(줄여 뽑는 길)
            self.enc1 = DoubleConv(in_channels, 64)      # 256x256 -> 256x256
            self.pool1 = nn.MaxPool2d(2)                  # 256x256 -> 128x128

            self.enc2 = DoubleConv(64, 128)               # 128x128 -> 128x128
            self.pool2 = nn.MaxPool2d(2)                  # 128x128 -> 64x64

            self.enc3 = DoubleConv(128, 256)              # 64x64 -> 64x64
            self.pool3 = nn.MaxPool2d(2)                  # 64x64 -> 32x32

            self.enc4 = DoubleConv(256, 512)              # 32x32 -> 32x32
            self.pool4 = nn.MaxPool2d(2)                  # 32x32 -> 16x16

            # 병목
            self.bottleneck = DoubleConv(512, 1024)       # 16x16 -> 16x16

            # 디코더(키우는 길)
            self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 16x16 -> 32x32
            self.dec4 = DoubleConv(1024, 512)             # 32x32 -> 32x32(이어 붙여서 1024)

            self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # 32x32 -> 64x64
            self.dec3 = DoubleConv(512, 256)              # 64x64 -> 64x64

            self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # 64x64 -> 128x128
            self.dec2 = DoubleConv(256, 128)              # 128x128 -> 128x128

            self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    # 128x128 -> 256x256
            self.dec1 = DoubleConv(128, 64)               # 256x256 -> 256x256

            # 마지막 내놓는 레이어
            self.out = nn.Conv2d(64, num_classes, kernel_size=1)  # 클래스 점수를 얻는 1x1 컨볼루션

        def forward(self, x):
            """
            U-넷을 지나는 피드포워드.

            인수:
                x: 꼴이 (batch_size, 3, H, W)인 들임 텐서

            반환값:
                꼴이 (batch_size, num_classes, H, W)인 내놓는 텐서
            """
            # 건너뛰는 이음을 갈무리한 인코더
            enc1 = self.enc1(x)           # 64채널
            x = self.pool1(enc1)

            enc2 = self.enc2(x)           # 128채널
            x = self.pool2(enc2)

            enc3 = self.enc3(x)           # 256채널
            x = self.pool3(enc3)

            enc4 = self.enc4(x)           # 512채널
            x = self.pool4(enc4)

            # 병목
            x = self.bottleneck(x)        # 1024채널

            # 건너뛰는 이음을 갖춘 디코더
            x = self.upconv4(x)           # 키우기
            x = torch.cat([x, enc4], dim=1)  # 건너뛰는 이음 이어 붙이기
            x = self.dec4(x)

            x = self.upconv3(x)
            x = torch.cat([x, enc3], dim=1)
            x = self.dec3(x)

            x = self.upconv2(x)
            x = torch.cat([x, enc2], dim=1)
            x = self.dec2(x)

            x = self.upconv1(x)
            x = torch.cat([x, enc1], dim=1)
            x = self.dec1(x)

            # 최종 출력
            x = self.out(x)

            return x


    # 모델 생성
    print("\nCreating U-Net model...")
    model = UNet(in_channels=3, num_classes=2)  # 두 클래스: 뒷바탕(0), 앞바탕(1)
    model = model.to(device)

    # 파라메터 개수 세기
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ============================================================================
    # 4단계: 로스 함수와 가장 좋게 하개
    # ============================================================================
    """
    두 분류에는 로짓을 쓴 두 클래스 엇갈린 엔트로피 로스를 쓴다.
    이는 시그모이드 활성화와 두 클래스 엇갈린 엔트로피를 한 함수로 아울러
    수치로 든든하게 만든다.

    로스는 화소마다 셈한다:
    로스 = -[y*log(p) + (1-y)*log(1-p)]
    여기서 y은 참값(0 또는 1), p은 어림 확률
    """

    criterion = nn.CrossEntropyLoss()  # 여러 클래스용(클래스가 둘뿐이어도)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 러닝레이트 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    print("\nTraining configuration:")
    print(f"Loss function: CrossEntropyLoss")
    print(f"Optimizer: Adam (lr=0.001)")
    print(f"Scheduler: ReduceLROnPlateau (monitor IoU)")

    # ============================================================================
    # 5단계: 밸류 잣대
    # ============================================================================
    """
    나누기에는 자카드 지수라고도 하는
    겹침 비(IoU)를 쓴다. 어림과 참값 사이의 겹침을 잰다.

    겹침 비 = |A ∩ B| / |A ∪ B|
        = TP / (TP + FP + FN)

    여기서 각 기호는 다음과 같다.
    - TP(참양성): 올바로 어림한 앞바탕 화소
    - FP(헛양성): 앞바탕으로 어림한 뒷바탕 화소
    - FN(헛음성): 뒷바탕으로 어림한 앞바탕 화소
    """

    def calculate_iou(pred, target, threshold=0.5):
        """
        두 분류의 겹침 비(IoU)를 셈한다.

        인수:
            pred: 꼴이 (batch, 2, H, W)인 어림 로짓
            target: 값이 {0, 1}이고 꼴이 (batch, 1, H, W)인 참값 마스크
            threshold: 확률을 두 클래스 어림으로 바꾸는 임계값

        반환값:
            iou: 배치에 걸친 평균 겹침 비
        """
        with torch.no_grad():
            # 어림한 클래스 얻기(0 또는 1)
            pred_class = torch.argmax(pred, dim=1)  # 꼴: (배치, H, W)
            target_class = target.squeeze(1).long()  # 꼴: (배치, H, W)

            # 앞바탕 클래스(클래스 1)의 교집합과 합집합 셈하기
            pred_fg = (pred_class == 1)
            target_fg = (target_class == 1)

            intersection = (pred_fg & target_fg).float().sum(dim=(1, 2))
            union = (pred_fg | target_fg).float().sum(dim=(1, 2))

            # 0으로 나누기를 피한다
            iou = (intersection + 1e-6) / (union + 1e-6)

            return iou.mean().item()


    def calculate_pixel_accuracy(pred, target):
        """화소마다의 정확도를 셈한다."""
        with torch.no_grad():
            pred_class = torch.argmax(pred, dim=1)
            target_class = target.squeeze(1).long()
            correct = (pred_class == target_class).float()
            accuracy = correct.mean()
            return accuracy.item()

    # ============================================================================
    # 6단계: 익히기 함수
    # ============================================================================

    def train_one_epoch(model, loader, criterion, optimizer, device):
        """모델을 한 세대 학습한다."""
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        running_accuracy = 0.0

        for batch_idx, (images, masks) in enumerate(loader):
            images = images.to(device)
            masks = masks.to(device)

            # 순전파
            optimizer.zero_grad()
            outputs = model(images)

            # 로스 계산
            # CrossEntropyLoss는 클래스 번호를 담은 (배치, H, W) 꼴의 목표를 바란다
            masks_class = masks.squeeze(1).long()
            loss = criterion(outputs, masks_class)

            # 역전파
            loss.backward()
            optimizer.step()

            # 지표를 계산한다
            iou = calculate_iou(outputs, masks)
            accuracy = calculate_pixel_accuracy(outputs, masks)

            # 통계
            running_loss += loss.item()
            running_iou += iou
            running_accuracy += accuracy

            # 진행 상황 출력
            if (batch_idx + 1) % 20 == 0:
                print(f'  Batch {batch_idx + 1}/{len(loader)}: '
                      f'Loss: {loss.item():.4f}, '
                      f'IoU: {iou:.4f}, '
                      f'Acc: {accuracy:.4f}')

        epoch_loss = running_loss / len(loader)
        epoch_iou = running_iou / len(loader)
        epoch_accuracy = running_accuracy / len(loader)

        return epoch_loss, epoch_iou, epoch_accuracy


    # ============================================================================
    # 7단계: 검증 함수
    # ============================================================================

    def validate(model, loader, criterion, device):
        """모델을 검증한다."""
        model.eval()
        running_loss = 0.0
        running_iou = 0.0
        running_accuracy = 0.0

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(device)
                masks = masks.to(device)

                # 순전파
                outputs = model(images)

                # 로스 계산
                masks_class = masks.squeeze(1).long()
                loss = criterion(outputs, masks_class)

                # 지표를 계산한다
                iou = calculate_iou(outputs, masks)
                accuracy = calculate_pixel_accuracy(outputs, masks)

                # 통계
                running_loss += loss.item()
                running_iou += iou
                running_accuracy += accuracy

        avg_loss = running_loss / len(loader)
        avg_iou = running_iou / len(loader)
        avg_accuracy = running_accuracy / len(loader)

        return avg_loss, avg_iou, avg_accuracy


    # ============================================================================
    # 8단계: 그려 보기 함수
    # ============================================================================

    def visualize_predictions(model, dataset, device, num_samples=3):
        """모델의 어림을 그려 본다."""
        model.eval()

        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))

        for i in range(num_samples):
            # 표본 하나 얻기
            image, mask = dataset[i]
            image_input = image.unsqueeze(0).to(device)

            # 예측한다
            with torch.no_grad():
                output = model(image_input)
                pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()

            # 그려 보려고 바꾸기
            image_np = image.permute(1, 2, 0).cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()

            # 그림
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')

        plt.tight_layout()
        plt.savefig('segmentation_results.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'segmentation_results.png'")
        plt.close()


    # ============================================================================
    # 9단계: 학습 고리
    # ============================================================================

    NUM_EPOCHS = 20

    print(f"\n{'='*70}")
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    print(f"{'='*70}\n")

    best_iou = 0.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 70)

        # 학습
        train_loss, train_iou, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 검증
        val_loss, val_iou, val_acc = validate(model, val_loader, criterion, device)

        # 결과 출력
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, IoU: {train_iou:.4f}, Acc: {train_acc:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, IoU: {val_iou:.4f}, Acc: {val_acc:.4f}")

        # 러닝레이트 스케줄링
        scheduler.step(val_iou)

        # 최고 성능 모델 저장
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print(f"  ✓ New best IoU! Model saved.")

        print()

    total_time = time.time() - start_time
    print(f"{'='*70}")
    print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best validation IoU: {best_iou:.4f}")
    print(f"{'='*70}\n")

    # ============================================================================
    # 10단계: 마지막 평가
    # ============================================================================

    # 가장 좋은 모델을 불러온다
    model.load_state_dict(torch.load('best_unet_model.pth'))

    print("Final Test Evaluation:")
    print("="*70)
    test_loss, test_iou, test_acc = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test IoU: {test_iou:.4f}")
    print(f"Test Pixel Accuracy: {test_acc:.4f}")

    # 어림 몇 개 그려 보기
    print("\nGenerating visualizations...")
    visualize_predictions(model, test_dataset, device, num_samples=5)

    # ============================================================================
    # 요약
    # ============================================================================

    print("\n" + "="*70)
    print("BASIC U-NET SEGMENTATION COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Implemented U-Net architecture from scratch")
    print("2. Used skip connections to preserve spatial information")
    print("3. Learned binary segmentation (background vs foreground)")
    print("4. Used IoU as the primary evaluation metric")
    print("5. Applied pixel-wise cross-entropy loss")
    print("\nImportant Concepts:")
    print("- Encoder-Decoder architecture")
    print("- Skip connections preserve fine details")
    print("- IoU measures segmentation quality better than accuracy")
    print("- Each pixel is classified independently")
    print("\nNext Steps:")
    print("- Try Example 2 to use pre-trained encoders")
    print("- Experiment with different loss functions (Dice loss)")
    print("- Try multi-class segmentation (3+ classes)")
    print("="*70)


    if __name__ == "__main__":
        pass
    ```


## 2. 논의

### 데이터셋과 목표 레이블

`SyntheticShapesDataset`은 생성 시 모든 이미지와 마스크를 메모리에 저장하고 `__getitem__`으로 이미지3×H×W와 마스크1×H×W를 반환한다. `forward`를 갖는 뉴럴 넷트웍이 아니라 Dataset이다. 각 이미지에는 원·사각형·삼각형 중 **하나**가 있고 모든 도형을 전경1로 취급한다. 모양별 세 클래스 분류나 인스턴스 구분이 아니다.

배경 색은 낮은 밝기, 도형 색은 높은 밝기 범위로 생성되어 단순 밝기 단서가 강하다. 높은 IoU가 나와도 다양한 실제 색·질감·가림·객체를 학습했다고 해석하지 않는다. 중심을 randint(64,image_size−64)에서 뽑으므로 image_size>128이 필요하다. 구조가64×64를 받을 수 있다는 사실과 현재 데이터 생성기가64×64를 지원한다는 것은 다르다. 일부 큰 도형은 영상 경계에서 잘릴 수도 있다.

`simple_augment`는 이미지와 마스크를 함께 좌우 반전하지만 생성한 train_dataset에 transform으로 전달하지 않아 실제 학습에는 적용되지 않는다. 함수 정의와 사용을 구분한다. 훈련·검증·테스트는 같은 합성 생성 규칙의 다른 표본이며 분포 변화에 대한 일반화 평가는 아니다.

### 두 클래스 교차 엔트로피

모델은 N×2×H×W의 배경·전경 로짓을 반환한다. 학습 루프는 float 마스크의 채널을 제거해 N×H×W의 long0/1로 바꾸고 `CrossEntropyLoss`를 사용한다. 코드 문자열에는 BCE 설명이 남아 있지만 실제 로스는 **두 클래스 softmax 교차 엔트로피**다. 한 로짓 sigmoid BCE의 형태와 혼용하지 않는다.

픽셀별 로스를 합·평균해도 각 픽셀의 예측이 통계적으로 독립인 것은 아니다. 넓은 수용영역의 피처와 공유 웨이트가 인접 픽셀의 출력에 함께 영향을 준다. 정답 마스크를 변환할 때는 정확한0/1을 보존해야 하며 연속 보간 뒤 long으로 절삭해 클래스를 만드는 방식은 다른 목표를 만들 수 있다.

### 구조와 공간 조건

폭64·128·256·512와1024 병목을 사용하고 네 번 풀링한 뒤 전치 컨볼루션과 스킵 연결로 복원한다.64×64 입력을 모델에 직접 주면 병목은1024×4×4, 최종은2×64×64다. 그러나 임의 홀수 공간 크기에 대한 crop·padding은 없어 스킵과 업샘플 형태가 맞지 않을 수 있다. 메모리는 큰 활성과 연결, BN 및 옵티마이저 상태까지 고려해야 한다.

### 실제 IoU 집계와 종료

`calculate_iou`는 argmax의 전경 클래스1을 사용하며 `threshold` 인수는 사용하지 않는다. 이미지별 IoU를 평균한 **전경 지표**로, 배경과 전경의 macro mIoU나 데이터 전체 교집합/합집합 비율과 다르다. smooth 때문에 양쪽 모두 빈 전경은1이 된다.

epoch에서는 배치별 평균을 다시 단순 평균하므로 마지막 작은 배치에 상대적으로 큰 웨이트가 생길 수 있다. 검증·테스트100개를 배치8로 나누면 이 차이가 발생한다. 픽셀 정확도도 전경이 작은 자료에서는 배경 성능에 지배될 수 있다.

검증 IoU가0보다 엄격히 커질 때만 체크포인트를 저장한다. 한 번도 개선되지 않으면 뒤의 고정 파일 로드가 실패하거나 이전 실행의 파일을 읽을 수 있다. scheduler는 러닝레이트를 바꾸지만 루프 자체에 조기 종료는 없다. 마지막 `if __name__` 안의 pass와 달리 생성·학습이 모듈 최상위에 있어 import도 작업을 시작한다. 이번 편집에서는 코드를 실행하지 않았다.

### 검증 범위

그림은 선택한 몇 표본의 마스크 비교이며 전체 평가·경계 품질·새 분포의 성능을 대신하지 않는다. 같은 생성 분포의 독립 테스트를 검증 선택과 구분하고, 실제 이미지로 확장할 때는 마스크 라벨 규약·배경 단서·데이터 분할을 다시 설계한다. 코드·출력·그림은 요청대로 유지했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
Dataset과 모델의 역할 및 배치4의 입력·목표·출력 형태를 구분하라.

</div>

??? success "연습문제 1 풀이"
    Dataset은 __getitem__으로 이미지(3,H,W), 마스크(1,H,W)를 반환한다. 배치4면 입력(4,3,H,W), 마스크(4,1,H,W), CE 목표는 채널을 제거한 long(4,H,W), 모델 로짓은(4,2,H,W)다. Dataset에 뉴럴 넷트웍 forward나 컨볼루션 레이어를 추적하는 문제는 맞지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64로 바꾸는 경우 모델과 합성 생성기의 조건을 따로 설명하라.

</div>

??? success "연습문제 2 풀이"
    모델의 네 풀링과 전치 컨볼루션은64×64에서 스킵 형태가 맞는다. 하지만 생성기의 randint(64,image_size−64)는64에서 유효하지 않다. 중심·크기 표집 범위도 해상도에 맞춰 설계해야 하며 첫 채널이나 fc를 고치는 문제는 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
보고되는 IoU와 어그멘테이션의 실제 동작을 설명하라.

</div>

??? success "연습문제 3 풀이"
    IoU는 argmax 뒤 전경1의 이미지별 비율 평균이며 threshold를 사용하지 않는다. 배치 평균의 평균은 마지막 배치 크기에 영향을 받는다. simple_augment는 정의만 되어 있고 train_dataset에 전달되지 않아 적용되지 않는다. 코드의 인수·주석보다 실제 호출 경로를 확인해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 ModuleList 골격이 SyntheticShapesDataset의 확장으로 적절한지 평가하라.

</div>

??? success "연습문제 4 풀이"

    적절하지 않다. Dataset은 표본 생성·조회 객체이고 학습 가능한 레이어의 목록이 아니다. YourBlock·dim도 정의되지 않았다. 깊이를 바꾸려면 UNet의 단계·채널·스킵과 대응 decoder를 설계하고, 데이터 변형은 __getitem__ 또는 transform에서 별도로 구현해야 한다. 기존 골격은 요청대로 보존했다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

합성 도형 예제의 실제 목표는 두 클래스 CE와 전경 IoU다. Dataset과 모델, 정의한 어그멘테이션과 실제 호출, 이미지·배치·데이터 전체의 지표 평균을 구분한다. 단순한 합성 단서에서의 성능을 실제 분할 능력으로 일반화하지 않는다.
