# 의료 이미지 분할

이 예제는 피부색 배경과 어두운 다각형으로 만든 합성 영상에서 이진 분할을 학습한다. 의료 영상을 연상시키는 모양과 대비를 사용하지만 실제 환자 영상이나 임상 검증 데이터는 아니다. 한 채널 U-Net, BCE와 soft Dice의 결합, 픽셀 단위 민감도·특이도 등의 계산을 설명하는 교육용 실험이다.

## 1. 코드

??? note "코드 (671줄)"

    ```python
    """
    보기 3: 의료 그림 나누기
    ======================================

    이 각본은 그 분야에 맞춘 의료 그림 나누기를 보여 준다
    피부 확대경 그림의 병터 나누기에 초점을 둔다.

    핵심 개념:
    - 의료 영상을 위한 다이스 손실
    - 극단적 갈래 치우침 다루기
    - 임상 잣대(다이스, 민감도, 특이도)
    - 의료 영상 앞손질
    - 제대로 된 검증 전략

    지은이: PyTorch Semantic Segmentation Tutorial
    날짜: 2025
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFilter
    import time

    # 난수 씨앗을 설정한다
    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================================
    # 1단계: 장치 설정
    # ============================================================================

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ============================================================================
    # 2단계: 인공 의료 자료 뭉치
    # ============================================================================
    """
    피부 병터의 피부 확대경 그림을 흉내낸 인공 자료 뭉치를 만든다.
    실전에서는 ISIC, DRIVE, BraTS 같은 실제 의료 자료 뭉치를 쓴다.

    의료 그림의 성질:
    - 극단적 갈래 치우침(병터가 뒷바탕보다 작다)
    - 들쭉날쭉한 꼴
    - 들쭉날쭉한 대비
    - 잡음과 찌꺼기
    """

    class SyntheticMedicalDataset(Dataset):
        """
        병터가 있는 피부 확대경 비슷한 인공 그림을 만든다.

        다음을 갖춘 피부 병터 나누기 일을 흉내낸다:
        - 들쭉날쭉한 병터 꼴
        - 들쭉날쭉한 크기(그림 넓이의 5~15%)
        - 잡음과 흐림(실제 같은 찌꺼기)
        - 갈래 치우침이 심하다
        """

        def __init__(self, num_samples=1000, image_size=256):
            self.num_samples = num_samples
            self.image_size = image_size

            print(f"Generating {num_samples} synthetic medical images...")
            self.samples = []
            for i in range(num_samples):
                img, mask = self._generate_sample()
                self.samples.append((img, mask))
                if (i + 1) % 200 == 0:
                    print(f"  Generated {i + 1}/{num_samples} images")

            # 갈래 분포 셈하기
            total_pixels = 0
            lesion_pixels = 0
            for _, mask in self.samples:
                mask_np = np.array(mask)
                total_pixels += mask_np.size
                lesion_pixels += np.sum(mask_np > 0)

            self.lesion_ratio = lesion_pixels / total_pixels
            print(f"\nDataset statistics:")
            print(f"  Lesion pixels: {self.lesion_ratio:.2%}")
            print(f"  Background pixels: {1-self.lesion_ratio:.2%}")
            print(f"  Class imbalance ratio: 1:{(1-self.lesion_ratio)/self.lesion_ratio:.1f}")

        def _generate_sample(self):
            """피부 확대경 그림을 닮은 그림-마스크 짝 하나를 만든다."""
            # 살빛 뒷바탕 만들기
            skin_color = (
                np.random.randint(200, 240),  # R
                np.random.randint(160, 200),  # G
                np.random.randint(140, 180)   # B
            )
            img = Image.new('RGB', (self.image_size, self.image_size), skin_color)
            mask = Image.new('L', (self.image_size, self.image_size), 0)

            draw_img = ImageDraw.Draw(img)
            draw_mask = ImageDraw.Draw(mask)

            # 병터 빛깔(어둡고 갈색기)
            lesion_color = (
                np.random.randint(80, 140),
                np.random.randint(60, 100),
                np.random.randint(40, 80)
            )

            # 들쭉날쭉한 병터 꼴 만들기
            center_x = np.random.randint(64, self.image_size - 64)
            center_y = np.random.randint(64, self.image_size - 64)

            # 여러 번 뒤튼 들쭉날쭉한 타원
            num_points = 20
            angles = np.linspace(0, 2*np.pi, num_points)
            base_radius = np.random.randint(30, 50)

            # 들쭉날쭉한 테두리 만들기
            radii = base_radius + np.random.randint(-15, 15, num_points)
            points = [
                (
                    center_x + int(radii[i] * np.cos(angles[i])),
                    center_y + int(radii[i] * np.sin(angles[i]))
                )
                for i in range(num_points)
            ]

            # 병터 그리기
            draw_img.polygon(points, fill=lesion_color)
            draw_mask.polygon(points, fill=255)

            # 실제 같은 찌꺼기 더하기
            # 1. 살짝 흐리기(사진기/살결 흉내)
            img = img.filter(ImageFilter.GaussianBlur(radius=1))

            # 2. 그림에 잡음 더하기
            img_array = np.array(img).astype(np.float32)
            noise = np.random.normal(0, 5, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_array)

            return img, mask

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            img, mask = self.samples[idx]

            # 텐서로 바꾼다
            img_array = np.array(img).astype(np.float32) / 255.0
            mask_array = np.array(mask).astype(np.float32) / 255.0

            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)     # (1, H, W)

            return img_tensor, mask_tensor


    # 환자 단위로 나눈 자료 뭉치 만들기
    # 중요: 실제 의료 인공지능에서는 그림이 아니라 환자 단위로 나눈다!
    print("Creating medical imaging datasets...\n")
    train_dataset = SyntheticMedicalDataset(num_samples=800, image_size=256)
    val_dataset = SyntheticMedicalDataset(num_samples=100, image_size=256)
    test_dataset = SyntheticMedicalDataset(num_samples=100, image_size=256)

    # 데이터 로더 생성
    BATCH_SIZE = 16
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ============================================================================
    # 3단계: 의료 영상을 위한 U-넷
    # ============================================================================
    """
    U-넷은 의료 그림 나누기의 으뜸 잣대이다.
    본디 생의학 그림 나누기를 위해 꾸며졌다.
    """

    class DoubleConv(nn.Module):
        """겹 누비기 덩이."""
        def __init__(self, in_channels, out_channels):
            super().__init__()
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


    class MedicalUNet(nn.Module):
        """
        의료 영상에 맞게 다듬은 U-넷.
        특징을 더 잘 뽑으려 기본 U-넷보다 조금 깊다.
        """
        def __init__(self, in_channels=3, num_classes=1):
            super().__init__()

            # 인코더
            self.enc1 = DoubleConv(in_channels, 64)
            self.pool1 = nn.MaxPool2d(2)

            self.enc2 = DoubleConv(64, 128)
            self.pool2 = nn.MaxPool2d(2)

            self.enc3 = DoubleConv(128, 256)
            self.pool3 = nn.MaxPool2d(2)

            self.enc4 = DoubleConv(256, 512)
            self.pool4 = nn.MaxPool2d(2)

            # 병목
            self.bottleneck = DoubleConv(512, 1024)

            # 디코더
            self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
            self.dec4 = DoubleConv(1024, 512)

            self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec3 = DoubleConv(512, 256)

            self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec2 = DoubleConv(256, 128)

            self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec1 = DoubleConv(128, 64)

            # 마지막 내놓음(활성화 없음, BCEWithLogitsLoss를 쓴다)
            self.out = nn.Conv2d(64, num_classes, kernel_size=1)

        def forward(self, x):
            # 인코더
            enc1 = self.enc1(x)
            x = self.pool1(enc1)

            enc2 = self.enc2(x)
            x = self.pool2(enc2)

            enc3 = self.enc3(x)
            x = self.pool3(enc3)

            enc4 = self.enc4(x)
            x = self.pool4(enc4)

            # 병목
            x = self.bottleneck(x)

            # 건너뛰는 이음을 갖춘 디코더
            x = self.upconv4(x)
            x = torch.cat([x, enc4], dim=1)
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

            # 마지막 내놓음(로짓)
            x = self.out(x)

            return x


    model = MedicalUNet(in_channels=3, num_classes=1)  # 두 갈래: 병터와 뒷바탕
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: Medical U-Net")
    print(f"Total parameters: {total_params:,}")

    # ============================================================================
    # 4단계: 의료 영상을 위한 다이스 손실
    # ============================================================================
    """
    다이스 손실은 의료 그림 나누기의 표준 손실이다.

    엇갈린 엔트로피보다 나은 점:
    1. 값매김 잣대인 다이스 계수를 곧바로 가장 좋게 한다
    2. 갈래 치우침을 자연스레 다룬다
    3. 낱낱의 화소가 아니라 겹침에 초점을 둔다
    4. 작은 물체에 더 낫다

    식:
        Dice = 2 * |A ∩ B| / (|A| + |B|)
        다이스 잃음 = 1 - 다이스
    """

    class DiceLoss(nn.Module):
        """
        두 갈래 나누기를 위한 다이스 손실.
        """
        def __init__(self, smooth=1e-6):
            """
            인수:
                smooth: 0으로 나누지 않도록 하는 부드럽게 하기 상수
            """
            super().__init__()
            self.smooth = smooth

        def forward(self, predictions, targets):
            """
            인수:
                predictions: 꼴이 (batch, 1, H, W)인 로짓
                targets: 값이 [0, 1]이고 꼴이 (batch, 1, H, W)인 참값

            반환값:
                dice_loss: 홑값 손실
            """
            # 로짓을 확률로 바꾸려 시그모이드 쓰기
            predictions = torch.sigmoid(predictions)

            # 어림과 목표 펴기
            predictions = predictions.view(-1)
            targets = targets.view(-1)

            # 교집합과 합집합 계산
            intersection = (predictions * targets).sum()
            union = predictions.sum() + targets.sum()

            # 다이스 계수 셈하기
            dice = (2. * intersection + self.smooth) / (union + self.smooth)

            # 다이스 손실은 1 - 다이스
            return 1 - dice


    class CombinedLoss(nn.Module):
        """
        다이스와 두 갈래 엇갈린 엔트로피를 아우른 손실.

        두 갈래 엇갈린 엔트로피는 화소 정확도를, 다이스는 겹침을 돕는다.
        이 아우름이 실전에서 흔히 가장 잘 된다.
        """
        def __init__(self, alpha=0.5, beta=0.5):
            """
            인수:
                alpha: 두 갈래 엇갈린 엔트로피 손실의 무게
                beta: 다이스 손실의 무게
            """
            super().__init__()
            self.alpha = alpha
            self.beta = beta
            self.bce = nn.BCEWithLogitsLoss()
            self.dice = DiceLoss()

        def forward(self, predictions, targets):
            bce_loss = self.bce(predictions, targets)
            dice_loss = self.dice(predictions, targets)
            return self.alpha * bce_loss + self.beta * dice_loss


    # 가장 좋은 결과를 위해 아우른 손실 쓰기
    criterion = CombinedLoss(alpha=0.5, beta=0.5)

    print("\nLoss function: Combined Dice + BCE Loss")
    print("  Dice Loss: Optimizes overlap (better for small objects)")
    print("  BCE Loss: Pixel-wise accuracy")

    # ============================================================================
    # 5단계: 의료 잣대
    # ============================================================================
    """
    의료 나누기는 일반 나누기와 다른 잣대를 쓴다:
    - 다이스 계수: 으뜸 잣대
    - 민감도(재현율): 병터 화소를 얼마나 찾아냈는가?
    - 특이도: 뒷바탕 화소 가운데 맞은 것이 얼마인가?
    - 정밀도: 병터로 어림한 화소 가운데 맞은 것이 얼마인가?
    """

    def calculate_dice(pred, target, threshold=0.5, smooth=1e-6):
        """다이스 계수를 셈한다."""
        with torch.no_grad():
            pred = torch.sigmoid(pred)
            pred = (pred > threshold).float()

            pred = pred.view(-1)
            target = target.view(-1)

            intersection = (pred * target).sum()
            union = pred.sum() + target.sum()

            dice = (2. * intersection + smooth) / (union + smooth)
            return dice.item()


    def calculate_medical_metrics(pred, target, threshold=0.5):
        """
        두루 살피는 의료 잣대를 셈한다.

        반환값:
            다이스, 민감도, 특이도, 정밀도를 담은 사전
        """
        with torch.no_grad():
            pred = torch.sigmoid(pred)
            pred = (pred > threshold).float()

            pred = pred.view(-1)
            target = target.view(-1)

            # 헷갈림 행렬 요소 셈하기
            tp = ((pred == 1) & (target == 1)).float().sum()  # 참양성
            tn = ((pred == 0) & (target == 0)).float().sum()  # 참음성
            fp = ((pred == 1) & (target == 0)).float().sum()  # 헛양성
            fn = ((pred == 0) & (target == 1)).float().sum()  # 헛음성

            # 지표를 계산한다
            smooth = 1e-6

            dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
            sensitivity = (tp + smooth) / (tp + fn + smooth)  # 재현율, 참양성 비율
            specificity = (tn + smooth) / (tn + fp + smooth)  # 참음성 비율
            precision = (tp + smooth) / (tp + fp + smooth)    # 양성 예측도

            return {
                'dice': dice.item(),
                'sensitivity': sensitivity.item(),
                'specificity': specificity.item(),
                'precision': precision.item()
            }


    # ============================================================================
    # 6단계: 가장 좋게 하개
    # ============================================================================

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )

    # ============================================================================
    # 7단계: 학습 함수
    # ============================================================================

    def train_one_epoch(model, loader, criterion, optimizer, device):
        """한 에폭을 학습한다."""
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            # 순전파
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            # 역전파
            loss.backward()
            optimizer.step()

            # 다이스 셈하기
            dice = calculate_dice(outputs, masks)

            # 통계
            running_loss += loss.item()
            running_dice += dice

        epoch_loss = running_loss / len(loader)
        epoch_dice = running_dice / len(loader)

        return epoch_loss, epoch_dice


    # ============================================================================
    # 8단계: 검증 함수
    # ============================================================================

    def validate(model, loader, criterion, device):
        """두루 살피는 의료 잣대로 검증한다."""
        model.eval()
        running_loss = 0.0
        metrics_sum = {'dice': 0.0, 'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0}

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(device)
                masks = masks.to(device)

                # 순전파
                outputs = model(images)
                loss = criterion(outputs, masks)

                # 지표를 계산한다
                metrics = calculate_medical_metrics(outputs, masks)

                # 통계
                running_loss += loss.item()
                for key in metrics_sum:
                    metrics_sum[key] += metrics[key]

        # 잣대 고루내기
        avg_loss = running_loss / len(loader)
        avg_metrics = {key: val / len(loader) for key, val in metrics_sum.items()}

        return avg_loss, avg_metrics


    # ============================================================================
    # 9단계: 그려 보기
    # ============================================================================

    def visualize_medical_predictions(model, dataset, device, num_samples=3):
        """의료용 겹쳐 놓기 방식으로 어림을 그려 본다."""
        model.eval()

        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

        for i in range(num_samples):
            image, mask = dataset[np.random.randint(len(dataset))]
            image_input = image.unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image_input)
                pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
                pred_mask = (pred_prob > 0.5).astype(np.float32)

            # 그려 보려고 바꾸기
            image_np = image.permute(1, 2, 0).cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()

            # 겹쳐 놓기 만들기
            overlay = image_np.copy()
            # 참값은 초록으로
            overlay[mask_np > 0.5] = [0, 1, 0]

            overlay_pred = image_np.copy()
            # 어림은 빨강으로
            overlay_pred[pred_mask > 0.5] = [1, 0, 0]

            # 그림
            axes[i, 0].imshow(image_np)
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')

            axes[i, 1].imshow(mask_np, cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred_prob, cmap='hot')
            axes[i, 2].set_title('Prediction Probability')
            axes[i, 2].axis('off')

            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title('Overlay (GT=Green)')
            axes[i, 3].axis('off')

        plt.tight_layout()
        plt.savefig('medical_segmentation_results.png', dpi=150, bbox_inches='tight')
        print("Visualization saved as 'medical_segmentation_results.png'")
        plt.close()


    # ============================================================================
    # 10단계: 익히기 되풀이
    # ============================================================================

    NUM_EPOCHS = 30

    print(f"\n{'='*70}")
    print(f"Starting medical segmentation training for {NUM_EPOCHS} epochs...")
    print(f"{'='*70}\n")

    best_dice = 0.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 70)

        # 학습
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 검증
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        # 결과 출력
        print(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Dice: {val_metrics['dice']:.4f}, "
              f"Sens: {val_metrics['sensitivity']:.4f}, Spec: {val_metrics['specificity']:.4f}")

        # 학습률 스케줄링
        scheduler.step(val_metrics['dice'])

        # 최고 성능 모델 저장
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(model.state_dict(), 'best_medical_segmentation.pth')
            print("✓ New best Dice! Model saved.")

        print()

    total_time = time.time() - start_time
    print(f"{'='*70}")
    print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"{'='*70}\n")

    # ============================================================================
    # 11단계: 마지막 시험 값매김
    # ============================================================================

    model.load_state_dict(torch.load('best_medical_segmentation.pth'))

    print("Final Test Evaluation (Clinical Metrics):")
    print("="*70)
    test_loss, test_metrics = validate(model, test_loader, criterion, device)

    print(f"Test Dice Coefficient: {test_metrics['dice']:.4f}")
    print(f"Test Sensitivity (Recall): {test_metrics['sensitivity']:.4f}")
    print(f"Test Specificity: {test_metrics['specificity']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")

    print("\nClinical Interpretation:")
    print(f"- Dice {test_metrics['dice']:.2%}: Overall segmentation quality")
    print(f"- Sensitivity {test_metrics['sensitivity']:.2%}: Detected {test_metrics['sensitivity']:.1%} of lesion pixels")
    print(f"- Specificity {test_metrics['specificity']:.2%}: Correctly identified {test_metrics['specificity']:.1%} of healthy skin")

    # 어림 그려 보기
    print("\nGenerating visualizations...")
    visualize_medical_predictions(model, test_dataset, device, num_samples=5)

    # ============================================================================
    # 요약
    # ============================================================================

    print("\n" + "="*70)
    print("MEDICAL IMAGE SEGMENTATION COMPLETE!")
    print("="*70)
    print("\nKey Takeaways:")
    print("1. Used Dice Loss to handle extreme class imbalance")
    print("2. Evaluated with clinical metrics (Dice, Sensitivity, Specificity)")
    print("3. Applied U-Net architecture (gold standard for medical imaging)")
    print("4. Handled small objects better than cross-entropy")
    print("5. Combined Dice + BCE for optimal results")
    print("\nMedical AI Best Practices:")
    print("✓ Use Dice loss for imbalanced medical data")
    print("✓ Report Sensitivity and Specificity, not just accuracy")
    print("✓ Split data by patient, not by image")
    print("✓ Visualize predictions for clinical validation")
    print("✓ Consider false negative cost (missing lesions is dangerous)")
    print("\nNext Steps:")
    print("- Try Example 4 for advanced techniques")
    print("- Apply to real medical datasets (ISIC, DRIVE, BraTS)")
    print("- Implement uncertainty quantification")
    print("- Add 3D volumetric segmentation (CT/MRI)")
    print("="*70)


    if __name__ == "__main__":
        pass
    ```


## 2. 논의

### 합성 영상의 범위

`SyntheticMedicalDataset`은 밝은 배경 위 어두운 불규칙 다각형 하나를 그리고 이미지에 blur·노이즈를 더한다. 마스크는0/255로 그린 뒤0/1 float로 반환한다. 이미지와 마스크의 경계는 blur 때문에 시각적으로 완전히 일치하지 않을 수 있지만 정답은 원래 다각형 영역이다. 이 데이터는 실제 피부 병변의 색·구조·촬영 다양성을 모델링한 것이 아니다.

모든 표본을 메모리에 저장하고 이미지3×H×W, 목표1×H×W를 반환하는 Dataset이며 신경망의 forward를 정의하는 객체가 아니다. 중심은64부터image_size−64 미만에서 표집하므로 image_size>128이 필요하다. 0개 표본이나 전경이 없는 자료에서는 `lesion_ratio`와 비율 출력의 분모도 유효하지 않을 수 있다.

훈련·검증·테스트는 같은 생성 규칙의 다른 표본이다. 환자·기관·시점의 차이나 실제 진단 대상의 일반화는 평가하지 않는다. 실제 자료를 다룬다면 같은 환자의 관련 영상이 서로 다른 평가 분할로 섞이지 않는 등 데이터 단위에 맞는 독립성을 설계해야 한다. 여기에는 환자 ID 자체가 없다.

### 한 채널 로짓과 결합 손실

`MedicalUNet`은 네 번 풀링하고 스킵을 연결하는 패딩 U-Net이며 출력은 N×1×H×W 로짓이다. BCEWithLogitsLoss는 같은 형태의 float0/1 목표를 사용한다. DiceLoss는 내부에서 sigmoid를 취하고 배치 전체를 펼쳐 다음 soft Dice를 계산한다.

$$L_{Dice}=1-\frac{2\sum_i p_i y_i+\epsilon}{\sum_i p_i+\sum_i y_i+\epsilon}.$$

전체 손실은0.5L_BCE+0.5L_Dice다. 두 목적의 단위·기울기 크기가 같다는 뜻은 아니며 가중치 선택에 따라 균형이 달라진다. soft Dice는 임계화한 평가 Dice의 미분 가능한 대리 목적이다. BCE는 정확도 자체가 아니라 픽셀별 로그 손실이며, Dice가 모든 작은 객체·불균형 데이터에서 더 낫거나 결합이 항상 최적이라고 단정할 수 없다.

### 픽셀 단위 지표

평가에서는 sigmoid>0.5로 전경을 만든 뒤 Dice와 다음 비율을 계산한다.

$$\mathrm{민감도}=\frac{TP}{TP+FN},\quad\mathrm{특이도}=\frac{TN}{TN+FP},\quad\mathrm{정밀도}=\frac{TP}{TP+FP}.$$

이 값의 단위는 **병변 픽셀**이다. 환자·병변 객체 단위의 탐지 민감도나 진단 정확도와 같지 않다. 출력의 ‘clinical’이라는 표시는 이런 단위 차이를 없애지 않으며 임상적 유효성을 입증한 결과도 아니다. 배경이 많으면 특이도가 높아도 작은 병변을 놓칠 수 있다.

코드는 분자·분모에 smooth를 더해 분모0인 경우에도1을 반환할 수 있다. 예를 들어 정답 양성 픽셀이 없으면 민감도는 원래 정의되지 않는데1이 된다. 정밀도·빈 마스크도 이 규칙의 영향을 받으므로 결측·제외 또는 별도 보고 정책을 명시해야 한다.

지표는 배치 전체 픽셀을 합쳐 계산한 비율을 다시 배치별로 단순 평균한다. 데이터 전체 혼동행렬의 비율이나 이미지별 macro 평균과 일반적으로 다르다. 검증·테스트100개를 배치16으로 나눠 마지막4개도 다른 배치와 같은 비중을 받는 점을 구분한다.

### 구조·체크포인트·시각화

RGB64×64 입력을 모델에 직접 주면 인코더 폭64·128·256·512를 거쳐 병목1024×4×4, 최종1×64×64가 된다. 데이터 생성기의 고정 중심 범위는 이 크기를 지원하지 않는다. 임의 홀수 크기에 대한 스킵 정렬도 없다.

검증 Dice가0보다 엄격히 개선될 때만 고정 경로에 저장하며 한 번도 저장하지 않으면 마지막 로드가 실패하거나 이전 파일을 읽을 수 있다. scheduler는 학습률을 조절하고 기본 루프에는 조기 종료가 없다. 생성·학습은 모듈 최상위에 있어 마지막 pass guard와 무관하게 import에서 시작할 수 있다. 코드를 실행하지 않고 정적으로 검토했다.

시각화는 확률과 정답을 보여 주지만 `overlay_pred`는 계산만 하고 그림에는 사용하지 않는다. 마지막 패널은 녹색 GT 오버레이이며 예측 오버레이가 아니다. 몇 장의 합성 영상과 픽셀 지표를 실제 임상 성능이나 최적 손실의 근거로 해석하지 않는다. 코드·출력·그림은 그대로 보존했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
배치4에서 Dataset 반환값과 MedicalUNet의 로짓 형태를 설명하라.

</div>

??? success "연습문제 1 풀이"
    영상은(4,3,H,W), 마스크는(4,1,H,W) float0/1이다. 모델은 같은 공간 크기의(4,1,H,W) 로짓을 반환하고 BCE는 이를 직접 사용한다. sigmoid는 DiceLoss 내부와 평가에서 적용한다. Dataset은 신경망이 아니라 표본을 반환하는 객체다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
64×64 모델 입력과 합성 생성기의 크기 조건을 구분하라.

</div>

??? success "연습문제 2 풀이"
    U-Net은64×64에서 네 풀링 뒤4×4 병목을 거쳐 원 크기로 복원할 수 있다. 그러나 중심 randint(64,image_size−64)는 image_size=64에서 유효하지 않다. 생성 범위를 크기에 맞춰 설계해야 하며 RGB 모델의 fc를 바꾸는 문제는 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
코드의 민감도가 환자 진단 민감도와 같은가? 분모가0이면 무엇이 일어나는가?

</div>

??? success "연습문제 3 풀이"
    아니다. TP·FN은 합성 병변의 픽셀 수다. 환자·객체 단위 지표는 별도의 판단 단위와 정답이 필요하다. 양성 픽셀이 없으면 smooth로 계산한 민감도는1이 될 수 있지만 원래 비율은0/0으로 정의되지 않는다. 빈 경우와 집계 규칙을 명시해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 반복 골격이 Dataset에 필요한 확장인가? 손실·평가 실험을 어떻게 구분해야 하는가?

</div>

??? success "연습문제 4 풀이"

    Dataset에 ModuleList로 신경망 깊이를 넣는 것은 역할이 맞지 않고 YourBlock·dim도 정의되지 않았다. 모델 깊이는 MedicalUNet의 채널·스킵·decoder에서 설계한다. 손실 비교는 같은 데이터와 초기·학습 예산에서 BCE·softDice·결합을 구분하고, 임계화한 픽셀·객체·환자 지표를 혼동하지 않도록 평가 단위를 명시한다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

이 예제는 합성 병변 모양에서 한 채널 BCE·soft Dice 학습과 픽셀 지표를 설명한다. 실제 환자 데이터·진단 단위와 다르며 빈 분모·배치 평균·임계값의 규칙을 명시해야 한다. 합성 결과나 시각화를 임상 검증으로 해석하지 않는다.
