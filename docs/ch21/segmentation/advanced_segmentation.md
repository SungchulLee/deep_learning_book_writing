# 이미지 분할 심화

이 예제는 합성 도형의 이진 분할에 CBAM, 여러 손실의 결합, 배치별 해상도 변경, 혼합 정밀도와 테스트 시 증강(TTA)을 연결한다. 각 방법은 비용과 목적을 다르게 바꾸며 함께 넣었다고 성능이 항상 좋아지는 것은 아니다. 실제로 켜지는 경로와 비교 실험의 기준을 코드에 맞춰 해석한다.

## 1. 코드

??? note "코드 (687줄)"

    ```python
    """
    보기 4: 앞선 뜻 나누기 재주
    =====================================================

    이 각본은 뜻 나누기의 가장 앞선 재주를 보여 준다:
    - 어텐션 구조(CBAM)
    - 앞선 손실 함수(초점 + 다이스 + 테두리)
    - 여러 잣수 익히기
    - 시험 때 불리기
    - 뒷손질
    - 섞인 정밀도 익히기

    이 재주들은 어려운 일에서 성능을 크게 올릴 수 있다.

    지은이: PyTorch Semantic Segmentation Tutorial
    날짜: 2025
    """

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from torch.cuda.amp import autocast, GradScaler
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    import time
    from scipy.ndimage import binary_erosion, binary_dilation

    # 난수 씨앗을 설정한다
    torch.manual_seed(42)
    np.random.seed(42)

    # ============================================================================
    # 설정
    # ============================================================================

    USE_MIXED_PRECISION = True
    USE_MULTI_SCALE = True
    USE_TTA = True  # 시험 때 불리기

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if USE_MIXED_PRECISION and not torch.cuda.is_available():
        print("⚠ Mixed precision requires CUDA. Disabling.")
        USE_MIXED_PRECISION = False

    print(f"\nConfiguration:")
    print(f"  Mixed Precision: {USE_MIXED_PRECISION}")
    print(f"  Multi-scale Training: {USE_MULTI_SCALE}")
    print(f"  Test-time Augmentation: {USE_TTA}\n")

    # ============================================================================
    # 1단계: 어텐션 단원
    # ============================================================================
    """
    어텐션 구조는 모델이 중요한 자리와 특징에 초점을 두게 돕는다.
    CBAM(누비기 덩이 어텐션 단원)은 어텐션을
    채널 차원과 자리 차원 모두에 준다.
    """

    class ChannelAttention(nn.Module):
        """
        채널 어텐션 단원.
        어떤 채널(특징)이 중요한지 배운다.
        """
        def __init__(self, in_channels, reduction=16):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

            self.fc = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
            )
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            out = avg_out + max_out
            return self.sigmoid(out)


    class SpatialAttention(nn.Module):
        """
        자리 어텐션 단원.
        어떤 자리가 중요한지 배운다.
        """
        def __init__(self, kernel_size=7):
            super().__init__()
            self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            x = torch.cat([avg_out, max_out], dim=1)
            x = self.conv(x)
            return self.sigmoid(x)


    class CBAM(nn.Module):
        """
        누비기 덩이 어텐션 단원.
        채널 어텐션과 자리 어텐션을 아우른다.
        """
        def __init__(self, in_channels, reduction=16):
            super().__init__()
            self.channel_attention = ChannelAttention(in_channels, reduction)
            self.spatial_attention = SpatialAttention()

        def forward(self, x):
            # 채널 어텐션 쓰기
            x = x * self.channel_attention(x)
            # 자리 어텐션 쓰기
            x = x * self.spatial_attention(x)
            return x


    # ============================================================================
    # 2단계: 어텐션을 갖춘 앞선 U-넷
    # ============================================================================

    class AttentionDoubleConv(nn.Module):
        """어텐션을 갖춘 겹 누비기 덩이."""
        def __init__(self, in_channels, out_channels, use_attention=True):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.attention = CBAM(out_channels) if use_attention else nn.Identity()

        def forward(self, x):
            x = self.conv(x)
            x = self.attention(x)
            return x


    class AdvancedUNet(nn.Module):
        """
        다음을 갖춘 앞선 U-넷:
        - 어텐션 단원(CBAM)
        - 깊은 이끎(있어도 되고 없어도 됨)
        - 잔차 이음(있어도 되고 없어도 됨)
        """
        def __init__(self, in_channels=3, num_classes=1, use_attention=True):
            super().__init__()

            # 인코더
            self.enc1 = AttentionDoubleConv(in_channels, 64, use_attention)
            self.pool1 = nn.MaxPool2d(2)

            self.enc2 = AttentionDoubleConv(64, 128, use_attention)
            self.pool2 = nn.MaxPool2d(2)

            self.enc3 = AttentionDoubleConv(128, 256, use_attention)
            self.pool3 = nn.MaxPool2d(2)

            self.enc4 = AttentionDoubleConv(256, 512, use_attention)
            self.pool4 = nn.MaxPool2d(2)

            # 어텐션을 갖춘 병목
            self.bottleneck = AttentionDoubleConv(512, 1024, use_attention=True)

            # 디코더
            self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dec4 = AttentionDoubleConv(1024, 512, use_attention)

            self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec3 = AttentionDoubleConv(512, 256, use_attention)

            self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec2 = AttentionDoubleConv(256, 128, use_attention)

            self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = AttentionDoubleConv(128, 64, use_attention)

            # 최종 출력
            self.out = nn.Conv2d(64, num_classes, 1)

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

            # 디코더
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

            x = self.out(x)
            return x


    model = AdvancedUNet(in_channels=3, num_classes=1, use_attention=True)
    model = model.to(device)

    print(f"Model: Advanced U-Net with CBAM Attention")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # ============================================================================
    # 3단계: 앞선 손실 함수
    # ============================================================================
    """
    앞선 손실 함수는 나누기의 좋음을 크게 올릴 수 있다.
    가장 좋은 결과를 위해 여러 손실을 아우른다.
    """

    class FocalLoss(nn.Module):
        """
        갈래 치우침을 다루는 초점 손실.
        쉬운 보기의 무게를 낮추고 어려운 음성에 초점을 둔다.
        """
        def __init__(self, alpha=0.25, gamma=2.0):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            pt = torch.exp(-BCE_loss)  # 참 갈래의 확률
            focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
            return focal_loss.mean()


    class DiceLoss(nn.Module):
        """나누기를 위한 다이스 손실."""
        def __init__(self, smooth=1e-6):
            super().__init__()
            self.smooth = smooth

        def forward(self, inputs, targets):
            inputs = torch.sigmoid(inputs)
            inputs = inputs.view(-1)
            targets = targets.view(-1)

            intersection = (inputs * targets).sum()
            dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
            return 1 - dice


    class BoundaryLoss(nn.Module):
        """
        테두리 자리를 도드라지게 하는 손실.
        테두리에 가까운 화소에 더 큰 무게를 준다.
        """
        def __init__(self, theta=5):
            super().__init__()
            self.theta = theta  # 테두리 너비를 다스린다

        def forward(self, inputs, targets):
            # 테두리 무게 셈하기
            # 테두리에 가까운 화소에 더 큰 무게
            targets_np = targets.cpu().numpy()
            boundary_weights = np.zeros_like(targets_np)

            for i in range(targets_np.shape[0]):
                mask = targets_np[i, 0]
                # 테두리를 찾으려 깎고 부풀리기
                eroded = binary_erosion(mask > 0.5, iterations=2)
                dilated = binary_dilation(mask > 0.5, iterations=2)
                boundary = dilated & ~eroded
                boundary_weights[i, 0] = boundary.astype(np.float32) * self.theta + 1.0

            boundary_weights = torch.from_numpy(boundary_weights).to(inputs.device)

            # 무게를 준 두 갈래 엇갈린 엔트로피
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            weighted_bce = bce * boundary_weights
            return weighted_bce.mean()


    class CombinedLoss(nn.Module):
        """
        아우른 손실: 초점 + 다이스 + 테두리
        여러 손실의 센 점을 써먹는다.
        """
        def __init__(self, alpha=0.3, beta=0.4, gamma=0.3):
            super().__init__()
            self.alpha = alpha  # 초점 손실의 무게
            self.beta = beta    # 다이스 손실의 무게
            self.gamma = gamma  # 테두리 손실의 무게

            self.focal = FocalLoss()
            self.dice = DiceLoss()
            self.boundary = BoundaryLoss()

        def forward(self, inputs, targets):
            focal_loss = self.focal(inputs, targets)
            dice_loss = self.dice(inputs, targets)
            boundary_loss = self.boundary(inputs, targets)

            total = (self.alpha * focal_loss + 
                    self.beta * dice_loss + 
                    self.gamma * boundary_loss)
            return total


    criterion = CombinedLoss()
    print("Loss: Combined (Focal + Dice + Boundary)")

    # ============================================================================
    # 4단계: 여러 잣수를 갖춘 인공 자료 뭉치
    # ============================================================================

    class MultiScaleDataset(Dataset):
        """여러 잣수의 그림을 돌려주는 자료 뭉치."""
        def __init__(self, num_samples=800, scales=[256, 384] if USE_MULTI_SCALE else [256]):
            self.num_samples = num_samples
            self.scales = scales
            self.samples = []

            print(f"Generating {num_samples} samples at scales {scales}...")
            for i in range(num_samples):
                # 가장 큰 잣수에서 만들기
                img, mask = self._generate_sample(max(scales))
                self.samples.append((img, mask))

        def _generate_sample(self, size):
            """인공 나누기 표본을 만든다."""
            img = Image.new('RGB', (size, size), 
                           tuple(np.random.randint(100, 200, 3).tolist()))
            mask = Image.new('L', (size, size), 0)

            draw_img = ImageDraw.Draw(img)
            draw_mask = ImageDraw.Draw(mask)

            # 여러 꼴
            for _ in range(np.random.randint(1, 4)):
                shape_type = np.random.choice(['circle', 'rectangle'])
                center_x = np.random.randint(size//4, 3*size//4)
                center_y = np.random.randint(size//4, 3*size//4)
                shape_size = np.random.randint(size//10, size//5)

                color = tuple(np.random.randint(50, 150, 3).tolist())

                if shape_type == 'circle':
                    bbox = [center_x - shape_size, center_y - shape_size,
                           center_x + shape_size, center_y + shape_size]
                    draw_img.ellipse(bbox, fill=color)
                    draw_mask.ellipse(bbox, fill=255)
                else:
                    bbox = [center_x - shape_size, center_y - shape_size,
                           center_x + shape_size, center_y + shape_size]
                    draw_img.rectangle(bbox, fill=color)
                    draw_mask.rectangle(bbox, fill=255)

            return img, mask

        def __len__(self):
            return self.num_samples

        def __getitem__(self, idx):
            img, mask = self.samples[idx]

            # 잣수는 보기마다가 아니라 배치마다 골라야 한다. 보기마다 고르면
            # 한 배치에 크기가 다른 텐서가 섞여 stack이 되지 않는다.
            # 여기서는 가장 큰 잣수로 내놓고, 줄이는 일은 collate가 맡는다.
            scale = max(self.scales)
            img = img.resize((scale, scale), Image.BILINEAR)
            mask = mask.resize((scale, scale), Image.NEAREST)

            # 텐서로 바꾼다
            img_array = np.array(img).astype(np.float32) / 255.0
            mask_array = np.array(mask).astype(np.float32) / 255.0

            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0)

            return img_tensor, mask_tensor


    # 데이터셋들을 만든다
    train_dataset = MultiScaleDataset(num_samples=800)
    val_dataset = MultiScaleDataset(num_samples=100, scales=[256])  # 검증용 붙박이 잣수
    test_dataset = MultiScaleDataset(num_samples=100, scales=[256])

    BATCH_SIZE = 4  # 여러 잣수와 어텐션 때문에 더 작다
    def multiscale_collate(batch):
        """배치마다 잣수 하나를 골라 그 배치 전체를 같은 크기로 맞춘다."""
        scale = int(np.random.choice(train_dataset.scales))
        imgs, masks = zip(*batch)
        imgs = F.interpolate(torch.stack(imgs), size=(scale, scale),
                             mode='bilinear', align_corners=False)
        masks = F.interpolate(torch.stack(masks), size=(scale, scale), mode='nearest')
        return imgs, masks


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=multiscale_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\nDataset created: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test\n")

    # ============================================================================
    # 5단계: 가장 좋게 하개와 일정 짜개
    # ============================================================================

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-6)

    if USE_MIXED_PRECISION:
        scaler = GradScaler()

    # ============================================================================
    # 6단계: 섞인 정밀도로 익히기
    # ============================================================================

    def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
        """켜져 있으면 섞인 정밀도로 익힌다."""
        model.train()
        running_loss = 0.0

        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            if USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(images)
                    # 마스크 크기에 맞게 내놓음 크기 바꾸기(여러 잣수용)
                    if outputs.shape[-2:] != masks.shape[-2:]:
                        outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                              mode='bilinear', align_corners=False)
                    loss = criterion(outputs, masks)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                if outputs.shape[-2:] != masks.shape[-2:]:
                    outputs = F.interpolate(outputs, size=masks.shape[-2:], 
                                          mode='bilinear', align_corners=False)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

            running_loss += loss.item()

        return running_loss / len(loader)


    def calculate_dice(pred, target):
        """다이스 계수를 셈한다."""
        with torch.no_grad():
            pred = torch.sigmoid(pred)
            pred = (pred > 0.5).float()

            pred = pred.view(-1)
            target = target.view(-1)

            intersection = (pred * target).sum()
            dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
            return dice.item()


    def validate(model, loader, criterion, device):
        """다이스 셈하기를 곁들인 검증."""
        model.eval()
        running_loss = 0.0
        running_dice = 0.0

        with torch.no_grad():
            for images, masks in loader:
                images = images.to(device)
                masks = masks.to(device)

                if USE_MIXED_PRECISION:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, masks)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                dice = calculate_dice(outputs, masks)

                running_loss += loss.item()
                running_dice += dice

        return running_loss / len(loader), running_dice / len(loader)


    # ============================================================================
    # 7단계: 시험 때 불리기
    # ============================================================================

    def test_time_augmentation(model, image, device):
        """
        시험 때 불리기를 여러 번 하고 어림을 고루낸다.
        튼튼함을 낫게 하며 흔히 성능을 1~3% 올린다.
        """
        model.eval()
        predictions = []

        with torch.no_grad():
            # 원래
            pred = torch.sigmoid(model(image))
            predictions.append(pred)

            # 가로 뒤집기
            pred = torch.sigmoid(model(torch.flip(image, dims=[3])))
            predictions.append(torch.flip(pred, dims=[3]))

            # 세로 뒤집기
            pred = torch.sigmoid(model(torch.flip(image, dims=[2])))
            predictions.append(torch.flip(pred, dims=[2]))

            # 두 방향 다 뒤집기
            pred = torch.sigmoid(model(torch.flip(image, dims=[2, 3])))
            predictions.append(torch.flip(pred, dims=[2, 3]))

        # 모든 어림 고루내기
        avg_pred = torch.stack(predictions).mean(dim=0)
        return avg_pred


    def evaluate_with_tta(model, loader, device):
        """시험 때 불리기로 평가한다."""
        model.eval()
        running_dice = 0.0

        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            if USE_TTA:
                predictions = test_time_augmentation(model, images, device)
            else:
                with torch.no_grad():
                    predictions = torch.sigmoid(model(images))

            # 다이스 셈하기
            pred = (predictions > 0.5).float()
            target = masks

            pred = pred.view(-1)
            target = target.view(-1)

            intersection = (pred * target).sum()
            dice = (2. * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
            running_dice += dice.item()

        return running_dice / len(loader)


    # ============================================================================
    # 8단계: 익히기 되풀이
    # ============================================================================

    NUM_EPOCHS = 30

    print(f"{'='*70}")
    print(f"Starting advanced training for {NUM_EPOCHS} epochs...")
    print(f"{'='*70}\n")

    best_dice = 0.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        # 학습
        train_loss = train_one_epoch(model, train_loader, criterion, 
                                     optimizer, scaler if USE_MIXED_PRECISION else None, device)

        # 검증
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # 결과 출력
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        # 학습률 스케줄링
        scheduler.step()

        # 최고 성능 모델 저장
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'best_advanced_model.pth')
            print("  ✓ Best model saved")

        print()

    total_time = time.time() - start_time
    print(f"{'='*70}")
    print(f"Training completed in {total_time//60:.0f}m {total_time%60:.0f}s")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"{'='*70}\n")

    # ============================================================================
    # 9단계: 시험 때 불리기를 쓴 마지막 값매김
    # ============================================================================

    model.load_state_dict(torch.load('best_advanced_model.pth'))

    print("Final Test Evaluation:")
    print("="*70)

    # 시험 때 불리기 안 씀
    test_dice_no_tta = evaluate_with_tta(model, test_loader, device)
    print(f"Test Dice (no TTA): {test_dice_no_tta:.4f}")

    # 시험 때 불리기 씀
    if USE_TTA:
        print("\nEvaluating with Test-Time Augmentation...")
        USE_TTA_TEMP = True
        test_dice_tta = evaluate_with_tta(model, test_loader, device)
        print(f"Test Dice (with TTA): {test_dice_tta:.4f}")
        print(f"TTA Improvement: +{(test_dice_tta - test_dice_no_tta):.4f}")

    # ============================================================================
    # 요약
    # ============================================================================

    print("\n" + "="*70)
    print("ADVANCED SEGMENTATION TECHNIQUES COMPLETE!")
    print("="*70)
    print("\nTechniques Applied:")
    print("✓ Attention mechanisms (CBAM)")
    print("✓ Advanced loss (Focal + Dice + Boundary)")
    if USE_MULTI_SCALE:
        print("✓ Multi-scale training")
    if USE_MIXED_PRECISION:
        print("✓ Mixed precision training (FP16)")
    if USE_TTA:
        print("✓ Test-time augmentation")

    print("\nPerformance Gains:")
    print(f"  Final Dice Score: {test_dice_tta if USE_TTA else test_dice_no_tta:.4f}")
    print(f"  Training Time: {total_time//60:.0f}m {total_time%60:.0f}s")

    print("\nKey Takeaways:")
    print("1. Attention mechanisms focus on important features")
    print("2. Combined losses leverage multiple objectives")
    print("3. Multi-scale training improves scale invariance")
    print("4. TTA boosts performance with minimal code")
    print("5. Mixed precision speeds up training")

    print("\nYou've completed all 4 examples!")
    print("You're now ready for:")
    print("- Real-world segmentation projects")
    print("- Kaggle competitions")
    print("- Research in semantic segmentation")
    print("- Production deployment")
    print("="*70)


    if __name__ == "__main__":
        pass
    ```


## 2. 논의

### CBAM과 비교의 조건

ChannelAttention은 공간 평균·최대 요약을 공유 MLP에 넣어 채널 게이트를 만들고, SpatialAttention은 채널 평균·최대를 연결해 공간 게이트를 만든다. 같은 채널의 모든 위치 또는 같은 위치의 모든 채널에 브로드캐스트하는 마스크이며 위치 쌍의 셀프 어텐션이 아니다. sigmoid의 게이트 합은1로 정규화되지 않는다.

U-Net의 합성곱 블록마다 CBAM을 붙이지만 `use_attention=False`여도 **bottleneck은 use_attention=True로 고정**되어 있다. 따라서 그 인수만 바꿔 얻은 모델을 ‘어텐션이 전혀 없는 기준 모델’이라고 보고하면 안 된다. 낮은 채널 수에 일반화하면 축소 폭 in_channels//16이 양수인지도 확인해야 한다.

### Focal·Dice·경계 가중 손실

FocalLoss는 pt=exp(−BCE)와 (1−pt)^γ로 어려운 픽셀에 상대적으로 더 큰 비중을 준다. 그러나 코드의 alpha는 양성·음성 모두에 동일한0.25를 곱한다. 원래의 α_y처럼 클래스별 균형을 다르게 주는 식이 아니므로 ‘양성 클래스 가중치0.25’라고 해석하지 않는다. gamma=0이면 이 alpha를 곱한 BCE가 된다.

DiceLoss는 배치 전체 확률을 펼친 soft Dice이며 임계화한 평가 Dice의 직접 최적값과 같지 않다. BoundaryLoss는 GT 마스크를2회 erosion·dilation한 띠에1+theta의 가중치를 주는 **경계 가중 BCE**다. 거리 변환에 기반한 특정 이름의 boundary loss나 경계 정확도 지표와 동일한 알고리즘은 아니다. GT에서 만든 가중치는 상수지만 BCE를 통한 입력 로짓 경사는 계산된다.

현재 형태학 연산은 매 배치 GPU→CPU→NumPy→GPU를 거쳐 수행해 동기화·전송 비용이 들 수 있다. 띠의 폭은 고정 픽셀 수여서 학습 해상도를 바꾸면 원 물체 크기에 대한 상대 폭도 달라진다. CombinedLoss의0.3·0.4·0.3은 서로 다른 스케일의 항을 조합하는 선택이며 항상 최적인 비율은 아니다.

### 다중 해상도 데이터의 실제 경로

Dataset은 각 표본을 최대 해상도로 생성·반환한다. **훈련 collate 함수**가 배치마다256 또는384 중 하나를 고르고 이미지를 bilinear, 마스크를 nearest로 함께 resize한다. 한 배치의 크기를 맞추는 것이며 각 표본이 다른 크기인 텐서를 그대로 stack하는 방식이 아니다. 이 collate는 전역 train_dataset의 scales에 의존하므로 다른 dataset에 재사용할 때 해당 의존성을 확인한다.

검증·테스트는256으로 고정한다. 두 크기는16의 배수라 네 번 풀링한 스킵과 맞지만 일반 크기의 crop·padding은 없다. 모델 출력 뒤 보간 코드는 이미 실패한 내부 스킵 연결을 고치지 못한다. 두 해상도 학습이 임의 스케일에 대한 완전한 불변성을 보장하지도 않는다.

### 혼합 정밀도의 범위

이 예제는 CUDA AMP 경로를 선택하고 CUDA가 없으면 해당 flag를 끈다. 이를 모든 형태의 혼합 정밀도가 CUDA에서만 가능하다는 일반 설명으로 읽지 않는다. 설치 버전에 맞는 autocast·GradScaler API, 모델·입력 장치와 연산 정밀도를 확인해야 한다. 가속과 메모리 절감은 장치·연산·입력에 따라 달라진다.

특히 soft Dice의 `inputs.sum()`은 큰 FP16 텐서에서 표현 범위를 넘을 수 있다. targets와의 곱이 float32로 승격되더라도 별도로 계산한 확률 합의 overflow가 자동으로 복구되지는 않는다. 수치적으로 민감한 합·분모는 적절한 누적 정밀도를 검토해야 한다. GradScaler는 경사 스케일을 다루며 forward의 모든 overflow를 해결하는 장치가 아니다. CPU 형태학 처리의 비용 역시 AMP만으로 사라지지 않는다.

### TTA의 역변환과 잘못된 기준 비교

TTA 함수는 원 영상·좌우 반전·상하 반전·두 축 반전의 네 예측을 sigmoid 확률로 만든 뒤 각 반전을 **출력에서 되돌리고** 평균한다. 정렬하지 않고 평균하면 다른 위치의 물체를 섞게 된다. 확률 평균 후0.5 임계화를 사용하며 로짓 평균이나 이진 마스크 평균과 다른 계산이다. 추론 횟수도4회로 늘어난다.

그러나 최종 코드의 `test_dice_no_tta`를 계산할 때도 전역 USE_TTA=True다. 첫 호출부터 TTA 경로를 사용한다. 뒤의 USE_TTA_TEMP=True는 평가 함수에서 사용하지 않아 두 번째 호출도 같은 경로다. 따라서 인쇄한 ‘no TTA’, ‘with TTA’, ‘TTA Improvement’를 실제 TTA의 유무 비교나 성능 향상의 근거로 사용할 수 없다. 서로 다른 두 조건을 명시적으로 평가해야 한다.

### 평가와 실험 보고

Dice는 배치 전체 픽셀의 비율을 계산한 뒤 배치별로 평균한다. 이미지별 평균이나 데이터 전체 누적 Dice와 다른 집계이며 빈 양쪽 마스크는 smooth로1이 된다. 현재 검증100개·배치4는 크기가 같지만 데이터 구성을 바꾸면 마지막 작은 배치의 가중치도 달라질 수 있다.

최선 체크포인트는0보다 엄격한 검증 개선에서만 저장한다. 저장하지 못한 실행의 고정 파일 로드는 유효성을 따로 확인해야 한다. 모듈 최상위의 생성·학습도 import에서 실행될 수 있다. 기법별 이득은 어텐션·손실·해상도·AMP·TTA를 통제한 ablation과 올바른 평가로 비교해야 하며, 이 코드를 실행하거나 그림을 다시 생성해 개선을 검증하지는 않았다. 기존 코드·출력은 그대로 보존했다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
CBAM의 게이트 형태와 어텐션 비활성화 flag의 한계를 설명하라.

</div>

??? success "연습문제 1 풀이"
    N×C×H×W 입력의 채널 게이트는N×C×1×1, 공간 게이트는N×1×H×W다. 채널 먼저 곱한 특징에서 공간 게이트를 구한다. use_attention=False로 만들더라도 병목 AttentionDoubleConv는True로 고정되어 완전한 무어텐션 기준이 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
다중 해상도 배치를 만드는 위치와 마스크 보간을 설명하라.

</div>

??? success "연습문제 2 풀이"
    Dataset은 최대 크기로 반환하고 collate가 배치별 공통 scale을 선택해 이미지 bilinear·마스크 nearest로 변환한다. 입력은이미3채널이며 네 풀링과 스킵의 내부 크기가 맞아야 한다. 모델 최종 출력의 보간은 내부의 실패한 스킵 덧붙이기를 고칠 수 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
최종 두 테스트 호출이 TTA 유무의 비교가 아닌 이유를 설명하라.

</div>

??? success "연습문제 3 풀이"
    첫 test_dice_no_tta 호출에서도 USE_TTA가True이므로 TTA를 수행한다. USE_TTA_TEMP는 함수가 읽지 않으므로 뒤 호출도 같은 경로다. 출력 라벨만 다르며 TTA의 성능 이득을 측정하지 않았다. 같은 최선 가중치·데이터에서 실제 flag를 달리하고 역변환·확률 집계를 일관되게 비교해야 한다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 일반 반복 골격 대신 올바른 ablation을 설계하려면 무엇을 확인해야 하는가?

</div>

??? success "연습문제 4 풀이"

    YourBlock·dim 등은 정의되지 않았고 목록만으로 실험 조건이 바뀌지 않는다. 모든 블록의 실제 어텐션 사용, focal alpha의 클래스 가중치 여부, 다중 해상도 collate, AMP 누적 정밀도와 TTA 경로를 확인한다. 동일한 초기값·데이터·예산·평가에서 바꾸는 요인을 명시해야 개선의 원인을 비교할 수 있다.

    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```


## 정리하며

여러 기법의 조합은 실제 호출·손실 정의·평가 조건을 기준으로 해석한다. 이 예제의 어텐션 flag는 병목을 끄지 않고 TTA 비교의 두 호출은 같은 경로를 사용한다. 경계 가중 BCE와 focal·Dice의 목적 및 AMP의 수치 한계를 구분한다.
