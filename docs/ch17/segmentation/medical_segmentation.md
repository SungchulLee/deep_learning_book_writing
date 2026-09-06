# 보기 3

보기 3: 의료 그림 나누기. 이 각본은 그 분야에 맞춘 의료 그림 나누기를 보여 준다.

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 그림 나누기를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 코드

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
        
        # 부호기
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
        
        # 복호기
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # 마지막 내놓음(깨어남 없음, BCEWithLogitsLoss를 쓴다)
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)
    
    def forward(self, x):
        # 부호기
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
        
        # 건너뛰는 이음을 갖춘 풀개
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
    Dice Loss = 1 - Dice
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

## 논의

여기 짠 것은 함께 어울려 온전한 그림 나누기 얼개를 이루는 클래스 5개(`SyntheticMedicalDataset`, `DoubleConv`, `MedicalUNet`, `DiceLoss`, 그 밖 1개)를 정한다. 클래스마다 뚜렷한 조각 하나를 감싸므로 코드가 단원별로 나뉘고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 셈 그래프를 정한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 컴퓨터 비전 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`SyntheticMedicalDataset`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 묶음에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    Change the first convolutional layer's `in_channels` from the current value to 3. Recalculate the spatial dimensions after each convolution and pooling layer using the formulas $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$. Update the first linear layer's `in_features` to match the flattened output of the last convolutional/pooling layer. Verify with: `model = SyntheticMedicalDataset(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `SyntheticMedicalDataset`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = SyntheticMedicalDataset(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
