# 보기 4

보기 4: 앞선 뜻 나누기 재주. 이 각본은 뜻 나누기의 가장 앞선 재주를 보여 준다:

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 그림 나누기를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 코드

```python
"""
보기 4: 앞선 뜻 나누기 재주
=====================================================

이 각본은 뜻 나누기의 가장 앞선 재주를 보여 준다:
- 눈길 얼개(CBAM)
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
# 1단계: 눈길 단원
# ============================================================================
"""
눈길 얼개는 모델이 중요한 자리와 특징에 초점을 두게 돕는다.
CBAM(누비기 덩이 눈길 단원)은 눈길을
채널 차원과 자리 차원 모두에 준다.
"""

class ChannelAttention(nn.Module):
    """
    채널 눈길 단원.
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
    자리 눈길 단원.
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
    누비기 덩이 눈길 단원.
    채널 눈길과 자리 눈길을 아우른다.
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        # 채널 눈길 쓰기
        x = x * self.channel_attention(x)
        # 자리 눈길 쓰기
        x = x * self.spatial_attention(x)
        return x


# ============================================================================
# 2단계: 눈길을 갖춘 앞선 U-넷
# ============================================================================

class AttentionDoubleConv(nn.Module):
    """눈길을 갖춘 겹 누비기 덩이."""
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
    - 눈길 단원(CBAM)
    - 깊은 이끎(있어도 되고 없어도 됨)
    - 잔차 이음(있어도 되고 없어도 됨)
    """
    def __init__(self, in_channels=3, num_classes=1, use_attention=True):
        super().__init__()
        
        # 부호기
        self.enc1 = AttentionDoubleConv(in_channels, 64, use_attention)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = AttentionDoubleConv(64, 128, use_attention)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = AttentionDoubleConv(128, 256, use_attention)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = AttentionDoubleConv(256, 512, use_attention)
        self.pool4 = nn.MaxPool2d(2)
        
        # 눈길을 갖춘 병목
        self.bottleneck = AttentionDoubleConv(512, 1024, use_attention=True)
        
        # 복호기
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
        
        # 복호기
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
        
        # 여러 잣수 익히기를 위한 마구잡이 잣수 고르기
        scale = np.random.choice(self.scales)
        
        # 고른 잣수로 크기 바꾸기
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

BATCH_SIZE = 4  # 여러 잣수와 눈길 때문에 더 작다
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
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

## 논의

여기 짠 것은 함께 어울려 온전한 그림 나누기 얼개를 이루는 클래스 10개(`ChannelAttention`, `SpatialAttention`, `CBAM`, `AttentionDoubleConv`, 그 밖 6개)를 정한다. 클래스마다 뚜렷한 조각 하나를 감싸므로 코드가 단원별로 나뉘고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 셈 그래프를 정한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 컴퓨터 비전 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`ChannelAttention`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 묶음에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = ChannelAttention(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**연습문제 3.**
자기 어텐션의 계산 복잡도를 열의 길이 $n$과 모델 차원 $d$의 함수로 설명하라. 이것이 왜 긴 열에 대해 Longformer나 Linformer 같은 구조의 동기가 되는가?

??? success "연습문제 3 풀이"
    표준 자기 어텐션은 $n \times n$ 어텐션 행렬을 계산하므로 시간 복잡도가 $O(n^2 d)$이고 어텐션 가중치에 $O(n^2)$의 메모리가 든다. 열이 길면(예: $n = 4096$) 감당하기 어려워진다. Longformer는 국소적인 미끄럼창 어텐션($w$이 창 크기일 때 $O(n \cdot w \cdot d)$)과 선택된 토큰에 대한 희소한 전역 어텐션을 결합한다. Linformer는 키와 값을 더 낮은 차원 $k \ll n$으로 사영하여 복잡도를 $O(n \cdot k \cdot d)$으로 줄인다. 둘 다 표현력을 조금 내주고 긴 입력에서의 실용적인 효율을 얻는다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `ChannelAttention`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = ChannelAttention(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
