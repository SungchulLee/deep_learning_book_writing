# 보기 1

보기 1: 기본 U-넷 뜻 나누기. 이 각본은 온전한 U-넷 얼개를 맨바닥부터 짠다.

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 그림 나누기를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 1. 코드

```python
"""
보기 1: 기본 U-넷 뜻 나누기
=============================================

이 각본은 온전한 U-넷 얼개를 맨바닥부터 짠다
두 갈래 뜻 나누기를 한다. 단순한 꼴로 된 인공 자료 뭉치를 쓴다.

핵심 개념:
- U-넷 얼개(건너뛰는 이음을 갖춘 부호기-풀개)
- 두 갈래 나누기(갈래 2개)
- 화소마다의 엇갈린 엔트로피 손실
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
    - 마스크: 1=꼴, 0=뒷바탕인 두 갈래 마스크
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
        
        # 꼴 갈래를 마구잡이로 고르기
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
# 3단계: U-넷 얼개 짜기
# ============================================================================
"""
U-넷은 다음으로 이루어진다:
1. 부호기(오그라드는 길): 맥락을 담아낸다
2. 병목: 가장 깊은 특징
3. 풀개(부풀어 오르는 길): 정밀한 자리 잡기를 가능하게 한다
4. 건너뛰는 이음: 부호기 특징을 풀개에 이어 붙인다
"""

class DoubleConv(nn.Module):
    """
    겹 누비기 덩이: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU
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
    뜻 나누기를 위한 U-넷 얼개.
    
    구조:
        들임(3채널) -> 부호기(건너뛰는 이음 포함) -> 병목 ->
        풀개(건너뛰는 이음 포함) -> 내놓음(num_classes 채널)
    
    인수:
        in_channels: 들임 채널의 개수(RGB는 3)
        num_classes: 내놓는 갈래의 개수(두 갈래는 2: 뒷바탕, 앞바탕)
    """
    
    def __init__(self, in_channels=3, num_classes=2):
        super(UNet, self).__init__()
        
        # 부호기(줄여 뽑는 길)
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
        
        # 풀개(키우는 길)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)  # 16x16 -> 32x32
        self.dec4 = DoubleConv(1024, 512)             # 32x32 -> 32x32(이어 붙여서 1024)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)   # 32x32 -> 64x64
        self.dec3 = DoubleConv(512, 256)              # 64x64 -> 64x64
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)   # 64x64 -> 128x128
        self.dec2 = DoubleConv(256, 128)              # 128x128 -> 128x128
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)    # 128x128 -> 256x256
        self.dec1 = DoubleConv(128, 64)               # 256x256 -> 256x256
        
        # 마지막 내놓는 층
        self.out = nn.Conv2d(64, num_classes, kernel_size=1)  # 갈래 점수를 얻는 1x1 누비기
    
    def forward(self, x):
        """
        U-넷을 지나는 앞먹임.
        
        인수:
            x: 꼴이 (batch_size, 3, H, W)인 들임 텐서
        
        반환값:
            꼴이 (batch_size, num_classes, H, W)인 내놓는 텐서
        """
        # 건너뛰는 이음을 갈무리한 부호기
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
        
        # 건너뛰는 이음을 갖춘 풀개
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
model = UNet(in_channels=3, num_classes=2)  # 두 갈래: 뒷바탕(0), 앞바탕(1)
model = model.to(device)

# 매개변수 개수 세기
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ============================================================================
# 4단계: 손실 함수와 가장 좋게 하개
# ============================================================================
"""
두 갈래 나누기에는 로짓을 쓴 두 갈래 엇갈린 엔트로피 손실을 쓴다.
이는 시그모이드 깨어남과 두 갈래 엇갈린 엔트로피를 한 함수로 아울러
수치로 든든하게 만든다.

손실은 화소마다 셈한다:
손실 = -[y*log(p) + (1-y)*log(1-p)]
여기서 y은 참값(0 또는 1), p은 어림 확률
"""

criterion = nn.CrossEntropyLoss()  # 여러 갈래용(갈래가 둘뿐이어도)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습률 스케줄러
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

print("\nTraining configuration:")
print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: Adam (lr=0.001)")
print(f"Scheduler: ReduceLROnPlateau (monitor IoU)")

# ============================================================================
# 5단계: 값매김 잣대
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
    두 갈래 나누기의 겹침 비(IoU)를 셈한다.
    
    인수:
        pred: 꼴이 (batch, 2, H, W)인 어림 로짓
        target: 값이 {0, 1}이고 꼴이 (batch, 1, H, W)인 참값 마스크
        threshold: 확률을 두 갈래 어림으로 바꾸는 문턱값
    
    반환값:
        iou: 묶음에 걸친 평균 겹침 비
    """
    with torch.no_grad():
        # 어림한 갈래 얻기(0 또는 1)
        pred_class = torch.argmax(pred, dim=1)  # 꼴: (묶음, H, W)
        target_class = target.squeeze(1).long()  # 꼴: (묶음, H, W)
        
        # 앞바탕 갈래(갈래 1)의 교집합과 합집합 셈하기
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
        
        # 손실 계산
        # CrossEntropyLoss는 갈래 번호를 담은 (묶음, H, W) 꼴의 목표를 바란다
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
            
            # 손실 계산
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
    
    # 학습률 스케줄링
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

여기 짠 것은 함께 어울려 온전한 그림 나누기 얼개를 이루는 클래스 3개(`SyntheticShapesDataset`, `DoubleConv`, `UNet`)를 정한다. 클래스마다 뚜렷한 조각 하나를 감싸므로 코드가 단원별로 나뉘고 넓히기 쉽다. `forward` 메서드가 PyTorch의 자동 미분에 쓰이는 셈 그래프를 정한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 컴퓨터 비전 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
`SyntheticShapesDataset`의 앞먹임을 따라가며 텐서 꼴을 좇아라. 붙박이 매개변수로 들임 표본 4개짜리 묶음에 대해 주요 연산(누비기, 모으기, 선형 층)마다 그 뒤의 꼴을 적어라.

??? success "연습문제 1 풀이"
    입력 모양에서 출발하여 각 층을 차례로 적용한다. `Conv2d(in_c, out_c, k)`마다 공간 차원은 (덧대기가 없으면) $H_{\text{out}} = H_{\text{in}} - k + 1$으로 바뀌거나 (`padding=k//2`이면) 그대로 유지된다. 커널이 2인 풀링은 공간 차원을 절반으로 만든다. 선형 층은 마지막 차원을 바꾼다. 배치 차원은 내내 그대로임에 유의하며 추적한다. 중간 모양을 합성곱 층에서는 $(B, C, H, W)$로, 평탄화 후에는 $(B, F)$로 적는다.

---

**연습문제 2.**
$64 \times 64$ 크기의 RGB 이미지(입력 모양 $3 \times 64 \times 64$)를 받도록 구조를 수정하라. 모든 층의 차원을 그에 맞게 고치고 모델이 오류 없이 실행되는지 확인하라.

??? success "연습문제 2 풀이"
    첫 엮음 켜의 `in_channels`을 이제 값에서 3으로 바꾼다. 엮음과 모으기 켜마다 $H_{\text{out}} = \lfloor(H_{\text{in}} + 2p - k) / s\rfloor + 1$으로 자리 차수를 다시 셈한다. 첫 선형 켜의 `in_features`을 마지막 엮음/모으기 켜의 편 날임에 맞춘다. `model = SyntheticShapesDataset(...); x = torch.randn(1, 3, 64, 64); print(model(x).shape)`으로 따진다.

---

**연습문제 3.**
같은 입출력 차원에서 표준 합성곱과 깊이별 분리 합성곱의 매개변수 개수와 FLOPs를 비교하라. 계산 절감이 가장 큰 것은 언제인가?

??? success "연습문제 3 풀이"
    표준 `Conv2d(C_in, C_out, k)`은 $C_{{\text{{in}}}} \times C_{{\text{{out}}}} \times k^2$개의 매개변수를 갖는다. 깊이별 분리 합성곱은 이를 둘로 나눈다. (1) 깊이별: $C_{{\text{{in}}}} \times k^2$개(입력 채널마다 필터 하나), (2) 점별: $C_{{\text{{in}}}} \times C_{{\text{{out}}}}$개(1x1 합성곱)이다. 매개변수의 비는 대략 $1/C_{{\text{{out}}}} + 1/k^2$이다. $k=3$이고 $C_{{\text{{out}}}}=256$이면 매개변수가 약 $8{-}9\times$ 적어진다. 절감은 $C_{{\text{{out}}}}$과 $k$가 모두 클 때 가장 크다.

---

**연습문제 4.**
층이나 덩이의 개수를 정할 수 있도록 `SyntheticShapesDataset`을 넓혀라. `__init__`에 `num_layers` 매개변수를 더하고 `nn.ModuleList`로 깊이가 바뀌는 얼개를 만들어라. 층 2, 4, 8개로 시험하여라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`로 되풀이한다. (그냥 파이썬 목록이 아니라) `nn.ModuleList`를 써야 PyTorch가 가장 좋게 하기에 쓸 매개변수를 모두 등록한다. 시험: `for n in [2, 4, 8]: model = SyntheticShapesDataset(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 보기 1

여기 짠 것은 함께 어울려 온전한 그림 나누기 얼개를 이루는 클래스 3개(`SyntheticShapesDataset`, `DoubleConv`, `UNet`)를 정한다.

고갱이 갈래는 `SyntheticShapesDataset`, `DoubleConv`, `UNet`이며 앞의 연습문제 4개로 스스로 따져 볼 수 있다.
