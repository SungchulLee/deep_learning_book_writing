# 보기 2

보기 2: 미리 익힌 부호기로 나누기에 옮겨 배우기. 이 각본은 미리 익힌 부호기(ResNet, VGG 등)를 쓰는 법을 보여 준다.

이 단원은 셈틀 보기라는 더 넓은 맥락 안에서 그림 나누기를 살펴본다. 여기 짠 것은 요즘 체계에서 쓰는 얼개의 새로움과 익히기 전략을 보여 주는 실전 PyTorch 코드이다.

## 코드

```python
"""
보기 2: 미리 익힌 부호기로 나누기에 옮겨 배우기
========================================================================

이 각본은 미리 익힌 부호기(ResNet, VGG 등)를 쓰는 법을 보여 준다
뜻 나누기에 쓰는 법을 보여 준다. segmentation_models_pytorch 라이브러리를 쓰는데
이는 ImageNet에서 미리 익힌 등뼈를 갖춘 여러 얼개를 준다.

핵심 개념:
- 나누기를 위한 옮겨 배우기
- 미리 익힌 부호기(ResNet, EfficientNet)
- DeepLabV3+ 얼개
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
         cat, chair, cow, dining table, dog, horse, motorbike, person,
         potted plant, sheep, sofa, train, tv/monitor
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
# 3단계: 미리 익힌 부호기로 모델 고르기
# ============================================================================
"""
segmentation_models_pytorch(smp)는 미리 익힌 부호기를 갖춘 여러 얼개를 준다
부호기를 쓴다. 여기서는 ResNet-50 부호기를 쓴 DeepLabV3+를 쓴다.

얼개 고름:
- Unet: 건너뛰는 이음을 갖춘 고전 U-넷
- UnetPlusPlus: 겹겹이 든 건너뛰는 이음을 갖춘 나아진 U-넷
- DeepLabV3+: 구멍 뚫린 누비기와 ASPP를 쓴 가장 앞선 것
- FPN: 특징 피라미드 그물
- PSPNet: 피라미드 장면 뜯어 읽기 그물

부호기 고름:
- resnet18, resnet34, resnet50, resnet101
- efficientnet-b0 to b7
- mobilenet_v2
- vgg16, vgg19
"""

print("Creating DeepLabV3+ model with pre-trained ResNet-50 encoder...")

# 미리 익힌 부호기로 모델 만들기
model = smp.DeepLabV3Plus(
    encoder_name="resnet50",           # 부호기 고르기(resnet50, efficientnet-b0 등)
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
- 부호기(미리 익힘): 작은 배움 비율(0.0001) — 조심스레 곱게 다듬는다
- 풀개(마구잡이 첫자리매김): 큰 배움 비율(0.001) — 더 많이 배워야 한다

이를 "층마다 다른 배움 비율"이라 하며
나누기에서 옮겨 배우기를 잘하는 데 결정적이다.
"""

# 부호기와 풀개의 매개변수 나누기
encoder_params = []
decoder_params = []

for name, param in model.named_parameters():
    if 'encoder' in name:
        encoder_params.append(param)
    else:
        decoder_params.append(param)

# 배움 비율이 다른 가장 좋게 하개 만들기
optimizer = optim.Adam([
    {'params': encoder_params, 'lr': 0.0001},   # 미리 익힌 부호기에는 작은 배움 비율
    {'params': decoder_params, 'lr': 0.001},    # 풀개에는 더 큰 배움 비율
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

## 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 넓혀진다. 초매개변수와 구조의 변형, 다른 데이터셋으로 실험해 보면 이해가 깊어지고 컴퓨터 비전 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

---

**연습문제 2.**
최적화기를 Adam으로 바꾸고(`torch.optim.Adam`에 `lr=0.001`을 쓴다) 원래 최적화기와 학습 수렴을 비교하라. 두 손실 곡선을 같은 그래프에 그려라.

??? success "연습문제 2 풀이"
    최적화기를 만드는 줄을 `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`으로 바꾼다. Adam은 매개변수마다 적응적인 학습률과 운동량 추정값을 유지하므로 초반 에폭에서 대체로 더 빠르게 수렴한다. Adam의 손실 곡선은 보통 처음 몇 에폭에서 더 가파르게 떨어지지만, 최적점 근처에서는 운동량을 쓴 SGD보다 조금 더 흔들릴 수 있다. 공정한 비교를 위해 둘을 같은 난수 씨앗과 같은 에폭 수로 실행하라.

---

**연습문제 3.**
이 구현에서 생길 수 있는 실패 양상 두 가지를 서술하고, 각각을 어떻게 진단하고 고칠지 설명하라.

??? success "연습문제 3 풀이"
    흔한 실패 양상은 다음과 같다. (1) **경사 소실/폭발** — 경사의 노름을 지켜보아 진단한다(`torch.nn.utils.clip_grad_norm_`을 쓰거나 층마다 `param.grad.norm()`을 기록한다). 경사 자르기, 더 나은 초기화(Xavier/Kaiming), 또는 구조 변경(잔차 연결, 정규화)으로 고친다. (2) **과적합** — 학습 손실은 줄어드는데 검증 손실이 늘어나면 진단된다. 정칙화(드롭아웃, 가중치 감쇠, 데이터 증강)나 모델 용량 축소로 고친다. 이런 문제를 일찍 잡아내려면 언제나 학습 지표와 검증 지표를 함께 살펴라.

---

**연습문제 4.**
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
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
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.
