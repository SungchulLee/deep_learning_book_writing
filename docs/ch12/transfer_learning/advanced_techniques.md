# 고급 기법

이 예제는 최고 성능을 얻으려고 연구와 실무에서 쓰는 최신 전이 학습 기법을 보여 준다. 더 빠른 계산을 위한 섞인 정밀도 학습(FP16), 실효 배치 크기를 키우는 기울기 모으기, 코사인 담금질 학습률 일정, 파국적 망각을 막는 차츰 녹이기, 추론 정확도를 높이는 시험 때 데이터 불리기를 다룬다. 이 기법들을 함께 짜 맞추면 사전 학습된 모델에서 새 과제의 성능을 최대로 짜낼 수 있다.

## 1. 코드

```python
"""
예제 4: 고급 전이 학습 기법
=================================================

이 스크립트는 최신 전이 학습 기법을 보여 준다.
- 섞인 정밀도 학습 (FP16)
- 기울기 모으기
- 고급 학습률 일정
- 층을 차츰 녹이기
- 모델 앙상블
- 여러 구조

이 기법들은 최고의 성능을 얻으려고 연구와 실무에서
쓰인다.

지은이: PyTorch Transfer Learning Tutorial
날짜: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision
import torchvision.transforms as transforms
from torchvision.models import (resnet18, resnet50,
                                ResNet18_Weights, ResNet50_Weights)
import time
import copy
import numpy as np

# 난수 씨앗 고정
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 설정
# ============================================================================
"""
이 깃발을 켜고 끄며 여러 고급 기법을 쓰거나 끈다.
여러 조합을 시험해 보라!
"""

# 학습 설정
BATCH_SIZE = 32
NUM_EPOCHS = 20
BASE_LR = 0.001

# 고급 기법 깃발
USE_MIXED_PRECISION = True        # FP16 학습을 켠다 (GPU가 필요하다)
GRADIENT_ACCUMULATION_STEPS = 2   # 실효 배치 크기 = BATCH_SIZE × 이 값
USE_COSINE_SCHEDULE = True        # 코사인 담금질을 쓸지 정체 감지를 쓸지
GRADUAL_UNFREEZING = True         # 학습 중에 층을 차츰 녹인다
USE_ENSEMBLE = False              # 앙상블을 위해 모델을 여럿 학습한다 (느리다)

print("Advanced Transfer Learning Configuration:")
print("=" * 70)
print(f"Batch Size: {BATCH_SIZE}")
print(f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Mixed Precision: {USE_MIXED_PRECISION}")
print(f"Gradient Accumulation: {GRADIENT_ACCUMULATION_STEPS} steps")
print(f"Cosine Annealing Schedule: {USE_COSINE_SCHEDULE}")
print(f"Gradual Unfreezing: {GRADUAL_UNFREEZING}")
print(f"Model Ensemble: {USE_ENSEMBLE}")
print("=" * 70 + "\n")

# ============================================================================
# 1단계: 장치 설정
# ============================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 섞인 정밀도를 쓸 수 있는지 살핀다
if USE_MIXED_PRECISION and not torch.cuda.is_available():
    print("Warning: Mixed precision requires CUDA. Disabling mixed precision.")
    USE_MIXED_PRECISION = False

if USE_MIXED_PRECISION:
    print("Mixed precision (FP16) training enabled")

# ============================================================================
# 2단계: 고급 데이터 불리기
# ============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 학습 불리기 파이프라인
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# 시험 때 불리기 변환
tta_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.FiveCrop(224),
    transforms.Lambda(lambda crops: torch.stack([
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)(
            transforms.ToTensor()(crop)
        ) for crop in crops
    ]))
])

print("\nLoading CIFAR-10 dataset with advanced augmentation...")

# 데이터셋 불러오기
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=val_transform
)

# 학습 자료를 학습과 검증으로 나눈다
from torch.utils.data import random_split
train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(
    train_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

# 데이터 로더 생성
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True if torch.cuda.is_available() else False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True if torch.cuda.is_available() else False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=2, pin_memory=True if torch.cuda.is_available() else False
)

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

print(f"Dataset loaded: Train={len(train_dataset)}, "
      f"Val={len(val_dataset)}, Test={len(test_dataset)}")

# ============================================================================
# 3단계: 구조를 골라 모델 만들기
# ============================================================================

def create_model(architecture='resnet18', num_classes=10, pretrained=True):
    """
    정한 구조로 전이 학습 모델을 만든다.
    """
    print(f"\nCreating {architecture} model...")

    if architecture == 'resnet18':
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        num_features = model.fc.in_features
    elif architecture == 'resnet50':
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet50(weights=weights)
        num_features = model.fc.in_features
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

    # 처음에는 모든 층을 얼린다
    for param in model.parameters():
        param.requires_grad = False

    # 마지막 층을 바꾼다
    model.fc = nn.Linear(num_features, num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model

# 주 모델(ResNet18)을 만든다
model = create_model('resnet18', num_classes=len(classes))
model = model.to(device)

# ============================================================================
# 4단계: 차츰 녹이기 도우미
# ============================================================================

def unfreeze_layers(model, epoch, total_epochs):
    """
    학습이 나아가는 정도에 따라 층을 차츰 녹인다.
    """
    if not GRADUAL_UNFREEZING:
        for name, param in model.named_parameters():
            if any(layer in name for layer in ['layer3', 'layer4']):
                param.requires_grad = True
        return

    progress = epoch / total_epochs

    if progress < 0.25:
        pass
    elif progress < 0.5:
        for name, param in model.named_parameters():
            if 'layer4' in name:
                param.requires_grad = True
    elif progress < 0.75:
        for name, param in model.named_parameters():
            if any(layer in name for layer in ['layer3', 'layer4']):
                param.requires_grad = True
    else:
        for name, param in model.named_parameters():
            if any(layer in name for layer in ['layer2', 'layer3', 'layer4']):
                param.requires_grad = True

# ============================================================================
# 5단계: 최적화기와 일정 조정기 설정
# ============================================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=BASE_LR,
    weight_decay=0.01
)

if USE_COSINE_SCHEDULE:
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )
    print("\nUsing CosineAnnealingWarmRestarts scheduler")
else:
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    print("\nUsing ReduceLROnPlateau scheduler")

# ============================================================================
# 6단계: 섞인 정밀도 설정
# ============================================================================

if USE_MIXED_PRECISION:
    scaler = GradScaler()
    print("Mixed precision scaler initialized")

# ============================================================================
# 7단계: 고급 기능을 갖춘 학습 함수
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """섞인 정밀도와 기울기 모으기로 한 세대를 학습한다."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    optimizer.zero_grad()

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if USE_MIXED_PRECISION:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss = loss / GRADIENT_ACCUMULATION_STEPS

        if USE_MIXED_PRECISION:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            if USE_MIXED_PRECISION:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'  Batch {batch_idx + 1}/{len(loader)}: '
                  f'Loss: {running_loss / (batch_idx + 1):.3f}, '
                  f'Acc: {100. * correct / total:.2f}%, '
                  f'LR: {current_lr:.6f}')

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """표준 평가 함수."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

# ============================================================================
# 8단계: 주 학습 고리
# ============================================================================

print(f"\n{'='*70}")
print(f"Starting advanced training for {NUM_EPOCHS} epochs...")
print(f"{'='*70}\n")

best_val_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 70)

    prev_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    unfreeze_layers(model, epoch, NUM_EPOCHS)
    curr_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if curr_trainable != prev_trainable:
        print(f"Unfroze additional layers: {curr_trainable:,} trainable parameters")
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=optimizer.param_groups[0]['lr'],
            weight_decay=0.01
        )

    if USE_MIXED_PRECISION:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
    else:
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, None, device, epoch
        )

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")

    if USE_COSINE_SCHEDULE:
        scheduler.step()
    else:
        scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        print(f"  New best model saved")

    print()

total_time = time.time() - start_time
print(f"{'='*70}")
print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
print(f"Best validation accuracy: {best_val_acc:.2f}%")
print(f"{'='*70}\n")

model.load_state_dict(best_model_weights)

# ============================================================================
# 9단계: 고급 평가
# ============================================================================

print("Final Test Evaluation:")
print("=" * 70)

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Single Model Test Accuracy: {test_acc:.2f}%")

# ============================================================================
# 10단계: 시험 때 불리기 (TTA)
# ============================================================================

print("\nTest-Time Augmentation Evaluation:")
print("-" * 70)

def evaluate_with_tta(model, loader, device):
    """시험 때 불리기로 평가한다."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            batch_size = inputs.size(0)

            flipped = torch.flip(inputs, dims=[3])
            all_inputs = torch.cat([inputs, flipped], dim=0)
            all_inputs = all_inputs.to(device)

            if USE_MIXED_PRECISION:
                with autocast():
                    outputs = model(all_inputs)
            else:
                outputs = model(all_inputs)

            outputs1 = outputs[:batch_size]
            outputs2 = outputs[batch_size:]
            avg_outputs = (outputs1 + outputs2) / 2

            labels = labels.to(device)
            _, predicted = avg_outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return accuracy

tta_acc = evaluate_with_tta(model, test_loader, device)
print(f"With TTA Test Accuracy: {tta_acc:.2f}%")
print(f"TTA Improvement: +{tta_acc - test_acc:.2f}%")

# ============================================================================
# 마무리 요약
# ============================================================================

print("\n" + "="*70)
print("ADVANCED TRANSFER LEARNING COMPLETE!")
print("="*70)

# 마지막 모델을 저장한다
torch.save(model.state_dict(), 'advanced_transfer_model.pth')
print("\nModel saved as 'advanced_transfer_model.pth'")


if __name__ == "__main__":
    pass
```

## 2. 논의

**섞인 정밀도 학습**은 모델 가중치를 FP32으로 담되 앞먹임과 역전파는 FP16으로 하여 기억 사용을 대략 절반으로 줄이고 요즘 GPU의 텐서 코어를 써서 2~3배 빨라진다. `GradScaler`는 FP16에서 기울기가 밑으로 넘치는 것을 막으려고 손실의 크기를 저절로 조정한다. 반정밀도에서 0으로 내림될 작은 기울기를, 역전파 전에 손실에 유동적인 배수를 곱하고 최적화기 단계 전에 기울기를 같은 배수로 나누어 지켜 낸다.

**기울기 모으기**는 기억을 더 쓰지 않고 큰 배치를 흉내 낸다. 배치마다 가중치를 갱신하는 대신 최적화기 단계 하나 전에 앞먹임과 역전파를 여러 번 하며 기울기를 쌓는다. 배치 크기가 32이고 모으기 단계가 2이면 실효 배치 크기가 64가 된다. 기울기의 크기가 진짜 배치 64가 낼 것과 맞도록 손실을 모으기 단계 수로 나눈다. GPU 기억이 배치 크기를 옥죄지만 학습의 움직임이 큰 실효 배치의 덕을 볼 때 꼭 필요한 기법이다.

**차츰 녹이기**는 사전 학습된 모델을 미세 조정할 때 생기는 파국적 망각 문제를 다룬다. 처음에는 사전 학습 층을 모두 얼려 둔 채 새 분류 머리만 학습한다. 학습이 나아가면서 깊은 층을 차츰 녹인다. 먼저 마지막 잔차 블록(layer4), 그다음 layer3 하는 식이다. 그러면 사전 학습된 특징이 맞추어지기 전에 무작위로 초기화된 분류기가 수렴할 틈이 생겨, 분류기의 큰 기울기 신호가 정성껏 배운 특징의 위계를 망가뜨리는 일을 막는다. (앞 층에 더 낮은 학습률을 주는) 층별 학습률과 함께 쓰면 앞 층의 일반 특징을 지키면서 뒤 층은 과제에 맞추어 갈 수 있다.

## 연습문제

**연습문제 1.**
배치 크기 32, 기울기 모으기 4단계, 기억이 8GB인 GPU에서 실효 배치 크기를 어림하고, (배치 128이 기억에 들어가지 않는다고 할 때) 왜 이것이 배치 크기를 그냥 128로 두는 것보다 나을 수 있는지 설명하라.

??? success "연습문제 1 풀이"
    실효 배치 크기는 $32 \times 4 = 128$이다. 배치를 곧바로 128로 두는 것보다 나은 까닭은 다음과 같다.

    1. **기억**: ResNet18의 활성까지 더하면 $224 \times 224 \times 3$ 그림 128장의 배치가 8GB를 넘을 수 있지만 32장은 넉넉히 들어간다.
    2. **기울기가 같음**: (코드가 하듯) 단계마다 손실에 $1/4$을 곱하면 쌓은 기울기가 배치 128의 기울기와 수학적으로 같다.
    3. **맞바꿈**: 나쁜 점은 배치 정규화 통계를 128개가 아니라 32개 표본으로 셈하여 학습의 움직임이 조금 달라질 수 있다는 것뿐이다. 또 앞먹임·역전파 4번은 표본 128개를 처리하는 것과 벽시계 시간이 같으므로 빨라지지는 않는다. 이점은 오로지 기억에 있다.

---

**연습문제 2.**
`CosineAnnealingWarmRestarts`와 `ReduceLROnPlateau` 일정 조정기의 차이를 설명하라. 언제 어느 쪽이 더 나은가? 코사인 담금질에서 `T_0`과 `T_mult` 매개변수는 어떤 몫을 하는가?

??? success "연습문제 2 풀이"
    **CosineAnnealingWarmRestarts**는 미리 정한 일정을 따른다. 학습률이 `T_0` 세대에 걸쳐 코사인 곡선을 따라 처음 값에서 `eta_min`까지 내려간 뒤 처음 값으로 "다시 시작"한다. `T_0`이 첫 주기의 길이를 정하고 `T_mult`이 다시 시작할 때마다 주기에 곱해진다(그래서 주기가 점점 길어진다). `T_0=5`, `T_mult=2`이면 주기가 5, 10, 20 세대이다.

    **ReduceLROnPlateau**는 반응하는 쪽이다. (검증 손실 같은) 지표를 살피다가 `patience` 세대 동안 지표가 나아지지 않으면 학습률에 `factor`를 곱해 줄인다. 정해진 일정을 따르는 대신 실제 학습의 움직임에 맞추어 간다.

    학습 예산이 정해져 있고 국소 최솟값을 벗어나는 데 도움이 되는 주기적인 따뜻한 다시 시작으로 두루 살피고 싶으면 **코사인 담금질**이 낫다. 학습 기간이 자유롭고 학습률이 실제 수렴 모습에 맞추어 가기를 바라면 **ReduceLROnPlateau**가 낫다. 전이 학습에서는 대개 학습 예산이 정해져 있고 따뜻한 다시 시작이 이로운 규제가 되므로 코사인 담금질을 흔히 더 좋아한다.

---

**연습문제 3.**
모델 둘(ResNet18과 ResNet50)을 따로 학습해 시험 때 소프트맥스 예측을 평균 내는 간단한 모델 앙상블을 구현하라. 앙상블의 정확도를 낱낱 모델의 정확도와 견주고 앙상블이 언제 가장 도움이 되는지 설명하라.

??? success "연습문제 3 풀이"
    ```python
    def ensemble_evaluate(models, loader, device):
        """소프트맥스 출력을 평균 내어 모델 앙상블을 평가한다."""
        for m in models:
            m.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 모든 모델의 소프트맥스 예측을 평균 낸다
                avg_probs = None
                for m in models:
                    outputs = m(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    if avg_probs is None:
                        avg_probs = probs
                    else:
                        avg_probs += probs
                avg_probs /= len(models)

                _, predicted = avg_probs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return 100. * correct / total

    # 쓰는 법:
    # model_r18 = create_model('resnet18', num_classes=10).to(device)
    # model_r50 = create_model('resnet50', num_classes=10).to(device)
    # ... 두 모델을 학습시킨다 ...
    # ensemble_acc = ensemble_evaluate([model_r18, model_r50], test_loader, device)
    ```

    앙상블은 낱낱 모델이 **서로 다른 잘못**을 저지를 때(곧 오차의 상관이 낮을 때) 가장 도움이 된다. 구조가 다른 모델(ResNet18과 ResNet50)을 쓰면 서로 다른 특징 위계를 배우므로 다양함이 생긴다. 앙상블은 대개 가장 좋은 낱낱 모델보다 정확도를 1~3% 높이며, 한 모델은 자신 있게 맞히고 다른 모델은 헷갈리는 어렵거나 애매한 예에서 이득이 가장 크다.

## 정리하며

**다룬 것** — 고급 기법

**섞인 정밀도 학습**은 모델 가중치를 FP32으로 담되 앞먹임과 역전파는 FP16으로 하여 기억 사용을 대략 절반으로 줄이고 요즘 GPU의 텐서 코어를 써서 2~3배 빨라진다.

앞의 연습문제 3개로 직접 확인할 수 있다.
