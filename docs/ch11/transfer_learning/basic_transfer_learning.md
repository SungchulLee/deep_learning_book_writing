# 기본 전이 학습

전이 학습은 (대개 ImageNet인) 큰 데이터셋으로 학습한 모델을 새롭고 흔히 더 작은 데이터셋에 맞추는 기법이다. 가장 간단한 꼴인 **특징 뽑기**에서는 사전 학습 층을 모두 얼리고 새 분류 머리만 학습하여, 그림 수백만 장에서 배운 강력한 특징 표현을 쓴다. 이 방법은 놀랍도록 잘 통한다. 학습하는 매개변수가 아주 적어도, 특히 목표 데이터셋이 작을 때 전이 학습이 맨바닥부터 학습하는 것을 앞서는 일이 많다.

## 1. 코드

```python
"""
예제 1: 기본 전이 학습 (특징 뽑기)
========================================================

이 스크립트는 파이토치로 전이 학습의 기본을 보여 준다.
사전 학습된 ResNet18 모델을 CIFAR-10 분류에 맞추어 쓴다.

지은이: PyTorch Transfer Learning Tutorial
날짜: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
import time
import copy

# 재현성을 위해 난수 씨앗 고정
torch.manual_seed(42)

# ============================================================================
# 1단계: 장치 설정
# ============================================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================================
# 2단계: 데이터 마련
# ============================================================================

# ImageNet 정규화 값
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 학습 데이터의 변환을 정한다
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# 시험 데이터의 변환을 정한다
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

print("\nDownloading and loading CIFAR-10 dataset...")

train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=test_transform
)

BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

classes = ['plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

print(f"\nDataset loaded successfully!")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: {len(classes)}")

# ============================================================================
# 3단계: 사전 학습 모델 싣기
# ============================================================================

print("\nLoading pre-trained ResNet18 model...")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print("Pre-trained model loaded successfully!")

# ============================================================================
# 4단계: 특징 뽑개 얼리기
# ============================================================================

print("\nFreezing all layers except the final classifier...")
for param in model.parameters():
    param.requires_grad = False

# ============================================================================
# 5단계: 마지막 층 바꾸기
# ============================================================================

num_features = model.fc.in_features
print(f"\nReplacing final layer:")
print(f"- Input features: {num_features}")
print(f"- Output classes: {len(classes)} (CIFAR-10)")

model.fc = nn.Linear(num_features, len(classes))
model = model.to(device)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")

# ============================================================================
# 6단계: 손실 함수와 최적화기 정하기
# ============================================================================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
)

# ============================================================================
# 7단계: 학습 함수
# ============================================================================

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """모델을 한 세대 학습한다."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch {batch_idx + 1}/{len(train_loader)}: '
                  f'Loss: {running_loss / (batch_idx + 1):.3f}, '
                  f'Acc: {100. * correct / total:.2f}%')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

# ============================================================================
# 8단계: 평가 함수
# ============================================================================

def evaluate(model, test_loader, criterion, device):
    """시험 데이터로 모델을 평가한다."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# ============================================================================
# 9단계: 학습 고리
# ============================================================================

NUM_EPOCHS = 10

print(f"\n{'='*70}")
print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"{'='*70}\n")

best_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 70)

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)

    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        print(f"  New best accuracy! Saving model...")

    print()

total_time = time.time() - start_time
print(f"{'='*70}")
print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
print(f"Best test accuracy: {best_acc:.2f}%")
print(f"{'='*70}\n")

model.load_state_dict(best_model_weights)

# ============================================================================
# 10단계: 마지막 평가
# ============================================================================

print("Final Evaluation on Test Set:")
print("-" * 70)
final_loss, final_acc = evaluate(model, test_loader, criterion, device)
print(f"Final Test Loss: {final_loss:.4f}")
print(f"Final Test Accuracy: {final_acc:.2f}%")

# ============================================================================
# 11단계: 부류별 정확도
# ============================================================================

print("\nPer-class accuracy:")
print("-" * 70)

class_correct = [0] * len(classes)
class_total = [0] * len(classes)

model.eval()
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

for i in range(len(classes)):
    accuracy = 100 * class_correct[i] / class_total[i]
    print(f"{classes[i]:>10s}: {accuracy:>6.2f}%")

# 모델을 저장한다
torch.save(model.state_dict(), 'resnet18_cifar10_transfer.pth')
print("\nModel saved as 'resnet18_cifar10_transfer.pth'")


if __name__ == "__main__":
    pass
```

## 2. 논의

특징 뽑기의 핵심 생각은 ImageNet으로 사전 학습한 신경망의 합성곱 층이 이미 넉넉한 시각 특징의 위계를 배웠다는 것이다. 앞 층의 모서리와 결에서 뒤 층의 물체 부분과 뜻 개념까지이다. 이 층을 얼려(`requires_grad = False`으로 두어) 그 배운 지식을 지키고, 뽑은 특징을 우리 부류로 잇대는 마지막 분류 층만 학습한다. 부류가 10개인 CIFAR-10과 특징 출력이 512차원인 ResNet18에서는 모두 약 1120만 개 가운데 $512 \times 10 + 10 = 5{,}130$개만 학습한다는 뜻이다.

사전 학습된 ImageNet 모델을 쓸 때 매우 중요한 세부는 **입력 정규화**이다. 사전 학습 가중치는 ImageNet 통계(RGB 통로마다 평균 $[0.485, 0.456, 0.406]$, 표준편차 $[0.229, 0.224, 0.225]$)로 정규화한 입력을 바란다. 다른 정규화 통계를 쓰면 입력 분포가 모델이 사전 학습 때 본 것에서 벗어나 뽑은 특징의 질이 떨어진다. 마찬가지로 ResNet18은 $224 \times 224$ 그림을 바라므로 CIFAR-10의 본디 $32 \times 32$ 그림을 키워야 한다. `Resize(256)` 뒤에 `CenterCrop(224)`을 두는 파이프라인이 이를 다룬다.

특징 뽑기의 학습 움직임은 남다르다. 선형 분류기만 학습하므로 수렴이 대개 빠르고(흔히 5~10 세대 안에) 최적화 지형이 볼록하다. 최적화기는 고정된 특징 공간에서 가장 좋은 선형 결정 경계만 찾으면 된다. 그래서 초매개변수 선택에 튼튼하다. 기본 설정의 간단한 Adam 최적화기로도 잘 된다. 가장 큰 한계는 얼린 특징이 목표 분야에 딱 맞지 않을 수 있다는 것인데, 목표 분야가 ImageNet과 크게 다르면 (층을 얼마간 녹이는) 미세 조정을 하게 되는 까닭이다.

## 연습문제

**연습문제 1.**
이 코드는 학습과 시험에 모두 `CenterCrop(224)`을 쓴다. 학습 변환을 `RandomCrop(224, padding=4)`과 `RandomHorizontalFlip()`으로 바꾸고 10 세대 뒤의 시험 정확도를 견주어라. 맨바닥부터 학습할 때에 견주어 특징 뽑기에서 데이터 불리기의 영향이 작을 법한 까닭은 무엇인가?

??? success "연습문제 1 풀이"
    ```python
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    ```

    특징 뽑기에서는 사전 학습된 특징 뽑개가 얼어 있어 데이터 불리기의 영향이 작다. 뒤집기나 자르기 같은 불리기가 입력을 바꾸지만 얼린 합성곱 신경망은 같은 고정 가중치로 처리한다. 그 특징 위에서 선형 분류기만 학습하고 선형 분류기는 그릇이 작으므로, 불리기는 주로 (매개변수가 5,130개뿐이어서 어차피 심하게 과적합될 일이 없는) 선형 층의 과적합을 막는 데 도움이 된다. 맨바닥부터 학습할 때는 불리기가 신경망 전체가 학습 집합을 외우는 것을 막고 더 일반화되는 특징을 배우도록 북돋우므로 매우 중요하다.

---

**연습문제 2.**
ResNet18을 특징 뽑개로 쓸 때(마지막 선형 층만 학습할 때) 학습하는 매개변수가 전체의 몇 퍼센트인지 셈하라. ResNet50에 대해서도 되풀이하라. 모델이 클수록 그 비율이 줄어드는 까닭은 무엇이며 전이 학습의 효율에 대해 무엇을 뜻하는가?

??? success "연습문제 2 풀이"
    **ResNet18**: 전체 매개변수는 약 $11{,}689{,}512$개이다. 마지막 층은 $512 \times 10 + 10 = 5{,}130$개이다. 학습 비율은 $5{,}130 / 11{,}689{,}512 \approx 0.044\%$이다.

    **ResNet50**: 전체 매개변수는 약 $25{,}557{,}032$개이다. 마지막 층은 $2{,}048 \times 10 + 10 = 20{,}490$개이다. 학습 비율은 $20{,}490 / 25{,}557{,}032 \approx 0.080\%$이다.

    흥미롭게도 ResNet50의 비율이 조금 더 높은데, 특징 차원(2048)이 ResNet18(512)보다 커서 선형 층이 더 크기 때문이다. 그래도 둘 다 0.1%에 한참 못 미친다. 이는 전이 학습의 놀라운 효율을 드러낸다. 매개변수 수백만 개어치의 배운 표현을 쓰면서 아주 작은 몫만 다듬는다. 큰 모델은 미세 조정을 그에 비례해 더 하지 않고도 (차원이 높은) 더 넉넉한 특징을 주는데, 전이 학습이 모델 크기에 따라 유리하게 커짐을 뜻한다.

---

**연습문제 3.**
시험 그림 500장에 대해 얼린 ResNet18 부호기에서 512차원 특징을 뽑고 t-SNE로 2차원으로 줄인 뒤 부류 이름표로 색을 칠한 산점도를 그려 특징 공간을 보여 주는 함수를 구현하라. 선형 머리를 학습하기 전과 후에 돌려 보아라. 특징 공간이 바뀌는가?

??? success "연습문제 3 풀이"
    ```python
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    def visualize_features(model, loader, device, n_samples=500, title="Feature Space"):
        """특징을 뽑아 t-SNE로 그려 본다."""
        model.eval()
        features = []
        labels = []

        with torch.no_grad():
            for inputs, targets in loader:
                if len(features) * BATCH_SIZE >= n_samples:
                    break
                inputs = inputs.to(device)
                # 마지막 층 앞에서 특징을 뽑는다
                x = inputs
                for name, module in model.named_children():
                    if name == 'fc':
                        break
                    x = module(x)
                x = x.view(x.size(0), -1)
                features.append(x.cpu().numpy())
                labels.append(targets.numpy())

        features = np.concatenate(features)[:n_samples]
        labels = np.concatenate(labels)[:n_samples]

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)

        plt.figure(figsize=(10, 8))
        for i, cls in enumerate(classes):
            mask = labels == i
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       label=cls, alpha=0.6, s=10)
        plt.legend()
        plt.title(title)
        plt.show()
    ```

    부호기 층이 얼어 있으므로(`requires_grad = False`) 선형 머리를 학습하기 전과 후에 특징 공간은 바뀌지 **않는다**. 512차원 특징은 두 경우에 똑같다. 바뀌는 것은 그 특징에서 부류 로짓으로 가는 선형 잇댐뿐이다. t-SNE 그림은 학습 전후가 같아 보일 것이다. 사전 학습된 특징이 이미 잘 갈라진 무리를 이루면 선형 머리가 할 일이 쉬운데, 특징 뽑기가 그토록 잘 통하는 까닭이 이것이다.

## 정리하며

**다룬 것** — 기본 전이 학습

특징 뽑기의 핵심 생각은 ImageNet으로 사전 학습한 신경망의 합성곱 층이 이미 넉넉한 시각 특징의 위계를 배웠다는 것이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
