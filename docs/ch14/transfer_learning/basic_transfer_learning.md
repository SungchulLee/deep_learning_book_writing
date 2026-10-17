# 기본 전이 학습

ImageNet 사전 학습 ResNet18의 계수를 동결하고 CIFAR-10의 새 분류 헤드를 학습한다. 고정 피처를 활용하는 전이 학습의 출발점이다. 계수 동결과 모델의 평가 모드·BatchNorm 버퍼 고정은 서로 다른 설정이라는 점을 함께 확인한다.

## 1. 코드

```python
"""
예제 1: 기본 전이 학습 (피처 뽑기)
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
# 4단계: 피처 뽑개 얼리기
# ============================================================================

print("\nFreezing all layers except the final classifier...")
for param in model.parameters():
    param.requires_grad = False

# ============================================================================
# 5단계: 마지막 레이어 바꾸기
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
# 6단계: 로스 함수와 최적화기 정하기
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

ResNet18의 모든 계수를 requires_grad=False로 둔 뒤 fc를 512→10으로 교체한다. 새 헤드는 5,130개, 전체는 11,181,642개 계수다. 새 레이어는 기본적으로 경사가 켜져 있으며 옵티마이저에는 학습 가능한 계수만 전달한다.

**동결의 제한:** train_one_epoch는 model.train()을 호출해 BatchNorm의 이동 통계도 갱신한다. 컨볼루션·아핀 계수가 고정되어도 평가 피처가 학습 전후 바뀔 수 있으므로 “완전히 고정한 피처 추출기”는 아니다. 피처를 고정하려면 기반 모델의 모드·버퍼 폴리시까지 정해야 한다.

전처리는 RGB를 256으로 확대하고 224 중앙 크롭·ImageNet 정규화를 적용한다. 이는 사용한 사전 학습 웨이트의 입력 관례와 맞추는 선택이다. ResNet의 연산이 모든 다른 해상도를 원천적으로 거부한다는 뜻은 아니다. 목표 데이터에서 다른 해상도·정규화를 선택할 때는 품질·비용을 평가한다.

같은 고정 피처에 선형 레이어와 교차 엔트로피만 최적화하면 계수에 대해 볼록한 문제다. 현재 코드는 BatchNorm 통계와 입력 조건이 달라질 수 있어 그 단순한 조건으로 모든 수렴을 설명할 수 없다. 정해진 5~10에포크 수렴이나 하이퍼파라미터 둔감성이 보장되지는 않는다.

각 에포크의 공식 테스트 정확도로 최고 웨이트를 골라 최종 테스트를 다시 평가한다. 그 테스트는 이미 모델 선택에 사용되어 독립 최종 성능이 아니다. 학습셋 안에 별도 검증 분할을 두어야 한다. 로스는 배치 평균의 단순 평균이며 저장 파일에는 웨이트만 들어 있어 학습 재시작용 상태는 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
아래 어그멘테이션 설정의 적용 위치와 공정한 성능 비교 조건을 설명하라.

</div>

??? success "풀이"
    ```python
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    ```

    train_transform 변수만 재정의해도 이미 생성한 train_dataset.transform은 자동으로 교체되지 않는다. 데이터셋·로더 연결을 확인하고 검증 분할에서 비교해야 한다. 동결 인코더도 어그멘테이션 입력마다 다른 피처를 내므로 어그멘테이션 영향이 반드시 작거나 작은 헤드가 과적합하지 않는다는 주장은 성립하지 않는다. 각 조건의 초기 웨이트·BatchNorm 폴리시와 학습 예산을 맞춘다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
10개 클래스 헤드로 교체한 ResNet18·50의 전체·학습 계수 비율을 계산하라.

</div>

??? success "풀이"
    ResNet18은 전체 11,181,642개 중 5,130개로 약 0.0459%, ResNet50은 전체 23,528,522개 중 20,490개로 약 0.0871%다. 1000클래스 원래 전체 수를 분모로 두면 현재 모델의 비율이 아니다. 학습 비율이 작아도 전체 추론 계수·연산은 남으며 큰 모델의 전이가 항상 유리하다는 결론은 아니다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
아래 t-SNE 함수로 헤드 학습 전후 피처가 같은지 판단할 때 무엇을 확인해야 하는가?

</div>

??? success "풀이"
    ```python
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    def visualize_features(model, loader, device, n_samples=500, title="Feature Space"):
        """피처를 뽑아 t-SNE로 그려 본다."""
        model.eval()
        features = []
        labels = []

        with torch.no_grad():
            for inputs, targets in loader:
                if len(features) * BATCH_SIZE >= n_samples:
                    break
                inputs = inputs.to(device)
                # 마지막 레이어 앞에서 피처를 뽑는다
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

    계수뿐 아니라 BatchNorm 버퍼·평가 모드·입력 변환이 같아야 같은 피처가 나온다. 현재 학습은 BatchNorm 통계를 갱신하므로 학습 전후 같다고 보장할 수 없다. 먼저 원래 고차원 피처를 비교하고 t-SNE는 이웃 구조의 시각 보조로만 쓴다. 표본 수는 perplexity보다 커야 하며 함수의 전역 BATCH_SIZE·classes 의존과 실제 로더 배치 크기도 확인한다. 새 그림은 이번 편집에서 만들지 않았다.

## 정리하며

새 분류 헤드 학습과 기반 피처의 동결 범위를 명확히 한다. BatchNorm 버퍼·입력 전처리·모델 선택 분할을 함께 관리하고, 작은 학습 계수 비율이나 t-SNE 그림을 일반화 품질의 보장으로 해석하지 않는다.
