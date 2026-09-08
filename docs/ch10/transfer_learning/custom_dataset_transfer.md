# 사용자 데이터셋 전이

이 예제는 부류 불균형, 알맞은 학습·검증·시험 나누기, 단순한 정확도를 넘는 두루 갖춘 평가 같은 실제 어려움을 다루며 사용자 데이터셋에 전이 학습을 쓰는 법을 보여 준다. 파이토치의 `ImageFolder`로 그림 데이터를 짜 놓기, 가중치를 준 손실 함수와 가중치를 준 무작위 표집으로 부류 불균형 다루기, 미세 조정에 층별 학습률 쓰기, 정밀도와 재현율과 F1 점수와 혼동 행렬로 평가하기를 다룬다.

## 1. 코드

```python
"""
예제 3: 사용자 데이터셋을 쓰는 전이 학습
==================================================

이 스크립트는 자기 데이터셋에 전이 학습을 쓰는 법을 보여 준다.
보여 주려고 인공 데이터셋을 만들지만, 이 코드는 알맞게 짜인
어떤 이미지 데이터셋에서도 돌도록 설계했다.

핵심 개념:
- 사용자 Dataset 클래스
- 불균형 데이터셋 다루기
- 알맞은 학습·검증·시험 나누기
- 고급 데이터 불리기
- 정확도를 넘는 평가 지표

지은이: PyTorch Transfer Learning Tutorial
날짜: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from PIL import Image
import time
import copy

# 재현성을 위한 난수 시드 설정
torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 1단계: 보여 주기 위한 인공 데이터셋 만들기
# ============================================================================

def create_synthetic_dataset():
    """사용자 데이터셋 다루기를 보이려고 인공 데이터셋을 만든다."""
    print("Creating synthetic dataset for demonstration...")

    base_dir = './custom_dataset'
    splits = ['train', 'val', 'test']
    classes = ['cat', 'dog', 'bird']

    class_counts = {
        'train': {'cat': 500, 'dog': 300, 'bird': 200},
        'val': {'cat': 100, 'dog': 60, 'bird': 40},
        'test': {'cat': 100, 'dog': 60, 'bird': 40}
    }

    for split in splits:
        for cls in classes:
            path = os.path.join(base_dir, split, cls)
            os.makedirs(path, exist_ok=True)

            n_images = class_counts[split][cls]
            for i in range(n_images):
                img = Image.new('RGB', (64, 64),
                               color=(np.random.randint(0, 255),
                                     np.random.randint(0, 255),
                                     np.random.randint(0, 255)))
                img.save(os.path.join(path, f'{cls}_{i:04d}.jpg'))

    print(f"Synthetic dataset created at: {base_dir}")
    return base_dir

data_dir = create_synthetic_dataset()

# ============================================================================
# 2단계: 장치 설정
# ============================================================================

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# ============================================================================
# 3단계: 고급 데이터 불리기
# ============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    transforms.RandomErasing(p=0.2)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# ============================================================================
# 4단계: ImageFolder로 사용자 데이터셋 싣기
# ============================================================================

print("\nLoading custom dataset...")

train_dataset = ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transform)
val_dataset = ImageFolder(root=os.path.join(data_dir, 'val'), transform=val_transform)
test_dataset = ImageFolder(root=os.path.join(data_dir, 'test'), transform=val_transform)

classes = train_dataset.classes
class_to_idx = train_dataset.class_to_idx

print(f"\nDataset Information:")
print(f"Number of classes: {len(classes)}")
print(f"Classes: {classes}")
print(f"Training: {len(train_dataset)} images")
print(f"Validation: {len(val_dataset)} images")
print(f"Test: {len(test_dataset)} images")

# ============================================================================
# 5단계: 부류 분포 살피기
# ============================================================================

def get_class_distribution(dataset):
    """부류마다의 표본 수를 셈한다."""
    class_counts = {}
    for _, label in dataset.imgs:
        class_name = dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    return class_counts

train_class_counts = get_class_distribution(train_dataset)

print("\nTraining set class distribution:")
total = sum(train_class_counts.values())
for cls, count in sorted(train_class_counts.items()):
    percentage = 100 * count / total
    print(f"{cls:>10s}: {count:>4d} images ({percentage:>5.1f}%)")

# ============================================================================
# 6단계: 부류 불균형 다루기
# ============================================================================

class_weights = []
for cls in classes:
    count = train_class_counts[cls]
    weight = total / (len(classes) * count)
    class_weights.append(weight)

class_weights = torch.FloatTensor(class_weights).to(device)

print("\nClass weights for balanced loss:")
for cls, weight in zip(classes, class_weights):
    print(f"{cls:>10s}: {weight:.3f}")

# ============================================================================
# 7단계: 데이터 로더 만들기
# ============================================================================

BATCH_SIZE = 32

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

# ============================================================================
# 8단계: 모델 싣고 설정하기
# ============================================================================

print("\nLoading pre-trained ResNet18...")
model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if any(layer in name for layer in ['layer3', 'layer4']):
        param.requires_grad = True

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))
model = model.to(device)

# ============================================================================
# 9단계: 손실 함수와 최적화기
# ============================================================================

criterion = nn.CrossEntropyLoss(weight=class_weights)

finetune_params = []
for name, param in model.named_parameters():
    if param.requires_grad and 'fc' not in name:
        finetune_params.append(param)

classifier_params = [model.fc.weight, model.fc.bias]

optimizer = optim.Adam([
    {'params': finetune_params, 'lr': 0.0001},
    {'params': classifier_params, 'lr': 0.001}
])

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3, verbose=True
)

# ============================================================================
# 10단계: 학습 함수와 평가 함수
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    """한 에폭을 학습한다."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
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

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """모델을 평가한다."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)

# ============================================================================
# 11단계: 학습 고리
# ============================================================================

NUM_EPOCHS = 20

print(f"\n{'='*70}")
print(f"Starting training for {NUM_EPOCHS} epochs...")
print(f"{'='*70}\n")

best_val_acc = 0.0
best_model_weights = copy.deepcopy(model.state_dict())
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    print("-" * 70)

    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device
    )
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")

    scheduler.step(val_loss)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_weights = copy.deepcopy(model.state_dict())
        print("New best model saved")
    print()

total_time = time.time() - start_time
print(f"Training completed in {total_time // 60:.0f}m {total_time % 60:.0f}s")
print(f"Best validation accuracy: {best_val_acc:.2f}%")

model.load_state_dict(best_model_weights)

# ============================================================================
# 12단계: 두루 갖춘 시험 평가
# ============================================================================

print("\nFinal Test Evaluation:")
print("="*70)

test_loss, test_acc, test_preds, test_labels = evaluate(
    model, test_loader, criterion, device
)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.2f}%")

print("\nDetailed Classification Report:")
print("-" * 70)
print(classification_report(test_labels, test_preds, target_names=classes))

print("\nConfusion Matrix:")
cm = confusion_matrix(test_labels, test_preds)
print("Predicted ->")
print(f"{'Actual':>10s} | " + " | ".join(f"{c:>6s}" for c in classes))
print("-" * (12 + 10 * len(classes)))
for i, cls in enumerate(classes):
    print(f"{cls:>10s} | " + " | ".join(f"{cm[i][j]:>6d}" for j in range(len(classes))))

torch.save(model.state_dict(), 'custom_dataset_model.pth')
print("\nModel saved as 'custom_dataset_model.pth'")


if __name__ == "__main__":
    pass
```

## 2. 논의

사용자 데이터셋을 다룰 때 **부류 불균형**은 가장 흔하고 영향이 큰 어려움 가운데 하나이다. 학습 집합에 고양이 그림이 500장인데 새 그림이 200장뿐이면(2.5 대 1의 비율) 순진한 모델이 그저 "고양이"를 더 자주 맞혀도 어지간한 정확도를 얻는다. 가중치를 준 교차 엔트로피 손실은 적게 나타나는 부류에 더 큰 가중치를 주어 이를 다룬다. $w_k = N / (K \cdot n_k)$이며 여기서 $N$은 전체 표본 수, $K$은 부류 수, $n_k$은 부류 $k$의 개수이다. 그러면 드문 부류를 잘못 맞히는 비용이 커져 모델이 모든 부류에 고르게 마음을 쓰게 된다.

**층별 학습률**은 모델의 부분마다 다른 학습률을 준다. 미세 조정되는 사전 학습 층은 사전 학습된 특징을 조금씩 조심스레 고치도록 더 낮은 학습률(이를테면 $10^{-4}$)을 받고, 새로 더한 분류기는 무작위 초기화에서 시작해 더 빨리 배워야 하므로 더 높은 학습률(이를테면 $10^{-3}$)을 받는다. 그러면 무작위로 초기화된 분류기의 큰 기울기가 학습 초기에 배운 특징 표현을 흐트러뜨리는 일을 막는다.

정확도 너머로 불균형 데이터셋의 평가에는 **부류별 지표**가 필요하다. **정밀도**는 양성 예측 가운데 얼마가 맞는지를 재고(거짓 양성이 비쌀 때 중요하다), **재현율**은 실제 양성 가운데 얼마를 찾았는지를 재며(거짓 음성이 비쌀 때 중요하다), **F1 점수**는 둘의 조화 평균으로 둘의 균형을 잡는 하나의 지표를 준다. **혼동 행렬**은 더 자세히, 어느 부류끼리 헷갈리는지를 꼭 집어 보여 준다. 이를테면 고양이가 개로는 자주 잘못 분류되는데 새로는 결코 그러지 않는다면, 겨냥한 데이터 모으기나 불리기로 다룰 수 있는 특정한 실패 모습이 드러난다.

## 연습문제

**연습문제 1.**
부류 A의 표본 1000개, 부류 B의 표본 200개, 부류 C의 표본 50개인 데이터셋이 주어졌을 때 역빈도 공식 $w_k = N / (K \cdot n_k)$으로 부류 가중치를 셈하라. 부류 C의 가중치가 부류 A보다 훨씬 큰 까닭을 설명하라.

??? success "연습문제 1 풀이"
    전체 표본은 $N = 1000 + 200 + 50 = 1250$개이고 부류 수는 $K = 3$이다.

    - $w_A = 1250 / (3 \times 1000) = 0.417$
    - $w_B = 1250 / (3 \times 200) = 2.083$
    - $w_C = 1250 / (3 \times 50) = 8.333$

    부류 C의 가중치(8.333)는 부류 A의 가중치(0.417)보다 $20$배 크다. 잘못 분류한 C 표본 하나가 잘못 분류한 A 표본보다 손실에 20배 더 이바지한다는 뜻이다. 부류 C가 20배 드물게 나타나므로 가중치를 다시 주지 않으면 모델이 세대마다 C의 예를 훨씬 적게 보아 그 결정 경계를 배울 까닭이 거의 없다는 것이 그 이치이다. 이 가중치가 부류마다 기대 손실 몫을 같게 하여 사실상 균형 잡힌 데이터셋을 흉내 낸다.

---

**연습문제 2.**
부류 불균형을 다루는 방법으로 `WeightedRandomSampler`와 가중치를 준 `CrossEntropyLoss`를 견주어라. 언제 어느 쪽이 더 나은가? 둘을 함께 쓸 수 있는가?

??? success "연습문제 2 풀이"
    **WeightedRandomSampler**는 데이터를 실을 때 소수 부류를 더 많이 뽑아 배치마다 부류별 표본 수가 얼추 같아진다. 모델이 소수 부류의 예를 더 자주 본다는 뜻이지만 손실 계산 하나하나에는 가중치가 없다.

    **가중치를 준 CrossEntropyLoss**는 고르게 뽑되 소수 부류에 더 큰 손실 가중치를 주어 드문 부류의 잘못을 더 비싸게 만든다.

    데이터 불리기가 중요하면 **WeightedRandomSampler**가 낫다. 소수 부류의 그림이 더 자주 뽑히므로 불린 판이 더 많이 나오기 때문이다. 코드가 더 간단하고 세대 길이가 정해지기를 바라면 **가중치를 준 손실**이 낫다. 둘을 함께 쓰면 지나치게 바로잡을 수 있다. 소수 부류가 더 많이 뽑히고 가중치도 더 받아 모델이 소수 예에 과적합될 수 있다. 실제로는 한 가지를 고르라. 소수 표본의 불린 시야가 더 많은 다양함을 주므로 이미지 과제에서는 WeightedRandomSampler를 흔히 더 좋아한다.

---

**연습문제 3.**
거시 평균 F1 점수와 가중 평균 F1 점수도 함께 셈해 돌려주도록 평가 함수를 고쳐라. 이 두 평균 방식의 차이와 각각이 언제 더 알맞은지 설명하라.

??? success "연습문제 3 풀이"
    ```python
    from sklearn.metrics import f1_score

    def evaluate_with_f1(model, loader, criterion, device, classes):
        """거시 F1과 가중 F1 점수로 평가한다."""
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        preds = np.array(all_preds)
        labels = np.array(all_labels)

        accuracy = 100.0 * (preds == labels).sum() / len(labels)
        macro_f1 = f1_score(labels, preds, average='macro')
        weighted_f1 = f1_score(labels, preds, average='weighted')

        return running_loss / len(loader), accuracy, macro_f1, weighted_f1
    ```

    **거시 평균 F1**은 부류마다 따로 F1을 셈해 가중치 없이 평균 낸다. 빈도와 무관하게 모든 부류를 똑같이 다루므로, 흔한 부류를 잘해도 드문 부류를 못하면 거시 F1이 낮게 나온다. 모든 부류가 똑같이 중요할 때 쓴다.

    **가중 평균 F1**은 부류마다 F1에 그 뒷받침(참 사례의 수)으로 가중치를 준다. 흔한 부류에 더 큰 영향을 주어 이름표 불균형을 셈에 넣는다. 부류의 흔함에 비례하는 전체 성능을 하나의 지표로 보고 싶을 때 쓴다. 불균형 데이터셋에서 거시 F1과 가중 F1의 차이가 크면 모델이 소수 부류에서 헤맨다는 뜻이다.

## 정리하며

**다룬 것** — 사용자 데이터셋 전이

사용자 데이터셋을 다룰 때 **부류 불균형**은 가장 흔하고 영향이 큰 어려움 가운데 하나이다.

앞의 연습문제 3개로 직접 확인할 수 있다.
