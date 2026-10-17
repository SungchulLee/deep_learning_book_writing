# 사용자 데이터셋 전이

ImageFolder의 학습·검증·테스트 분할에 ResNet18 전이 학습을 적용하고 클래스 웨이트·평가 지표를 살펴본다. 실제 사용자 자료를 읽기 전에 합성 단색 이미지 폴더를 만드는 데모이며, 동물 이름이 붙어 있어도 실제 동물 분류 데이터는 아니다.

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
    # num_workers 를 0으로 둔다. 1 이상이면 맥과 윈도가 일꾼 프로세스를
    # **새로 띄우며**(spawn) 이 각본을 처음부터 다시 읽는다. 맨 바깥에서 자료를
    # 돌리고 있으면 일꾼이 또 일꾼을 띄워 끝내 죽는다. 1 이상을 쓰려면 돌리는
    # 부분을 모두 `if __name__ == "__main__":` 안으로 넣어야 한다.
    num_workers=0, pin_memory=True if torch.cuda.is_available() else False
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True if torch.cuda.is_available() else False
)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=True if torch.cuda.is_available() else False
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
# 9단계: 로스 함수와 최적화기
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

create_synthetic_dataset은 클래스 이름과 무관하게 무작위 단색 RGB 이미지를 만든다. 따라서 cat·dog·bird의 의미 피처를 배우는 실험이 아니며 별도 테스트가 있어도 실제 과제의 품질을 입증하지 않는다. 디렉터리를 반복 사용하면 기존 파일이 남을 수 있어 실제 자료와 섞이지 않는지 확인한다.

ImageFolder의 클래스 인덱스는 폴더 이름의 정렬 순서다. 이 예시는 bird·cat·dog 순서가 되며 생성 함수의 목록 순서와 다르다. train·val·test의 class_to_idx가 같고 각 분할에 필요한 클래스가 존재하는지 확인한다. 웨이트는 학습 표본 수로만 계산하며 빈 클래스·표본과 중복을 처리해야 한다.

실제 사용하는 방식은 shuffle=True 배치와 역빈도 가중 CrossEntropyLoss다. WeightedRandomSampler는 import만 하고 사용하지 않는다. w_k=N/(K n_k)는 소수 클래스 로스의 상대 비중을 높이는 선택이지만 클래스별 성능·확률 보정을 자동으로 같게 만들지는 않는다.

모델은 layer3·layer4와 새 fc를 학습한다. 기반에는 lr=1e-4, 헤드에는 1e-3을 주며 이것이 망각을 보장해 막는 것은 아니다. model.train은 동결한 앞부분의 BatchNorm 이동 통계도 갱신한다. 검증 로스로 스케줄을 조정하지만 최고 웨이트는 검증 정확도로 선택하므로 선택 기준이 다르다.

배치 가중 로스의 단순 평균은 전체 표본 가중 평균과 다르다. classification_report·confusion_matrix의 클래스 목록·빈 클래스·zero_division 폴리시를 명시한다. ReduceLROnPlateau의 verbose 인수는 사용하는 PyTorch 버전에서 지원되는지 확인해야 한다. 이 페이지는 모듈 상단에서 자료 생성·학습을 수행하므로 import만 해도 해당 작업이 시작될 수 있다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
A=1000, B=200, C=50에서 역빈도 웨이트를 계산하고 해석하라.

</div>

??? success "풀이"
    N=1250, K=3이므로 A≈0.4167, B≈2.0833, C≈8.3333이다. 같은 비가중 로스값의 C 표본은 A보다 가중 계수가 20배다. 클래스별 로스의 실제 평균·난이도·잡음이 달라 기대 오차나 경사 기여가 정확히 같아지는 것은 아니다. 소수 클래스의 노이즈도 증폭할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
가중 샘플링과 가중 로스의 차이 및 결합 조건을 설명하라.

</div>

??? success "풀이"
    샘플러는 어떤 표본을 얼마나 자주 보는지, 가중 로스는 각 오차의 비중을 바꾼다. 둘 다 역빈도를 그대로 적용하면 소수 클래스를 이중 강조할 수 있지만 목적·목표 분포에 따라 함께 설계할 수도 있다. 배치마다 정확히 균형을 보장하지 않으며 어느 방식이 항상 좋은지는 독립 검증으로 비교한다. 이 코드에는 샘플러가 실제 연결되지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
아래 매크로·가중 F1의 의미와 누락 클래스·단위 처리 조건을 설명하라.

</div>

??? success "풀이"
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

    매크로 F1은 클래스별 F1을 같은 웨이트로, 가중 F1은 평가셋의 실제 클래스 표본 수로 평균한다. 어느 쪽도 모든 실패 비용을 대표하지 않아 클래스별 값·혼동 행렬과 함께 본다. 두 평균의 차이가 크다는 사실만으로 소수 클래스 실패를 단정하지 않는다. 이 함수의 accuracy는 0~100, F1은 0~1이고 classes 인수는 사용되지 않아 labels·zero_division 폴리시를 따로 정해야 한다.

    평가셋의 클래스 비율이 달라지면 가중 F1도 달라질 수 있다. 비교하는 분할·클래스 집합과 표본 수를 함께 보고한다.

## 정리하며

분할별 클래스 대응·가중 로스·계수 갱신 범위와 지표를 함께 확인한다. 합성 단색 자료의 학습을 실제 동물 분류 결과로 해석하지 않고, 샘플링·평가 클래스·모드와 로스 집계의 조건을 명시한다.
