# 2단계: 선형 모델과 소프트맥스

[1단계](01_template_learning.md)에서 템플릿을 클래스 평균으로 **못박아** 82.03%를 얻었다. 그 절 끝에서 보았듯 이 방법은 선형 분류기이고, 가중치가 평균으로 고정되어 있을 뿐이다. 이제 그 가중치를 풀어 데이터에 맞추어 움직인다. 이것이 이 장에서 처음으로 "학습"이라 부를 만한 일이다.

그런데 무엇을 기준으로 움직여야 하는가? 이 절은 그 물음에 답하고, 답을 따라가면 **최대가능도 추정 → 가능도 → 로그가능도 → 손실 → 경사 하강법**이라는 하나의 사슬이 나온다. 딥러닝의 모든 학습이 이 사슬 위에 있다.

## 1. 가중치를 어떻게 찾는가

### 모델

28×28 이미지를 784차원 벡터로 펼치고, 아핀 변환으로 클래스마다 점수 하나씩을 만든다.

$$
y = xA + b, \qquad A \in \mathbb{R}^{784 \times 10}, \quad b \in \mathbb{R}^{10}
$$

찾아야 할 수는 $784 \times 10 + 10 = 7850$개다. 점수 $y$는 아직 확률이 아니므로 소프트맥스로 바꾼다.

$$
p_k = \frac{e^{y_k}}{\sum_{j=0}^{9} e^{y_j}}
$$

모델의 정의는 여기서 끝이다. 남은 것은 7850개의 수를 정하는 일뿐이다. 자세한 유도는 [3장의 소프트맥스 회귀 기초](../../ch03/softmax_regression/01_fundamentals.md)에 있다.

### 최대가능도 추정

기준은 이것이다. **관찰한 데이터를 가장 그럴듯하게 만드는 매개변수를 고른다.** 이를 최대가능도 추정(maximum likelihood estimation, MLE)이라 한다.

말이 추상적이니 매개변수가 하나뿐인 예로 먼저 보자. 호수의 물고기 수 $N$을 추정한다. 3마리를 잡아 표지를 붙여 놓아주고, 나중에 5마리를 잡았더니 그중 1마리에 표지가 있었다. 이 결과가 나올 확률은 $N$의 함수이다.

$$
P_N = \frac{\binom{3}{1}\binom{N-3}{4}}{\binom{N}{5}}
$$

$N$이 데이터가 아니라 **매개변수**이고, $P_N$이 그 매개변수의 함수인 **가능도**라는 점이 핵심이다. 최대가능도 추정은 $P_N$을 가장 크게 하는 $N$을 고르는 것이며, 계산해 보면 $N = 14$와 $15$에서 $45/91 \approx 0.4945$로 비긴다. 유도는 [표지-재포획 MLE](../../ch03/mle/capture_recapture_mle.md)에 있다.

여기서 매개변수가 하나이고 정수였기 때문에 이웃한 값끼리 견주는 것만으로 최댓값을 찾을 수 있었다. MNIST에서는 매개변수가 7850개이고 모두 실수이므로 그 방법을 쓸 수 없다. 그래서 사슬의 나머지가 필요해진다.

### 가능도에서 손실로

학습 데이터 $(x_1, y_1), \ldots, (x_n, y_n)$이 서로 독립이라고 보면 가능도는 곱으로 쓰인다.

$$
L(A, b) = \prod_{i=1}^{n} p_{y_i}(x_i; A, b)
$$

이 곱을 그대로 다루기는 어렵다. 항이 6만 개나 되고 각 항이 1보다 작아 곱하면 순식간에 0으로 내려가 버린다(부동소수점에서 실제로 0이 된다). 그래서 **로그**를 취한다.

$$
\ell(A, b) = \log L(A, b) = \sum_{i=1}^{n} \log p_{y_i}(x_i; A, b)
$$

로그는 단조 증가 함수이므로 최댓값의 **위치**를 바꾸지 않는다. 곱이 합으로 바뀌어 미분이 쉬워지고, 아주 작은 확률도 큰 음수로 표현되어 수치적으로 안정해진다. 이것이 로그가능도이다.

마지막으로 부호를 뒤집는다. 최적화 도구는 관례상 최소화를 하도록 만들어져 있기 때문이다.

$$
\text{loss} = -\ell(A, b) = -\sum_{i=1}^{n} \log p_{y_i}(x_i; A, b)
$$

가능도를 **최대화**하는 일과 이 손실을 **최소화**하는 일은 정확히 같은 문제이다. 봉우리를 뒤집어 골짜기로 만든 것뿐이다.

그리고 이 손실에는 이미 이름이 있다. **교차 엔트로피**이다. 분류 문제에서 교차 엔트로피를 쓰는 까닭이 여기 있다. 임의로 고른 편리한 함수가 아니라, 최대가능도 추정에서 저절로 따라 나온 것이다.

### 경사 하강법

이제 7850차원 공간에서 손실이 가장 낮은 지점을 찾아야 한다. 손실을 각 매개변수로 미분해 얻은 경사는 가장 가파르게 **올라가는** 방향을 가리키므로, 그 반대로 조금씩 내려간다.

$$
\theta_{n+1} = \theta_n - \lambda \frac{\partial \ell}{\partial \theta}
$$

$\lambda$는 **학습률**이며 한 걸음의 크기를 정한다. 너무 작으면 수렴이 더디고 얕은 극소점에 갇히기 쉬우며, 너무 크면 최솟값을 지나쳐 진동하거나 아예 발산한다. 자세한 내용은 [학습률과 이동 폭](../gradient_descent/learning_rate.md)에서 다룬다.

미분값 $\partial \ell / \partial \theta$을 손으로 구할 필요는 없다. PyTorch의 [자동 미분](../autograd/01_basic_scalar_backward.md)이 계산 그래프를 거슬러 올라가며 대신 구해 준다.

### 사슬 전체

| 단계 | 하는 일 | 왜 |
|---|---|---|
| 최대가능도 추정 | 데이터를 가장 그럴듯하게 만드는 $(A,b)$를 고른다 | 학습의 기준을 정한다 |
| 가능도 $L$ | 확률들의 곱 | 매개변수의 함수로 본다 |
| 로그가능도 $\ell$ | 곱을 합으로 | 미분이 쉽고 수치적으로 안정 |
| 손실 $-\ell$ | 부호 뒤집기 | 최대화를 최소화로 |
| 경사 하강법 | 조금씩 내려가기 | 7850차원에서 최솟값 찾기 |

아래 코드는 이 사슬을 그대로 옮긴 것이다. `nn.CrossEntropyLoss`가 손실 $-\ell$이고, `loss.backward()`가 $\partial \ell / \partial \theta$이며, `optimizer.step()`이 $\theta - \lambda g$이다.

---

## 2. 코드

```python
"""
===============================================================================
2단계: MNIST에서의 선형 모델과 소프트맥스 회귀
===============================================================================
어려움: 가운데
미리 알아 둘 것: 2.1 템플릿 학습, 그림 데이터 이해
학습 목표:
  - 참 세상 그림 데이터셋(MNIST)을 다룬다
  - 데이터 불러오기와 배치 만들기를 다룬다
  - 학습/검증/시험 나누기를 제대로 짠다
  - 데이터 로더와 작은 배치 익히기를 쓴다
  - 그림에 대한 예측을 그림으로 본다

소요 시간: 45~60분
===============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import time

# 재현성을 위한 난수 시드 설정
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("STAGE 2: LINEAR MODEL + SOFTMAX ON MNIST")
print("=" * 80)


# =============================================================================
# 1부: MNIST 데이터셋 불러와 살펴보기
# =============================================================================
print("\n" + "=" * 80)
print("PART 1: Loading MNIST Dataset")
print("=" * 80)

"""
MNIST 데이터셋:
--------------
- 손글씨 숫자 그림 70,000장(0~9)
- 28x28 화소, 잿빛
- 학습 그림 60,000장
- 시험 그림 10,000장
- 기계 학습에서 가장 이름난 데이터셋의 하나다!
"""

# 데이터에 적용할 변환을 정의한다
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL Image to tensor (0-1 range)
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with MNIST mean and std
])

# 학습 데이터를 내려받아 불러온다
print("Downloading MNIST dataset...")
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# 시험 데이터를 내려받아 불러온다
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

print(f"✅ Dataset loaded successfully!")
print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Image shape: {train_dataset[0][0].shape}")  # (1, 28, 28) - CHW format
print(f"Number of classes: {len(train_dataset.classes)}")


# =============================================================================
# 2부: 표본 이미지 시각화
# =============================================================================
print("\n" + "=" * 80)
print("PART 2: Exploring the Data")
print("=" * 80)

def show_images(dataset, num_images=10):
    """
    데이터셋에서 보기 그림을 격자로 보인다.
    
    Args:
        dataset: PyTorch 데이터셋
        num_images: 보일 그림의 수
    """
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(num_images):
        img, label = dataset[i]
        # CHW에서 HW로 바꾼다 (회색조이므로 채널 차원을 없앤다)
        img = img.squeeze().numpy()
        
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# 표본 이미지를 보려면 주석을 푼다
# show_images(train_dataset)
# plt.show()

# 표본 하나를 살펴본다
sample_img, sample_label = train_dataset[0]
print(f"\nSample image shape: {sample_img.shape}")  # (1, 28, 28)
print(f"Sample label: {sample_label}")
print(f"Pixel value range: [{sample_img.min():.3f}, {sample_img.max():.3f}]")
print("(Values are normalized around 0)")


# =============================================================================
# 3부: 데이터 로더 만들기
# =============================================================================
print("\n" + "=" * 80)
print("PART 3: Setting Up Data Loaders")
print("=" * 80)

"""
데이터 로더:
-------------
데이터를 한꺼번에 올리는 대신 데이터 로더로 다음을 한다.
- 데이터를 작은 배치으로 불러온다(기억 자리를 아낀다)
- 에폭마다 데이터를 섞는다(과적합을 막는다)
- 나란히 데이터를 불러온다(학습이 빨라진다)
"""

# 학습 데이터를 학습 집합과 검증 집합으로 나눈다
train_size = int(0.8 * len(train_dataset))  # 80% for training
val_size = len(train_dataset) - train_size  # 20% for validation

train_dataset, val_dataset = random_split(
    train_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print(f"Training set: {len(train_dataset)} samples")
print(f"Validation set: {len(val_dataset)} samples")
print(f"Test set: {len(test_dataset)} samples")

# 데이터 로더 생성
batch_size = 128  # Process 128 images at a time

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,      # Shuffle training data each epoch
    num_workers=0      # Parallel data loading
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,     # Don't shuffle validation data
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

print(f"\nBatch size: {batch_size}")
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")
print(f"Number of test batches: {len(test_loader)}")


# =============================================================================
# 4부: 모델 정의
# =============================================================================
print("\n" + "=" * 80)
print("PART 4: Building the Neural Network")
print("=" * 80)

class MNISTClassifier(nn.Module):
    """
    MNIST 숫자 분류을 위한 신경망.
    
    Architecture:
        펼치기(28×28) → 특징 784개
          ↓
        선형 층(784 → 256) + ReLU + 드롭아웃
          ↓
        선형 층(256 → 128) + ReLU + 드롭아웃
          ↓
        출력 층(128 → 클래스 10개)
    
    눈여겨볼 것: 드롭아웃은 익히는 동안 뉴런을 마구잡이로 떨어뜨려 지나치게
          맞춰짐을 막는다(추론 때는 늘 살아 있다).
    """
    
    def __init__(self, input_size=784, hidden_size1=256, hidden_size2=128, 
                 num_classes=10, dropout_rate=0.2):
        """
        망의 초기화한다.
        
        Args:
            input_size: 입력 특징의 수(MNIST에서는 28*28=784)
            hidden_size1: 첫 은닉층의 뉴런 수
            hidden_size2: 둘째 은닉층의 뉴런 수
            num_classes: 출력 클래스의 수(숫자 10개)
            dropout_rate: 드롭아웃 확률(0.2이면 뉴런의 20%를 떨어뜨린다)
        """
        super(MNISTClassifier, self).__init__()
        
        # 2차원 이미지를 1차원 벡터로 바꾸는 평탄화 층
        self.flatten = nn.Flatten()
        
        # 첫 번째 은닉층
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 두 번째 은닉층
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # 출력층 (활성화 없음 - 로짓을 반환한다)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
    def forward(self, x):
        """
        순전파.
        
        Args:
            x: 모양이 (batch_size, 1, 28, 28)인 입력 텐서
        
        Returns:
            모양이 (batch_size, num_classes)인 로짓
        """
        # 이미지를 평탄화한다: (batch, 1, 28, 28) → (batch, 784)
        x = self.flatten(x)
        
        # 첫 번째 은닉층
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        # 두 번째 은닉층
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        # 출력층 (로짓)
        logits = self.fc3(x)
        return logits


# 모델을 만든다
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = MNISTClassifier(
    input_size=784,
    hidden_size1=256,
    hidden_size2=128,
    num_classes=10,
    dropout_rate=0.2
).to(device)

print("\nModel Architecture:")
print(model)

# 매개변수 개수 세기
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")


# =============================================================================
# 5부: 미니배치로 학습하기
# =============================================================================
print("\n" + "=" * 80)
print("PART 5: Training the Model")
print("=" * 80)

# 학습을 준비한다
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

print(f"Loss function: CrossEntropyLoss")
print(f"Optimizer: Adam (lr=0.001)")
print(f"Training epochs: {num_epochs}")


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    모델을 한 에폭 익힌다.
    
    Returns:
        avg_loss: 그 판의 평균 손실
        accuracy: 학습 정확도
    """
    model.train()  # 학습 결로 둔다
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 데이터를 장치(GPU/CPU)로 옮긴다
        images = images.to(device)
        labels = labels.to(device)
        
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 역전파와 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 지표를 추적한다
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """
    검증 배치에서 모델을 평가한다.
    
    Returns:
        avg_loss: 평균 검증 손실
        accuracy: 검증 정확도
    """
    model.eval()  # Set to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy


# 학습 루프
print("\nStarting training...\n")
history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': []
}

start_time = time.time()

for epoch in range(num_epochs):
    # 학습
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, 
                                            optimizer, device)
    
    # 검증
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    # 이력 저장
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    
    # 진행 상황 출력
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print()

training_time = time.time() - start_time
print(f"✅ Training complete! Time taken: {training_time:.2f} seconds")


# =============================================================================
# 6부: 시험 집합에서 평가
# =============================================================================
print("\n" + "=" * 80)
print("PART 6: Final Evaluation on Test Set")
print("=" * 80)

def test_model(model, test_loader, device):
    """
    시험 배치에서 모델을 따지고 자세한 자를 돌려준다.
    """
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = correct / total
    return accuracy, np.array(all_predictions), np.array(all_labels)


test_accuracy, predictions, true_labels = test_model(model, test_loader, device)
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")


# =============================================================================
# 7부: 예측 시각화
# =============================================================================
print("\n" + "=" * 80)
print("PART 7: Visualizing Predictions")
print("=" * 80)

def visualize_predictions(model, test_loader, device, num_images=10):
    """
    보기 예측과 그 확률을 보인다.
    """
    model.eval()
    
    # 배치 하나를 얻는다
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # 예측한다
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predictions = torch.max(outputs, 1)
    
    # 그림을 그리기 위해 CPU로 되돌린다
    images = images.cpu()
    labels = labels.cpu()
    predictions = predictions.cpu()
    probabilities = probabilities.cpu()
    
    # 그림
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    axes = axes.ravel()
    
    for i in range(num_images):
        img = images[i].squeeze().numpy()
        true_label = labels[i].item()
        pred_label = predictions[i].item()
        prob = probabilities[i][pred_label].item()
        
        axes[i].imshow(img, cmap='gray')
        
        # 색 구분: 맞으면 초록, 틀리면 빨강
        color = 'green' if pred_label == true_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}\nConf: {prob:.2f}',
                         color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig


# 예측을 보려면 주석을 푼다
# visualize_predictions(model, test_loader, device)
# plt.show()


# =============================================================================
# 8부: 혼동 행렬
# =============================================================================
print("\n" + "=" * 80)
print("PART 8: Confusion Matrix")
print("=" * 80)

from sklearn.metrics import confusion_matrix, classification_report

# 혼동 행렬을 계산한다
cm = confusion_matrix(true_labels, predictions)

print("Confusion Matrix:")
print("-" * 80)
print(cm)
print("\nRow = True digit, Column = Predicted digit")

# 가장 자주 혼동되는 숫자를 찾는다
print("\nMost Common Misclassifications:")
print("-" * 80)
misclassifications = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i, j] > 0:
            misclassifications.append((i, j, cm[i, j]))

# 빈도순으로 정렬한다
misclassifications.sort(key=lambda x: x[2], reverse=True)

for true_digit, pred_digit, count in misclassifications[:5]:
    print(f"Digit {true_digit} predicted as {pred_digit}: {count} times")

# 분류 보고서
print("\n\nDetailed Classification Report:")
print("-" * 80)
print(classification_report(true_labels, predictions))


# =============================================================================
# 9부: 클래스별 정확도
# =============================================================================
print("\n" + "=" * 80)
print("PART 9: Per-Class Performance")
print("=" * 80)

# 클래스별 정확도를 계산한다
class_correct = [0] * 10
class_total = [0] * 10

for i in range(len(true_labels)):
    label = true_labels[i]
    class_total[label] += 1
    if predictions[i] == label:
        class_correct[label] += 1

print("Accuracy for each digit:")
print("-" * 80)
for i in range(10):
    accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
    print(f"Digit {i}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")


# =============================================================================
# 10부: 모델 저장하기
# =============================================================================
print("\n" + "=" * 80)
print("PART 10: Saving the Model")
print("=" * 80)

# 모델 정보를 모두 저장한다
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs,
    'train_acc': history['train_acc'][-1],
    'val_acc': history['val_acc'][-1],
    'test_acc': test_accuracy,
}

model_path = '/home/claude/softmax_regression_tutorial/level_03_mnist_model.pth'
torch.save(checkpoint, model_path)
print(f"✅ Model saved to: {model_path}")

# 모델을 불러온다 (시연)
loaded_checkpoint = torch.load(model_path)
loaded_model = MNISTClassifier().to(device)
loaded_model.load_state_dict(loaded_checkpoint['model_state_dict'])
loaded_model.eval()

print(f"✅ Model loaded successfully")
print(f"   Saved test accuracy: {loaded_checkpoint['test_acc']:.4f}")


# =============================================================================
# 요약
# =============================================================================
print("\n" + "=" * 80)
print("SUMMARY - What You Learned")
print("=" * 80)

print(f"""
✅ MNIST 데이터셋을 불러와 살펴보았다
✅ 효율적인 배치 다루기에 DataLoader를 썼다
✅ 학습/검증/시험 나누기를 짰다
✅ 드롭아웃을 곁들인 더 깊은 신경망을 지었다
✅ 참 세상 데이터셋으로 익혔다
✅ 시험 배치에서 정확도 {test_accuracy:.2%}을 얻었다
✅ 예측을 그림으로 보고 오차을 살폈다
✅ 혼동 행렬과 클래스마다의 자를 만들었다
✅ 익힌 모델을 저장하고 불러왔다

핵심 개념:
-------------
• 작은 배치 익히기: 데이터를 작은 배치으로 다룬다
• 데이터 로더: 섞기를 곁들여 데이터를 잘 다룬다
• 드롭아웃: 과적합을 막는 정칙화
• train()과 eval() 결: 드롭아웃과 배치 정규화가 다르게 움직인다
• 혼동 행렬: 어떤 클래스가 서로 헷갈리는지 안다

성능 간추림:
--------------------
학습 시간: {training_time:.2f}초
마지막 학습 정확도: {history['train_acc'][-1]:.2%}
마지막 검증 정확도: {history['val_acc'][-1]:.2%}
시험 정확도: {test_accuracy:.2%}

다음 걸음:
-----------
→ 4단계: 맞춤 학습을 밑바닥부터 짜기
→ 5단계: 앞선 기법(학습률 짜기, 데이터 불리기)
→ 실험: 여러 구조와 초매개변수를 써 보기

🎉 훌륭하다! MNIST으로 익히기를 해냈다!
""")


if __name__ == "__main__":
    pass
```

## 3. 논의

`MNISTClassifier` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 다중 클래스 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `MNISTClassifier`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

??? success "연습문제 1 풀이"
    `nn.Linear(in_features, out_features)` 각각에는 `in_features * out_features`개의 가중치 매개변수와 (`bias=False`가 아닌 한) `out_features`개의 편향 매개변수가 있다. `nn.Conv2d(in_c, out_c, k)`에는 `in_c * out_c * k * k`개의 가중치와 `out_c`개의 편향이 있다. `nn.Embedding(num, dim)`에는 `num * dim`개의 매개변수가 있다. 모든 층에 대해 더하면 된다. `sum(p.numel() for p in model.parameters())`로 확인할 수 있다.

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
층이나 블록의 개수를 설정할 수 있도록 `MNISTClassifier`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = MNISTClassifier(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — MNIST

`MNISTClassifier` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다.

핵심 클래스는 `MNISTClassifier`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
