# MNIST 분류 자세히 보기

튜토리얼 06: MNIST 숫자 분류. 배울 내용:

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 코드

```python
"""
==============================================================================
튜토리얼 06: MNIST 숫자 분류
==============================================================================
난이도: ⭐⭐ 중급

배울 내용:
- 실제 데이터셋 다루기
- DataLoader로 데이터 불러오기
- 학습 분할과 검증 분할
- 배치 처리
- 모델 평가 지표

선수 지식:
- 튜토리얼 05 (nn.Module과 최적화기)

핵심 개념:
- torchvision.datasets
- torch.utils.data.DataLoader
- 배치 학습
- 학습/시험 분할
- 정확도 계산
==============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

torch.manual_seed(42)

# ==============================================================================
# 들어가며: 실제 데이터셋
# ==============================================================================
print("=" * 70)
print("Welcome to MNIST Classification!")
print("=" * 70)
print("\nMNIST Dataset:")
print("  - 70,000 handwritten digit images (0-9)")
print("  - 28x28 grayscale images")
print("  - Classic machine learning benchmark")
print("  - Real-world computer vision task!")

# ==============================================================================
# 1단계: MNIST 데이터셋 불러오고 살펴보기
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 1: Loading MNIST Dataset")
print("=" * 70)

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# 변환: PIL 이미지를 텐서로
# ToTensor()는 [0, 1]로 자동 정규화한다
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 학습 데이터셋 내려받아 불러오기
print("Downloading MNIST dataset (if not already present)...")
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

# 시험 데이터셋 불러오기
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True
)

print(f"\nDataset loaded successfully!")
print(f"  Training samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

# 표본 하나 살펴보기
sample_image, sample_label = train_dataset[0]
print(f"\nSample exploration:")
print(f"  Image shape: {sample_image.shape}")  # [채널, 높이, 너비]
print(f"  Label: {sample_label}")
print(f"  Image value range: [{sample_image.min():.2f}, {sample_image.max():.2f}]")

# ==============================================================================
# 2단계: 데이터 로더 만들기
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Creating Data Loaders")
print("=" * 70)

print("\nWhat is a DataLoader?")
print("  - Batches data for efficient training")
print("  - Shuffles data each epoch")
print("  - Handles parallel data loading")
print("  - Essential for large datasets!")

batch_size = 64

# 데이터 로더 만들기
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,  # 더 나은 학습을 위해 섞기
    num_workers=2  # 병렬 데이터 적재
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False  # 시험 데이터는 섞을 필요가 없다
)

print(f"\nData loaders created:")
print(f"  Batch size: {batch_size}")
print(f"  Training batches: {len(train_loader)}")
print(f"  Test batches: {len(test_loader)}")

# 예: 배치 하나를 훑어보기
images, labels = next(iter(train_loader))
print(f"\nExample batch:")
print(f"  Images shape: {images.shape}")  # [batch_size, 채널, 높이, 너비]
print(f"  Labels shape: {labels.shape}")  # [batch_size]

# ==============================================================================
# 3단계: 표본 이미지 시각화
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Visualizing Sample Images")
print("=" * 70)

# 시각화를 위해 배치 하나 가져오기
examples = iter(test_loader)
example_images, example_labels = next(examples)

# 표본 12개 그리기
fig, axes = plt.subplots(2, 6, figsize=(12, 4))
axes = axes.ravel()

for i in range(12):
    axes[i].imshow(example_images[i].squeeze(), cmap='gray')
    axes[i].set_title(f'Label: {example_labels[i].item()}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/06_mnist_samples.png', dpi=100)
print("Sample images saved as '06_mnist_samples.png'")

# ==============================================================================
# 4단계: 신경망 정의
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Defining the Neural Network")
print("=" * 70)

class MNISTNet(nn.Module):
    """
    MNIST 분류를 위한 순방향 신경망
    
    구조:
      입력 (784) -> 은닉1 (128) -> 은닉2 (64) -> 출력 (10)
    
    왜 이런 구조인가?
      - 입력 784개: 28x28 화소를 펼친 것
      - 128, 64: 특징을 배우기 위한 은닉층
      - 출력 10개: 숫자마다 하나 (0~9)
    """
    def __init__(self):
        super(MNISTNet, self).__init__()
        
        self.network = nn.Sequential(
            # 첫 번째 은닉층
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            
            # 두 번째 은닉층
            nn.Linear(128, 64),
            nn.ReLU(),
            
            # 출력층 (활성화 없음 - CrossEntropyLoss를 쓸 것이다)
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        # 이미지 펼치기: [batch, 1, 28, 28] -> [batch, 784]
        x = x.view(x.size(0), -1)
        return self.network(x)

# 모델 만들기
model = MNISTNet().to(device)
print("Model architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# ==============================================================================
# 5단계: 손실과 최적화기 정의
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Loss Function and Optimizer")
print("=" * 70)

# 다중 클래스 분류를 위한 CrossEntropyLoss
# LogSoftmax와 NLLLoss를 합친다
criterion = nn.CrossEntropyLoss()
print("Loss function: CrossEntropyLoss")
print("  - Perfect for multi-class classification")
print("  - Expects raw logits (no softmax needed in model)")
print("  - Numerically stable")

# Adam 최적화기
learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
print(f"\nOptimizer: Adam (lr={learning_rate})")

# ==============================================================================
# 6단계: 학습 루프
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Training the Model")
print("=" * 70)

n_epochs = 5
train_losses = []
train_accuracies = []

print(f"Training for {n_epochs} epochs...\n")

for epoch in range(n_epochs):
    model.train()  # 모델을 학습 모드로
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 진행 상황 기록
    for batch_idx, (images, labels) in enumerate(train_loader):
        # 데이터를 장치로 옮기기
        images = images.to(device)
        labels = labels.to(device)
        
        # 기울기 초기화
        optimizer.zero_grad()
        
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 역전파와 최적화
        loss.backward()
        optimizer.step()
        
        # 통계
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 100 배치마다 진행 상황 출력
        if (batch_idx + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], '
                  f'Step [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    # 에포크 통계
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    print(f'\nEpoch [{epoch+1}/{n_epochs}] Summary:')
    print(f'  Average Loss: {epoch_loss:.4f}')
    print(f'  Training Accuracy: {epoch_acc:.2f}%\n')

# ==============================================================================
# 7단계: 시험 집합에서 평가
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Testing the Model")
print("=" * 70)

model.eval()  # 모델을 평가 모드로

test_correct = 0
test_total = 0
class_correct = [0] * 10
class_total = [0] * 10

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()
        
        # 클래스별 정확도
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

test_accuracy = 100 * test_correct / test_total
print(f'\nOverall Test Accuracy: {test_accuracy:.2f}%')

print('\nPer-class accuracy:')
for i in range(10):
    acc = 100 * class_correct[i] / class_total[i]
    print(f'  Digit {i}: {acc:.2f}%')

# ==============================================================================
# 8단계: 예측 시각화
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 8: Visualizing Predictions")
print("=" * 70)

# 시험 이미지 몇 장 가져오기
model.eval()
with torch.no_grad():
    test_images, test_labels = next(iter(test_loader))
    test_images = test_images.to(device)
    outputs = model(test_images)
    _, predictions = torch.max(outputs, 1)

# 예측 그리기
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.ravel()

for i in range(12):
    img = test_images[i].cpu().squeeze()
    true_label = test_labels[i].item()
    pred_label = predictions[i].item()
    
    axes[i].imshow(img, cmap='gray')
    color = 'green' if true_label == pred_label else 'red'
    axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/06_predictions.png', dpi=100)
print("Predictions saved as '06_predictions.png'")

# ==============================================================================
# 9단계: 학습 과정 시각화
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 9: Training Progress Visualization")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 학습 손실 그리기
ax1.plot(range(1, n_epochs + 1), train_losses, 'b-o', linewidth=2, markersize=8)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.grid(True, alpha=0.3)

# 학습 정확도 그리기
ax2.plot(range(1, n_epochs + 1), train_accuracies, 'g-o', linewidth=2, markersize=8)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Accuracy')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/06_training_progress.png', dpi=100)
print("Training progress saved as '06_training_progress.png'")

# ==============================================================================
# 10단계: 모델 저장
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 10: Saving the Model")
print("=" * 70)

model_path = '/home/claude/pytorch_feedforward_tutorial/mnist_model.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# ==============================================================================
# 핵심 정리:
# ==============================================================================
print("\n" + "=" * 70)
print("핵심 정리")
print("=" * 70)
print("""
1. 실제 데이터 작업 흐름:
   a) 데이터셋을 불러온다(torchvision.datasets)
   b) 데이터 로더를 만든다(배치 구성, 섞기)
   c) 모델 구조를 정의한다
   d) 배치로 학습한다
   e) 시험 집합에서 평가한다

2. DataLoader의 이점:
   - 자동 배치 구성
   - 데이터 섞기
   - 병렬 적재
   - 기억 자리 아끼기

3. 분류에 쓰는 CrossEntropyLoss:
   - LogSoftmax와 NLLLoss를 합친다
   - 원본 로짓을 받는다(모델에 소프트맥스를 두지 않는다)
   - 수치적으로 안정적이다

4. 학습 모범 사례:
   - 학습 앞에 model.train()
   - 평가 앞에 model.eval()
   - 추론 중에는 torch.no_grad()
   - 손실과 정확도를 함께 추적한다

5. MNIST의 일반적인 정확도:
   - 단순 순전파 신경망: 95~97%
   - CNN(뒤에서 다룬다): 99% 이상
   - 우리 모델: 약 {test_accuracy:.1f}%

다음 단계:
- 튜토리얼 07: 검증 집합과 정칙화 더하기
- 튜토리얼 08: 배치 정규화
- 튜토리얼 09: 더 깊은 신경망
- 튜토리얼 10: 고급 기법
""")

print("Training completed successfully! ✓")
# ==============================================================================


if __name__ == "__main__":
    pass
```

## 논의

`MNISTNet` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `MNISTNet`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `MNISTNet`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = MNISTNet(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
