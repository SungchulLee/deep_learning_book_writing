# CIFAR-10 분류기

12_cifar10_classifier.py - 컬러 이미지 분류. 제대로 된 평가까지 갖춘 완전한 CIFAR-10 분류기를 만든다.

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 코드

```python
"""
12_cifar10_classifier.py - 컬러 이미지 분류

제대로 된 평가를 갖춘 완전한 CIFAR-10 분류기를 만든다.
CIFAR-10: 10개 클래스로 이루어진 32x32 컬러 이미지 60,000장.

클래스: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

데이터 적재부터 배포까지 완전한 기계학습 파이프라인을 보인다.

소요 시간: 40~50분 | 난이도: ⭐⭐⭐⭐☆
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ========================================================================
# 메인
# ========================================================================

print("="*80)
print("CIFAR-10 Image Classification Pipeline")
print("="*80)

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# CIFAR-10 클래스
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
          'dog', 'frog', 'horse', 'ship', 'truck')

print("\n" + "="*80)
print("STEP 1: Data Loading and Augmentation")
print("="*80)

# 학습을 위한 데이터 증강 (일반화를 개선한다)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),      # 덧대기와 함께 무작위 잘라내기
    transforms.RandomHorizontalFlip(),         # 50% 확률로 뒤집기
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),     # [-1, 1]로 정규화
                        (0.5, 0.5, 0.5))
])

# 시험에는 증강 없음 (정규화만)
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 데이터셋 불러오기
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

# 데이터 로더 만들기
batch_size = 128
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {batch_size}")

print("\n" + "="*80)
print("STEP 2: Model Architecture")
print("="*80)

class CIFAR10Net(nn.Module):
    """
    CIFAR-10을 위한 깊은 순방향 신경망.
    
    구조: 배치 정규화와 드롭아웃을 갖춘 5개 층.
    입력: 32x32x3 = 특징 3072개
    출력: 클래스 10개
    """
    
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        
        self.network = nn.Sequential(
            # 입력: 특징 3072개 (32*32*3)
            nn.Linear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 10)  # 클래스 10개
        )
    
    def forward(self, x):
        # 이미지 펼치기: (B, 3, 32, 32) → (B, 3072)
        x = x.view(x.size(0), -1)
        return self.network(x)

model = CIFAR10Net().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

print("\n" + "="*80)
print("STEP 3: Training Setup")
print("="*80)

# 손실과 최적화기
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# 학습률 스케줄러
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

print(f"Loss: CrossEntropyLoss")
print(f"Optimizer: Adam (lr=0.001, weight_decay=1e-4)")
print(f"Scheduler: StepLR (step_size=20, gamma=0.5)")

print("\n" + "="*80)
print("STEP 4: Training")
print("="*80)

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

# 학습 루프
num_epochs = 50
best_acc = 0
train_losses, test_losses = [], []
train_accs, test_accs = [], []

print(f"Training for {num_epochs} epochs...")
print("-"*80)

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    scheduler.step()
    
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_cifar10_model.pth')
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

print(f"\nBest Test Accuracy: {best_acc:.2f}%")

print("\n" + "="*80)
print("STEP 5: Evaluation and Analysis")
print("="*80)

# 가장 좋은 모델을 불러온다
model.load_state_dict(torch.load('best_cifar10_model.pth'))

# 최종 평가
_, final_acc, all_preds, all_labels = evaluate(model, test_loader, criterion)

# 클래스별 정확도
class_correct = [0] * 10
class_total = [0] * 10
for pred, label in zip(all_preds, all_labels):
    class_correct[label] += (pred == label)
    class_total[label] += 1

print("\nPer-Class Accuracy:")
print("-"*60)
for i in range(10):
    acc = 100 * class_correct[i] / class_total[i]
    print(f"{classes[i]:10s}: {acc:5.2f}% ({class_correct[i]}/{class_total[i]})")

# 혼동 행렬
cm = confusion_matrix(all_labels, all_preds)

print("\n" + "="*80)
print("STEP 6: Visualization")
print("="*80)

# 시각화 만들기
fig = plt.figure(figsize=(18, 5))

# 그림 1: 학습 곡선
ax1 = plt.subplot(1, 3, 1)
ax1.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
ax1.plot(test_losses, 'r-', label='Test Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(1, 3, 2)
ax2.plot(train_accs, 'b-', label='Train Acc', linewidth=2)
ax2.plot(test_accs, 'r-', label='Test Acc', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 그림 3: 혼동 행렬
ax3 = plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes, ax=ax3)
ax3.set_xlabel('Predicted')
ax3.set_ylabel('True')
ax3.set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('12_cifar10_results.png', dpi=150)
print("Results saved as '12_cifar10_results.png'")

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print(f"""
ACHIEVED: {final_acc:.2f}% accuracy on CIFAR-10
(State-of-the-art CNNs achieve ~99%)

COMPLETE PIPELINE DEMONSTRATED:
✓ Data augmentation for better generalization
✓ Proper train/test split
✓ Deep architecture with regularization
✓ 배움 빠르기 짜기
✓ Model checkpointing (save best model)
✓ Comprehensive evaluation (per-class, confusion matrix)
✓ Visualization of results

PRODUCTION CONSIDERATIONS:
- Always use validation set for hyperparameter tuning
- Monitor multiple metrics, not just accuracy
- Save checkpoints regularly
- Analyze errors (confusion matrix)
- Test final model only once on test set
""")
plt.show()


if __name__ == "__main__":
    pass```

## 논의

`CIFAR10Net` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `CIFAR10Net`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `CIFAR10Net`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = CIFAR10Net(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
