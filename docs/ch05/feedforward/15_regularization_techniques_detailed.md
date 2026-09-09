# 정칙화 기법 자세히 보기

튜토리얼 07: 정칙화 기법. 배울 내용:

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 1. 코드

```python
"""
==============================================================================
튜토리얼 07: 정칙화 기법
==============================================================================
난이도: ⭐⭐⭐ 중급~고급

배울 내용:
- 과적합과 과소적합
- 드롭아웃 정칙화
- L2 정칙화 (가중치 감쇠)
- 검증 집합 쓰기
- 조기 종료

선수 지식:
- 튜토리얼 06 (MNIST 분류)

핵심 개념:
- 과적합 예방
- nn.Dropout
- 가중치 감쇠
- 모델 검증
- 학습 과정 감시
==============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import numpy as np

torch.manual_seed(42)

# ==============================================================================
# 들어가며: 과적합 문제
# ==============================================================================
print("=" * 70)
print("Understanding Overfitting and Regularization")
print("=" * 70)
print("""
과적합이란 무엇인가?
  - 모델이 학습 데이터를 지나치게 잘 외운다
  - 학습 집합에서는 아주 잘한다
  - 처음 보는 데이터(시험 집합)에서 성능이 나쁘다
  - 일반화하지 않고 암기한다

과적합을 어떻게 막는가?
  1. 학습 데이터를 더 모은다
  2. 더 단순한 모델(매개변수가 적다)
  3. 정칙화 기법:
     - Dropout
     - L2 정칙화 (가중치 감쇠)
     - 데이터 증강
  4. 조기 종료
""")

# ==============================================================================
# 1단계: 검증 분할과 함께 데이터 준비
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 1: Creating Train/Validation/Test Split")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

# MNIST 불러오기
transform = transforms.Compose([transforms.ToTensor()])

train_val_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

# 학습 데이터를 학습용과 검증용으로 나눈다 (80/20)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(
    train_val_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

print("Dataset split:")
print(f"  Training:   {len(train_dataset):,} samples (80%)")
print(f"  Validation: {len(val_dataset):,} samples (20%)")
print(f"  Test:       {len(test_dataset):,} samples")

print("\nWhy use validation set?")
print("  - Monitor overfitting during training")
print("  - Tune hyperparameters")
print("  - Test set remains untouched until final evaluation")

# 데이터 로더 만들기
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ==============================================================================
# 2단계: 모델 정의 (정칙화 있는 것과 없는 것)
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Defining Models")
print("=" * 70)

class SimpleNet(nn.Module):
    """
    정칙화가 없는 간단한 신경망
    과적합하기 쉽다!
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

class RegularizedNet(nn.Module):
    """
    정칙화 기법을 쓰는 신경망
    
    쓰인 정칙화 방법:
      1. 드롭아웃: 학습 중에 일부 뉴런을 무작위로 0으로 만든다
      2. 가중치 감쇠: 최적화기를 통해 더한다 (L2 정칙화)
    """
    def __init__(self, dropout_rate=0.5):
        super(RegularizedNet, self).__init__()
        
        self.network = nn.Sequential(
            # 첫 번째 층
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # 활성화 뒤의 드롭아웃
            
            # 두 번째 층
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 출력층 (여기에는 드롭아웃을 두지 않는다)
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

print("Two models defined:")
print("  1. SimpleNet: No regularization")
print("  2. RegularizedNet: Dropout + Weight decay")
print(f"\nDropout explanation:")
print("  - Randomly sets neurons to 0 during training")
print("  - Forces network to learn robust features")
print("  - Prevents co-adaptation of neurons")
print("  - Automatically disabled during eval mode")

# ==============================================================================
# 3단계: 학습 함수
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Defining Training and Evaluation Functions")
print("=" * 70)

def train_epoch(model, train_loader, criterion, optimizer, device):
    """한 에포크 동안 학습한다"""
    model.train()  # 드롭아웃 켜기
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        # 여기서 세는 정확도는 드롭아웃이 켜진 채, 그것도 갱신 전
        # 가중치로 계산한 outputs에서 나온 값이다. 학습이 도는 동안
        # 대강의 진행을 보기에는 쓸 만하지만 모델끼리 견주는 데에는
        # 쓸 수 없다. 드롭아웃을 쓰는 모델에서만 낮게 잡히기 때문이다.
        # 그래서 아래 4단계는 이 값을 버리고 evaluate()로 다시 잰다
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def evaluate(model, data_loader, criterion, device):
    """데이터셋에서 모델을 평가한다"""
    model.eval()  # 드롭아웃 끄기
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(data_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

print("Functions defined:")
print("  - train_epoch(): Trains for one epoch")
print("  - evaluate(): Evaluates model (no gradient computation)")

# ==============================================================================
# 4단계: 두 모델 모두 학습
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 4: Training Both Models")
print("=" * 70)

n_epochs = 15
learning_rate = 0.001

# 모델 1: 정칙화 없음
print("\n" + "-" * 70)
print("Training Model 1: SimpleNet (No Regularization)")
print("-" * 70)

# 주의: 씨앗을 심지 않아 두 모델의 초기 가중치가 다르다. 차이를
# 정칙화 탓으로만 돌리려면 두 모델을 만들기 직전마다
# torch.manual_seed(42)를 불러야 한다
model1 = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer1 = optim.Adam(model1.parameters(), lr=learning_rate)
# 모델 1에는 가중치 감쇠를 쓰지 않는다

history1 = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

for epoch in range(n_epochs):
    train_loss, _ = train_epoch(model1, train_loader, criterion, optimizer1, device)
    # 학습 정확도를 eval 모드에서 다시 잰다. train_epoch가 돌려주는 값은
    # 드롭아웃이 켜진 채, 그것도 갱신 전 출력으로 계산한 것이라 정칙화를
    # 쓰는 모델에서만 낮게 잡힌다. 그대로 쓰면 아래 "학습-검증 차이"가
    # 모델 2에서만 작아 보여, 정칙화의 효과를 실제보다 부풀린다
    _, train_acc = evaluate(model1, train_loader, criterion, device)
    val_loss, val_acc = evaluate(model1, val_loader, criterion, device)
    
    history1['train_loss'].append(train_loss)
    history1['train_acc'].append(train_acc)
    history1['val_loss'].append(val_loss)
    history1['val_acc'].append(val_acc)
    
    if (epoch + 1) % 3 == 0:
        print(f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# 모델 2: 정칙화 있음
print("\n" + "-" * 70)
print("Training Model 2: RegularizedNet (With Dropout + Weight Decay)")
print("-" * 70)

# 기본값 0.5가 아니라 0.3을 넘긴다. 폭이 256과 128인 층이라
# 절반을 끄면 남는 뉴런이 적어 학습이 더디기 때문이다
model2 = RegularizedNet(dropout_rate=0.3).to(device)
# 가중치 감쇠 추가 (L2 정칙화).
# 모델 1에는 이 인자가 없으므로, 위와 달리 여기 기준선은 정말로
# 정칙화가 없는 모델이다.
# 다만 한 번에 두 가지(드롭아웃과 가중치 감쇠)를 바꾸었으므로,
# 아래 결과에서 어느 쪽이 일했는지는 갈라낼 수 없다.
# 그것까지 보려면 셋째 모델로 하나씩만 켜서 견주어야 한다
optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate, weight_decay=1e-4)

print(f"Regularization parameters:")
print(f"  - Dropout rate: 0.3 (30% of neurons dropped)")
print(f"  - Weight decay: 1e-4 (L2 penalty)\n")

history2 = {
    'train_loss': [], 'train_acc': [],
    'val_loss': [], 'val_acc': []
}

for epoch in range(n_epochs):
    train_loss, _ = train_epoch(model2, train_loader, criterion, optimizer2, device)
    # 학습 정확도를 eval 모드에서 다시 잰다. train_epoch가 돌려주는 값은
    # 드롭아웃이 켜진 채, 그것도 갱신 전 출력으로 계산한 것이라 정칙화를
    # 쓰는 모델에서만 낮게 잡힌다. 그대로 쓰면 아래 "학습-검증 차이"가
    # 모델 2에서만 작아 보여, 정칙화의 효과를 실제보다 부풀린다
    _, train_acc = evaluate(model2, train_loader, criterion, device)
    val_loss, val_acc = evaluate(model2, val_loader, criterion, device)
    
    history2['train_loss'].append(train_loss)
    history2['train_acc'].append(train_acc)
    history2['val_loss'].append(val_loss)
    history2['val_acc'].append(val_acc)
    
    if (epoch + 1) % 3 == 0:
        print(f"Epoch {epoch+1:2d}: Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

# ==============================================================================
# 5단계: 결과 비교
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Comparing Models on Test Set")
print("=" * 70)

# 두 모델을 시험 집합에서 평가
test_loss1, test_acc1 = evaluate(model1, test_loader, criterion, device)
test_loss2, test_acc2 = evaluate(model2, test_loader, criterion, device)

print("\nFinal Test Results:")
print("-" * 70)
print(f"Model 1 (No Regularization):")
print(f"  Test Accuracy: {test_acc1:.2f}%")
print(f"  Test Loss: {test_loss1:.4f}")
print(f"\nModel 2 (With Regularization):")
print(f"  Test Accuracy: {test_acc2:.2f}%")
print(f"  Test Loss: {test_loss2:.4f}")

# 과적합 지표 계산 (학습-검증 차이)
# 과적합을 "학습 정확도 빼기 검증 정확도"로 잰다. 학습 데이터에서만
# 잘하고 처음 보는 데이터에서 못하는 정도를 뜻하므로 타당한 잣대다.
# 두 모델의 학습 정확도를 모두 eval() 상태에서 다시 재었으므로, 이
# 차이는 정칙화의 효과만 담는다. 드롭아웃이 켜진 채로 잰 값을 그대로
# 썼다면 model2의 차이가 실제보다 작게 나와 정칙화를 과대평가하게 된다
overfit1 = history1['train_acc'][-1] - history1['val_acc'][-1]
overfit2 = history2['train_acc'][-1] - history2['val_acc'][-1]
print(f"\nOverfitting Analysis (Train-Val Accuracy Gap):")
print(f"  Model 1: {overfit1:.2f}% gap")
print(f"  Model 2: {overfit2:.2f}% gap")
print(f"  {'Model 2 has less overfitting! ✓' if overfit2 < overfit1 else 'Unexpected result'}")

# ==============================================================================
# 6단계: 시각화
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Visualizing Training Progress")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

epochs = range(1, n_epochs + 1)

# 모델 1: 손실
axes[0, 0].plot(epochs, history1['train_loss'], 'b-o', label='Train Loss', alpha=0.7)
axes[0, 0].plot(epochs, history1['val_loss'], 'r-s', label='Val Loss', alpha=0.7)
axes[0, 0].set_title('Model 1 (No Regularization): Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 모델 1: 정확도
axes[0, 1].plot(epochs, history1['train_acc'], 'b-o', label='Train Acc', alpha=0.7)
axes[0, 1].plot(epochs, history1['val_acc'], 'r-s', label='Val Acc', alpha=0.7)
axes[0, 1].set_title('Model 1 (No Regularization): Accuracy')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 모델 2: 손실
axes[1, 0].plot(epochs, history2['train_loss'], 'b-o', label='Train Loss', alpha=0.7)
axes[1, 0].plot(epochs, history2['val_loss'], 'r-s', label='Val Loss', alpha=0.7)
axes[1, 0].set_title('Model 2 (With Regularization): Loss')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 모델 2: 정확도
axes[1, 1].plot(epochs, history2['train_acc'], 'b-o', label='Train Acc', alpha=0.7)
axes[1, 1].plot(epochs, history2['val_acc'], 'r-s', label='Val Acc', alpha=0.7)
axes[1, 1].set_title('Model 2 (With Regularization): Accuracy')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/07_regularization_comparison.png', dpi=100)
print("Comparison saved as '07_regularization_comparison.png'")

# ==============================================================================
# 핵심 정리:
# ==============================================================================
print("\n" + "=" * 70)
print("핵심 정리")
print("=" * 70)
print("""
1. 과적합의 징후:
   - 학습 정확도와 검증 정확도의 간격이 크다
   - 학습 정확도는 계속 오르는데 검증은 정체된다
   - 처음 보는 데이터에서 성능이 나쁘다

2. Dropout (nn.Dropout):
   - 학습 중 뉴런을 무작위로 0으로 만든다
   - dropout_rate: 떨어뜨릴 확률(보통 0.2~0.5)
   - model.eval()을 부르면 자동으로 꺼진다
   - 뉴런들이 함께 적응하는 것을 막는다

3. 가중치 감쇠(L2 정칙화):
   - 큰 가중치에 벌점을 더한다: 손실 = 데이터 손실 + λ * Σ(가중치²)
   - 최적화기에서 설정한다: weight_decay=1e-4
   - 가중치를 작게 유지하도록 북돋운다
   - 두루 미침이 나아진다

4. 검증 집합:
   - 과적합을 살피는 데 꼭 필요하다
   - 초매개변수 조율에 쓴다
   - 기울기 갱신에는 쓰지 않는다
   - 시험 집합과 따로 둔다

5. 그 밖의 정칙화 기법(여기서는 다루지 않는다):
   - 데이터 증강
   - 배치 정규화
   - 조기 종료
   - L1 regularization

6. 모범 사례:
   - 늘 검증 집합을 쓰라
   - 학습과 검증의 간격을 살피라
   - 가벼운 정칙화로 시작하여 필요하면 늘리라
   - 드롭아웃은 ReLU 앞이 아니라 뒤에 두라

다음 단계:
- 튜토리얼 08: 배치 정규화
- 튜토리얼 09: 학습률 스케줄링
- 튜토리얼 10: 고급 구조
""")

print("\nTraining completed successfully! ✓")
# ==============================================================================


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 2개의 클래스(`SimpleNet`, `RegularizedNet`)를 정의하며, 이들이 함께 작동하여 완전한 순방향 신경망 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `SimpleNet`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `SimpleNet`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = SimpleNet(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 정칙화 기법 자세히 보기

이 구현은 2개의 클래스(`SimpleNet`, `RegularizedNet`)를 정의하며, 이들이 함께 작동하여 완전한 순방향 신경망 구조를 이룬다.

핵심 클래스는 `SimpleNet`, `RegularizedNet`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
