# 다중 클래스 분류

02_multiclass_classification.py - 소프트맥스와 다중 클래스 문제

이 튜토리얼은 PyTorch에서 로지스틱 회귀에 대한 기초적인 이해를 쌓는다. 코드를 따라가 보면 모델을 세우고, 손실 함수를 정의하고, 경사 하강법으로 학습시키고, 분류 과제에서 성능을 평가하는 법을 알 수 있다.

## 코드

```python
"""
==============================================================================
02_multiclass_classification.py - 소프트맥스와 여러 클래스 문제
================================================================================

학습 목표:
- 로지스틱 회귀를 여러 클래스로 넓힌다
- 소프트맥스 살림을 이해한다
- CrossEntropyLoss을 쓴다
- 원핫 부호를 다룬다
- 여러 클래스 모델을 평가한다

소요 시간: 1시간 반쯤
어려움: ⭐⭐⭐⭐☆ (앞선)
================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("MULTI-CLASS CLASSIFICATION")
print("="*80)

# =============================================================================
# 1부: 이진 대 다중 클래스
# =============================================================================

print("\n" + "="*80)
print("PART 1: UNDERSTANDING MULTI-CLASS CLASSIFICATION")
print("="*80)

print("""
둘 분류:
  2 classes: 0 or 1
  출력: 확률 하나
  살림: 시그모이드
  손실: 둘 교차 엔트로피(BCE)

여러 클래스 분류:
  K classes: 0, 1, 2, ..., K-1
  출력: 확률 K개(합이 1이다)
  살림: 소프트맥스
  손실: 교차 엔트로피

소프트맥스 함수:
  클래스가 K개일 때 소프트맥스는 로짓을 확률로 바꾼다.
  
  P(class=k) = exp(logit_k) / sum(exp(logit_j) for all j)
  
  Properties:
  ✓ 확률이 모두 양수다
  ✓ 확률의 합이 1이다
  ✓ Differentiable
""")

# =============================================================================
# 2부: 다중 클래스 데이터 생성
# =============================================================================

print("\n" + "="*80)
print("PART 2: PREPARING MULTI-CLASS DATA")
print("="*80)

# 3개 클래스 데이터셋을 생성한다
n_classes = 3
n_samples = 1500

torch.manual_seed(42)
np.random.seed(42)

X, y = make_classification(
    n_samples=n_samples,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=n_classes,
    n_clusters_per_class=1,
    random_state=42
)

print(f"Dataset: {n_samples} samples, {n_classes} classes")
print(f"Class distribution:")
for i in range(n_classes):
    count = (y == i).sum()
    pct = 100 * count / len(y)
    print(f"  Class {i}: {count:4d} samples ({pct:.1f}%)")

# 나누고 표준화한다
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = torch.FloatTensor(scaler.fit_transform(X_train))
X_test = torch.FloatTensor(scaler.transform(X_test))

# 중요: CrossEntropyLoss에서 목표는 클래스 인덱스(Long)여야 한다
# 원-핫 부호가 아니다!
y_train = torch.LongTensor(y_train)  # Shape: (N,) with values 0, 1, 2
y_test = torch.LongTensor(y_test)

print(f"\nData shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape} (class indices)")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

# DataLoader들을 만든다
train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=64,
    shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=64,
    shuffle=False
)

# =============================================================================
# 3부: 다중 클래스 모델
# =============================================================================

print("\n" + "="*80)
print("PART 3: MULTI-CLASS LOGISTIC REGRESSION MODEL")
print("="*80)

class MultiClassLogisticRegression(nn.Module):
    """
    여러 클래스 로지스틱 회귀(소프트맥스 회귀)
    
    Args:
        input_dim: 입력 특징의 수
        num_classes: 출력 클래스의 수
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        # 선형 층: (input_dim) -> (num_classes)
        # 각 클래스가 자기만의 특징 선형결합을 갖는다
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        """
        순전파
        
        Args:
            x: 모양이 (batch_size, input_dim)인 입력 텐서
            
        Returns:
            logits: 모양이 (batch_size, num_classes)인 날것 점수
                   눈여겨볼 것: 확률이 아니라 로짓을 돌려준다!
                   CrossEntropyLoss이 안에서 소프트맥스를 건다
        """
        logits = self.linear(x)  # Shape: (batch_size, num_classes)
        return logits
    
    def predict_proba(self, x):
        """클래스 확률을 얻는다 (추론용)"""
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities


# 모델 생성
input_dim = X_train.shape[1]
model = MultiClassLogisticRegression(input_dim, n_classes)

print(f"Model created:")
print(f"  Input dimension: {input_dim}")
print(f"  Number of classes: {n_classes}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters())}")
print(f"\nModel architecture:")
print(model)

# =============================================================================
# 4부: 손실 함수와 학습
# =============================================================================

print("\n" + "="*80)
print("PART 4: TRAINING MULTI-CLASS MODEL")
print("="*80)

print("""
CrossEntropyLoss:
  - LogSoftmax과 NLLLoss을 아우른다
  - Expects:
    * 예측: (batch_size, num_classes) - 날것 로짓
    * 과녁: (batch_size,) - 클래스 번호(Long 텐서)
  - 소프트맥스를 절로 건다
  - 손수 하는 소프트맥스 + 로그보다 수치가 든든하다
""")

# 준비
criterion = nn.CrossEntropyLoss()  # Combines softmax + log + NLL
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
num_epochs = 100

# 학습 기록
history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': []
}

print(f"\nTraining for {num_epochs} epochs...")
print("-" * 60)

for epoch in range(num_epochs):
    # 학습
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_y in train_loader:
        # 순전파
        logits = model(batch_X)  # (batch_size, num_classes)
        loss = criterion(logits, batch_y)  # Expects logits and class indices
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 정확도 계산
        _, predicted = torch.max(logits, 1)  # Get class with highest logit
        train_loss += loss.item() * len(batch_X)
        correct += (predicted == batch_y).sum().item()
        total += len(batch_X)
    
    avg_train_loss = train_loss / total
    train_acc = correct / total
    
    # 검증
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            
            _, predicted = torch.max(logits, 1)
            test_loss += loss.item() * len(batch_X)
            correct += (predicted == batch_y).sum().item()
            total += len(batch_X)
    
    avg_test_loss = test_loss / total
    test_acc = correct / total
    
    # 이력 저장
    history['train_loss'].append(avg_train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(avg_test_loss)
    history['test_acc'].append(test_acc)
    
    # 진행 상황 출력
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} Acc: {test_acc:.4f}")

print("\n✓ Training completed!")

# =============================================================================
# 5부: 평가
# =============================================================================

print("\n" + "="*80)
print("PART 5: COMPREHENSIVE EVALUATION")
print("="*80)

# 예측을 얻는다
model.eval()
all_predictions = []
all_targets = []
all_probabilities = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        logits = model(batch_X)
        probabilities = torch.softmax(logits, dim=1)
        _, predicted = torch.max(logits, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_targets.extend(batch_y.cpu().numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

all_predictions = np.array(all_predictions)
all_targets = np.array(all_targets)
all_probabilities = np.array(all_probabilities)

# 분류 보고서
print("\nClassification Report:")
print(classification_report(all_targets, all_predictions,
                          target_names=[f'Class {i}' for i in range(n_classes)]))

# 혼동 행렬
cm = confusion_matrix(all_targets, all_predictions)
print("\nConfusion Matrix:")
print(cm)

# =============================================================================
# 6부: 시각화
# =============================================================================

print("\n" + "="*80)
print("PART 6: VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(16, 10))

# 그림 1: 학습 곡선
ax1 = plt.subplot(2, 3, 1)
ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
ax1.plot(history['test_loss'], label='Test Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Curves', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 그림 2: 정확도 곡선
ax2 = plt.subplot(2, 3, 2)
ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
ax2.plot(history['test_acc'], label='Test Acc', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Accuracy Curves', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 그림 3: 혼동 행렬
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=[f'C{i}' for i in range(n_classes)],
            yticklabels=[f'C{i}' for i in range(n_classes)])
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')
ax3.set_title('Confusion Matrix', fontweight='bold')

# 그림 4: 클래스별 정확도
ax4 = plt.subplot(2, 3, 4)
per_class_acc = []
for i in range(n_classes):
    mask = all_targets == i
    acc = (all_predictions[mask] == all_targets[mask]).mean()
    per_class_acc.append(acc)

bars = ax4.bar(range(n_classes), per_class_acc, color='steelblue', alpha=0.7)
ax4.set_xlabel('Class')
ax4.set_ylabel('Accuracy')
ax4.set_title('Per-Class Accuracy', fontweight='bold')
ax4.set_xticks(range(n_classes))
ax4.set_xticklabels([f'Class {i}' for i in range(n_classes)])
ax4.grid(True, alpha=0.3, axis='y')

for bar, acc in zip(bars, per_class_acc):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{acc:.3f}', ha='center', va='bottom')

# 그림 5: 확률 분포
ax5 = plt.subplot(2, 3, 5)
for i in range(n_classes):
    mask = all_targets == i
    probs = all_probabilities[mask, i]  # Probability of correct class
    ax5.hist(probs, bins=30, alpha=0.5, label=f'Class {i}')

ax5.set_xlabel('Predicted Probability (for true class)')
ax5.set_ylabel('Count')
ax5.set_title('Confidence Distribution', fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 그림 6: 모델 가중치 시각화
ax6 = plt.subplot(2, 3, 6)
weights = model.linear.weight.data.cpu().numpy()  # Shape: (n_classes, input_dim)
im = ax6.imshow(weights, aspect='auto', cmap='RdBu_r', center=0)
ax6.set_xlabel('Feature Index')
ax6.set_ylabel('Class')
ax6.set_title('Model Weights', fontweight='bold')
ax6.set_yticks(range(n_classes))
ax6.set_yticklabels([f'Class {i}' for i in range(n_classes)])
plt.colorbar(im, ax=ax6)

plt.tight_layout()
plt.show()

print("Visualizations created!")

# =============================================================================
# 핵심 요점
# =============================================================================

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
1. 여러 클래스와 둘 분류 견주기
   둘 분류: 출력 1개 → 시그모이드 → BCE
   여러 클래스: 출력 K개 → 소프트맥스 → 교차 엔트로피

2. 종요로운 다름
   ✓ 출력 층: Linear(input_dim, num_classes)
   ✓ 로짓을 돌려준다(확률이 아니다)
   ✓ CrossEntropyLoss을 쓴다
   ✓ 과녁은 클래스 번호다(Long)
   ✓ 예측: 로짓의 argmax

3. CROSSENTROPYLOSS
   ✓ 소프트맥스 + 로그 + NLL을 아우른다
   ✓ 수치가 더 든든하다
   ✓ 입력으로 로짓을 바란다
   ✓ 과녁으로 클래스 번호를 바란다

4. EVALUATION
   ✓ 클래스마다의 정확도
   ✓ 혼동 행렬
   ✓ 클래스마다의 정밀도/재현율
   ✓ 클래스마다의 F1 점수

5. 언제 쓸까
   ✓ 클래스가 둘을 넘을 때
   ✓ 클래스가 서로 겹치지 않을 때
   ✓ 레이블가 하나뿐인 분류
""")

print("\n" + "="*80)
print("EXERCISES")
print("="*80)
print("""
1. 쉬움: 클래스 수를 바꾸어 보아라(5, 10)

2. 보통: 위 k개 정확도을 짜라.
   - 참 클래스가 위 2개나 위 3개 예측에 드는지 살핀다

3. 보통: 편향을 다루도록 클래스 가중치를 더하여라.
   - CrossEntropyLoss의 class_weight 매개변수를 쓴다

4. 어려움: 하나 대 나머지 길을 짜라.
   - 둘 분류기 K개를 익힌다
   - 곧바로 하는 여러 클래스 분류과 견준다

5. 어려움: 레이블 스무딩을 더하여라.
   - 딱딱한 0/1 대신 부드러운 과녁
   - 일반화이 나아진다
""")

print("\n" + "="*80)
print("NEXT: 03_regularization.py - Preventing overfitting")
print("="*80)


if __name__ == "__main__":
    pass
```

## 논의

`MultiClassLogisticRegression` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `MultiClassLogisticRegression`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `MultiClassLogisticRegression`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = MultiClassLogisticRegression(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
