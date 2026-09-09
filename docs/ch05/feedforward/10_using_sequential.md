# Sequential 사용하기

04_using_sequential.py - nn.Sequential로 모델 빠르게 만들기. nn.Sequential은 별도의 클래스를 쓰지 않고도 모델을 빠르게 만드는 PyTorch의 방법이다

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 1. 코드

```python
"""
================================================================================
04_using_sequential.py - nn.Sequential로 모델 빠르게 만들기
================================================================================

nn.Sequential은 사용자 정의 클래스를 쓰지 않고 모델을 빠르게 만드는 PyTorch의 방법이다.
간단한 순방향 구조에 안성맞춤이다!

nn.Sequential을 쓸 때:
    ✓ 간단한 순방향 신경망
    ✓ 빠른 시제품 만들기
    ✓ 선형층을 쌓은 구조
    ✗ 복잡하게 갈라지는 구조
    ✗ 여러 입력/출력
    ✗ 사용자 정의 순전파 논리

학습 목표:
    1. nn.Sequential로 모델 만들기
    2. 사용자 정의 nn.Module과의 절충 이해하기
    3. Sequential 블록 조립하는 법 배우기
    4. 여러 구조 양식으로 연습하기

난이도: ⭐⭐☆☆☆ (초급~중급)
소요 시간: 15~20분
================================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ================================================================================
# 1부: 간단한 Sequential 모델
# ================================================================================
print("=" * 80)
print("PART 1: Building a Simple Sequential Model")
print("=" * 80)

# PyTorch에서 신경망을 만드는 가장 간단한 방법이다!
# 각 층이 지정한 순서대로 차례차례 적용된다
simple_model = nn.Sequential(
    nn.Linear(784, 256),    # 입력층: 784 → 256
    nn.ReLU(),              # 활성화
    nn.Linear(256, 128),    # 은닉층: 256 → 128
    nn.ReLU(),              # 활성화
    nn.Linear(128, 10)      # 출력층: 128 → 10
)

print("Simple Sequential Model:")
print(simple_model)
print(f"\nTotal parameters: {sum(p.numel() for p in simple_model.parameters()):,}")

# ================================================================================
# 2부: 이름 붙인 층을 쓰는 Sequential
# ================================================================================
print("\n" + "=" * 80)
print("PART 2: Sequential with Named Layers")
print("=" * 80)

# 층에 이름을 붙이면 디버깅과 이해에 도움이 된다
# 나중에 특정 층에 접근하고 싶을 때 편리하다
named_model = nn.Sequential(
    # OrderedDict를 쓰거나 키-값 쌍으로 직접 이름을 준다
    ('flatten', nn.Flatten()),              # 입력 펼치기
    ('fc1', nn.Linear(784, 256)),           # 첫 완전 연결층
    ('relu1', nn.ReLU()),                   # 첫 활성화
    ('dropout1', nn.Dropout(0.2)),          # 정칙화를 위한 드롭아웃
    ('fc2', nn.Linear(256, 128)),           # 둘째 완전 연결층
    ('relu2', nn.ReLU()),                   # 둘째 활성화
    ('dropout2', nn.Dropout(0.2)),          # 드롭아웃 한 번 더
    ('fc3', nn.Linear(128, 10))             # 출력층
)

print("Named Sequential Model:")
for name, module in named_model.named_children():
    print(f"  {name}: {module}")

# ================================================================================
# 3부: 모듈식 Sequential (블록 조립하기)
# ================================================================================
print("\n" + "=" * 80)
print("PART 3: Composing Sequential Blocks")
print("=" * 80)

# 다시 쓸 수 있는 구성 블록을 만들 수 있다!
def make_fc_block(in_features, out_features, dropout=0.2):
    """
    완전 연결 블록 만들기: 선형 → ReLU → 드롭아웃
    
    흔히 쓰는 양식이다. 함수로 감싸 두면
    코드가 더 깔끔해지고 관리하기 쉬워진다.
    """
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Dropout(dropout)
    )

# 블록으로 모델 만들기
modular_model = nn.Sequential(
    nn.Flatten(),                          # 28×28을 784로 펼치기
    make_fc_block(784, 512, dropout=0.3),  # 블록 1
    make_fc_block(512, 256, dropout=0.3),  # 블록 2
    make_fc_block(256, 128, dropout=0.2),  # 블록 3
    nn.Linear(128, 10)                     # 출력 (로짓이므로 활성화 없음)
)

print("Modular Sequential Model:")
print(modular_model)
print(f"\nNumber of layers: {len(list(modular_model.children()))}")

# ================================================================================
# 4부: MNIST로 학습하기
# ================================================================================
print("\n" + "=" * 80)
print("PART 4: Training Sequential Model on MNIST")
print("=" * 80)

# 장치 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# MNIST 불러오기
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=2
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=128, shuffle=False, num_workers=2
)

# 학습에는 간단한 모델을 쓴다
model = simple_model.to(device)

# 손실과 최적화기
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 함수
def train_epoch(model, loader, criterion, optimizer, device):
    """한 에폭을 학습한다."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        # 이미지를 (batch, 1, 28, 28)에서 (batch, 784)로 펼친다
        images = images.view(images.size(0), -1).to(device)
        labels = labels.to(device)
        
        # 순전파
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 통계
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

# 평가 함수
def evaluate(model, loader, criterion, device):
    """시험 집합에서 평가한다."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.view(images.size(0), -1).to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

# 학습 루프
print("\nTraining...")
num_epochs = 5
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")

# ================================================================================
# 5부: 구조 비교하기
# ================================================================================
print("\n" + "=" * 80)
print("PART 5: Architecture Comparison")
print("=" * 80)

# 여러 모델을 비교해 보자
models_dict = {
    'Simple (3 layers)': simple_model,
    'Named (with dropout)': named_model,
    'Modular (4 layers)': modular_model
}

print("\nModel Comparison:")
print("-" * 80)
print(f"{'Model':<25} {'Parameters':<15} {'Layers':<10}")
print("-" * 80)
for name, model in models_dict.items():
    params = sum(p.numel() for p in model.parameters())
    layers = len([m for m in model.modules() if not isinstance(m, nn.Sequential)])
    print(f"{name:<25} {params:<15,} {layers:<10}")
print("-" * 80)

# ================================================================================
# 6부: Sequential 모델에 접근하고 고치기
# ================================================================================
print("\n" + "=" * 80)
print("PART 6: Accessing Sequential Model Components")
print("=" * 80)

# 인덱스로 층에 접근할 수 있다
print("First layer of simple_model:")
print(simple_model[0])

print("\nThird layer (second ReLU):")
print(simple_model[3])

# 층을 훑어볼 수 있다
print("\nAll layers:")
for idx, layer in enumerate(simple_model):
    print(f"  Layer {idx}: {layer.__class__.__name__}")

# Sequential 모델을 잘라 낼 수 있다
print("\nFirst 3 layers:")
feature_extractor = simple_model[:3]  # 처음 세 층을 얻는다
print(feature_extractor)

# Sequential 모델을 고칠 수 있다
print("\nModifying model by adding a new layer:")
extended_model = nn.Sequential(
    *simple_model,  # 기존 층 펼치기
    nn.ReLU(),      # 활성화 하나 더 추가
    nn.Linear(10, 5)  # 마지막 층 추가
)
print(f"Original output size: 10")
print(f"Extended output size: 5")

# ================================================================================
# 7부: 시각화
# ================================================================================
print("\n" + "=" * 80)
print("PART 7: Visualizing Training Progress")
print("=" * 80)

fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(14, 5))

# 손실을 그린다
ax1.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
ax1.plot(test_losses, 'r-', label='Test Loss', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Test Loss', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 정확도를 그린다
ax2.plot(train_accs, 'b-', label='Train Accuracy', linewidth=2)
ax2.plot(test_accs, 'r-', label='Test Accuracy', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Training and Test Accuracy', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([90, 100])

plt.tight_layout()
plt.savefig('04_sequential_training.png', dpi=150, bbox_inches='tight')
print("Training progress saved as '04_sequential_training.png'")
plt.show()

# ================================================================================
# 핵심 정리
# ================================================================================
print("\n" + "=" * 80)
print("KEY TAKEAWAYS")
print("=" * 80)
print("""
1. nn.Sequential은 단순한 순전파 구조에 안성맞춤이다
   - 깔끔하고 읽기 쉬운 코드
   - 빠른 시제품 제작
   - 맞춤 nn.Module보다 군더더기가 적다

2. Sequential을 쓰는 세 가지 방법:
   - 단순: 층을 순서대로 넘기기만 한다
   - 이름 붙이기: 디버깅을 돕도록 튜플을 쓴다
   - 모듈화: 다시 쓸 수 있는 블록을 조립한다

3. Sequential의 한계:
   ✗ 여러 입력이나 출력을 다룰 수 없다
   ✗ 맞춤 순전파 논리를 넣을 수 없다
   ✗ 조건부 실행이 안 된다
   → 이런 경우에는 맞춤 nn.Module을 쓰라

4. Sequential 모델은 다음과 온전히 호환된다.
   ✓ 모든 PyTorch 학습 API
   ✓ 모델 저장과 불러오기
   ✓ 전이 학습
   ✓ 모델 점검 도구

5. Sequential 모델은 접근, 자르기, 고치기가 쉽다
   - 인덱스로 접근: model[0]
   - Slice: model[:3]
   - 순회: for layer in model

언제 쓰는가:
  - 층을 단순히 쌓을 때는 Sequential을 쓴다
  - 복잡한 구조에는 맞춤 Module을 쓴다
""")

# ================================================================================
# 학생을 위한 연습문제
# ================================================================================
print("=" * 80)
print("EXERCISES TO TRY")
print("=" * 80)
print("""
1. Sequential로 5~6층짜리 더 깊은 모델을 만들어 보라
2. 층 사이에 배치 정규화를 넣어 보라
3. 드롭아웃 비율을 달리하여 실험해 보라
4. "넓은" 신경망(층마다 뉴런이 많다)과 "깊은" 신경망(층이 많다)을 만들어 견주어 보라
5. 설정 목록으로 Sequential 모델을 만들어 내는 함수를 짜 보라
6. 자르기로 중간 특징을 뽑아 보라
7. 여러 활성화 함수(LeakyReLU, ELU 등)를 써 보라
8. Sequential 모델의 앙상블을 만들어 보라
9. 첫 층의 가중치를 시각화해 보라
10. 층을 없애는 방식으로 모델 가지치기를 구현해 보라
""")


if __name__ == "__main__":
    pass
```

## 2. 논의

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
학습 루프에서 `optimizer.zero_grad()` 호출을 없애면 어떤 일이 일어나는지 설명하라. 고친 코드를 실행하고 학습 손실의 수렴에 미치는 영향을 서술하라.

??? success "연습문제 1 풀이"
    `optimizer.zero_grad()`가 없으면 PyTorch가 새 경사를 기존 `.grad` 텐서에 덮어쓰지 않고 더하기 때문에 반복에 걸쳐 경사가 누적된다. 이는 사실상 학습률에 누적된 단계 수를 곱하는 셈이어서 최적화가 점점 크고 불규칙한 걸음을 내딛게 된다. 학습 손실은 매끄럽게 수렴하는 대신 심하게 진동하거나 발산한다. 해결책은 간단하다. `loss.backward()`를 호출하기 전에 언제나 경사를 0으로 만들어라.

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
조기 종료를 구현하라. 매 에폭 후 검증 손실을 추적하고, 10 에폭 연속으로 개선이 없으면 학습을 멈춘다. 가장 좋은 모델 가중치를 저장하고 복원하라.

??? success "연습문제 4 풀이"
    인내 횟수 카운터와 최저 손실 추적기를 추가한다.
    ```python
    best_loss = float('inf')
    patience_counter = 0
    best_state = None
    for epoch in range(num_epochs):
        # ... 학습 단계 ...
        val_loss = evaluate(model, val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        if patience_counter >= 10:
            print(f'Early stopping at epoch {epoch}')
            model.load_state_dict(best_state)
            break
    ```
    이렇게 하면 따로 떼어 둔 데이터에서 모델이 더 나아지지 않을 때 멈추므로 과적합을 막을 수 있다.

## 정리하며

**다룬 것** — Sequential 사용하기

학습 루프는 표준적인 PyTorch 패턴을 따른다.

앞의 연습문제 4개로 직접 확인할 수 있다.
