# 깊은 신경망

15_deep_network.py - 아주 깊은 신경망 만들기. 다음을 사용하여 아주 깊은 순방향 신경망을 만들고 학습시키는 법을 배운다:

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 코드

```python
"""
15_deep_network.py - 아주 깊은 신경망 만들기

다음을 써서 아주 깊은 순방향 신경망을 만들고 학습시키는 법을 배운다:
- 잔차 연결 (건너뛰기 연결)
- 세심한 초기화
- 기울기 자르기
- 고급 기법

깊이 들어가기: 층이 많을수록 용량은 커지지만 학습은 어려워진다!

소요 시간: 45~60분 | 난이도: ⭐⭐⭐⭐⭐
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================

print("="*80)
print("Building Very Deep Networks")
print("="*80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST 불러오기
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

print("\n" + "="*80)
print("Architecture Design")
print("="*80)

class ResidualBlock(nn.Module):
    """
    건너뛰기 연결을 갖는 잔차 블록.
    출력 = ReLU(Block(x) + x)
    """
    
    def __init__(self, size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual  # 건너뛰기 연결!
        out = self.relu(out)
        return out

class DeepNet(nn.Module):
    """잔차 연결을 갖는 아주 깊은 신경망."""
    
    def __init__(self, input_size=784, hidden_size=256, num_blocks=10, num_classes=10):
        super().__init__()
        
        # 입력 사영
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU()
        )
        
        # 잔차 블록 더미
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])
        
        # 출력층
        self.output = nn.Linear(hidden_size, num_classes)
        
        # 가중치를 알맞게 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_proj(x)
        
        # 잔차 블록 통과
        for block in self.blocks:
            x = block(x)
        
        x = self.output(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# 깊이가 다른 모델 만들기
models = {
    'Shallow (3 blocks)': DeepNet(num_blocks=3),
    'Medium (7 blocks)': DeepNet(num_blocks=7),
    'Deep (15 blocks)': DeepNet(num_blocks=15)
}

print("Model Comparison:")
print("-"*80)
for name, model in models.items():
    params = model.count_parameters()
    layers = len(list(model.modules()))
    print(f"{name:20s} | Parameters: {params:,} | Modules: {layers}")

print("\n" + "="*80)
print("Training Deep Network")
print("="*80)

# 깊은 모델 쓰기
model = models['Deep (15 blocks)'].to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# 워밍업이 있는 학습률 스케줄러
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 안정성을 위한 기울기 자르기
max_grad_norm = 1.0

def train_epoch(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # 기울기 자르기 (기울기 폭발을 막는다)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / len(loader), 100 * correct / total

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# 학습
epochs = 20
train_losses, train_accs, test_accs = [], [], []

print(f"Training deep network ({model.count_parameters():,} parameters)...")
print("-"*80)

for epoch in range(epochs):
    train_loss, train_acc = train_epoch(model, train_loader)
    test_acc = evaluate(model, test_loader)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    
    scheduler.step()
    
    print(f"Epoch [{epoch+1:2d}/{epochs}] | "
          f"Loss: {train_loss:.4f} | "
          f"Train Acc: {train_acc:.2f}% | "
          f"Test Acc: {test_acc:.2f}%")

# 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, linewidth=2, label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss (Deep Network)', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, linewidth=2, label='Train Accuracy')
ax2.plot(test_accs, linewidth=2, label='Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training Progress', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('15_deep_network_results.png', dpi=150)
print("\nResults saved!")

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print("""
BUILDING DEEP NETWORKS:
✓ Use residual connections (skip connections)
✓ Batch normalization after linear layers
✓ Proper weight initialization (Kaiming for ReLU)
✓ Gradient clipping to prevent explosion
✓ 배움 빠르기 짜기

RESIDUAL CONNECTIONS:
  H(x) = F(x) + x
  - Easier to optimize (gradients flow directly)
  - Enable training of 100+ layer networks
  - Used in ResNet, DenseNet, Transformers

TRAINING STABILITY:
- Gradient clipping: Limit gradient magnitude
- Batch normalization: Stabilize activations
- Skip connections: Direct gradient flow
- Proper initialization: Good starting point

DEPTH vs WIDTH:
- Deeper: More hierarchical features
- Wider: More capacity per layer
- Trade-off depends on problem

CHALLENGES WITH DEPTH:
⚠ Vanishing/exploding gradients
⚠ Degradation (accuracy saturates)
⚠ More memory and compute
⚠ Longer training time

SOLUTIONS:
✓ Residual connections
✓ Normalization layers
✓ Careful initialization
✓ Gradient clipping
✓ Learning rate warmup

CONGRATULATIONS!
You've completed the full tutorial! 🎉
You now know how to build production-ready neural networks!
""")
plt.show()


if __name__ == "__main__":
    pass```

## 논의

이 구현은 2개의 클래스(`ResidualBlock`, `DeepNet`)를 정의하며, 이들이 함께 작동하여 완전한 순방향 신경망 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `ResidualBlock`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `ResidualBlock`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = ResidualBlock(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.
