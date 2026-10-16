# 깊은 뉴럴 넷트웍

같은 피처 차원을 유지하는 잔차 블록을 ModuleList로 쌓아 깊은 MLP를 구성한다. 블록 수에 따른 크기를 비교한 뒤 한 구조를 학습한다.

## 1. 코드

```python
"""
15_deep_network.py - 아주 깊은 뉴럴 넷트웍 만들기

다음을 써서 아주 깊은 순방향 뉴럴 넷트웍을 만들고 학습시키는 법을 배운다:
- 잔차 연결 (건너뛰기 연결)
- 세심한 초기화
- 그래디언트 자르기
- 고급 기법

깊이 들어가기: 레이어가 많을수록 용량은 커지지만 학습은 어려워진다!

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
            # 블록이 정규화로 끝나고 활성화는 밖에서 더한 뒤에 건다.
            # 그래야 항등 경로가 활성화를 거치지 않고 그대로 흘러,
            # 미분이 1 + f'(x)가 되어 그래디언트가 반드시 살아남는다
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.block(x)
        # 건너뛰기 연결!
        # 여기서는 제자리 덧셈이 통하지만 아슬아슬하다. block이 배치
        # 정규화로 끝나고 그 역전파는 자기 출력을 다시 보지 않기
        # 때문이다. 만약 block이 ReLU로 끝났다면 ReLU의 역전파가
        # 자기 출력을 필요로 해서 "inplace operation" 오류가 난다.
        # out = out + residual로 쓰면 어느 경우에도 안전하다
        out += residual
        out = self.relu(out)
        return out

class DeepNet(nn.Module):
    """잔차 연결을 갖는 아주 깊은 뉴럴 넷트웍."""
    
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
        
        # 출력 레이어
        self.output = nn.Linear(hidden_size, num_classes)
        
        # 웨이트를 알맞게 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # mode='fan_out'은 기본값인 fan_in과 다르다. 순전파의
                # 분산이 아니라 역전파의 분산을 지키는 쪽이며, ResNet
                # 원 논문이 쓴 설정이다
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                # gamma=1, beta=0이면 정규화한 값을 그대로 통과시킨다.
                # 곧 학습 초반에는 정규화 레이어가 아무것도 바꾸지 않는다.
                # 참고로 ResNet 계열에서는 블록의 "마지막" 정규화만
                # gamma=0으로 두는 요령도 쓴다. 그러면 블록의 출력이
                # 0에서 시작해 뉴럴 넷트웍 전체가 항등 사상으로 출발하므로
                # 아주 깊을 때 더 안정적이다
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

# 워밍업이 있는 러닝레이트 스케줄러
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

# 안정성을 위한 그래디언트 자르기
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
        
        # 그래디언트 자르기 (그래디언트 폭발을 막는다).
        # 자리가 정해져 있다. backward "뒤", step "앞"이다. 순서를
        # 어기면 자르기가 갱신에 반영되지 않는다.
        # 전체 그래디언트를 하나의 벡터로 보고 그 노름이 1을 넘으면
        # 방향은 그대로 둔 채 길이만 1로 줄인다. 값 하나하나를 자르는
        # clip_grad_value_와 달리 방향이 보존된다
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
깊은 뉴럴 넷트웍 짓기:
✓ 잔차 연결(건너뛰기 연결)을 쓴다
✓ 선형 레이어 뒤의 배치 정규화
✓ 알맞은 웨이트 초기화(ReLU에는 카이밍)
✓ 폭발을 막는 그래디언트 절단
✓ 러닝 빠르기 짜기

잔차 연결:
  H(x) = F(x) + x
  - 최적화가 쉽다(그래디언트가 곧장 흐른다)
  - 100레이어가 넘는 뉴럴 넷트웍도 학습할 수 있게 한다
  - ResNet, DenseNet, 트랜스포머에 쓴다

학습 안정성:
- 그래디언트 절단: 그래디언트의 크기를 제한한다
- 배치 정규화: 엑티베이션을 안정시킨다
- 건너뛰기 연결: 그래디언트가 곧장 흐르게 한다
- 알맞은 초기화: 좋은 출발점

깊이 대 너비:
- 더 깊게: 위계적인 피처가 늘어난다
- 더 넓게: 레이어마다 수용력이 커진다
- 절충점은 문제에 달렸다

깊이가 주는 어려움:
⚠ 그래디언트 소실과 폭발
⚠ 성능 퇴화(정확도가 더 오르지 않는다)
⚠ 메모리와 계산이 더 든다
⚠ 학습 시간이 길어진다

SOLUTIONS:
✓ 잔차 연결
✓ 정규화 레이어
✓ 신중한 초기화
✓ 그래디언트 절단
✓ 러닝레이트 워밍업

CONGRATULATIONS!
튜토리얼 전체를 마쳤다! 🎉
이제 실서비스에 쓸 수 있는 뉴럴 넷트웍을 만들 줄 안다!
""")
plt.show()


if __name__ == "__main__":
    pass
```

## 2. 논의

ResidualBlock은 선형 레이어 두 개와 배치 정규화를 거친 값에 입력을 더하고 마지막에 ReLU를 적용한다. 따라서 블록의 야코비안은 단순한 I+J_F에 마지막 ReLU의 도함수까지 곱한 형태이다. 잔차 연결만으로 경사가 반드시 보존되는 것은 아니다.

DeepNet의 num_blocks는 선형 레이어 수가 아니다. 입력·출력 선형 레이어 두 개와 블록당 두 개가 있으므로, 블록 수 B일 때 선형 레이어는 2B+2개이다. 기본 10블록과 실제 학습에 선택한 15블록도 구분한다.

클리핑은 계산된 경사의 노름을 제한하고, AdamW는 분리된 웨이트 감쇠를 적용한다. 매 에포크 테스트 정확도를 확인하므로 이를 구조 선택에 사용하면 독립 평가 역할이 사라진다. 별도의 검증셋이 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
너비 d인 ResidualBlock의 학습 파라메터 수를 구하라.

</div>

??? success "연습문제 1 풀이"
    선형 레이어 두 개에 $2(d^2+d)$개, BatchNorm 두 개의 배율과 이동에 $4d$개가 있으므로 총 $2d^2+6d$개이다. 이동 통계량은 버퍼라 제외한다. $d=256$이면 $132608$개이다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
15개 잔차 블록의 선형 레이어 수를 구하고 블록 수와 구분하라.

</div>

??? success "연습문제 2 풀이"
    입력 투영 한 개, 블록마다 두 개, 출력 한 개로 총 32개이다. 배치 정규화와 활성화까지 포함해 레이어를 세는 관례는 다를 수 있으므로 무엇을 센 값인지 명시한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
잔차 합 뒤 ReLU가 경사에 미치는 영향을 설명하라.

</div>

??? success "연습문제 3 풀이"
    블록 야코비안은 $D_{\mathrm{ReLU}}(I+J_F)$이다. 합이 음수인 성분은 마지막 ReLU에서 경사가 차단되므로 항등 분기가 있다고 모든 성분에 경사가 그대로 전달되는 것은 아니다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 풀이의 `ModuleList` 코드가 원래 모델에 바로 적용될 수 있는지 검토하라. 입력·출력 차원, 활성화, 생성자와 `forward`에서 필요한 수정을 설명하라.

</div>

??? success "연습문제 4 풀이"
    아래 코드는 깊이를 설정하는 아이디어를 위한 조각이다. 원래 모델의 생성자와 텐서 차원을 반영한 완성 구현은 아니다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `ModuleList`는 하위 모듈을 등록하지만 실행 순서나 활성화를 자동으로 정하지 않는다. 입력 레이어와 출력 레이어는 과제의 차원을 유지하고, 은닉 레이어 사이에 필요한 비선형성을 넣어야 한다. 선형 레이어만 반복하면 아핀 함수 하나로 합쳐진다. 원래 클래스에 없는 `num_layers` 인수를 그대로 넘길 수 없으므로 생성자와 `forward`를 함께 수정해야 한다.

## 정리하며

깊이를 비교할 때 블록 수와 선형 레이어 수를 구분한다. 잔차 연결, 배치 정규화, 클리핑과 AdamW는 서로 다른 역할을 하며 각각의 효과는 검증 실험에서 확인해야 한다.
