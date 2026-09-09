# 배치 정규화 자세히 보기

튜토리얼 08: 배치 정규화. 배울 내용:

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 1. 코드

```python
"""
==============================================================================
튜토리얼 08: 배치 정규화
==============================================================================
난이도: ⭐⭐⭐ 고급

배울 내용:
- 내부 공변량 변화 문제
- 배치 정규화 층
- 배치 정규화가 학습을 개선하는 방식
- 배치 정규화를 언제 어디에 쓰는지

선수 지식:
- 튜토리얼 07 (정칙화 기법)

핵심 개념:
- nn.BatchNorm1d
- 학습의 안정성
- 더 빠른 수렴
- 배치 정규화와 층 정규화
==============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import time

torch.manual_seed(42)

# ==============================================================================
# 들어가며: 내부 공변량 변화 문제
# ==============================================================================
print("=" * 70)
print("Understanding Batch Normalization")
print("=" * 70)
print("""
배치 정규화는 어떤 문제를 푸는가?

내부 공변량 이동:
  - 학습이 진행되면 층 입력의 분포가 바뀐다
  - 층마다 이 변화에 적응해야 한다
  - 학습이 느려진다
  - 가중치 초기화에 신경 써야 한다
  - 학습률에 민감하다

배치 정규화라는 해법:
  - 미니배치마다 층 입력을 정규화한다
  - 안쪽 함께 바뀜의 옮겨감을 줄인다
  - 배움 빠르기를 더 크게 쓸 수 있다
  - 초기화에 대한 민감도를 낮춘다
  - 정칙화 노릇을 한다(드롭아웃처럼)

식(배치 안의 특징마다):
  1. μ = mean(batch)
  2. σ² = variance(batch)
  3. x̂ = (x - μ) / √(σ² + ε)     # 정규화
  4. y = γ * x̂ + β                # 배율 조정과 이동

여기서 γ(감마)와 β(베타)는 학습되는 매개변수이다!
""")

# ==============================================================================
# 1단계: 데이터 불러오기
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 1: Loading MNIST Data")
print("=" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}\n")

transform = transforms.Compose([transforms.ToTensor()])

train_val_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transform, download=True
)

# 80/20 학습/검증 분할
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(
    train_val_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

batch_size = 128  # 배치 정규화 통계를 안정시키려고 배치 크기를 키운다
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Dataset loaded (batch size: {batch_size})")

# ==============================================================================
# 2단계: 모델 정의
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 2: Defining Models (With and Without BatchNorm)")
print("=" * 70)

class NetWithoutBN(nn.Module):
    """
    배치 정규화가 없는 깊은 신경망
    특히 더 깊은 구조에서는 학습이 어려울 수 있다
    """
    def __init__(self):
        super(NetWithoutBN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

class NetWithBN(nn.Module):
    """
    배치 정규화가 있는 깊은 신경망
    
    배치 정규화의 위치:
    - 보통 선형층 뒤, 활성화 앞
    - 활성화 뒤에 두어도 된다는 연구도 있다
    - 여기서는 표준을 따른다: 선형 -> 배치 정규화 -> 활성화
    """
    def __init__(self):
        super(NetWithBN, self).__init__()
        
        self.network = nn.Sequential(
            # 1층
            nn.Linear(28*28, 256),
            # 선형층 뒤, ReLU 앞의 배치 정규화.
            # 이 자리에 두면 ReLU에 들어가는 값이 0 언저리에 모여
            # 절반쯤 살고 절반쯤 죽으므로 신호가 한쪽으로 쏠리지 않는다.
            # 바로 앞 Linear의 편향은 사실상 쓸모가 없다. 여기서 평균을
            # 빼 버리기 때문이며, 그래서 실무에서는 bias=False를 주기도 한다
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # 2층
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            # 3층
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # 출력층 (여기에는 배치 정규화를 두지 않는다)
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)

print("Two models defined:")
print("  1. NetWithoutBN: Standard network")
print("  2. NetWithBN: Network with BatchNorm layers")
print("\nBatchNorm1d:")
print("  - For fully connected layers (1D data)")
print("  - BatchNorm2d exists for CNNs (2D data)")
print("  - BatchNorm3d for 3D data (videos, medical images)")

# ==============================================================================
# 3단계: 배치 정규화의 거동 이해하기
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 3: Understanding BatchNorm Behavior")
print("=" * 70)

# 시연을 위해 간단한 배치 정규화 층 만들기.
# 갓 만든 층이라 아래 출력이 gamma=[1,1,1], beta=[0,0,0],
# running_mean=[0,0,0], running_var=[1,1,1]로 찍힌다. 곧 학습을
# 한 번도 하지 않은 배치 정규화는 eval() 상태에서 입력을 거의
# 그대로 통과시킨다. (x - 0) / sqrt(1 + eps)이기 때문이다.
# 이동 통계는 학습을 거치며 배치마다 조금씩 갱신되어 쌓이는 값이다
demo_bn = nn.BatchNorm1d(3)
print("\nBatchNorm1d layer parameters:")
print(f"  gamma (weight): {demo_bn.weight.data}")
print(f"  beta (bias):    {demo_bn.bias.data}")
print(f"  running_mean:   {demo_bn.running_mean}")
print(f"  running_var:    {demo_bn.running_var}")

print("\nKey points:")
print("  - gamma and beta are learnable (updated by optimizer)")
print("  - running_mean and running_var are NOT learnable")
print("  - During training: uses batch statistics")
print("  - During evaluation: uses running statistics")
print("  - This is why model.train() and model.eval() matter!")

# ==============================================================================
# 4단계: 학습 함수
# ==============================================================================

def train_and_evaluate(model, train_loader, val_loader, optimizer, 
                       criterion, n_epochs, device):
    """모델을 학습시키고 지표를 기록한다"""
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'epoch_time': []
    }
    
    for epoch in range(n_epochs):
        # 주의: GPU에서는 이 시간이 부정확하다. CUDA 연산이 비동기라
        # 파이썬이 먼저 앞서 나가므로, 제대로 재려면 앞뒤로
        # torch.cuda.synchronize()를 불러야 한다.
        # 배치 정규화는 층마다 계산이 더해지므로 에포크 시간이 늘지만,
        # 대신 수렴이 빨라 전체 학습 시간은 줄어드는 것이 보통이다
        start_time = time.time()
        
        # 학습
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # 검증.
        # 배치 정규화가 있는 모델에서 eval()은 선택이 아니다. 이 줄이
        # 없으면 평가에서도 배치 통계를 쓰게 되어, 같은 표본이라도
        # 어떤 표본들과 한 배치에 묶였느냐에 따라 예측이 달라진다.
        # 배치 크기가 1이면 분산이 정의되지 않아 오류까지 난다
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # 지표를 계산한다
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_time'].append(epoch_time)
        
        if (epoch + 1) % 3 == 0:
            print(f"Epoch {epoch+1:2d}/{n_epochs}: "
                  f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, "
                  f"Time: {epoch_time:.2f}s")
    
    return history

# ==============================================================================
# 5단계: 두 모델 모두 학습
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 5: Training Both Models")
print("=" * 70)

n_epochs = 15
# 참고: 배치 정규화에서는 학습률을 높게 잡는 편이 낫다!
# 다만 아래에서 두 모델이 이 값을 함께 쓴다. 정규화의 강점 하나가
# "더 큰 학습률을 견딘다"는 것인데, 같은 값을 주면 그 강점은
# 표에 드러나지 않는다. 정규화 없는 쪽에 0.01이 이미 큰 값이라
# 학습이 불안정해 보인다면, 그것도 이 실험이 말해 주는 바의 하나다
learning_rate = 0.01

criterion = nn.CrossEntropyLoss()

# 모델 1: 배치 정규화 없음
print("\n" + "-" * 70)
print("Training Model WITHOUT BatchNorm")
print("-" * 70)

model1 = NetWithoutBN().to(device)
optimizer1 = optim.SGD(model1.parameters(), lr=learning_rate, momentum=0.9)
history1 = train_and_evaluate(model1, train_loader, val_loader, 
                               optimizer1, criterion, n_epochs, device)

# 모델 2: 배치 정규화 있음
print("\n" + "-" * 70)
print("Training Model WITH BatchNorm")
print("-" * 70)

model2 = NetWithBN().to(device)
optimizer2 = optim.SGD(model2.parameters(), lr=learning_rate, momentum=0.9)
history2 = train_and_evaluate(model2, train_loader, val_loader, 
                               optimizer2, criterion, n_epochs, device)

# ==============================================================================
# 6단계: 결과 비교
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 6: Comparison")
print("=" * 70)

# 시험 집합 평가
model1.eval()
model2.eval()

def test_accuracy(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

test_acc1 = test_accuracy(model1, test_loader, device)
test_acc2 = test_accuracy(model2, test_loader, device)

print("\nFinal Results:")
print("-" * 70)
print(f"Without BatchNorm:")
print(f"  Best Val Accuracy: {max(history1['val_acc']):.2f}%")
print(f"  Test Accuracy: {test_acc1:.2f}%")
print(f"  Avg Time/Epoch: {sum(history1['epoch_time'])/len(history1['epoch_time']):.2f}s")

print(f"\nWith BatchNorm:")
print(f"  Best Val Accuracy: {max(history2['val_acc']):.2f}%")
print(f"  Test Accuracy: {test_acc2:.2f}%")
print(f"  Avg Time/Epoch: {sum(history2['epoch_time'])/len(history2['epoch_time']):.2f}s")

improvement = test_acc2 - test_acc1
print(f"\nImprovement with BatchNorm: {improvement:+.2f}% {'✓' if improvement > 0 else ''}")

# ==============================================================================
# 7단계: 시각화
# ==============================================================================
print("\n" + "=" * 70)
print("STEP 7: Visualizing Training Progress")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
epochs = range(1, n_epochs + 1)

# 학습 손실
axes[0, 0].plot(epochs, history1['train_loss'], 'b-', label='Without BN', linewidth=2)
axes[0, 0].plot(epochs, history2['train_loss'], 'r-', label='With BN', linewidth=2)
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Comparison')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 검증 손실
axes[0, 1].plot(epochs, history1['val_loss'], 'b-', label='Without BN', linewidth=2)
axes[0, 1].plot(epochs, history2['val_loss'], 'r-', label='With BN', linewidth=2)
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_title('Validation Loss Comparison')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 학습 정확도
axes[1, 0].plot(epochs, history1['train_acc'], 'b-', label='Without BN', linewidth=2)
axes[1, 0].plot(epochs, history2['train_acc'], 'r-', label='With BN', linewidth=2)
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy (%)')
axes[1, 0].set_title('Training Accuracy Comparison')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 검증 정확도
axes[1, 1].plot(epochs, history1['val_acc'], 'b-', label='Without BN', linewidth=2)
axes[1, 1].plot(epochs, history2['val_acc'], 'r-', label='With BN', linewidth=2)
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy (%)')
axes[1, 1].set_title('Validation Accuracy Comparison')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/pytorch_feedforward_tutorial/08_batchnorm_comparison.png', dpi=100)
print("Comparison saved as '08_batchnorm_comparison.png'")

# ==============================================================================
# 핵심 정리:
# ==============================================================================
print("\n" + "=" * 70)
print("핵심 정리")
print("=" * 70)
print("""
1. 배치 정규화가 하는 일:
   - 미니배치별로 층 입력을 정규화한다
   - 최적의 크기(γ)와 이동(β)을 학습한다
   - 추론을 위해 이동 통계를 유지한다

2. Benefits:
   ✓ 더 빨리 수렴한다(더 큰 학습률을 쓸 수 있다)
   ✓ 초기화에 대한 민감도를 낮춘다
   ✓ 정칙화 노릇을 한다(일반화가 조금 나아진다)
   ✓ 학습이 더 안정적이다(특히 깊은 신경망에서)

3. 배치 정규화를 어디에 둘까:
   - 보통: 선형 -> 배치 정규화 -> 활성화
   - 선형층 뒤, 활성화 앞
   - 출력층에는 필요 없다
   - 완전 연결층에는 BatchNorm1d를 쓴다
   - 합성곱층에는 BatchNorm2d를 쓴다

4. 학습과 평가:
   - 학습: 배치 통계를 쓴다(현재 배치의 평균과 분산)
   - 평가: 이동 통계를 쓴다(학습 중에 쌓은 값)
   - model.train()과 model.eval()을 늘 알맞게 부르라!

5. 배치 정규화의 변형:
   - BatchNorm1d: 완전 연결층용
   - BatchNorm2d: 2차원 합성곱층용
   - BatchNorm3d: 3차원 합성곱층용
   - LayerNorm: 대안(배치가 아니라 특징 방향으로 정규화한다)
   - GroupNorm: 배치 크기가 작아도 잘 작동한다

6. 배치 정규화를 쓰지 말아야 할 때:
   - 배치 크기가 아주 작을 때(통계를 믿기 어렵다)
   - 순환 신경망(대신 LayerNorm을 쓴다)
   - 배치 사이의 독립성이 중요할 때

7. 흔한 함정:
   ✗ 추론할 때 model.eval() 부르기를 잊는다
   ✗ 배치 크기를 1로 쓴다(통계가 뜻이 없다)
   ✗ 학습 모드와 평가 모드를 헷갈리기

다음 단계:
- 튜토리얼 09: 학습률 스케줄링
- 튜토리얼 10: 고급 구조
- 튜토리얼 11: CIFAR-10 도전
""")

print("\nTraining completed successfully! ✓")
# ==============================================================================


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 2개의 클래스(`NetWithoutBN`, `NetWithBN`)를 정의하며, 이들이 함께 작동하여 완전한 순방향 신경망 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `NetWithoutBN`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `NetWithoutBN`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = NetWithoutBN(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 배치 정규화 자세히 보기

이 구현은 2개의 클래스(`NetWithoutBN`, `NetWithBN`)를 정의하며, 이들이 함께 작동하여 완전한 순방향 신경망 구조를 이룬다.

핵심 클래스는 `NetWithoutBN`, `NetWithBN`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
