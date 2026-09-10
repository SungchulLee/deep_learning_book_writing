# 드롭아웃 정칙화

08_dropout_regularization.py - 과적합 막기. 드롭아웃을 비롯한 정칙화 기법을 쓰는 법을 배운다

순방향 신경망을 이해하는 것은 깊은 신경망을 효과적으로 만들고 학습시키는 데 필수적이다. 이 구현은 그 핵심 개념을 PyTorch로 보여주며, 현대적인 구조의 구성 요소를 직접 다뤄 볼 기회를 준다.

## 1. 코드

```python
"""
08_dropout_regularization.py - 과적합 막기

드롭아웃을 비롯한 정칙화 기법으로 모델이 학습 데이터를
외우지 못하게 하는 법을 배운다.

과적합: 학습 데이터에서는 잘하지만 시험 데이터에서는 못한다
해결책: 일반화를 북돋우는 정칙화 기법

소요 시간: 25~30분 | 난이도: ⭐⭐⭐☆☆
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

print("="*70)
print("Dropout and Regularization")
print("="*70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MNIST 불러오기
transform = transforms.ToTensor()
train_dataset = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128)

# 드롭아웃이 없는 모델 (과적합하기 쉽다)
class NoDropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 드롭아웃이 있는 모델 (일반화가 더 좋다)
class DropoutNet(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        # 인스턴스 하나를 아래에서 두 번 쓴다. 드롭아웃은 배울 것도
        # 기억할 것도 없고 부를 때마다 새 마스크를 뽑을 뿐이라 안전하다.
        # 배치 정규화라면 이동 통계가 섞여 이렇게 쓰면 안 된다
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        # 마지막 층 앞에서 멈추고 출력에는 걸지 않는다. 로짓을 지우면
        # 그 표본의 예측이 통째로 망가진다.
        # 학습 때 살아남은 활성값은 1/(1-p)배로 부풀려진다. 그래야
        # 평균이 유지되어, 평가 때 드롭아웃을 꺼도 눈금이 어긋나지 않는다
        x = self.dropout(x)  # 활성화 뒤의 드롭아웃
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def train_and_evaluate(model, name, epochs=10):
    """모델을 학습시키고 학습/시험 정확도를 기록한다."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    # 주의: 이 weight_decay가 두 모델 모두에 걸린다. 곧 "드롭아웃이
    # 없는 모델"도 정칙화가 아주 없는 것은 아니다. 드롭아웃만의 효과를
    # 보려면 이 값을 0으로 두고 견주어야 한다
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)  # L2 정칙화
    
    train_accs, test_accs = [], []
    
    for epoch in range(epochs):
        # 학습
        model.train()  # 드롭아웃 켜기
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 주의: outputs는 드롭아웃이 켜진 채, 그것도 step() 앞에서
            # 계산한 값이다. 그래서 여기 나오는 학습 정확도는 드롭아웃
            # 모델 쪽이 실제보다 낮게 잡힌다.
            # 이 페이지가 보려는 것이 학습 정확도와 시험 정확도의
            # 간격이므로, 그 간격의 일부는 드롭아웃 자체가 만든 착시다.
            # 정확히 재려면 에포크 끝에 eval()로 학습 집합을 다시 훑어야 한다
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        
        train_acc = 100 * correct / total
        train_accs.append(train_acc)
        
        # 시험
        model.eval()  # 드롭아웃 끄기
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        
        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        
        print(f"{name} - Epoch {epoch+1}: Train={train_acc:.1f}%, Test={test_acc:.1f}%")
    
    return train_accs, test_accs

# 주의: 씨앗을 심지 않아 두 모델의 초기 가중치가 서로 다르다.
# 차이를 온전히 드롭아웃 탓으로 돌리려면 모델을 만들기 직전마다
# torch.manual_seed(42)를 불러야 한다
print("\nTraining WITHOUT Dropout:")
no_drop_train, no_drop_test = train_and_evaluate(NoDropoutNet(), "No Dropout")

print("\nTraining WITH Dropout:")
drop_train, drop_test = train_and_evaluate(DropoutNet(0.5), "Dropout 0.5")

# 비교 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(no_drop_train, 'b-', label='No Dropout', linewidth=2)
ax1.plot(drop_train, 'r-', label='With Dropout', linewidth=2)
ax1.set_title('Training Accuracy', fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(no_drop_test, 'b-', label='No Dropout', linewidth=2)
ax2.plot(drop_test, 'r-', label='With Dropout', linewidth=2)
ax2.set_title('Test Accuracy', fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('08_dropout_comparison.png', dpi=150)
print("\nPlot saved as '08_dropout_comparison.png'")

print("\n" + "="*70)
print("KEY TAKEAWAYS")
print("="*70)
print("""
DROPOUT:
  - 학습 중 뉴런을 무작위로 떨어뜨린다
  - 특징들이 함께 적응하는 것을 막는다
  - 여러 신경망의 앙상블처럼 작동한다
  - 은닉층의 일반적인 비율: 0.2~0.5
  
USAGE:
  model.train()  # 학습 중에는 드롭아웃 켜기
  model.eval()   # 평가 중에는 드롭아웃 끄기
  
그 밖의 정칙화:
  - L2(최적화기의 weight_decay): 큰 가중치에 벌점을 준다
  - L1: 희소성을 북돋운다
  - 조기 종료: 검증 손실이 늘면 멈춘다
  - 데이터 증강: 데이터셋을 인위적으로 늘린다
  
언제 쓰는가:
  ✓ 작은 데이터셋에 큰 모델을 쓸 때
  ✓ 모델이 과적합한다(학습 정확도 >> 시험 정확도)
  ✓ 깊은 신경망
  
드롭아웃 비율:
  - 입력층: 0.1~0.2(낮게)
  - 은닉층: 0.3~0.5(높게)
  - 출력층: 절대 쓰지 마라!
""")

print("\nEXERCISES:")
print("1. Try different dropout rates (0.1, 0.3, 0.7)")
print("2. Compare L1 vs L2 regularization")
print("3. Implement early stopping")
print("4. Add dropout to different positions in network")
plt.show()


if __name__ == "__main__":
    pass
```

## 2. 논의

이 구현은 2개의 클래스(`NoDropoutNet`, `DropoutNet`)를 정의하며, 이들이 함께 작동하여 완전한 순방향 신경망 구조를 이룬다. 각 클래스가 서로 다른 구성 요소를 감싸므로 코드가 모듈식이 되고 확장하기 쉬워진다. `forward` 메서드들이 PyTorch가 자동 미분에 사용하는 계산 그래프를 정의한다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 딥러닝의 기초 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `NoDropoutNet`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `NoDropoutNet`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = NoDropoutNet(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — 드롭아웃 정칙화

이 구현은 2개의 클래스(`NoDropoutNet`, `DropoutNet`)를 정의하며, 이들이 함께 작동하여 완전한 순방향 신경망 구조를 이룬다.

핵심 클래스는 `NoDropoutNet`, `DropoutNet`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
