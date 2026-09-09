# 4단계: 합성곱 신경망

[3단계](03_mlp.md)의 다층 퍼셉트론은 층을 쌓아 비선형성을 얻었지만, 여전히 첫 줄에서 이미지를 784차원 벡터로 펼친다. 그 순간 어느 화소가 어느 화소 옆에 있었는지가 사라진다. 화소를 무작위로 뒤섞어 놓아도(모든 이미지에 같은 순서라면) 결과가 똑같다는 뜻이다.

합성곱 신경망은 그 잃어버린 이웃 관계를 되찾는다. 작은 필터를 이미지 위로 미끄러뜨리며 국소적인 무늬를 읽고, 그 필터를 모든 위치에서 공유한다. 덕분에 숫자가 조금 옆으로 밀려도 같은 특징이 잡히며, 이것이 [1단계](01_template_learning.md)의 템플릿 학습이 가장 약했던 지점이다.

아래 코드는 합성곱 층 두 개와 완전 연결층 두 개만으로 이 장의 네 걸음을 마무리한다.

## 1. 코드

```python
"""
4단계: MNIST 합성곱 신경망

합성곱 층 두 개와 완전 연결층 두 개로 손글씨 숫자를 분류한다.
이 장의 네 걸음 가운데 마지막이며, 앞의 세 걸음과 같은 데이터셋을 쓴다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 데이터
# =============================================================================
# 앞의 세 걸음과 같은 정규화를 쓴다. 0.1307과 0.3081은 MNIST 학습 집합의
# 화소 평균과 표준편차이며, 이 값으로 맞추어야 네 걸음의 결과를 나란히
# 견줄 수 있다
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# =============================================================================
# 모델
# =============================================================================

class CNN(nn.Module):
    """합성곱 층 두 개와 완전 연결층 두 개."""

    def __init__(self):
        super().__init__()
        # 앞의 세 걸음과 결정적으로 다른 점이 여기 있다. 이미지를 펼치지
        # 않고 (1, 28, 28) 모양 그대로 받는다. 그래야 이웃 화소를 함께
        # 볼 수 있다.
        # 3x3 필터 32개가 이미지 전체를 훑는다. 필터는 위치마다 다시
        # 배우는 것이 아니라 모든 위치에서 공유되므로, 왼쪽 위에서 배운
        # 무늬를 오른쪽 아래에서도 그대로 알아본다. 이 가중치 공유가
        # 평행 이동에 강해지는 까닭이다
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 2x2 최대 풀링이 가로세로를 각각 절반으로 줄인다.
        # 28 -> 14 -> 7이 되어 완전 연결층에 들어가는 수가 크게 준다
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        # 두 번 풀링한 뒤의 모양이 (64, 7, 7)이므로 64*7*7 = 3136이다
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 합성곱 -> 활성화 -> 풀링을 두 번 되풀이한다
        x = self.pool(torch.relu(self.conv1(x)))   # (B, 32, 14, 14)
        x = self.pool(torch.relu(self.conv2(x)))   # (B, 64,  7,  7)
        # 공간 구조를 다 쓰고 난 뒤에야 펼친다. 1~3단계가 맨 처음에
        # 펼쳤던 것과 달리, 여기서는 합성곱이 이웃 관계를 이미 활용한
        # 뒤이므로 잃을 것이 없다
        x = x.flatten(1)
        x = self.dropout(torch.relu(self.fc1(x)))
        # 마지막에 소프트맥스를 걸지 않는다. CrossEntropyLoss가 안에
        # 품고 있기 때문이며, 이는 2단계와 같은 규칙이다
        return self.fc2(x)


model = CNN().to(device)
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

# =============================================================================
# 학습
# =============================================================================
# 손실과 최적화 방식은 2단계에서 세운 사슬 그대로다. 바뀐 것은 모델뿐이다
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 5
for epoch in range(1, EPOCHS + 1):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    # 에포크마다 시험 정확도를 재어 진행을 살핀다
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    print(f"Epoch {epoch}/{EPOCHS}  Test Accuracy: {100 * correct / total:.2f}%")

print(f"\nFinal Test Accuracy: {100 * correct / total:.2f}%")
```


**출력:**

```
Trainable parameters: 421,642
Epoch 1/5  Test Accuracy: 98.37%
Epoch 2/5  Test Accuracy: 98.35%
Epoch 3/5  Test Accuracy: 99.02%
Epoch 4/5  Test Accuracy: 99.05%
Epoch 5/5  Test Accuracy: 99.22%

Final Test Accuracy: 99.22%
```

3단계의 97.42%에서 **99.22%**로 올랐다. 남은 오차의 3분의 2가 사라진 셈이다(2.58% → 0.78%).

### 네 걸음을 돌아보며

| 걸음 | 모델 | 매개변수 | 시험 정확도 | 더한 생각 |
|---|---|---|---|---|
| 1 | 템플릿 학습 | 7,850 (고정) | 82.03% | — |
| 2 | 선형 + 소프트맥스 | 7,850 | 92.51% | 가중치를 학습한다 |
| 3 | 다층 퍼셉트론 | 약 100,000 | 97.42% | 비선형성 |
| 4 | 합성곱 신경망 | 421,642 | 99.22% | 이웃 관계 |

1단계와 2단계는 매개변수 수가 **똑같다**. 달라진 것은 그 수를 평균으로 못박느냐 데이터에 맞추어 학습하느냐뿐인데 10%포인트가 넘게 벌어진다. 학습이라는 일이 그 자체로 얼마나 큰 몫인지를 보여 준다.

3단계와 4단계 사이에서는 매개변수가 네 배로 늘지만 정확도는 1.8%포인트만 오른다. 하지만 그 1.8%포인트가 남은 오차의 3분의 2라는 점이 중요하다. 정확도가 높아질수록 한 걸음의 값어치는 남은 오차로 재야 한다.

## 2. 논의

여기서 쓴 CNN 구조는 고전적인 방식을 따른다. 특징을 뽑는 합성곱 층 뒤에 분류를 맡는 완전 연결층이 온다. 첫 합성곱 층은 단일 채널 입력을 특징 맵 32개로 바꾸는데, 각 맵이 모서리나 꼭짓점 같은 서로 다른 저수준 무늬를 잡는다. 둘째 합성곱 층은 이를 엮어 숫자의 모양을 담은 더 높은 수준의 특징 맵 64개를 만든다. 합성곱 블록마다 뒤따르는 최댓값 풀링이 공간 차원을 절반으로 줄여, 가장 두드러진 특징은 지키면서 표현을 간결하게 만든다.

학습 반복문은 표준적인 경사 하강 주기를 구현한다. 순전파로 예측을 구하고, 교차 엔트로피로 손실을 계산하고, 역전파로 기울기를 구하고, 관성을 쓰는 SGD로 매개변수를 갱신한다. 관성은 지난 기울기의 이동 평균을 쌓아 학습을 빠르게 하며, 손실 지형의 평평한 곳에서도 최적화기가 한결같이 나아가도록 돕는다. 학습률 스케줄러는 세대마다 걸음의 크기를 $\gamma = 0.7$배로 줄여, 학습 초반에는 큼직하게 움직이고 나중에는 미세하게 다듬게 한다.

완전 연결층에 확률 0.5로 적용하는 드롭아웃 규제는 학습 동안 뉴런의 절반을 무작위로 0으로 만들어, 어느 한 뉴런에도 기대지 않는 여벌 있는 표현을 배우게 한다. 이는 일반화에 매우 중요하다. 드롭아웃이 없으면 모델이 학습 집합을 외워(학습 정확도 100%) 처음 보는 시험 숫자에는 형편없을 수 있다. 드롭아웃과 관성과 학습률 감쇠를 함께 쓰면 대개 시험 정확도가 99%를 넘는다.

## 연습문제

**연습문제 1.**
모양이 $(1, 28, 28)$인 입력에서 시작하여 Conv2d(kernel=3, padding=1)과 MaxPool2d(2) 블록 두 개를 지나는 CNN 구조에서, 층마다 특징 맵의 공간 차원을 계산하라.

??? success "연습문제 1 풀이"
    입력 모양 $(1, 28, 28)$에서 시작하면 다음과 같다.

    - Conv2d(1, 32, 3, padding=1) 뒤: $(32, 28, 28)$ — padding=1이 공간 크기를 지킨다
    - MaxPool2d(2) 뒤: $(32, 14, 14)$ — 공간 차원이 절반이 된다
    - Conv2d(32, 64, 3, padding=1) 뒤: $(64, 14, 14)$ — padding=1이 공간 크기를 지킨다
    - MaxPool2d(2) 뒤: $(64, 7, 7)$ — 공간 차원이 다시 절반이 된다
    - 펼치기: 특징 $64 \times 7 \times 7 = 3{,}136$개
    - FC(3136, 128) 뒤: 특징 128개
    - FC(128, 10) 뒤: 로짓 10개 (숫자 부류마다 하나)

---

**연습문제 2.**
분류 과제에서 평균 제곱 오차(MSE)보다 `CrossEntropyLoss`을 즐겨 쓰는 까닭을 설명하라. 모델의 예측이 크게 틀렸을 때의 기울기 거동을 살펴보라.

??? success "연습문제 2 풀이"
    `CrossEntropyLoss`은 `LogSoftmax`과 음의 로그 가능도를 합친 것이다. 모델이 옳은 부류에 아주 낮은 확률을 매기면 손실 $-\log(p)$이 매우 커지고 기울기도 강해, 잘못을 바로잡는 강력한 학습 신호를 준다. 반면 분류에 MSE를 쓰면 옳은 부류에 대해 $(1 - p)^2$을 계산하는데, 예측이 아무리 틀려도 기울기의 최댓값이 2이다.

    게다가 교차 엔트로피는 범주형 분포에 자연스러운 손실이다. 예측 분포와 참된 원-핫 이름표 사이의 정보 이론적 거리를 잰다. MSE는 출력 값 10개를 서로 무관한 회귀 표적으로 다루어, 그 값들이 합이 1이어야 하는 확률 분포를 나타낸다는 사실을 무시한다. 그래서 MSE로 하는 최적화는 덜 효율적이고 보정이 나쁜 예측으로 이어질 수 있다.

---

**연습문제 3.**
관성을 쓰는 SGD 대신 Adam 최적화기를 쓰도록 학습 스크립트를 고쳐라. 같은 세대 수만큼 돌리고 마지막 시험 정확도를 견주어라. Adam과 관성 SGD의 핵심 차이를 설명하라.

??? success "연습문제 3 풀이"
    최적화기 줄을 다음으로 바꾼다.
    ```python
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    ```
    학습률이 더 낮다는 점(0.001 대 0.01)에 주의하라. Adam은 대체로 더 작은 학습률에서 가장 잘 통한다.

    핵심 차이는 Adam이 매개변수마다 적응적인 학습률을 지닌다는 점이다. 일차 적률(관성처럼 기울기의 평균)과 이차 적률(기울기 제곱의 평균)을 함께 좇는다. 지금까지 기울기가 컸던 매개변수는 실효 학습률이 작아지고, 기울기가 작았던 매개변수는 커진다. 이런 적응적 거동 덕분에 특히 학습 초반에 더 빨리 수렴할 때가 많지만, 관성 SGD도 잘 맞추면 일반화가 더 나을 수 있다. MNIST에서는 두 최적화기 모두 어렵지 않게 99% 넘는 정확도에 이른다.

## 정리하며

**다룬 것** — MNIST 분류기

여기서 쓴 CNN 구조는 고전적인 방식을 따른다.

앞의 연습문제 3개로 직접 확인할 수 있다.
