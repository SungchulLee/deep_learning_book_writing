# CIFAR-10 기본

이 스크립트는 CIFAR-10 분류를 위한 단순한 CNN을 구현하여 색 이미지 인식의 기준선을 보인다. 일부러 최소한으로 만든 구조, 곧 필터 수가 적은 합성곱 층 두 개는 정확도가 60~70%에 그치는데, 이 뚜렷한 기준선이 뒤이은 실습에서 다룰 더 깊고 정교한 구조의 필요를 일깨운다.

## 1. 코드

```python
"""
06_cifar10_basic.py
===================
CIFAR-10을 위한 기본 CNN (이해하기 쉬운 판본)

난이도: 중간~어려움
예상 시간: 1~2시간

지은이: PyTorch CNN 실습
날짜: 2025년 11월
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 5
batch_size = 64
learning_rate = 0.001

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class SimpleCNN(nn.Module):
    """
    CIFAR-10을 위한 단순한 CNN.
    Conv1: 3->6, 5x5 | Pool | Conv2: 6->16, 5x5 | Pool | FC: 400->120->84->10
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 학습 루프
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (i + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {running_loss/200:.4f}')
            running_loss = 0.0

# 평가
model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Overall Accuracy: {acc:.2f}%')


if __name__ == "__main__":
    pass
```

## 2. 논의

이 단순한 CNN은 LeNet에서 비롯한 고전적인 구조를 따라 $5 \times 5$ 핵과 아주 적은 필터(6개와 16개)를 쓰는 합성곱 층 두 개로 이루어진다. 매개변수의 총수가 적어 자연 이미지에 있는 복잡한 시각 무늬를 배울 용량이 제한된다. 첫 층에 필터가 6개뿐이면 신경망이 잡아낼 수 있는 저수준 특징이 많아야 6가지인데, 이는 CIFAR-10 사진의 다양한 모서리와 질감과 색을 담기에는 턱없이 모자라다.

이 구조는 합성곱-풀링 단계 두 개를 지나며 공간 차원을 $32 \times 32$에서 $5 \times 5$으로 줄이고, 400차원 벡터로 펼쳐 완전 연결층 세 개에 넣는다. 배치 정규화도, 드롭아웃도, Adam이나 학습률 스케줄링 같은 요즘 최적화 기법도 없어 성능이 더 제한된다. 이는 일부러 그런 것이다. 약한 기준선에서 출발해야 구조를 하나씩 개선할 때마다 그 효과를 또렷이 잴 수 있다.

이 구조의 60~70%라는 정확도 천장은 핵심 원리 하나를 보여 준다. 모델의 용량은 문제의 복잡함에 걸맞아야 한다. CIFAR-10의 자연 이미지에는 이 신경망의 매개변수 62,006개가 담을 수 있는 것보다 훨씬 많은 시각적 변화가 들어 있다. 경쟁력 있는 정확도에 다가가려면 필터가 더 많은 깊은 구조와 배치 정규화, 알맞은 규제가 필요하다.

## 연습문제

**연습문제 1.**
$3 \times 32 \times 32$ 입력에서 시작하여 SimpleCNN 구조를 따라 공간 차원을 추적하라. 층마다 출력 모양을 보이고 펼친 크기가 정말 $16 \times 5 \times 5 = 400$인지 확인하라.

??? success "연습문제 1 풀이"

    - 입력: $(3, 32, 32)$
    - 덧대기 없는 Conv2d(3, 6, 5) 뒤: $(6, 32 - 5 + 1, 32 - 5 + 1) = (6, 28, 28)$
    - MaxPool2d(2, 2) 뒤: $(6, 14, 14)$
    - 덧대기 없는 Conv2d(6, 16, 5) 뒤: $(16, 14 - 5 + 1, 14 - 5 + 1) = (16, 10, 10)$
    - MaxPool2d(2, 2) 뒤: $(16, 5, 5)$
    - 펼치기: $16 \times 5 \times 5 = 400$

    핵심은 덧대기가 없으면 $5 \times 5$ 합성곱마다 공간 차원이 4씩 줄고 $2 \times 2$ 최댓값 풀링마다 절반이 된다는 것이다: $32 \to 28 \to 14 \to 10 \to 5$.

---

**연습문제 2.**
(관성 없는) SGD를 Adam으로 바꾸고 세대 수를 5에서 10으로 늘려라. 정확도가 얼마나 오를지 보고하고 이 문제에서 Adam이 더 빨리 수렴하는 까닭을 설명하라.

??? success "연습문제 2 풀이"
    `optim.SGD(model.parameters(), lr=0.001)`을 `optim.Adam(model.parameters(), lr=0.001)`으로 바꾸고 10세대를 학습시키면 대개 정확도가 약 63%에서 68~72%로 오른다.

    Adam이 더 빨리 수렴하는 까닭은 기울기의 일차·이차 적률 추정값으로 매개변수마다 적응적인 학습률을 지니기 때문이다. 기울기가 드물게 오거나 작은 매개변수(특정 무늬에만 반응하는 합성곱 필터에서 흔하다)는 실효 학습률이 커지고, 늘 큰 기울기를 받는 매개변수는 눌린다. 관성 없는 평범한 SGD는 모든 매개변수에 같은 학습률을 쓰는데, 매개변수마다 기울기의 크기가 크게 다를 때는 이것이 최적이 아니다.

---

**연습문제 3.**
합성곱 층마다 (ReLU 활성화 앞에) 배치 정규화를 더하라. 고친 구조를 구현하고 배치 정규화가 학습을 어떻게 돕는지 설명하라.

??? success "연습문제 3 풀이"
    ```python
    class SimpleCNNWithBN(nn.Module):
        def __init__(self):
            super(SimpleCNNWithBN, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.bn1 = nn.BatchNorm2d(6)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.bn2 = nn.BatchNorm2d(16)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.bn1(self.conv1(x))))
            x = self.pool(F.relu(self.bn2(self.conv2(x))))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    ```
    배치 정규화는 미니배치 안의 활성값을 평균 0, 분산 1로 표준화한 뒤 학습된 아핀 변환을 적용한다. 이는 앞선 층이 매개변수를 갱신하면서 층 입력의 분포가 학습 중에 바뀌는 현상인 내부 공변량 이동을 줄여 준다. 배치 정규화를 쓰면 발산하지 않고도 더 높은 학습률을 쓸 수 있어 학습이 더 빨리 수렴한다. 정규화 통계량이 미니배치마다 달라 학습 과정에 잡음이 더해지므로 가벼운 규제 효과도 있다.

## 정리하며

**다룬 것** — CIFAR-10 기본

이 단순한 CNN은 LeNet에서 비롯한 고전적인 구조를 따라 $5 \times 5$ 핵과 아주 적은 필터(6개와 16개)를 쓰는 합성곱 층 두 개로 이루어진다.

핵심 클래스는 `SimpleCNN`, `SimpleCNNWithBN`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
