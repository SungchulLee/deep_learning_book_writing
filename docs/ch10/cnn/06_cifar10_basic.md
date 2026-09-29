# CIFAR-10 기본

이 쪽은 CIFAR-10을 분류하는 가장 단순한 CNN을 끝까지 돌리고 그 출력을 통째로 싣는다. 필터가 6개와 16개뿐인 $5 \times 5$ 합성곱 층 둘에 완전 연결층 셋을 얹은, 매개변수 62,006개짜리 LeNet 꼴 신경망이다. 데이터는 [CIFAR-10 데이터셋 시각화](03_cifar10_dataset.md)에서 들여다본 그 5만 장이다.

앞선 판본은 이 구조가 "60~70%에 그친다"고 적어 두었다. 그러나 그 수 아래에는 출력 블록이 없었고, 코드를 실제로 돌려서 나오는 시험 정확도는 **10.70%**이다. 부류가 열인 문제에서 무작위로 찍는 값이 10%이니, 이 모델은 다섯 세대 동안 사실상 아무것도 배우지 않았다. 손실도 같은 말을 한다 — 세대 평균이 2.3045에서 2.3010까지, 곧 $\ln 10 \approx 2.3026$ 언저리에서 꼼짝하지 않는다.

까닭은 용량이 아니다. `optim.SGD` 줄에 `momentum=0.9` 한 낱말만 더하고 같은 씨앗으로 같은 5세대를 돌리면 41.07%가 나온다(2.2절). 배치 정규화까지 더하면 57.17%가 되어, 매개변수가 34.9배 많고 세대 수도 2.8배인 [CIFAR-10 심화](07_cifar10_advanced.md)의 54.82%를 넘어선다(연습문제 4). 그러니 이 쪽이 가르치는 것은 "작은 모델은 여기까지"가 아니라, **학습 절차가 기회를 주지 않으면 용량은 아무 말도 하지 않는다**는 것이다.

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

# =============================================================================
# 1절: 설정
# =============================================================================

# nn.Conv2d 와 nn.Linear 의 초기 가중치도, DataLoader 의 뒤섞기도 무작위다.
# 씨앗을 고정해야 아래 출력 블록의 수가 다시 나온다.
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_epochs = 5
batch_size = 64
learning_rate = 0.001

# =============================================================================
# 2절: 데이터 적재
# =============================================================================

# 채널마다 (x - 0.5) / 0.5 = 2x - 1 이므로 [0, 1] 이 [-1, 1] 로 옮겨 간다.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 학습 50,000장 / 시험 10,000장, 부류마다 각각 5,000장과 1,000장이다.
# download=True 인 호출이 둘이라 'Files already downloaded and verified' 가
# 두 줄 나온다.
train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# =============================================================================
# 3절: 모델
# =============================================================================


class SimpleCNN(nn.Module):
    """
    CIFAR-10을 위한 단순한 CNN.
    Conv1: 3->6, 5x5 | Pool | Conv2: 6->16, 5x5 | Pool | FC: 400->120->84->10

    덧대기가 없어 5x5 합성곱마다 가로세로가 4씩 줄고 2x2 풀링마다 절반이 된다:
    32 -> 28 -> 14 -> 10 -> 5. 그래서 펼친 벡터가 16 * 5 * 5 = 400 차원이다.
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

# 62,006개다. 그 가운데 fc1 하나가 400 x 120 + 120 = 48,120개로 77.61%를 차지한다.
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')

criterion = nn.CrossEntropyLoss()
# 관성이 없다 -- optim.SGD 의 momentum 기본값은 0이다. 2.2절에서 이 한 낱말이
# 무엇을 바꾸는지 같은 씨앗으로 재어 본다.
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# =============================================================================
# 4절: 학습
# =============================================================================

for epoch in range(num_epochs):
    running_loss = 0.0      # 200걸음마다 찍고 0으로 되돌리는 값
    epoch_loss = 0.0        # 세대 전체를 더하는 값
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        epoch_loss += loss.item()
        # 한 세대가 782걸음이므로 200, 400, 600 에서 세 줄만 찍힌다. 남은 182걸음은
        # running_loss 에 쌓였다가 다음 세대 첫머리에서 버려지므로, 세대마다 뒤쪽
        # 23.3%는 이 세 줄에 들어오지 않는다. 그래서 그 182걸음까지 포함한 세대
        # 전체의 평균을 epoch_loss 로 따로 찍는다.
        if (i + 1) % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], '
                  f'Loss: {running_loss/200:.4f}')
            running_loss = 0.0
    print(f'Epoch [{epoch+1}/{num_epochs}] Avg Loss: {epoch_loss/len(train_loader):.4f}')

# =============================================================================
# 5절: 평가
# =============================================================================

model.eval()
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0] * 10
    n_class_samples = [0] * 10
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        for label, pred in zip(labels, predicted):
            n_class_samples[label.item()] += 1
            if label == pred:
                n_class_correct[label.item()] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Overall Accuracy: {acc:.2f}%')
    # 부류마다 시험 이미지가 정확히 1,000장씩이라 이 열 줄은 바로 견줄 수 있다.
    for j in range(10):
        print(f'  {classes[j]:>5}: {100.0 * n_class_correct[j] / n_class_samples[j]:5.2f}%'
              f'  ({n_class_correct[j]}/{n_class_samples[j]})')


if __name__ == "__main__":
    pass
```

**출력:**

```
Files already downloaded and verified
Files already downloaded and verified
Total parameters: 62,006
Epoch [1/5], Step [200/782], Loss: 2.3058
Epoch [1/5], Step [400/782], Loss: 2.3032
Epoch [1/5], Step [600/782], Loss: 2.3048
Epoch [1/5] Avg Loss: 2.3045
Epoch [2/5], Step [200/782], Loss: 2.3038
Epoch [2/5], Step [400/782], Loss: 2.3036
Epoch [2/5], Step [600/782], Loss: 2.3035
Epoch [2/5] Avg Loss: 2.3036
Epoch [3/5], Step [200/782], Loss: 2.3032
Epoch [3/5], Step [400/782], Loss: 2.3022
Epoch [3/5], Step [600/782], Loss: 2.3033
Epoch [3/5] Avg Loss: 2.3028
Epoch [4/5], Step [200/782], Loss: 2.3023
Epoch [4/5], Step [400/782], Loss: 2.3015
Epoch [4/5], Step [600/782], Loss: 2.3020
Epoch [4/5] Avg Loss: 2.3019
Epoch [5/5], Step [200/782], Loss: 2.3018
Epoch [5/5], Step [400/782], Loss: 2.3006
Epoch [5/5], Step [600/782], Loss: 2.3012
Epoch [5/5] Avg Loss: 2.3010
Overall Accuracy: 10.70%
  plane:  0.00%  (0/1000)
    car:  0.00%  (0/1000)
   bird:  0.00%  (0/1000)
    cat:  9.00%  (90/1000)
   deer:  0.00%  (0/1000)
    dog:  0.00%  (0/1000)
   frog: 98.00%  (980/1000)
  horse:  0.00%  (0/1000)
   ship:  0.00%  (0/1000)
  truck:  0.00%  (0/1000)
```

## 2. 논의

### 2.1 손실이 ln 10 에서 내려오지 않는다

출력에서 먼저 읽을 것은 정확도가 아니라 손실이다. 세대마다 찍힌 평균이 다섯 개 있다.

| 세대 | 세대 평균 손실 | 앞 세대 대비 |
|---|---|---|
| 1 | 2.3045 | — |
| 2 | 2.3036 | −0.0009 |
| 3 | 2.3028 | −0.0008 |
| 4 | 2.3019 | −0.0009 |
| 5 | 2.3010 | −0.0009 |

부류가 열인 분류 문제에서, 아무것도 배우지 않아 열 부류에 똑같이 $1/10$씩 나누어 주는 모델의 교차 엔트로피는 다음과 같다.

$$-\ln \frac{1}{10} = \ln 10 \approx 2.3026$$

표의 다섯 수가 모두 이 값에서 0.002 안에 있다. 다섯 세대에 걸쳐 손실이 내려간 폭은 $2.3045 - 2.3010 = 0.0035$이고 세대마다 평균 0.00088이다. 이 걸음이 그대로 이어진다고 치면 손실을 1.5까지 끌어내리는 데 $(2.3010 - 1.5) / 0.00088 \approx 910$세대가 걸린다. 손실이 내려가기 시작하면 기울기가 커져 실제로는 더 빨라질 것이므로 910은 예측이 아니다. 다만 지금 걸음이 얼마나 작은지를 재는 눈금으로는 쓸 만하다.

정확도도 같은 말을 한다. 10.70%는 무작위로 찍는 10%보다 0.70%p 높을 뿐이다. 부류별 열 줄을 보면 그 10.70%가 어디서 왔는지가 드러난다. 열 부류 가운데 **여덟 부류가 정확히 0.00%**이고, 맞힌 1,070장은 개구리 980장과 고양이 90장이 전부다. 시험 집합에는 부류마다 정확히 1,000장씩 고르게 들어 있으므로, 이런 표가 나온다는 것은 모델의 답이 몇몇 부류로 쏠려 있다는 뜻이다. "조금 배웠다"가 아니라 **무너졌다**고 읽어야 한다.

### 2.2 걸음이 모자란 것인가, 용량이 모자란 것인가

앞 절이 남긴 물음은 하나다. 손실이 내려오지 않는 것이 이 신경망이 작아서인가, 걸음이 작아서인가? 고쳐야 할 곳이 전혀 다르다. 용량이 문제라면 층과 필터를 늘려야 하고, 걸음이 문제라면 최적화기 한 줄만 고치면 된다.

가르는 방법은 간단하다. 구조도 씨앗도 세대 수도 그대로 두고 `optim.SGD` 줄에 `momentum=0.9` 하나만 더해 보는 것이다. 씨앗이 같으므로 두 학습은 완전히 같은 초기 가중치에서 출발하며, 다른 것은 걸음을 쌓는 방식뿐이다.

```python
# =============================================================================
# 6절: 관성 한 낱말만 더해 본다
# =============================================================================
# 4절의 학습은 손실이 ln 10 (= 2.302585...) 언저리에서 내려오지 않았다. 까닭이 용량이라면
# 최적화 방법을 바꾸어도 달라질 것이 없어야 한다. optimizer 줄에 momentum=0.9
# 하나만 더해, 같은 구조를 같은 씨앗 0으로 같은 5세대 동안 다시 돌려 견준다.


def train_and_score(momentum, seed):
    """SimpleCNN 을 5세대 학습시키고 (시험 정확도, 부류별 맞힌 수, 부류별 장 수)를 돌려준다."""
    torch.manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    net = SimpleCNN().to(device)
    loss_ftn = nn.CrossEntropyLoss()
    opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    net.train()
    for _ in range(num_epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            loss = loss_ftn(net(images), labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

    net.eval()
    correct = [0] * 10
    seen = [0] * 10
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            _, predicted = torch.max(net(images), 1)
            for label, pred in zip(labels, predicted):
                seen[label.item()] += 1
                if label == pred:
                    correct[label.item()] += 1
    return 100.0 * sum(correct) / sum(seen), correct, seen


# 관성 없는 쪽(씨앗 0)은 4절에서 이미 돌렸으므로 acc 를 그대로 가져다 쓴다.
acc_momentum, mom_correct, mom_seen = train_and_score(0.9, 0)
print(f'momentum=0.0 (seed 0, from above): {acc:.2f}%')
print(f'momentum=0.9 (seed 0)            : {acc_momentum:.2f}%')
print(f'gap                              : {acc_momentum - acc:.2f}%p')
for j in range(10):
    print(f'  {classes[j]:>5}: {100.0 * mom_correct[j] / mom_seen[j]:5.2f}%'
          f'  ({mom_correct[j]}/{mom_seen[j]})')
```

**출력:**

```
momentum=0.0 (seed 0, from above): 10.70%
momentum=0.9 (seed 0)            : 41.07%
gap                              : 30.37%p
  plane: 41.00%  (410/1000)
    car: 55.60%  (556/1000)
   bird: 18.50%  (185/1000)
    cat: 26.00%  (260/1000)
   deer: 26.10%  (261/1000)
    dog: 26.10%  (261/1000)
   frog: 60.30%  (603/1000)
  horse: 55.00%  (550/1000)
   ship: 57.60%  (576/1000)
  truck: 44.50%  (445/1000)
```

30.37%p이다. 구조는 한 글자도 바뀌지 않았고 매개변수도 62,006개 그대로인데 정확도가 10.70%에서 41.07%로, 곧 3.84배가 되었다. 그러므로 4절의 10.70%는 용량의 천장이 아니다. 그 수가 말하던 것은 **걸음**이었다.

부류별 줄도 함께 달라진다. 앞 학습에서 여덟 부류가 0.00%였던 자리에 이제 열 부류가 모두 18.50%(새)에서 60.30%(개구리) 사이에 들어와 있다. 무너져 있던 모델이 적어도 열 부류를 구분하려 들기 시작한 것이다.

관성이 이만큼을 바꾸는 까닭은 갱신식으로 보면 분명하다. 관성 없는 SGD는 $\theta \leftarrow \theta - \eta g$로 한 걸음마다 그때의 기울기만큼만 간다. 관성 $\beta$를 붙이면 $v \leftarrow \beta v + g$와 $\theta \leftarrow \theta - \eta v$가 되어, 기울기가 여러 걸음 동안 같은 쪽을 가리키면 $v$가 최대 $1 / (1 - \beta)$배까지 커진다. $\beta = 0.9$이면 10배다. 방향이 꾸준한 성분에서만 학습률을 0.001에서 0.01로 올린 것과 비슷해지고, 걸음마다 방향이 뒤집히는 성분에서는 그렇게 커지지 않는다.

!!! warning "두 수는 각각 한 번씩만 돌린 값이다"

    41.07%도 10.70%도 씨앗 0 하나로 한 번 돌린 결과다. 한 번 돌린 것은 잰 것이 아니므로, 30.37%p라는 차이를 그대로 믿기 전에 씨앗을 바꾸어 퍼짐을 함께 보아야 한다. 같은 두 학습을 씨앗 1로 다시 돌리면 관성 없는 쪽은 **11.82%**, 관성 있는 쪽은 **41.55%**가 나온다.

    | 설정 | 씨앗 0 | 씨앗 1 | 퍼짐 |
    |---|---|---|---|
    | momentum=0.0 | 10.70% | 11.82% | 1.12%p |
    | momentum=0.9 | 41.07% | 41.55% | 0.48%p |
    | 차이 | 30.37%p | 29.73%p | |

    설정 안의 퍼짐이 1.12%p와 0.48%p인데 설정 사이의 차이는 두 씨앗 모두에서 30%p 안팎이다. 큰 쪽 퍼짐의 27배이므로 이 차이는 씨앗이 만든 것이 아니다. 씨앗이 둘뿐이라는 점은 그대로 남으므로, 셋 이상으로 늘려 다시 확인하는 것이 좋다.

### 2.3 32에서 5까지, 그리고 매개변수가 놓인 자리

이 구조는 LeNet에서 비롯한 고전적인 얼개를 따른다. 덧대기가 없어 $5 \times 5$ 합성곱마다 가로세로가 4씩 줄고, $2 \times 2$ 최댓값 풀링마다 절반이 된다.

$$32 \to 28 \to 14 \to 10 \to 5$$

그래서 펼친 벡터가 $16 \times 5 \times 5 = 400$차원이고, 여기에 완전 연결층 셋이 400 → 120 → 84 → 10으로 이어진다.

마지막 $5 \times 5$ 특징 맵의 화소 하나가 원본에서 보는 넓이는 $16 \times 16$이다. 수용 영역은 층을 따라 $r \leftarrow r + (k - 1) j$와 $j \leftarrow j s$로 쌓이므로 $1 \to 5 \to 6 \to 14 \to 16$이 되고, $32 \times 32$ 그림의 넓이로 정확히 4분의 1이다. 합성곱 층을 넷 쓰는 [심화 모델](07_cifar10_advanced.md)의 마지막 특징 맵도 같은 $16 \times 16$을 본다. 두 신경망을 가르는 것은 보는 넓이가 아니라, 그 넓이 안에서 몇 가지를 구별할 수 있는가이다.

매개변수 62,006개는 다음과 같이 나뉜다.

| 층 | 가중치 | 편향 | 합 |
|---|---|---|---|
| conv1 | $6 \times 3 \times 5 \times 5 = 450$ | 6 | 456 |
| conv2 | $16 \times 6 \times 5 \times 5 = 2{,}400$ | 16 | 2,416 |
| fc1 | $120 \times 400 = 48{,}000$ | 120 | 48,120 |
| fc2 | $84 \times 120 = 10{,}080$ | 84 | 10,164 |
| fc3 | $10 \times 84 = 840$ | 10 | 850 |
| **합계** | | | **62,006** |

합성곱 두 층을 합쳐도 2,872개로 전체의 4.63%이고, `fc1` 하나가 48,120개로 77.61%를 차지한다. 이름은 합성곱 신경망이지만 무게는 펼치기 바로 다음의 완전 연결층에 쏠려 있다. 심화 모델에서도 같은 쏠림이 `fc1` 96.74%로 나타나므로, 이것은 이 작은 모델만의 성질이 아니다.

배치 정규화도, 드롭아웃도, Adam이나 학습률 스케줄링도 없다. 이것은 일부러 그런 것이다. 약한 기준선에서 출발해야 하나씩 더할 때마다 그 효과를 따로 잴 수 있다. 다만 2.2절이 보여 준 대로, **맨 처음 더해야 할 것은 층이 아니라 관성**이다.

### 2.4 이 수를 형제 쪽들과 어떻게 놓을 것인가

이 책이 실제로 잰 값들을 한자리에 모으면 다음과 같다.

| 쪽 | 모델 | 매개변수 | 학습 | 시험 정확도 |
|---|---|---|---|---|
| [MNIST CNN](../../ch03/mnist/04_cnn.md) | 합성곱 2층 | 421,642 | Adam, 10세대 | 99.21% |
| [Fashion-MNIST 분류기](05_fashion_mnist_classifier.md) | 같은 구조 | 421,642 | SGD(관성 0.5), 14세대 | 85.84% |
| [CIFAR-10 심화](07_cifar10_advanced.md) | 합성곱 4층 | 2,168,362 | SGD(관성 0.5), 14세대 | 54.82% |
| 이 쪽, 1절 코드 그대로 | 합성곱 2층 | 62,006 | SGD(관성 없음), 5세대 | 10.70% |
| 이 쪽 + 관성 0.9 | 같은 구조 | 62,006 | SGD(관성 0.9), 5세대 | 41.07% |
| 이 쪽 + 관성 0.9 + 배치 정규화 | 같은 구조 | 62,050 | SGD(관성 0.9), 5세대 | 57.17% |

마지막 줄이 이 표에서 가장 할 말이 많다. 매개변수 62,050개짜리 신경망이 5세대 만에 57.17%를 내어, 매개변수가 34.9배 많고 세대도 2.8배인 심화 모델의 54.82%를 넘어선다. "깊고 넓게 만들면 정확도가 따라 오른다"는 이야기는 이 책이 잰 수로는 뒷받침되지 않는다.

다만 이 표의 줄들을 곧바로 빼서는 안 된다. 최적화기·관성·학습률·세대 수·규제가 줄마다 다르고, 위의 두 줄은 데이터셋도 다르며, 무엇보다 모두 씨앗 하나로 한 번씩 돌린 값이다. 표가 보여 주는 것은 **자릿수**이지 통제된 비교가 아니다.

마지막으로 한 가지를 적어 둔다. [CIFAR-10 심화](07_cifar10_advanced.md)는 이 쪽을 가리켜 "기본 쪽이 적어 놓은 60~70%"라고 쓰고, 그 쪽 2.4절의 논지는 이 쪽에 출력 블록이 없다는 사실 위에 서 있다. 이 쪽에 출력이 실린 지금 그 문장들은 낡았다. 견주어야 할 값은 60~70%가 아니라 10.70%(1절 코드 그대로)와 41.07%(관성 한 낱말)이며, 그렇게 바꾸어 놓으면 심화 쪽의 결론 — 용량을 늘려도 학습 절차가 그 용량을 쓰게 해 주지 않으면 소용이 없다 — 은 약해지기는커녕 더 또렷해진다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
1절 출력의 마지막 열한 줄을 근거로, 시험 정확도 10.70%를 "무작위보다 조금 나은 성능"으로 읽어서는 안 되는 까닭을 밝혀라. 세대 평균 손실 2.3010이 $\ln 10$과 얼마나 떨어져 있는지도 함께 적어라.

</div>

??? success "연습문제 1 풀이"
    무작위로 찍는 모델이라면 열 부류의 정확도가 모두 10% 언저리에 고르게 흩어져야 한다. 실제 출력은 그렇지 않다.

    - 여덟 부류(plane, car, bird, deer, dog, horse, ship, truck)가 **정확히 0.00%**이다. 1,000장 가운데 한 장도 맞히지 못했다.
    - 맞힌 1,070장은 개구리 980장(98.00%)과 고양이 90장(9.00%)이 전부다. $980 + 90 = 1{,}070$이고 $1{,}070 / 10{,}000 = 10.70\%$이다.

    시험 집합은 부류마다 정확히 1,000장씩이므로 이 쏠림은 데이터의 불균형 탓이 아니다. 모델의 답이 몇몇 부류에 몰려 있다는 뜻이며, 전체 정확도가 우연히 무작위 수준으로 보일 뿐 속은 무작위와 전혀 다르다. 무작위 모델은 모든 부류를 고르게 조금씩 맞히고, 이 모델은 한 부류를 거의 다 맞히면서 여덟 부류를 통째로 버린다.

    손실 쪽도 같다. $\ln 10 = 2.302585\ldots$이고 마지막 세대 평균이 2.3010이므로 차이는 $2.302585 - 2.3010 \approx 0.0016$에 지나지 않는다. 학습 전 모델이 열 부류에 $1/10$씩 나누어 줄 때의 손실이 바로 $\ln 10$이니, 다섯 세대 동안 손실이 움직인 거리는 그 출발점의 0.07%에 지나지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
$3 \times 32 \times 32$ 입력에서 시작하여 `SimpleCNN` 구조를 따라 공간 차원을 추적하라. 층마다 출력 모양을 보이고 펼친 크기가 정말 $16 \times 5 \times 5 = 400$인지 확인하라. 이어서 마지막 특징 맵의 화소 하나가 원본에서 보는 넓이(수용 영역)를 구하라.

</div>

??? success "연습문제 2 풀이"
    - 입력: $(3, 32, 32)$
    - 덧대기 없는 `Conv2d(3, 6, 5)` 뒤: $(6, 32 - 5 + 1, 32 - 5 + 1) = (6, 28, 28)$
    - `MaxPool2d(2, 2)` 뒤: $(6, 14, 14)$
    - 덧대기 없는 `Conv2d(6, 16, 5)` 뒤: $(16, 14 - 5 + 1, 14 - 5 + 1) = (16, 10, 10)$
    - `MaxPool2d(2, 2)` 뒤: $(16, 5, 5)$
    - 펼치기: $16 \times 5 \times 5 = 400$

    핵심은 덧대기가 없으면 $5 \times 5$ 합성곱마다 공간 차원이 4씩 줄고 $2 \times 2$ 최댓값 풀링마다 절반이 된다는 것이다: $32 \to 28 \to 14 \to 10 \to 5$.

    수용 영역은 $r \leftarrow r + (k - 1) j$와 $j \leftarrow j s$를 층 순서대로 적용하여 쌓는다. $r$은 지금까지 본 넓이, $j$는 한 칸 옮길 때 원본에서 움직이는 거리이다.

    | 층 | $k$ | $s$ | $r$ | $j$ |
    |---|---|---|---|---|
    | 입력 | — | — | 1 | 1 |
    | conv1 | 5 | 1 | $1 + 4 \times 1 = 5$ | 1 |
    | pool1 | 2 | 2 | $5 + 1 \times 1 = 6$ | 2 |
    | conv2 | 5 | 1 | $6 + 4 \times 2 = 14$ | 2 |
    | pool2 | 2 | 2 | $14 + 1 \times 2 = 16$ | 4 |

    마지막 $5 \times 5$ 맵의 화소 하나가 원본에서 $16 \times 16$을 본다. $32 \times 32$ 넓이의 $16^2 / 32^2 = 25\%$이다. 자동 미분으로도 확인할 수 있다. 가운데 칸 하나의 출력에 대해 입력의 기울기를 구하면 0이 아닌 자리가 행과 열 모두 8부터 23까지, 곧 16칸이다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
(관성 없는) SGD를 Adam으로 바꾸고 세대 수를 5에서 10으로 늘려라. 시험 정확도를 재고, 2.2절이 얻은 관성 판본(5세대, 41.07%)과 견주어라. 이 문제에서 Adam이 관성 없는 SGD보다 빨리 나아가는 까닭도 설명하라.

</div>

??? success "연습문제 3 풀이"
    `optim.SGD(model.parameters(), lr=0.001)`을 `optim.Adam(model.parameters(), lr=0.001)`으로 바꾸고 `num_epochs`를 10으로 두면 된다. 씨앗 0으로 재면 시험 정확도는 **62.84%**이다.

    1절 코드 그대로의 10.70%보다 52.14%p 높고, 2.2절의 관성 판본 41.07%보다 21.77%p 높다. 다만 세대 수가 두 배이므로 이 21.77%p를 모두 Adam의 몫으로 돌릴 수는 없다. 두 판본을 제대로 견주려면 세대 수를 맞추어야 한다.

    앞선 판본은 이 연습문제의 답을 "약 63%에서 68~72%로 오른다"고 적어 두었다. 출발점 63%는 이 쪽 코드가 내는 값이 아니다 — 1절이 내는 값은 10.70%다. 공교롭게도 Adam 10세대의 62.84%가 그 63%에 가까운데, 적혀 있던 수가 기준선이 아니라 Adam 쪽 값이었음을 짐작하게 한다.

    Adam이 더 빨리 나아가는 까닭은 기울기의 일차·이차 적률 추정값으로 매개변수마다 적응적인 학습률을 지니기 때문이다. 갱신식은 대략 $\theta \leftarrow \theta - \eta \, \hat{m} / (\sqrt{\hat{v}} + \epsilon)$이고, 분모가 그 매개변수가 최근에 받은 기울기 크기의 제곱근이다. 기울기가 작은 매개변수는 실효 학습률이 커지고, 늘 큰 기울기를 받는 매개변수는 눌린다. 관성 없는 평범한 SGD는 모든 매개변수에 같은 $\eta$를 쓰므로, 기울기의 크기가 층마다 크게 다를 때 — 이 신경망처럼 입력 쪽 합성곱의 기울기가 출력 쪽 완전 연결층보다 훨씬 작을 때 — 앞쪽 층은 거의 움직이지 않는다. 일차 적률 $\hat{m}$ 자체가 관성과 같은 역할을 하므로, 2.2절에서 관성 하나로 얻은 이득의 상당 부분이 Adam 안에도 들어 있다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
합성곱 층마다 (ReLU 활성화 앞에) 배치 정규화를 더하라. 고친 구조를 구현하고 매개변수가 몇 개 늘었는지 세어라. 그런 다음 두 가지를 재어 견주어라. (가) 배치 정규화 + 관성 0.9, (나) 배치 정규화 + 관성 없음. 곧 배치 정규화만으로 1절의 무너진 학습을 되살릴 수 있는가?

</div>

??? success "연습문제 4 풀이"
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

    `BatchNorm2d(C)`는 채널마다 배울 값이 둘(눈금 $\gamma$와 치우침 $\beta$)이므로 $2C$개를 더한다. 여기서는 $2 \times 6 + 2 \times 16 = 44$개이고, 전체는 $62{,}006 + 44 = 62{,}050$개가 된다. 0.07% 늘어난 셈이다. 이동 평균과 이동 분산은 학습되는 값이 아니라 버퍼라서 이 수에 들어가지 않는다.

    씨앗 0으로 5세대씩 재면 다음과 같다.

    | 설정 | 매개변수 | 시험 정확도 |
    |---|---|---|
    | 1절 코드 그대로 | 62,006 | 10.70% |
    | 배치 정규화만, 관성 없음 | 62,050 | 30.78% |
    | 관성 0.9만 | 62,006 | 41.07% |
    | 배치 정규화 + 관성 0.9 | 62,050 | 57.17% |

    답은 "되살릴 수 있다, 다만 관성만 못하다"이다. 매개변수 44개를 더한 것만으로 10.70%가 30.78%로, 20.08%p 오른다. 그러나 관성 하나만 더한 41.07%에는 10.29%p 못 미친다. 그리고 둘을 함께 쓴 57.17%는 각각의 이득을 그냥 더한 $10.70 + 20.08 + 30.37 = 61.15$보다 3.98%p 낮다. 두 장치가 같은 병 — 걸음이 기울기 크기에 견주어 너무 작다 — 을 서로 다른 쪽에서 고치고 있어 효과가 일부 겹치기 때문이다. 관성은 걸음을 키우고, 배치 정규화는 기울기의 크기를 층마다 고르게 만들어 같은 학습률이 모든 층에서 뜻을 갖게 한다.

    배치 정규화가 하는 일은 미니배치 안의 활성값을 채널마다 평균 0, 분산 1로 표준화한 뒤 학습된 아핀 변환 $\gamma x + \beta$를 적용하는 것이다. 이렇게 하면 앞선 층이 갱신될 때마다 뒤 층이 받는 입력의 분포가 흔들리는 일이 줄고, 무엇보다 각 층의 기울기 크기가 층마다 제멋대로 작아지거나 커지지 않는다. 학습률 하나를 모든 층에 똑같이 쓰는 SGD에서 이것이 크게 도움이 되는 까닭이 여기에 있다. 정규화 통계량이 미니배치마다 달라 학습에 잡음이 섞이므로 가벼운 규제 효과도 따라온다.

    표에서 읽을 마지막 사실은 44개의 매개변수로 얻은 57.17%가 매개변수 2,168,362개짜리 [심화 모델](07_cifar10_advanced.md)의 54.82%보다 높다는 것이다. 다만 세대 수와 최적화기 설정이 다르므로 이 비교 역시 자릿수를 보는 것이지 통제된 측정이 아니다.

---

## 정리하며

**다룬 것** — CIFAR-10 기본

필터가 6개와 16개뿐인 매개변수 62,006개짜리 LeNet 꼴 CNN을 1절 코드 그대로 다섯 세대 돌리면 시험 정확도가 **10.70%**, 곧 열 부류 무작위 찍기와 같다. 손실은 다섯 세대 내내 $\ln 10 \approx 2.3026$ 에서 0.002 안에 머물고, 부류별로 보면 여덟 부류가 정확히 0.00%이며 맞힌 1,070장은 개구리 980장과 고양이 90장이 전부다. 이 수는 용량의 천장이 아니다. 같은 씨앗으로 `momentum=0.9` 하나만 더하면 41.07%(+30.37%p), 배치 정규화까지 더하면 매개변수 44개를 보태고 57.17%가 되어 매개변수가 34.9배 많은 심화 모델의 54.82%를 넘어선다. 구조는 $32 \to 28 \to 14 \to 10 \to 5$로 줄어들고 마지막 화소 하나가 원본의 $16 \times 16$을 보며, 매개변수의 77.61%는 합성곱이 아니라 `fc1` 하나에 들어 있다.

앞의 연습문제 4개로 직접 확인할 수 있다.
