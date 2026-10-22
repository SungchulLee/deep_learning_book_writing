# CIFAR-10 기본

두 컨볼루션 레이어와 세 완전 연결 레이어로 CIFAR-10을 분류하는 예제가다. SimpleCNN의 파라메터는 62006개이며 기록된 기본 설정의 테스트 정확도는 10.70%이다. 모멘텀과 BatchNorm을 추가한 비교를 통해 구조와 학습 절차의 영향을 구분해 살펴본다.

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

# nn.Conv2d 와 nn.Linear 의 초기 웨이트도, DataLoader 의 뒤섞기도 무작위다.
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

    덧대기가 없어 5x5 컨볼루션마다 가로세로가 4씩 줄고 2x2 풀링마다 절반이 된다:
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
    running_loss = 0.0      # 200스텝마다 찍고 0으로 되돌리는 값
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
        # 한 세대가 782스텝이므로 200, 400, 600 에서 세 줄만 찍힌다. 남은 182스텝은
        # running_loss 에 쌓였다가 다음 세대 첫머리에서 버려지므로, 세대마다 뒤쪽
        # 23.3%는 이 세 줄에 들어오지 않는다. 그래서 그 182스텝까지 포함한 세대
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

### 2.1 로스가 ln 10 에서 내려오지 않는다

출력에서 먼저 읽을 것은 정확도가 아니라 로스다. 에포크마다 찍힌 평균이 다섯 개 있다.

| 에포크 | 에포크 평균 로스 | 앞 에포크 대비 |
|---|---|---|
| 1 | 2.3045 | — |
| 2 | 2.3036 | −0.0009 |
| 3 | 2.3028 | −0.0008 |
| 4 | 2.3019 | −0.0009 |
| 5 | 2.3010 | −0.0009 |

클래스가 열인 분류 문제에서, 아무것도 배우지 않아 열 클래스에 똑같이 $1/10$씩 나누어 주는 모델의 교차 엔트로피는 다음과 같다.

$$-\ln \frac{1}{10} = \ln 10 \approx 2.3026$$

기록된 평균 로스는 2.3045에서 2.3010으로 작게 줄며 균등 예측의 기준값 $\ln10\approx2.3026$에 가깝다. 그러나 로스가 가깝다는 사실만으로 예측 확률이 모두 균등하거나 그래디언트가 작다는 결론은 나오지 않는다. 이 초반 변화량을 선형으로 외삽해 필요한 학습 기간을 예측할 수도 없다.

클래스별 재현율을 보면 여덟 클래스가 0%이고 개구리 980장과 고양이 90장을 맞힌다. 전체 정확도 10.70%만으로는 이런 불균형한 오류를 알 수 없다. 다만 정답을 맞힌 클래스의 분포만으로 모든 예측 레이블의 분포를 복원할 수는 없으므로 혼동 행렬과 예측 클래스 빈도도 확인한다.

### 2.2 스텝이 모자란 것인가, 용량이 모자란 것인가

낮은 정확도의 원인은 모델 표현력, 러닝레이트·옵티마이저·기간, 전처리와 구현 등 여러 가지이다. 한 번의 결과를 모델 용량의 상한으로 해석하지 않고 조건을 하나씩 바꾸는 비교로 가능한 원인을 좁힌다.

다음 비교는 같은 구조에서 momentum=0.9를 추가한 설정이다. 공통 초기 상태와 데이터 순서까지 맞추면 모멘텀의 효과를 비교하기 쉽다. 시드 번호가 같다는 사실만으로 모든 실행 조건이 같아지는 것은 아니므로 초기 웨이트와 난수 소비 경로도 확인한다.

```python
# =============================================================================
# 6절: 관성 한 낱말만 더해 본다
# =============================================================================
# 4절의 학습은 로스가 ln 10 (= 2.302585...) 언저리에서 내려오지 않았다. 까닭이 용량이라면
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

기록된 정확도는 10.70%에서 41.07%로 30.37%포인트 높아졌다. 이는 기본 실행의 낮은 성능이 해당 구조가 낼 수 있는 정확도의 상한은 아니라는 근거이다. 다만 다른 러닝레이트·기간에서도 모멘텀이 같은 크기의 향상을 준다는 뜻은 아니다.

비교 실행의 클래스별 재현율은 모두 18.50~60.30% 범위에 있다. 어떤 클래스의 정답을 맞히는지가 달라진 관찰이며 학습된 피처의 종류나 개선의 원인을 모두 설명하는 지표는 아니다.

기본 모멘텀 식은 $v_t=\beta v_{t-1}+g_t$, $\theta_t=\theta_{t-1}-\eta v_t$이다. 초기 버퍼가 0이고 그래디언트가 일정한 g라면 $v_t=g(1-\beta^t)/(1-\beta)$이므로 장기적으로 $g/(1-\beta)$에 접근한다. 변화하는 임의의 그래디언트에서 현재 그래디언트의 정확히 10배가 된다는 보장은 없으며, 모멘텀을 단순히 러닝레이트를 10배 키운 것과 동일시하지 않는다.

!!! warning "두 수는 각각 한 번씩만 돌린 값이다"

    표에는 시드 0과 1에서 얻은 두 비교 결과가 있다. 각 실행은 실제 관찰값이지만 두 시드만으로 전체 시드 분포나 신뢰구간을 안정적으로 추정할 수는 없다.

    | 설정 | 시드 0 | 시드 1 | 범위 |
    |---|---|---|---|
    | momentum=0.0 | 10.70% | 11.82% | 1.12%p |
    | momentum=0.9 | 41.07% | 41.55% | 0.48%p |
    | 차이 | 30.37%p | 29.73%p | |

    두 시드 모두에서 모멘텀 설정의 정확도가 약 30%포인트 높다. 이는 이 두 실행에서 방향이 일관된다는 뜻이다. 표의 최댓값−최솟값은 두 값의 범위이며 표준편차나 검정 통계가 아니다. 이를 27배 같은 비율로 비교해 시드의 영향을 통계적으로 배제했다고 주장하지 않는다.

### 2.3 32에서 5까지, 그리고 파라메터가 놓인 자리

이 구조는 LeNet에서 비롯한 고전적인 구조를 따른다. 패딩가 없어 $5 \times 5$ 컨볼루션마다 가로세로가 4씩 줄고, $2 \times 2$ 최댓값 풀링마다 절반이 된다.

$$32 \to 28 \to 14 \to 10 \to 5$$

그래서 펼친 벡터가 $16 \times 5 \times 5 = 400$차원이고, 여기에 완전 연결 레이어 셋이 400 → 120 → 84 → 10으로 이어진다.

마지막 5×5 피처 맵의 한 위치는 컨볼루션·풀링 경로에서 이론적으로 16×16 입력 범위를 참조한다. 완전 연결 레이어는 여러 위치를 결합하므로 최종 클래스 로짓의 참조 범위는 다르다. [심화 모델](07_cifar10_advanced.md)의 마지막 국소 범위와 같아도 채널 수·해상도·경계와 비선형 경로가 달라 표현력이 같지는 않다.

파라메터 62,006개는 다음과 같이 나뉜다.

| 레이어 | 웨이트 | 바이어스 | 합 |
|---|---|---|---|
| conv1 | $6 \times 3 \times 5 \times 5 = 450$ | 6 | 456 |
| conv2 | $16 \times 6 \times 5 \times 5 = 2{,}400$ | 16 | 2,416 |
| fc1 | $120 \times 400 = 48{,}000$ | 120 | 48,120 |
| fc2 | $84 \times 120 = 10{,}080$ | 84 | 10,164 |
| fc3 | $10 \times 84 = 840$ | 10 | 850 |
| **합계** | | | **62,006** |

컨볼루션 두 레이어를 합쳐도 2,872개로 전체의 4.63%이고, `fc1` 하나가 48,120개로 77.61%를 차지한다. 이름은 컨볼루션 뉴럴 넷트웍이지만 무게는 펼치기 바로 다음의 완전 연결 레이어에 쏠려 있다. 심화 모델에서도 같은 쏠림이 `fc1` 96.74%로 나타나므로, 이것은 이 작은 모델만의 성질이 아니다.

기본 구성에는 BatchNorm·드롭아웃·러닝레이트 일정이 없다. 간단한 기준 설정에서 요소를 추가하면 효과를 조사할 수 있지만, 모든 문제에서 모멘텀을 가장 먼저 추가해야 한다는 우선순위가 정해지는 것은 아니다.

### 2.4 이 수를 형제 쪽들과 어떻게 놓을 것인가

이 책이 실제로 잰 값들을 한자리에 모으면 다음과 같다.

| 쪽 | 모델 | 파라메터 | 학습 | 테스트 정확도 |
|---|---|---|---|---|
| [MNIST CNN](../../ch03/mnist/04_cnn.md) | 컨볼루션 2레이어 | 421,642 | Adam, 10에포크 | 99.21% |
| [Fashion-MNIST 분류기](05_fashion_mnist_classifier.md) | 같은 구조 | 421,642 | SGD(모멘텀 0.5), 14에포크 | 85.84% |
| [CIFAR-10 심화](07_cifar10_advanced.md) | 컨볼루션 4레이어 | 2,168,362 | SGD(모멘텀 0.5), 14에포크 | 54.82% |
| 이 예제, 1절 코드 그대로 | 컨볼루션 2레이어 | 62,006 | SGD(모멘텀 없음), 5에포크 | 10.70% |
| 이 예제 + 모멘텀 0.9 | 같은 구조 | 62,006 | SGD(모멘텀 0.9), 5에포크 | 41.07% |
| 이 예제 + 모멘텀 0.9 + 배치 정규화 | 같은 구조 | 62,050 | SGD(모멘텀 0.9), 5에포크 | 57.17% |

표의 기본+BatchNorm+모멘텀 설정은 57.17%, 심화 예제는 54.82%를 기록했다. 파라메터가 많다는 사실만으로 더 높은 정확도를 보장하지 않는 예이다. 각 설정의 모델 선택 과정과 학습 조건이 달라 구조만의 우열을 분리할 수는 없다.

다만 이 표의 줄들을 곧바로 빼서는 안 된다. 옵티마이저·모멘텀·러닝레이트·에포크 수·규제가 줄마다 다르고, 위의 두 줄은 데이터셋도 다르며, 무엇보다 모두 시드 하나로 한 번씩 돌린 값이다. 표가 보여 주는 것은 **자릿수**이지 통제된 비교가 아니다.

다른 페이지의 결과를 참조할 때도 각 실행의 모델·옵티마이저·러닝레이트·기간과 분할을 함께 기록한다. 원인 비교에는 통제된 추가 실험을, 최종 성능 평가에는 선택에 사용하지 않은 테스트셋을 사용한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff easy" title="쉬움"></span>
기록된 전체 정확도 10.70%와 클래스별 재현율은 서로 어떤 정보를 주는가?

</div>

??? success "풀이"
    전체 정확도는 1070/10000이다. 정답 중 개구리가 980개, 고양이가 90개이고 여덟 클래스는 재현율 0%이므로 오류가 클래스별로 다르다. 전체 정확도가 균등 무작위 예측의 기대값 10%와 가깝더라도 같은 예측 분포라는 뜻은 아니다. 마지막 평균 로스와 $\ln10$의 차이는 약 0.0016이지만 이것도 모든 확률이 균등하다는 증명은 아니다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
SimpleCNN의 공간 크기와 마지막 피처 한 위치의 이론적 리셉티브 필드를 구하라.

</div>

??? success "풀이"
    입력 3×32×32는 conv1 뒤 6×28×28, 풀링 뒤 6×14×14, conv2 뒤 16×10×10, 풀링 뒤 16×5×5가 된다. 펼친 크기는 400이다. 리셉티브 필드 r과 누적 스트라이드 j를 각각 1에서 시작해 갱신하면 r은 5,6,14,16, j는 1,2,2,4가 된다. 이는 구조적 경계 범위이다. MaxPool의 선택과 ReLU 비활성화 때문에 한 입력에서 0이 아닌 그래디언트가 그 안의 모든 위치에 나타나지는 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
SGD 5에포크와 Adam 10에포크를 비교할 때 무엇을 알 수 있고, 옵티마이저만의 효과를 알려면 무엇이 필요한가?

</div>

??? success "풀이"
    이전 설명에 기록된 Adam 10에포크 정확도는 62.84%이다. 기본 10.70%와 모멘텀 SGD 41.07%보다 높지만 학습 기간도 달라 옵티마이저만의 효과는 아니다. 같은 초기 상태·분할과 갱신 예산에서 방법별로 정한 조율 범위를 비교하고 반복 실행한다. Adam의 적응형 모멘트가 유용할 수 있지만 이 표만으로 어느 레이어의 그래디언트가 원인인지 확인할 수 없다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
컨볼루션 레이어마다 (ReLU 활성화 앞에) 배치 정규화를 더하라. 고친 구조를 구현하고 파라메터가 몇 개 늘었는지 세어라. 그런 다음 두 가지를 재어 비교하여라. (a) 배치 정규화 + 모멘텀 0.9, (b) 배치 정규화 + 모멘텀 없음. 곧 배치 정규화만으로 1절의 무너진 학습을 되살릴 수 있는가?

</div>

??? success "풀이"
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

    `BatchNorm2d(C)`는 채널마다 배울 값이 둘(눈금 $\gamma$와 바이어스 $\beta$)이므로 $2C$개를 더한다. 여기서는 $2 \times 6 + 2 \times 16 = 44$개이고, 전체는 $62{,}006 + 44 = 62{,}050$개가 된다. 0.07% 늘어난 계산이다. 이동 평균과 이동 분산은 학습되는 값이 아니라 버퍼라서 이 수에 들어가지 않는다.

    시드 0으로 5에포크씩 재면 다음과 같다.

    | 설정 | 파라메터 | 테스트 정확도 |
    |---|---|---|
    | 1절 코드 그대로 | 62,006 | 10.70% |
    | 배치 정규화만, 모멘텀 없음 | 62,050 | 30.78% |
    | 모멘텀 0.9만 | 62,006 | 41.07% |
    | 배치 정규화 + 모멘텀 0.9 | 62,050 | 57.17% |

    기록된 실행에서는 BatchNorm만 추가한 정확도가 30.78%, 모멘텀만 추가한 경우가 41.07%, 둘 다 사용하면 57.17%이다. 효과가 단순 합과 다르다는 사실만으로 두 기법이 같은 원인을 고쳤다고 결론 내릴 수는 없다. 상호작용과 학습 설정의 영향을 확인할 추가 비교가 필요하다.

    BatchNorm은 학습 모드에서 배치·공간 축의 채널별 통계를 사용하고 학습 가능한 스케일·바이어스를 적용한다. epsilon 때문에 분산이 정확히 1이 되는 것은 아니며 선형 변환 후 통계도 달라진다. 레이어별 그래디언트 크기를 항상 같게 만들거나 입력 분포 변화를 완전히 제거하는 기법은 아니다. 평가에서는 기본적으로 이동 통계를 사용한다.

    표에서 읽을 마지막 사실은 44개의 파라메터로 얻은 57.17%가 파라메터 2,168,362개짜리 [심화 모델](07_cifar10_advanced.md)의 54.82%보다 높다는 것이다. 다만 에포크 수와 옵티마이저 설정이 다르므로 이 비교 역시 자릿수를 보는 것이지 통제된 측정이 아니다.

---

## 정리하며

기본 설정의 낮은 정확도는 구조의 성능 상한이 아니다. 모멘텀·BatchNorm 변경 결과는 해당 실행의 관찰로 읽고 원인이나 보편적인 우위를 단정하지 않는다. 공간 크기·국소 리셉티브 필드·최종 분류기의 참조 범위를 구분하고 조건과 반복 실행의 변동을 함께 보고한다.
