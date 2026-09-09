# 2단계: 선형 모델과 소프트맥스

[1단계](01_template_learning.md)에서 템플릿을 클래스 평균으로 **못박아** 82.03%를 얻었다. 그 절 끝에서 보았듯 이 방법은 선형 분류기이고, 가중치가 평균으로 고정되어 있을 뿐이다. 이제 그 가중치를 풀어 데이터에 맞추어 움직인다. 이것이 이 장에서 처음으로 "학습"이라 부를 만한 일이다.

그런데 무엇을 기준으로 움직여야 하는가? 이 절은 그 물음에 답하고, 답을 따라가면 **최대가능도 추정 → 가능도 → 로그가능도 → 손실 → 경사 하강법**이라는 하나의 사슬이 나온다. 딥러닝의 모든 학습이 이 사슬 위에 있다.

## 1. 가중치를 어떻게 찾는가

### 모델

28×28 이미지를 784차원 벡터로 펼치고, 아핀 변환으로 클래스마다 점수 하나씩을 만든다.

$$
y = xA + b, \qquad A \in \mathbb{R}^{784 \times 10}, \quad b \in \mathbb{R}^{10}
$$

찾아야 할 수는 $784 \times 10 + 10 = 7850$개다. 점수 $y$는 아직 확률이 아니므로 소프트맥스로 바꾼다.

$$
p_k = \frac{e^{y_k}}{\sum_{j=0}^{9} e^{y_j}}
$$

모델의 정의는 여기서 끝이다. 남은 것은 7850개의 수를 정하는 일뿐이다. 자세한 유도는 [3장의 소프트맥스 회귀 기초](../../ch04/softmax_regression/01_fundamentals.md)에 있다.

### 최대가능도 추정

기준은 이것이다. **관찰한 데이터를 가장 그럴듯하게 만드는 매개변수를 고른다.** 이를 최대가능도 추정(maximum likelihood estimation, MLE)이라 한다.

말이 추상적이니 매개변수가 하나뿐인 예로 먼저 보자. 호수의 물고기 수 $N$을 추정한다. 3마리를 잡아 표지를 붙여 놓아주고, 나중에 5마리를 잡았더니 그중 1마리에 표지가 있었다. 이 결과가 나올 확률은 $N$의 함수이다.

$$
P_N = \frac{\binom{3}{1}\binom{N-3}{4}}{\binom{N}{5}}
$$

$N$이 데이터가 아니라 **매개변수**이고, $P_N$이 그 매개변수의 함수인 **가능도**라는 점이 핵심이다. 최대가능도 추정은 $P_N$을 가장 크게 하는 $N$을 고르는 것이며, 계산해 보면 $N = 14$와 $15$에서 $45/91 \approx 0.4945$로 비긴다. 유도는 [표지-재포획 MLE](../../ch04/mle/capture_recapture_mle.md)에 있다.

여기서 매개변수가 하나이고 정수였기 때문에 이웃한 값끼리 견주는 것만으로 최댓값을 찾을 수 있었다. MNIST에서는 매개변수가 7850개이고 모두 실수이므로 그 방법을 쓸 수 없다. 그래서 사슬의 나머지가 필요해진다.

### 가능도에서 손실로

학습 데이터 $(x_1, y_1), \ldots, (x_n, y_n)$이 서로 독립이라고 보면 가능도는 곱으로 쓰인다.

$$
L(A, b) = \prod_{i=1}^{n} p_{y_i}(x_i; A, b)
$$

이 곱을 그대로 다루기는 어렵다. 항이 6만 개나 되고 각 항이 1보다 작아 곱하면 순식간에 0으로 내려가 버린다(부동소수점에서 실제로 0이 된다). 그래서 **로그**를 취한다.

$$
\ell(A, b) = \log L(A, b) = \sum_{i=1}^{n} \log p_{y_i}(x_i; A, b)
$$

로그는 단조 증가 함수이므로 최댓값의 **위치**를 바꾸지 않는다. 곱이 합으로 바뀌어 미분이 쉬워지고, 아주 작은 확률도 큰 음수로 표현되어 수치적으로 안정해진다. 이것이 로그가능도이다.

마지막으로 부호를 뒤집는다. 최적화 도구는 관례상 최소화를 하도록 만들어져 있기 때문이다.

$$
\text{loss} = -\ell(A, b) = -\sum_{i=1}^{n} \log p_{y_i}(x_i; A, b)
$$

가능도를 **최대화**하는 일과 이 손실을 **최소화**하는 일은 정확히 같은 문제이다. 봉우리를 뒤집어 골짜기로 만든 것뿐이다.

그리고 이 손실에는 이미 이름이 있다. **교차 엔트로피**이다. 분류 문제에서 교차 엔트로피를 쓰는 까닭이 여기 있다. 임의로 고른 편리한 함수가 아니라, 최대가능도 추정에서 저절로 따라 나온 것이다.

### 경사 하강법

이제 7850차원 공간에서 손실이 가장 낮은 지점을 찾아야 한다. 손실을 각 매개변수로 미분해 얻은 경사는 가장 가파르게 **올라가는** 방향을 가리키므로, 그 반대로 조금씩 내려간다.

$$
\theta_{n+1} = \theta_n - \lambda \frac{\partial \ell}{\partial \theta}
$$

$\lambda$는 **학습률**이며 한 걸음의 크기를 정한다. 너무 작으면 수렴이 더디고 얕은 극소점에 갇히기 쉬우며, 너무 크면 최솟값을 지나쳐 진동하거나 아예 발산한다. 자세한 내용은 [학습률과 이동 폭](../../ch03/gradient_descent/learning_rate.md)에서 다룬다.

미분값 $\partial \ell / \partial \theta$을 손으로 구할 필요는 없다. PyTorch의 [자동 미분](../../ch03/autograd/01_basic_scalar_backward.md)이 계산 그래프를 거슬러 올라가며 대신 구해 준다.

### 사슬 전체

| 단계 | 하는 일 | 왜 |
|---|---|---|
| 최대가능도 추정 | 데이터를 가장 그럴듯하게 만드는 $(A,b)$를 고른다 | 학습의 기준을 정한다 |
| 가능도 $L$ | 확률들의 곱 | 매개변수의 함수로 본다 |
| 로그가능도 $\ell$ | 곱을 합으로 | 미분이 쉽고 수치적으로 안정 |
| 손실 $-\ell$ | 부호 뒤집기 | 최대화를 최소화로 |
| 경사 하강법 | 조금씩 내려가기 | 7850차원에서 최솟값 찾기 |

아래 코드는 이 사슬을 그대로 옮긴 것이다. `nn.CrossEntropyLoss`가 손실 $-\ell$이고, `loss.backward()`가 $\partial \ell / \partial \theta$이며, `optimizer.step()`이 $\theta - \lambda g$이다.

---

## 2. 코드

```python
"""
2단계: MNIST 선형 모델 + 소프트맥스

은닉층이 하나도 없다. 784차원 입력을 10개 로짓으로 옮기는 아핀 변환
하나뿐이며, 학습할 수는 784*10 + 10 = 7850개다.
1단계가 이 가중치를 클래스 평균으로 못박았다면, 여기서는 데이터에
맞추어 학습한다.
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
# 네 걸음 모두 같은 정규화를 쓴다. 0.1307과 0.3081은 MNIST 학습 집합의
# 화소 평균과 표준편차이며, 이래야 결과를 나란히 견줄 수 있다
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# =============================================================================
# 모델: y = xA + b
# =============================================================================
# nn.Flatten이 (B, 1, 28, 28)을 (B, 784)로 편다. 1단계에서 view로 하던
# 일과 같으며, 여기서도 이 순간 화소의 이웃 관계가 사라진다.
# 그 뒤는 nn.Linear 하나뿐이다. 은닉층도 활성화 함수도 없으므로 이
# 모델이 표현할 수 있는 결정 경계는 선형이다.
# 소프트맥스를 붙이지 않는 까닭은 아래 CrossEntropyLoss가 안에 품고
# 있기 때문이다. 여기서 또 걸면 두 번 적용된다
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 10),
).to(device)

n_params = sum(p.numel() for p in model.parameters())
print(f"Trainable parameters: {n_params:,}")   # 7,850

# =============================================================================
# 손실과 최적화: 1절의 사슬 그대로
# =============================================================================
# CrossEntropyLoss가 곧 손실 -l이다. 안에서 log_softmax와 NLLLoss를
# 합쳐 계산하므로, 소프트맥스를 따로 적용하는 것보다 수치적으로 안정하다
criterion = nn.CrossEntropyLoss()
# Adam이 theta <- theta - lambda * g 갱신을 매개변수마다 조절해 수행한다
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def evaluate(loader):
    """시험 정확도를 백분율로 돌려준다."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            preds = model(images).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


# =============================================================================
# 학습
# =============================================================================
EPOCHS = 10
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()          # 기울기는 누적되므로 매번 지운다
        loss = criterion(model(images), labels)
        loss.backward()                # dl/dtheta 를 채운다
        optimizer.step()               # theta - lambda * g
        running += loss.item() * images.size(0)

    print(f"Epoch {epoch:2d}/{EPOCHS}  "
          f"Loss: {running / len(train_dataset):.4f}  "
          f"Test Accuracy: {evaluate(test_loader):.2f}%")

print(f"\nFinal Test Accuracy: {evaluate(test_loader):.2f}%")
```

**출력:**

```
Trainable parameters: 7,850
Epoch  1/10  Loss: 0.4700  Test Accuracy: 90.86%
...
Epoch 10/10  Loss: 0.2575  Test Accuracy: 92.51%

Final Test Accuracy: 92.51%
```

1단계의 82.03%에서 **92.51%**로 올랐다. 같은 선형 분류기인데 가중치를 클래스 평균으로 못박는 대신 데이터에 맞추어 학습한 것만으로 10%포인트 넘게 얻은 셈이다. 매개변수 수는 7850개로 1단계와 정확히 같다.

여기서 막히는 이유도 분명하다. 이 모델이 그릴 수 있는 결정 경계는 여전히 선형이다. 다음 걸음은 층을 쌓아 그 제약을 푼다.

## 3. 논의

`MNISTClassifier` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다. `forward` 메서드가 계산 그래프를 정의하므로, 학습 중에 PyTorch의 autograd 체계가 경사 계산을 자동으로 처리한다. 이런 모듈식 설계 덕분에 개별 구성 요소를 고치거나 모델을 더 큰 파이프라인에 넣기가 쉬워진다.

학습 루프는 표준적인 PyTorch 패턴을 따른다. 예측을 계산하는 순전파, 손실 계산, 경사를 구하는 역전파, 그리고 최적화기를 통한 매개변수 갱신이다. 에폭에 걸쳐 지표를 추적하면 수렴 양상이 드러나고 과소적합이나 과적합 같은 문제를 진단하는 데 도움이 된다.

시각화는 모델의 거동을 이해하고 학습 문제를 진단하는 데 중요한 역할을 한다. 그림을 그리는 코드는 학습된 표현, 수렴의 움직임, 평가 지표에 대한 통찰을 주어 추상적인 계산을 손에 잡히게 만든다.

여기서 보인 방식은 더 복잡한 상황으로 자연스럽게 확장된다. 초매개변수, 구조의 변형, 여러 데이터셋을 두루 실험해 보면 이해가 깊어지고 다중 클래스 분류 과제에 대한 실용적인 직관이 쌓인다.

## 연습문제

**연습문제 1.**
기본 초기화 상태의 `MNISTClassifier`에서 학습 가능한 매개변수의 총 개수를 계산하라. 가중치와 편향을 모두 포함하여 층별로 나누어 세어라.

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
층이나 블록의 개수를 설정할 수 있도록 `MNISTClassifier`를 확장하라. `__init__`에 `num_layers` 매개변수를 추가하고 `nn.ModuleList`로 깊이가 변하는 구조를 만들어라. 층 2개, 4개, 8개로 시험하라.

??? success "연습문제 4 풀이"
    고정되어 있던 층들을 다음으로 바꾼다.
    ```python
    self.layers = nn.ModuleList()
    for i in range(num_layers):
        self.layers.append(YourBlock(dim, ...))
    ```
    `forward` 메서드에서 `for layer in self.layers: x = layer(x)`처럼 순회한다. (평범한 파이썬 리스트가 아니라) `nn.ModuleList`를 써야 PyTorch가 모든 매개변수를 최적화 대상으로 등록한다. 시험은 다음과 같이 한다. `for n in [2, 4, 8]: model = MNISTClassifier(num_layers=n); print(f'Layers={n}, params={sum(p.numel() for p in model.parameters()):,}')`.

## 정리하며

**다룬 것** — MNIST

`MNISTClassifier` 클래스는 PyTorch의 `nn.Module` 인터페이스를 사용하여 모델 구조를 감싼다.

핵심 클래스는 `MNISTClassifier`이며 앞의 연습문제 4개로 직접 확인할 수 있다.
