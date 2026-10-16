# 엑티베이션 함수 비교

알맞은 엑티베이션 함수를 고르는 일은 뉴럴 넷트웍의 학습 속도, 최종 정확도, 경사의 흐름에 큰 영향을 줄 수 있다. 이 체계적인 비교는 흔히 쓰이는 엑티베이션 함수 일곱 가지(ReLU, Leaky ReLU, ELU, GELU, SiLU(Swish), Tanh, Sigmoid)를 이진 분류 과제에서 평가하며 시험 정확도, 수렴 속도, 학습 시간을 잰다.

## 1. 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

# 씨앗은 여기서 한 번만 심는다. 아래 반복문은 모델을 일곱 개 잇달아
# 만들므로, 활성화마다 초기 웨이트가 서로 다르다. 즉 아래 표의 차이에는
# 활성화의 효과와 초기화 운이 섞여 있다. 엄밀히 견주려면 모델을 만들기
# 직전마다 torch.manual_seed(42)를 다시 불러야 한다
torch.manual_seed(42)
np.random.seed(42)

class ComparisonNetwork(nn.Module):
    """여러 엑티베이션 함수를 비교하기 위한 뉴럴 넷트웍"""
    def __init__(self, activation_type='relu'):
        super().__init__()
        # 폭과 깊이는 일곱 경우 모두 같게 고정한다. 바꾸는 것은
        # 활성화 하나뿐이어야 결과를 활성화 탓으로 돌릴 수 있다
        self.fc1 = nn.Linear(20, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        # 주의: get은 이름을 잘못 적어도 조용히 ReLU로 넘어간다.
        # 'gelu'를 'gleu'로 오타 내면 ReLU 결과가 GELU 이름표를 달고
        # 표에 찍힌다. 실험용 코드라면 activations[activation_type]로
        # 두어 KeyError가 나게 하는 편이 안전하다
        self.activation = activations.get(activation_type, nn.ReLU())
        self.name = activation_type

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        # 마지막 레이어에는 활성화를 걸지 않는다. 아래에서 쓰는
        # BCEWithLogitsLoss가 시그모이드를 안에 품고 있어서,
        # 여기서 또 걸면 시그모이드가 두 번 적용된다
        x = self.fc4(x)
        return x

def generate_comparison_data():
    # 피처 20개 가운데 15개만 이름표와 관계가 있고 5개는 그 15개의
    # 선형 결합이다. 즉 겉보기 차원보다 실제 정보 차원이 낮은,
    # 적당히 어려운 이진 분류 과제를 만든 것이다
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, n_classes=2, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    return X_train, y_train, X_test, y_test

def train_and_evaluate(model, X_train, y_train, X_test, y_test,
                       epochs=150, lr=0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_accs, test_accs, losses = [], [], []
    start_time = time.time()

    for epoch in range(epochs):
        # 미니배치를 나누지 않고 학습 집합 800개를 한꺼번에 넣는다.
        # 그래서 에포크 하나가 곧 경사 하강 한 걸음이다. 150 에포크는
        # 150 걸음일 뿐이므로, 여기서 재는 "수렴 속도"는 미니배치
        # 학습에서의 수렴 속도와 다를 수 있다
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # 주의: train_acc는 step() 앞에서 계산해 둔 logits를 쓰므로
            # 갱신 "전" 웨이트의 정확도이고, test_acc는 갱신 "후"
            # 웨이트로 새로 계산한다. 같은 에포크에 찍힌 두 값이
            # 사실은 다른 모델의 값이라는 뜻이다
            train_acc = ((torch.sigmoid(logits) > 0.5) == y_train).float().mean()
            test_logits = model(X_test)
            test_acc = ((torch.sigmoid(test_logits) > 0.5) == y_test).float().mean()

        train_accs.append(train_acc.item())
        test_accs.append(test_acc.item())
        losses.append(loss.item())

    # 평가까지 포함된 시간이라 순수 학습 시간은 아니다. 다만 일곱 경우가
    # 똑같이 평가를 하므로, 활성화끼리 견주는 용도로는 쓸 만하다
    training_time = time.time() - start_time
    return {
        'final_test_acc': test_accs[-1],
        # best_test_acc는 150 에포크 가운데 시험 정확도가 가장 높았던
        # 값이다. 시험 집합을 보고 고른 값이므로 일반화 성능을 낙관적으로
        # 부풀린다. 정직하게 보고하려면 final_test_acc를 쓰거나,
        # 검증 집합을 따로 떼어 거기서 최고점을 골라야 한다
        'best_test_acc': max(test_accs),
        'training_time': training_time,
        'losses': losses,
    }

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = generate_comparison_data()
    activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'silu', 'tanh', 'sigmoid']
    for act in activations:
        model = ComparisonNetwork(activation_type=act)
        result = train_and_evaluate(model, X_train, y_train, X_test, y_test)
        print(f"{act:12s} | Acc: {result['final_test_acc']:.4f} | "
              f"Best: {result['best_test_acc']:.4f} | "
              f"Time: {result['training_time']:.2f}s")
```

**출력:**

```
relu         | Acc: 0.9550 | Best: 0.9550 | Time: 0.12s
leaky_relu   | Acc: 0.9450 | Best: 0.9450 | Time: 0.10s
elu          | Acc: 0.9450 | Best: 0.9450 | Time: 0.17s
gelu         | Acc: 0.9450 | Best: 0.9550 | Time: 0.27s
silu         | Acc: 0.9500 | Best: 0.9500 | Time: 0.13s
tanh         | Acc: 0.9150 | Best: 0.9150 | Time: 0.16s
sigmoid      | Acc: 0.8700 | Best: 0.8700 | Time: 0.11s
```

## 2. 논의

이 실험의 순위는 특정 데이터, 구조, 초기값과 학습 예산에서 나온 결과이다. 활성화가 달라지면 적절한 초기화와 러닝레이트도 달라질 수 있으므로 같은 고정 설정의 결과와 각 방법을 충분히 조율한 결과를 구분한다.

ReLU의 양수 도함수는 1이지만 전체 뉴럴 넷트웍의 경사 소실을 막지는 않는다. ELU는 음수 영역에서 포화할 수 있고, GELU·SiLU·Mish는 매끄럽지만 작은 도함수나 비단조 영역이 있다. 어느 함수도 학습 성공을 보장하지 않는다.

정확도뿐 아니라 여러 초기값의 변동, 시간, 메모리를 함께 비교한다. GPU의 시간 측정은 비동기 실행을 고려해야 하며, 테스트 정확도로 활성화를 선택한다면 별도의 독립 평가가 필요하다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
더 깊은 뉴럴 넷트웍(4레이어 대신 6레이어)으로 비교를 실행하라. 엑티베이션 함수의 상대적인 순위가 바뀌는가? 깊이가 늘어날 때 가장 크게 손해를 보는 활성화는 무엇인가?

</div>

??? success "연습문제 1 풀이"
    `ComparisonNetwork`에 레이어 두 개를 더한다.
    ```python
    class DeeperNetwork(nn.Module):
        def __init__(self, activation_type='relu'):
            super().__init__()
            # ModuleList로 담으면 레이어가 파라미터로 등록되면서도
            # forward에서 순서를 직접 다룰 수 있다. 여기서는
            # 마지막 레이어만 활성화를 건너뛰어야 해서 이 형태가 편하다
            self.layers = nn.ModuleList([
                nn.Linear(20, 64), nn.Linear(64, 64),
                nn.Linear(64, 64), nn.Linear(64, 32),
                nn.Linear(32, 32), nn.Linear(32, 1)
            ])
            activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), ...}
            self.activation = activations[activation_type]

        def forward(self, x):
            # 마지막 레이어를 뺀 다섯 레이어에만 활성화를 건다. 4레이어에서 6레이어로
            # 늘리면 시그모이드는 0.25 이하인 도함수를 곱하는 횟수가
            # 늘어나 입력 쪽 그래디언트가 더 빠르게 사그라든다
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))
            return self.layers[-1](x)
    ```
    깊어지면 시그모이드와 tanh의 포화가 경사를 줄일 수 있지만 순위가 항상 같은 방향으로 바뀐다고 보장할 수 없다. 초기화와 레이어별 경사를 확인하고 여러 시드에서 비교한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
입력이 클 때 시그모이드의 경사가 사라지는 이유를 수학적으로 설명하라. $\frac{d}{dx}\sigma(x)$을 계산하고 $x = 10$과 $x = -10$에서 값을 구하라.

</div>

??? success "연습문제 2 풀이"
    시그모이드 함수는 $\sigma(x) = \frac{1}{1 + e^{-x}}$이다. 그 도함수는 다음과 같다.

    $$
    \sigma'(x) = \sigma(x)(1 - \sigma(x))
    $$

    $x = 10$일 때 $\sigma(10) \approx 0.99995$이므로 $\sigma'(10) \approx 0.99995 \times 0.00005 \approx 4.5 \times 10^{-5}$이다.

    $x = -10$일 때 $\sigma(-10) \approx 0.00005$이므로 $\sigma'(-10) \approx 0.00005 \times 0.99995 \approx 4.5 \times 10^{-5}$이다.

    최대 경사는 $\sigma'(0) = 0.25$이다. 봉우리에서조차 경사가 0.25에 불과하며, 0에서 멀어질수록 지수적으로 줄어든다. 깊은 뉴럴 넷트웍에서 이렇게 작은 경사를 여러 번 곱하면 경사 소실 문제가 생긴다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
사용자 정의 엑티베이션 함수 $f(x) = x \cdot \tanh(\text{softplus}(x))$(Mish라 부른다)을 구현하라. 비교에 추가하여 다른 활성화들과 비교하여 성능을 평가하라.

</div>

??? success "연습문제 3 풀이"
    ```python
    class Mish(nn.Module):
        def forward(self, x):
            return x * torch.tanh(F.softplus(x))

    # 활성화 사전에 추가하기:
    activations['mish'] = Mish()
    ```
    Mish는 매끄럽고 비단조인 함수이다. ReLU처럼 음수 영역을 정확히 0으로 만드는 함수는 아니므로 같은 희소성 효과를 갖는다고 볼 수 없다. 이 실험 조건에서 검증 성능과 비용을 비교하고 보편적인 우열로 일반화하지 않는다.

## 정리하며

활성화 비교는 초기화, 러닝레이트와 예산을 함께 정의한 실험이다. 한 번의 순위로 모든 과제의 기본 선택을 정하지 말고, 레이어별 경사와 검증 성능·비용을 함께 판단한다.
