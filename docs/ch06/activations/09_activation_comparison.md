# 활성화 함수 비교

알맞은 활성화 함수를 고르는 일은 신경망의 학습 속도, 최종 정확도, 경사의 흐름에 큰 영향을 줄 수 있다. 이 체계적인 비교는 흔히 쓰이는 활성화 함수 일곱 가지(ReLU, Leaky ReLU, ELU, GELU, SiLU(Swish), Tanh, Sigmoid)를 이진 분류 과제에서 평가하며 시험 정확도, 수렴 속도, 학습 시간을 잰다.

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
# 만들므로, 활성화마다 초기 가중치가 서로 다르다. 즉 아래 표의 차이에는
# 활성화의 효과와 초기화 운이 섞여 있다. 엄밀히 견주려면 모델을 만들기
# 직전마다 torch.manual_seed(42)를 다시 불러야 한다
torch.manual_seed(42)
np.random.seed(42)

class ComparisonNetwork(nn.Module):
    """여러 활성화 함수를 비교하기 위한 신경망"""
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
        # 마지막 층에는 활성화를 걸지 않는다. 아래에서 쓰는
        # BCEWithLogitsLoss가 시그모이드를 안에 품고 있어서,
        # 여기서 또 걸면 시그모이드가 두 번 적용된다
        x = self.fc4(x)
        return x

def generate_comparison_data():
    # 특징 20개 가운데 15개만 이름표와 관계가 있고 5개는 그 15개의
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
            # 갱신 "전" 가중치의 정확도이고, test_acc는 갱신 "후"
            # 가중치로 새로 계산한다. 같은 에포크에 찍힌 두 값이
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

## 2. 논의

복잡한 과제에서는 GELU와 SiLU(Swish) 같은 현대적인 활성화 함수가 고전적인 ReLU보다 나은 경우가 많다. BERT와 GPT 같은 트랜스포머 구조에서 널리 쓰이는 GELU는 매끄러운 근사를 제공하여 경사의 흐름을 더 좋게 만들 수 있다. SiLU(Swish라고도 한다)는 $x \cdot \sigma(x)$으로, 매끄럽고 단조롭지 않으며 스스로 문을 여닫는다. 두 함수 모두 작은 음수 값을 통과시켜, 뉴런이 영영 0만 내놓는 "죽은 ReLU" 문제를 피한다.

ReLU는 계산이 단순하고 깊은 신경망에서 시그모이드와 tanh를 괴롭히는 경사 소실 문제를 피한다는 점에서 여전히 경쟁력이 있다. Leaky ReLU와 ELU는 계산 효율을 대부분 유지하면서 음수 입력에서 ReLU의 경사가 0이 되는 문제를 해결한다. 이 ReLU 변형들 사이의 실험적 차이는 대체로 작고 과제에 따라 달라진다.

시그모이드와 tanh 활성화는 입력의 절댓값이 클 때 경사가 0에 다가가 역전파 중 경사 소실을 일으키므로 깊은 신경망에서 수렴이 느려지는 경향이 있다. 다만 특정 역할에서는 여전히 유용하다. 시그모이드는 LSTM의 출력 게이트와 이진 분류 출력에, tanh는 순환 신경망의 유계 활성화에 쓰인다. 일반적인 권고는 은닉층에 ReLU로 시작하고, 성능이 부족하면 GELU나 SiLU를 시도하며, 언제나 자신의 문제에서 직접 견주어 보라는 것이다.

## 연습문제

**연습문제 1.**
더 깊은 신경망(4층 대신 6층)으로 비교를 실행하라. 활성화 함수의 상대적인 순위가 바뀌는가? 깊이가 늘어날 때 가장 크게 손해를 보는 활성화는 무엇인가?

??? success "연습문제 1 풀이"
    `ComparisonNetwork`에 층 두 개를 더한다.
    ```python
    class DeeperNetwork(nn.Module):
        def __init__(self, activation_type='relu'):
            super().__init__()
            # ModuleList로 담으면 층이 파라미터로 등록되면서도
            # forward에서 순서를 직접 다룰 수 있다. 여기서는
            # 마지막 층만 활성화를 건너뛰어야 해서 이 형태가 편하다
            self.layers = nn.ModuleList([
                nn.Linear(20, 64), nn.Linear(64, 64),
                nn.Linear(64, 64), nn.Linear(64, 32),
                nn.Linear(32, 32), nn.Linear(32, 1)
            ])
            activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), ...}
            self.activation = activations[activation_type]

        def forward(self, x):
            # 마지막 층을 뺀 다섯 층에만 활성화를 건다. 4층에서 6층으로
            # 늘리면 시그모이드는 0.25 이하인 도함수를 곱하는 횟수가
            # 늘어나 입력 쪽 기울기가 더 빠르게 사그라든다
            for layer in self.layers[:-1]:
                x = self.activation(layer(x))
            return self.layers[-1](x)
    ```
    경사 소실 때문에 시그모이드와 tanh가 깊이 증가에 가장 큰 손해를 본다. ReLU 계열과 GELU/SiLU는 더 안정적인 학습을 유지한다. 깊이가 깊어질수록 현대적인 활성화와 시그모이드/tanh의 격차가 벌어진다.

---

**연습문제 2.**
입력이 클 때 시그모이드의 경사가 사라지는 이유를 수학적으로 설명하라. $\frac{d}{dx}\sigma(x)$을 계산하고 $x = 10$과 $x = -10$에서 값을 구하라.

??? success "연습문제 2 풀이"
    시그모이드 함수는 $\sigma(x) = \frac{1}{1 + e^{-x}}$이다. 그 도함수는 다음과 같다.

    $$
    \sigma'(x) = \sigma(x)(1 - \sigma(x))
    $$

    $x = 10$일 때 $\sigma(10) \approx 0.99995$이므로 $\sigma'(10) \approx 0.99995 \times 0.00005 \approx 4.5 \times 10^{-5}$이다.

    $x = -10$일 때 $\sigma(-10) \approx 0.00005$이므로 $\sigma'(-10) \approx 0.00005 \times 0.99995 \approx 4.5 \times 10^{-5}$이다.

    최대 경사는 $\sigma'(0) = 0.25$이다. 봉우리에서조차 경사가 0.25에 불과하며, 0에서 멀어질수록 지수적으로 줄어든다. 깊은 신경망에서 이렇게 작은 경사를 여러 번 곱하면 경사 소실 문제가 생긴다.

---

**연습문제 3.**
사용자 정의 활성화 함수 $f(x) = x \cdot \tanh(\text{softplus}(x))$(Mish라 부른다)을 구현하라. 비교에 추가하여 다른 활성화들과 견주어 성능을 평가하라.

??? success "연습문제 3 풀이"
    ```python
    class Mish(nn.Module):
        def forward(self, x):
            return x * torch.tanh(F.softplus(x))

    # 활성화 사전에 추가하기:
    activations['mish'] = Mish()
    ```
    Mish는 SiLU처럼 매끄럽고 단조롭지 않다. GELU 및 SiLU와 비슷한 성능을 내는 경향이 있으며 어떤 과제에서는 조금 더 낫기도 하다. 핵심은 $x \cdot \tanh(\text{softplus}(x))$ 형태가 모든 곳에서 매끄러우면서도 작은 음의 경사를 보존한다는 점이며, 이로써 ReLU 같은 희소성의 이점과 매끄러운 경사의 흐름을 함께 얻는다.

## 정리하며

**다룬 것** — 활성화 함수 비교

복잡한 과제에서는 GELU와 SiLU(Swish) 같은 현대적인 활성화 함수가 고전적인 ReLU보다 나은 경우가 많다.

핵심 클래스는 `ComparisonNetwork`, `DeeperNetwork`, `Mish`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
