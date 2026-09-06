# 활성화 함수 비교

알맞은 활성화 함수를 고르는 일은 신경망의 학습 속도, 최종 정확도, 경사의 흐름에 큰 영향을 줄 수 있다. 이 체계적인 비교는 흔히 쓰이는 활성화 함수 일곱 가지(ReLU, Leaky ReLU, ELU, GELU, SiLU(Swish), Tanh, Sigmoid)를 이진 분류 과제에서 평가하며 시험 정확도, 수렴 속도, 학습 시간을 잰다.

## 코드

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import time

torch.manual_seed(42)
np.random.seed(42)

class ComparisonNetwork(nn.Module):
    """여러 활성화 함수를 비교하기 위한 신경망"""
    def __init__(self, activation_type='relu'):
        super().__init__()
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
        self.activation = activations.get(activation_type, nn.ReLU())
        self.name = activation_type

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

def generate_comparison_data():
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
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            train_acc = ((torch.sigmoid(logits) > 0.5) == y_train).float().mean()
            test_logits = model(X_test)
            test_acc = ((torch.sigmoid(test_logits) > 0.5) == y_test).float().mean()

        train_accs.append(train_acc.item())
        test_accs.append(test_acc.item())
        losses.append(loss.item())

    training_time = time.time() - start_time
    return {
        'final_test_acc': test_accs[-1],
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

## 논의

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
            self.layers = nn.ModuleList([
                nn.Linear(20, 64), nn.Linear(64, 64),
                nn.Linear(64, 64), nn.Linear(64, 32),
                nn.Linear(32, 32), nn.Linear(32, 1)
            ])
            activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), ...}
            self.activation = activations[activation_type]

        def forward(self, x):
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
