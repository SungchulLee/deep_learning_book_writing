# 2레이어 뉴럴 넷트웍

XOR의 네 점은 직선 하나로 분리할 수 없다. 이 예제는 ReLU 은닉 레이어로 비선형 표현을 만들고 시그모이드와 BCE로 이진 분류를 학습한다. 유한한 네 점을 분류하는 예는 연속 함수의 보편 근사 정리와 구별한다.

## 1. 코드

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

# XOR 문제. 입력 네 가지와 그 정답이 전부다.
# 이 문제가 유명한 까닭은 선형으로 갈리지 않기 때문이다. 평면에
# 직선 하나를 어떻게 그어도 (0,1),(1,0)을 (0,0),(1,1)에서 떼어 낼 수 없다.
# 곧 은닉 레이어가 없는 뉴럴 넷트웍은 정확도 75%를 넘지 못한다.
X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = torch.tensor([[0.], [1.], [1.], [0.]])

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 이 ReLU가 이 예제의 전부다. 이것이 없으면 두 선형 레이어의 합성이
        # 다시 선형이 되어 레이어를 아무리 쌓아도 XOR을 풀 수 없다.
        # 비선형이 있어야 은닉 레이어가 입력을 갈릴 수 있는 자리로 옮겨 준다
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        # 출력을 [0,1]로 눌러 확률로 읽는다. 아래 BCELoss가 확률을 받기 때문이다
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        hidden = self.relu(self.fc1(x))
        output = self.sigmoid(self.fc2(hidden))
        return output

# 은닉 유닛 4개. XOR은 2개로도 풀리지만, 초기값을 잘못 만나면
# 국소 최적에 갇히는 일이 잦아 넉넉히 잡았다
model = TwoLayerNet(2, 4, 1)

# BCELoss는 시그모이드를 이미 지난 확률을 받는다.
# 로짓을 그대로 넘기려면 BCEWithLogitsLoss를 써야 하며, 그쪽이
# 수치적으로 더 안정적이라 실무에서는 보통 그쪽을 쓴다
criterion = nn.BCELoss()

# 러닝레이트 0.1은 꽤 크지만, 표본이 4개뿐이라 이 정도라야 5000걸음 안에 수렴한다
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

for epoch in range(5000):
    outputs = model(X)              # 순전파
    loss = criterion(outputs, y)    # 얼마나 틀렸는지
    optimizer.zero_grad()           # 지난 그래디언트를 지우고
    loss.backward()                 # 역전파
    optimizer.step()                # 한 걸음 옮긴다

model.eval()   # 평가 결로 바꾼다(여기서는 드롭아웃이 없어 차이가 없지만 버릇을 들인다)
with torch.no_grad():   # 그래디언트가 필요 없으므로 추적을 끈다
    predictions = model(X)
    # 확률 0.5를 임계값으로 0과 1을 정한다
    predicted_labels = (predictions > 0.5).float()
    accuracy = (predicted_labels == y).float().mean() * 100
print(f"Final Accuracy: {accuracy:.1f}%")
```

**출력:**

```
Final Accuracy: 100.0%
```

## 2. 논의

XOR은 고전적인 비선형 문제이다. 입력이 서로 다르면 출력이 1이고 같으면 0이다. 입력 공간에서 어떤 직선 하나로도 양성 예와 음성 예를 나눌 수 없다. ReLU를 쓰는 은닉 레이어가 데이터를 선형 분리 가능하게 만드는 변환된 표현을 만들어 낸다. 은닉 뉴런 4개면 학습 가능한 파라메터가 17개뿐인데도 XOR 사상을 학습하기에 충분하고도 남는 용량이 된다.

시그모이드 출력에는 확률 입력을 받는 `BCELoss`를 연결한다. 로짓을 직접 출력하도록 바꾼다면 `BCEWithLogitsLoss`를 사용한다. Adam도 러닝레이트 선택이 필요하며, 특정 초기값에서 학습이 정체하거나 수렴하지 않을 수 있다. 반복 횟수와 초기값을 명시해 결과를 해석한다.

결정 경계를 시각화해 보면 은닉 레이어가 입력 공간을 어떻게 여러 영역으로 나누는지 드러난다. 각 은닉 뉴런이 반평면 경계를 구현하고, 그 조합이 비선형 XOR 경계를 만든다. 이것이 핵심 통찰이다. 선형 레이어에 비선형 활성화를 합성하면 비선형 함수가 나온다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
`hidden_size`를 2로 줄여라. 뉴럴 넷트웍이 여전히 XOR을 배울 수 있는가? 여러 난수 시드로 시도하고 10회 중 성공률을 보고하라.

</div>

??? success "연습문제 1 풀이"
    ```python
    successes = 0
    for seed in range(10):
        torch.manual_seed(seed)
        model = TwoLayerNet(2, 2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in range(5000):
            loss = nn.BCELoss()(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            acc = ((model(X) > 0.5).float() == y).float().mean()
            if acc == 1.0:
                successes += 1
    print(f"Success rate: {successes}/10")
    ```
    ReLU 은닉 뉴런 두 개로 $|x_1-x_2|=\operatorname{ReLU}(x_1-x_2)+\operatorname{ReLU}(x_2-x_1)$을 만들면 XOR의 네 입력을 구분할 수 있다. 이런 표현의 존재가 학습 성공을 보장하지는 않는다. 성공 기준을 정하고 실제 시도 횟수와 결과를 보고하며, 근거 없이 특정 성공률을 가정하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
`nn.ReLU()`를 `nn.Identity()`(활성화 없음)로 바꿔라. 뉴럴 넷트웍을 학습시키고 XOR을 배우지 못하는 이유를 설명하라.

</div>

??? success "연습문제 2 풀이"
    활성화가 없으면 뉴럴 넷트웍은 $\sigma(W_2(W_1 x + b_1) + b_2) = \sigma(W_2 W_1 x + W_2 b_1 + b_2)$을 계산한다. 곱 $W_2 W_1$은 하나의 행렬이므로 뉴럴 넷트웍 전체가 선형 레이어 하나에 시그모이드를 붙인 것과 같아진다. 양성 예 $(0,1)$과 $(1,0)$을 음성 예 $(0,0)$과 $(1,1)$로부터 선형으로 분리할 수 없으므로 선형 분류기는 XOR을 풀 수 없다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
$[{-0.5}, 1.5] \times [{-0.5}, 1.5]$의 격자 점들에서 모델을 계산하여 결정 경계를 시각화하라. 예측 곡면을 그리는 데 `plt.contourf`를 쓰라.

</div>

??? success "연습문제 3 풀이"
    ```python
    xx, yy = np.meshgrid(np.arange(-0.5, 1.5, 0.01),
                         np.arange(-0.5, 1.5, 0.01))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        Z = model(grid).numpy().reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap='RdYlBu',
                edgecolors='k', s=200, zorder=10)
    plt.colorbar(label='Prediction')
    plt.show()
    ```
    그림은 네 개의 XOR 점을 가르는 비선형 경계를 보여준다. 경계의 모양은 학습된 웨이트에 따라 달라지며 보통 대각선 방향의 두 영역으로 나타난다.

## 정리하며

ReLU 은닉 레이어는 XOR을 선형으로 구분할 수 있는 표현을 만들 수 있다. 출력 확률에는 BCE를 사용하고, 표현 가능성과 실제 학습 성공을 구분한다. 초기값과 러닝레이트에 따라 수렴 결과가 달라질 수 있다.
