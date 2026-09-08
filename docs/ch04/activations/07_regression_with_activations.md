# 활성화 함수를 쓰는 회귀

회귀 과제는 이산적인 클래스가 아니라 연속적인 값을 예측한다. 구조상 결정적인 선택은 출력층에 활성화 함수를 두지 않아 신경망이 유계가 아닌 예측을 낼 수 있게 하는 것이다. 은닉층에서는 여전히 ReLU 같은 활성화를 써서 비선형성을 들여오며, 덕분에 신경망이 사인 곡선 같은 복잡한 함수도 근사할 수 있다.

## 1. 코드

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)

class RegressionNetwork(nn.Module):
    """회귀 신경망: 연속값을 예측한다"""
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)  # 출력에는 활성화를 쓰지 마라!
        return x

def generate_regression_data(n_samples=500):
    X = np.linspace(-3*np.pi, 3*np.pi, n_samples).reshape(-1, 1)
    y = np.sin(X) + 0.2 * np.random.randn(n_samples, 1)
    X_train = torch.FloatTensor(X)
    y_train = torch.FloatTensor(y)
    return X_train, y_train

def train_regression(model, X, y, epochs=300, lr=0.01):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

if __name__ == "__main__":
    X_train, y_train = generate_regression_data()
    model = RegressionNetwork()
    losses = train_regression(model, X_train, y_train, epochs=300)
    print(f"Final Loss: {losses[-1]:.6f}")
```

## 2. 논의

회귀 신경망과 분류 신경망의 핵심 차이는 출력층에 있다. 회귀에서는 출력 뉴런에 활성화 함수가 없어서 마지막 은닉층 출력의 날것의 선형결합을 낸다. 덕분에 신경망이 어떤 실숫값이든 예측할 수 있다. 출력에 시그모이드나 tanh 같은 활성화를 쓰면 예측이 유계인 범위로 제한되어 그 구간 밖의 목표값에는 적합할 수 없게 된다.

`MSELoss`(평균제곱오차)는 회귀의 표준 손실 함수이다. 큰 오차에 제곱으로 벌점을 주므로 모델이 이상치 예측을 줄이는 데 크게 집중한다. 이상치가 많은 데이터셋에서는 `L1Loss`(평균절대오차)나 `SmoothL1Loss`가 더 강건한 대안이 될 수 있다. 손실 함수의 선택은 모델이 무엇을 최적화하는지를 곧바로 좌우한다. MSE는 조건부 평균을, MAE는 조건부 중앙값을 겨냥한다.

출력의 정의역이 유계라면 출력에 활성화를 두는 것이 적절할 수 있다. $(0, 1)$ 범위의 출력에는 시그모이드를, $(-1, 1)$에는 tanh를, 음이 아닌 출력에는 ReLU나 softplus를 쓴다. 온도, 가격, 좌표처럼 정말로 유계가 아닌 예측에는 활성화를 두지 말아야 한다.

## 연습문제

**연습문제 1.**
`MSELoss`를 `L1Loss`로 바꾸고 같은 데이터로 모델을 다시 학습시켜라. 최종 손실값을 비교하고 예측을 그려라. 어느 손실 함수가 더 매끄러운 곡선을 내는가?

??? success "연습문제 1 풀이"
    ```python
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(300):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
    ```
    MSE는 큰 편차에 더 무거운 벌점을 주어 사실상 잡음을 매끄럽게 다듬으므로 더 매끄러운 곡선을 내는 경향이 있다. L1 손실은 이상치에 더 강건해서 잡음까지 포함해 데이터를 더 가깝게 따라갈 수 있다. 두 손실은 서로 다른 양을 재므로 최종 L1 손실값과 MSE 값을 직접 비교할 수는 없다.

---

**연습문제 2.**
출력층에 시그모이드 활성화를 추가하고 같은 사인 데이터로 학습시켜라. 모델이 데이터를 제대로 적합하지 못하는 이유를 설명하라.

??? success "연습문제 2 풀이"
    시그모이드 출력을 쓰면 모델은 $(0, 1)$ 범위의 값만 예측할 수 있는데, 목표 $\sin(x)$은 $-1$에서 $1$까지 걸쳐 있다. 모델은 출력의 경계에서 포화하고 음수 값은 아예 표현하지 못한다. 모델 구조가 근본적으로 목표 함수를 표현할 수 없으므로 손실은 높은 값에서 정체된다. 이는 출력 활성화가 목표의 정의역과 맞아야 하는 이유를 보여준다.

---

**연습문제 3.**
구간 $[-5, 5]$에서 $y = x^2$을 예측하도록 신경망을 수정하라. 데이터 점 200개를 쓰고 표준편차 2인 가우스 잡음을 더하라. 500 에폭 동안 학습시키고 따로 떼어 둔 시험 집합에서 $R^2$ 점수를 평가하라.

??? success "연습문제 3 풀이"
    ```python
    # 데이터를 생성한다
    X = np.linspace(-5, 5, 200).reshape(-1, 1)
    y = X**2 + np.random.randn(200, 1) * 2
    X_train, X_test = torch.FloatTensor(X[:160]), torch.FloatTensor(X[160:])
    y_train, y_test = torch.FloatTensor(y[:160]), torch.FloatTensor(y[160:])

    # 학습
    model = RegressionNetwork(input_size=1, hidden_size=64)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(500):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    # R^2 평가
    model.eval()
    with torch.no_grad():
        preds = model(X_test)
        ss_res = ((y_test - preds) ** 2).sum()
        ss_tot = ((y_test - y_test.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        print(f"R^2 score: {r2.item():.4f}")
    ```
    잘 학습된 모델이라면 $R^2 > 0.9$을 얻어야 한다. `hidden_size`를 키우거나 층을 더하면 신경망이 이차 함수를 더 정확하게 근사하는 데 도움이 된다.

## 정리하며

**다룬 것** — 활성화 함수를 쓰는 회귀

회귀 신경망과 분류 신경망의 핵심 차이는 출력층에 있다.

핵심 클래스는 `RegressionNetwork`이며 앞의 연습문제 3개로 직접 확인할 수 있다.
