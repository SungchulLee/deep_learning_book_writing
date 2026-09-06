# 간단한 퍼셉트론

퍼셉트론은 신경망의 가장 기본이 되는 단위로, 선형 변환 $y = wx + b$을 계산하는 뉴런 하나이다. 단순한 선형 관계에 퍼셉트론을 학습시켜 보면 PyTorch의 `nn.Linear`, `nn.MSELoss`, `torch.optim.SGD`을 써서 순전파, 손실 계산, 역전파, 가중치 갱신으로 이어지는 학습 루프 전체를 볼 수 있다.

## 코드

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)

X = torch.randn(100, 1) * 10
y = 2 * X + 3 + torch.randn(100, 1) * 0.5

model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

loss_history = []
for epoch in range(100):
    predictions = model(X)
    loss = loss_fn(predictions, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

print(f"Learned weight: {model.weight.item():.4f} (target: 2.0)")
print(f"Learned bias: {model.bias.item():.4f} (target: 3.0)")
```

## 논의

`nn.Linear(1, 1)` 층은 입력 하나와 출력 하나를 갖는 뉴런 하나를 만들며, 무작위로 초기화된 가중치와 편향 매개변수를 담는다. 학습 루프는 정석적인 네 단계 패턴을 따른다. 예측 계산(순전파), 오차 측정(손실), 경사 계산(역전파), 매개변수 갱신(최적화기 단계)이다. 100 에폭이 지나면 학습된 가중치와 편향이 참값 2.0과 3.0에 가깝게 맞아야 한다.

SGD(확률적 경사 하강법)는 갱신 규칙 $\theta \leftarrow \theta - \alpha \nabla_\theta L$을 그대로 적용하는 가장 단순한 최적화기이다. 학습률 $\alpha = 0.01$이 이동 폭을 정한다. 너무 크면 발산하고 너무 작으면 수렴이 느려진다. 이 단순한 문제에서는 손실 곡면이 전역 최솟값이 하나뿐인 볼록 포물면이므로 SGD가 빠르게 수렴한다.

뉴런 하나만으로도 선형 관계는 완벽하게 배울 수 있다. 퍼셉트론의 한계는 선형 결정 경계나 선형 함수만 표현할 수 있다는 점이다. 비선형 문제에는 비선형 활성화를 갖는 은닉층이 필요하며 이는 뒤의 튜토리얼에서 다룬다. 다만 복잡한 것을 더하기 전에 이 단순한 경우를 깊이 이해해 두는 것이 필수적이다.

## 연습문제

**연습문제 1.**
학습률을 0.001과 0.1로 바꿔 보라. 세 학습률의 손실 곡선을 같은 그래프에 그려라. 발산하지 않으면서 가장 빠르게 수렴하는 것은 어느 쪽인가?

??? success "연습문제 1 풀이"
    ```python
    for lr in [0.001, 0.01, 0.1]:
        model = nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        losses = []
        for epoch in range(100):
            loss = nn.MSELoss()(model(X), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        plt.plot(losses, label=f'lr={lr}')
    plt.legend()
    plt.yscale('log')
    plt.show()
    ```
    `lr=0.01`에서는 빠르고 안정적으로 수렴한다. `lr=0.001`에서는 느리지만 꾸준히 수렴한다. `lr=0.1`에서는 가장 빠르게 수렴하지만 데이터의 규모에 따라 초반에 진동이 보일 수 있다.

---

**연습문제 2.**
`optimizer.zero_grad()` 호출을 없애면 어떻게 되는가? 20 에폭 동안 학습을 돌리며 매 단계 가중치의 경사를 출력하라. 관찰된 거동을 설명하라.

??? success "연습문제 2 풀이"
    `optimizer.zero_grad()`이 없으면 경사가 반복에 걸쳐 누적된다. 경사의 크기가 에폭 수에 비례해 커지면서 매개변수 갱신이 점점 커진다. 이는 사실상 매 단계 학습률을 키우는 셈이어서 진동이나 발산으로 이어진다. 에폭 $n$에서의 가중치 경사는 한 단계 경사의 약 $n$배가 되어 학습이 불안정해진다.

---

**연습문제 3.**
참 관계를 $y = 5x - 2$로 바꾸고 다시 학습시켜라. 모델이 올바른 가중치와 편향을 배우는지 확인하라. 그다음 $y = x^2$을 시도해 보라. 퍼셉트론이 이것을 배울 수 있는가?

??? success "연습문제 3 풀이"
    $y = 5x - 2$은 선형 관계이므로 퍼셉트론이 `weight = 5.0`과 `bias = -2.0`을 배운다. $y = x^2$은 비선형 함수라 퍼셉트론이 배울 수 없다. 모델은 최선의 선형 근사(접선)를 찾겠지만 MSE 손실은 높은 채로 남는다. 이차 함수를 근사하려면 비선형 활성화를 갖는 은닉층이 필요하다.
