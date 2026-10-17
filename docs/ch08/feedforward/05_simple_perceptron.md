# 간단한 퍼셉트론

이 예제는 `nn.Linear(1, 1)`로 $wx+b$를 계산하는 선형 회귀를 학습한다. 고전적인 퍼셉트론의 계단 출력과 분류 갱신 규칙을 구현한 코드는 아니다. 단일 아핀 레이어와 MSE, 옵티마이저가 어떻게 연결되는지 살펴본다.

## 1. 코드

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(42)   # 초기 웨이트와 잡음을 고정해 결과를 재현 가능하게 한다

# ── 1. 답을 아는 데이터를 만든다 ────────────────────────────────────
# y = 2x + 3 에 표준편차 0.5짜리 잡음을 얹는다. 학습이 끝난 뒤
# 웨이트가 2, 바이어스가 3에 가까워지는지로 성공 여부를 판정할 수 있다.
# 모양이 (100, 1)인 까닭은 nn.Linear가 (배치, 특성) 꼴을 받기 때문이다
X = torch.randn(100, 1) * 10
y = 2 * X + 3 + torch.randn(100, 1) * 0.5

# ── 2. 가장 단순한 뉴럴 넷트웍: 뉴런 하나 ────────────────────────────────
# 입력 1개, 출력 1개. 곧 y = wx + b 이며 배울 파라메터는 w와 b 둘뿐이다
model = nn.Linear(1, 1)
loss_fn = nn.MSELoss()                                    # 제곱오차
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)  # 가장 기본적인 경사 하강법

# ── 3. 학습 루프: 네 줄이 전부다 ────────────────────────────────────
# 이 네 줄(순전파-로스-역전파-갱신)이 아무리 큰 뉴럴 넷트웍에서도 그대로다.
# 달라지는 것은 model의 속뿐이다
loss_history = []
for epoch in range(100):
    predictions = model(X)              # 순전파: 지금 웨이트로 예측한다
    loss = loss_fn(predictions, y)      # 얼마나 틀렸는지 잰다

    optimizer.zero_grad()               # 지난 스텝의 그래디언트를 지운다.
                                        # 빼먹으면 그래디언트가 누적되어 발산한다
    loss.backward()                     # 역전파: 로스를 w와 b로 미분한다
    optimizer.step()                    # 그래디언트 반대 방향으로 한 스텝 옮긴다

    loss_history.append(loss.item())    # item()으로 그래프에서 떼어 낸 수만 담는다.
                                        # 텐서째 담으면 계산 그래프가 쌓여 메모리가 샌다

# ── 4. 참값과 견준다 ────────────────────────────────────────────────
print(f"Learned weight: {model.weight.item():.4f} (target: 2.0)")
print(f"Learned bias: {model.bias.item():.4f} (target: 3.0)")
```

**출력:**

```
Learned weight: 2.0008 (target: 2.0)
Learned bias: 2.7283 (target: 3.0)
```

## 2. 논의

`nn.Linear(1, 1)`은 웨이트 하나와 바이어스 하나를 갖는다. 학습은 예측, 로스, 역전파, 갱신의 순서로 진행된다. 입력을 10배 확대한 이 예제에서는 MSE의 곡률이 커지므로 고정 러닝레이트 0.01이 적절한지 확인해야 한다. 100 에포크 후 참 계수에 가까워진다는 결론은 수렴 여부를 확인한 뒤에만 내릴 수 있다.

모멘텀과 웨이트 감쇠를 끈 SGD는 $\theta\leftarrow\theta-\alpha\nabla L$을 수행한다. 이 코드처럼 모든 표본을 한 번에 쓰면 실제 갱신은 전체 배치 경사 하강법이다. 선형 최소제곱은 볼록하지만, 수렴 속도와 안정성은 설계행렬의 고윳값과 러닝레이트에 달려 있다.

뉴런 하나만으로도 선형 관계는 완벽하게 배울 수 있다. 퍼셉트론의 한계는 선형 결정 경계나 선형 함수만 표현할 수 있다는 점이다. 비선형 문제에는 비선형 활성화를 갖는 은닉 레이어가 필요하며 이는 뒤의 튜토리얼에서 다룬다. 다만 복잡한 것을 더하기 전에 이 단순한 경우를 깊이 이해해 두는 것이 필수적이다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
러닝레이트를 0.001과 0.1로 바꿔 보라. 세 러닝레이트의 로스 곡선을 같은 그래프에 그려라. 발산하지 않으면서 가장 빠르게 수렴하는 것은 어느 쪽인가?

</div>

??? success "풀이"
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
    입력이 크게 스케일된 이 예제에서는 0.01이나 0.1을 안정적인 러닝레이트라고 단정할 수 없다. 절편을 포함한 MSE의 헤시안에서 안정 범위를 계산하고 각 로스 곡선을 대조한다. 발산했다면 반복 횟수를 늘리기보다 러닝레이트와 스케일을 점검한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
경사 초기화 호출을 제거하면 현재 단계에서 사용하는 경사는 무엇인가? 에포크 수에 비례한다고 할 수 있는가?

</div>

??? success "풀이"
    현재 경사는 이전 단계에서 남은 경사에 새 경사를 더한 값이다. 갱신으로 파라메터가 바뀌면 새 경사의 크기와 방향도 달라지므로, 에포크 수에 비례하거나 러닝레이트를 같은 비율로 키운 것과 동등하다고 할 수 없다. 의도적인 누적이 아니라면 각 갱신 전에 초기화한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
목표를 $y=5x-2$와 $y=x^2$로 각각 바꿀 때, 단일 아핀 레이어가 표현할 수 있는 관계를 설명하라.

</div>

??? success "풀이"
    $5x-2$는 모형에 포함되므로 적절한 최적화에서 그 계수를 찾을 수 있다. 잡음이 있으면 유한 표본의 최소제곱 계수는 참값과 다를 수 있다. $x^2$는 연속 구간에서 하나의 아핀 함수로 정확히 표현되지 않는다. MSE가 고르는 최선의 아핀 근사는 입력 분포에 따른 최소제곱 해이며 일반적으로 접선은 아니다. $x^2$를 피처로 추가하는 방법도 가능하다.

## 정리하며

단일 아핀 레이어로 선형 회귀의 학습 루프를 구성했다. 볼록 목적함수여도 러닝레이트가 크면 발산할 수 있으므로 입력 스케일을 함께 확인한다. 비선형 목표에는 새로운 피처가나 비선형 모델이 필요하다.
