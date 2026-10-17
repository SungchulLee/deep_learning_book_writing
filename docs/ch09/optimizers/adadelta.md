# Adadelta

[Adagrad](adagrad.md)는 지난 그래디언트 제곱의 합의 제곱근으로 나누어 파라메터마다 러닝레이트를 맞춘다. 희소한 문제에는 효과적이지만 이 누적이 한없이 커져 실효 러닝레이트가 시간이 갈수록 0으로 줄어든다. Adadelta(Zeiler, 2012)는 유계가 아닌 합을 그래디언트 제곱의 **지수 감쇠 평균**으로 바꾸어 이 문제를 다룬다. 또한 처음 러닝레이트라는 하이퍼파라미터를 아예 없애는 새로운 "단위 보정" 장치를 들여온다.

---

## 1. 동기: 단위가 맞지 않는 문제

표준 SGD에서 파라메터의 갱신 $\Delta\theta = -\eta \, g$은 파라메터 자체가 아니라 그래디언트에 비례하는 단위를 갖는다. $\theta$이 킬로그램 단위의 무게이고 $g = \partial L / \partial \theta$의 단위가 킬로그램당 로스라면, $\eta \, g$의 단위는 임의로 고른 $\eta$에 달려 있게 된다. Adadelta는 $\Delta\theta$이 $\theta$과 같은 "단위"를 갖도록 갱신의 배율을 조정하여 이를 해결한다.

---

## 2. 갱신 규칙

각 시각 $t$에서 파라메터에 대한 로스의 그래디언트를 $g_t = \nabla_\theta L(\theta_t)$이라 하자. 아래의 제곱, 제곱근, 나눗셈은 모두 **원소별** 연산이다.

**1단계.** 그래디언트 제곱의 지수 감쇠 평균 $s_t$을 갱신한다.

$$
s_t = \rho \, s_{t-1} + (1 - \rho) \, g_t^2
$$

**2단계.** 지난 파라메터 갱신의 RMS와 현재 그래디언트 누적값의 RMS의 비를 써서 파라메터의 갱신량 $\Delta\theta_t$을 계산한다.

$$
\Delta\theta_t = -\frac{\sqrt{\delta_{t-1} + \epsilon}}{\sqrt{s_t + \epsilon}} \, g_t
$$

**3단계.** 파라메터 갱신량 제곱의 지수 감쇠 평균 $\delta_t$을 갱신한다.

$$
\delta_t = \rho \, \delta_{t-1} + (1 - \rho) \, (\Delta\theta_t)^2
$$

**4단계.** 갱신을 적용한다.

$$
\theta_{t+1} = \theta_t + \Delta\theta_t
$$

여기서 $\rho \in [0, 1)$은 **감쇠율**(보통 $0.9$)이고, $\epsilon$은 0으로 나누는 것을 막는 작은 상수(예: $10^{-6}$)이다.

---

## 3. 핵심 착상: 단위 보정

분모 $\sqrt{s_t + \epsilon}$은 최근 그래디언트의 RMS이고 분자 $\sqrt{\delta_{t-1} + \epsilon}$은 최근 파라메터 갱신량의 RMS이다. 그 비가 "그래디언트의 단위"를 없애므로 갱신량 $\Delta\theta_t$이 $\theta$과 같은 단위를 갖게 된다. Adadelta가 명시적인 러닝레이트 없이도 작동하는 까닭이 여기에 있다.

!!! note "RMSProp과의 비교"
    (따로, 그리고 같은 무렵에 나온) RMSProp도 분모에 그래디언트 제곱의 지수 감쇠 평균을 쓰지만, 분자에는 지난 갱신량의 RMS 대신 고정된 러닝레이트 $\eta$을 쓴다. Adadelta의 단위 보정이 더 원칙 있는 정식화이지만, 실무에서는 둘 다 [Adam](adam.md)에 자리를 내주었다.

---

## 4. PyTorch 예제

```python
"""간단한 회귀 과제에서 Adadelta 최적화기 시연."""

import torch
import torch.nn as nn

# === 모델과 최적화기 준비 ===

if __name__ == "__main__":
    # 간단한 선형 회귀
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.9, eps=1e-6)

    # 학습 루프 (예시)
    for step in range(100):
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        loss = nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Final loss: {loss.item():.4f}")
```

**출력:**

```
Final loss: 1.2525
```

!!! warning "PyTorch에서의 러닝레이트"
    원래 식은 전역 러닝레이트를 1로 둘 수 있도록 설계되었지만 PyTorch는 lr 배율을 제공한다. 이를 바꾸면 갱신 크기가 달라진다. lr가 무차원 배율이라는 사실만으로 단위 보정이 깨지는 것은 아니며, 효과를 검증해야 한다.

---

## 5. 성질

- **러닝레이트가 필요 없다**(원래 정식화에서는): RMS의 비가 배율을 저절로 맞춰 준다.
- **유계인 누적**: $s_t$의 지수 감쇠가 Adagrad를 괴롭히는 걷잡을 수 없는 누적을 막는다.
- **단위가 맞는 갱신**: 파라메터의 갱신량이 파라메터와 같은 차원의 단위를 갖는다.
- **역사적 의의**: Adadelta와 RMSProp은 거의 같은 무렵에 나왔고 그래디언트 제곱의 지수 이동 평균이라는 같은 핵심 착상을 공유한다.
- **요즘의 쓰임**: 지금은 거의 쓰이지 않는다. 대부분의 응용에서 [Adam](adam.md)과 [AdamW](adamw.md)를 선호한다.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
Adadelta가 보관하는 두 상태를 설명하라. Adam과 무엇이 다른가?

</div>

??? success "풀이"
    경사 제곱의 지수 평균 s와 갱신 제곱의 지수 평균 delta를 보관한다. 갱신량은 sqrt(delta 이전값+epsilon)/sqrt(s 현재값+epsilon)에 현재 경사를 곱해 정한다. 기본식에는 Adam의 1차 모멘텀과 바이어스 보정이 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
그래디언트가 유계가 아닐 때 지수 이동 평균도 언제나 유계라고 할 수 있는가?

</div>

??? success "풀이"
    아니다. 과거 값을 감쇠하지만 계속 커지는 입력 경사를 받으면 상태도 커질 수 있다. 경사가 유계라는 등의 조건에서 상태를 유계로 분석할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
Adadelta의 갱신을 경사열만으로 계산하려면 어떤 정보가 더 필요한가?

</div>

??? success "풀이"
    rho, epsilon, 초기 s와 delta, 적용할 lr 배율이 필요하다. delta는 과거 갱신 제곱을 누적하므로 상태를 순서대로 계산해야 한다. 누적 경사 제곱 하나만으로 실효 러닝레이트를 구하는 방법은 아니다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
Adadelta가 전역 러닝레이트 없이 제안되었다고 조율이 필요 없는가?

</div>

??? success "풀이"
    rho와 epsilon, 입력·목표 스케일과 구현의 lr가 결과에 영향을 준다. 적응형 상태를 가진 알고리즘의 안정성을 일반 GD의 eta<2/L 하나로 판정할 수는 없다. 실제 조건과 검증 결과를 살펴야 한다.

## 정리하며

Adadelta는 경사와 갱신의 RMS 비로 스케일을 조절한다. 초기 러닝레이트를 줄이려는 원래 동기와 PyTorch의 lr 옵션을 구분하고, 지수 평균이 모든 입력에서 유계라는 보장은 피한다.

**참고 문헌**

- Zeiler, M. D. (2012). ADADELTA: An Adaptive Learning Rate Method. *arXiv:1212.5701*.
