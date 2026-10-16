# Adagrad

표준 SGD는 모든 파라메터에 같은 러닝레이트 $\eta$을 쓰지만, 파라메터마다 필요한 스텝의 크기가 다를 때가 많다. 자주 활성화되는 피처에 이어진 웨이트는 그래디언트 갱신을 많이 받으므로 진동을 피하려면 스텝이 작아야 하고, 드문 피처에 이어진 웨이트는 갱신이 적으므로 스텝이 클수록 이롭다. **Adagrad**(적응형 그래디언트, Duchi 등, 2011)는 파라메터마다 지난 그래디언트 제곱을 누적해 두고 그것으로 러닝레이트의 배율을 따로 맞추어 이를 다룬다.

---

## 1. 갱신 규칙

각 시각 $t$에서 파라메터에 대한 로스의 그래디언트를 $g_t = \nabla_\theta L(\theta_t)$이라 하자. 아래의 제곱, 제곱근, 나눗셈은 모두 **원소별** 연산이다.

**1단계.** 상태 변수 $s_t$에 그래디언트 제곱의 합을 누적한다.

$$
s_t = s_{t-1} + g_t^2
$$

**2단계.** 파라메터별 실효 러닝레이트 $\eta / (\sqrt{s_t}+\epsilon)$으로 파라메터를 갱신한다.

$$
\theta_{t+1} = \theta_t - \frac{\eta}{(\sqrt{s_t}+\epsilon)} \, g_t
$$

여기서 $\eta > 0$은 전역 러닝레이트고 $\epsilon$은 0으로 나누는 것을 막는 작은 상수(구현별 기본값은 다름)이다. 상태 $s_0$은 0으로 초기화한다.

---

## 2. 희소한 피처 다루기

Adagrad의 파라메터별 배율 조정은 희소한 데이터에 특히 효과적이다. 드문 피처(예: 자연어 처리의 희귀 단어, 추천 시스템의 드문 사용자-항목 상호작용)에 딸린 파라메터는 $s_t$이 작게 쌓이므로 실효 러닝레이트 $\eta / (\sqrt{s_t}+\epsilon)$을 비교적 크게 받는다. 반대로 잦은 피처에 딸린 파라메터는 $s_t$이 크게 쌓여 갱신이 작아진다. 이러한 자동 균형 덕분에 Adagrad는 흔한 피처에서 지나치지 않으면서도 드문 피처에서 뜻있는 진전을 이룬다.

??? example "파라메터 두 개로 보는 직관"
    파라메터 $\theta_1$은 매 단계 크기가 $\approx 10$인 그래디언트를 받고, 파라메터 $\theta_2$은 (작은 크기의 경사에 해당하여) 크기가 $\approx 0.1$인 그래디언트를 받는다고 하자. $T$ 단계 뒤에는 다음과 같다.

    - $s_{T,1} \approx 100T$이므로 $\theta_1$의 실효 러닝레이트는 $\eta / \sqrt{100T} = \eta / (10\sqrt{T})$이다.
    - $s_{T,2} \approx 0.01T$이므로 $\theta_2$의 실효 러닝레이트는 $\eta / \sqrt{0.01T} = 10\eta / \sqrt{T}$이다.

    파라메터 $\theta_2$은 $\theta_1$보다 $100$배 큰 실효 러닝레이트를 받아 그래디언트 신호가 드문 것을 메운다.

---

## 3. 감쇠 문제

$s_t$은 음이 아닌 항의 합이므로 단조 증가한다. 따라서 실효 러닝레이트 $\eta / (\sqrt{s_t}+\epsilon)$은 시간이 갈수록 **단조 감소**한다. 학습 기간이 짧은 볼록 문제에서는 이 자동 어닐링이 장점이다. 그러나 딥러닝에서는 학습이 수백만 단계에 이를 수 있고, 실효 러닝레이트가 너무 세게 줄어들어 수렴하기 훨씬 전에 모델이 사실상 학습을 멈출 수 있다.

이 근본적인 한계가 [RMSProp](rmsprop.md), [Adadelta](adadelta.md), 그리고 마침내 [Adam](adam.md)이 나오게 된 계기였다. 이들은 모두 유계가 아닌 합을 지수 감쇠 평균으로 바꾼다.

---

## 4. PyTorch 예제

```python
"""간단한 회귀 과제에서 Adagrad 최적화기 시연."""

import torch
import torch.nn as nn

# === 모델과 최적화기 준비 ===

if __name__ == "__main__":
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01, eps=1e-10)

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
Final loss: 0.6877
```

---

## 5. Adagrad를 쓸 때

Adagrad는 다음에 알맞다.

- **희소한 피처**: 어휘가 큰 자연어 처리 과제, 항목이 많은 추천 시스템, 그리고 표본마다 대부분의 피처가 꺼져 있는 모든 상황.
- **짧은 학습**: 학습을 일찍 끝내면 감쇠의 해가 적다.
- **볼록 목적 함수**: 단조 감쇠가 볼록 문제의 수렴에 도움이 됨이 증명되어 있다.

비볼록한 딥러닝 목적 함수를 오래 학습시킬 때에는 [Adam](adam.md)이나 [AdamW](adamw.md)를 쓰라.

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
Adagrad의 누적 상태와 갱신을 쓰고 모멘텀·바이어스 보정의 포함 여부를 설명하라.

</div>

??? success "연습문제 1 풀이"
    $s_t=s_{t-1}+g_t^2$, $\theta^+=\theta-\eta g_t/(\sqrt{s_t}+\epsilon)$이다. 기본 Adagrad에는 경사의 1차 모멘텀이나 Adam식 바이어스 보정이 없다. 초기 누적값·러닝레이트 감쇠 등 옵션과 epsilon 위치는 구현을 확인한다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
그래디언트가 작다는 사실과 희소하다는 사실을 구분하라.

</div>

??? success "연습문제 2 풀이"
    희소 경사는 많은 단계나 좌표에서 0이라는 뜻이다. 매번 작은 경사가 있는 경우는 크기가 작지만 희소하지 않다. Adagrad의 누적량은 빈도와 크기를 모두 반영하므로 둘을 동일하게 설명하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
$s_0=0$에서 경사 $[0.1,0.2,0.1,0.3]$을 받았을 때 상태와 4단계 배율을 구하라.

</div>

??? success "연습문제 3 풀이"
    상태는 0.01, 0.05, 0.06, 0.15이다. epsilon을 무시하면 네 번째 배율은 $\eta/\sqrt{0.15}\approx2.582\eta$, 갱신량의 절댓값은 약 $0.7746\eta$이다. 제곱근을 빠뜨려 누적 제곱합 자체로 나누면 다른 알고리즘이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
Adagrad의 분모가 커지면 학습이 반드시 멈추는가?

</div>

??? success "연습문제 4 풀이"
    $s_t$는 감소하지 않지만 $\sum_t g_t^2$가 유한하면 유한한 값에 수렴할 수 있다. 분모가 제한 없이 커지는 경우에 좌표별 배율이 0으로 간다. 이 성질만으로 모든 학습의 수렴이나 실패를 판단할 수 없다.

## 정리하며

Adagrad는 경사 제곱의 누적합 제곱근으로 좌표별 배율을 조정한다. 빈도와 크기를 모두 반영하며 기본식에는 1차 모멘텀이나 바이어스 보정이 없다. 누적량의 장기 변화와 epsilon 관례를 확인한다.

**참고 문헌**

- Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive Subgradient Methods for Online Learning and Stochastic Optimization. *Journal of Machine Learning Research*, 12, 2121--2159.
