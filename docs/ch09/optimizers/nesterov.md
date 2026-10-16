# 네스테로프 가속 경사

[고전적인 모멘텀](momentum.md)은 지난 그래디언트로 속도 벡터를 쌓아 파라메터의 갱신을 가속한다. 곡률이 낮은 방향에서는 수렴이 빨라지지만, 로스 곡면이 급하게 굽는 곳, 예를 들어 그래디언트의 방향이 빠르게 바뀌는 극소점 근처에서는 쌓인 속도가 지나쳐 버릴 수 있다. **네스테로프 가속 경사**(NAG, Nesterov, 1983)는 현재 위치가 아니라 **미리 본 위치**에서 그래디언트를 재어 이를 누그러뜨린다. 이 "미리 엿보기"가 곡률의 변화를 미리 알려 주어 옵티마이저가 지나치기 전에 속도를 늦추게 한다.

---

## 1. 갱신 규칙

각 시각 $t$에서 현재 파라메터를 $\theta_t$, 모멘텀 계수를 $\mu \in [0, 1)$, 러닝레이트를 $\eta > 0$이라 하자. 속도 벡터 $v_t$은 0으로 초기화한다.

**1단계.** 미리 볼 위치를 계산하고 거기서 그래디언트를 잰다.

$$
\tilde{\theta}_t = \theta_t + \mu \, v_{t-1}
$$

**2단계.** 미리 본 위치의 그래디언트로 속도를 갱신한다.

$$
v_t = \mu \, v_{t-1} - \eta \, \nabla_\theta \mathcal{L}(\tilde{\theta}_t)
$$

**3단계.** 파라메터를 갱신한다.

$$
\theta_{t+1} = \theta_t + v_t
$$

고전적인 모멘텀과의 결정적인 차이는 1단계에 있다. 그래디언트를 $\theta_t$ 자체가 아니라 (모멘텀이 데려갈 곳인) $\tilde{\theta}_t = \theta_t + \mu \, v_{t-1}$에서 잰다.

!!! note "동등한 다시 쓰기"
    PyTorch를 비롯한 여러 프레임워크는 미리 볼 위치를 겉으로 계산하지 않는 대수적으로 동등한 형태를 구현한다. $\hat{\theta}_t = \theta_t + \mu \, v_t$으로 두고 대입하면 갱신이 $\theta_t$에서 바로 적용되는 변형된 속도 스텝으로 줄어듦을 보일 수 있다. 수학적인 효과는 같다.

---

## 2. 직관: 바로잡아 주는 미리 보기

관성을 지니고 비탈을 굴러 내려가는 공을 생각해 보자. 고전적인 모멘텀은 공의 현재 위치에서 그래디언트를 재어 쌓인 속도에 더한다. 공이 골짜기 바닥에 다가가고 있어도 현재 그래디언트는 여전히 내리막을 가리키므로 공은 더 빨라져 지나쳐 버린다.

네스테로프 모멘텀은 먼저 공을 모멘텀이 데려갈 곳으로 옮긴 뒤 그 미래 위치에서 그래디언트를 잰다. 그 위치가 이미 골짜기 바닥을 지났다면 거기서의 그래디언트는 *오르막*을 가리키므로 속도가 줄어 지나침을 막는다.

??? example "1차원 이차 함수에서의 비교"
    $\eta = 0.1$, $\mu = 0.9$으로 $\theta_0 = 10$에서 시작하여 $f(\theta) = \frac{1}{2}\theta^2$을 최소화한다고 하자.

    **고전적인 모멘텀**은 현재 위치에서 $\nabla f(\theta_t) = \theta_t$을 재므로 파라메터가 0에 다가가는 동안에도 속도에 강한 내리막 신호가 쌓여 진동이 생긴다.

    **네스테로프 모멘텀**은 미리 본 위치에서 $\nabla f(\theta_t + \mu v_{t-1})$을 잰다. 속도가 $\theta$을 0 너머로 데려가려 할 때 미리 본 그래디언트가 반대 방향을 가리켜 진동을 더 일찍 누그러뜨린다.

---

## 3. 수렴 성질

매끄러운 볼록 함수에서는 적절한 러닝레이트와 시간별 모멘텀을 쓰는 가속 방법의 함수값 오차가 $O(1/t^2)$로 줄어드는 결과가 있다. 같은 조건의 기본 경사 하강법은 $O(1/t)$ 상계를 갖는다. 여기의 속도는 파라메터 오차 자체나 고정 모멘텀 설정의 보편적 속도가 아니다.

비볼록 뉴럴 넷트웍에서는 그 결과를 바로 적용할 수 없다. 네스테로프 갱신도 러닝레이트와 모멘텀에 따라 진동하거나 발산할 수 있으므로 고전 모멘텀과 같은 검증 예산으로 비교한다.

## 4. PyTorch 예제

```python
"""PyTorch의 SGD로 보는 네스테로프 가속 경사 시연."""

import torch
import torch.nn as nn

# === 모델과 최적화기 준비 ===

if __name__ == "__main__":
    model = nn.Linear(10, 1)

    # nesterov=True로 두어 네스테로프 모멘텀 켜기
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9,
        nesterov=True,
    )

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
Final loss: 0.8982
```

---

## 5. 언제 쓰는가

모멘텀 SGD의 후보로 네스테로프를 비교할 수 있다. PyTorch의 재표현은 추가적인 전체 순전파 없이 계산하지만 버퍼 연산과 제약은 남는다. 같은 초기 계수의 매 단계 궤적이 미리 보기 표기와 단순히 같다고 가정하지 말고 변수 변환·초기 상태를 맞춘다. PyTorch에서는 양의 모멘텀과 dampening=0 등 해당 옵션 조건을 확인한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
네스테로프의 미리 보기 갱신에 적응형 2차 적률이나 바이어스 보정이 포함되는가?

</div>

??? success "연습문제 1 풀이"
    본문의 갱신은 $\tilde\theta_t=\theta_t+\mu v_{t-1}$, $v_t=\mu v_{t-1}-\eta\nabla L(\tilde\theta_t)$, $\theta_{t+1}=\theta_t+v_t$이다. 이 기본식에는 Adam식 2차 적률 상태나 바이어스 보정이 없다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
네스테로프가 지나침을 줄일 수 있는 직관과 한계를 설명하라.

</div>

??? success "연습문제 2 풀이"
    속도가 데려갈 위치의 경사를 이용하면 현재 경사만 보는 경우와 다른 보정 방향을 얻을 수 있다. 그러나 모든 로스에서 진동을 줄이거나 더 빨리 수렴하는 것은 아니며, 러닝레이트와 모멘텀의 안정 조건이 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
그래디언트 값 네 개만으로 네스테로프의 실효 러닝레이트를 유일하게 정할 수 있는가?

</div>

??? success "연습문제 3 풀이"
    러닝레이트, 모멘텀과 초기 속도가 없으면 갱신을 정할 수 없다. 또한 그래디언트가 현재점인지 미리 본 점의 값인지 명시해야 한다. 상태가 속도 이력이므로 Adam처럼 제곱 경사 분모에 나눠 구하는 방법은 적용되지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
$f(\theta)=a\theta^2/2$, $a>0$에서 고정 $0\leq\mu<1$인 본문의 미리 보기 방식의 안정 조건을 유도하라.

</div>

??? success "연습문제 4 풀이"
    $q=1-\eta a$라 두면 $\theta_{t+1}=q[(1+\mu)\theta_t-\mu\theta_{t-1}]$이고 특성식은 $r^2-q(1+\mu)r+q\mu=0$이다. 두 근이 단위원 안에 있으려면 $0<\eta a<2(1+\mu)/(1+2\mu)$가 필요하다. 이는 이 이차 함수와 고정 모멘텀 방식의 조건이며 모든 최적화 알고리즘에 $\eta<2/L$를 그대로 적용할 수는 없다.

## 정리하며

네스테로프는 모멘텀의 미리 보기 관점으로 갱신한다. 적응형 2차 적률 방법과 구분하고, 볼록 가속 이론의 스케줄 조건과 실제 PyTorch 고정 옵션을 나눠 이해한다.

**참고 문헌**

- Nesterov, Y. (1983). A Method of Solving a Convex Programming Problem with Convergence Rate $O(1/k^2)$. *Soviet Mathematics Doklady*, 27(2), 372--376.
- Sutskever, I., Martens, J., Dahl, G., & Hinton, G. (2013). On the Importance of Initialization and Momentum in Deep Learning. *ICML 2013*.
