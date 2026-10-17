# 확산 모델의 역방향 과정

확산 모델은 노이즈 관측에서 이전 상태를 생성하는 역전이를 학습한다. 원본 데이터를 알고 계산할 수 있는 정방향 사후분포를 학습 목표로 활용하지만, 생성할 때는 원본을 모르므로 네트워크의 예측으로 역전이를 구성한다. 원본에 조건을 둔 사후분포와 실제 주변 역전이를 구분하는 것이 핵심이다.

## 1. 생성용 마르코프 사슬

원본을 x₀, 노이즈 단계를 1~T로 표기하자. 생성 모델의 결합분포는 다음과 같다.

$$p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^Tp_\theta(x_{t-1}\mid x_t)$$

종단 사전분포는 흔히 N(0,I)로 두고 중간 역전이는 학습한 평균과 고정·학습 분산의 가우시안으로 근사한다. 실제 정방향 종단 주변 분포가 이 사전과 충분히 가까워지는 노이즈 스케줄을 선택해야 한다. 데이터가 이산 이미지라면 마지막 관측 우도·양자화 처리도 따로 정의한다.

## 2. 원본에 조건을 둔 사후분포

정방향 전이가 $q(x_t\mid x_{t-1})=N(\sqrt{\alpha_t}x_{t-1},\beta_tI)$이고 αt=1−βt, ᾱt=∏s≤tαs라고 하자. 원본 x₀를 알면 다음 사후분포를 계산할 수 있다.

$$q(x_{t-1}\mid x_t,x_0)=N(\widetilde\mu_t(x_t,x_0),\widetilde\beta_tI)$$

$$\widetilde\beta_t=\frac{1-\bar\alpha_{t-1}}{1-\bar\alpha_t}\beta_t$$

$$\widetilde\mu_t=
\frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0+
\frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$

이는 **x₀까지 주어진 조건부 사후분포**다. 원본을 모르는 실제 역전이 q(xₜ₋₁|xₜ)는 가능한 x₀에 대한 혼합이며 일반적으로 같은 가우시안이 아니다. 가우시안 역모델은 이를 근사하는 선택이다. t=1에서는 x₀를 조건으로 이미 알고 있어 사후가 x₀에 집중하고 분산은 0이다. 이 경계는 비퇴화 가우시안의 일반식을 그대로 나누기보다 별도로 해석한다.

## 3. 가우시안 곱으로 유도하기

베이즈 규칙과 정방향 마르코프성을 사용한다.

$$q(x_{t-1}\mid x_t,x_0)\propto
q(x_t\mid x_{t-1})q(x_{t-1}\mid x_0)$$

t>1에서 xₜ₋₁에 대한 두 이차형의 정밀도는 αt/βt와 1/(1−ᾱt−1)다. 선형항의 계수는 각각 √αt·xₜ/βt와 √ᾱt−1·x₀/(1−ᾱt−1)다. 두 정밀도를 더한 역수가 사후분산이고, 선형항의 합에 그 분산을 곱하면 사후평균이다.

이 식을 ᾱt=αtᾱt−1로 정리하면 위 공식을 얻는다. 가중하는 관측은 x₀와 xₜ 자체가 아니라 각각 이전 상태에 대한 스케일된 정보다. 따라서 표시된 x₀·xₜ의 두 계수 합이 반드시 1인 볼록평균이라고 해석하지 않는다.

## 4. 노이즈 예측과 역평균

$x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$에서 노이즈를 예측하면 다음 원본 추정값을 구성할 수 있다.

$$\widehat x_0=\frac{x_t-\sqrt{1-\bar\alpha_t}\epsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}$$

이를 사후평균에 대입하면 다음과 같다.

$$\mu_\theta(x_t,t)=\frac1{\sqrt{\alpha_t}}
\left(x_t-\frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t,t)\right)$$

원본 예측을 clipping하면 더는 같은 단순 대입식과 동일하지 않을 수 있다. 학습에서 선택한 예측·로스·분산 관례를 생성에도 일치시킨다.

v 예측이라는 다른 파라미터화에서는 $a_t=\sqrt{\bar\alpha_t}$, $b_t=\sqrt{1-\bar\alpha_t}$로 두고 $v=a_t\epsilon-b_tx_0$를 정의한다. 그러면 $\widehat x_0=a_tx_t-b_tv_\theta$, $\widehat\epsilon=b_tx_t+a_tv_\theta$로 변환한다. 이는 두 예측을 임의로 보간하는 것이 아니라 정해진 선형 좌표 변환이다.

## 5. 역분산 선택의 범위

고정 분산으로 β̃t 또는 βt를 선택하거나 두 로그 분산 사이를 학습하는 방법이 있다. β̃t≤βt라는 관계는 정방향 조건부 사후와 노이즈 스케줄의 관계다. 임의 데이터에서 원본을 모르는 실제 역전이의 최적 분산이 항상 이 둘 사이에 있다는 보편적 보장은 아니다.

[Improved DDPM](https://arxiv.org/abs/2102.09672)의 학습 분산은 특정 모델·목적·평가에서 다루는 선택이다. 시간별 값이나 입력 의존 출력을 사용할 수 있으며, 로그 보간의 계수가 0~1로 제한되는지도 구현에 따라 확인한다. 마지막 β̃₁=0의 로그는 별도 처리해야 한다.

## 6. 연속시간 역 SDE와 랑주뱅의 차이

상태에 의존하지 않는 g(t)에 대해 정방향 SDE가 $dX_t=f(X_t,t)dt+g(t)dW_t$이면 역시간 식은 다음과 같다.

$$dX_t=[f(X_t,t)-g(t)^2\nabla_x\log p_t(X_t)]dt+g(t)d\overline W_t$$

여기서는 t가 감소한다. 양의 h로 t→t−h를 계산할 때는 표시된 드리프트에 −h를 곱해야 한다. VP의 f=−βx/2, g²=β라면 감소 시간 τ=T−t에서의 드리프트는 **+βx/2+βs**다. 원래 시간과 감소 시간을 섞어 원점 수축 항과 스코어 항의 부호를 임의로 결합하지 않는다.

고정 분포의 랑주뱅은 평형분포를 향해 이동하고, 역확산은 시간에 따라 변하는 주변 분포를 거슬러 이동한다. 어닐링된 랑주뱅, DDPM 역전이와 역 SDE는 관련되지만 이산화만 제외하면 모두 같은 알고리즘이라고 할 수 없다. [스코어 SDE 논문](https://arxiv.org/abs/2011.13456)의 과정과 시간 관례를 참고한다.

노이즈 예측과 스코어의 $s_\theta=-\epsilon_\theta/\sqrt{1-\bar\alpha_t}$ 관계도 가우시안 관측과 해당 좌표에서 성립하는 변환이다. 모든 종류의 확산·이산 상태 모델에 조건 없이 같은 식을 적용하지 않는다.

## 7. 구현의 인덱스와 반환 모양

```python
import torch

def compute_posterior_params(
    x_0: torch.Tensor,
    x_t: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
    betas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute parameters of q(x_{t-1} | x_t, x_0).

    반환값:
        (posterior_mean, posterior_variance)
    """
    alpha_bar_t = alpha_bars[t]
    alpha_bar_prev = torch.where(t > 0, alpha_bars[t - 1], torch.ones_like(alpha_bar_t))
    beta_t = betas[t]
    alpha_t = 1.0 - beta_t

    # 사후 흩어짐: β̃_t = β_t (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
    posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)

    # 사후 평균 계수
    coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1 - alpha_bar_t)
    coef_xt = torch.sqrt(alpha_t) * (1 - alpha_bar_prev) / (1 - alpha_bar_t)

    # 퍼뜨리기에 맞게 꼴을 바꾼다
    for _ in range(len(x_0.shape) - 1):
        coef_x0 = coef_x0.unsqueeze(-1)
        coef_xt = coef_xt.unsqueeze(-1)

    posterior_mean = coef_x0 * x_0 + coef_xt * x_t
    return posterior_mean, posterior_var

def reverse_mean_from_noise(
    x_t: torch.Tensor,
    t: torch.Tensor,
    eps_pred: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> torch.Tensor:
    """잡음 헤아림에서 뒤 평균을 셈한다.

    μ_θ(x_t, t) = (1/√α_t)(x_t - (1-α_t)/√(1-ᾱ_t) · ε_θ)
    """
    alpha_t = alphas[t]
    alpha_bar_t = alpha_bars[t]

    for _ in range(len(x_t.shape) - 1):
        alpha_t = alpha_t.unsqueeze(-1)
        alpha_bar_t = alpha_bar_t.unsqueeze(-1)

    return (1.0 / torch.sqrt(alpha_t)) * (
        x_t - (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) * eps_pred
    )
```

코드는 이론의 1~T 대신 **0~T−1 인덱스**를 쓴다. t=0에서는 이전 누적곱을 1로 놓아 사후평균 x₀, 사후분산 0을 만든다. `torch.where`는 인수 표현식을 모두 계산하므로 alpha_bars[t−1]의 −1 인덱스도 먼저 조회되지만, 비어 있지 않은 정상 스케줄에서는 마지막 원소를 조회한 뒤 선택에서 제외된다.

길이 B의 t와 B×d 또는 이미지 입력을 전제로 계수를 뒤쪽 축에 확장한다. 사후평균은 입력과 같은 모양이지만 반환 posterior_var는 **길이 B**다. 이미지에 곱하거나 잡음을 만들 때는 그 분산도 B×1×1×1 등으로 확장해야 한다. 두 함수는 같은 β·α·누적곱, 유효 인덱스와 일치하는 device·dtype을 전제로 하며 자동 검증·이동하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
q(xₜ₋₁|xₜ,x₀)와 q(xₜ₋₁|xₜ)를 구분하라.

</div>

??? success "풀이"

    전자는 원본까지 알고 있어 가우시안 전이의 곱으로 닫힌 식을 얻는다. 후자는 가능한 원본에 대한 사후 혼합이므로 일반적으로 가우시안이 아니다. 생성 모델은 원본 없이 네트워크로 그 역전이를 근사하며 이 차이를 학습·분산 선택에서 고려한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
코드의 t=0에서 사후평균과 사후분산은 무엇이며 이론의 어느 단계인가?

</div>

??? success "풀이"

    코드 인덱스 0은 이론의 첫 노이즈 단계 t=1에 대응한다. 이전 누적곱을 1로 두면 x₀의 계수는 1, xₜ 계수는 0이고 분산도 0이다. 원본 x₀를 조건으로 이미 알고 있으므로 퇴화 사후다. 생성에서 원본을 실제로 알고 있다는 뜻은 아니다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
B×C×H×W 입력에서 두 반환값의 모양과 역시간 SDE의 시간 부호를 설명하라.

</div>

??? success "풀이"

    posterior_mean은 B×C×H×W이고 posterior_var는 B다. 분산을 이미지별로 곱할 때는 뒤쪽 축을 확장해야 한다. 별도로 역 SDE의 f−g²s는 감소하는 t에 대해 적힌 식이므로 양의 시간 폭으로 역진할 때는 드리프트 부호를 반전한다.



---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
아래 보존된 중요도 샘플링 코드는 역확산을 구현하는가?

</div>

??? success "풀이"

    아니다. 정규분포의 E[X²]=1을 t분포 제안으로 추정하는 독립적인 통계 예제다. 역전이·노이즈 예측과 연결된 코드는 아니므로 원래 주제와 구분한다. 이 특정 정규 목표·다항 함수·양의 자유도 t 제안에서는 필요한 제곱 적분이 유한하지만 무거운 꼬리만으로 모든 중요도 추정의 분산을 보장할 수는 없다.

    ```python
    import numpy as np
    from scipy import stats

    def importance_sampling_x_squared(n_samples=10000, df=5):
        target = stats.norm(0, 1)
        proposal = stats.t(df=df)
        x = proposal.rvs(n_samples)
        weights = target.pdf(x) / proposal.pdf(x)
        f_x = x ** 2
        estimate = np.mean(weights * f_x)
        return estimate  # 1.0에 가까워야 함

    print(f"Estimate: {importance_sampling_x_squared():.4f}")
    print(f"True value: 1.0000")
    ```


## 정리하며

역확산에서는 원본에 조건을 둔 가우시안 사후와 생성용 주변 역전이를 구분한다. 노이즈 예측에서 역평균을 얻되 분산·인덱스·입력 축의 관례를 맞춘다. 연속시간으로 연결할 때도 감소 시간과 스코어 드리프트 부호를 일치시켜야 한다.
