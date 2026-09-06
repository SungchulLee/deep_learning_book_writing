# 뒤 과정
**뒤 과정**은 퍼짐 모델에서 만들어 내는 알맹이이다. 앞 퍼짐을 한 걸음씩 거꾸로 돌리는 법을 배워 잡음을 짜임 있는 자료로 바꾼다.

## 띄엄띄엄한 때로 적기

뒤 과정은 $t=T$에서 $t=0$으로 거슬러 도는, 배운 마르코프 사슬이다.

$$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^{T} p_\theta(x_{t-1} | x_t)$$

여기서 $p(x_T) = \mathcal{N}(0, I)$은 잡음 사전 분포이고 뒤 옮아감마다 정규 분포로 나타낸다.

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}\bigl(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I\bigr)$$

모델은 평균 $\mu_\theta$을 배운다. 흩어짐 $\sigma_t^2$은 흔히 $\beta_t$이나 $\tilde{\beta}_t$(아래에서 뜻매김한다)으로 붙박는다.

## 참 사후 분포 q(x_(t-1) | x_t, x_0)

깨끗한 자료 $x_0$을 알면 참 뒤 옮아감은 닫힌 꼴이 된다. 이것이 배운 뒤 과정이 어림하는 목표이다.

$$\boxed{q(x_{t-1} | x_t, x_0) = \mathcal{N}\bigl(x_{t-1};\, \tilde{\mu}_t(x_t, x_0),\, \tilde{\beta}_t I\bigr)}$$

### 뒤확률의 흩어짐

$$\tilde{\beta}_t = \frac{(1 - \bar{\alpha}_{t-1})}{(1 - \bar{\alpha}_t)} \beta_t$$

### 뒤확률의 평균

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\, \beta_t}{1 - \bar{\alpha}_t}\, x_0 + \frac{\sqrt{\alpha_t}\,(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t}\, x_t$$

### 이끌어 내기: 정규 분포 둘의 곱

사후 분포는 마르코프 사슬에 베이즈 규칙을 써서 나온다.

$$q(x_{t-1} | x_t, x_0) = \frac{q(x_t | x_{t-1}) \cdot q(x_{t-1} | x_0)}{q(x_t | x_0)} \propto q(x_t | x_{t-1}) \cdot q(x_{t-1} | x_0)$$

두 갑절 모두 $x_{t-1}$에 대해 정규 분포이다.

**가능도**: $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{\alpha_t}\, x_{t-1},\, (1-\alpha_t)I)$이 정밀도 $A_1 = \alpha_t / (1-\alpha_t)$을 보탠다.

**사전 분포**: $q(x_{t-1} | x_0) = \mathcal{N}(\sqrt{\bar{\alpha}_{t-1}}\, x_0,\, (1-\bar{\alpha}_{t-1})I)$이 정밀도 $A_2 = 1 / (1-\bar{\alpha}_{t-1})$을 보탠다.

정규 분포 둘의 곱은 정밀도가 $A = A_1 + A_2$이고 평균이 $B/A$인 또 다른 정규 분포이며, $B = B_1 + B_2$은 정밀도로 무게를 준 평균을 모은다. $\bar{\alpha}_t = \alpha_t \bar{\alpha}_{t-1}$과 $\beta_t = 1 - \alpha_t$을 쓰면 대수가 위 공식으로 단순해진다.

사후 평균은 **무게를 준 평균**이다. 곧 (사전 분포로) $x_0$이 $x_{t-1}$에 대해 헤아리는 것과 (가능도로) $x_t$이 $x_{t-1}$에 대해 헤아리는 것 사이를 메우며, 무게는 서로의 정밀도로 정해진다.

## 잡음 헤아리기와 뒤 평균

실제로는 $x_0$을 모르므로 신경망의 잡음 헤아림에서 어림한다. $x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\, \epsilon$이므로 깨끗한 자료 어림은 다음과 같다.

$$\hat{x}_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}\bigl(x_t - \sqrt{1-\bar{\alpha}_t}\, \epsilon_\theta(x_t, t)\bigr)$$

사후 평균 공식에 넣고 단순하게 하면:

$$\boxed{\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\, \epsilon_\theta(x_t, t)\right)}$$

이 공식이 말하는 바는 이렇다. $x_t$에서 시작해 헤아린 잡음을 (알맞게 잣수를 맞추어) 빼고 신호가 줄어든 것을 헤아려 $1/\sqrt{\alpha_t}$으로 다시 잣수를 맞춘다. 이끌어 내기는 $\hat{x}_0$을 $\tilde{\mu}_t$에 넣고 $x_t$의 계수($1/\sqrt{\alpha_t}$으로 단순해진다)와 $\epsilon_\theta$의 계수($-(1-\alpha_t)/(\sqrt{1-\bar{\alpha}_t}\sqrt{\alpha_t})$으로 단순해진다)를 모으면 된다.

### 다른 매개변수화

| 헤아릴 목표 | 뒤 평균 공식 |
|-------------------|---------------------|
| 잡음 $\epsilon_\theta$ | $\frac{1}{\sqrt{\alpha_t}}\bigl(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta\bigr)$ |
| 깨끗한 자료 $\hat{x}_0$ | $\frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} \hat{x}_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t$ |
| 빠르기 $v_\theta$ | $\epsilon$ 매개변수화와 $x_0$ 매개변수화 사이의 사이 메우기 |

## 흩어짐 고르기

뒤 흩어짐 $\sigma_t^2$은 다음으로 둘 수 있다.

| 고르기 | 공식 | 성질 |
|--------|---------|------------|
| 사후 흩어짐 | $\tilde{\beta}_t = \frac{(1-\bar{\alpha}_{t-1})}{(1-\bar{\alpha}_t)}\beta_t$ | $x_0$을 알 때 가장 좋다 |
| 앞 흩어짐 | $\beta_t = 1 - \alpha_t$ | 위 한계이며 잡음이 더 많다 |
| 배운 것(사이 메움) | $\exp(v \log\beta_t + (1-v)\log\tilde{\beta}_t)$ | 나아진 DDPM |

붙박인 두 고르기가 가장 좋은 흩어짐을 사이에 둔다. 때 걸음마다 $v$을 배우면(Nichol와 Dhariwal, 2021) 특히 $T$이 작을 때 조금 나아진다.

## 이어진 때: 뒤 확률 미분 방정식

이어진 때에서 앞 과정은 확률 미분 방정식으로 적는다.

$$dx = f(x, t)\, dt + g(t)\, dW$$

앤더슨의 때 뒤집기 정리(1982)가 뒤 과정을 준다.

$$\boxed{dx = \bigl[f(x,t) - g(t)^2 \nabla_x \log p_t(x)\bigr] dt + g(t)\, d\bar{W}}$$

여기서 $d\bar{W}$은 거꾸로 된 때의 위너 과정이고 $\nabla_x \log p_t(x)$은 때 $t$의 **점수 함수**이다.

$f(x,t) = -\frac{1}{2}\beta(t)x$이고 $g(t) = \sqrt{\beta(t)}$인 흩어짐 지키기 적기에서:

$$dx = \left[-\frac{1}{2}\beta(t)\, x - \beta(t)\, \nabla_x \log p_t(x)\right] dt + \sqrt{\beta(t)}\, d\bar{W}$$

항 $-\beta(t) \nabla_x \log p_t(x)$은 과정을 잡음에서 자료 쪽으로 이끄는 **점수가 이끄는 떠돎**이다.

## 랑주뱅 움직임과의 이음

뒤 퍼짐 확률 미분 방정식은 배운 점수를 지닌 **때에 따라 달라지는 랑주뱅 움직임**이다.

$$dx_t = \underbrace{-\frac{1}{2}\beta(t)\, x_t}_{\text{drift toward origin}} + \underbrace{\beta(t)\, s_\theta(x_t, t)}_{\text{score-guided drift}}\, dt + \sqrt{\beta(t)}\, d\bar{W}_t$$

이는 마르코프 사슬 몬테카를로와 만들어 내는 모델의 관점을 하나로 묶는다.

| 마르코프 사슬 몬테카를로(랑주뱅) | 퍼짐 모델 |
|-----------------|-----------------|
| 아는 목표 $\pi(x)$ | 알 수 없는 $p_{\text{data}}(x)$ |
| 닫힌 꼴 점수 $\nabla \log \pi$ | 배운 점수 $s_\theta(x, t)$ |
| 멈춰 있는 분포 | 때에 따라 바뀌는 가장자리 분포 $p_t(x)$ |
| 온도 하나 | 때에 따라 바뀌는 잡음 차례표 |

여느 랑주뱅 움직임은 **때에 따라 달라지지 않는 특별한 경우**이다. 곧 점수가 $t$에 매이지 않고 목표 분포가 붙박여 있다. 잡음 수준이 $\sigma_1 > \cdots > \sigma_L$인 식힘 랑주뱅 움직임이라는 띄엄띄엄한 짝은 DDPM보다 앞서며 띄엄띄엄하게 만든 것만 빼면 같다.

## 점수와 잡음의 같음

DDPM의 잡음 헤아리개와 점수 함수는 다음으로 이어진다.

$$s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

잡음을 헤아리는 것은 점수를 어림하는 것과 같다. 이 같음이 DDPM(잡음 헤아리기)과 점수 바탕(점수 어림) 적기를 하나로 묶는다. 둘은 같은 수학의 것을 서로 다른 눈으로 그린 것이다.

## PyTorch 구현

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

## 요약

뒤 과정은 참 잡음 없애기 사후 분포 $q(x_{t-1}|x_t, x_0)$을 어림하는 법을 배워 잡음을 자료로 바꾼다. 잡음 헤아리기 매개변수화에서는 모델이 $\epsilon$을 어림해 닫힌 꼴 평균 공식에 넣는다. 이어진 때에서 뒤 과정은 떠돎이 점수 함수에 매이는 확률 미분 방정식이며, 점수 함수는 모든 퍼짐 모델 적기를 잇는 핵심 양이다.

## 연습문제

**연습문제 1.**
과녁 적분이 끝이 있는데도 중요도 표집의 흩어짐이 왜 끝없을 수 있는지 설명하여라.

??? success "연습문제 1 풀이"
    중요도 표집 어림자의 흩어짐은 $\text{Var}_q[w(x) f(x)]$에 비례하며, 여기서 $w(x) = p(x)/q(x)$은 중요도 무게이다. $q(x)$의 꼬리가 $p(x) f(x)$보다 가벼우면, $q$은 확률을 거의 주지 않는데 $p$은 주는 구역에서 비 $p(x)/q(x)$이 한없이 커질 수 있다. 그러면 이따금 어림값을 좌우하는 몹시 큰 무게가 생겨, 적분 $\mathbb{E}_p[f(X)]$이 끝이 있는데도 흩어짐이 끝없어진다(또는 사실상 끝없어진다).

---

**연습문제 2.**
중요도 무게 $w_1, \ldots, w_N$으로 나타낸 실효 표본 크기(ESS)의 공식을 이끌어 내어라.

??? success "연습문제 2 풀이"
    ESS은 무게 준 표본이 과녁 분포의 독립 표본 몇 개에 맞먹는지를 잰다:

    $$\text{ESS} = \frac{\left(\sum_{i=1}^N w_i\right)^2}{\sum_{i=1}^N w_i^2}$$

    무게가 모두 같으면($w_i = c$) ESS $= N$이다. 무게 하나가 좌우하면 ESS $\approx 1$이다. 이는 스스로 고르게 하는 중요도 표집 어림자의 흩어짐을 과녁에서 뽑은 독립 동일 분포 표본의 흩어짐에 견주어 뜯어보면 나온다.

---

**연습문제 3.**
중요도 표집으로 $\mathbb{E}_p[f(X)]$을 어림할 때 가장 좋은 제안 분포가 $q^*(x) \propto |f(x)| p(x)$임을 보여라.

??? success "연습문제 3 풀이"
    중요도 표집 어림자의 흩어짐은 $\text{Var}_q\left[\frac{f(X)p(X)}{q(X)}\right] / N$이다. 제약 $\int q(x) dx = 1$ 아래 라그랑주 곱수로 이를 $q$에 대해 가장 작게 하면 $q^*(x) = |f(x)| p(x) / \int |f(x')| p(x') dx'$이 나온다. $f \geq 0$일 때 이것이 흩어짐 0인 제안이다(어림자가 표본 하나로 정확한 답을 되돌린다). 실전에서 $q^*$은 우리가 셈하려는 바로 그 적분을 필요로 하므로 쓸 수 없다.

---

**연습문제 4.**
$X \sim \mathcal{N}(0,1)$일 때 $t$분포를 제안으로 써서 $\mathbb{E}[X^2]$의 단순한 중요도 표집 어림자를 구현하여라.

??? success "연습문제 4 풀이"
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
    $t$분포는 가우스보다 꼬리가 무거워 중요도 무게의 흩어짐이 끝이 있음을 보장한다.
