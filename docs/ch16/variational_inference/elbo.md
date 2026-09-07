# 증거 아래 경계(ELBO)
증거 아래 경계는 변분 추론, EM 알고리즘, VAE 익힘의 한가운데 있는 양이다. 다룰 수 없는 로그 주변 가능도를 아래에서 받치는, 다룰 수 있는 최적화 목표를 주며, 참 가능도와의 틈은 KL 벌어짐으로 또렷이 특징지어진다. 이 절에서는 서로 보완하는 세 가지 이끌어 내기를 보이고, 틈과 팽팽함의 조건을 살피며, 핵심이 되는 다른 표현들을 펼치고, ELBO를 EM 알고리즘과 변분 자동부호기에 이어 준다.

## 문제: 다룰 수 없는 주변 가능도

숨은 변수 모형에서 주변 가능도(또는 *증거*)에는 숨은 변수 $z$(상황에 따라서는 매개변수 $\theta$)에 걸친 적분이 들어 있다:

$$p(\mathbf{X} | \theta) = \int p(\mathbf{X}, \mathbf{Z} | \theta) \, d\mathbf{Z}$$

신경망 풀개나 복잡한 가능도 함수에서는 이 적분을 다룰 수 없다. 곧 닫힌 꼴로 값을 매길 수도, 효율적으로 어림할 수도 없다. ELBO는 최적화할 *수 있는* 아래 경계를 준다.

## 이끌어 내기 1: 옌센 부등식

### 옌센 부등식

($\log$ 같은) 오목 함수 $\varphi$과 아무 확률 변수 $Y$에 대해:

$$\varphi(\mathbb{E}[Y]) \geq \mathbb{E}[\varphi(Y)]$$

거의 확실히 $Y$이 상수일 때 그리고 오직 그때만 등호가 성립한다.

!!! note "증명 밑그림"
    볼록한 $\varphi$에 대해 $\mu = \mathbb{E}[X]$에서 기울기가 $\alpha$인 받침 초평면이 있어 $\varphi(X) \geq \alpha(X - \mu) + \varphi(\mu)$이다. 기댓값을 취하고 $\mathbb{E}[X - \mu] = 0$임을 쓰면 $\mathbb{E}[\varphi(X)] \geq \varphi(\mathbb{E}[X])$을 얻는다. 오목한 $\varphi$에서는 부등호가 뒤집힌다.

### 옌센 부등식 쓰기

적분 안에서 곱하고 나누어 숨은 변수에 대한 아무 분포 $q(\mathbf{Z})$을 들여온다:

$$\log p(\mathbf{X} | \theta) = \log \int q(\mathbf{Z}) \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})} \, d\mathbf{Z} = \log \, \mathbb{E}_{q}\!\left[\frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})}\right]$$

$\log$이 오목하므로 옌센 부등식은 다음을 준다:

$$\log p(\mathbf{X} | \theta) \geq \mathbb{E}_{q}\!\left[\log \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})}\right] \;\equiv\; \mathcal{L}(q, \theta)$$

오른쪽 변이 **증거 아래 경계(ELBO)**이다.

### 등호가 성립할 때

기댓값 안의 확률 변수가 상수일 때 옌센 부등식이 팽팽해진다. 이는 $q$의 받침에 있는 모든 $\mathbf{Z}$에 대해 $p(\mathbf{X}, \mathbf{Z} | \theta) / q(\mathbf{Z}) = c$이어야 함을 뜻한다. 양변을 고르게 하면 $c = p(\mathbf{X} | \theta)$임이 드러나고 따라서 다음이 성립한다:

$$q(\mathbf{Z}) = \frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{p(\mathbf{X} | \theta)} = p(\mathbf{Z} | \mathbf{X}, \theta)$$

$q$이 참 뒤확률과 같을 때 그리고 오직 그때만 이 경계가 팽팽해진다.

## 이끌어 내기 2: KL 벌어짐 쪼개기

$q(\mathbf{Z})$에서 참 뒤확률 $p(\mathbf{Z} | \mathbf{X}, \theta)$까지의 KL 벌어짐을 쓴다:

$$D_{\text{KL}}\!\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr) = \mathbb{E}_{q}\!\left[\log \frac{q(\mathbf{Z})}{p(\mathbf{Z} | \mathbf{X}, \theta)}\right]$$

베이즈 정리 $p(\mathbf{Z} | \mathbf{X}, \theta) = p(\mathbf{X}, \mathbf{Z} | \theta) / p(\mathbf{X} | \theta)$을 넣으면:

$$D_{\text{KL}} = \mathbb{E}_{q}[\log q(\mathbf{Z})] - \mathbb{E}_{q}[\log p(\mathbf{X}, \mathbf{Z} | \theta)] + \log p(\mathbf{X} | \theta)$$

정리하고 ELBO를 알아보면:

$$\boxed{\log p(\mathbf{X} | \theta) = \mathcal{L}(q, \theta) + D_{\text{KL}}\!\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)}$$

$D_{\text{KL}} \geq 0$이므로 $\log p(\mathbf{X} | \theta) \geq \mathcal{L}(q, \theta)$을 되찾는다. 이 쪼갬은 틈을 정확히 짚어 주므로 옌센의 길보다 알려 주는 것이 많다.

## 이끌어 내기 3: 중요도 표집의 눈

중요도 표집의 눈으로 보면 주변 가능도는 중요도 무게 $p(\mathbf{X}, \mathbf{Z} | \theta) / q(\mathbf{Z})$의 기댓값이다:

$$p(\mathbf{X} | \theta) = \mathbb{E}_{q}\!\left[\frac{p(\mathbf{X}, \mathbf{Z} | \theta)}{q(\mathbf{Z})}\right]$$

ELBO는 옌센 부등식으로 얻은 이 기댓값의 아래 경계의 로그이다. 이 눈은 표본 여럿으로 경계를 팽팽하게 하는 **중요도 무게 자동부호기(IWAE)** 목표로 이어진다.

## 근본 항등식

세 가지 이끌어 내기가 서로 다른 각도에서 같은 결과에 이른다:

$$\log p(\mathbf{X} | \theta) = \mathcal{L}(q, \theta) + D_{\text{KL}}\!\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)$$

이 항등식에서 곧바로 셋이 따라 나온다. 첫째, $D_{\text{KL}} \geq 0$이므로 ELBO는 늘 로그 증거의 아래 경계이다. 둘째, 참 로그 가능도와 ELBO 사이의 틈은 정확히 $q$에서 참 뒤확률까지의 KL 벌어짐이다. 셋째, $q$에 대해 ELBO를 가장 크게 하는 것은 $D_{\text{KL}}(q \| p_{\text{posterior}})$을 가장 작게 하는 것과 같고, 이것이 변분 추론의 목표이다.

## 다른 표현들

ELBO는 여러 같은 뜻의 꼴로 다시 쓸 수 있으며 저마다 다른 통찰을 준다.

### 결합 꼴

$$\mathcal{L}(q, \theta) = \mathbb{E}_{q}[\log p(\mathbf{X}, \mathbf{Z} | \theta)] - \mathbb{E}_{q}[\log q(\mathbf{Z})]$$

첫 항은 기댓값 완전 자료 로그 가능도이고, 둘째 항은 $q$의 엔트로피의 음수이다.

### 되살림 + 벌주기 꼴

결합 분포를 $p(\mathbf{X}, \mathbf{Z} | \theta) = p(\mathbf{X} | \mathbf{Z}, \theta) \, p(\mathbf{Z})$으로 인수 나누면:

$$\mathcal{L}(q, \theta) = \underbrace{\mathbb{E}_{q}[\log p(\mathbf{X} | \mathbf{Z}, \theta)]}_{\text{reconstruction}} - \underbrace{D_{\text{KL}}\!\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z})\bigr)}_{\text{regularisation}}$$

되살림 항은 자료를 정확히 본뜨는 것을 북돋우고, KL 항은 앞확률에서 벗어나는 것에 벌을 준다. 이것이 VAE 익힘에 쓰는 표준 꼴이다.

### 엔트로피 꼴

$$\mathcal{L}(q, \theta) = \mathbb{E}_{q}[\log p(\mathbf{X} | \mathbf{Z}, \theta)] + \mathbb{E}_{q}[\log p(\mathbf{Z})] + H[q]$$

여기서 $H[q] = -\mathbb{E}_{q}[\log q(\mathbf{Z})]$이다. 엔트로피 항은 $q$이 넓게 퍼지도록 북돋아 한 점으로 찌부러지는 것을 막는다.

### 자유 에너지의 음수

물리 문헌에서 ELBO는 **변분 자유 에너지**의 음수이다. 곧 $\mathcal{F}(q) = -\mathcal{L}(q, \theta)$이다. 자유 에너지를 가장 작게 하는 것은 ELBO를 가장 크게 하는 것과 같다.

## 틈 살피기

### 틈이 곧 KL 벌어짐이다

$$\text{Gap} = \log p(\mathbf{X} | \theta) - \mathcal{L}(q, \theta) = D_{\text{KL}}\!\bigl(q(\mathbf{Z}) \,\|\, p(\mathbf{Z} | \mathbf{X}, \theta)\bigr)$$

$q = p(\mathbf{Z} | \mathbf{X}, \theta)$일 때 그리고 오직 그때만 틈이 0이다.

### 틈에 영향을 주는 것들

$q$ 집안의 표현력이 좋으면 틈이 작아지고, 참 뒤확률이 복잡하거나 봉우리가 여럿이면 커진다. 참 뒤확률의 봉우리가 여럿인데 $q$이 (이를테면 대각 가우스처럼) 봉우리 하나인 분포로 옭매여 있으면 0이 아닌 틈을 피할 수 없다.

### 경계 팽팽하게 하기

틈을 줄이는 방법으로는 더 풍성한 변분 집안(고르게 하는 흐름, 자기 회귀 뒤확률), 중요도 무게 주기(IWAE), 확률 층이 여럿인 층 숨은 짜임이 있다.

## EM 알고리즘과의 이음

EM 알고리즘은 $q$을 정확한 뒤확률로 두는, ELBO 최대화의 특별한 경우이다.

**E 걸음.** $q^{(t+1)}(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta^{(t)})$으로 둔다. 그러면 $D_{\text{KL}} = 0$이 되어 경계가 팽팽해진다. 곧 $\mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \log p(\mathbf{X} | \theta^{(t)})$이다.

**M 걸음.** $\theta$에 대해 $\mathcal{L}(q^{(t+1)}, \theta)$을 가장 크게 한다:

$$\theta^{(t+1)} = \arg\max_\theta \, \mathbb{E}_{q^{(t+1)}}[\log p(\mathbf{X}, \mathbf{Z} | \theta)]$$

엔트로피 $H[q^{(t+1)}]$이 $\theta$에 달려 있지 않으므로 이는 기댓값 완전 자료 로그 가능도를 가장 크게 하는 것으로 줄어든다.

**단조롭게 나아짐.** M 걸음 뒤에 $\theta$이 $\theta^{(t+1)}$으로 바뀌면서 양수인 틈이 다시 생길 수 있다. 그러나 가능도는 줄어들 수 없다:

$$\log p(\mathbf{X} | \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t+1)}) \geq \mathcal{L}(q^{(t+1)}, \theta^{(t)}) = \log p(\mathbf{X} | \theta^{(t)})$$

### 기하로 풀이하기: 접하는 아래 경계

$q$을 $\theta^{(t)}$에서의 뒤확률로 두면 ELBO는 $\theta^{(t)}$에서 로그 가능도 곡선에 닿고 다른 곳에서는 그 아래에 있다. M 걸음은 이 아래 경계의 봉우리로 옮겨 간다. 이는 **아래 받치고 최대화하기(MM)**의 한 보기이다.

### EM, 변분 추론, VAE의 견줌

| 살필 점 | EM | 변분 추론 | VAE |
|--------|----|-----------------------|-----|
| **뒤확률** | 정확한 $p(\mathbf{Z} \| \mathbf{X}, \theta)$ | 옭맨 집안 $q \in \mathcal{Q}$ | 신경망 $q_\phi(\mathbf{Z} \| \mathbf{X})$ |
| **새로 고치기** | E와 M을 번갈아 | 좌표 오르기 변분 추론 | $(\theta, \phi)$에 대한 결합 확률 기울기 내리기 |
| **추론** | 자료 점마다 | 자료 점마다 | 자료에 걸쳐 나눠 갚음 |
| **단조로움** | 예 | 예(집안 안에서) | 보장 없음 |

## VAE과의 이음

VAE에서는 부호기 $q_\phi(\mathbf{Z} | \mathbf{X})$와 풀개 $p_\theta(\mathbf{X} | \mathbf{Z})$을 되살림 + 벌주기 꼴의 ELBO를 가장 크게 하여 함께 익힌다:

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x) \| p(z))$$

되살림 항은 몬테카를로로 어림하며(흔히 표본 하나면 넉넉하다), KL 항은 $q_\phi$과 앞확률 $p(z)$이 모두 가우스이면 닫힌 꼴이 된다. 기울기는 **매개변수 바꾸기 재주**로 표집 걸음을 지나 흐른다. 곧 $\epsilon \sim \mathcal{N}(0, I)$일 때 $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$이다.

### 풀개 고르기와 되살림 손실

되살림 항 $\mathbb{E}_q[\log p_\theta(x|z)]$은 풀개의 분포에 따라 다른 손실에 맞대응된다. 곧 가우스 풀개는 평균 제곱 오차 손실($-\|x - \hat{x}\|^2 / 2\sigma^2$에 상수를 더한 것)을 주고, 베르누이 풀개는 두 값 교차 엔트로피 손실을 준다.

### 베타-VAE의 주고받음

$\beta$-VAE의 목표 $\mathcal{L} = \mathbb{E}_q[\log p_\theta(x|z)] - \beta \, D_{\text{KL}}$은 되살림의 질($\beta \to 0$)과 숨은 공간의 반듯함($\beta \to \infty$) 사이를 메운다. 표준 VAE은 $\beta = 1$에 해당한다.

(가우스의 닫힌 꼴을 비롯한) KL 벌어짐 셈하기를 자세히 다룬 것은 [KL 벌어짐](../../ch03/loss/kl_divergence.md)을 보아라. 매개변수 바꾸기 재주는 매개변수 바꾸기 쪽을 보아라. VAE 목표의 온전한 PyTorch 구현은 PyTorch 구현을 보아라.

## 가우스 모형의 ELBO(풀어 본 보기)

켤레 가우스 모형을 보자. 앞확률은 $\theta \sim \mathcal{N}(\mu_0, \sigma_0^2)$, 가능도는 $i = 1, \ldots, n$에 대해 $x_i | \theta \sim \mathcal{N}(\theta, \sigma^2)$, 변분 집안은 $q(\theta) = \mathcal{N}(m, s^2)$이다.

**기댓값 로그 가능도:**

$$\mathbb{E}_q[\log p(\mathcal{D} | \theta)] = -\frac{n}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\!\left[\sum_{i=1}^n (x_i - m)^2 + n s^2\right]$$

여기서 $\mathbb{E}_q[(x_i - \theta)^2] = (x_i - m)^2 + s^2$을 썼다.

**기댓값 로그 앞확률:**

$$\mathbb{E}_q[\log p(\theta)] = -\frac{1}{2}\log(2\pi\sigma_0^2) - \frac{1}{2\sigma_0^2}\!\left[(m - \mu_0)^2 + s^2\right]$$

**엔트로피:**

$$H[q] = \frac{1}{2}\log(2\pi e \, s^2)$$

ELBO는 이 세 항의 합이며 손으로 풀거나 $(m, s)$에 대한 기울기 오르기로 최적화할 수 있다.

## PyTorch 구현

```python
import torch
import torch.nn as nn
from typing import Tuple, Dict


class GaussianELBO:
    """흩어짐을 아는 가우스 평균 어림의 ELBO 셈하기."""

    def __init__(self, prior_mean: float, prior_std: float,
                 likelihood_std: float):
        self.mu_0 = prior_mean
        self.sigma_0 = prior_std
        self.sigma = likelihood_std

    def elbo(self, data: torch.Tensor, m: torch.Tensor,
             s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ELBO = E_q[log p(D|theta)] - KL(q || 앞확률) 셈하기.

        반환값:
            (elbo, reconstruction_term, kl_term)
        """
        n = len(data)

        # 되살림: E_q[log p(D|theta)]
        reconstruction = (
            -0.5 * n * torch.log(torch.tensor(2 * torch.pi * self.sigma**2))
            - 0.5 / self.sigma**2 * (torch.sum((data - m)**2) + n * s**2)
        )

        # KL(N(m, s^2) || N(mu_0, sigma_0^2))
        kl = (
            torch.log(torch.tensor(self.sigma_0) / s)
            + (s**2 + (m - self.mu_0)**2) / (2 * self.sigma_0**2)
            - 0.5
        )

        return reconstruction - kl, reconstruction, kl


def optimize_elbo(data: torch.Tensor, elbo_computer: GaussianELBO,
                  n_iterations: int = 500,
                  learning_rate: float = 0.05) -> Dict:
    """PyTorch의 autograd로 기울기 오르기를 해서 ELBO 최적화하기."""
    m = torch.tensor([0.0], requires_grad=True)
    log_s = torch.tensor([0.0], requires_grad=True)  # 양수로 만들려는 로그 눈금

    optimizer = torch.optim.Adam([m, log_s], lr=learning_rate)
    history = {'elbo': [], 'm': [], 's': []}

    for _ in range(n_iterations):
        optimizer.zero_grad()
        s = torch.exp(log_s)
        elbo, _, _ = elbo_computer.elbo(data, m, s)
        (-elbo).backward()  # ELBO의 음수를 가장 작게
        optimizer.step()
        history['elbo'].append(elbo.item())
        history['m'].append(m.item())
        history['s'].append(s.item())

    return history, m.detach(), torch.exp(log_s).detach()
```

## 모형 고르기 잣대로서의 ELBO

최적화한 ELBO $\mathcal{L}(q^*, \theta^*)$은 로그 모형 증거 $\log p(\mathcal{D} | \mathcal{M})$을 어림하며, 이는 모형을 견주는 표준 베이즈 양이다. BIC이나 AIC와 달리 ELBO는 점 어림값이 아니라 ($q$으로 어림한) 온전한 뒤확률을 쓰므로 모형의 복잡함에 더 풍성한 벌을 준다.

## 요약

| 이끌어 내기 | 핵심 통찰 |
|------------|-------------|
| 옌센 부등식 | $\log \mathbb{E}[Y] \geq \mathbb{E}[\log Y]$, $Y$이 상수일 때 등호 |
| KL 쪼개기 | $\log p(\mathbf{X}) = \mathcal{L} + D_{\text{KL}}(q \| p_{\text{post}})$ |
| 중요도 표집 | ELBO가 로그 중요도 무게 기댓값을 아래에서 받친다 |

| 표현 | 식 | 통찰 |
|-------------|------------|---------|
| 결합 | $\mathbb{E}_q[\log p(\mathbf{X}, \mathbf{Z})] - \mathbb{E}_q[\log q]$ | 기댓값 결합에 엔트로피를 더함 |
| 되살림 + KL | $\mathbb{E}_q[\log p(\mathbf{X} \| \mathbf{Z})] - D_{\text{KL}}(q \| p(\mathbf{Z}))$ | 자료에 맞음과 앞확률 |
| 엔트로피 | $\mathbb{E}_q[\log p(\mathbf{X} \| \mathbf{Z})] + \mathbb{E}_q[\log p(\mathbf{Z})] + H[q]$ | 드러난 엔트로피 덤 |

## 참고 문헌

1. Blei, D. M., Kucukelbir, A., & McAuliffe, J. D. (2017). "Variational Inference: A Review for Statisticians."
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*, 10장.
3. Hoffman, M. D., & Johnson, M. J. (2016). "ELBO Surgery: Yet Another Way to Carve up the Variational Evidence Lower Bound."
4. Kingma, D. P., & Welling, M. (2014). "Auto-Encoding Variational Bayes."

## 연습문제

### 연습 1: 옌센 등호의 조건

중요도 비 $p(\mathbf{X}, \mathbf{Z} | \theta) / q(\mathbf{Z})$이 상수임을 확인하여, $q(\mathbf{Z}) = p(\mathbf{Z} | \mathbf{X}, \theta)$일 때 ELBO에 대한 옌센 부등식이 등호가 됨을 보여라.

### 연습 2: 베타-이항의 ELBO

앞확률이 $\theta \sim \text{Beta}(\alpha_0, \beta_0)$, 가능도가 $x | \theta \sim \text{Binomial}(n, \theta)$, 변분 집안이 $q(\theta) = \text{Beta}(\alpha, \beta)$일 때의 ELBO를 이끌어 내어라.

### 연습 3: 수치로 본 팽팽함

풀어 본 보기의 가우스 모형에서 $q$을 정확한 뒤확률로 두면 ELBO가 로그 증거와 같아짐을 수치로 확인하여라.

### 연습 4: EM과 VAE

EM은 되풀이마다 로그 가능도가 단조롭게 나아짐을 보장하는데 VAE 익힘은 왜 그렇지 않은지 설명하여라.
