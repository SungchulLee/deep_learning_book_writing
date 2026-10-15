# 디노이징 스코어 매칭의 원리

디노이징 스코어 매칭(DSM)은 데이터에 노이즈를 더하고 계산 가능한 조건부 스코어를 회귀하는 방법이다. 미지의 데이터 밀도를 직접 평가하지 않아도 노이즈로 평활화된 주변 밀도의 스코어를 학습할 수 있다. 입력 야코비안의 trace를 계산하는 목적을 피할 수 있지만, 평활화와 유한 표본·학습 오차까지 제거하는 것은 아니다.

## 1. 노이즈 주변 분포와 조건부 목표

깨끗한 데이터 x₀에 σ>0의 등방 가우시안 노이즈를 더한다.

$$\widetilde x=x_0+\sigma\epsilon,\qquad \epsilon\sim N(0,I),\qquad
q_\sigma(\widetilde x\mid x_0)=N(\widetilde x;x_0,\sigma^2I)$$

노이즈 관측의 주변 밀도는 다음과 같다.

$$p_\sigma(\widetilde x)=\int q_\sigma(\widetilde x\mid x_0)p_{data}(x_0)dx_0$$

가우시안 평활화는 원본이 특이한 분포여도 σ>0에서 양의 매끄러운 밀도를 제공한다. 그러나 낮은 밀도 영역의 스코어가 유한 학습 데이터로 정확하게 추정된다고 보장하지는 않는다. 원본 밀도와 평활화 밀도의 차이도 남는다.

조건부 관측 스코어는 다음처럼 직접 계산할 수 있다.

$$a(\widetilde x,x_0)=\nabla_{\widetilde x}\log q_\sigma(\widetilde x\mid x_0)
=-\frac{\widetilde x-x_0}{\sigma^2}=-\frac\epsilon\sigma$$

## 2. 왜 주변 스코어를 학습하는가?

제곱오차 회귀의 목표를 a로 두자.

$$L_{DSM}(\theta)=\frac12\mathbb E_{x_0,\epsilon}
\left\|s_\theta(x_0+\sigma\epsilon)+\frac\epsilon\sigma\right\|^2$$

조건부 평균이 최적 회귀 함수이고, 적절한 미분·적분 조건에서 다음을 얻는다.

$$\mathbb E[a(\widetilde x,x_0)\mid\widetilde x]
=\nabla_{\widetilde x}\log p_\sigma(\widetilde x)=s_\sigma(\widetilde x)$$

제곱오차의 조건부 분해는 DSM과 주변 스코어 오차의 관계를 보여 준다.

$$L_{DSM}(\theta)=\frac12\mathbb E_{\widetilde x\sim p_\sigma}
\|s_\theta(\widetilde x)-s_\sigma(\widetilde x)\|^2
+\frac12\mathbb E\|a-s_\sigma\|^2$$

마지막 항은 θ와 무관하지만 일반적으로 0이 아니다. 따라서 DSM의 최소 손실이 양수여도 모순이 아니다. [Vincent의 논문](https://doi.org/10.1162/NECO_a_00142)은 디노이징과 스코어 매칭의 연결을 다룬다.

σ→0에서 평활화 밀도가 원본으로 접근한다는 사실만으로 **전체 DSM 손실값이 유한한 원본 스코어 매칭 손실로 수렴한다**고 말할 수는 없다. 위 상수는 σ에 의존하고 발산할 수 있다. 스코어의 수렴에도 매끄러움·적분 조건이 필요하며 특이한 원본 분포에는 일반적인 데이터 스코어가 없을 수도 있다.

## 3. 여러 노이즈 수준과 가중치

여러 양의 σi를 조건으로 주면 다음과 같은 목적을 구성할 수 있다.

$$L(\theta)=\sum_i\lambda_i\mathbb E\left\|s_\theta(x_0+\sigma_i\epsilon,\sigma_i)+\frac\epsilon{\sigma_i}\right\|^2$$

가중치 λi는 양수로 선택하고 수준을 뽑는 확률과 함께 해석한다. σ² 가중치는 목표 −ε/σ의 배율을 상쇄하는 다음 관계를 갖는다.

$$\sigma^2\left\|s_\theta+\frac\epsilon\sigma\right\|^2
=\|\sigma s_\theta+\epsilon\|^2$$

이를 노이즈 예측으로 바꾸려면 εθ=−σsθ로 정의한다. 이 대수적 관계가 실제 파라미터 기울기의 크기나 학습 난이도를 모든 수준에서 같게 만든다는 뜻은 아니다. 데이터 스케일·조건 입력·모델 표현력도 영향을 준다.

등비 σ 목록은 흔한 선택이지만 σmin=0.01이나 데이터의 최대 쌍별 거리를 σmax로 쓰는 규칙은 보편적인 최적값이 아니다. 이상값, 차원과 전처리에 민감하고 전체 쌍별 거리 계산도 비싸다. 학습 범위와 샘플링 초기분포·최종 노이즈를 함께 정한다.

## 4. DDPM과 연결할 때 좌표를 구분한다

DDPM 관측은 $x_t=\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\epsilon$이다. **xₜ 좌표**에서는 조건부 노이즈 표준편차가 √(1−ᾱt)이므로 다음과 같다.

$$s_{x_t}(x_t,t)=-\frac{\epsilon_\theta(x_t,t)}{\sqrt{1-\bar\alpha_t}}$$

반면 $u_t=x_t/\sqrt{\bar\alpha_t}$로 재스케일하면 $u_t=x_0+\tau_t\epsilon$이고 $\tau_t=\sqrt{(1-\bar\alpha_t)/\bar\alpha_t}$다. 이때 스코어는 좌표 변환에 따라 달라진다.

$$s_{u_t}(u_t,t)=\sqrt{\bar\alpha_t}\,s_{x_t}(\sqrt{\bar\alpha_t}u_t,t)
=-\frac{\epsilon_\theta}{\tau_t}$$

τ를 쓰면서 xₜ 좌표의 스코어 배율을 그대로 사용하는 것은 맞지 않는다. 단순 노이즈 MSE는 가중 DSM과 연결되지만 전체 변분 우도 목적과 항상 같은 수치·가중치를 갖지는 않는다. [DDPM 논문](https://arxiv.org/abs/2006.11239)의 전이와 목적 정의를 함께 확인한다.

## 5. 다른 노이즈 커널

라플라스 커널은 관측과 원본이 다른 위치에서 목표 스코어 −sign(x̃−x₀)/b를 갖는다. 일치점에서는 로그 밀도가 미분 가능하지 않아 별도 관례가 필요하다. 등방 D차원 t 커널의 스케일 행렬을 σ²I라 정의하면 목표는 다음과 같다.

$$-\frac{(\nu+D)(\widetilde x-x_0)}{\nu\sigma^2+\|\widetilde x-x_0\|^2}$$

t분포의 σ²I는 일반적으로 공분산과 다르며 ν>2일 때 공분산은 ν/(ν−2)배다. 커널을 바꾸면 회귀 목표·주변 분포와 생성 절차도 달라진다. 꼬리가 무거우면 임의 데이터에서 더 좋은 생성이나 강건성이 보장된다고 단정하지 않는다.

## 6. 기본 손실 함수의 인터페이스

```python
import torch
import torch.nn as nn

def dsm_loss(score_net: nn.Module, x: torch.Tensor,
             sigma: float) -> torch.Tensor:
    """잡음 수준 하나의 잡음 없애는 점수 맞추기 손실.

    인수:
        score_net: [batch, dim]을 [batch, dim]으로 보내는 그물 s(x).
        x: 깨끗한 자료 [batch, dim].
        sigma: 잡음의 표준 편차.

    반환값:
        낱값 손실.
    """
    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    score = score_net(x_noisy)
    target = -noise / sigma
    return 0.5 * ((score - target)**2).sum(dim=-1).mean()
```

첫 함수는 N×d 입력과 단일 인수 score_net을 가정하며 좌표 합·표본 평균에 1/2을 곱한다. 이미지에 쓰면 sum(dim=−1)은 너비만 합산하므로 이미지 전체 제곱 노름과 같지 않다. 입력 축을 명시해 집계해야 한다.

```python
def dsm_loss_energy(energy_net: nn.Module, x: torch.Tensor,
                    sigma: float) -> torch.Tensor:
    """에너지 바탕 모델의 잡음 없애는 점수 맞추기 손실.

    점수는 자동 미분으로 ∇_x E_θ(x)으로 셈한다.

    인수:
        energy_net: 낱값 에너지 신경망 E(x).
        x: 깨끗한 자료 [batch, dim].
        sigma: 잡음의 표준 편차.

    반환값:
        낱값 손실.
    """
    noise = torch.randn_like(x)
    x_noisy = (x + sigma * noise).requires_grad_(True)

    energy = energy_net(x_noisy)
    score = torch.autograd.grad(
        energy.sum(), x_noisy, create_graph=True
    )[0]

    target = -noise / sigma
    return ((score - target)**2).sum(dim=1).mean()
```

에너지 형태의 함수는 energy.sum의 입력 기울기를 **양의 부호**로 사용한다. 이는 log 비정규화 밀도 E, 즉 p∝exp(E)라는 관례다. 보통 p∝exp(−E)의 에너지 관례를 쓰면 부호가 반대다. 표본별 스칼라 출력을 전제로 하고 create_graph=True로 파라미터 미분을 유지한다. 따라서 직접 벡터 스코어 DSM과 달리 에너지에 대한 입력 미분과 혼합 미분 비용이 있다. 이 함수에는 1/2 배율도 없다.

## 7. 다중 수준과 연속 σ 구현

```python
def multi_scale_dsm_loss(
    score_net: nn.Module, x: torch.Tensor,
    sigmas: list[float], weights: list[float] | None = None
) -> torch.Tensor:
    """기본으로 σ² 무게를 준 여러 잣수의 잡음 없애는 점수 맞추기 손실.

    인수:
        score_net: 잡음 조건을 갖춘 신경망 s(x, sigma).
        x: 깨끗한 자료 [batch, dim].
        sigmas: 잡음 수준.
        weights: 수준마다 무게(기본: σ²으로 고르게 맞춘 것).

    반환값:
        잣수에 걸친 무게 있는 잡음 없애는 점수 맞추기 손실.
    """
    if weights is None:
        raw = [s**2 for s in sigmas]
        total = sum(raw)
        weights = [w / total for w in raw]

    loss = torch.tensor(0.0, device=x.device)
    for sigma, w in zip(sigmas, weights):
        noise = torch.randn_like(x)
        x_noisy = x + sigma * noise
        sigma_t = torch.full((x.shape[0], 1), sigma, device=x.device)
        score = score_net(x_noisy, sigma_t)
        target = -noise / sigma
        loss = loss + w * 0.5 * ((score - target)**2).sum(dim=-1).mean()

    return loss
```

각 수준을 모두 평가해 손실을 합산한다. 기본 가중치는 σ²를 합이 1이 되도록 정규화한다. 사용자 가중치와 σ 목록 길이가 다르면 zip이 짧은 쪽에 맞춰 잘라 버리므로 길이·양수·유한성 검사가 필요하다. 아래 ScoreNet처럼 두 인수를 받는 조건 모델과 연결한다. 단일 인수인 첫 함수에 이 모델을 바로 전달하면 호출 계약이 맞지 않는다.

```python
def dsm_loss_random_sigma(
    score_net: nn.Module, x: torch.Tensor,
    sigma_min: float = 0.01, sigma_max: float = 10.0
) -> torch.Tensor:
    """표본마다 아무 시그마(로그 고른 분포)를 쓴 잡음 없애는 점수 맞추기 손실.

    모든 시그마 수준을 되풀이하는 것보다 효율이 좋다.
    """
    log_sigma = (
        torch.rand(x.shape[0], 1, device=x.device)
        * (torch.log(torch.tensor(sigma_max)) - torch.log(torch.tensor(sigma_min)))
        + torch.log(torch.tensor(sigma_min))
    )
    sigma = torch.exp(log_sigma)

    noise = torch.randn_like(x)
    x_noisy = x + sigma * noise
    score = score_net(x_noisy, sigma)
    target = -noise / sigma

    # σ² 무게를 준 손실
    return 0.5 * (sigma**2 * (score - target)**2).sum(dim=-1).mean()
```

이 함수는 log σ를 균등하게 뽑으므로 σ 자체의 균등 선택이 아니다. σ는 N×1이고 좌표별 오차도 N×d여서 여기서는 가중치가 **좌표 합산 전** 올바르게 브로드캐스팅된다. 입력은 표 형태 N×d를 가정하며 σmin>0, σmax≥σmin이 필요하다. 이미지에는 별도의 축 확장이 필요하다.

```python
class ScoreNet(nn.Module):
    """잡음 수준을 조건으로 삼는 여러 층 신경망 점수 신경망."""

    def __init__(self, data_dim: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """인자: x [batch, dim], sigma [batch, 1]."""
        inp = torch.cat([x, torch.log(sigma) / 4.0], dim=-1)
        return self.net(inp)
```

조건은 log(σ)/4이며 σ는 N×1이어야 한다. 단일 스칼라를 자동 복제하지 않고 x와 조건의 dtype·device도 호출자가 맞춰야 한다.

```python
import numpy as np

def train_dsm(data: torch.Tensor, n_epochs: int = 5000,
              batch_size: int = 256, lr: float = 1e-3):
    """아무 시그마 잡음 없애는 점수 맞추기로 점수 신경망을 익힌다."""
    model = ScoreNet(data_dim=data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epochs):
        idx = torch.randperm(len(data))[:batch_size]
        batch = data[idx]

        loss = dsm_loss_random_sigma(model, batch,
                                     sigma_min=0.01, sigma_max=5.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}: loss = {loss.item():.4f}")

    return model
```

학습 함수는 각 반복에 무작위 순열의 앞 batch_size개를 사용해 **미니배치 하나만** 갱신한다. 따라서 n_epochs는 전체 데이터 순회 수가 아니라 갱신 횟수다. 모델은 CPU에 생성하며 외부 데이터를 자동 이동시키지 않는다. 검증·체크포인트·샘플링 실험은 포함하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
DSM의 회귀 최적점이 개별 조건부 목표와 어떻게 다른지 설명하라.

</div>

??? success "연습문제 1 풀이"
    개별 목표는 −ε/σ이고 최적 출력은 관측 x̃가 주어졌을 때의 목표 조건부 평균이다. 이것이 노이즈 주변 밀도의 스코어다. 같은 관측에 가능한 원본이 여럿이면 목표의 조건부 분산이 남아 최소 손실도 양수일 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
σ→0에서 DSM 손실값 전체가 원본 스코어 매칭 손실로 수렴한다고 말할 수 있는가?

</div>

??? success "연습문제 2 풀이"
    일반적으로 없다. DSM과 주변 스코어 오차의 차이는 θ에 무관하지만 σ에 의존하는 조건부 분산 항이다. 매끄러운 원본에서도 목표의 제곱 노름은 D/σ² 규모여서 그 상수가 발산할 수 있다. 원본이 특이하면 원래 스코어의 존재부터 검토해야 한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
연속 σ 손실에서 N×1 가중치와 N×d 오차를 곱한 뒤 좌표를 합하면 어떤 값을 얻는가?

</div>

??? success "연습문제 3 풀이"
    같은 표본의 가중치가 그 표본의 d개 좌표에 곱해진 뒤 좌표를 합산하므로 표본별 가중 제곱 노름이 된다. 좌표를 먼저 합해 길이 N으로 만든 뒤 N×1을 곱하면 N×N 외적이 되어 다른 계산이다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
DDPM의 xₜ와 재스케일한 uₜ=xₜ/√ᾱₜ에서 노이즈 크기와 스코어 배율을 비교하라.

</div>

??? success "연습문제 4 풀이"
    xₜ의 조건부 노이즈 표준편차는 √(1−ᾱₜ)이고 스코어는 −εθ/√(1−ᾱₜ)다. uₜ에서는 τₜ=√((1−ᾱₜ)/ᾱₜ)이고 스코어는 −εθ/τₜ, 즉 xₜ 스코어의 √ᾱₜ배다. 동일한 노이즈 예측이라도 미분 좌표에 따라 스코어의 배율이 달라진다.

## 정리하며

DSM은 조건부 노이즈 스코어의 회귀로 평활화된 주변 스코어를 학습한다. θ에 무관한 상수의 σ 의존성, 손실 집계 축과 에너지 부호를 명시한다. DDPM과 연결할 때는 노이즈 배율뿐 아니라 좌표 재스케일과 시간별 가중치를 구분한다.

**참고 문헌**

1. Vincent, P. (2011). "A Connection Between Score Matching and Denoising Autoencoders." *Neural Computation*.
2. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
3. Song, Y., & Ermon, S. (2020). "Improved Techniques for Training Score-Based Generative Models." *NeurIPS*.
4. Ho, J., et al. (2020). "Denoising Diffusion Probabilistic Models." *NeurIPS*.
