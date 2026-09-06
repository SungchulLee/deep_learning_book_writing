# 점수 확률 미분 방정식

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 점수 확률 미분 방정식을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""
FILE: 08_score_sde.py
어려움: 나아간 단계
걸리는 시간: 4~5시간
미리 알 것: 07_noise_conditional_scores.py, 기본 확률 미분 방정식, 미적분

학습 목표:
    1. 점수 바탕 확률 미분 방정식 틀을 이해한다
    2. 흩어짐 터짐(VE) 확률 미분 방정식을 짠다
    3. 흩어짐 지키기(VP) 확률 미분 방정식을 짠다
    4. 확률 흐름 상미분 방정식을 이해한다
    5. 거꾸로 된 때의 확률 미분 방정식 뽑기를 짠다

MATHEMATICAL BACKGROUND:
    점수 바탕 확률 미분 방정식은 퍼짐을 이어진 때로 적어 준다.
    
    FORWARD SDE:
    dx = f(x, t)dt + g(t)dw
    
    여기서 w은 브라운 움직임이다.
    
    REVERSE SDE:
    dx = [f(x, t) - g(t)²∇log p_t(x)]dt + g(t)dw̄
    
    우리가 배우는 것은 점수 ∇log p_t(x)이다!
    
    VARIANCE EXPLODING (VE):
    f(x, t) = 0
    g(t) = σ(t)√(dσ(t)²/dt)
    
    VARIANCE PRESERVING (VP):
    f(x, t) = -β(t)x/2
    g(t) = √β(t)
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================


class ScoreSDE:
    """점수 바탕 확률 미분 방정식의 바탕 갈래."""
    
    def __init__(self, beta_min=0.1, beta_max=20.0, T=1.0):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T
    
    def f(self, x, t):
        """떠돎 계수."""
        raise NotImplementedError
    
    def g(self, t):
        """퍼짐 계수."""
        raise NotImplementedError
    
    def marginal_prob(self, x0, t):
        """
        Compute mean and std of p_t(x|x₀).
        
        반환값:
            mean: E[x_t | x₀]
            std: √Var[x_t | x₀]
        """
        raise NotImplementedError


class VESDE(ScoreSDE):
    """
    흩어짐 터짐 확률 미분 방정식.
    
    dx = σ(t)√(dσ(t)²/dt) dw
    
    Marginal: p_t(x|x₀) = N(x|x₀, σ(t)²I)
    """
    
    def __init__(self, sigma_min=0.01, sigma_max=50.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def f(self, x, t):
        return torch.zeros_like(x)
    
    def g(self, t):
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        return sigma_t * np.sqrt(2 * np.log(self.sigma_max / self.sigma_min))
    
    def marginal_prob(self, x0, t):
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x0
        std = sigma_t
        return mean, std


class VPSDE(ScoreSDE):
    """
    흩어짐 지키기 확률 미분 방정식.
    
    dx = -β(t)x/2 dt + √β(t) dw
    
    Marginal: p_t(x|x₀) = N(x | α_t x₀, (1-α_t²)I)
    """
    
    def __init__(self, beta_min=0.1, beta_max=20.0):
        super().__init__(beta_min, beta_max)
    
    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min)
    
    def f(self, x, t):
        return -0.5 * self.beta(t) * x
    
    def g(self, t):
        return torch.sqrt(self.beta(t))
    
    def marginal_prob(self, x0, t):
        log_alpha_t = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
        alpha_t = torch.exp(log_alpha_t)
        mean = alpha_t * x0
        std = torch.sqrt(1 - alpha_t ** 2)
        return mean, std


def sde_loss(model, sde, x0):
    """
    확률 미분 방정식의 점수 맞추기 손실을 셈한다.
    
    L = E_t E_x₀ E_x_t[||s_θ(x_t, t) - ∇log p_t(x_t|x₀)||²]
    """
    # 아무 때를 뽑는다
    batch_size = len(x0)
    t = torch.rand(batch_size, device=x0.device)
    
    # 가장자리 분포에서 뽑는다
    mean, std = sde.marginal_prob(x0, t.view(-1, 1))
    z = torch.randn_like(x0)
    x_t = mean + std * z
    
    # 점수 미리보기
    score_pred = model(x_t, t)
    
    # 참 점수: ∇log p_t(x_t|x₀) = -(x_t - mean)/std²
    score_true = -z / std
    
    # 표준 편차 제곱으로 무게를 준 평균 제곱 어긋남 손실
    loss = torch.mean(std ** 2 * torch.sum((score_pred - score_true) ** 2, dim=1))
    
    return loss


def reverse_sde_sampling(model, sde, shape, n_steps=1000):
    """
    거꾸로 된 때의 확률 미분 방정식으로 뽑는다.
    
    Implements: dx = [f - g²∇log p]dt + g dw̄
    """
    x = torch.randn(shape)
    dt = 1.0 / n_steps
    
    with torch.no_grad():
        for i in range(n_steps):
            t = 1.0 - i * dt
            t_tensor = torch.full((shape[0],), t)
            
            # 계수를 셈한다
            f = sde.f(x, t)
            g = sde.g(t)
            
            # 점수 셈하기
            score = model(x, t_tensor)
            
            # 뒤 확률 미분 방정식 걸음
            drift = f - (g ** 2) * score
            diffusion = g
            
            x = x + drift * dt + diffusion * np.sqrt(dt) * torch.randn_like(x)
    
    return x


def demo_sde():
    """2차원 정규 분포에서 점수 확률 미분 방정식을 보여 준다."""
    print("Score-based SDE Demo")
    print("=" * 80)
    
    # 단순한 2차원 정규 분포 자료
    data = torch.randn(1000, 2)
    
    # 확률 미분 방정식을 만든다
    sde = VPSDE(beta_min=0.1, beta_max=20.0)
    
    # 단순한 점수 모델(때 조건)
    class SimpleScoreModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3, 128),  # x(2차원) + t(1차원)
                nn.SiLU(),
                nn.Linear(128, 128),
                nn.SiLU(),
                nn.Linear(128, 2)
            )
        
        def forward(self, x, t):
            t = t.view(-1, 1)
            inp = torch.cat([x, t], dim=-1)
            return self.net(inp)
    
    model = SimpleScoreModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nTraining score model...")
    for epoch in range(3000):
        optimizer.zero_grad()
        loss = sde_loss(model, sde, data)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")
    
    # 뽑기
    print("\nGenerating samples via reverse SDE...")
    samples = reverse_sde_sampling(model, sde, (500, 2), n_steps=1000)
    
    # 시각화한다
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=1, alpha=0.5)
    axes[0].set_title('Training Data')
    axes[0].set_aspect('equal')
    
    axes[1].scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=1, alpha=0.5)
    axes[1].set_title('Generated Samples (Reverse SDE)')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('/home/claude/demo_score_sde.png', dpi=150, bbox_inches='tight')
    print("\nSaved demo_score_sde.png")
    print("\n✓ Score SDE successfully implemented!")


if __name__ == "__main__":
    demo_sde()```

## 논의

점수 확률 미분 방정식의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

이 짜기의 핵심에는 수치의 안정을 꼼꼼히 다루기, 고르게 맞추기 재주를 제대로 쓰기, 효율 좋은 셈 결이 든다. 익히기 절차에는 잡음 차례표, 기울기 다루기, 이따금의 따지기가 들며 모두 품질 높은 결과를 내는 데 결정적이다.

이 단원은 이론의 개념이 실제 짜기로 어떻게 옮겨지는지 보이며 만들어 내는 모델의 더 넓은 틀과 이어진다. 여기서 보이는 재주는 만들어 내는 모델이 이룰 수 있는 것의 가장자리를 넓히는 더 앞선 변형과 넓힘을 이해하는 바탕이 된다.

## 연습문제

**연습문제 1.**
구체적인 자료 묶음으로 이 단원의 으뜸 셈을 좇아라. 큰 걸음마다 텐서 꼴을 적고 모든 차원이 서로 맞는지 확인하라.

??? success "연습문제 1 풀이"
    모델에 알맞은 꼴의 들임 묶음에서 시작한다. 층이나 함수 부르기마다 셈을 따라가며 바뀜 뒤 텐서 꼴을 적는다. 겹말기 층에서는 내놓기 차원 공식을 쓴다. 눈길 얼개에서는 물음, 열쇠, 값의 차원이 맞는지 확인한다. 마지막 내놓기 꼴이 바라던 목표 차원과 맞는지 굳힌다. 이 익힘은 자료가 얼개를 어떻게 흐르는지에 대한 직관을 쌓아 준다.

---

**연습문제 2.**
이 단원에 쓰인 손실 함수를 가려내고 모델 매개변수에 대한 기울기를 이끌어 내라. 왜 이 손실 함수가 이 일에 알맞은지 설명하라.

??? success "연습문제 2 풀이"
    손실 함수는 모델이 헤아린 값과 목표 사이의 어긋남을 잰다. 잡음 헤아리기에서는 평균 제곱 어긋남 손실 $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$을 쓰는데, 이것이 로그 가능도의 변분 아래 한계에 맞물리기 때문이다. 매개변수 $\theta$에 대한 기울기는 $-2(\epsilon - \epsilon_\theta) \nabla_\theta \epsilon_\theta$이며 헤아림 어긋남을 줄이는 방향을 가리킨다. 이 손실을 가장 작게 하는 것이 퍼짐 모델에서 자료 로그 가능도의 아래 한계를 가장 크게 하는 것과 같으므로 알맞다.

---

**연습문제 3.**
다른 잡음 차례표를 받쳐 주도록 이 짜기를 고쳐라(예컨대 선형에서 코사인으로, 또는 그 반대로). 두 차례표의 익히기 움직임과 표본 품질을 견주어라.

??? success "연습문제 3 풀이"
    두 차례표를 모두 짜고 각각으로 모델을 익힌다. $\bar{\alpha}_t = \cos^2\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)$으로 뜻매김한 코사인 차례표는 선형 차례표 $\beta_t = \beta_{\min} + t(\beta_{\max} - \beta_{\min})/T$에 견주어 잡음이 더 매끄럽게 늘어난다. 손실 곡선을 좇고 일정한 사이마다 표본을 만든다. 코사인 차례표는 신호 대 잡음비가 더 완만하게 줄어 때 걸음에 걸쳐 배움 신호가 더 고르므로 흔히 더 좋은 결과를 낸다.
