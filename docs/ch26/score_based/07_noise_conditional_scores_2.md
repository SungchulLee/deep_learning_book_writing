# 잡음 조건 점수 2

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 잡음 조건 점수 2을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 1. 코드

```python
"""
FILE: 07_noise_conditional_scores.py
어려움: 나아간 단계
걸리는 시간: 4~5시간
미리 알 것: 04-06, 여러 잣수 나타내기 이해

학습 목표:
    1. 잡음 조건 점수 그물(NCSN)을 짠다
    2. 식힘 랑주뱅 움직임을 이해한다
    3. 여러 잡음 잣수에서 모델을 익힌다
    4. 복잡한 분포에서 품질 높은 표본을 만든다

수학 바탕:
    NCSN은 여러 잡음 층에서 점수를 배운다: s_θ(x, σ) ≈ ∇log p_σ(x)
    
    여기서 p_σ(x) = ∫ p(y)N(x|y, σ²I)dy은 가우스 잡음으로 부드럽게 한 자료다.
    
    익힘 목표:
    L = E_σ E_x E_ε[λ(σ)||s_θ(x+ε, σ) + ε/σ²||²]
    
    여기서 각 기호는 다음과 같다.
    - σ ~ p(σ)은 잡음 분포에서 뽑는다
    - ε ~ N(0, σ²I)
    - λ(σ) = σ² is a weighting function
    
    달군 랑주뱅 움직임:
    잡음 층을 낮춰 가며 랑주뱅을 돌려 표본을 뽑는다.
    σ₁ > σ₂ > ... > σ_T
    
    이는 봉우리 무너짐을 이겨 내고 표본 품질을 높이는 데 도움이 된다.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================


class NCSN(nn.Module):
    """잡음 조건 점수 신경망."""
    
    def __init__(self, data_dim=2, hidden_dim=128, n_layers=4):
        super().__init__()
        
        # 잡음 박아 넣기
        self.sigma_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 점수 신경망
        layers = []
        input_dim = data_dim + hidden_dim
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.GroupNorm(8, hidden_dim),
                nn.SiLU(),
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dim, data_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, sigma):
        """s_θ(x, σ)을 셈한다."""
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.full((len(x), 1), sigma, device=x.device)
        else:
            sigma = sigma.view(-1, 1)
        
        # 잡음 수준을 박아 넣는다
        sigma_emb = self.sigma_embed(sigma)
        
        # 이어 붙이고 다룬다
        h = torch.cat([x, sigma_emb], dim=-1)
        return self.net(h)


def ncsn_loss(model, x, sigmas):
    """
    여러 잡음 수준으로 잡음 조건 점수 신경망 손실을 셈한다.
    
    인수:
        model: 잡음 조건 점수 신경망 모델
        x: 자료 표본, 꼴 (N, D)
        sigmas: 잡음 수준의 목록
    
    반환값:
        loss: 잡음 수준에 걸쳐 평균 낸 무게 있는 잡음 없애는 점수 맞추기 손실
    """
    # 자료 점마다 아무 잡음 수준을 뽑는다
    N = len(x)
    sigma_idx = torch.randint(0, len(sigmas), (N,))
    sigma = torch.tensor([sigmas[i] for i in sigma_idx], device=x.device)
    
    # 잡음 더하기
    noise = torch.randn_like(x)
    x_noisy = x + noise * sigma.view(-1, 1)
    
    # 점수 미리보기
    pred_score = model(x_noisy, sigma)
    
    # 목표 점수: -noise/σ²
    target_score = -noise / (sigma.view(-1, 1) ** 2)
    
    # 무게를 준 평균 제곱 어긋남(무게 = σ²)
    weights = sigma.view(-1, 1) ** 2
    loss = torch.mean(weights * torch.sum((pred_score - target_score) ** 2, dim=1))
    
    return loss


def anneal_langevin_sampling(model, sigmas, n_samples=100, n_steps_per_sigma=100,
                             step_size_ratio=0.00002):
    """
    식힘 랑주뱅 움직임 뽑기.
    
    잡음 수준을 낮춰 가며 랑주뱅을 돌려 뽑는다.
    이는 잡음이 클 때 공간을 살피고 잡음이 작을 때 다듬는 데 도움이 된다.
    
    인수:
        model: 익힌 잡음 조건 점수 신경망 모델
        sigmas: 잡음 짜임(줄어드는 차례), 보기: [10, 1, 0.1, 0.01]
        n_samples: 만들 표본의 개수
        n_steps_per_sigma: 잡음 수준마다 랑주뱅 걸음 수
        step_size_ratio: σ²에 대한 걸음 크기의 몫
    
    반환값:
        samples: 마지막 표본, 꼴 (n_samples, dim)
        trajectory: 뽑기 자취
    """
    # 큰 잡음에서 첫자리매김한다
    dim = 2  # 그려 보려 2차원이라 본다
    x = torch.randn(n_samples, dim) * sigmas[0]
    
    trajectory = [x.clone()]
    
    with torch.no_grad():
        for sigma in sigmas:
            step_size = step_size_ratio * (sigma ** 2)
            
            for _ in range(n_steps_per_sigma):
                # 지금 잡음 수준의 점수를 셈한다
                score = model(x, sigma)
                
                # 랑주뱅 걸음
                noise = torch.randn_like(x)
                x = x + (step_size / 2) * score + np.sqrt(step_size) * noise
            
            trajectory.append(x.clone())
    
    return x, trajectory


def train_ncsn(data, sigmas, epochs=5000, lr=1e-3):
    """2차원 자료로 잡음 조건 점수 신경망을 익힌다."""
    model = NCSN(data_dim=2, hidden_dim=128, n_layers=3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = ncsn_loss(model, data, sigmas)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")
    
    return model, losses


if __name__ == "__main__":
    print("Noise Conditional Score Networks Demo")
    print("=" * 80)
    
    # 바둑판 자료 묶음을 만든다
    def checkerboard_data(n_samples=2000):
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,)).float()
        x2 = x2_ + torch.floor(x1) % 2
        return torch.stack([x1, x2], dim=1)
    
    data = checkerboard_data(2000)
    
    # 등비 잡음 차례표
    sigmas = np.exp(np.linspace(np.log(20), np.log(0.01), 10))
    print(f"\nNoise schedule: {sigmas}")
    
    # 학습
    print("\nTraining NCSN...")
    model, losses = train_ncsn(data, sigmas, epochs=3000, lr=1e-3)
    
    # 뽑기
    print("\nGenerating samples via annealed Langevin dynamics...")
    samples, trajectory = anneal_langevin_sampling(
        model, sigmas, n_samples=2000, n_steps_per_sigma=100
    )
    
    # 시각화한다
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), s=1, alpha=0.5)
    axes[0].set_title('Training Data')
    axes[0].set_aspect('equal')
    
    axes[1].scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), s=1, alpha=0.5)
    axes[1].set_title('Generated Samples')
    axes[1].set_aspect('equal')
    
    axes[2].plot(losses)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Training Curve')
    axes[2].set_yscale('log')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('/home/claude/demo_ncsn_checkerboard.png', dpi=150, bbox_inches='tight')
    print("\nSaved demo_ncsn_checkerboard.png")
    print("\n✓ NCSN successfully learned multi-modal distribution!")
```

## 2. 논의

잡음 조건 점수 2의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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

## 정리하며

**다룬 것** — 잡음 조건 점수 2

잡음 조건 점수 2의 짜기는 이 마당에 자리 잡은 방식을 따른다.

고갱이 갈래는 `NCSN`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
