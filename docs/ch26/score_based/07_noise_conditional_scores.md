# 잡음 조건 점수

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 잡음 조건 점수을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 1. 코드

```python
"""
7단원: 잡음 조건 점수 그물(NCSN)
=================================================

어려움: 나아간 단계
시간: 3~4시간
미리 알 것: 단원 01-06

학습 목표:
- 여러 잣수의 점수 나타내기를 이해한다
- 식힘 랑주뱅 움직임을 짠다
- 퍼짐의 앞 과정과 잇는다

핵심 생각: 여러 잡음 수준에서 점수를 배운다
s_θ(x, σ_i) for σ_1 > σ_2 > ... > σ_L

지은이: 이성철 @ 연세대학교
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ========================================================================
# 메인
# ========================================================================

print("MODULE 07: Noise Conditional Score Networks")
print("="*80)

print("""
왜 잡음 수준이 여럿인가?
-------------------------
σ 하나만 쓸 때의 문제:
- σ이 작으면: 자료 가까이서는 정확하나 뽑기 어렵다(자료에서 멀면 점수가 사라진다)
- σ이 크면: 어디서나 뽑기 쉽지만 정밀하지 않다

풀이: 둘 다 쓴다!
- σ이 큰 데서 비롯한다(뽑기 쉽고 온 공간을 덮는다)
- σ을 차츰 줄인다(자료 분포에 맞추어 다듬는다)
- 이것이 식힘 랑주뱅 움직임이다

퍼짐과의 이음:
----------------------
앞으로 가는 흐름: x_0 → x_1 → ... → x_T(잡음을 더한다)
거꾸로 가는 흐름: x_T ← ... ← x_1 ← x_0(잡음을 지운다)

걸음마다 알맞은 잡음 수준의 점수를 쓴다!
이것이 바로 퍼짐 모델의 틀이다!
""")

class NCSN(nn.Module):
    """잡음 조건 점수 신경망"""
    def __init__(self, data_dim=2, noise_levels=10):
        super().__init__()
        self.noise_levels = noise_levels
        
        # 잡음 조건을 갖춘 함께 쓰는 신경망
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, 128),  # 잡음 수준을 위해 +1
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, data_dim)
        )
    
    def forward(self, x, sigma_idx):
        """
        인수:
            x: Data [B, D]
            sigma_idx: 잡음 층 번호 [B], 값은 [0, noise_levels-1]
        """
        # sigma_idx을 [0, 1]으로 고르게 맞춘다
        sigma_embed = (sigma_idx.float() / self.noise_levels).unsqueeze(-1)
        x_with_sigma = torch.cat([x, sigma_embed], dim=-1)
        return self.net(x_with_sigma)

def train_ncsn(data, n_epochs=2000):
    """자료로 잡음 조건 점수 신경망을 익힌다"""
    # 등비 잡음 차례표
    sigma_min, sigma_max = 0.01, 1.0
    n_sigmas = 10
    sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), n_sigmas))
    
    model = NCSN(data_dim=data.shape[1], noise_levels=n_sigmas)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    data_tensor = torch.FloatTensor(data)
    
    print(f"Training NCSN with {n_sigmas} noise levels...")
    print(f"σ range: [{sigma_min:.3f}, {sigma_max:.3f}]")
    
    for epoch in range(n_epochs):
        # 아무 잡음 수준
        sigma_idx = torch.randint(0, n_sigmas, (len(data),))
        sigma_vals = torch.FloatTensor([sigmas[i] for i in sigma_idx])
        
        # 잡음 더하기
        noise = torch.randn_like(data_tensor)
        noisy_data = data_tensor + sigma_vals.unsqueeze(-1) * noise
        
        # 점수 미리보기
        pred_score = model(noisy_data, sigma_idx)
        target_score = -noise / sigma_vals.unsqueeze(-1)
        
        # 무게를 준 손실(균형을 위해 1/σ²으로 무게를 준다)
        weights = 1.0 / (sigma_vals ** 2)
        loss = torch.mean(weights.unsqueeze(-1) * (pred_score - target_score) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 400 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")
    
    return model, sigmas

def annealed_langevin_sampling(model, sigmas, n_samples=500, n_steps_per_sigma=100):
    """
    식힘 랑주뱅 움직임 뽑기
    
    큰 잡음에서 시작해 차츰 줄인다
    """
    x = torch.randn(n_samples, 2) * sigmas[0]  # 사전 분포에서 첫자리매김
    
    trajectory = []
    
    for sigma_idx, sigma in enumerate(sigmas):
        # 이 잡음 수준에서의 랑주뱅 걸음
        epsilon = 2 * (sigma ** 2) / (sigmas[-1] ** 2) * 0.01  # 맞추어 가는 걸음 크기
        
        for step in range(n_steps_per_sigma):
            with torch.no_grad():
                sigma_idx_tensor = torch.ones(n_samples, dtype=torch.long) * sigma_idx
                score = model(x, sigma_idx_tensor)
            
            # 랑주뱅 고침
            x = x + epsilon * score + np.sqrt(2 * epsilon) * torch.randn_like(x)
        
        trajectory.append(x.clone().detach().numpy())
        print(f"  Annealing step {sigma_idx+1}/{len(sigmas)}: σ = {sigma:.4f}")
    
    return x.detach().numpy(), trajectory

# 달 모양 자료 묶음으로 익힌다
from sklearn.datasets import make_moons
data, _ = make_moons(n_samples=2000, noise=0.05)

model, sigmas = train_ncsn(data, n_epochs=2000)

print("\nGenerating samples via annealed Langevin dynamics...")
samples, trajectory = annealed_langevin_sampling(model, sigmas, n_samples=500, n_steps_per_sigma=50)

# 시각화한다
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()

for idx, (ax, samples_at_sigma) in enumerate(zip(axes, trajectory)):
    ax.scatter(data[:, 0], data[:, 1], s=1, alpha=0.2, c='blue', label='Data')
    ax.scatter(samples_at_sigma[:, 0], samples_at_sigma[:, 1], s=1, alpha=0.5, c='red', label='Samples')
    ax.set_title(f'σ = {sigmas[idx]:.3f}', fontweight='bold')
    ax.set_xlim(-2, 3)
    ax.set_ylim(-1.5, 2)
    ax.set_aspect('equal')
    if idx == 0:
        ax.legend()

plt.suptitle('Annealed Langevin Dynamics: Gradual Denoising', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('07_annealed_langevin.png', dpi=150)
plt.close()
print("✓ Saved: 07_annealed_langevin.png")

print("""
고갱이 눈썰미:
------------
1. 큰 잡음(σ_1)에서 시작한다: 표본이 온 공간을 덮는다
2. 잡음을 차츰 줄인다: 표본이 자료 다양체로 모인다
3. 잡음 수준마다 앞 수준을 다듬는다
4. 이것이 바로 뒤 퍼짐 과정이다!

잡음 짜임 설계:
---------------------
- 등비 수열: σ_i = σ_max * (σ_min/σ_max)^(i/L)
- 층이 많을수록 옮겨감이 매끄럽지만 뽑기가 느리다
- 맞추어 가는 걸음 크기: ε_i ∝ σ_i²

DDPM과의 이음:
-----------------
DDPM forward: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
→ 차례표에 따라 잡음을 더하는 것과 같다

DDPM 거꾸로: 점수로 p(x_{t-1}|x_t)을 배운다
→ 식힘 랑주뱅과 같다!

이제 온전한 점수 바탕 얼개를 지었다!
다음: 이어진 때의 꼴(SDE)
""")

print("\n✓ Module 07 complete!")


if __name__ == "__main__":
    pass
```

## 2. 논의

잡음 조건 점수의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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

**다룬 것** — 잡음 조건 점수

잡음 조건 점수의 짜기는 이 마당에 자리 잡은 방식을 따른다.

고갱이 갈래는 `NCSN`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
