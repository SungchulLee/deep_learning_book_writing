# 노이즈 조건부 스코어와 어닐링된 샘플링

노이즈 조건 스코어 네트워크(NCSN)는 여러 표준편차로 평활화한 데이터 분포의 스코어를 함께 학습한다. 큰 노이즈 수준에서 시작해 작은 수준으로 이동하는 어닐링된 랑주뱅 샘플링과 결합할 수 있다. 이 페이지는 두 반달 데이터의 예제를 살펴보되, 로스 웨이트와 스트라이드의 실제 설정을 함께 확인한다.

## 1. 코드와 기존 출력

```python
"""
7단원: 잡음 조건 점수 넷트웍(NCSN)
=================================================

어려움: 나아간 단계
시간: 3~4시간
미리 알 것: 단원 01-06

학습 목표:
- 여러 잣수의 점수 나타내기를 이해한다
- 식힘 랑주뱅 액션을 짠다
- 확산의 앞 과정과 잇는다

핵심 생각: 여러 잡음 수준에서 점수를 배운다
s_θ(x, σ_i) for σ_1 > σ_2 > ... > σ_L

지은이: 이성철 @ 연세대학교
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# 무작위로 뽑는 값이 아래에 나온다. 씨앗을 고정해야 이 쪽에 실린
# 수가 다시 나온다 — 고정하지 않으면 돌릴 때마다 다른 수가 찍힌다.
# 익힘 자료를 만드는 make_moons 는 numpy 쪽 무작위를 쓰므로 함께 고정한다
torch.manual_seed(0)
np.random.seed(0)

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
- 이것이 식힘 랑주뱅 액션이다

확산과의 이음:
----------------------
앞으로 가는 흐름: x_0 → x_1 → ... → x_T(잡음을 더한다)
거꾸로 가는 흐름: x_T ← ... ← x_1 ← x_0(잡음을 지운다)

스텝마다 알맞은 잡음 수준의 점수를 쓴다!
이것이 바로 확산 모델의 틀이다!
""")

class NCSN(nn.Module):
    """잡음 조건 점수 뉴럴 넷트웍"""
    def __init__(self, data_dim=2, noise_levels=10):
        super().__init__()
        self.noise_levels = noise_levels
        
        # 잡음 조건을 갖춘 함께 쓰는 뉴럴 넷트웍
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
            sigma_idx: 잡음 레이어 번호 [B], 값은 [0, noise_levels-1]
        """
        # sigma_idx을 [0, 1]으로 고르게 맞춘다
        sigma_embed = (sigma_idx.float() / self.noise_levels).unsqueeze(-1)
        x_with_sigma = torch.cat([x, sigma_embed], dim=-1)
        return self.net(x_with_sigma)

def train_ncsn(data, n_epochs=2000):
    """자료로 잡음 조건 점수 뉴럴 넷트웍을 익힌다"""
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
        
        # 무게를 준 로스(균형을 위해 1/σ²으로 무게를 준다)
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
    식힘 랑주뱅 액션 뽑기
    
    큰 잡음에서 시작해 차츰 줄인다
    """
    x = torch.randn(n_samples, 2) * sigmas[0]  # 사전 분포에서 첫자리매김
    
    trajectory = []
    
    for sigma_idx, sigma in enumerate(sigmas):
        # 이 잡음 수준에서의 랑주뱅 스텝
        epsilon = 2 * (sigma ** 2) / (sigmas[-1] ** 2) * 0.01  # 맞추어 가는 스텝 크기
        
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
핵심 눈썰미:
------------
1. 큰 잡음(σ_1)에서 시작한다: 표본이 온 공간을 덮는다
2. 잡음을 차츰 줄인다: 표본이 자료 다양체로 모인다
3. 잡음 수준마다 앞 수준을 다듬는다
4. 이것이 바로 뒤 확산 과정이다!

잡음 짜임 설계:
---------------------
- 등비 수열: σ_i = σ_max * (σ_min/σ_max)^(i/L)
- 레이어가 많을수록 옮겨감이 매끄럽지만 뽑기가 느리다
- 맞추어 가는 스텝 크기: ε_i ∝ σ_i²

DDPM과의 이음:
-----------------
DDPM forward: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
→ 차례표에 따라 잡음을 더하는 것과 같다

DDPM 거꾸로: 점수로 p(x_{t-1}|x_t)을 배운다
→ 식힘 랑주뱅과 같다!

이제 온전한 점수 바탕 구조를 지었다!
다음: 이어진 때의 꼴(SDE)
""")

print("\n✓ Module 07 complete!")


if __name__ == "__main__":
    pass
```

```
MODULE 07: Noise Conditional Score Networks
================================================================================

왜 잡음 수준이 여럿인가?
-------------------------
σ 하나만 쓸 때의 문제:
- σ이 작으면: 자료 가까이서는 정확하나 뽑기 어렵다(자료에서 멀면 점수가 사라진다)
- σ이 크면: 어디서나 뽑기 쉽지만 정밀하지 않다

풀이: 둘 다 쓴다!
- σ이 큰 데서 비롯한다(뽑기 쉽고 온 공간을 덮는다)
- σ을 차츰 줄인다(자료 분포에 맞추어 다듬는다)
- 이것이 식힘 랑주뱅 액션이다

확산과의 이음:
----------------------
앞으로 가는 흐름: x_0 → x_1 → ... → x_T(잡음을 더한다)
거꾸로 가는 흐름: x_T ← ... ← x_1 ← x_0(잡음을 지운다)

스텝마다 알맞은 잡음 수준의 점수를 쓴다!
이것이 바로 확산 모델의 틀이다!

Training NCSN with 10 noise levels...
σ range: [0.010, 1.000]
Epoch 0: Loss = 11250250.000000
Epoch 400: Loss = 10413595.000000
Epoch 800: Loss = 12017707.000000
Epoch 1200: Loss = 11552046.000000
Epoch 1600: Loss = 11454724.000000

Generating samples via annealed Langevin dynamics...
  Annealing step 1/10: σ = 1.0000
  Annealing step 2/10: σ = 0.5995
  Annealing step 3/10: σ = 0.3594
  Annealing step 4/10: σ = 0.2154
  Annealing step 5/10: σ = 0.1292
  Annealing step 6/10: σ = 0.0774
  Annealing step 7/10: σ = 0.0464
  Annealing step 8/10: σ = 0.0278
  Annealing step 9/10: σ = 0.0167
  Annealing step 10/10: σ = 0.0100
✓ Saved: 07_annealed_langevin.png

핵심 눈썰미:
------------
1. 큰 잡음(σ_1)에서 시작한다: 표본이 온 공간을 덮는다
2. 잡음을 차츰 줄인다: 표본이 자료 다양체로 모인다
3. 잡음 수준마다 앞 수준을 다듬는다
4. 이것이 바로 뒤 확산 과정이다!

잡음 짜임 설계:
---------------------
- 등비 수열: σ_i = σ_max * (σ_min/σ_max)^(i/L)
- 레이어가 많을수록 옮겨감이 매끄럽지만 뽑기가 느리다
- 맞추어 가는 스텝 크기: ε_i ∝ σ_i²

DDPM과의 이음:
-----------------
DDPM forward: x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
→ 차례표에 따라 잡음을 더하는 것과 같다

DDPM 거꾸로: 점수로 p(x_{t-1}|x_t)을 배운다
→ 식힘 랑주뱅과 같다!

이제 온전한 점수 바탕 구조를 지었다!
다음: 이어진 때의 꼴(SDE)


✓ Module 07 complete!
```

## 2. 노이즈 수준과 모델의 조건

코드는 1.0에서 0.01까지 로그 간격으로 표준편차 10개를 만든다. 0부터 세는 i=0,…,L−1에 대해 다음과 같다.

$$\sigma_i=\sigma_{\max}\left(\frac{\sigma_{\min}}{\sigma_{\max}}\right)^{i/(L-1)}$$

출력의 i/L 식과 달리 실제 np.linspace는 양 끝을 모두 포함한다. 모델은 σ 자체가 아니라 **인덱스 i를 10으로 나눈 스칼라**를 데이터에 연결한다. 따라서 마지막 조건은 0.9이며 인덱스·σ 대응을 보존해야 한다. 중간 인덱스가 새로운 σ를 정확히 나타낸다는 보장은 없다.

2차원 데이터에서는 입력 폭 3, 은닉 폭 128 두 개, 출력 폭 2의 SiLU MLP다. 파라미터는 17,282개다. 학습은 2,000개 반달 데이터에서 표본별로 노이즈 인덱스를 새로 선택해 2,000번 전체 배치 갱신을 수행한다.

## 3. DSM 목표와 로스 웨이트

관측은 x̃=x+σiξ, 목표는 −ξ/σi다. 코드의 로스는 다음과 같이 **1/σi²**로 가중한 원소 평균이다.

$$L=\frac1{Nd}\sum_{j=1}^N\frac1{\sigma_{i_j}^2}
\left\|s_\theta(\widetilde x_j,i_j)+\frac{\xi_j}{\sigma_{i_j}}\right\|^2$$

각 수준에서 양의 웨이트만 적용하면 충분한 표현력을 가진 모집단의 조건별 최적점 자체는 바뀌지 않는다. 하지만 공유 모델의 학습에서 수준별 중요도와 그래디언트 크기는 크게 달라진다. 목표 크기가 대략 1/σ인 데 다시 역분산 웨이트를 곱하므로 작은 σ를 매우 강하게 강조할 수 있다.

노이즈 수준을 균형 있게 학습하려고 σ² 웨이트를 사용하는 구성과 이 코드는 반대다. 출력의 설명만으로 표준적인 NCSN 웨이트를 구현했다고 볼 수 없다. [노이즈 조건 스코어 모델 논문](https://arxiv.org/abs/1907.05600)의 방법과 현재 선택을 구분한다.

작은 σ에서 참 스코어가 데이터로부터 멀어지면 항상 사라진다는 출력 설명도 일반적으로 맞지 않는다. 가우시안 스코어는 평균에서 멀수록 커진다. 낮은 밀도 영역에서 학습 데이터와 신호가 부족해 추정이 어려울 수 있다는 문제와 스코어 자체가 0이 된다는 주장은 다르다.

## 4. 실제 샘플링 스트라이드의 문제

샘플러는 500×2 표준 정규 상태에서 시작하고 각 수준에서 50회 갱신하므로 총 500회다. 갱신식은 다음과 같다.

$$x\leftarrow x+\epsilon_i s_\theta(x,i)+\sqrt{2\epsilon_i}\,\xi$$

실제 설정은 $\epsilon_i=2(\sigma_i^2/\sigma_{min}^2)0.01=200\sigma_i^2$다. 따라서 첫 수준 σ=1의 스트라이드는 **200**, 마지막 σ=0.01에서도 0.02다. 비율이 σ²에 비례한다는 사실만으로 안정적이지는 않다.

예를 들어 정확한 N(0,σ²) 스코어 −x/σ²를 사용해도 이 식의 AR 계수는 1−εi/σ²=−199여서 불안정하다. 반달의 평활화 분포가 이 가우시안과 같다는 뜻은 아니지만 스트라이드 상수를 무조건 안전하다고 설명할 수 없는 분명한 검사다. 코드에 안정성 검사나 발산 감지·스트라이드 조정은 없다.

## 5. 그림과 확산 과정의 구분

각 노이즈 수준이 끝난 뒤의 상태를 저장해 열 개 패널로 보여 준다. 축 범위 밖의 상태는 그림에서 가려지므로 샘플이 거의 보이지 않는 상황을 성공적인 수렴으로 읽으면 안 된다. 유한성, 스코어 크기와 분포 통계를 별도로 확인해야 하지만 이 코드는 그런 검증을 하지 않는다.

초기 N(0,σmax²I)가 실제 최대 노이즈 주변 분포와 정확히 같은 것도 아니다. 데이터의 위치와 크기에 비해 노이즈가 충분한지 확인해야 한다. 마지막 σ도 0이 아니므로 이상적으로 각 수준을 정확히 샘플링해도 원본이 아닌 마지막 평활화 분포가 대상이다.

어닐링된 랑주뱅은 관련된 생성 방법이지만 DDPM의 역전이나 일반 역시간 SDE와 동일한 식은 아니다. 시간별 원래 드리프트·확산 계수를 포함하는 역과정과, 고정 수준에서 여러 번 평형을 향해 이동하는 반복을 구분한다. 이 예제의 성공 메시지는 실제 생성 품질 검증이 아니다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
학습에서 노이즈 인덱스, 표준편차, 입력과 목표의 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"

    인덱스는 길이 N의 정수 벡터이고 sigma_vals도 길이 N이다. unsqueeze 후 N×1로 만들어 N×2 데이터·노이즈와 브로드캐스팅한다. 모델 조건은 i/10 스칼라를 연결하므로 N×3 입력이며 출력과 목표는 N×2다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
σ=0.01과 σ=1의 역분산 웨이트는 각각 얼마인가? 회귀 목표와 결합하면 어떤 차이를 만드는가?

</div>

??? success "연습문제 2 풀이"

    웨이트는 각각 10,000과 1이다. 작은 σ에서는 −ξ/σ의 크기도 커져 초기 오차의 제곱이 대략 1/σ² 규모일 수 있다. 여기에 다시 1/σ²를 곱해 작은 수준이 로스를 지배할 수 있다. 이는 σ²로 가중하는 균형화와 반대며 유한 모델의 최적화에 영향을 준다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
가우시안 N(0,σ²)의 정확한 스코어로 코드의 ε=200σ²를 검사하라.

</div>

??? success "연습문제 3 풀이"

    반복은 x′=(1−ε/σ²)x+√(2ε)ξ이므로 AR 계수는 −199다. 절댓값이 1보다 커 안정적인 정상분산이 없다. 여러 노이즈 수준을 사용하더라도 잘못된 스트라이드 상수가 자동으로 수정되지는 않는다. 이 예제에는 선형·코사인 DDPM 스케줄 비교가 없다.



## 정리하며

다중 노이즈 학습에서는 조건 인덱스와 σ의 대응, DSM 목표와 로스 웨이트, 샘플링 스트라이드를 함께 확인한다. 이 코드는 역분산 로스 웨이트와 매우 큰 상대 스트라이드를 사용하므로 출력의 성공 주장을 그대로 받아들일 수 없다. 어닐링된 랑주뱅과 DDPM 역과정도 구분한다.
