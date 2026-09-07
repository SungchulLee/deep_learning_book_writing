# 랑주뱅 움직임

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 랑주뱅 움직임을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""
단원 03: 뽑기를 위한 랑주뱅 움직임
==========================================

어려움: 중간
시간: 2~3시간
미리 알 것: 단원 01-02, 기본 마르코프 사슬 몬테카를로 이해

학습 목표:
-------------------
1. 랑주뱅 마르코프 사슬 몬테카를로를 기울기 바탕 뽑기로 이해한다
2. 베이즈 추론의 메트로폴리스-헤이스팅스와 잇는다
3. 점수 바탕 뽑기를 위한 랑주뱅 움직임을 짠다
4. 모임의 성질을 살핀다
5. 퍼짐의 뒤 과정과의 이음을 이해한다

수학 바탕:
-----------------------
랑주뱅 움직임(랑주뱅 1908, 이후 여러 사람이 넓힘):

dx_t = ∇_x log p(x_t) dt + √(2)dW_t

띄엄띄엄한 때의 꼴(랑주뱅 MCMC):

x_{t+1} = x_t + ε * ∇_x log p(x_t) + √(2ε) * z_t

where z_t ~ N(0, I)

고갱이 눈썰미:
- 떠돎 항: ε * ∇_x log p(x_t)이 확률 높은 쪽으로 옮긴다
- 퍼짐 항: √(2ε) * z_t이 살펴보기 잡음을 더한다
- 캐내기(기울기)와 살펴보기(잡음)의 균형을 잡는다
- 목표 분포 p(x)으로 모인다!

점수 맞추기와의 이음:
s(x) = ∇_x log p(x)이므로 배운 점수로 표본을 뽑을 수 있다!

x_{t+1} = x_t + ε * s_θ(x_t) + √(2ε) * z_t

이것이 퍼짐 모델이 표본을 만드는 방식이다!

지은이: 이성철 @ 연세대학교
날짜: 2025년 11월
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)
torch.manual_seed(42)

print("=" * 80)
print("MODULE 03: LANGEVIN DYNAMICS")
print("=" * 80)

# ============================================================================
# 마디 1: 메트로폴리스-헤이스팅스에서 랑주뱅으로
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 1: Evolution from Metropolis-Hastings to Langevin")
print("=" * 80)

print("""
떠올리기: 메트로폴리스-헤이스팅스(베이즈 미룸에서)
----------------------------------------------------
1. Propose: x' ~ q(x'|x_t)  (e.g., x' = x_t + N(0, σ²))
2. 확률 α = min(1, p(x')/p(x_t))로 받아들인다
3. If accepted: x_{t+1} = x', else x_{t+1} = x_t

문제:
- 받아들임 비율이 낮을 수 있다(물리치는 제안이 많다)
- 고르게 맞추기까지 담아 p(x)을 셈해야 한다
- 아무 걸음 제안은 효율이 나쁘다

랑주뱅 마르코프 사슬 몬테카를로: 기울기에 바탕한 개선
----------------------------------------
핵심 통찰: 기울기 앎을 쓴다!

마구잡이 걸음 대신:
x' = x_t + N(0, σ²)

앎을 담은 제안을 쓴다.
x' = x_t + ε * ∇log p(x_t) + N(0, 2ε)

이점:
- 제안이 확률이 높은 쪽으로 움직인다(기울기 표류)
- 그러면서도 살펴본다(가우스 잡음)
- ε→0의 끝에서는 제안이 늘 받아들여진다!
- 온전한 확률이 아니라 점수만 필요하다!

이것이 고치지 않은 랑주뱅 알고리즘(ULA)이다.
""")

# 차이를 보여 준다
def metropolis_hastings_1d(target_logpdf, n_samples=5000, sigma=0.5):
    """아무 걸음을 쓴 여느 메트로폴리스-헤이스팅스"""
    samples = [0.0]  # 원점에서 시작
    x = 0.0
    n_accepted = 0
    
    for _ in range(n_samples):
        # 내놓기
        x_prop = x + np.random.randn() * sigma
        
        # 받아들이거나 물리치기
        log_alpha = target_logpdf(x_prop) - target_logpdf(x)
        if np.log(np.random.rand()) < log_alpha:
            x = x_prop
            n_accepted += 1
        
        samples.append(x)
    
    acceptance_rate = n_accepted / n_samples
    return np.array(samples), acceptance_rate

def langevin_mcmc_1d(target_logpdf, score_fn, n_samples=5000, epsilon=0.01):
    """기울기 앎을 쓴 랑주뱅 마르코프 사슬 몬테카를로"""
    samples = [0.0]
    x = 0.0
    
    for _ in range(n_samples):
        # 랑주뱅 고침
        x = x + epsilon * score_fn(x) + np.sqrt(2 * epsilon) * np.random.randn()
        samples.append(x)
    
    return np.array(samples)

# 목표: 정규 분포 섞기
def target_logpdf(x):
    """섞임 분포의 로그 확률밀도: 0.3*N(-2,0.5²) + 0.7*N(2,0.5²)"""
    from scipy.stats import norm
    p1 = 0.3 * norm.pdf(x, -2, 0.5)
    p2 = 0.7 * norm.pdf(x, 2, 0.5)
    return np.log(p1 + p2 + 1e-10)

def score_fn(x):
    """섞기의 점수"""
    from scipy.stats import norm
    p1 = 0.3 * norm.pdf(x, -2, 0.5)
    p2 = 0.7 * norm.pdf(x, 2, 0.5)
    
    s1 = -(x + 2) / 0.25  # N(-2, 0.5²)의 점수
    s2 = -(x - 2) / 0.25  # N(2, 0.5²)의 점수
    
    total_p = p1 + p2 + 1e-10
    return (p1 * s1 + p2 * s2) / total_p

print("\nComparison: Metropolis-Hastings vs Langevin MCMC")
print("-" * 80)

# 둘 다 돌린다
mh_samples, acc_rate = metropolis_hastings_1d(target_logpdf, n_samples=5000)
langevin_samples = langevin_mcmc_1d(target_logpdf, score_fn, n_samples=5000)

print(f"Metropolis-Hastings acceptance rate: {acc_rate:.2%}")
print(f"Langevin MCMC: No rejection (always accepts in continuous limit)")

# 시각화한다
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 참 분포
x_plot = np.linspace(-4, 4, 1000)
true_pdf = np.exp([target_logpdf(x) for x in x_plot])
true_pdf /= np.trapezoid(true_pdf, x_plot)

# 그림 1: 메트로폴리스-헤이스팅스 자취
axes[0, 0].plot(mh_samples[:500], alpha=0.7, linewidth=0.5)
axes[0, 0].set_title('Metropolis-Hastings Trajectory', fontweight='bold')
axes[0, 0].set_xlabel('Iteration')
axes[0, 0].set_ylabel(r'$x$')
axes[0, 0].grid(True, alpha=0.3)

# 그림 2: 랑주뱅 자취
axes[0, 1].plot(langevin_samples[:500], alpha=0.7, linewidth=0.5)
axes[0, 1].set_title('Langevin MCMC Trajectory', fontweight='bold')
axes[0, 1].set_xlabel('Iteration')
axes[0, 1].set_ylabel(r'$x$')
axes[0, 1].grid(True, alpha=0.3)

# 그림 3: 메트로폴리스-헤이스팅스 히스토그램
axes[1, 0].hist(mh_samples[1000:], bins=50, density=True, alpha=0.6, label='MH samples')
axes[1, 0].plot(x_plot, true_pdf, 'r-', linewidth=2, label='True distribution')
axes[1, 0].set_title(f'Metropolis-Hastings\n(Acc. rate: {acc_rate:.1%})', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 그림 4: 랑주뱅 히스토그램
axes[1, 1].hist(langevin_samples[1000:], bins=50, density=True, alpha=0.6, label='Langevin samples')
axes[1, 1].plot(x_plot, true_pdf, 'r-', linewidth=2, label='True distribution')
axes[1, 1].set_title('Langevin MCMC\n(Gradient-guided)', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_mh_vs_langevin.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: 03_mh_vs_langevin.png")
plt.close()

print("""
고갱이 살핌:
- 랑주뱅은 기울기 앎을 써서 더 효율 좋게 살핀다
- 둘 다 목표 분포로 모인다
- 랑주뱅은 복잡하고 봉우리 여럿인 분포를 더 잘 다룬다
- M-H의 받아들임 비율 절충은 랑주뱅에는 들어맞지 않는다
""")

# ============================================================================
# 마디 2: 랑주뱅 움직임 이론
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Theory of Langevin Dynamics")
print("=" * 80)

print("""
이어진 때의 랑주뱅 움직임:
---------------------------------
확률 미분 방정식(SDE):

dx_t = ∇_x log p(x_t) dt + √2 dW_t

여기서 W_t은 여느 브라운 움직임이다.

느낌으로 보면:
- 정해진 떠돎: ∇log p(x)이 확률 높은 쪽으로 끈다
- 확률 퍼짐: √2 dW_t이 공간을 살핀다
- 균형이 p(x)로 모여듦을 보장한다

포커-플랑크 방정식:
----------------------
x_t의 분포 p_t(x)는 다음과 같이 바뀐다.

∂p_t/∂t = -∇·(p_t ∇log p) + Δp_t
        = ∇·(p_t ∇log(p/p_t))

At equilibrium (∂p_t/∂t = 0), we have p_t = p!

띄엄띄엄한 때의 꼴:
--------------------
오일러-마루야마 띄엄띄엄하게 하기:

x_{t+1} = x_t + ε ∇_x log p(x_t) + √(2ε) z_t

여기서 z_t ~ N(0, I)이고 ε은 걸음 크기이다.

CONVERGENCE:
- ε→0이면 띄엄띄엄한 과정 → 이어진 랑주뱅 확률 미분 방정식
- ε이 넉넉히 작으면 목표 분포로 모인다
- 모여드는 빠르기는 p(x)의 성질에 달렸다

실제로 살필 점:
- 걸음 크기 ε: 너무 크면 흔들리고 너무 작으면 느리게 모여든다
- 걸음 수: 많을수록 표본이 좋아지지만 느려진다
- 첫자리: 어떤 분포에서 비롯해도 된다(보기: N(0,I))

점수 맞추기와의 이음:
----------------------------
자료에서 s_θ(x) ≈ ∇log p(x)를 배웠으므로,
다음으로 표본을 뽑을 수 있다.

x_{t+1} = x_t + ε s_θ(x_t) + √(2ε) z_t

이것이 점수 바탕 뽑기이다!
고르게 맞춘 확률이 필요 없다!
""")

# ============================================================================
# 마디 3: 랑주뱅 뽑기 짜기
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Langevin Sampling with Learned Scores")
print("=" * 80)

print("\nExample: 2D Swiss Roll with Learned Score Network")
print("-" * 80)

# 스위스 롤을 만든다
def make_swiss_roll(n_samples=2000, noise=0.1):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x = t * np.cos(t)
    y = t * np.sin(t)
    X = np.stack([x, y], axis=1)
    X += noise * np.random.randn(n_samples, 2)
    return X

swiss_data = make_swiss_roll(2000)

# 점수 신경망을 익힌다(단원 02을 단순하게 만든 것)
class ScoreNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        return self.net(x)

def train_score_dsm(data, sigma=0.5, n_epochs=1000):
    """잡음 없애는 점수 맞추기로 점수 신경망을 익힌다"""
    model = ScoreNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    data_tensor = torch.FloatTensor(data)
    
    for epoch in range(n_epochs):
        # 잡음 없애는 점수 맞추기 손실
        noise = torch.randn_like(data_tensor)
        noisy_data = data_tensor + sigma * noise
        predicted_score = model(noisy_data)
        target_score = -noise / sigma
        
        loss = torch.mean((predicted_score - target_score) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0:
            print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")
    
    return model

print("Training score network...")
score_net = train_score_dsm(swiss_data, sigma=0.5, n_epochs=1000)
print("✓ Training complete!")

# 랑주뱅 뽑기 함수
def langevin_sampling(score_model, n_samples=500, n_steps=1000, epsilon=0.01, 
                     init_x=None):
    """
    배운 점수로 랑주뱅 움직임을 써서 뽑는다.
    
    인수:
        score_model: 익힌 점수 신경망
        n_samples: 만들 표본의 개수
        n_steps: 랑주뱅 걸음 수
        epsilon: 걸음 크기
        init_x: 첫 표본(None이면 N(0, I)을 쓴다)
    
    반환값:
        samples: 만들어 낸 표본 [n_samples, dim]
        trajectory: 뽑기의 온 자취 [n_steps, n_samples, dim]
    """
    if init_x is None:
        x = torch.randn(n_samples, 2) * 3  # 더 넓은 분포에서 첫자리매김
    else:
        x = init_x.clone()
    
    trajectory = [x.clone().detach().numpy()]
    
    for step in range(n_steps):
        with torch.no_grad():
            score = score_model(x)
        
        # 랑주뱅 고침
        x = x + epsilon * score + np.sqrt(2 * epsilon) * torch.randn_like(x)
        
        if step % 100 == 0:
            trajectory.append(x.clone().detach().numpy())
    
    samples = x.detach().numpy()
    return samples, trajectory

print("\nGenerating samples via Langevin dynamics...")
samples, trajectory = langevin_sampling(score_net, n_samples=500, n_steps=1000, epsilon=0.01)
print("✓ Sampling complete!")

# 결과를 그려 본다
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 그림 1: 참 자료
axes[0].scatter(swiss_data[:, 0], swiss_data[:, 1], s=1, alpha=0.5, c='blue')
axes[0].set_title('True Data Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel(r'$x_1$')
axes[0].set_ylabel(r'$x_2$')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# 그림 2: 만든 표본
axes[1].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.5, c='red')
axes[1].set_title('Generated via Langevin Sampling', fontsize=14, fontweight='bold')
axes[1].set_xlabel(r'$x_1$')
axes[1].set_ylabel(r'$x_2$')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)

# 그림 3: 견주기
axes[2].scatter(swiss_data[:, 0], swiss_data[:, 1], s=1, alpha=0.3, c='blue', label='True')
axes[2].scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3, c='red', label='Generated')
axes[2].set_title('Overlay Comparison', fontsize=14, fontweight='bold')
axes[2].set_xlabel(r'$x_1$')
axes[2].set_ylabel(r'$x_2$')
axes[2].legend()
axes[2].set_aspect('equal')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('03_langevin_sampling_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 03_langevin_sampling_results.png")
plt.close()

# 자취가 바뀌는 모습을 그려 본다
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, (ax, traj_snapshot) in enumerate(zip(axes, trajectory)):
    step = idx * 100
    ax.scatter(traj_snapshot[:, 0], traj_snapshot[:, 1], s=1, alpha=0.5, c='purple')
    ax.scatter(swiss_data[:, 0], swiss_data[:, 1], s=0.5, alpha=0.1, c='blue')
    ax.set_title(f'Step {step}', fontsize=12, fontweight='bold')
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle('Langevin Sampling Evolution Over Time', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('03_langevin_trajectory.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 03_langevin_trajectory.png")
plt.close()

print("""
✓ 랑주뱅 뽑기가 스위스 롤에서 표본을 잘 만들었다!

눈여겨볼 점은 다음과 같다.
- 아무 정규 잡음에서 시작했다
- 점수 이끌기로 자료 분포 쪽으로 차츰 옮겨 갔다
- 마지막 표본이 참 자료 분포와 잘 맞는다
- 이것이 바로 퍼짐 모델이 만들어 내는 방식이다!
""")

# ============================================================================
# 마디 4: 퍼짐 모델과의 이음
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: Connection to Diffusion Models")
print("=" * 80)

print("""
퍼짐 모델은 여러 자의 랑주뱅 움직임이다
-----------------------------------------------

핵심 통찰: 여러 잡음 수준에서 랑주뱅 뽑기를 쓴다!

앞으로 가는 흐름(퍼짐):
x_0 → x_1 → x_2 → ... → x_T
x_T ~ N(0, I)이 될 때까지 잡음을 차츰 더한다

거꾸로 가는 흐름(만들어 내기):
x_T ← x_{T-1} ← ... ← x_1 ← x_0
배운 점수로 차츰 잡음을 없앤다!

거꾸로 걸음:
x_{t-1} = x_t + ε * s_θ(x_t, t) + √(2ε) * z_t

여기서 s_θ(x_t, t)은 잡음 수준 t의 점수이다.

이것이 바로 랑주뱅 움직임이다!
다만 때에 따라 달라지는 점수를 쓸 뿐이다!

달군 랑주뱅 움직임:
--------------------------
잡음 층의 이음을 쓴다: σ_1 > σ_2 > ... > σ_L

층 i마다:
1. Learn score s_θ(x, σ_i)
2. 랑주뱅을 돌린다: x ← x + ε * s_θ(x, σ_i) + √(2ε) * z

잡음이 큰 데서 비롯한다(뽑기 쉬우나 성기다)
잡음을 차츰 줄인다(뽑기 어려우나 더 촘촘하다)

이것이 다음의 밑바탕이다.
- 점수 바탕 만들어 내는 모델(Song & Ermon, 2019)
- 잡음 지우기 퍼짐 확률 모델(Ho 외, 2020)
- 요즘의 모든 퍼짐 모델이다!

여태 배운 것:
------------------------
✓ 단원 01: 점수 함수란 무엇인가?
✓ 단원 02: 어떻게 배우는가(점수 맞추기)?
✓ 단원 03: 어떻게 쓰는가(랑주뱅 뽑기)?

다음: 실제 쓰임새를 위한 여러 잣수 틀!
""")

# ============================================================================
# 간추리기와 익힘
# ============================================================================
print("\n" + "=" * 80)
print("MODULE SUMMARY")
print("=" * 80)

print("""
배운 것:
---------------
1. 랑주뱅 움직임은 기울기 바탕 MCMC 뽑기다
2. 점수 함수로 뽑기를 이끈다
3. 아무 걸음 메트로폴리스-헤이스팅스보다 효율이 좋다
4. 점수 맞추기로 배운 점수와 함께 통한다
5. 퍼짐 모델 뽑기의 바탕이다!

고갱이 식:
------------
1. 이어진 랑주뱅 SDE:
   dx_t = ∇log p(x_t) dt + √2 dW_t

2. 띄엄띄엄한 랑주뱅 MCMC:
   x_{t+1} = x_t + ε * ∇log p(x_t) + √(2ε) * z_t

3. 점수 바탕 뽑기:
   x_{t+1} = x_t + ε * s_θ(x_t) + √(2ε) * z_t

4. 달군 랑주뱅(미리 보기):
   잡음 층 σ_1 > ... > σ_L에서 점수를 쓴다

만들어진 파일:
---------------
1. 03_mh_vs_langevin.png - 메트로폴리스-헤이스팅스와의 견줌
2. 03_langevin_sampling_results.png - 만들어 낸 표본
3. 03_langevin_trajectory.png - 때에 따른 바뀜
""")

print("\n" + "=" * 80)
print("EXERCISES")
print("=" * 80)

print("""
익힘 1: 모임 살피기
-------------------------------
For 1D Gaussian N(0,1):
a) 걸음 크기 ε을 달리해 랑주뱅을 돌린다
b) 모여드는 빠르기를 재어라(과녁까지의 KL 갈림)
c) ε에 따른 모임을 그린다
d) 가장 좋은 걸음 크기를 찾는다

익힘 2: 봉우리 여럿인 분포
-----------------------------------
가우스 4개(정사각형의 꼭짓점)로 2차원 섞임 분포를 만들어라.
a) 잡음 없애는 점수 맞추기로 점수 신경망을 익힌다
b) 랑주뱅 뽑기를 돌린다
c) 모든 봉우리를 들르는가?
d) 여러 첫자리매김과 걸음 크기를 시험한다

익힘 3: 메트로폴리스로 고친 랑주뱅
---------------------------------------
MALA를 짜라(메트로폴리스 받아들임을 더한다).
a) 랑주뱅으로 제안한다: x' = x + ε*∇log p(x) + √(2ε)*z
b) 메트로폴리스-헤이스팅스 확률로 받아들인다
c) 고치지 않은 랑주뱅과 견준다
d) 메트로폴리스로 고친 랑주뱅은 언제 도움이 되는가?

익힘 4: 점수의 어긋남
-----------------------
점수 그물에 어긋남이 있어 s_θ(x) ≠ ∇log p(x)이면:
a) 이것이 표본에 어떤 영향을 주는가?
b) 일부러 점수를 치우치게 하여 짠다
c) 분포의 어긋남을 잰다
d) 퍼짐 모델에 어떤 뜻이 있는가?

익힘 5: 식힘 차례표
-----------------------------
달군 랑주뱅을 짜라.
a) 잡음 짜임 σ_1 > σ_2 > ... > σ_L을 뜻매김하여라
b) 층마다 점수를 익혀라(2단원을 보라)
c) 큰 잡음에서 시작해 뽑는다
d) 수준 하나의 랑주뱅과 견준다

도전 익힘: 3차원 뽑기
------------------------------
Extend to 3D:
a) 3차원 스위스 롤이나 나선 자료를 만든다
b) 3차원 점수 신경망을 익힌다
c) 랑주뱅 뽑기를 짠다
d) 3차원 자취를 그려 본다
e) 차원이 모임에 어떤 영향을 주는가?
""")

print("\n" + "=" * 80)
print("NEXT: Multi-Scale Score Modeling")
print("=" * 80)
print("""
이제 벽돌이 모두 갖추어졌다.
✓ 점수 함수(1단원)
✓ 그것을 배우는 점수 맞추기(2단원)
✓ 표본을 뽑는 랑주뱅 움직임(3단원)

다음: 실제 자료(그림 등)에서 이것이 통하게 하려면?

답: 여러 잡음 수준에서 점수를 배운다!
이것이 온전한 퍼짐 틀로 이어진다!
""")

print("\n✓ Module 03 complete!")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

랑주뱅 움직임의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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
