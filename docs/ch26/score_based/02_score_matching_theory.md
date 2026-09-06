# 점수 맞추기 이론

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 점수 맞추기 이론을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""
단원 02: 점수 맞추기 이론
================================

어려움: 처음~중간
시간: 3~4시간
PREREQUISITES: Module 01 (Score Functions Basics)

학습 목표:
-------------------
1. Understand why we can't compute scores directly from data
2. 드러난 점수 맞추기(ESM)의 목표를 배운다
3. 잡음 없애는 점수 맞추기(DSM)를 이해한다 - 핵심 생각이다!
4. 잡음 없애는 점수 맞추기를 베이즈 잡음 없애기와 잇는다
5. 장난감 자료의 기본 점수 맞추기를 짠다

MATHEMATICAL FOUNDATION:
-----------------------
THE PROBLEM:
Given dataset {x_i}_{i=1}^N ~ p_data(x), learn s(x) = ∇_x log p_data(x)

Challenge: We don't know p_data(x)! Only have samples!

NAIVE APPROACH (doesn't work):
Fit p_θ(x) to data, then compute s_θ(x) = ∇_x log p_θ(x)
문제: p_θ(x)을 고르게 맞추어야 하는데 이는 다룰 수 없다!

SCORE MATCHING SOLUTION:
p_data(x) 없이 점수 s_θ(x)을 참 점수에 곧바로 맞춘다!

핵심 통찰: 어느 분포도 몰라도 점수를 맞출 수 있다!

지은이: 이성철 @ 연세대학교
날짜: 2025년 11월
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from scipy.stats import multivariate_normal

plt.style.use('seaborn-v0_8-darkgrid')
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("MODULE 02: SCORE MATCHING THEORY")
print("=" * 80)

# ============================================================================
# 마디 1: 바탕이 되는 문제
# ============================================================================
print("\n" + "="  * 80)
print("SECTION 1: Why Direct Score Computation Fails")
print("=" * 80)

print("""
SCENARIO:
--------
Given: Dataset {x₁, x₂, ..., x_N} sampled from unknown p_data(x)
Goal: Learn score function s(x) = ∇_x log p_data(x)

WHY WE CAN'T COMPUTE SCORE DIRECTLY:
-----------------------------------
1. Don't have formula for p_data(x)
2. 모델 p_θ(x)을 자료에 맞출 수 있다
3. But normalizing p_θ(x) requires computing:
   Z_θ = ∫ p̃_θ(x) dx  (intractable!)
   
4. So can't compute: log p_θ(x) = log p̃_θ(x) - log Z_θ
5. Therefore can't compute: ∇_x log p_θ(x)

EXAMPLE:
-------
Say we parameterize: p̃_θ(x) = exp(E_θ(x))
여기서 E_θ은 신경망(에너지 함수)이다.

Then: p_θ(x) = exp(E_θ(x)) / Z_θ
where: Z_θ = ∫ exp(E_θ(x)) dx  ← INTRACTABLE!

But we want: s_θ(x) = ∇_x log p_θ(x) = ∇_x E_θ(x)

Good news: Score doesn't need Z_θ!
Bad news: Still can't train without knowing p_data(x)!

풀이: 점수 맞추기!
""")

# ============================================================================
# 마디 2: 드러난 점수 맞추기(ESM)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 2: Explicit Score Matching (ESM)")
print("=" * 80)

print("""
FISHER DIVERGENCE:
-----------------
Define distance between score functions:

D_Fisher(p||q) = (1/2) 𝔼_{x~p} ||∇_x log p(x) - ∇_x log q(x)||²

Minimize w.r.t. θ:
J_ESM(θ) = (1/2) 𝔼_{x~p_data} ||∇_x log p_data(x) - s_θ(x)||²

Problem: Still need ∇_x log p_data(x) which we don't have!

HYVÄRINEN'S TRICK (2005):
------------------------
Using integration by parts (Stein's identity):

J_ESM(θ) = 𝔼_{x~p_data} [ tr(∇_x s_θ(x)) + (1/2)||s_θ(x)||² ] + const

KEY INSIGHT:
- 목표에서 ∇_x log p_data(x)을 없앴다!
- 필요한 것은 p_data의 표본과 s_θ의 미분뿐이다
- tr(∇_x s_θ(x)) = sum of diagonal elements of Jacobian

DRAWBACK:
- tr(∇_x s_θ(x))을 셈하려면 헤세를 셈해야 한다
- 차원이 높으면 아주 비싸다!
- 그림 같은 데는 쓸 수 없다

더 나은 방식이 필요하다 → 잡음 없애는 점수 맞추기!
""")

# 장난감 자료에서 드러난 점수 맞추기를 보여 준다
print("\nDemonstration: ESM on 2D Gaussian")
print("-" * 80)

# 참 분포: 2차원 정규 분포
mu_true = np.array([0, 0])
Sigma_true = np.array([[1.0, 0.5], [0.5, 1.0]])

# 표본 만들기
n_samples = 1000
samples = np.random.multivariate_normal(mu_true, Sigma_true, n_samples)

# 참 점수 함수(견주기용)
Sigma_inv = np.linalg.inv(Sigma_true)
def true_score(x):
    return -Sigma_inv @ (x - mu_true)

# 단순한 선형 점수 모델: s_θ(x) = A(x - b)
# 정규 분포에서는 A = -Σ^(-1), b = μ이 가장 좋다
class LinearScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.randn(2, 2) * 0.1)
        self.b = nn.Parameter(torch.zeros(2))
    
    def forward(self, x):
        # s_θ(x) = A(x - b)
        return (self.A @ (x - self.b).T).T

# 드러난 점수 맞추기 목표: 𝔼[tr(∇s_θ) + 0.5||s_θ||²]
def esm_loss(model, x_batch):
    """드러난 점수 맞추기 손실"""
    x_batch.requires_grad_(True)
    
    # 점수 셈하기
    score = model(x_batch)
    
    # 항 1: 0.5 * ||s_θ(x)||²
    score_norm = 0.5 * torch.sum(score ** 2, dim=-1).mean()
    
    # 항 2: tr(∇_x s_θ(x)) - 야코비의 대각합
    # 점수의 성분마다 x에 대해 미분한다
    trace_term = 0
    for i in range(2):
        grad_outputs = torch.zeros_like(score)
        grad_outputs[:, i] = 1
        grads = torch.autograd.grad(score, x_batch, grad_outputs, 
                                     create_graph=True)[0]
        trace_term += grads[:, i].mean()
    
    return score_norm + trace_term

# 드러난 점수 맞추기로 익힌다
model_esm = LinearScoreModel()
optimizer = torch.optim.Adam(model_esm.parameters(), lr=0.01)

x_tensor = torch.FloatTensor(samples)
print("Training with Explicit Score Matching...")
for epoch in range(500):
    optimizer.zero_grad()
    loss = esm_loss(model_esm, x_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"  Epoch {epoch}: Loss = {loss.item():.4f}")

print("\n✓ ESM training complete!")
print(f"  Learned A ≈ -Σ^(-1):")
print(f"    {model_esm.A.detach().numpy()}")
print(f"  True -Σ^(-1):")
print(f"    {-Sigma_inv}")

# ============================================================================
# 마디 3: 잡음 없애는 점수 맞추기(DSM)
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 3: Denoising Score Matching (DSM) - The Key Idea!")
print("=" * 80)

print("""
THE PROBLEM WITH ESM:
-------------------
tr(∇_x s_θ(x))을 셈하는 것은 비싸다(헤세가 필요하다).
차원이 높으면(그림 등) 키울 수 없다.

DENOISING SCORE MATCHING (Vincent 2011):
---------------------------------------
핵심 통찰: 대신 잡음 없애는 법을 배운다!

PROCEDURE:
1. Take clean data x ~ p_data(x)
2. Add Gaussian noise: x̃ = x + σ * ε, where ε ~ N(0, I)
3. 잡음을 헤아리는 법을 배운다: s_θ(x̃) ≈ -(x̃ - x)/σ²

OBJECTIVE:
J_DSM(θ) = 𝔼_{x~p_data} 𝔼_{ε~N(0,I)} ||s_θ(x + σε) + ε/σ||²

Why this works:
--------------
The score of the noisy distribution q_σ(x̃|x) = N(x̃; x, σ²I) is:

∇_{x̃} log q_σ(x̃|x) = -(x̃ - x)/σ²

And the marginal noisy distribution q_σ(x̃) = ∫ q_σ(x̃|x)p_data(x)dx
이것으로 어림할 수 있는 점수를 가진다!

CONNECTION TO BAYESIAN INFERENCE:
--------------------------------
잡음 없애기가 곧 베이즈 사후 추론이다!

Given noisy observation x̃ = x + noise,
infer clean x via posterior: p(x|x̃)

The score ∇_x log p(x|x̃) tells us how to denoise!

This is exactly what diffusion models do:
- Forward: Add noise (known)
- Reverse: Denoise using learned score (learned)

ADVANTAGES OF DSM:
-----------------
✓ 헤세를 셈할 필요가 없다
✓ 기울기 셈하기가 단순하다
✓ 차원이 높아도 키울 수 있다
✓ 잡음 없애는 자기 부호기와 이어진다
✓ 퍼짐 모델의 바탕이다
""")

# 잡음 없애는 점수 맞추기를 보여 준다
print("\nDemonstration: DSM on 2D Swiss Roll")
print("-" * 80)

# 스위스 롤 자료를 만든다
def generate_swiss_roll(n_samples=1000, noise=0.1):
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    x1 = t * np.cos(t)
    x2 = t * np.sin(t)
    data = np.stack([x1, x2], axis=1)
    data += noise * np.random.randn(n_samples, 2)
    return data

swiss_roll = generate_swiss_roll(1000, noise=0.1)

# 여러 층 신경망 점수 신경망
class MLPScoreNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        return self.net(x)

# 잡음 없애는 점수 맞추기 손실
def dsm_loss(model, x_batch, sigma=0.1):
    """잡음 없애는 점수 맞추기 손실"""
    # 잡음 더하기
    noise = torch.randn_like(x_batch)
    x_noisy = x_batch + sigma * noise
    
    # 점수 미리보기
    predicted_score = model(x_noisy)
    
    # 목표 점수: -(x_noisy - x)/sigma^2 = -noise/sigma
    target_score = -noise / sigma
    
    # 평균 제곱 어긋남 손실
    loss = torch.mean((predicted_score - target_score) ** 2)
    return loss

# 잡음 없애는 점수 맞추기로 익힌다
model_dsm = MLPScoreNet()
optimizer_dsm = torch.optim.Adam(model_dsm.parameters(), lr=1e-3)

x_swiss = torch.FloatTensor(swiss_roll)
sigma_noise = 0.1

print("Training with Denoising Score Matching...")
for epoch in range(2000):
    optimizer_dsm.zero_grad()
    loss = dsm_loss(model_dsm, x_swiss, sigma=sigma_noise)
    loss.backward()
    optimizer_dsm.step()
    
    if epoch % 400 == 0:
        print(f"  Epoch {epoch}: Loss = {loss.item():.6f}")

print("\n✓ DSM training complete!")

# 배운 점수 마당을 그려 본다
x1_grid = np.linspace(-15, 15, 20)
x2_grid = np.linspace(-15, 15, 20)
X1, X2 = np.meshgrid(x1_grid, x2_grid)
grid_points = np.stack([X1.ravel(), X2.ravel()], axis=1)

with torch.no_grad():
    scores = model_dsm(torch.FloatTensor(grid_points)).numpy()

scores = scores.reshape(X1.shape + (2,))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 그림 1: 자료
axes[0].scatter(swiss_roll[:, 0], swiss_roll[:, 1], s=1, alpha=0.5, c='blue')
axes[0].set_title('Swiss Roll Data', fontsize=14, fontweight='bold')
axes[0].set_xlabel(r'$x_1$')
axes[0].set_ylabel(r'$x_2$')
axes[0].set_aspect('equal')
axes[0].grid(True, alpha=0.3)

# 그림 2: 배운 점수 마당
axes[1].scatter(swiss_roll[:, 0], swiss_roll[:, 1], s=1, alpha=0.3, c='blue')
axes[1].quiver(X1, X2, scores[:, :, 0], scores[:, :, 1], 
               color='red', alpha=0.6, scale=50, width=0.003)
axes[1].set_title('Learned Score Field via DSM\n(Points toward data manifold)', 
                  fontsize=14, fontweight='bold')
axes[1].set_xlabel(r'$x_1$')
axes[1].set_ylabel(r'$x_2$')
axes[1].set_aspect('equal')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('02_dsm_swiss_roll.png', dpi=150, bbox_inches='tight')
print("✓ Saved: 02_dsm_swiss_roll.png")
plt.close()

# ============================================================================
# 마디 4: 드러난 점수 맞추기와 잡음 없애는 점수 맞추기 견주기
# ============================================================================
print("\n" + "=" * 80)
print("SECTION 4: ESM vs DSM Comparison")
print("=" * 80)

comparison_table = """
|  Aspect                | Explicit Score Matching (ESM)     | Denoising Score Matching (DSM)  |
|-----------------------|-----------------------------------|----------------------------------|
| Objective             | 𝔼[tr(∇s_θ) + 0.5||s_θ||²]        | 𝔼||s_θ(x̃) + ε/σ||²             |
| Derivatives needed    | Second-order (Hessian)            | First-order only                 |
| Computational cost    | O(d²) per sample                  | O(d) per sample                  |
| Scalability           | Poor (high dimensions)            | Excellent                        |
| Connection            | Fisher divergence                 | Bayesian denoising               |
| Practical use         | Rare (too expensive)              | Standard (all modern models)     |
| Noise parameter       | Not needed                        | Requires σ choice                |
"""

print(comparison_table)

print("""
KEY TAKEAWAYS:
-------------
1. 드러난 점수 맞추기는 이론으로 아름답지만 셈이 비싸다
2. 잡음 없애는 점수 맞추기는 쓸모 있고 키울 수 있어 요즘 모든 모델이 쓴다
3. 잡음 없애는 점수 맞추기는 베이즈 잡음 없애기와 이어져 퍼짐의 바탕이 된다
4. 둘 다 고르게 맞추는 상수를 셈하지 않는다
5. 잡음 없애는 점수 맞추기는 자료만으로 점수를 배울 수 있게 한다

다음 걸음: 여러 잣수의 잡음 없애는 점수 맞추기
--------------------------
문제: 잡음 수준 σ 하나로는 어디서나 잘 듣지 않을 수 있다
풀이: 여러 잡음 수준에서 점수를 배운다
이것이 잡음 조건 점수 신경망(NCSN)으로 이어진다!
""")

# ============================================================================
# 간추리기와 익힘
# ============================================================================
print("\n" + "=" * 80)
print("MODULE SUMMARY")
print("=" * 80)

print("""
WHAT WE LEARNED:
---------------
1. 자료에서 곧바로 점수를 셈하는 것은 다룰 수 없다
2. 드러난 점수 맞추기는 고르게 맞추기를 피하지만 헤세가 필요하다
3. 잡음 없애는 점수 맞추기는 쓸모 있고 키울 수 있다
4. DSM = learning to denoise = Bayesian inference!
5. 잡음 없애는 점수 맞추기는 퍼짐 모델의 바탕이다

KEY FORMULAS:
------------
1. ESM objective:
   J_ESM = 𝔼[tr(∇_x s_θ(x)) + 0.5||s_θ(x)||²]

2. DSM objective:
   J_DSM = 𝔼_{x,ε} ||s_θ(x + σε) + ε/σ||²

3. Denoising score:
   ∇_x log q_σ(x̃|x) = -(x̃ - x)/σ²

FILES GENERATED:
---------------
1. 02_dsm_swiss_roll.png - 스위스 롤 자료의 잡음 없애는 점수 맞추기

""")

print("=" * 80)
print("EXERCISES")
print("=" * 80)

print("""
익힘 1: 피셔 벌어짐에서 드러난 점수 맞추기 이끌어 내기
-------------------------------------------
Starting from:
D_Fisher(p||q) = (1/2)𝔼_{x~p}||∇log p(x) - ∇log q(x)||²

Expand and use integration by parts to derive:
J_ESM = 𝔼[tr(∇s_θ) + 0.5||s_θ||²] + const

익힘 2: 드러난 점수 맞추기 짜기
------------------------
a) 정규 분포 섞기의 드러난 점수 맞추기를 짠다
b) 잡음 없애는 점수 맞추기와 셈 비용을 견준다
c) 결과가 비슷한지 확인한다

익힘 3: 잡음 없애는 점수 맞추기 이끌어 내기
-------------------------
Show that for q_σ(x̃|x) = N(x̃; x, σ²I):
∇_{x̃} log q_σ(x̃|x) = -(x̃ - x)/σ²

Interpret: denoising = computing posterior score!

익힘 4: 잡음 수준 살피기
-------------------------------
Train DSM with different σ values:
a) Very small σ (σ=0.01)
b) Medium σ (σ=0.1)
c) Large σ (σ=1.0)

How does choice of σ affect:
- 익히기가 안정된가?
- 배운 점수의 품질은?
- 뽑기의 움직임은?

익힘 5: 자기 부호기와의 이음
-------------------------------------
잡음 없애는 자기 부호기가 배우는 것: f(x̃) ≈ x
Show that:
a) 가장 좋은 잡음 없애기 함수는 점수와 이어진다
b) Connection: ∇_{x̃} log p(x̃) ∝ f(x̃) - x̃
c) 둘 다 짜서 견준다

도전 익힘: 봉우리 여럿인 분포
------------------------------------------
정규 분포 봉우리 9개인 2차원 "바둑판"을 만들어라.
a) σ 하나로 잡음 없애는 점수 맞추기를 익힌다
b) 모든 봉우리를 담는가?
c) 여러 σ 값을 시험한다
d) Motivate multi-scale approach (next module!)
""")

print("\n" + "=" * 80)
print("NEXT MODULE: 03_langevin_dynamics.py")
print("=" * 80)
print("""
Now that we can learn scores from data,
이를 어떻게 써서 표본을 만드는가?

답: 랑주뱅 움직임 - 점수를 쓴 마르코프 사슬 몬테카를로 뽑기!
이는 단원 01의 베이즈 셈으로 되돌아 이어진다!
""")

print("\n✓ Module 02 complete!")
print("=" * 80)


if __name__ == "__main__":
    pass
```

## 논의

점수 맞추기 이론의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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
