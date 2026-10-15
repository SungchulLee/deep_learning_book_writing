# 다중 노이즈 스코어의 목표와 브로드캐스팅

이 페이지는 σ를 MLP에 입력하는 NCSN과 어닐링된 랑주뱅 샘플러를 살펴본다. 여러 수준의 노이즈를 다룰 때는 목표 스코어의 배율과 표본별 손실 가중치의 축이 중요하다. 보존된 코드에는 두 부분의 불일치가 있으므로 의도한 DSM 식과 실제 계산을 구분한다.

## 1. 코드와 기존 출력

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
        
        # 잡음 임베딩
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
    # sigmas는 numpy에서 왔으므로 float64이다. dtype을 지정하지 않으면
    # Double 텐서가 만들어져 Float인 모델 가중치와 곱해지지 못한다.
    sigma = torch.tensor([sigmas[i] for i in sigma_idx],
                         dtype=torch.float32, device=x.device)
    
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
    plt.savefig('demo_ncsn_checkerboard.png', dpi=150, bbox_inches='tight')
    print("\nSaved demo_ncsn_checkerboard.png")
    print("\n✓ NCSN successfully learned multi-modal distribution!")
```

```
Noise Conditional Score Networks Demo
================================================================================

Noise schedule: [2.00000000e+01 8.59505945e+00 3.69375235e+00 1.58740105e+00
 6.82190321e-01 2.93173318e-01 1.25992105e-01 5.41454816e-02
 2.32691817e-02 1.00000000e-02]

Training NCSN...
Epoch     0 | Loss: 967414656.000000
Epoch   500 | Loss: 1014138368.000000
Epoch  1000 | Loss: 1145161472.000000
Epoch  1500 | Loss: 1030166976.000000
Epoch  2000 | Loss: 994114368.000000
Epoch  2500 | Loss: 1175464320.000000

Generating samples via annealed Langevin dynamics...

Saved demo_ncsn_checkerboard.png

✓ NCSN successfully learned multi-modal distribution!
```

## 2. σ 조건과 실제 구조

데이터가 N×2이면 σ를 N×1로 만들고 1→128→128 임베딩 MLP를 통과시킨다. N×128 조건을 데이터와 연결해 N×130 입력을 만들고, main에서 선택한 은닉층 세 개를 거쳐 N×2를 출력한다. 은닉층에는 GroupNorm(8,128)과 SiLU를 사용한다.

클래스 기본 n_layers는 4지만 학습 함수는 3을 지정한다. GroupNorm 때문에 hidden_dim은 8로 나누어져야 한다. Python·NumPy 스칼라 σ는 배치 길이만큼 복제하지만 **스칼라 torch.Tensor**는 1×1로 바꾸기만 해 배치 N>1에서 연결이 맞지 않는다. 텐서 σ는 길이 N이고 데이터·모델과 장치·dtype이 맞아야 한다.

main의 데이터는 두 좌표의 균등 난수와 바닥 함수로 만드는 체크무늬다. 20에서 0.01까지 로그 간격 표준편차 열 개를 사용한다. σ=20의 주변 분포도 데이터와 노이즈의 합이며 초기 N(0,20²I)와 정확히 같은 분포는 아니다.

## 3. 조건부 목표의 배율 오류

코드의 noise는 표준 정규 ξ이고 관측은 x̃=x+σξ다. 따라서 올바른 조건부 스코어 목표는 다음과 같다.

$$\nabla_{\widetilde x}\log q_\sigma(\widetilde x\mid x)
=-\frac{\widetilde x-x}{\sigma^2}=-\frac\xi\sigma$$

그런데 구현은 **−noise/σ²**를 사용한다. noise에 이미 σ가 곱해진 경우라면 맞지만 여기서는 그렇지 않아 목표가 1/σ배 더 크다. 조건별 회귀의 모집단 최적점도 의도한 주변 스코어의 1/σ배가 되어, 샘플러가 기대하는 스코어와 맞지 않는다.

## 4. 손실 가중치의 축 오류

표본별 제곱오차를 좌표 방향으로 합산하면 길이 N 벡터가 된다. weights는 N×1인데 이를 그대로 곱하면 브로드캐스팅 결과가 **N×N**이 된다. 각 표본 오차에 그 표본의 σ²를 곱하는 것이 아니라 모든 가중치·모든 표본 오차의 외적을 계산한다.

결과 스칼라는 이 배치에서 다음과 같다.

$$L_{code}=\left(\frac1N\sum_i\sigma_i^2\right)
\left(\frac1N\sum_j\|e_j\|^2\right)$$

의도한 표본별 가중 평균 $(1/N)\sum_i\sigma_i^2\|e_i\|^2$와 다르다. N=2, 가중치 (1,4), 오차 제곱합 (9,16)이면 실제 값은 2.5×12.5=31.25이고 의도한 값은 (9+64)/2=36.5다. 입력이 2,000개면 불필요한 2,000×2,000 중간 결과도 만든다.

따라서 현재 손실을 표준적인 σ² 가중 DSM으로 설명할 수 없다. 코드와 출력은 보존하되 목표 배율과 축 문제가 해결되기 전까지 성공적인 다중 노이즈 학습의 근거로 해석하지 않는다.

## 5. 샘플링의 시간 스케일

샘플러는 2,000×2 상태를 초기화하고 각 σ에서 100번, 총 1,000번 갱신한다. hᵢ=2×10⁻⁵σᵢ²이고 식은 다음과 같다.

$$x\leftarrow x+\frac{h_i}{2}s_\theta(x,\sigma_i)+\sqrt{h_i}\,\xi$$

보폭이 작아도 각 수준에서 충분히 혼합된다는 보장은 없다. 예를 들어 정확한 N(0,σ²) 스코어를 사용하면 한 단계 수축 계수는 1−10⁻⁵이고 100번 후 약 0.999다. 큰 초기 상태를 충분히 바꾸는 시간인지 확인해야 한다. 이는 실제 체크무늬 목표를 가우시안이라고 가정한 결론이 아니라 보폭·반복 수의 규모를 읽는 검사다.

초기 상태와 각 수준이 끝난 상태를 저장해 trajectory 길이는 11이다. main은 마지막 샘플과 학습 손실만 그리고 이 궤적은 사용하지 않는다. 상태는 CPU에 생성되므로 CUDA 모델로 자동 대응하지 않으며 데이터 차원도 2로 고정되어 있다.

## 6. 학습과 결과 해석

main은 3,000번 전체 배치를 갱신하며 매번 표본별로 σ를 선택한다. 시드를 고정하지 않고 검증·체크포인트·스코어 오차 검사를 하지 않는다. 학습 곡선과 마지막 산점도만으로 목표 배율·축 오류가 사라지거나 원본 분포를 학습했다고 결론 내릴 수 없다. 마지막 σ도 0이 아니므로 올바르게 학습·혼합하더라도 마지막 평활화 분포와 원본의 차이가 남는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
N×1 가중치와 길이 N의 오차 벡터를 곱하면 왜 N×N이 되는가?

</div>

??? success "연습문제 1 풀이"

    브로드캐스팅에서 길이 N은 1×N으로 해석된다. 따라서 행 방향 가중치와 열 방향 오차의 외적이 된다. 평균은 평균 가중치×평균 오차이며 표본별 대응 곱의 평균과 다르다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
x̃=x+σξ이고 ξ~N(0,I)일 때 −ξ/σ²를 회귀하면 의도한 주변 스코어와 어떻게 다른가?

</div>

??? success "연습문제 2 풀이"

    올바른 조건부 목표는 −ξ/σ다. 잘못된 목표는 같은 σ에서 1/σ배이므로 조건부 평균도 주변 스코어의 1/σ배다. 샘플러가 그 출력을 올바른 스코어로 사용하면 드리프트와 잡음의 균형이 달라진다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
샘플러의 작은 hᵢ와 성공 메시지가 충분한 생성 검증이 아닌 이유는 무엇인가?

</div>

??? success "연습문제 3 풀이"

    작은 보폭은 이산화 오차를 줄일 수 있어도 같은 반복 수에서는 이동·혼합 시간이 부족할 수 있다. 목표 스코어의 학습 오차와 이 코드의 배율·가중치 오류도 별개다. 유한성, 분포 포괄성, 여러 초기화와 검증 지표를 확인해야 하며 출력 문장 자체가 검증 결과는 아니다.



## 정리하며

다중 노이즈 모델에서는 표준 정규 노이즈와 실제 섭동을 구분하고 가중치 축을 표본별 오차와 맞춰야 한다. 이 코드의 −ξ/σ² 목표와 N×N 브로드캐스팅은 의도한 DSM과 다르다. 샘플링의 작은 보폭과 긴 반복 수 역시 별도로 검증해야 한다.
