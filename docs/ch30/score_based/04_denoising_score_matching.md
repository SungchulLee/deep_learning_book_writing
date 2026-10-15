# 디노이징 스코어 매칭

디노이징 스코어 매칭(DSM)은 깨끗한 데이터에 가우시안 노이즈를 더하고, 그 조건부 스코어를 회귀 목표로 사용한다. 이 페이지에서는 2차원 나선 데이터를 고정된 노이즈 수준으로 평활화한 스코어를 학습하고 랑주뱅 반복에 사용한다. 원본 분포를 정확히 복원하거나 완성된 다단계 확산 모델을 학습하는 예제는 아니다.

## 1. 코드와 기존 출력

```python
"""
FILE: 04_denoising_score_matching.py
어려움: 중간
걸리는 시간: 3~4시간
미리 알 것: 01-03, 기본 신경망, 잡음 흔들기 이해

학습 목표:
    1. 쓸모 있는 잡음 없애는 점수 맞추기를 짠다
    2. 2차원 자료 묶음으로 점수 신경망을 익힌다
    3. 잡음 알맹이 고르기를 이해한다
    4. 배운 모델에서 표본을 만든다

수학 바탕:
    잡음 없애는 점수 맞추기는 자료에 잡음을 더하고 잡음의 방향을 헤아리는 법을 배운다.
    
    Training: Minimize E_x E_ε[||s_θ(x + ε) + ε/σ²||²]
    where ε ~ N(0, σ²I)
    
    이는 잡음 알맹이 q에 대해 ∇log q(x̃|x)를 배우는 것과 같다.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
import os

# ========================================================================
# 메인
# ========================================================================


class ScoreNetwork(nn.Module):
    """2차원 자료의 점수 신경망."""
    
    def __init__(self, hidden_dims=[128, 128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = 2
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.Softplus(),  # 매끄러운 깨움
            ])
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def dsm_loss(model, x, sigma=0.5):
    """잡음 없애는 점수 맞추기 손실을 셈한다."""
    # 잡음 더하기
    noise = torch.randn_like(x) * sigma
    x_noisy = x + noise
    
    # 점수 미리보기
    pred_score = model(x_noisy)
    
    # 목표 점수: -noise/sigma²
    target_score = -noise / (sigma ** 2)
    
    # 평균 제곱 어긋남 손실
    loss = torch.mean(torch.sum((pred_score - target_score) ** 2, dim=1))
    return loss


def train_score_model(data, sigma=0.5, epochs=5000, lr=1e-3):
    """2차원 자료로 점수 모델을 익힌다."""
    model = ScoreNetwork()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = dsm_loss(model, data, sigma)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 500 == 0:
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.6f}")
    
    return model, losses


def sample_langevin(model, n_samples=100, n_steps=1000, step_size=0.01, noise_scale=1.0):
    """배운 점수로 랑주뱅 움직임을 써서 표본을 만든다."""
    # 사전 분포에서 첫자리매김한다
    x = torch.randn(n_samples, 2) * noise_scale
    
    samples = []
    
    with torch.no_grad():
        for step in range(n_steps):
            score = model(x)
            noise = torch.randn_like(x)
            x = x + (step_size / 2) * score + np.sqrt(step_size) * noise
            
            if step % 10 == 0:
                samples.append(x.clone())
    
    return x, samples


if __name__ == "__main__":
    print("Denoising Score Matching on 2D Datasets")
    print("=" * 80)
    
    # 스위스 롤 자료 묶음을 만든다
    from sklearn.datasets import make_swiss_roll
    data_np, _ = make_swiss_roll(n_samples=2000, noise=0.5)
    data = torch.tensor(data_np[:, [0, 2]] / 10, dtype=torch.float32)  # X-Z 평면을 쓴다
    
    print(f"\nDataset: Swiss Roll")
    print(f"Number of samples: {len(data)}")
    print(f"Data range: [{data.min():.2f}, {data.max():.2f}]")
    
    # 학습
    print("\nTraining score model...")
    model, losses = train_score_model(data, sigma=0.5, epochs=3000, lr=1e-3)
    
    # 뽑기
    print("\nGenerating samples via Langevin dynamics...")
    final_samples, trajectory = sample_langevin(
        model, n_samples=500, n_steps=1000, step_size=0.05
    )
    
    # 시각화한다
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 학습 데이터
    axes[0].scatter(data[:, 0].numpy(), data[:, 1].numpy(), 
                    s=1, alpha=0.5, c='blue')
    axes[0].set_title('Training Data')
    axes[0].axis('equal')
    axes[0].grid(True, alpha=0.3)
    
    # 만든 표본
    axes[1].scatter(final_samples[:, 0].numpy(), final_samples[:, 1].numpy(),
                    s=1, alpha=0.5, c='red')
    axes[1].set_title('Generated Samples')
    axes[1].axis('equal')
    axes[1].grid(True, alpha=0.3)
    
    # 익히기 곡선
    axes[2].plot(losses)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('DSM Loss')
    axes[2].set_title('Training Curve')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_dsm_swiss_roll.png', dpi=150, bbox_inches='tight')
    print("\nSaved demo_dsm_swiss_roll.png")
    
    print("\n✓ Successfully trained and sampled from Swiss roll distribution!")
```

```
Denoising Score Matching on 2D Datasets
================================================================================

Dataset: Swiss Roll
Number of samples: 2000
Data range: [-1.23, 1.49]

Training score model...
Epoch     0 | Loss: 8.178442
Epoch   500 | Loss: 5.123600
Epoch  1000 | Loss: 5.013676
Epoch  1500 | Loss: 4.852315
Epoch  2000 | Loss: 4.981414
Epoch  2500 | Loss: 4.870563

Generating samples via Langevin dynamics...

Saved demo_dsm_swiss_roll.png

✓ Successfully trained and sampled from Swiss roll distribution!
```

## 2. 입력과 네트워크

main은 scikit-learn의 3차원 Swiss roll 데이터에서 첫 번째와 세 번째 좌표를 선택하고 10으로 나눈다. 입력은 2,000×2 float32다. 데이터 생성에 지정한 noise=0.5도 좌표 스케일을 나누면서 함께 축소되며, DSM에서 더하는 σ=0.5와는 서로 다른 단계의 노이즈다.

`ScoreNetwork`는 기본적으로 2→128→128→128→2 구조이며 은닉층마다 Softplus를 사용한다. 파라미터는 33,666개다. 데이터 차원이 2로 고정되어 있어 이미지나 다른 차원에 그대로 적용하는 구조는 아니다. σ를 네트워크 입력으로 받지도 않으므로 학습된 모델은 선택한 한 노이즈 수준에 대응한다.

## 3. 회귀 목표와 주변 스코어

노이즈 η를 $N(0,\sigma^2I)$에서 뽑고 $\widetilde x=x_0+\eta$로 둔다. 코드의 noise에는 이미 σ가 곱해져 있으므로 목표는 −noise/σ²다.

$$\nabla_{\widetilde x}\log q_\sigma(\widetilde x\mid x_0)
=-\frac{\widetilde x-x_0}{\sigma^2}=-\frac\eta{\sigma^2}$$

`dsm_loss`는 좌표별 제곱오차를 **합산하고 표본 사이에서 평균** 낸다. 모델 출력, 목표, 노이즈 관측의 모양은 모두 N×2다. 앞 페이지의 원소 전체 평균과는 배율이 다르므로 손실값을 그대로 비교하지 않는다.

$$J(\theta)=\mathbb E_{x_0,\eta}\left\|s_\theta(x_0+\eta)+\frac\eta{\sigma^2}\right\|^2$$

모집단 최적 회귀 함수는 목표의 관측 조건부 평균, 즉 $s^*(\widetilde x)=\nabla_{\widetilde x}\log p_\sigma(\widetilde x)$다. pσ는 가우시안으로 평활화한 데이터의 주변 밀도다. 유한 학습 데이터에서는 그 경험 분포를 평활화한 목표에 맞추게 되므로 모집단 데이터에 대한 일반화는 별도로 평가해야 한다.

각 관측값에 가능한 깨끗한 표본이 여럿 대응하면 목표의 조건부 분산이 남아 최적 DSM 손실도 0일 필요가 없다. 단순히 손실이 줄었다는 사실이 샘플의 품질·다양성을 보장하지 않는다. 여기의 MSE를 임의의 확산 ELBO와 정확히 같은 목적이라고 부를 수도 없다.

## 4. 실제 학습 설정

함수 기본값은 epochs=5,000이지만 main은 3,000을 전달한다. 매번 2,000개 전체 데이터에 새 노이즈를 추가해 Adam, 학습률 0.001로 한 번 갱신한다. 미니배치 순회가 아니라 전체 배치 갱신 3,000회다. 검증 분할, 체크포인트, 조기 종료와 난수 시드 고정은 없다.

표준편차 σ는 양수여야 한다. 작은 σ에서는 목표 스코어의 크기가 커지고, 큰 σ에서는 데이터 모양이 더 평활화된다. 특정 σ가 항상 더 좋은 샘플을 준다고 단정할 수 없으며 데이터의 스케일과 함께 해석한다.

## 5. 랑주뱅 생성과 그림 해석

샘플러는 500×2 표준 정규 초기 상태에서 다음 식을 h=0.05로 1,000번 적용한다.

$$x_{n+1}=x_n+\frac h2s_\theta(x_n)+\sqrt h\,\xi_n$$

no_grad로 그래프를 기록하지 않지만 모델을 eval로 전환하지는 않는다. 현재 구조에는 배치 정규화·드롭아웃이 없어 모드 차이가 없으나 다른 구조를 넣을 때는 확인해야 한다. 상태는 CPU에서 생성하므로 CUDA 모델로 자동 대응하는 함수가 아니다. `noise_scale`은 **초기 상태**의 표준편차이며 반복 잡음의 크기나 DSM의 σ를 바꾸는 인수가 아니다.

기록은 갱신 1,11,…,991회 시점의 100개 스냅샷이며 초기·최종 상태를 포함한 전체 궤적은 아니다. 최종 상태는 별도의 반환값으로 받는다. main의 그림은 원본 데이터, 최종 상태와 학습 손실을 보여 주며 스냅샷은 그리지 않는다.

학습 스코어의 근사 오차, 고정 σ의 평활화와 고정 보폭의 이산화 편향을 구분한다. 출력의 성공 메시지는 별도의 검정 결과가 아니며, 그림만으로 원래 Swiss roll 분포와 정확히 일치한다고 결론 내릴 수 없다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
2,000×2 입력이 기본 ScoreNetwork를 통과할 때 모양을 설명하고 파라미터 수를 구하라.

</div>

??? success "연습문제 1 풀이"

    은닉 출력은 차례로 2000×128 세 개, 마지막 출력은 2000×2다. 파라미터는 첫 층 384, 두 은닉 전이 각각 16,512, 출력층 258로 합계 33,666개다. Softplus에는 학습 파라미터가 없다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
노이즈가 σξ일 때 목표가 −noise/σ²인 이유와 손실의 파라미터 기울기를 적어라.

</div>

??? success "연습문제 2 풀이"

    조건부 관측 밀도는 N(x₀,σ²I)이므로 관측 좌표의 로그 밀도 기울기는 −(x̃−x₀)/σ²=−ξ/σ다. eᵢ=sθ(x̃ᵢ)−targetᵢ라 하면 표본 평균 제곱 노름 손실의 기울기는 (2/N)ΣJθsθ(x̃ᵢ)ᵀeᵢ다. 이 페이지는 스코어를 직접 출력하며 노이즈 예측 파라미터화와는 배율이 다르다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
학습 σ, 샘플러 noise_scale과 step_size가 각각 무엇을 바꾸는가?

</div>

??? success "연습문제 3 풀이"

    σ는 학습의 노이즈 주변 분포와 회귀 목표를 바꾼다. noise_scale은 초기 상태의 분산만 바꾼다. step_size는 반복의 스코어 이동과 가우시안 잡음 분산을 함께 바꾼다. 이 코드에는 시간별 선형·코사인 스케줄이 없으며 학습 후 이 인수들을 바꾸는 것이 다중 노이즈 스코어 학습을 대신하지 않는다.



## 정리하며

고정 σ의 DSM은 계산 가능한 조건부 노이즈 스코어를 회귀해 평활화된 주변 분포의 스코어를 학습한다. 학습 노이즈, 초기 상태 분산과 샘플링 보폭을 구분하며, 손실 감소와 성공 출력만으로 원래 데이터 분포의 정확한 생성을 주장하지 않는다.
