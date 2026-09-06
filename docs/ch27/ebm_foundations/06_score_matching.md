# 점수 맞추기

점수 맞추기는 에너지 바탕 모델을 익히는 데 최대 가능도의 대안을 주며 다룰 수 없는 나눔 함수를 아름답게 피한다. 확률 밀도를 맞추는 대신 점수 맞추기는 고르게 맞추는 상수에 매이지 않는 로그 밀도의 기울기("점수")를 나란히 맞춘다. 점수 함수와의 이 이음이 요즘 퍼짐 모델의 이론 바탕을 이룬다.

## 코드

```python
"""
에너지 바탕 모델 익히기를 위한 점수 맞추기
============================================

Score matching provides an alternative to maximum likelihood for training EBMs,
다룰 수 없는 나눔 함수를 피한다. 점수는 로그 확률의 기울기이다.

학습 목표:
-------------------
1. 점수 함수와 그 성질을 이해한다
2. 점수 맞추기 목표를 짠다
3. 잡음 없애는 점수 맞추기를 배운다
4. 최대 가능도와 견준다
5. 이어진 자료 분포에 쓴다

핵심 개념:
------------
- Score: ∇ₓ log p(x) = -∇ₓ E(x) / T
- Score Matching: min E_p[(∇ₓ log p(x) - ∇ₓ log q(x))²]
- 잡음 없애는 점수 맞추기: 쓸모 있는 어림
- 나눔 함수를 셈할 필요가 없다

걸리는 시간: 90~120분
Prerequisites: Modules 01-05, Calculus (gradients)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ========================================================================
# 메인
# ========================================================================

torch.manual_seed(42)


class EnergyNetwork(nn.Module):
    """점수 맞추기를 위해 에너지를 내놓는 신경망."""
    
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()


def score_matching_loss(energy_net, x):
    """
    점수 맞추기 손실을 셈한다.
    
    L = ½ E[‖∇ₓ E(x)‖² + 2 Tr(∇²ₓ E(x))]
    
    이는 나눔 함수를 셈하지 않아도 되게 한다!
    """
    x = x.requires_grad_(True)
    
    # 에너지를 셈한다
    energy = energy_net(x)
    
    # 점수(에너지의 기울기)를 셈한다
    score = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=x,
        create_graph=True
    )[0]
    
    # 첫 항: ‖∇E‖²
    score_norm = (score ** 2).sum(dim=1).mean()
    
    # 둘째 항: Tr(∇²E) - 헤세의 대각합
    trace = 0
    for i in range(x.shape[1]):
        grad2 = torch.autograd.grad(
            outputs=score[:, i].sum(),
            inputs=x,
            create_graph=True
        )[0]
        trace += grad2[:, i]
    
    trace_term = 2 * trace.mean()
    
    return 0.5 * score_norm + trace_term


def denoising_score_matching_loss(energy_net, x, noise_std=0.1):
    """
    잡음 없애는 점수 맞추기: 온전한 점수 맞추기보다 셈하기 쉽다.
    
    Add noise: x̃ = x + ε, ε ~ N(0, σ²I)
    Loss: E[‖∇ₓ̃ E(x̃) + (x̃ - x)/σ²‖²]
    """
    # 가우스 잡음을 더한다
    noise = torch.randn_like(x) * noise_std
    x_noisy = x + noise
    x_noisy = x_noisy.requires_grad_(True)
    
    # 에너지와 그 기울기를 셈한다
    energy = energy_net(x_noisy)
    score = torch.autograd.grad(
        outputs=energy.sum(),
        inputs=x_noisy,
        create_graph=True
    )[0]
    
    # 목표 점수: -(x̃ - x)/σ²
    target_score = -noise / (noise_std ** 2)
    
    # 헤아린 점수와 목표 점수의 평균 제곱 어긋남
    loss = ((score - target_score) ** 2).sum(dim=1).mean()
    
    return loss


def train_score_matching_2d():
    """점수 맞추기로 2차원 정규 분포 섞기에서 에너지 바탕 모델을 익힌다."""
    print("\n" + "="*70)
    print("SCORE MATCHING TRAINING (2D Example)")
    print("="*70)
    
    # 2차원 정규 분포 섞기를 만든다
    n_samples = 5000
    means = [torch.tensor([-2., -2.]), torch.tensor([2., 2.]), 
             torch.tensor([-2., 2.])]
    
    samples = []
    for _ in range(n_samples):
        mean = means[np.random.choice(len(means))]
        sample = mean + torch.randn(2) * 0.5
        samples.append(sample)
    
    data = torch.stack(samples)
    print(f"Generated {n_samples} samples from 3-component mixture")
    
    # 에너지 신경망을 만든다
    energy_net = EnergyNetwork(input_dim=2, hidden_dim=64)
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=0.001)
    
    # 학습
    n_epochs = 1000
    batch_size = 128
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    
    for epoch in range(n_epochs):
        epoch_loss = 0
        for (batch_x,) in loader:
            optimizer.zero_grad()
            
            # 잡음 없애는 점수 맞추기를 쓴다(셈하기 쉽다)
            loss = denoising_score_matching_loss(energy_net, batch_x, noise_std=0.5)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(loader))
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {losses[-1]:.4f}")
    
    print("\n✓ Score matching training complete")
    return energy_net


def main():
    print("="*70)
    print("SCORE MATCHING FOR ENERGY-BASED MODELS")
    print("="*70)
    
    train_score_matching_2d()
    
    print("\n" + "="*70)
    print("MODULE COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("  ✓ Score matching avoids partition function")
    print("  ✓ Denoising score matching is practical")
    print("  ✓ Connection to diffusion models")
    print("\nNext: 07_neural_ebms.py")


if __name__ == "__main__":
    main()
```

## 논의

점수 함수 $s(x) = \nabla_x \log p(x)$은 로그 확률이 가장 빠르게 커지는 방향을 담는다. $p(x) = \exp(-E(x))/Z$인 에너지 바탕 모델에서 점수는 그저 $-\nabla_x E(x)$이며 나눔 함수 $Z$이 들어 있지 않다. 점수 맞추기는 이를 써서 자료 점수와 모델 점수 사이의 피셔 벌어짐을 가장 작게 하여 다룰 수 없는 고르게 맞추기를 비켜 가는 익히기 목표를 얻는다.

드러난 점수 맞추기 목표 $\frac{1}{2}\mathbb{E}_p[\|\nabla_x E(x)\|^2 + 2\,\text{Tr}(\nabla_x^2 E(x))]$에는 헤세의 대각합을 셈해야 하는데 차원이 높은 자료에서는 비쌀 수 있다. 잡음 없애는 점수 맞추기가 쓸 만한 대안을 준다. 자료에 알려진 정규 잡음을 더하면 잡음 낀 분포의 가장 좋은 점수를 닫힌 꼴 $-({\tilde{x} - x})/{\sigma^2}$으로 적을 수 있다. 이 목표 점수를 헤아리도록 모델을 익히는 것은 상수만큼의 차이를 빼면 점수 맞추기와 같으며 이계 미분을 아예 셈하지 않아도 된다.

점수 맞추기와 퍼짐 모델의 이음은 깊고 바탕이 된다. 여러 잡음 수준의 잡음 없애는 점수 맞추기가 점수 바탕 만들어 내는 모델(송과 에르몬, 2019)의 바탕을 이루었고 이는 뒤에 잡음 없애는 퍼짐 확률 모델로 자라났다. 잡음 수준을 차츰 낮추며 점수를 배워, 이 모델들은 랑주뱅 움직임이나 확률 흐름 상미분 방정식으로 배운 점수 마당을 따라 잡음에서 자료로 가며 표본을 만들 수 있다.

## 연습문제

**연습문제 1.**
1차원 에너지 함수 $E(x) = \frac{1}{2}(x - \mu)^2 / \sigma^2$에서 점수 함수를 이끌어 내고, 모델이 자료 분포 $\mathcal{N}(\mu, \sigma^2)$과 맞을 때 점수 맞추기 손실이 0임을 확인하라.

??? success "연습문제 1 풀이"
    점수는 다음과 같다:
    
    $$
    s(x) = -\nabla_x E(x) = -\frac{x - \mu}{\sigma^2}
    $$
    
    자료 분포 $p(x) = \mathcal{N}(\mu, \sigma^2)$의 자료 점수는 다음과 같다:
    
    $$
    \nabla_x \log p(x) = -\frac{x - \mu}{\sigma^2}
    $$
    
    점수 맞추기 손실은 $\frac{1}{2}\mathbb{E}_p[\|s_\theta(x) - s_{\text{data}}(x)\|^2]$이다. 모델이 자료와 맞으면 어디서나 $s_\theta(x) = s_{\text{data}}(x)$이므로 손실은 0이다.

---

**연습문제 2.**
잡음 수준 $\sigma$의 잡음 없애는 점수 맞추기가 왜 참 자료 분포가 아니라 매끄럽게 한 분포 $p_\sigma(x) = \int p(y) \mathcal{N}(x; y, \sigma^2 I)\, dy$의 점수를 배우는지 밝혀라. $\sigma \to 0$이면 어떻게 되는가?

??? success "연습문제 2 풀이"
    잡음 없애는 점수 맞추기는 $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$인 $\tilde{x} = x + \epsilon$에서 $-(\tilde{x} - x)/\sigma^2$을 헤아리도록 모델을 익힌다. 이 일의 가장 좋은 헤아리개는 잡음 낀 분포의 점수 $\nabla_{\tilde{x}} \log p_\sigma(\tilde{x})$이며, 여기서 $p_\sigma$은 자료 분포를 정규 알맹이와 겹쳐 만 것이다.
    
    $\sigma \to 0$이면 잡음 낀 분포 $p_\sigma$이 참 자료 분포 $p$으로 모이므로 배운 점수도 참 자료 점수로 모인다. 그러나 $\sigma$이 아주 작으면 기울기의 크기가 아주 커져($1/\sigma^2$으로 자란다) 가장 좋게 하기가 어려워진다. 이것이 여러 잡음 수준에서 모델을 한꺼번에 익히는 여러 잣수 방식의 까닭이 된다.

---

**연습문제 3.**
헤세의 대각합을 아무 쏘기를 쓴 확률 어림 $\mathbb{E}_v[v^\top \nabla_x^2 E(x) v + \|v^\top \nabla_x E(x)\|^2]$($v$은 아무 방향)으로 바꾸는 저민 점수 맞추기를 짜라. 그 셈 비용을 드러난 점수 맞추기 및 잡음 없애는 점수 맞추기와 견주어라.

??? success "연습문제 3 풀이"
    ```python
    def sliced_score_matching_loss(energy_net, x, n_projections=1):
        x = x.requires_grad_(True)
        energy = energy_net(x)
        score = torch.autograd.grad(
            energy.sum(), x, create_graph=True
        )[0]
        
        loss = 0
        for _ in range(n_projections):
            v = torch.randn_like(x)
            v = v / v.norm(dim=1, keepdim=True)
            
            # v^T 점수
            vT_score = (v * score).sum(dim=1)
            
            # v^T 점수의 이계 미분으로 얻은 v^T H v
            grad_vT_score = torch.autograd.grad(
                vT_score.sum(), x, create_graph=True
            )[0]
            vT_Hv = (v * grad_vT_score).sum(dim=1)
            
            loss += (0.5 * vT_score**2 + vT_Hv).mean()
        
        return loss / n_projections
    ```
    
    셈 비용 견주기: 드러난 점수 맞추기는 헤세의 대각합을 위해 뒷먹임을 $d$번(차원마다 한 번) 해야 하므로 앞뒤 한 번보다 $O(d)$배 비싸다. 저민 점수 맞추기는 차원과 상관없이 뒷먹임을 $n_\text{proj}$번만 하므로 차원에 대해 $O(1)$이다. 잡음 없애는 점수 맞추기는 앞뒤 한 번만 하면 되지만 잡음 수준에서 치우침이 생긴다. 저민 점수 맞추기는 치우치지 않은 기울기와 다스릴 수 있는 셈량으로 가운데 길을 준다.
