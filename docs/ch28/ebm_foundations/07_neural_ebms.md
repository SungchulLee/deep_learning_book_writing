# 신경망 에너지 바탕 모델

요즘 에너지 바탕 모델은 깊은 신경망을 너그러운 에너지 함수로 삼아 자연 그림처럼 복잡하고 차원 높은 분포를 나타낼 수 있게 한다. 이 모델을 익히는 일은 맞댐 배움 목표와 랑주뱅 움직임 뽑기를 합쳐 고전 통계 역학과 요즘 깊은 배움 얼개를 잇는다.

## 1. 코드

```python
"""
신경망 에너지 바탕 모델: 에너지 함수를 위한 깊은 배움
============================================================

요즘 에너지 바탕 모델은 깊은 신경망을 너그러운 에너지 함수로 쓴다.
이 단원은 익히기 재주, 랑주뱅 움직임, 그림 만들어 내기를 다룬다.

걸리는 시간: 120~150분
미리 알 것: 단원 01-06, 깊은 배움 바탕
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ========================================================================
# 메인
# ========================================================================

class ConvEnergyNetwork(nn.Module):
    """그림 에너지를 위한 겹말기 신경망."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # 14x14
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 7x7
            nn.SiLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.SiLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        x = self.conv(x)
        return self.fc(x).squeeze()


def langevin_dynamics(energy_net, x_init, n_steps=100, step_size=0.01, noise_scale=0.005):
    """
    랑주뱅 움직임으로 뽑는다: x ← x - ε∇E(x) + √(2ε)ξ
    """
    x = x_init.clone().requires_grad_(True)
    
    for _ in range(n_steps):
        energy = energy_net(x).sum()
        grad = torch.autograd.grad(energy, x)[0]
        
        noise = torch.randn_like(x) * noise_scale
        x = x.data - step_size * grad + noise
        x = x.requires_grad_(True)
    
    return x.detach()


def train_neural_ebm():
    """MNIST으로 신경망 에너지 바탕 모델을 익힌다."""
    print("Training Neural EBM on MNIST...")
    
    # 데이터를 불러온다
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    # 모델 생성
    energy_net = ConvEnergyNetwork()
    optimizer = torch.optim.Adam(energy_net.parameters(), lr=0.0001)
    
    # 익히기 되풀이(단순하게)
    n_epochs = 2
    for epoch in range(n_epochs):
        for batch_idx, (data, _) in enumerate(loader):
            if batch_idx > 50:  # Quick demo
                break
            
            # 양의 표본(자료)
            pos_energy = energy_net(data).mean()
            
            # 음의 표본(랑주뱅으로)
            neg_samples = torch.rand_like(data)
            neg_samples = langevin_dynamics(energy_net, neg_samples, n_steps=20)
            neg_energy = energy_net(neg_samples).mean()
            
            # 대조 손실
            loss = pos_energy - neg_energy
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    print("✓ Neural EBM training complete")
    return energy_net


def main():
    print("="*70)
    print("NEURAL ENERGY-BASED MODELS")
    print("="*70)
    
    train_neural_ebm()
    
    print("\nKey Takeaways:")
    print("  ✓ Deep networks as flexible energy functions")
    print("  ✓ Langevin dynamics for sampling")
    print("  ✓ Modern EBM architectures")


if __name__ == "__main__":
    main()
```

## 2. 논의

신경망 에너지 바탕 모델은 에너지 함수 $E_\theta(x)$을 깊은 신경망으로 매개변수화하여 복잡한 자료 분포를 나타내는 데 엄청난 너그러움을 준다. 확률 모델은 $p_\theta(x) \propto \exp(-E_\theta(x))$이며 신경망은 들임 $x$(보기로 그림)을 받아 낱값 에너지를 내놓는다. 겹말기 얼개는 공간의 켜와 그 자리의 무늬를 담을 수 있어 그림 자료에 자연스러운 고르기이다.

익히기는 맞댐 벌어짐이나 그와 비슷한 방법으로 나아가며, 실제 자료 표본의 에너지를 낮추고 모델이 만든 표본의 에너지를 높인다. 결정적 어려움은 모델 분포에서 음의 표본을 만드는 것이다. 랑주뱅 움직임이 기울기 바탕 뽑기 절차를 준다: $\xi_t \sim \mathcal{N}(0, I)$일 때 $x_{t+1} = x_t - \epsilon \nabla_x E_\theta(x_t) + \sqrt{2\epsilon}\,\xi_t$이다. 이 마르코프 사슬 몬테카를로 방법은 에너지 기울기를 따라 내려가면서 잡음을 넣어 분포를 제대로 살피게 한다.

실제로 중요한 것은 음의 표본의 품질이다. 짧게 돌린 마르코프 사슬 몬테카를로는 참 모델 분포를 나타내지 못하는 치우친 표본을 만들어 익히기를 흔들리게 한다. 되돌림 곳간(앞의 표본을 담아 두었다가 다시 시작하기), 에너지 신경망의 스펙트럼 고르게 맞추기, 에너지 크기의 규칙 세우기 같은 재주가 익히기를 안정시키는 데 도움이 된다. 이 신경망 에너지 바탕 모델은 더 넓은 은근한 만들어 내는 모델 무리와 이어지며 점수 바탕 퍼짐 모델과 깊이 이어져 있다.

## 연습문제

**연습문제 1.**
랑주뱅 움직임 고침 규칙 $x_{t+1} = x_t - \epsilon \nabla_x E(x_t) + \sqrt{2\epsilon}\,\xi_t$이 주어질 때 $\epsilon \to 0$이고 $T\epsilon \to \infty$이면서 걸음 수 $T \to \infty$이면 $x_T$의 분포가 $p(x) \propto \exp(-E(x))$으로 모임을 보여라.

??? success "연습문제 1 풀이"
    랑주뱅 움직임은 이어진 때의 랑주뱅 확률 미분 방정식 $dx = -\nabla_x E(x)\,dt + \sqrt{2}\,dW_t$을 띄엄띄엄하게 만든 것이다. 이 방정식의 포커-플랑크 식은 정상 분포가 $\nabla \cdot (p \nabla E + \nabla p) = 0$을 만족함을 보이며 이는 $p(x) \propto \exp(-E(x))$으로 풀린다. $\epsilon \to 0$이면 띄엄띄엄한 사슬이 이어진 확률 미분 방정식으로 모이고 $T\epsilon \to \infty$이면 사슬이 정상 분포로 섞일 시간을 얻는다. 잡음 항 $\sqrt{2\epsilon}\,\xi$이 결정적이다. 그것이 없으면 움직임이 온 분포에서 뽑는 대신 $E(x)$의 봉우리 하나로 모인다.

---

**연습문제 2.**
신경망 에너지 바탕 모델 익히기에서 되돌림 곳간의 몫을 밝혀라. 음의 표본 가운데 일부를 아무 잡음이 아니라 앞의 마르코프 사슬 몬테카를로 사슬에서 첫자리매김하는 것이 왜 이로운가?

??? success "연습문제 2 풀이"
    되돌림 곳간은 앞 익히기 되풀이의 음의 표본을 담아 둔다. 새 음의 표본을 만들 때 일부를 아무 잡음 대신 곳간에서 첫자리매김한다. 이는 두 가지 이로움을 준다:
    
    1. **더 빠른 섞임**: 곳간의 표본은 이미 에너지가 낮은 자리에 가까우므로 거기서 시작한 짧은 마르코프 사슬 몬테카를로가 아무 잡음에서 시작한 것보다 모델 분포를 더 잘 어림한다.
    
    2. **익히기 안정**: 되돌림 곳간이 없으면 모델이 잡음에서 짧은 마르코프 사슬 몬테카를로로 닿을 수 있는 자리에만 낮은 에너지를 매기고 다른 낮은 에너지 자리를 지나칠 수 있다. 곳간은 다양함을 지켜 봉우리 무너짐을 막는다.
    
    보통 음의 표본의 95%을 곳간에서, 5%을 아무 잡음에서 뽑는다. 아무 잡음 몫이 모델이 새 자리를 계속 살피게 한다.

---

**연습문제 3.**
`ConvEnergyNetwork`을 모든 층에 스펙트럼 고르게 맞추기를 넣도록 고치고 익히기 손실에 에너지 규칙 세우기 항 $\lambda \cdot (E(x_+)^2 + E(x_-)^2)$을 더하라. 각 고침이 익히기 안정을 어떻게 낫게 하는지 밝혀라.

??? success "연습문제 3 풀이"
    ```python
    from torch.nn.utils import spectral_norm
    
    class StableConvEnergyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                spectral_norm(nn.Conv2d(1, 64, 3, 1, 1)),
                nn.SiLU(),
                spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
                nn.SiLU(),
                spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
                nn.SiLU(),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                spectral_norm(nn.Linear(256 * 7 * 7, 512)),
                nn.SiLU(),
                spectral_norm(nn.Linear(512, 1))
            )
        
        def forward(self, x):
            return self.fc(self.conv(x)).squeeze()
    
    # 고친 손실
    reg_lambda = 0.1
    loss = pos_energy - neg_energy + reg_lambda * (pos_energy**2 + neg_energy**2).mean()
    ```
    
    스펙트럼 고르게 맞추기는 무게 행렬을 가장 큰 특이값으로 나누어 신경망의 립시츠 상수를 가둔다. 이는 에너지가 너무 빨리 커지는 것을 막아 랑주뱅 움직임 뽑기를 안정시킨다(에너지 기울기가 크면 지나치게 된다). 에너지 규칙 세우기 항은 에너지의 절댓값이 큰 것을 벌하여, 양의 에너지와 음의 에너지가 틈은 그대로 둔 채 각각 $-\infty$과 $+\infty$으로 갈라지는 흔한 실패를 막는다.

## 정리하며

**다룬 것** — 신경망 에너지 바탕 모델

신경망 에너지 바탕 모델은 에너지 함수 $E_\theta(x)$을 깊은 신경망으로 매개변수화하여 복잡한 자료 분포를 나타내는 데 엄청난 너그러움을 준다.

고갱이 갈래는 `ConvEnergyNetwork`, `StableConvEnergyNetwork`이며 앞의 연습문제 3개로 스스로 따져 볼 수 있다.
