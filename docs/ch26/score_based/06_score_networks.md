# 점수 신경망

이 단원은 요즘 만들어 내는 모델의 핵심 부품인 점수 신경망을 짠다. 여기서 보이는 개념과 재주를 알면 퍼짐 모델과 점수 바탕 만들어 내는 방법을 다루는 데 꼭 필요한 앎을 얻는다. 이 짜기는 또렷함과 실제 쓸모의 균형을 맞추어 배우기에도 실험하기에도 알맞다.

## 코드

```python
"""
FILE: 06_score_networks.py
어려움: 중간
걸리는 시간: 2~3시간
미리 알 것: 04_denoising_score_matching.py

학습 목표:
    1. 잘 듣는 점수 신경망 얼개를 짠다
    2. 잡음 조건 점수 신경망을 짠다
    3. 고르게 맞추기와 깨움 고르기를 이해한다
    4. 점수 나타내기의 가장 좋은 방식을 배운다

수학 바탕:
    점수 신경망은 ∇log p(x)을 나타내는 벡터 마당을 내놓아야 한다.
    
    설계에서 살필 고갱이:
    1. 마지막 활성화가 없다(내놓음이 어떤 실수든 될 수 있다)
    2. 흔히 잡음 수준 σ을 조건으로 삼는다
    3. 안정을 위해 립시츠 이어짐이어야 한다
    4. 복잡한 분포에는 담이를 크게 한다
"""

import torch
import torch.nn as nn

# ========================================================================
# 메인
# ========================================================================


class NoiseConditionalScoreNetwork(nn.Module):
    """
    잡음 수준을 조건으로 삼는 점수 신경망.
    
    얼개: s_θ(x, σ)이 분포 p_σ(x)의 점수를 내놓는다.
    이는 여러 잣수의 점수 나타내기에 결정적이다.
    """
    
    def __init__(self, data_dim=2, hidden_dims=[128, 128, 128],
                 sigma_encoding_dim=32):
        super().__init__()
        
        # 잡음 수준 부호화(사인 꼴 자리 부호화)
        self.sigma_encoding_dim = sigma_encoding_dim
        
        # 으뜸 신경망
        layers = []
        input_dim = data_dim + sigma_encoding_dim
        
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, h_dim),
                nn.GroupNorm(min(8, h_dim//8), h_dim),  # 고르게 맞추기
                nn.SiLU(),  # 매끄러운 깨움
            ])
            input_dim = h_dim
        
        layers.append(nn.Linear(input_dim, data_dim))
        self.network = nn.Sequential(*layers)
    
    def sigma_embedding(self, sigma):
        """
        잡음 수준의 사인 꼴 박아 넣기.
        
        변환기의 자리 부호화와 비슷하다.
        """
        half_dim = self.sigma_encoding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=sigma.device) * -emb)
        emb = sigma[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb
    
    def forward(self, x, sigma):
        """
        잡음 수준을 조건으로 삼은 앞먹임.
        
        인수:
            x: 자료 점, 꼴 (N, D)
            sigma: 잡음 수준, 꼴 (N,)이거나 낱값
        
        반환값:
            score: 점수 벡터, 꼴 (N, D)
        """
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor([sigma] * len(x), device=x.device)
        elif sigma.dim() == 0:
            sigma = sigma.repeat(len(x))
        
        # 잡음 수준을 부호로 담는다
        sigma_emb = self.sigma_embedding(sigma)
        
        # 자료와 이어 붙인다
        x_cond = torch.cat([x, sigma_emb], dim=-1)
        
        # 그물에 통과시키기
        return self.network(x_cond)


class ResidualBlock(nn.Module):
    """더 깊은 점수 신경망을 위한 남은 덩이."""
    
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GroupNorm(8, dim),
        )
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class DeepScoreNetwork(nn.Module):
    """남은 이음을 갖춘 깊은 점수 신경망."""
    
    def __init__(self, data_dim=2, hidden_dim=128, n_blocks=4):
        super().__init__()
        
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, data_dim)
    
    def forward(self, x):
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


def demo_architectures():
    """여러 점수 신경망 얼개를 견준다."""
    print("Score Network Architectures Demo")
    print("=" * 80)
    
    x = torch.randn(32, 2)
    
    # 기본 신경망을 시험한다
    basic_net = nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 2)
    )
    
    # 잡음 조건 신경망을 시험한다
    cond_net = NoiseConditionalScoreNetwork(data_dim=2)
    
    # 깊은 신경망을 시험한다
    deep_net = DeepScoreNetwork(data_dim=2, hidden_dim=64, n_blocks=3)
    
    print("\nArchitecture comparison:")
    print(f"Basic network parameters: {sum(p.numel() for p in basic_net.parameters()):,}")
    print(f"Conditional network parameters: {sum(p.numel() for p in cond_net.parameters()):,}")
    print(f"Deep network parameters: {sum(p.numel() for p in deep_net.parameters()):,}")
    
    # 순전파 시험
    basic_out = basic_net(x)
    cond_out = cond_net(x, sigma=0.5)
    deep_out = deep_net(x)
    
    print(f"\nOutput shapes:")
    print(f"Basic: {basic_out.shape}")
    print(f"Conditional: {cond_out.shape}")
    print(f"Deep: {deep_out.shape}")
    
    print("\n✓ All architectures tested successfully!")


if __name__ == "__main__":
    demo_architectures()```

## 논의

점수 신경망의 짜기는 이 마당에 자리 잡은 방식을 따른다. 코드 짜임이 모델 뜻매김과 익히기 논리를 갈라 놓아 부품을 하나씩 고치기 쉽다. 얼개 고르기는 만들어 내는 모델 무리가 많은 실험에서 얻은 배움을 담고 있다.

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
