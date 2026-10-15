# 스코어 네트워크의 조건 입력과 잔차 구조

스코어 네트워크는 데이터와 같은 차원의 벡터를 출력한다. 여러 노이즈 수준을 다루려면 조건 입력을 추가할 수 있고, 깊이를 늘릴 때는 잔차 연결을 사용할 수 있다. 이 페이지는 세 MLP의 파라미터 수와 출력 모양을 비교하는 구조 예제이며 학습·생성 품질을 비교하는 실험은 아니다.

## 1. 코드와 기존 출력

```python
"""
FILE: 06_score_networks.py
어려움: 중간
걸리는 시간: 2~3시간
미리 알 것: 04_denoising_score_matching.py

학습 목표:
    1. 잘 듣는 점수 신경망 구조를 짠다
    2. 잡음 조건 점수 신경망을 짠다
    3. 고르게 맞추기와 깨움 고르기를 이해한다
    4. 점수 나타내기의 가장 좋은 방식을 배운다

수학 바탕:
    점수 신경망은 ∇log p(x)을 나타내는 벡터 마당을 내놓아야 한다.
    
    설계에서 살필 핵심:
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
    
    구조: s_θ(x, σ)이 분포 p_σ(x)의 점수를 내놓는다.
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
        잡음 수준의 사인 꼴 임베딩.
        
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
    """여러 점수 신경망 구조를 견준다."""
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
    demo_architectures()
```

```
Score Network Architectures Demo
================================================================================

Architecture comparison:
Basic network parameters: 4,482
Conditional network parameters: 38,530
Deep network parameters: 26,050

Output shapes:
Basic: torch.Size([32, 2])
Conditional: torch.Size([32, 2])
Deep: torch.Size([32, 2])

✓ All architectures tested successfully!
```

## 2. 기본 MLP와 노이즈 조건 MLP

기본 네트워크는 2→64→64→2이며 ReLU를 사용한다. 파라미터는 4,482개다. `NoiseConditionalScoreNetwork`는 σ의 사인·코사인 임베딩을 데이터에 연결한 뒤 기본 은닉 폭 128 세 개를 거친다. 2차원 데이터와 32차원 임베딩에서는 입력 폭 34, 출력 폭 2이고 파라미터는 38,530개다.

조건 임베딩은 **σ 자체**에 주파수를 곱한다. 앞 페이지의 목록 인덱스나 log σ 임베딩과 같은 대상이 아니다. σ=0.5라는 호출은 모든 배치 원소에 같은 조건을 복제한다. 길이 B의 텐서를 전달할 수도 있지만 그 장치와 배치 크기는 x와 맞아야 한다. 이미 텐서인 σ를 x의 장치로 자동 이동시키지는 않는다.

주파수 계산의 분모는 half_dim−1이다. 이 구현은 sigma_encoding_dim이 **4 이상인 짝수**라는 조건을 필요로 한다. 홀수면 실제 임베딩 폭이 지정값보다 작고, 폭 2이면 분모가 0이 된다. 임베딩을 계산할 수 있다는 것과 스코어 학습의 표준편차로 유효하다는 것은 다르며, DSM에 쓰는 σ는 양수여야 한다.

## 3. GroupNorm과 폭의 조건

조건 MLP는 각 선형층 뒤에 GroupNorm과 SiLU를 적용한다. B×H 입력에서는 각 표본의 H개 특징을 그룹으로 나누어 정규화하며, 배치 통계를 누적하는 BatchNorm과 다르다. 그렇다고 정규화가 목표 스코어를 자동으로 만들어 주는 것은 아니다.

그룹 수를 min(8,H//8)로 정하므로 H<8이면 그룹 수가 0이 되고, 더 큰 H에서도 그룹 수가 H의 약수여야 한다. 예를 들어 H=70에서는 8개 그룹으로 나눌 수 없다. 기본 폭 128은 이 조건을 만족한다. 폭을 바꿀 때는 Linear 차원뿐 아니라 정규화의 나눗셈 조건도 확인한다.

## 4. 잔차 네트워크

`ResidualBlock`은 Linear→GroupNorm→SiLU→Dropout→Linear→GroupNorm 뒤에 입력을 더하고 SiLU를 적용한다. 입력과 출력 폭이 같아야 하며 GroupNorm(8,dim)은 폭이 8의 배수여야 한다. 잔차 연결을 둔다고 기울기 문제나 학습 실패가 모두 사라지는 것은 아니다.

`DeepScoreNetwork`는 데이터 투영, 잔차 블록들, 출력 투영으로 구성된다. **σ나 시간 조건은 받지 않는다**. 따라서 기본 형태는 하나의 목표 스코어를 나타내며 조건 MLP를 단순히 깊게 만든 완전한 대체물은 아니다.

데모는 hidden_dim=64와 블록 3개를 사용해 파라미터 26,050개를 갖는다. 클래스 기본값인 폭 128·블록 4개는 데모 설정과 다르다. Dropout 확률은 기본 0.1이고 데모는 모델을 eval로 바꾸지 않으므로 같은 입력의 출력도 호출마다 달라질 수 있다. 스코어를 샘플링에 사용하는 추론에서는 모드를 명확히 정해야 한다.

## 5. 출력 모양과 검증의 범위

데모의 입력은 32×2이고 세 출력도 모두 32×2다. 이는 코드가 해당 모양의 입력을 처리한다는 검사일 뿐 올바른 스코어를 학습했음을 보여 주지 않는다. 학습 데이터, DSM·SSM 손실, 최적화 루프와 노이즈 스케줄은 없다.

스코어 추정에서는 출력 차원 외에도 대상 분포·노이즈 조건, 입력 미분의 존재와 계산비용, 모델 모드와 장치를 확인한다. 임의의 벡터장이 항상 어떤 정규화 가능한 밀도의 기울기인 것도 아니다. 구조 선택의 효과는 같은 학습·평가 조건에서 별도로 확인해야 한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
32×2 입력이 조건 MLP를 통과할 때 각 주요 모양을 설명하라.

</div>

??? success "연습문제 1 풀이"

    σ 임베딩은 32×32이고 입력과 연결하면 32×34다. 은닉층 3개는 각각 32×128, 마지막 출력은 32×2다. 길이 32의 σ 벡터를 쓰면 입력과 같은 장치에 있어야 한다.



---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
조건 MLP의 은닉 폭을 70, 임베딩 폭을 33으로 각각 바꾸면 어떤 문제가 생기는가?

</div>

??? success "연습문제 2 풀이"

    폭 70은 그룹 수 8의 배수가 아니라 GroupNorm을 구성할 수 없다. 임베딩 폭 33은 half_dim=16으로 32개 성분만 만들어 첫 Linear가 기대하는 35개 결합 입력과 맞지 않는다. 각각 정규화 그룹과 실제 임베딩 폭의 조건을 함께 확인해야 한다.



---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
세 구조의 출력 모양이 같으면 학습한 분포나 생성 품질도 같은가?

</div>

??? success "연습문제 3 풀이"

    그렇지 않다. 데모는 학습하지 않았고 조건 MLP만 σ를 받는다. 깊은 모델에는 Dropout도 있어 현재 학습 모드에서 출력이 확률적이다. 같은 차원은 필요한 인터페이스 조건일 뿐이며 공통 목적·데이터·조건과 독립 평가가 필요하다.



## 정리하며

노이즈 임베딩과 잔차 연결은 서로 다른 구조 선택이다. 입력·출력 모양뿐 아니라 임베딩 폭, GroupNorm의 나눗셈 조건과 추론 모드를 확인한다. 파라미터 수와 한 번의 순방향 계산은 스코어 정확도나 생성 성능의 검증을 대신하지 않는다.
