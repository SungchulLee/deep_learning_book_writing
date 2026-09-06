# 에너지 바탕 신용 그물 모델



## 학습 목표

이 절을 마치면 다음을 할 수 있다:

1. 볼츠만 기계 얼개로 신용 부도의 서로 매임을 나타낸다
2. 짝별 그리고 더 높은 차수의 부도 얽힘을 담는 에너지 함수를 짠다
3. 에너지 모델에서 깁스 뽑기로 신용 위험 어림을 짠다
4. 체계 위험을 살피는 데 에너지 바탕 이상 찾기를 쓴다

## 들어가며

신용 위험 나타내기는 근본으로 부도 사건 사이의 매임을 이해하는 일이다. 한 회사가 부도 나면 거래 상대, 공급자, 경쟁자에게 영향을 주어 잇달아 무너지는 효과를 낳는데 이는 서로 얽매이지 않는 부도 모델로는 잘 담기지 않는다. 볼츠만 기계의 그물 얼개는 이 문제에 자연스럽게 들어맞는다. 이진 단위가 회사의 부도와 생존 상태를 나타내고 무게 있는 이음이 부도 매임의 세기를 적는다.

## 볼츠만 기계로서의 신용 그물

### 문제의 얼개

회사 $N$개로 이루어진 꾸러미를 살펴보자. 회사 $i$마다 이진 상태를 가진다:

$$s_i = \begin{cases} 1 & \text{if firm } i \text{ defaults} \\ 0 & \text{if firm } i \text{ survives} \end{cases}$$

결합 부도 분포를 볼츠만 분포로 나타낸다:

$$P(\mathbf{s}) = \frac{1}{Z} \exp(-E(\mathbf{s}))$$

### 에너지 함수 짜기

에너지 함수는 세 갈래의 앎을 적는다:

**낱낱의 부도 성향**(치우침 항):

$$E_{\text{individual}}(\mathbf{s}) = -\sum_i \theta_i s_i$$

여기서 $\theta_i$은 회사 $i$ 홀로의 부도 확률을 적는다. $\theta_i$이 더 음수일수록 부도 성향이 높다.

**짝별 부도 매임**(이음 무게):

$$E_{\text{pairwise}}(\mathbf{s}) = -\sum_{i < j} w_{ij} s_i s_j$$

여기서 $w_{ij} > 0$은 회사 $i$과 $j$이 함께 부도 나는 쪽(양의 얽힘)이고 $w_{ij} < 0$은 부도가 서로 갈음되는 쪽(음의 얽힘)임을 뜻한다.

**업종 수준 요인**(숨은 단위):

$$E_{\text{sector}}(\mathbf{s}, \mathbf{h}) = -\sum_{i,k} W_{ik} s_i h_k - \sum_k b_k h_k$$

여기서 숨은 단위 $h_k$은 얽힌 부도를 이끄는 숨은 업종 요인이나 거시 요인을 나타낸다.

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class CreditNetworkEBM(nn.Module):
    """
    신용 부도 나타내기를 위한 볼츠만 기계.
    
    회사 꾸러미의 결합 부도 분포를 나타낸다
    드러난 짝별 매임과 숨은 업종 요인을 갖춘다.
    
    매개변수
    ----------
    n_firms : int
        Number of firms (visible units)
    n_sectors : int
        Number of latent sector factors (hidden units)
    """
    
    def __init__(self, n_firms: int, n_sectors: int = 5):
        super().__init__()
        self.n_firms = n_firms
        self.n_sectors = n_sectors
        
        # 낱낱의 부도 성향
        self.theta = nn.Parameter(torch.randn(n_firms) * 0.1 - 2.0)
        
        # 짝별 부도 매임
        W_vis = torch.randn(n_firms, n_firms) * 0.01
        W_vis = (W_vis + W_vis.t()) / 2
        W_vis.fill_diagonal_(0)
        self.W_visible = nn.Parameter(W_vis)
        
        # 회사와 업종 사이의 이음
        self.W_sector = nn.Parameter(torch.randn(n_firms, n_sectors) * 0.1)
        
        # 업종 치우침
        self.b_sector = nn.Parameter(torch.zeros(n_sectors))
    
    def energy(self, s: torch.Tensor, h: torch.Tensor = None) -> torch.Tensor:
        """
        부도 자리 얽이의 에너지를 셈한다.
        
        매개변수
        ----------
        s : torch.Tensor
            Default states (batch, n_firms), values in {0, 1}
        h : torch.Tensor, optional
            Sector states (batch, n_sectors), values in {0, 1}
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        
        # 낱낱의 항
        E = -torch.einsum('i,bi->b', self.theta, s)
        
        # 짝별 항
        W_sym = (self.W_visible + self.W_visible.t()) / 2
        W_sym.fill_diagonal_(0)
        E -= 0.5 * torch.einsum('bi,ij,bj->b', s, W_sym, s)
        
        # 업종 항
        if h is not None:
            E -= torch.einsum('bi,ij,bj->b', s, self.W_sector, h)
            E -= torch.einsum('j,bj->b', self.b_sector, h)
        
        return E
    
    def free_energy(self, s: torch.Tensor) -> torch.Tensor:
        """
        숨은 업종 단위를 가장자리로 몰아낸 뒤의 자유 에너지.
        
        F(s) = -θᵀs - ½sᵀWs - Σ_k log(1 + exp(Ws_k + b_k))
        """
        if s.dim() == 1:
            s = s.unsqueeze(0)
        
        # 드러난 항
        E_vis = -torch.einsum('i,bi->b', self.theta, s)
        W_sym = (self.W_visible + self.W_visible.t()) / 2
        W_sym.fill_diagonal_(0)
        E_vis -= 0.5 * torch.einsum('bi,ij,bj->b', s, W_sym, s)
        
        # 숨은 항(닫힌 꼴로 가장자리로 몰아냈다)
        activation = s @ self.W_sector + self.b_sector
        E_hid = -torch.log(1 + torch.exp(activation)).sum(dim=1)
        
        return E_vis + E_hid
    
    def conditional_default_prob(self, firm_idx: int, 
                                 s: torch.Tensor) -> torch.Tensor:
        """
        P(s_i = 1 | s_{-i}) via mean-field with marginalized sectors.
        """
        # 다른 회사에서 오는 마당
        W_sym = (self.W_visible + self.W_visible.t()) / 2
        field = self.theta[firm_idx] + (W_sym[firm_idx] * s).sum(dim=-1)
        
        # 업종 단위에서 오는 마당(평균 마당 어림)
        sector_activation = s @ self.W_sector + self.b_sector
        sector_means = torch.sigmoid(sector_activation)
        field += (self.W_sector[firm_idx] * sector_means).sum(dim=-1)
        
        return torch.sigmoid(field)
    
    def gibbs_sample(self, n_samples: int = 1000, n_steps: int = 500,
                     initial_state: torch.Tensor = None) -> torch.Tensor:
        """
        깁스 뽑기로 부도 시나리오를 만든다.
        """
        if initial_state is not None:
            s = initial_state.clone()
        else:
            # 서로 얽매이지 않은 부도로 첫자리매김한다
            probs = torch.sigmoid(self.theta)
            s = torch.bernoulli(probs.unsqueeze(0).expand(n_samples, -1))
        
        for _ in range(n_steps):
            # 아무 차례로 회사마다 고친다
            order = torch.randperm(self.n_firms)
            for i in order:
                prob = self.conditional_default_prob(i, s)
                s[:, i] = torch.bernoulli(prob)
        
        return s
    
    def estimate_default_probabilities(self, n_samples: int = 10000) -> dict:
        """
        표본에서 부도 확률과 얽힘을 어림한다.
        """
        samples = self.gibbs_sample(n_samples=n_samples, n_steps=1000)
        
        # 가장자리 부도 확률
        pd = samples.mean(dim=0)
        
        # 부도 얽힘
        corr = torch.corrcoef(samples.t())
        
        # 결합 부도 확률(짝)
        joint_pd = {}
        for i in range(self.n_firms):
            for j in range(i+1, self.n_firms):
                joint_pd[(i,j)] = (samples[:, i] * samples[:, j]).mean().item()
        
        # 기대 부도 개수
        n_defaults = samples.sum(dim=1)
        
        return {
            'marginal_pd': pd.numpy(),
            'correlation': corr.numpy(),
            'joint_pd': joint_pd,
            'expected_defaults': n_defaults.mean().item(),
            'default_std': n_defaults.std().item(),
            'max_defaults_99': np.percentile(n_defaults.numpy(), 99)
        }


def credit_network_demo():
    """
    에너지 바탕 모델로 신용 그물 나타내기를 보여 준다.
    """
    # 작은 신용 꾸러미를 만든다
    n_firms = 20
    n_sectors = 3
    
    model = CreditNetworkEBM(n_firms, n_sectors)
    
    # 뜻있는 매개변수를 둔다
    with torch.no_grad():
        # 부도 성향: 회사 대부분은 부도 확률이 낮다
        model.theta.copy_(torch.tensor(
            [-3.0] * 5 +    # Low risk (PD ~ 5%)
            [-2.0] * 10 +   # Medium risk (PD ~ 12%)
            [-1.0] * 5      # High risk (PD ~ 27%)
        ))
        
        # 업종 배정(회사가 업종으로 뭉친다)
        W_sector = torch.zeros(n_firms, n_sectors)
        W_sector[:7, 0] = 1.0    # Sector 1: firms 0-6
        W_sector[7:14, 1] = 1.0  # Sector 2: firms 7-13
        W_sector[14:, 2] = 1.0   # Sector 3: firms 14-19
        model.W_sector.copy_(W_sector * 0.5)
        
        # 몇몇 짝별 매임(공급 사슬 이음)
        W_vis = torch.zeros(n_firms, n_firms)
        W_vis[0, 7] = W_vis[7, 0] = 0.5   # Cross-sector link
        W_vis[3, 15] = W_vis[15, 3] = 0.3  # Cross-sector link
        model.W_visible.copy_(W_vis)
    
    # 위험 잣대를 어림한다
    print("Estimating default probabilities...")
    stats = model.estimate_default_probabilities(n_samples=20000)
    
    print(f"\nPortfolio Risk Metrics:")
    print(f"  Expected defaults: {stats['expected_defaults']:.2f}")
    print(f"  Default volatility: {stats['default_std']:.2f}")
    print(f"  99th percentile: {stats['max_defaults_99']:.0f}")
    
    print(f"\nMarginal default probabilities:")
    for i, pd in enumerate(stats['marginal_pd']):
        risk = "Low" if i < 5 else ("Medium" if i < 15 else "High")
        print(f"  Firm {i:2d} ({risk:6s}): {pd:.3f}")
    
    # 시각화한다
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 가장자리 부도 확률
    colors = ['green']*5 + ['orange']*10 + ['red']*5
    axes[0].bar(range(n_firms), stats['marginal_pd'], color=colors, alpha=0.7)
    axes[0].set_xlabel('Firm')
    axes[0].set_ylabel('Default Probability')
    axes[0].set_title('Marginal Default Probabilities')
    axes[0].grid(True, alpha=0.3)
    
    # 얽힘 행렬
    im = axes[1].imshow(stats['correlation'], cmap='RdBu_r', 
                        vmin=-1, vmax=1)
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title('Default Correlation Matrix')
    axes[1].set_xlabel('Firm')
    axes[1].set_ylabel('Firm')
    
    # 부도 개수 분포
    samples = model.gibbs_sample(n_samples=10000, n_steps=500)
    n_defaults = samples.sum(dim=1).numpy()
    axes[2].hist(n_defaults, bins=range(int(n_defaults.max())+2), 
                density=True, alpha=0.7, edgecolor='black')
    axes[2].axvline(np.percentile(n_defaults, 99), color='red', 
                   linestyle='--', label='99th percentile')
    axes[2].set_xlabel('Number of Defaults')
    axes[2].set_ylabel('Probability')
    axes[2].set_title('Default Count Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

credit_network_demo()
```

## 체계 위험 살피기

### 체계 위험 지표로서의 에너지

본 시장 상태의 자유 에너지가 자연스러운 체계 위험 지표를 준다:

$$\text{Systemic Risk Score} = F(\mathbf{s}_{\text{current}}) - \mathbb{E}[F(\mathbf{s})]$$

자유 에너지가 기대 수준보다 크게 높으면 지금의 신용 계 상태가 예사롭지 않다는 신호이며 체계 위험이 높아졌음을 가리킬 수 있다.

```python
def systemic_risk_monitor(model, current_state, historical_states):
    """
    자유 에너지의 이상으로 체계 위험을 살핀다.
    
    매개변수
    ----------
    model : CreditNetworkEBM
        익힌 신용 그물
    current_state : torch.Tensor
        지금의 부도와 버팀 지표
    historical_states : torch.Tensor
        맞추기에 쓸 지난 부도 무늬
    """
    with torch.no_grad():
        current_fe = model.free_energy(current_state).item()
        historical_fe = model.free_energy(historical_states).numpy()
    
    # 지난 분포에 견준 Z 점수
    z_score = (current_fe - historical_fe.mean()) / historical_fe.std()
    
    # 백분위
    percentile = (historical_fe < current_fe).mean() * 100
    
    return {
        'free_energy': current_fe,
        'z_score': z_score,
        'percentile': percentile,
        'alert': 'HIGH' if percentile > 95 else ('MEDIUM' if percentile > 80 else 'LOW')
    }
```

### 옮아감 살피기

에너지 함수의 짝별 무게는 부도가 퍼져 나갈 수 있는 통로를 드러낸다:

```python
def analyze_contagion(model, shocked_firm: int, n_samples: int = 5000):
    """
    Analyze default contagion from a specific firm's default.
    
    충격이 있을 때와 없을 때의 부도 확률을 견준다.
    """
    # 바탕선: 조건 없는 부도
    baseline_samples = model.gibbs_sample(n_samples=n_samples)
    baseline_pd = baseline_samples.mean(dim=0)
    
    # 충격 준 뒤: 회사가 부도 났다는 조건을 둔다
    shocked_samples = model.gibbs_sample(n_samples=n_samples)
    shocked_samples[:, shocked_firm] = 1.0
    # 다시 평형에 이르게 한다
    shocked_samples = model.gibbs_sample(
        n_samples=n_samples, n_steps=500,
        initial_state=shocked_samples
    )
    shocked_pd = shocked_samples.mean(dim=0)
    
    # 옮아감 효과
    contagion = shocked_pd - baseline_pd
    
    return {
        'baseline_pd': baseline_pd.numpy(),
        'shocked_pd': shocked_pd.numpy(),
        'contagion_effect': contagion.numpy()
    }
```

## 지난 자료로 익히기

신용 그물은 맞댐 벌어짐으로 지난 부도 관측에서 익힐 수 있다:

```python
def train_credit_network(model, default_history, n_epochs=100, lr=0.01):
    """
    지난 부도 자료로 신용 그물을 익힌다.
    
    매개변수
    ----------
    model : CreditNetworkEBM
        신용 그물 모델
    default_history : torch.Tensor
        본 부도의 이진 행렬 (n_periods, n_firms)
    """
    n_periods = default_history.shape[0]
    
    for epoch in range(n_epochs):
        perm = torch.randperm(n_periods)
        epoch_loss = 0
        
        for i in range(0, n_periods, 32):
            batch = default_history[perm[i:i+32]]
            batch_size = batch.shape[0]
            
            # 양의 국면: 자료의 자유 에너지
            pos_fe = model.free_energy(batch).mean()
            
            # 음의 국면: 모델 표본의 자유 에너지
            neg_samples = model.gibbs_sample(
                n_samples=batch_size, n_steps=50
            )
            neg_fe = model.free_energy(neg_samples).mean()
            
            # 맞댐 벌어짐 기울기
            loss = pos_fe - neg_fe
            
            # 손으로 하는 기울기 걸음(맞춤 뽑기를 쓰기 때문이다)
            loss.backward()
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        param -= lr * param.grad
                        param.grad.zero_()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}: loss = {epoch_loss:.4f}")
```

## 핵심 정리

!!! success "핵심 개념"

    1. 신용 부도 그물은 볼츠만 기계 얼개에 자연스럽게 대응한다. 이진 부도 상태가 드러난 단위이고 업종 요인이 숨은 단위이다
    2. 짝별 무게는 곧바른 부도 매임을 담고 숨은 단위는 숨은 공통 요인을 담는다
    3. 깁스 뽑기는 배운 매임 얼개를 지키면서 얽힌 부도 시나리오를 만든다
    4. 자유 에너지가 자연스러운 체계 위험 지표를 준다. 예사롭지 않은 시장 상태는 자유 에너지가 높다
    5. 옮아감 살피기는 간단하다. 충격을 조건으로 두고 에너지 풍경을 따라 퍼지는 모습을 본다

!!! warning "한계"

    - 성긴 부도 자료로 볼츠만 기계를 익히기는 어렵다. 부도는 드문 사건이다
    - 이진 부도 상태는 단순하게 만든 것이다. 실제 신용 품질은 이어져 바뀐다
    - 이 모델은 평형을 가정하지만 신용 위기는 평형이 아닌 움직임을 담는다
    - 지난 자료에 맞추려면 회사에 걸쳐 넉넉한 부도 관측이 필요하다

## 참고 문헌

- Dai Pra, P., & Tolotti, M. (2009). Heterogeneous credit portfolios and the dynamics of the aggregate losses. *Stochastic Processes and their Applications*.
- Giesecke, K., & Kim, B. (2011). Systemic Risk: What Defaults Are Telling Us. *Management Science*.
- Hinton, G. E., & Sejnowski, T. J. (1986). Learning and relearning in Boltzmann machines. In *Parallel Distributed Processing*.

## 연습문제

1. **업종 얼개**: 숨은 업종 단위가 있을 때와 없을 때 신용 그물의 부도 얽힘 행렬을 견주어라. 숨은 단위가 부도 뭉침 나타내기를 어떻게 낫게 하는가?

2. **버팀 시험**: 업종 하나 전체에 충격을 주고(업종 $k$의 모든 회사를 부도로 둔다) 다른 업종으로 잇달아 번지는 정도를 재는 버팀 시험을 짜라.

3. **신용 부도 스와프 값 매기기**: 바구니 신용 부도 스와프의 결합 부도 확률을 신용 그물로 어림하라. 정규 코퓰러 모델과 견주어라.
