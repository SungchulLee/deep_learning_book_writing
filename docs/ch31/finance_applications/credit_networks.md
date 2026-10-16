# 에너지 기반 신용 네트워크 예제

같은 기간에 여러 기업이 부도 나는 사건의 의존성을 이진 에너지 모델로 표현해 보자. 기업별 바이어스, 기업 간 결합과 잠재 업종 요인을 구분하면 개별·공동 부도 빈도의 관계를 살펴볼 수 있다. 이 페이지의 값은 수동으로 정한 교육용 모수이며, 실제 부도 데이터로 추정·검증한 위험 결과가 아니다.

## 1. 상태와 결합 에너지

기업 i의 상태 sᵢ=1은 정한 관측 기간의 부도, 0은 생존이다. 모든 기업에 같은 기간·상태 정의가 필요하다. 이산 잠재 요인 hₖ∈{0,1}을 추가하고 온도 1에서 다음 에너지를 사용한다.

$$E(s,h)=-\theta^Ts-\frac12s^TWs-s^TUh-b^Th$$

W는 대칭·대각 0이고 U는 기업과 잠재 요인의 결합이다. 모든 기업·요인 결합이 없으면 P(sᵢ=1)=sigmoid(θᵢ)이므로 **θᵢ가 더 음수일수록 부도 확률은 낮아진다**. 결합이 있으면 sigmoid(θᵢ) 자체가 주변 부도 확률은 아니다.

Wᵢⱼ는 다른 상태와 요인을 고정했을 때의 조건부 연관을 나타낸다. 양의 결합이 공동 활성화를 선호하더라도 W를 그대로 주변 상관계수나 인과적인 전염 강도로 읽을 수는 없다. 잠재 요인 역시 관측 업종의 인과 효과로 자동 식별되지 않는다.

## 2. 코드와 보존된 출력

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
    
    파라메터
    ----------
    n_firms : int
        회사의 수(보이는 알갱이)
    n_sectors : int
        숨은 클래스 요인의 수(숨은 알갱이)
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
        
        파라메터
        ----------
        s : torch.Tensor
            부도 상태 (batch, n_firms), 값은 {0, 1}
        h : torch.Tensor, optional
            클래스 상태 (batch, n_sectors), 값은 {0, 1}
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
        클래스를 주변화한 평균장으로 구한 P(s_i = 1 | s_{-i}).
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
            'marginal_pd': pd.detach().numpy(),
            'correlation': corr.detach().numpy(),
            'joint_pd': joint_pd,
            'expected_defaults': n_defaults.mean().item(),
            'default_std': n_defaults.std().item(),
            'max_defaults_99': np.percentile(n_defaults.detach().numpy(), 99)
        }

def credit_network_demo():
    """
    에너지 바탕 모델로 신용 넷트웍 나타내기를 보여 준다.
    """
    # 작은 신용 꾸러미를 만든다
    n_firms = 20
    n_sectors = 3
    
    model = CreditNetworkEBM(n_firms, n_sectors)
    
    # 뜻있는 파라메터를 둔다
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
    n_defaults = samples.sum(dim=1).detach().numpy()
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

```
Estimating default probabilities...

Portfolio Risk Metrics:
  Expected defaults: 3.62
  Default volatility: 1.71
  99th percentile: 8

Marginal default probabilities:
  Firm  0 (Low   ): 0.069
  Firm  1 (Low   ): 0.062
  Firm  2 (Low   ): 0.061
  Firm  3 (Low   ): 0.072
  Firm  4 (Low   ): 0.060
  Firm  5 (Medium): 0.156
  Firm  6 (Medium): 0.152
  Firm  7 (Medium): 0.157
  Firm  8 (Medium): 0.155
  Firm  9 (Medium): 0.160
  Firm 10 (Medium): 0.152
  Firm 11 (Medium): 0.155
  Firm 12 (Medium): 0.156
  Firm 13 (Medium): 0.152
  Firm 14 (Medium): 0.162
  Firm 15 (High  ): 0.353
  Firm 16 (High  ): 0.342
  Firm 17 (High  ): 0.344
  Firm 18 (High  ): 0.349
  Firm 19 (High  ): 0.348
```

데모는 기업 20개와 잠재 요인 3개를 만들고 바이어스를 −3·−2·−1인 세 집단으로 지정한다. U는 기업을 세 묶음으로 연결하고 일부 W만 수동 지정한다. 이 구분은 모델의 설정이며 실제 기업의 신용등급이나 추정 결과가 아니다.

energy는 기업·요인의 결합 에너지를, free_energy는 h를 합한 가시 자유 에너지를 계산한다.

$$F(s)=-\theta^Ts-\frac12s^TWs
-\sum_k\operatorname{softplus}\left(b_k+\sum_iU_{ik}s_i\right)$$

이론식과 달리 free_energy는 log(1+exp(a))를 직접 계산해 큰 a에서 overflow할 수 있다. 정확한 log p(s)는 −F(s)−log Z이므로 자유 에너지와 정규화한 확률도 구분한다.

## 3. 가시변수 조건부의 정확한 식

은닉변수를 고정하면 기업 단위의 조건부는 sigmoid(θᵢ+ΣⱼWᵢⱼsⱼ+ΣₖUᵢₖhₖ)다. 하지만 h를 주변화한 뒤에는 h를 평균으로 대체한 시그모이드가 정확한 조건부가 아니다.

다른 기업 상태만으로 aₖ=bₖ+Σⱼ≠ᵢUⱼₖsⱼ를 정의하면 정확한 주변 조건부는 다음과 같다.

$$P(s_i=1\mid s_{-i})=\operatorname{sigmoid}\left(
\theta_i+\sum_{j\ne i}W_{ij}s_j+
\sum_k[\operatorname{softplus}(a_k+U_{ik})-\operatorname{softplus}(a_k)]\right)$$

이는 F(sᵢ=0)−F(sᵢ=1)의 로그 오즈에서 나온다. conditional_default_prob는 대신 현재 **sᵢ까지 포함한** 은닉 엑티베이션의 평균을 국소장에 대입한다. 자기 현재값에 조건부가 의존하고 softplus 차이도 사용하지 않으므로 위 목표분포의 정확한 깁스 전이가 아니다.

또한 energy와 free_energy는 W의 대각을 0으로 만들지만 conditional_default_prob는 대각을 제거하지 않는다. 초기 설정은 0이어도 임의 모수를 불러오는 등 다른 사용에서는 이 불일치도 확인해야 한다. 사슬을 오래 돌리는 것만으로 전이와 목표 에너지의 차이가 사라지지는 않는다.

## 4. 반환 통계와 계산 범위

기업별 표본 평균, 쌍별 공동 빈도, 표본 상관, 부도 개수의 평균·표준편차·99% 분위수를 계산한다. 상관은 한 기업이 모든 표본에서 같은 상태이면 분산 0으로 정의되지 않을 수 있다. 출력된 “99th percentile”은 **기업 부도 개수**의 분위수이며 금액 로스의 VaR가 아니다. 부도 시 로스률·익스포저나 시간별 부도 모형은 포함하지 않는다.

n_steps는 MCMC sweep 수이며 실제 날짜가 아니다. 이진 상태는 사슬에서 1에서 0으로도 바뀌므로 실제 기업의 부도와 회복 과정을 시뮬레이션하는 시간 모형으로 읽지 않는다. 처음 생성한 여러 행을 병렬로 갱신하는 방식이며 평형 진단과 전이의 정확성 문제는 별도로 남는다.

샘플링에는 no_grad나 명시적인 detach가 없다. Parameter에서 만든 확률을 반복적으로 사용하고 상태를 제자리 수정하므로 불필요한 계산 그래프가 커지거나 역전파 때 버전 충돌이 생길 수 있다. 나중의 .detach().numpy()는 앞서 그래프를 만든 비용을 없애지 않는다. CPU NumPy 변환과 입력 장치 조건도 맞춰야 한다.

## 5. 자유 에너지 기반 상태 점검

```python
def systemic_risk_monitor(model, current_state, historical_states):
    """
    자유 에너지의 이상으로 체계 위험을 살핀다.
    
    파라메터
    ----------
    model : CreditNetworkEBM
        익힌 신용 넷트웍
    current_state : torch.Tensor
        지금의 부도와 버팀 지표
    historical_states : torch.Tensor
        맞추기에 쓸 지난 부도 무늬
    """
    with torch.no_grad():
        current_fe = model.free_energy(current_state).item()
        historical_fe = model.free_energy(historical_states).detach().numpy()
    
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

systemic_risk_monitor는 현재 F가 역사적 F 분포에서 어디에 있는지 점수화한다. 높은 F는 같은 모델 안에서 낮은 개별 상태 확률을 뜻할 수 있지만, 이것이 부도 개수·예상 금액 로스나 시스템 위험과 단조롭게 대응하지는 않는다. 80·95 분위수 구분도 검증된 위험 등급이 아니라 코드에 정한 규칙이다.

현재 상태 하나를 가정해 .item()을 호출하며 역사 분산이 0이면 z-score가 정의되지 않는다. 분위수와 임계값은 독립적인 검증 구간에서 정하고 시간 변화·관측 기간을 일치시켜야 한다.

## 6. 충격 함수가 조건화를 유지하는가

```python
def analyze_contagion(model, shocked_firm: int, n_samples: int = 5000):
    """
    어떤 회사의 부도에서 비롯한 부도 옮음을 살핀다.
    
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
        'baseline_pd': baseline_pd.detach().numpy(),
        'shocked_pd': shocked_pd.detach().numpy(),
        'contagion_effect': contagion.detach().numpy()
    }
```

analyze_contagion은 한 번 생성한 상태에서 특정 기업을 1로 만든 뒤 다시 모든 기업을 갱신한다. **충격 기업도 다시 샘플링하므로 부도 상태에 고정하지 않는다.** 따라서 고정된 부도 조건의 분포를 구하는 함수가 아니다. 충분히 혼합하는 같은 무조건 전이라면 초기 충격을 잊고 기저분포로 돌아갈 수 있다.

정확한 조건부 비교를 원하면 관측 기업을 계속 고정하고 나머지만 올바른 조건부로 갱신해야 한다. 그래도 통계적 조건화와 인과 개입은 다르며 대칭 결합만으로 전염 방향과 인과 효과가 식별되지는 않는다. 출력 차이를 검증된 전염 로스로 해석하지 않는다.

## 7. 데이터 학습 골격

```python
def train_credit_network(model, default_history, n_epochs=100, lr=0.01):
    """
    지난 부도 자료로 신용 넷트웍을 익힌다.
    
    파라메터
    ----------
    model : CreditNetworkEBM
        신용 넷트웍 모델
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
            
            # 맞댐 벌어짐 그래디언트
            loss = pos_fe - neg_fe
            
            # 손으로 하는 그래디언트 걸음(맞춤 뽑기를 쓰기 때문이다)
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

train_credit_network는 데이터와 생성 상태의 평균 자유 에너지 차이를 역전파하고 수동 SGD를 수행한다. 정확한 주변 모델 표본을 고정해 사용하면 우도 그래디언트를 근사할 수 있지만, 현재 샘플러는 조건부 식이 다르고 그래프를 끊지도 않는다. 따라서 이 코드를 정확한 CD 학습으로 설명하지 않는다.

부도 이력은 같은 기간의 기업별 이진 행렬이어야 하며 누락·관측 기간·기업 집합 변화의 처리가 필요하다. 드문 사건의 의존성은 표본 부족에 민감하다. 실제 신용 예측·가격 산정에는 시간별·조건별 확률과 로스 정의, 별도의 외부 검증이 추가로 필요하며 이 교육용 정적 모형의 출력으로 대신하지 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1. 바이어스의 부호**

결합이 없을 때 θ가 −3,−2,−1로 변하면 부도 확률은 어떻게 변하는가?

</div>

??? success "연습문제 1 풀이"
    확률은 sigmoid(θ)로 약 0.0474, 0.1192, 0.2689로 증가한다. 더 음수일수록 낮다. 결합이 있는 모델의 주변 확률은 θ만으로 정해지지 않는다.

---

<div class="drillbox" markdown>

**연습문제 2. 고정 충격과 인과 효과**

analyze_contagion이 충격 기업의 부도 조건부를 유지하는가? 조건을 유지해도 곧 인과 효과인가?

</div>

??? success "연습문제 2 풀이"
    충격 기업까지 전이에 포함해 고정하지 않는다. 조건부를 얻으려면 해당 좌표를 계속 1로 유지하고 다른 좌표만 정확한 조건부로 갱신해야 한다. 이 결과도 통계적 조건화이며, 인과 개입을 주장하려면 별도의 구조·식별 가정이 필요하다.

---

<div class="drillbox" markdown>

**연습문제 3. 정적 통계와 로스 가격**

부도 개수의 99% 분위수와 기업 쌍의 공동 부도 빈도만으로 바스켓 신용상품 가격을 계산할 수 있는가?

</div>

??? success "연습문제 3 풀이"
    아니다. 금액 로스와 지급·회수 구조, 부도 시점·관측 기간 및 가격 산정에 쓰는 확률 측도·할인 조건 등이 더 필요하다. 이 정적 교육 모형에는 그 요소가 없다. 우선 정확한 소규모 상태 열거와 조건부 전이를 비교해 코드의 분포 일치부터 확인해야 한다.

## 정리하며

정적 신용 네트워크는 기업별 상태와 의존성을 표현하는 교육용 예다. 바이어스의 부호, 정확한 주변 조건부와 코드의 근사 차이, 그래프 처리와 충격 고정 여부를 확인한다. 상태 희귀성·부도 개수와 금액 로스·인과 전염을 구분해야 한다.

**참고 문헌**

- Dai Pra, P., & Tolotti, M. (2009). Heterogeneous credit portfolios and the dynamics of the aggregate losses. *Stochastic Processes and their Applications*.
- Giesecke, K., & Kim, B. (2011). Systemic Risk: What Defaults Are Telling Us. *Management Science*.
- Hinton, G. E., & Sejnowski, T. J. (1986). Learning and relearning in Boltzmann machines. In *Parallel Distributed Processing*.
