# 슬라이스드 스코어 매칭의 원리

스코어 매칭의 입력 야코비안 trace는 차원이 크면 계산 부담이 될 수 있다. 슬라이스드 스코어 매칭(SSM)은 임의 방향의 미분을 이용해 전체 야코비안을 명시적으로 만들지 않고 목적을 추정한다. 비편향이라는 말은 특정 목적의 랜덤 투영 추정에 관한 것이며, 유한 데이터로 학습한 스코어가 정확하다는 보장은 아니다.

## 1. trace와 허친슨 추정

모델 스코어 sθ와 데이터 p에 대한 암시적 스코어 매칭 목적은 다음과 같다. 참 스코어의 제곱오차와는 θ에 무관한 상수만큼 다르며 부분적분 경계 조건을 전제로 한다.

$$J(\theta)=\mathbb E_p\left[\frac12\|s_\theta(x)\|^2+
\operatorname{Tr}(J_xs_\theta(x))\right]$$

d개 좌표별 미분으로 trace를 정확히 계산하는 직접 구현은 비쌀 수 있다. 그러나 모든 모델에서 전체 야코비안·헤시안을 반드시 저장해야 하거나 d번 이외의 계산법이 없는 것은 아니다. 구조를 활용한 정확 계산이나 확률적 추정을 구분한다.

v가 고정 행렬 A와 독립이고 $\mathbb E[vv^T]=I$이면 다음 항등식이 성립한다.

$$\mathbb E[v^TAv]=\operatorname{Tr}(A\mathbb E[vv^T])=\operatorname{Tr}(A)$$

A가 대칭일 필요는 없다. 이차형에는 A의 대칭 부분만 기여한다. 독립 좌표의 표준 정규 방향이나 균등 ±1인 라데마허 방향은 조건을 만족한다. 단위 구면 균등 방향은 E[vvᵀ]=I/d이므로 그대로 쓰면 trace의 1/d배다. I에 맞추려면 √d배 해야 한다.

## 2. 투영 목적과 분산

두 항을 모두 같은 방향에 투영하면 다음을 얻는다.

$$J_{SSM}(\theta)=\mathbb E_{x,v}\left[
\frac12(v^Ts_\theta(x))^2+v^T(J_xs_\theta(x))v\right]$$

E[vvᵀ]=I에서 투영 기대값은 앞의 trace 목적과 같다. 제곱 노름을 정확히 계산하고 trace만 확률적으로 추정하는 변형도 가능하지만 아래 코드는 두 항 모두 투영한다. [SSM 논문](https://proceedings.mlr.press/v115/song20a.html)은 이러한 방향 투영과 스코어 추정을 다룬다.

고정 데이터·모델에서 독립 방향 M개를 평균 내고 분산이 유한하면 투영에 의한 조건부 분산은 1/M으로 줄어든다. 데이터 표본 오차나 최적화 오차까지 이 비율로 줄어드는 것은 아니다. 분포별 전체 손실 분산을 비교할 때는 두 항의 공분산도 고려해야 한다.

고정 이차형에서 라데마허 방향은 가우시안 방향의 대각 성분 변동을 제거할 수 있다. 하지만 좌표계, 행렬 구조와 스케일을 무시하고 단위 구면이 언제나 최소 분산이거나 어떤 방향이 모든 학습에 최선이라고 단정하지 않는다.

## 3. 역방향 자동미분

v를 입력과 독립적으로 고정하면 $\nabla_x(v^Ts_\theta)=J_xs_\theta^Tv$다. 이 결과와 v를 내적해 필요한 이차형을 얻는다.

$$v^T\nabla_x(v^Ts_\theta)=v^T(J_xs_\theta)^Tv=v^T(J_xs_\theta)v$$

아래 `autograd.grad`는 역방향 벡터–야코비안 곱(VJP)을 계산한다. 방향당 입력 미분에 더해 목적을 파라미터로 미분하는 역전파도 필요하므로 총 비용을 '역전파 한 번'이라고만 세면 불완전하다. 실제 시간·메모리는 네트워크·차원·방향 수와 그래프 유지 방식에 달려 있다.

## 4. 기본 손실 함수

```python
import torch
import torch.nn as nn
from typing import Literal

def sample_rademacher(shape: tuple, device: torch.device) -> torch.Tensor:
    """라데마허 아무 벡터를 뽑는다(+1이나 -1이 같은 확률)."""
    return torch.randint(0, 2, shape, device=device).float() * 2 - 1

def ssm_loss(
    score_model: nn.Module,
    samples: torch.Tensor,
    n_projections: int = 1,
    projection_type: Literal['rademacher', 'gaussian'] = 'rademacher'
) -> torch.Tensor:
    """
    저민 점수 맞추기 손실.
    
    L_SSM = E_x,v [(v·s(x))²/2 + v·∇_x(v·s(x))]
    
    인수:
        score_model: 점수 신경망 s_θ: R^D → R^D
        samples: 자료 표본, 꼴 (N, D)
        n_projections: 아무 쏘기 횟수 M
        projection_type: 'rademacher'(권함) 또는 'gaussian'
    
    반환값:
        낱값 손실
    """
    samples = samples.requires_grad_(True)
    N, D = samples.shape
    device = samples.device
    
    # 점수를 한 번 셈한다
    scores = score_model(samples)  # (N, D)
    
    total_loss = 0.0
    
    for _ in range(n_projections):
        # 아무 쏘기 벡터를 뽑는다
        if projection_type == 'rademacher':
            v = sample_rademacher((N, D), device)
        else:
            v = torch.randn(N, D, device=device)
        
        # 항 1: (v·s(x))² / 2
        score_proj = torch.sum(v * scores, dim=1)  # (N,)
        squared_term = 0.5 * score_proj ** 2
        
        # 항 2: 벡터-야코비 곱으로 구한 v·∇_x(v·s(x))
        # 낱값 (v·s)의 x에 대한 기울기가 v·∇s을 준다
        vjp = torch.autograd.grad(
            outputs=score_proj.sum(),
            inputs=samples,
            create_graph=True,
            retain_graph=True
        )[0]  # (N, D)
        
        # v·(v·∇s) = v·∇_x(v·s)
        trace_term = torch.sum(vjp * v, dim=1)  # (N,)
        
        total_loss += torch.mean(squared_term + trace_term)
    
    return total_loss / n_projections
```

입력·스코어·방향은 N×d이며 표본별 독립 함수라는 전제를 갖는다. 다른 표본에 의존하는 계층에서는 score_proj.sum의 미분에 교차 표본 항이 섞일 수 있다. samples는 입력 미분이 가능한 텐서여야 하며 requires_grad_는 입력 객체 자체의 설정을 바꾼다.

n_projections는 양수여야 한다. projection_type을 엄밀히 검사하지 않고 rademacher가 아닌 모든 문자열을 가우시안 분기로 처리하므로 오타도 조용히 다른 동작을 만든다. 라데마허·가우시안 방향을 기본 float32로 생성해 모델·입력 dtype에 대한 완전한 일반화도 아니다.

## 5. 메모리 변형과 학습기

```python
def ssm_loss_memory_efficient(
    score_model: nn.Module,
    samples: torch.Tensor,
    n_projections: int = 1
) -> torch.Tensor:
    """
    쏘기마다 점수를 다시 셈해 기억을 아끼는 저민 점수 맞추기.
    GPU 기억이 모자랄 때 쓴다.
    """
    samples = samples.requires_grad_(True)
    N, D = samples.shape
    device = samples.device
    
    total_loss = 0.0
    
    for _ in range(n_projections):
        # 점수를 다시 셈한다(기억 대신 셈을 쓴다)
        scores = score_model(samples)
        
        # 라데마허 쏘기
        v = sample_rademacher((N, D), device)
        
        # 쏜 점수
        score_proj = torch.sum(v * scores, dim=1)
        
        # 저민 점수 맞추기 항
        squared_term = 0.5 * score_proj ** 2
        
        vjp = torch.autograd.grad(
            score_proj.sum(), samples,
            create_graph=True
        )[0]
        trace_term = torch.sum(vjp * v, dim=1)
        
        total_loss += torch.mean(squared_term + trace_term)
        
        # 중간 텐서를 비운다
        del scores, score_proj, vjp
    
    return total_loss / n_projections
```

`ssm_loss_memory_efficient`는 방향마다 순방향을 다시 계산한다. 그러나 각 방향의 미분 그래프가 total_loss에 연결되어 있어 **del로 이름을 지운다고 그 그래프가 바로 해제되는 것은 아니다**. 전체 평균을 한 번 backward하는 동안 필요한 그래프는 유지된다. 이름만으로 메모리 절감을 보장하는 구현이라고 설명할 수 없다. 확률적인 계층이 있으면 방향마다 모델 출력 자체도 달라질 수 있다.

```python
class SlicedScoreMatchingTrainer:
    """저민 점수 맞추기의 온전한 익히개."""
    
    def __init__(
        self,
        score_net: nn.Module,
        lr: float = 1e-3,
        n_projections: int = 1,
        projection_type: str = 'rademacher'
    ):
        self.score_net = score_net
        self.optimizer = torch.optim.Adam(score_net.parameters(), lr=lr)
        self.n_projections = n_projections
        self.projection_type = projection_type
        
    def train_step(self, x: torch.Tensor) -> dict:
        """익히기 걸음 하나."""
        self.score_net.train()
        self.optimizer.zero_grad()
        
        loss = ssm_loss(
            self.score_net, x, 
            self.n_projections, 
            self.projection_type
        )
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def train(
        self, 
        data: torch.Tensor, 
        n_epochs: int = 1000,
        batch_size: int = 256
    ) -> list:
        """온전한 익히기 되풀이."""
        losses = []
        N = len(data)
        
        for epoch in range(n_epochs):
            idx = torch.randperm(N)[:batch_size]
            batch = data[idx]
            
            metrics = self.train_step(batch)
            losses.append(metrics['loss'])
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}: Loss = {metrics['loss']:.6f}")
        
        return losses
```

학습기는 매 반복 무작위 미니배치 하나를 골라 갱신한다. n_epochs라는 이름을 쓰지만 전체 데이터 순회가 아니라 갱신 수다. 입력의 leaf 상태·기존 기울기·장치를 자동으로 모두 처리하지 않으며, 검증·조기 종료·체크포인트·샘플링도 제공하지 않는다.

## 6. 부호 반대 방향은 같은 목적값을 준다

SSM 한 방향의 목적은 v에 대해 짝함수다.

$$\ell(x,-v)=\frac12((-v)^Ts)^2+(-v)^TJ(-v)=\ell(x,v)$$

```python
def ssm_loss_antithetic(score_model, samples, n_projections=1):
    """흩어짐을 줄이려 맞짝 뽑기를 쓴 저민 점수 맞추기."""
    samples = samples.requires_grad_(True)
    scores = score_model(samples)
    
    total_loss = 0.0
    
    for _ in range(n_projections):
        v = sample_rademacher(samples.shape, samples.device)
        
        # 앞 쏘기
        score_proj_pos = torch.sum(v * scores, dim=1)
        vjp_pos = torch.autograd.grad(score_proj_pos.sum(), samples, 
                                       create_graph=True, retain_graph=True)[0]
        loss_pos = 0.5 * score_proj_pos**2 + torch.sum(vjp_pos * v, dim=1)
        
        # 맞짝 쏘기(-v)
        score_proj_neg = torch.sum(-v * scores, dim=1)
        vjp_neg = torch.autograd.grad(score_proj_neg.sum(), samples,
                                       create_graph=True, retain_graph=True)[0]
        loss_neg = 0.5 * score_proj_neg**2 + torch.sum(vjp_neg * (-v), dim=1)
        
        # 평균이 흩어짐을 줄인다
        total_loss += torch.mean((loss_pos + loss_neg) / 2)
    
    return total_loss / n_projections
```

따라서 위 'antithetic' 함수의 (v,−v) 평균은 이론적으로 원래 v 한 개와 **정확히 같은 값**이다. 두 번째 미분 계산을 추가해도 독립 방향을 늘린 효과가 없고 투영 분산을 줄이지 않는다. 부동소수점 차이를 분산 감소로 해석해서는 안 된다.

분산을 줄이려면 실제로 독립적인 방향을 더 사용하거나, 알려진 기대값을 가진 제어변량·목적 변형 등을 올바르게 설계해야 한다. 구체적인 조건과 계산비용을 함께 비교한다.

## 7. 혼합 가우시안 학습 예시

```python
import matplotlib.pyplot as plt
import numpy as np

# 정규 분포 섞기를 만든다
def sample_mog(n, n_components=4):
    """정규 분포 섞기에서 뽑는다."""
    angles = np.linspace(0, 2*np.pi, n_components, endpoint=False)
    centers = 2.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    
    idx = np.random.randint(0, n_components, n)
    samples = centers[idx] + 0.3 * np.random.randn(n, 2)
    return torch.tensor(samples, dtype=torch.float32)

# 단순한 점수 신경망
class ScoreNet(nn.Module):
    def __init__(self, dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 저민 점수 맞추기로 익힌다
data = sample_mog(5000)
model = ScoreNet()
trainer = SlicedScoreMatchingTrainer(model, lr=1e-3, n_projections=4)
losses = trainer.train(data, n_epochs=2000, batch_size=256)

# 배운 점수 마당을 그려 본다
def plot_scores(model, data, title="Learned Score Field"):
    x = torch.linspace(-4, 4, 20)
    y = torch.linspace(-4, 4, 20)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    grid = torch.stack([X.flatten(), Y.flatten()], dim=1)
    
    with torch.no_grad():
        scores = model(grid)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.3, s=1, label='Data')
    plt.quiver(X.numpy(), Y.numpy(), 
               scores[:, 0].reshape(20, 20).numpy(),
               scores[:, 1].reshape(20, 20).numpy(),
               alpha=0.7)
    plt.title(title)
    plt.axis('equal')
    plt.legend()

plot_scores(model, data)
```

원 위의 네 중심 주변에 표준편차 0.3의 가우시안을 배치해 5,000개 2차원 표본을 만든다. 2→128→128→2 SiLU MLP와 투영 네 개를 사용하며, 각 갱신마다 표본 256개를 새로 골라 총 2,000번 학습한다. 전체 데이터 2,000회 순회가 아니다.

그림은 20×20 격자에서 학습한 벡터장을 표시한다. 참 혼합분포 스코어와의 오차나 생성 샘플·일반화는 측정하지 않으며 난수 시드도 고정하지 않는다. 손실이 음수일 수 있는 것은 제거한 상수 때문이고, 더 낮은 경험 손실이 언제나 더 정확한 밀도를 뜻하는 것도 아니다.

## 8. DSM과 비교할 때

DSM은 보통 선택한 노이즈 주변 스코어를 회귀하고 SSM은 주어진 데이터 분포의 스코어 목적을 투영한다. 비교하려면 먼저 같은 목표인지와 데이터·모델·평가 조건을 맞춰야 한다. 고정 σ의 평활화 차이를 모든 분포에서 정확히 O(σ²)라고 단정하거나, 한 방법의 유한 표본 추정이 늘 편향이 없다고 말할 수 없다.

SSM이 관측에 노이즈를 더하지 않는다고 해서 이산 데이터에 일반적인 연속 스코어를 바로 정의할 수 있는 것은 아니다. 연속 표현이나 별도의 이산 방법이 필요할 수 있다. 차원만으로 사용할 방법이나 고정된 실용 한계를 정하지 말고 대상의 정칙성, 계산비용과 검증 결과를 기준으로 선택한다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
단위 구면 방향으로 허친슨 trace를 추정하면 어떤 배율 보정이 필요한가?

</div>

??? success "연습문제 1 풀이"
    단위 방향은 E[vvᵀ]=I/d이므로 이차형의 평균은 trace/d다. 방향을 √d배 하거나 목적값을 d배 해야 trace 배율과 맞는다. 두 항을 함께 같은 양의 상수로 바꾸면 최적점은 같을 수 있지만 손실값·기울기 크기는 달라진다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
투영 수 M을 늘릴 때 어떤 분산이 1/M으로 줄어드는가?

</div>

??? success "연습문제 2 풀이"
    같은 데이터·모델에서 독립이며 유한 분산을 가진 방향 목적을 평균 낼 때의 투영 조건부 분산이다. 데이터의 표본 오차·미니배치 변동·모델 오차 전체가 1/M으로 줄어드는 것은 아니다. 비용도 늘므로 원하는 정확도와 실제 실행비용을 함께 비교한다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
DSM과 SSM의 학습 손실 숫자를 그대로 비교해도 되는가?

</div>

??? success "연습문제 3 풀이"
    안 된다. DSM은 노이즈 수준에 따른 주변 분포와 조건부 분산 상수를 포함하고 SSM은 부분적분으로 상수를 뺀 투영 목적을 사용한다. 같은 평가 분포의 스코어 오차나 적절한 생성 지표로 비교하며 가중치·배율·연산량을 맞춘다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff hard" title="어려움"></span>
SSM에서 (v,−v) 짝을 평균 내면 분산이 줄어드는가?

</div>

??? success "연습문제 4 풀이"
    줄어들지 않는다. 제곱 투영과 야코비안 이차형 모두 v에 대해 짝함수라 두 목적값이 같다. 같은 값을 두 번 계산해 평균 낸 것이므로 독립 방향 두 개와 다르다.

---

<div class="drillbox" markdown>

**연습문제 5.** <span class="diff med" title="중간"></span>
메모리 변형에서 del scores를 호출해도 그래프가 유지될 수 있는 이유는 무엇인가?

</div>

??? success "연습문제 5 풀이"
    각 방향 손실이 total_loss를 통해 최종 backward에 연결되어 필요한 중간 값과 미분 그래프의 참조가 남기 때문이다. Python 변수 이름을 지우는 것과 자동미분 그래프의 수명이 끝나는 것은 다르다. 재계산·개별 역전파 등의 다른 설계는 기울기와 메모리 측정을 함께 검토해야 한다.

## 정리하며

SSM은 적절한 방향 공분산 아래 trace 스코어 매칭 목적을 비편향 투영으로 추정한다. 방향 배율과 VJP 비용, 유지되는 그래프를 확인한다. (v,−v) 짝은 같은 목적값을 주므로 분산 감소가 없으며, 목표 분포와 정칙 조건을 맞춰 DSM과 비교한다.

**참고 문헌**

1. Song, Y., et al. (2019). "Sliced Score Matching: A Scalable Approach to Density and Score Estimation." *UAI*.
2. Hutchinson, M. F. (1989). "A Stochastic Estimator of the Trace of the Influence Matrix for Laplacian Smoothing Splines." *Communications in Statistics*.
3. Song, Y., & Ermon, S. (2019). "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS*.
