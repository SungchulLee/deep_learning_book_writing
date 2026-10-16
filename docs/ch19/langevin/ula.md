# 비보정 랑주뱅 알고리즘(ULA)

ULA는 Langevin 확산을 MH 보정 없이 이산화한다. 고정된 유한 스텝에서는 원래 목표와 다른 정상분포를 가질 수 있고 불안정한 경우 정상분포 자체가 없을 수도 있다.

## 1. 알고리즘

아래 의사코드는 x′=x+ε∇log π(x)/2+√εη, η~N(0,I)를 사용한다. 뒤의 기본 ula 함수는 x′=x+h∇log π(x)+√(2h)η를 사용하므로 같은 수치 스텝을 비교하려면 ε=2h로 맞춘다.

두 식 모두 Euler–Maruyama 근사이고 수락·거절을 하지 않는다. 스코어에 정규화 상수는 필요 없지만 정의된 적절한 목표와 수치 안정성은 필요하다.

```
Algorithm: Unadjusted Langevin Algorithm (ULA)
──────────────────────────────────────────────
Input: ∇log π̃(θ), step size ε, n_samples T
Initialize: θ₀

For t = 0, 1, ..., T-1:
    η ~ N(0, I)
    θₜ₊₁ = θₜ + (ε/2) ∇log π̃(θₜ) + √ε η
```

## 2. 바이어스 분석

ULA의 이산 바이어스는 스텝·목표의 매끄러움·꼬리·차원과 측정 거리(TV·Wasserstein·함수 기대 오차)에 따라 다르다. O(h)와 특정 상수를 강볼록·Lipschitz 경사 조건만의 보편적인 TV 공식으로 제시하지 않는다. 비선형 Euler는 연속 확산이 안정적이어도 불안정할 수 있다.

총 오차는 유한 시간의 초기 영향, 상관된 표본 평균의 몬테카를로 오차, 고정 스텝의 정상 바이어스를 구분한다. 체인을 길게 실행하는 것은 마지막 바이어스를 없애지 못한다. 스텝을 줄여 같은 물리적 시간·비용에서 결과 민감도를 비교해야 한다.

## 3. 확률 그래디언트 랑주뱅 동역학(SGLD)

전체 로그사후 경사 대신 균등 미니배치의 추정값을 사용한다. 자료 N개, 배치 B의 크기 b에서 ∇log p(θ)+(N/b)∑_{i∈B}∇log p(y_i|θ)가 불편한 경사 추정의 대표 형태다. 사전항을 데이터 배율에 같이 곱하지 않는다.

스텝 감소의 ∑ε_t=∞, ∑ε_t²<∞ 조건은 자주 쓰이지만 이것만으로 모든 목표·확률 경사에서 표본의 사후 수렴을 보장하지 않는다. 목표 안정성·경사 잡음 적률·적절한 가중 평균 등의 조건이 더 필요하다. [SGLD 원 논문](https://icml.cc/2011/papers/398_icmlpaper.pdf)의 설정과 구분한다.

아래 optimizer는 음의 로그목표 경사 g에 −h g+noise_scale√(2h)η를 더한다. 앞 ε/2 관례와는 h=ε/2다. weight_decay는 해당 목적함수에 이차 사전항을 추가하는 효과지만 자료 로스의 배율과 함께 해석해야 한다.

sgld_sample은 MSE의 기본 평균 로스만 사용하고 명시적 사전·전체 데이터 우도 배율을 넣지 않는다. 지정한 관측분산의 베이즈 회귀 사후라고 곧바로 부를 수 없다. 매 반복 dataloader의 첫 배치에서 break하므로 shuffle=False이면 같은 배치만 계속 사용할 수 있다. lr 감소·잡음·thin이라는 이름만으로 올바른 사후 표본이 되는 것은 아니다.

```python
import torch

class SGLD:
    """
    확률 그래디언트 랑주뱅 액션.
    
    작은 배치 그래디언트와 랑주뱅 잡음을 어우러지게 하여
    규모를 키울 수 있는 어림 뒤확률 표집을 이룬다.
    """
    
    def __init__(self, params, lr=1e-3, weight_decay=0.0,
                 noise_scale=1.0, lr_decay=0.0):
        self.params = list(params)
        self.lr_init = lr
        self.weight_decay = weight_decay
        self.noise_scale = noise_scale
        self.lr_decay = lr_decay
        self.step_count = 0
    
    @property
    def lr(self):
        if self.lr_decay > 0:
            return self.lr_init / (1 + self.lr_decay * self.step_count)
        return self.lr_init
    
    def step(self):
        """SGLD 새로 고치기 한 번 하기."""
        eps = self.lr
        
        for p in self.params:
            if p.grad is None:
                continue
            
            # 그래디언트 걸음(자료 가능도와 weight_decay로 넣은 앞확률 포함)
            d_p = p.grad.data
            if self.weight_decay > 0:
                d_p = d_p + self.weight_decay * p.data
            
            # 랑주뱅 잡음
            noise = torch.randn_like(p.data) * (self.noise_scale * (2 * eps) ** 0.5)
            
            # 새로 고치기: θ ← θ - ε∇U(θ) + √(2ε)η
            p.data.add_(-eps * d_p + noise)
        
        self.step_count += 1
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

def sgld_sample(model, dataloader, n_samples=100, burnin=1000,
                thin=10, lr=1e-4):
    """
    SGLD로 뉴럴 넷트웍의 뒤확률 표본 모으기.
    """
    optimizer = SGLD(model.parameters(), lr=lr, lr_decay=1e-5)
    
    samples = []
    total_steps = burnin + n_samples * thin
    
    for step in range(total_steps):
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            break  # 걸음마다 배치 하나
        
        if step >= burnin and (step - burnin) % thin == 0:
            samples.append({
                name: p.data.clone() 
                for name, p in model.named_parameters()
            })
    
    return samples
```

## 4. ULA, MALA, HMC 비교

고정 스텝 ULA/SGLD에는 일반적으로 목표 근사 바이어스가 남는다. MALA·HMC는 정확한 목표와 올바른 제안·MH 보정이 있을 때 원래 목표를 불변으로 갖는다. 이 경우 정상 바이어스를 각각 O(ε³)·O(ε^L)로 적는 것은 틀리다. 수치 제안 오차와 보정 후 목표 불변성을 구분한다.

MALA/HMC도 유한 실행의 초기 영향·상관·구현·탐색 오차는 남는다. 확률 경사를 사용할 때는 제안 생성과 실제 MH 비를 일관되게 구성해야 하며 단순히 모든 평가를 미니배치 값으로 바꾼다고 정확해지지 않는다.

## 5. ULA/SGLD를 언제 쓰나

자료가 크고 정확한 전체 경사 비용이 크면 SGLD의 근사 추론을 검토할 수 있다. 필요한 것은 단순히 SGD에 잡음을 더하는 것보다 우도·사전·로스 배율과 잡음 스케일을 맞추는 일이다.

ULA·SGLD·보정된 MCMC 선택은 허용 바이어스와 함수별 오차·총비용에 따라 판단한다. 자료가 크다는 이유로 SGLD가 최소 조정으로 항상 적절하거나 자료가 작다는 이유로 HMC가 항상 우수한 것은 아니다.

## 6. 표준정규의 정확한 이산 바이어스

표준정규 스코어 s(x)=−x에서 h 관례의 ULA는 X_{k+1}=(1−h)X_k+√(2h)η다. 안정 범위 0<h<2의 정상 AR(1) 분산은 다음과 같다.

$$v_h=\frac{2h}{1-(1-h)^2}=\frac1{1-h/2}.$$

h=0.1이면 약 1.0526이다. 초기 평균은 (1−h)^k배, 초기분산 v₀의 영향은 v_h+(v₀−v_h)(1−h)^{2k}로 변한다. 긴 실행은 v_h로 접근할 뿐 원래 1로 가지 않는다.

앞 ε/2 관례에서는 0<ε<4, 정상분산 1/(1−ε/4)로 바꿔 써야 한다. h가 1보다 크고 2보다 작으면 평균 자기상관 계수는 음수지만 과정은 안정적일 수 있다.

## 7. 기본 ULA 구현

아래 함수는 h=epsilon 관례의 x+h score+√(2h) noise를 반복하고 마지막 상태만 반환한다. x0가 배치면 여러 체인의 마지막 상태이며 n_steps개 시점의 표본 전체가 아니다.

양의 스텝 크기·호환되는 스코어 형태·유한 입력을 확인한다. score_fn이 자동미분 그래프를 만들면 반환 상태를 반복 연결할 수 있으므로 샘플링과 학습의 그래프 관리도 구분해야 한다.

```python
import torch

def ula(score_fn, x0, n_steps, epsilon):
    """다듬지 않은 랑주뱅 알고리즘.

    인수:
        score_fn: x [batch, dim] → 점수 [batch, dim]으로 잇는다.
        x0: 첫 상태 [batch, dim].
        n_steps: 되풀이 횟수.
        epsilon: 걸음 크기.

    반환값:
        되풀이 n_steps번 뒤의 표본 [batch, dim].
    """
    x = x0.clone()
    sqrt_2eps = (2 * epsilon) ** 0.5
    for _ in range(n_steps):
        x = x + epsilon * score_fn(x) + sqrt_2eps * torch.randn_like(x)
    return x
```

## 8. 진단과 스텝 민감도

트레이스·R-hat·함수별 ESS는 실제 이산 체인의 탐색·정밀도를 진단할 수 있지만 원래 연속 목표에 대한 이산 바이어스를 검출하지 못할 수 있다. 여러 체인이 같은 잘못된 정상분포에 잘 섞일 수도 있다.

스텝 민감도와 알려진 모멘트, 독립적인 기준 방법을 비교하고 물리적 적분시간과 비용도 맞춘다. 스텝이 작다는 표현만으로 허용 바이어스가 입증되지는 않는다.

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
ULA에 수락확률이 없는 것이 어떤 바이어스를 남기는가?

</div>

??? success "연습문제 1 풀이"
    이산 전이의 정상분포가 원래 목표와 다를 수 있다. 체인을 오래 돌리거나 높은 ESS를 얻어도 고정 스텝 바이어스는 남는다. 불안정하면 정상분포가 없을 수도 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
표준정규에서 h=0.1의 정상분산과 ε/2 관례의 대응값을 구하라.

</div>

??? success "연습문제 2 풀이"
    분산은 1/(1−0.05)≈1.0526이고 같은 전이는 ε=0.2에 해당한다. 두 관례에 같은 숫자 0.1을 넣으면 다른 전이가 된다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
SGLD에서 평균 MSE와 임의 잡음만으로 특정 베이즈 사후를 얻는가?

</div>

??? success "연습문제 3 풀이"
    우도 분산·전체 자료 배율·사전과 잡음 온도를 일관되게 맞춰야 한다. 불편한 전체 로그사후 경사와 안정성 조건 없이 optimizer 이름만으로 목표가 결정되지 않는다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
ULA의 R-hat과 ESS가 좋아도 필요한 추가 확인은 무엇인가?

</div>

??? success "연습문제 4 풀이"
    원래 목표에 대한 이산 바이어스를 보기 위해 스텝을 줄인 비교·해석적 모멘트·독립 기준 방법을 사용한다. 이산 체인의 정상성 진단과 목표 근사 오차는 다르다.

## 정리하며

ULA/SGLD에서는 이산 바이어스·초기 영향·몬테카를로 오차를 나누어 평가한다. 두 스텝 관례와 로스·사전·잡음 스케일을 일관되게 맞추며, MH 보정 후의 목표 불변성을 수치 제안 차수와 혼동하지 않는다.

**참고 문헌**

- Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
- Dalalyan, A. S. (2017). Theoretical guarantees for approximate sampling from a smooth and log-concave density. *JRSS-B*, 79(3), 651-676.
- Chen, T., Fox, E., & Guestrin, C. (2014). Stochastic gradient Hamiltonian Monte Carlo. *ICML*.

---
