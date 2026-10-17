# 스코어 매칭, 랑주뱅 동역학, 확산 모델

스코어 함수는 고정 목표의 Langevin 샘플링과 데이터에서 학습하는 생성모형을 연결한다. 다만 온도 냉각, 가우스 잡음 컨볼루션과 역시간 확산은 서로 다른 구성이다. 각 목표·시간·노이즈 관례를 명시해 관계를 살펴본다.

## 1. 스코어 함수의 역할

상태 x의 스코어는 s(x)=∇_xlog p(x)다. 미정규화 밀도 γ에서 p=γ/Z이면 Z가 x에 의존하지 않아 ∇log p=∇log γ다. 고정 목표 MCMC에서는 보통 이 값을 모형에서 계산하고, 생성모형에서는 잡음 수준별 데이터 스코어를 표본으로 학습한다.

## 2. 왜 점수인가?

양의 매끄러운 밀도가 연결된 영역에 정의되고 스코어가 정확한 보존 벡터장이라면 기준점에서 선적분해 log p의 차이를 복원할 수 있다. 적절한 정규화가 존재해야 밀도가 결정된다. 연결되지 않은 지지집합에서는 성분별 상수가 남아 스코어만으로 성분 간 질량을 정할 수 없다.

학습한 임의 벡터장이 반드시 전역 로그밀도의 경사인 것도 아니다. 정확한 스코어로 dX=s(X)dt+√2dW를 정의하면 적절한 조건에서 p가 불변이지만 유한 스텝·유한 실행·학습 오차까지 정확한 데이터 표본을 보장하지는 않는다.

## 3. 스코어 매칭과 노이즈 제거 회귀

원래 스코어 매칭은 E_p‖s_θ−∇log p‖²/2를 최소화한다. 적분 부분의 경계항이 사라지고 필요한 미분·적률 조건이 맞으면 θ에 무관한 상수를 제외해 E_p[div s_θ+‖s_θ‖²/2]로 쓸 수 있다. 단지 매끄러운 아무 벡터장이라는 조건만으로 이 항등식이 성립하는 것은 아니다.

첫 코드에는 torch import와 x.requires_grad=True, 표본별 미분이 가능한 망이 필요하다. 배치 간 상호작용이 있으면 합을 미분하는 방식이 의도한 개별 표본 Jacobian과 달라질 수 있다.

가우스 오염 X=X₀+σε의 조건부 스코어는 −(X−X₀)/σ²다. 이를 회귀 목표로 쓰는 denoising score matching(DSM)의 모집단 최적 예측은 E[−(X−X₀)/σ²|X]=∇log q_σ(X)다. 각 쌍의 조건부 스코어가 그 점의 주변 스코어와 똑같은 것은 아니다. σ>0, 입력 형태와 노이즈 수준별 로스 가중을 확인한다. 두 번째 코드는 이 조건부 회귀를 계산한다.

```python
def score_matching_loss(score_network, x):
    """
    x: 자료 표본 배치, 꼴 (batch, dim)
    score_network: 점수 s_θ(x)을 내놓는 뉴럴 넷트웍
    """
    # 점수 셈하기
    s = score_network(x)  # 꼴: (batch, dim)
    
    # 발산 셈하기: tr(∇s_θ)
    # 이는 야코비 행렬을 셈해야 한다
    div_s = 0
    for i in range(x.shape[1]):
        div_s += torch.autograd.grad(
            s[:, i].sum(), x, create_graph=True
        )[0][:, i]
    
    # 점수 맞추기 로스
    loss = (div_s + 0.5 * (s ** 2).sum(dim=1)).mean()
    return loss
```
```python
def denoising_score_matching_loss(score_network, x_data, sigma):
    """
    x_data: 깨끗한 자료 표본
    sigma: 잡음 레이어
    """
    # 잡음 더하기
    noise = torch.randn_like(x_data) * sigma
    x_noisy = x_data + noise
    
    # 미리본 점수
    s_pred = score_network(x_noisy, sigma)
    
    # 잡음 섞인 자료의 참 점수
    s_true = -noise / (sigma ** 2)
    
    # 잡음 없애는 점수 맞추기 로스
    loss = 0.5 * ((s_pred - s_true) ** 2).sum(dim=1).mean()
    return loss
```

## 4. 학습 스코어를 이용한 샘플링

학습한 s_θ로 X′=X+h s_θ(X)+√(2h)η를 반복할 수 있다. 정확한 스코어라도 유한 h는 ULA 바이어스를 만들고, 부정확한 스코어는 드리프트 오차를 만든다. 초기화·유한 실행·미탐색 영역도 영향을 준다.

잡음에서 시작해 반복하면 정확히 p_data의 표본이 나온다는 단정은 하지 않는다. 학습 오차와 수치·혼합 오차를 구분하며, 스코어만 있는 경우 원래 목표의 정확한 밀도를 알지 못해 일반적인 MH 보정을 바로 적용하기 어려울 수 있다.

## 5. 여러 노이즈 수준과 annealed Langevin

여러 σ>0에서 q_σ=p_data*N(0,σ²I)의 스코어를 학습하면 성긴 영역을 매끄럽게 만들 수 있다. 샘플링은 보통 큰 σ에서 작은 σ 순서로 수준을 바꾸며 각 수준에서 유한 Langevin 전이를 수행한다.

큰 σ의 분포는 조건에 따라 넓은 가우스에 가까울 수 있지만 R^d의 균등분포가 되지는 않는다. 마지막 σ_min>0의 목표는 여전히 오염된 q_σmin이며 p_data와 정확히 같지 않다. 온도 거듭제곱 p^(1/T)와 컨볼루션 q_σ는 다르므로 일반적인 simulated annealing 최적화와 동일시하지 않는다.

## 6. 확산 모델과의 이음

DDPM의 한 단계 분산 β_t, α_t=1−β_t, 누적 곱 ᾱ_t=∏_{j≤t}α_j를 구분한다. 정방향 주변 오염은 다음과 같다.

$$X_t=\sqrt{\bar\alpha_t}X_0+\sqrt{1-\bar\alpha_t}\,\epsilon.$$

이는 X₀를 먼저 축소하므로 단순 X₀+σε와 같은 오염식이 아니다. 노이즈 예측을 학습한 모집단 최적 망은 스코어 s_θ≈−ε_θ/√(1−ᾱ_t)와 연결된다. 한 단계 역평균의 대표 파라메터화는 다음과 같다.

$$\mu_\theta(X_t,t)=\frac1{\sqrt{\alpha_t}}\left(X_t+\beta_t s_\theta(X_t,t)\right).$$

역분산은 β_t나 β̃_t=β_t(1−ᾱ_{t−1})/(1−ᾱ_t) 등 선택한 변형에 맞춰 정하며 마지막 단계의 잡음 처리도 다르다. 누적 ᾱ와 한 단계 α를 섞으면 역식이 틀린다. 이 역전이는 고정 목표 Langevin과 동일한 알고리즘이 아니다. [DDPM 원 논문](https://arxiv.org/abs/2006.11239)을 참고한다.

## 7. 연결 구조를 읽는 방법

아래 도식은 스코어를 중심으로 여러 방법을 연결하는 직관이다. DSM의 회귀 목표, 노이즈 수준을 바꾸는 Langevin 보정과 시간 의존 역확산을 구분해야 한다. 동일한 수학적 요소를 공유한다고 세 알고리즘의 목표·전이가 같아지는 것은 아니다.

```
Score Function s(x) = ∇log p(x)
           |
           |--- [Learning] ---> Score Matching
           |                     |
           |                     |--- Denoising Score Matching
           |                     |       |
           |                     |       |--- Multi-Scale
           |                     |               |
           |                     |               |--- Diffusion Models
           |
           |--- [Sampling] ---> Langevin Dynamics
                                 |
                                 |--- Annealed Langevin
                                 |       |
                                 |       |--- Reverse Diffusion
                                 |
                                 |--- SGLD (Stochastic Gradient LD)
```

## 8. 자세히 비교

고정 목표의 MCMC에서는 π 또는 그 스코어를 지정하고 그 목표를 보존하는 전이를 설계한다. 데이터 생성에서는 q_t의 스코어를 학습하고 노이즈 분포에서 데이터 쪽으로 이동하는 전이를 근사한다.

온도는 지정한 에너지의 거듭제곱 목표를, 오염 σ는 컨볼루션 또는 축소를 포함한 다른 목표군을 정한다. 학습 오차가 있는 생성모형과 정확 목표값이 있는 MCMC의 보정 가능성을 구분한다.

## 9. 다른 생성모형과의 비교

GAN·VAE·정규화 플로·스코어 모형은 학습 목적과 표현·생성 비용이 다르다. 일반적인 GAN은 명시적 정규화 밀도를 제공하지 않고, VAE는 잠재변수와 변분 추론을, 가역적 플로는 변수변환의 밀도 계산을 사용한다.

스코어 방식은 노이즈 제거 회귀와 반복 생성이 편리하지만 모든 모드를 보장하거나 항상 더 높은 품질·안정성을 주지는 않는다. 학습한 플로의 직접 표본이 데이터의 정확 표본이라는 뜻도 아니다. 품질·덮임·우도·비용을 같은 데이터와 예산에서 평가한다. 인코더 유무·이미지 선명도 등을 모든 변형의 절대적 특성으로 제시하지 않는다.

## 10. 실용적인 고려

노이즈 스케줄·로스 가중·데이터 표준화와 스텝 크기는 서로 연결된다. 선형 β 스케줄은 β_t가 시간에 따라 선형으로 변하는 것이며 β_t=상수와 다르다. 코사인 누적 신호 스케줄도 시간 정규화·오프셋·한 단계 β로의 변환을 명시해야 한다.

필요한 네트워크 평가 수는 샘플러·학습 방식·허용오차에 따라 다르다. 모든 확산모형이 50~100단계면 충분하거나 특정 σ_min이 적절하다고 단정하지 않는다. 동일한 품질·덮임·계산비용으로 비교하고 학습·추론의 기기와 자료형을 일관되게 맞춘다.

## 11. 더 깊은 주제

정방향 SDE dX=f(X,t)dt+g(t)dW에서 f·g는 보통 설계한 오염 과정이다. 학습 대상은 시간별 주변 스코어 s_t=∇log p_t다. 역시간 τ=T−t를 증가시키면 역확산의 드리프트는 −f(X,t)+g(t)²s_t(X)이며 잡음 계수는 g(t)다. 고정 목표 Langevin 드리프트와 일반적으로 다르다.

확률 흐름 ODE는 원래 시간 t에서 dX/dt=f−g²s_t/2다. 정확한 스코어·초기분포·정칙성에서 확산과 같은 시간별 주변을 갖는다. 실제 우도 계산에는 divergence 적분과 수치 ODE 오차가 들어가므로 유한 계산에서 자동으로 정확한 데이터 우도가 되는 것은 아니다. [스코어 SDE 논문](https://arxiv.org/abs/2011.13456)을 참고한다.

조건부 스코어는 Bayes 규칙으로 s_t(x|y)=s_t(x)+∇_xlog p_t(y|x)와 연결된다. 추가 배율 w≠1은 일반적으로 원래 조건부 스코어와 다른 장을 만들며 품질·다양성의 결과는 경험적으로 평가해야 한다. 시간별 분류기는 오염 상태에 맞춰야 한다. SGLD는 파라메터 사후의 확률 경사를 사용하는 별도 설정이다.

## 12. 예제 코드의 범위와 노이즈 순서

아래는 구조를 보여 주는 교육용 조각이며 load_data와 data_dim이 정의되지 않아 그대로 완결된 실행은 아니다. 학습은 전체 데이터와 무작위 한 σ의 DSM 로스를 사용한다. σ별 로스 가중이 없으면 작은 σ의 큰 목표값이 로스 스케일에 강한 영향을 줄 수 있다.

sigmas=torch.logspace(−2,1,10)는 작은 값부터 큰 값 순서다. 현재 샘플러는 입력 순서대로 진행하므로 의도한 큰 노이즈→작은 노이즈의 annealed Langevin과 반대다. 초기화도 첫 번째 작은 σ를 사용하고 마지막 값은 가장 큰 σ이므로 최종 데이터 노이즈 제거 결과로 해석할 수 없다.

샘플 함수의 스텝 크기 비율과 기준값, 분포별 안정성을 따로 검토해야 한다. torch.tensor(sigma)의 재생성은 기기·자료형을 의도치 않게 바꿀 수 있고, 추론에서 그래프를 끊지 않아 긴 반복의 계산 그래프가 누적될 수 있다. ScoreNetwork는 스칼라 또는 일괄 공유 σ를 전제로 하며 표본별 σ 벡터에는 그대로 맞지 않을 수 있다. 코드와 출력은 보존한 설명 자료다.

```python
import torch
import torch.nn as nn

class ScoreNetwork(nn.Module):
    """단순한 다층 퍼셉트론 점수 망"""
    def __init__(self, data_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim + 1, hidden_dim),  # sigma 몫으로 +1
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, data_dim)
        )
    
    def forward(self, x, sigma):
        # 자료와 잡음 레이어 이어 붙이기
        sigma_vec = sigma * torch.ones(x.shape[0], 1, device=x.device)
        inp = torch.cat([x, sigma_vec], dim=1)
        return self.net(inp)

def train_score_model(data, sigmas, epochs=1000):
    """잡음 없애는 점수 맞추기로 점수 망 익히기"""
    model = ScoreNetwork(data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        # 잡음 레이어 표집
        sigma = sigmas[torch.randint(len(sigmas), (1,))]
        
        # 자료에 잡음 더하기
        noise = torch.randn_like(data) * sigma
        x_noisy = data + noise
        
        # 점수 미리보기
        score_pred = model(x_noisy, sigma)
        
        # 참 점수
        score_true = -noise / (sigma ** 2)
        
        # 로스
        loss = 0.5 * ((score_pred - score_true) ** 2).sum(dim=1).mean()
        
        # 갱신
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model

def sample_annealed_langevin(model, sigmas, n_steps=100):
    """담금질한 랑주뱅 액션으로 표집하기"""
    x = torch.randn(100, data_dim) * sigmas[0]  # 잡음에서 시작
    
    for sigma in sigmas:
        epsilon = 0.1 * (sigma / sigmas[-1]) ** 2
        
        for _ in range(n_steps):
            # 랑주뱅 스텝
            score = model(x, torch.tensor(sigma))
            x = x + epsilon * score + torch.sqrt(2 * epsilon) * torch.randn_like(x)
    
    return x

# 사용법
data = load_data()  # 여러분의 자료
sigmas = torch.logspace(-2, 1, 10)  # 등비 늘어놓음

model = train_score_model(data, sigmas)
samples = sample_annealed_langevin(model, sigmas)
```

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
스코어만으로 분포가 유일하게 결정되는 조건은 무엇인가?

</div>

??? success "풀이"
    양의 매끄러운 밀도의 연결된 영역과 적분 가능한 정규화, 보존 벡터장 조건이 필요하다. 분리된 지지집합에서는 성분별 상수가 남아 상대 질량이 결정되지 않을 수 있다.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
DSM에서 −(x−x₀)/σ²는 바로 주변 스코어인가?

</div>

??? success "풀이"
    한 오염 쌍의 조건부 스코어다. x를 조건으로 이 값을 평균한 회귀 최적해가 ∇log q_σ(x)이며 각 훈련 목표값과 주변 스코어를 동일시하지 않는다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
DDPM의 α_t와 누적 ᾱ_t를 구분하라.

</div>

??? success "풀이"
    α_t=1−β_t는 한 단계 신호 비율, ᾱ_t는 처음부터 t까지의 곱이다. 정방향 주변과 노이즈-스코어 관계에는 ᾱ_t, 역평균의 바깥 계수에는 α_t가 들어간다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff easy" title="쉬움"></span>
현재 예제의 노이즈 순서가 annealed Langevin에 맞는가?

</div>

??? success "풀이"
    logspace가 작은 값부터 커지는데 그대로 반복하므로 의도한 큰→작은 순서와 반대다. 마지막도 가장 큰 오염 수준이라 학습·혼합 오차와 별개로 의도한 생성 절차를 구현하지 않는다.

## 정리하며

스코어는 MCMC와 생성모형을 연결하지만 고정 목표·온도·컨볼루션 노이즈·시간 역전의 역할은 다르다. DSM의 조건부 회귀와 주변 스코어, DDPM의 누적·한 단계 계수, 실제 노이즈 순서와 수치 오차를 구분한다.
