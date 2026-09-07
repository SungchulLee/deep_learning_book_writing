# 점수 맞추기, 랑주뱅 동역학, 퍼짐 모형
## 점수 함수: 하나로 꿰는 개념

**점수 함수**는 로그 밀도의 기울기이다:

$$
s(x) = \nabla_x \log p(x)
$$

얼핏 단순해 보이는 이 양이 다음을 잇는다:

- 랑주뱅 동역학(MCMC 표집)
- 점수 맞추기(밀도 어림)
- 퍼짐 모형(낳는 모형 만들기)

## 왜 점수인가?

### 성질 1: 점수가 (상수까지) 분포를 정한다

어디서나 $s(x) = \nabla \log p(x)$을 알면 고르게 하기까지 $p$을 되찾을 수 있다:

$$
p(x) \propto \exp\left(\int_0^x s(u) \cdot du\right)
$$

**뜻하는 바**: 점수를 배우는 것은 분포를 배우는 것과 같다!

### 성질 2: 점수는 고르게 하기가 필요 없다

$\nabla \log p(x)$을 셈하는 데 나눔 함수 $Z$이 필요 없다:

$$
\nabla \log p(x) = \nabla \log \frac{\tilde{p}(x)}{Z} = \nabla \log \tilde{p}(x) - \nabla \log Z = \nabla \log \tilde{p}(x)
$$

$Z$은 상수이므로 $\nabla \log Z = 0$이다.

**뜻하는 바**: 고르게 하지 않은 밀도를 곧바로 다룰 수 있다!

### 성질 3: 점수가 랑주뱅 동역학을 정한다

랑주뱅 SDE은 점수가 이끈다:

$$
dx_t = s(x_t)\,dt + \sqrt{2}\,dW_t
$$

$p$에서 표집하는 일은 잡음과 함께 점수 함수를 따라가는 일로 줄어든다.

## 점수 맞추기: 자료에서 점수 배우기

**문제**: 표본 $\{x_i\}_{i=1}^N \sim p_{\text{data}}$이 주어졌을 때 $s(x) = \nabla \log p_{\text{data}}(x)$을 배워라.

**길**: 참 점수를 어림하도록 신경망 $s_\theta(x)$을 가르친다.

### 소박한 길: 점수 차이를 가장 작게 하기

$$
\mathcal{L}(\theta) = \frac{1}{2}\mathbb{E}_{p_{\text{data}}}[\|s_\theta(x) - \nabla \log p_{\text{data}}(x)\|^2]
$$

**문제**: $\nabla \log p_{\text{data}}(x)$을 모른다(그것이 바로 배우려는 것이다)!

### 점수 맞추기 목적 함수(히바리넨 2005)

**핵심 항등식**: 매끄러운 아무 벡터 마당 $s_\theta(x)$에 대해:

$$
\frac{1}{2}\mathbb{E}_{p}[\|s_\theta(x) - \nabla \log p(x)\|^2] = \mathbb{E}_p[\text{tr}(\nabla s_\theta(x)) + \frac{1}{2}\|s_\theta(x)\|^2] + \text{const}
$$

오른쪽 변은 $p$을 알 필요가 없다. 거기서 뽑은 표본만 있으면 된다!

**점수 맞추기 목적 함수**:

$$
\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}}\left[\text{tr}(\nabla s_\theta(x)) + \frac{1}{2}\|s_\theta(x)\|^2\right]
$$

**구현**:
```python
def score_matching_loss(score_network, x):
    """
    x: 자료 표본 묶음, 꼴 (batch, dim)
    score_network: 점수 s_θ(x)을 내놓는 신경망
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
    
    # 점수 맞추기 손실
    loss = (div_s + 0.5 * (s ** 2).sum(dim=1)).mean()
    return loss
```

### 잡음 없애기 점수 맞추기(뱅상 2011)

**생각**: $p_{\text{data}}$의 점수를 맞추는 대신 잡음을 섞은 자료의 점수를 맞춘다.

**잡음 모형**: $q_\sigma(x|x_0) = \mathcal{N}(x | x_0, \sigma^2 I)$

**잡음 섞은 분포**: $q_\sigma(x) = \int p_{\text{data}}(x_0)q_\sigma(x|x_0)\,dx_0$

**잡음 없애기 점수 맞추기 목적 함수**:

$$
\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{x_0 \sim p_{\text{data}}, x \sim q_\sigma(\cdot|x_0)}\left[\frac{1}{2}\|s_\theta(x, \sigma) - \nabla \log q_\sigma(x|x_0)\|^2\right]
$$

**핵심 통찰**: $\nabla \log q_\sigma(x|x_0)$은 정확히 셈할 수 있다:

$$
\nabla \log q_\sigma(x|x_0) = -\frac{x - x_0}{\sigma^2}
$$

**구현**:
```python
def denoising_score_matching_loss(score_network, x_data, sigma):
    """
    x_data: 깨끗한 자료 표본
    sigma: 잡음 층
    """
    # 잡음 더하기
    noise = torch.randn_like(x_data) * sigma
    x_noisy = x_data + noise
    
    # 미리본 점수
    s_pred = score_network(x_noisy, sigma)
    
    # 잡음 섞인 자료의 참 점수
    s_true = -noise / (sigma ** 2)
    
    # 잡음 없애는 점수 맞추기 손실
    loss = 0.5 * ((s_pred - s_true) ** 2).sum(dim=1).mean()
    return loss
```

**좋은 점**:

- 구현이 더 단순하다(발산을 셈하지 않는다)
- 가르치기가 더 안정적이다
- 퍼짐 모형과 곧바로 이어진다

## 점수 맞추기에서 표집으로: 랑주뱅 MCMC

$s_\theta(x) \approx \nabla \log p_{\text{data}}(x)$을 배우고 나면 표집할 수 있다:

**랑주뱅 동역학**:

$$
x_{t+1} = x_t + \epsilon s_\theta(x_t) + \sqrt{2\epsilon}\,\eta_t, \quad \eta_t \sim \mathcal{N}(0, I)
$$

무작위 잡음에서 시작해 이 과정을 돌리면 → $p_{\text{data}}$의 표본이 나온다!

이것이 **점수 기반 낳는 모형 만들기**이다.

## 여러 눈금 점수 맞추기

**문제**: 밀도가 낮은 구역(자료가 성긴 곳)에서는 점수가 잘 정해지지 않는다.

**풀이**: 여러 잡음 수준에서 점수를 배운다.

**눈금마다의 잡음 섞은 분포**:

$$
q_{\sigma_i}(x) = \int p_{\text{data}}(x_0)\mathcal{N}(x|x_0, \sigma_i^2 I)\,dx_0
$$

여기서 $\sigma_1 < \sigma_2 < \cdots < \sigma_L$이다.

모든 잡음 수준의 점수를 맞추도록 **점수 망** $s_\theta(x, \sigma)$을 가르친다:

$$
\mathcal{L}(\theta) = \sum_{i=1}^L \lambda_i \mathbb{E}_{x \sim q_{\sigma_i}}\left[\frac{1}{2}\|s_\theta(x, \sigma_i) - \nabla \log q_{\sigma_i}(x)\|^2\right]
$$

표집을 위한 **담금질한 랑주뱅 동역학**:

1. 가장 높은 잡음 수준 $\sigma_L$에서 시작한다(거의 고르다)
2. $\sigma_L$에서 랑주뱅을 돌린다 → $q_{\sigma_L}$의 표본
3. 잡음을 줄인다: $\sigma_L \to \sigma_{L-1} \to \cdots \to \sigma_1$
4. 가장 낮은 잡음에서 끝낸다 → $p_{\text{data}}$의 표본

**이것이 바로 표집에 쓴 흉내 담금질이다!**

## 퍼짐 모형과의 이음

퍼짐 모형은 이어진 잡음 일정을 쓴 여러 눈금 점수 맞추기이다.

### 앞 과정(잡음 더하기)

$$
x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\,\epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

여기서 $t$이 0에서 $T$으로 갈 때 $\alpha_t$은 1에서 0으로 줄어든다.

**이는 $\sigma^2 = 1 - \alpha_t$인 잡음 모형** $q_\sigma(x|x_0)$**과 같다.**

### 뒤 과정(잡음 없애기)

뒤 과정은 점수가 이끄는 대로 잡음을 없앤다:

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t + (1-\alpha_t)s_\theta(x_t, t)\right) + \sqrt{1-\alpha_t}\,\eta
$$

**이것이 시간에 따라 달라지는 잡음을 쓴 랑주뱅 동역학이다!**

### 퍼짐 모형 속의 점수 함수

퍼짐 모형의 "잡음 예측 망" $\epsilon_\theta(x_t, t)$은 점수와 이어져 있다:

$$
s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\alpha_t}}
$$

**가르치기 목적 함수**(간추림):

$$
\mathcal{L}(\theta) = \mathbb{E}_{x_0, t, \epsilon}\left[\|\epsilon_\theta(x_t, t) - \epsilon\|^2\right]
$$

여기서 $x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon$이다.

**이것이 이어진 시간의 잡음 없애기 점수 맞추기이다!**

## 온전한 그림

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

## 자세히 견주기

| 개념 | MCMC의 눈 | 낳는 모형의 눈 |
|---------|-----------|-------------------------|
| **목표** | $\pi(x)$에서 표집 | $p_{\text{data}}$의 표본 만들기 |
| **점수** | $\nabla \log \pi(x)$ | $\nabla \log p_{\text{data}}(x)$ |
| **아는가?** | 예(베이즈에서 온 뒤확률) | 아니오(자료에서 배운다) |
| **방법** | 랑주뱅 동역학 | 점수 기반 낳는 모형 |
| **담금질** | 흉내 담금질 | 여러 눈금 / 퍼짐 |
| **온도** | 살펴보기를 다스린다 | 잡음 수준 $\sigma$ |

**핵심 통찰**: 같은 수학, 다른 눈길!

## 점수 기반 모형과 다른 낳는 모형

### GAN(맞서 겨루는 낳는 망)

**좋은 점**:

- 빠른 표집(앞먹임 한 번)
- 질 좋은 표본

**나쁜 점**:

- 가르치기가 불안정하다(맞서 겨루기)
- 봉우리 주저앉음
- 밀도를 어림하지 못한다

**점수 기반**:

- 가르치기가 안정적이다(맞서 겨루기가 아니라 회귀)
- 봉우리를 모두 덮는다(가르치기를 꼼꼼히 하면)
- 표집이 느리다(되풀이 과정)

### VAE(변분 자동 부호기)

**좋은 점**:

- 빠른 표집
- 부호기가 있다(추론에 쓴다)

**나쁜 점**:

- 표본이 흐릿하다(GAN이나 퍼짐 모형에 견주어)
- 어림 추론(변분)

**점수 기반**:

- 표본의 질이 더 높다
- 부호기가 없다(순수하게 낳기만 한다)
- 정확한 표집(극한에서)

### 고르게 하는 흐름

**좋은 점**:

- 정확한 가능도
- 정확한 표집(거꾸로 갈 수 있다)

**나쁜 점**:

- 구조에 제약이 있다(거꾸로 갈 수 있어야 한다)
- 크게 키우기 어렵다

**점수 기반**:

- 구조가 유연하다
- 어림이지만 질이 높다
- 높은 차원까지 감당한다

## 실용적인 고려

### 잡음 일정 짜기

**점수 맞추기**에서는 $\{\sigma_i\}$을 고른다:

- **등비수열**: $\sigma_i = \sigma_{\max} \cdot r^i$, $r < 1$
- 흔히: $\sigma_{\max} \approx \text{std}(\text{data})$, $\sigma_{\min} \approx 0.01$

**퍼짐 모형**에서는 $\{\alpha_t\}$을 고른다:

- **선형 일정**: $\beta_t = \text{const}$
- **코사인 일정**: $\alpha_t = \cos^2(\pi t / 2)$
- 요즘: 배운 일정

### 걸음의 개수

**랑주뱅 MCMC**: 보통 잡음 수준마다 100-1000 걸음

**담금질한 랑주뱅**: 잡음 수준 10-50개 × 100-1000 걸음 = 모두 1000-50,000

**퍼짐 모형**:

- 가르치기: 시간 걸음 1000개
- 표집: (DDIM이나 DDPM으로) 50-100 걸음이면 된다

### 계산 비용

**점수 망 값 매기기**:

- 보통 U-Net이나 트랜스포머
- 큰 모형에는 GPU이 필요하다
- 표집의 병목이다

**표집 값**:

- 점수 기반: $N_{\text{steps}}$ × (점수 망 값 매기기)
- GAN이나 VAE보다 훨씬 느리다
- 그러나 나아지고 있다(빠른 표집기, 증류)

## 더 깊은 주제

### 확률 기울기 랑주뱅 동역학(SGLD)

베이즈 신경망에 랑주뱅을 쓴다:

$$
\theta_{t+1} = \theta_t + \frac{\epsilon}{2}\nabla \log p(\theta | \mathcal{D}) + \sqrt{\epsilon}\,\eta_t
$$

작은 묶음 기울기를 쓴다(확률적으로).

**이음**: 점수 기반 표집 + 작은 묶음 = SGLD.

### 점수 기반 SDE

이어진 시간으로 일반화한다:

$$
dx = f(x,t)\,dt + g(t)\,dW
$$

여기서 $f$은 자료에서 배운다.

**확률 흐름 ODE**: 정해진 판

$$
dx = \left[f(x,t) - \frac{1}{2}g(t)^2 s(x,t)\right]dt
$$

정확한 가능도 셈하기를 가능하게 한다!

### 조건부 낳기

조건부 점수 $s_\theta(x, y, t)$을 배운다

다음으로 $x | y$을 표집한다:

$$
x_{t-1} = x_t + \epsilon s_\theta(x_t, y, t) + \sqrt{2\epsilon}\,\eta
$$

쓰임새:

- 그림 메우기: $y$ = 가린 그림
- 해상도 높이기: $y$ = 낮은 해상도 그림
- 글에서 그림으로: $y$ = 글 묻힘

### 분류기 이끌기

조건 없는 점수와 분류기 기울기를 합친다:

$$
s_\theta^{\text{guided}}(x, y, t) = s_\theta(x, t) + w \cdot \nabla_x \log p(y|x_t)
$$

여기서 $w$은 이끄는 세기이다.

**효과**: 여러 갈래를 내주고 질을 얻는다.

## 코드 보기: 온전한 흐름

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
        # 자료와 잡음 층 이어 붙이기
        sigma_vec = sigma * torch.ones(x.shape[0], 1, device=x.device)
        inp = torch.cat([x, sigma_vec], dim=1)
        return self.net(inp)

def train_score_model(data, sigmas, epochs=1000):
    """잡음 없애는 점수 맞추기로 점수 망 익히기"""
    model = ScoreNetwork(data.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        # 잡음 층 표집
        sigma = sigmas[torch.randint(len(sigmas), (1,))]
        
        # 자료에 잡음 더하기
        noise = torch.randn_like(data) * sigma
        x_noisy = data + noise
        
        # 점수 미리보기
        score_pred = model(x_noisy, sigma)
        
        # 참 점수
        score_true = -noise / (sigma ** 2)
        
        # 손실
        loss = 0.5 * ((score_pred - score_true) ** 2).sum(dim=1).mean()
        
        # 갱신
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model

def sample_annealed_langevin(model, sigmas, n_steps=100):
    """담금질한 랑주뱅 움직임으로 표집하기"""
    x = torch.randn(100, data_dim) * sigmas[0]  # 잡음에서 시작
    
    for sigma in sigmas:
        epsilon = 0.1 * (sigma / sigmas[-1]) ** 2
        
        for _ in range(n_steps):
            # 랑주뱅 걸음
            score = model(x, torch.tensor(sigma))
            x = x + epsilon * score + torch.sqrt(2 * epsilon) * torch.randn_like(x)
    
    return x

# 사용법
data = load_data()  # 여러분의 자료
sigmas = torch.logspace(-2, 1, 10)  # 등비 늘어놓음

model = train_score_model(data, sigmas)
samples = sample_annealed_langevin(model, sigmas)
```

## 요약

**점수 함수 $\nabla \log p(x)$은** 다음의 중심이다:

1. **랑주뱅 MCMC**: 아는 분포에서 표집하기
2. **점수 맞추기**: 자료에서 분포 배우기
3. **퍼짐 모형**: 낳는 모형 만들기

**핵심 이음**:

- **잡음 없애기 점수 맞추기** = 퍼짐 모형의 가르치기 목적 함수
- **담금질한 랑주뱅** = 뒤 퍼짐 과정
- **온도/잡음** = 두 얼개에서 같은 개념

**실전에서의 자취**:

- MCMC 방법이 낳는 모형 만들기에 알려 준다
- 낳는 모형의 통찰이 MCMC를 낫게 한다
- 분야끼리 서로 꽃가루를 옮긴다

**큰 그림**:
이들은 따로 떨어진 기법이 아니다. 확률 공간에서 점수 함수를 따라간다는 같은 근본 수학을 다르게 쓴 것일 뿐이다!

**요즘 가장 앞선 것**:

- 스테이블 디퓨전, DALL-E 2: 그림 만들기
- WaveGrad: 소리 만들기
- 알파폴드: 단백질 구조(점수 기반 생각을 쓴다)

점수 함수는 고전 MCMC와 가장 앞선 낳는 모형을 하나로 꿰며 현대 기계 학습에서 가장 중요한 개념 가운데 하나가 되었다.

## 연습문제

**연습문제 1.**
마르코프 사슬이 올바른 과녁 분포로 모이게 하는 데 받아들임 확률이 하는 몫을 설명하여라.

??? success "연습문제 1 풀이"
    받아들임 확률이 **자세한 균형** $\pi(x) T(x \to x') \alpha(x \to x') = \pi(x') T(x' \to x) \alpha(x' \to x)$을 보장한다. 여기서 $\pi$은 과녁 분포, $T$은 제안 분포, $\alpha$은 받아들임 확률이다. 자세한 균형은 $\pi$이 사슬의 멈춘 분포임을 뜻한다. 쪼갤 수 없음과 주기 없음까지 합치면 $\pi$으로의 에르고드 모임이 보장된다.

---

**연습문제 2.**
제안 분포가 너무 좁은 상황과 너무 넓은 상황을 밝혀라. 저마다 표집 효율에 어떤 영향을 주는가?

??? success "연습문제 2 풀이"
    **너무 좁을 때:** 제안이 거의 늘 받아들여지지만(받아들임 비율이 높지만) 사슬이 아주 작은 걸음을 떼어 과녁 분포를 느리게 살펴본다. 그러면 자기상관이 높고 실효 표본 크기가 작아진다. **너무 넓을 때:** 제안이 확률이 낮은 구역에 자주 떨어져 물리쳐지므로(받아들임 비율이 낮으므로) 사슬이 여러 되풀이 동안 지금 상태에 갇혀 있게 된다. 두 극단 모두 효율을 떨어뜨린다. 높은 차원에서 무작위 걸음 메트로폴리스의 가장 좋은 받아들임 비율은 대략 0.234이다(Roberts 외, 1997).

---

**연습문제 3.**
메트로폴리스-헤이스팅스 받아들임 비 $\alpha = \min\left(1, \frac{\pi(x') q(x|x')}{\pi(x) q(x'|x)}\right)$이 $\pi$에 대해 자세한 균형을 만족함을 증명하여라.

??? success "연습문제 3 풀이"
    일반성을 잃지 않고 $\pi(x') q(x|x') \leq \pi(x) q(x'|x)$이라 하자. 그러면 $\alpha(x \to x') = \frac{\pi(x') q(x|x')}{\pi(x) q(x'|x)}$이고 $\alpha(x' \to x) = 1$이다. 자세한 균형 조건은 다음을 요구한다:

    $$\pi(x) q(x'|x) \alpha(x \to x') = \pi(x) q(x'|x) \cdot \frac{\pi(x') q(x|x')}{\pi(x) q(x'|x)} = \pi(x') q(x|x')$$

    그리고 $\pi(x') q(x|x') \alpha(x' \to x) = \pi(x') q(x|x') \cdot 1 = \pi(x') q(x|x')$이다. 양변이 같다. $\square$

---

**연습문제 4.**
MCMC에서 태우기 기간이란 무엇이며, 처음 표본을 언제 버릴지 어떻게 정하는가?

??? success "연습문제 4 풀이"
    태우기 기간은 마르코프 사슬에서 아직 멈춘 분포로 모이지 않은 처음 부분이다. 치우침을 줄이려고 이 기간의 표본을 버린다. 태우기를 정하는 길은 다음과 같다. (1) 자취 그림으로 사슬이 언제 안정되는지 눈으로 살핀다. (2) 여러 사슬에서 사슬 안 흩어짐과 사슬 사이 흩어짐을 견주는 겔먼-루빈 진단($\hat{R}$)을 쓰며 $\hat{R} < 1.01$이면 모였다고 본다. (3) 실효 표본 크기(ESS) 어림값을 쓴다. (4) 흩어진 시작점에서 여러 사슬을 돌려 서로 맞는지 살핀다.
