# 잡음 없애는 은근한 퍼짐 모델(DDIM)
**DDIM**(Song 외, 2020)은 다시 익히지 않고도 퍼짐 모델을 더 빠르게, 흔히 정해진 방식으로 뽑는 방법을 준다.

## 왜 필요한가

DDPM 뽑기는 $T$개(흔히 1000개) 때 걸음을 모두 지나야 해서 만들어 내기가 느리다. DDIM은 다음으로 이를 다룬다.

1. 가장자리 분포가 같은 **마르코프가 아닌** 앞 과정을 뜻매김한다
2. 뽑는 동안 **걸음 건너뛰기**를 가능하게 한다
3. 바라면 **정해진** 만들어 내기를 할 수 있게 한다

## 핵심 통찰

DDPM 익히기 목표는 온 결합 분포 $q(x_{1:T}|x_0)$이 아니라 가장자리 분포 $q(x_t|x_0)$에만 매인다. 곧 서로 다른 앞 과정 여럿이 같은 익히기 손실을 나누어 가진다.

## DDIM 앞 과정

DDIM은 $\sigma$으로 어깨수를 매긴 마르코프가 아닌 과정의 무리를 뜻매김한다.

$$
q_\sigma(x_{t-1}|x_t, x_0) = \mathcal{N}\left(\sqrt{\bar{\alpha}_{t-1}}x_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\cdot\frac{x_t - \sqrt{\bar{\alpha}_t}x_0}{\sqrt{1-\bar{\alpha}_t}}, \sigma_t^2 I\right)
$$

### 특별한 경우

| $\sigma_t$ | 움직임 |
|------------|----------|
| $\sigma_t = \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t}$ | DDPM과 같다 |
| $\sigma_t = 0$ | 정해짐(DDIM) |

## DDIM 뽑기 고침

익힌 잡음 헤아리개 $\epsilon_\theta(x_t, t)$으로:

### 걸음 1: x_0 헤아리기

$$
\hat{x}_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}
$$

### 걸음 2: x_(t-1)으로 고치기

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2}\cdot\epsilon_\theta(x_t, t) + \sigma_t z
$$

여기서 $z \sim \mathcal{N}(0, I)$이다.

### 정해진 경우(sigma_t = 0)

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}}\cdot\epsilon_\theta(x_t, t)
$$

마구잡이가 없다! 첫 잡음 $x_T$마다 하나뿐인 그림 $x_0$으로 옮겨진다.

## 빠르게 뽑기

### 부분 차례 뽑기

때 걸음 $\{1, 2, \ldots, T\}$을 모두 쓰는 대신 $S \ll T$인 부분 차례 $\tau = \{t_1, t_2, \ldots, t_S\}$을 쓴다.

보기: $T=1000$이면 $\tau = \{1, 21, 41, \ldots, 981\}$(50걸음)을 쓴다.

### 부분 차례의 고침 규칙

$\tau$의 잇닿은 낱개 $t_i$과 $t_{i+1}$에 대해:

$$
x_{t_i} = \sqrt{\bar{\alpha}_{t_i}}\hat{x}_0 + \sqrt{1-\bar{\alpha}_{t_i}}\cdot\epsilon_\theta(x_{t_{i+1}}, t_{i+1})
$$

## 알고리즘

```
알고리즘: DDIM 뽑기
────────────────────────
Input: Trained ε_θ, subsequence τ = [t_S, ..., t_1], η (stochasticity)

x_T ~ N(0, I)

for i = S, S-1, ..., 1:
    t = τ[i]
    t_prev = τ[i-1] if i > 1 else 0
    
    # x_0을 헤아린다
    x̂_0 = (x_t - sqrt(1-ᾱ_t) * ε_θ(x_t, t)) / sqrt(ᾱ_t)
    
    # 분산 계산
    σ_t = η * sqrt((1-ᾱ_{t_prev})/(1-ᾱ_t)) * sqrt(1-ᾱ_t/ᾱ_{t_prev})
    
    # x_t을 가리키는 방향
    dir_xt = sqrt(1 - ᾱ_{t_prev} - σ_t²) * ε_θ(x_t, t)
    
    # x_{t_prev}을 뽑는다
    noise = N(0, I) if t > 1 else 0
    x_{t_prev} = sqrt(ᾱ_{t_prev}) * x̂_0 + dir_xt + σ_t * noise

return x_0
```

## 구현

```python
import torch

class DDIMSampler:
    def __init__(self, model, alphas_bar, T=1000):
        self.model = model
        self.alphas_bar = alphas_bar
        self.T = T
    
    def get_timestep_sequence(self, num_steps):
        """고르게 벌린 때 걸음 부분 차례를 만든다."""
        step_size = self.T // num_steps
        return list(range(0, self.T, step_size))[::-1]  # 내림 차례
    
    @torch.no_grad()
    def sample(self, shape, device, num_steps=50, eta=0.0):
        """
        DDIM 뽑기.
        
        인수:
            shape: 내놓기 꼴(묶음, 채널, 높이, 너비)
            device: 토치 장치
            num_steps: 뽑기 걸음 수
            eta: 마구잡이 정도(0이면 정해진 대로, 1이면 DDPM처럼)
        
        반환값:
            만든 표본
        """
        # 때 걸음 부분 차례를 얻는다
        timesteps = self.get_timestep_sequence(num_steps)
        
        # 잡음에서 시작한다
        x = torch.randn(shape, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # alpha_bar 값을 얻는다
            alpha_bar_t = self.alphas_bar[t]
            alpha_bar_prev = self.alphas_bar[timesteps[i+1]] if i < len(timesteps)-1 else torch.tensor(1.0)
            
            # 잡음을 헤아린다
            eps = self.model(x, t_batch)
            
            # x_0을 헤아린다
            x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            
            # 필요하면 x0 헤아림을 자른다
            x0_pred = torch.clamp(x0_pred, -1, 1)
            
            # 시그마를 셈한다
            sigma = eta * torch.sqrt(
                (1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_prev)
            )
            
            # x_t을 가리키는 방향
            dir_xt = torch.sqrt(1 - alpha_bar_prev - sigma**2) * eps
            
            # x_{t-1}을 뽑는다
            if i < len(timesteps) - 1:
                noise = torch.randn_like(x) if eta > 0 else 0
                x = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt + sigma * noise
            else:
                x = x0_pred
        
        return x
```

## 견주기: DDPM과 DDIM

| 갈래 | DDPM | DDIM |
|--------|------|------|
| 뽑기 갈래 | 확률 | 정해짐(η=0)이나 확률 |
| 필요한 걸음 | T(예컨대 1000) | 아무 S ≪ T(예컨대 50) |
| 숨은 공간 | 짜임이 없다 | 뜻있는 사이 메우기 |
| 표본 품질 | 바탕 | 걸음이 적어도 비슷하다 |
| 빠르기 | 느리다 | 10-50배 빠르다 |

## 정해진 뽑기의 좋은 점

### 숨은 공간의 짜임

$\eta = 0$이면 $x_T$과 $x_0$ 사이에 일대일 대응이 있다.

$$
x_0 = f_\theta(x_T)
$$

이로써 다음이 가능해진다:

1. **사이 메우기**: 숨은 값을 섞어 그림을 섞는다
2. **되돌리기**: 주어진 그림을 만드는 $x_T$을 찾는다
3. **고치기**: 다스린 바꿈을 위해 $x_T$을 고친다

### 일치성

같은 $x_T$은 늘 같은 $x_0$을 내며 다음에 쓸모 있다.

- 되풀이할 수 있음
- 벌레 잡기
- 떼어 보기 연구

## 신경 상미분 방정식과의 이음

정해진 DDIM은 상미분 방정식을 푸는 것으로 볼 수 있다.

$$
\frac{dx}{dt} = f_\theta(x, t)
$$

이 확률 흐름 상미분 방정식은 퍼짐 확률 미분 방정식과 가장자리 분포가 같아 다음이 가능하다.

- 상미분 방정식 풀개(오일러, 호인, RK45) 쓰기
- 맞추어 가는 걸음 크기
- 높은 차수 방법으로 더 빠르게 하기

## 실제의 요령

### 걸음 수 고르기

| 걸음 | 품질 | 빠르기 |
|-------|---------|-------|
| 10-20 | 받아들일 만함 | 아주 빠름 |
| 50 | 좋음 | 빠름 |
| 100 | 아주 좋음 | 보통 |
| 250 이상 | 뛰어남 | 느림 |

### η 고르기

- $\eta = 0$: 정해져 있어 한결같음에 좋다
- $\eta = 1$: DDPM 같은 확률성
- $\eta \in (0, 1)$: 다양함과 한결같음의 균형

### 때 걸음 사이 띄우기

- **고르게**: 단순하고 잘 듣는다
- **이차**: 잡음이 클 때 걸음을 더 둔다
- **배운 것**: 모델마다 가장 좋게 한다

## 요약

DDIM은 DDPM과 익히기 목표가 같은 마르코프가 아닌 적기를 써서 퍼짐 모델에서 빠르게 뽑을 수 있게 한다. $\eta=0$이면 뽑기가 정해져 숨은 공간을 다룰 수 있다. 실제의 핵심 이점은 품질을 지키면서 뽑기를 1000걸음에서 50걸음 이하로 줄이는 것이다.

## 연습문제

**연습문제 1.**
DDPM의 앞 퍼짐 과정을 설명하라. 왜 정규 분포 옮아감을 지닌 마르코프 사슬로 짰는가?

??? success "연습문제 1 풀이"
    앞 과정은 $T$걸음에 걸쳐 정규 분포 잡음을 차츰 더한다. 곧 $q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$이며 $\{\beta_t\}$은 잡음 차례표이다. 정규 분포 옮아감을 고른 까닭은 (1) 정규 분포가 선형 아우름에 닫혀 있어 $\bar{\alpha}_t = \prod_{s=1}^t (1-\beta_s)$일 때 닫힌 꼴 가장자리 분포 $q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t)I)$을 얻고, (2) 정규 분포 사이의 쿨백-라이블러 벌어짐이 닫힌 꼴이어서 익히기 손실이 단순해지며, (3) $\beta_t$이 작으면 뒤 과정도 거의 정규 분포이기 때문이다.

---

**연습문제 2.**
단순하게 만든 DDPM 익히기 목표를 이끌어 내고 그것이 잡음 헤아리기와 같음을 보여라.

??? success "연습문제 2 풀이"
    변분 한계는 정규 분포 사이의 쿨백-라이블러 항으로 나뉜다. 단순하게 하면 손실이 다음으로 줄어든다.

    $$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

    여기서 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$이고 $\epsilon \sim \mathcal{N}(0, I)$이다. 신경망 $\epsilon_\theta$은 걸음 $t$에서 더한 잡음을 헤아린다. $\epsilon_\theta(x_t, t) \approx -\sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log q(x_t | x_0)$이므로 이는 점수 $\nabla_{x_t} \log q(x_t)$을 헤아리는 것과 같다. 단순하게 만든 손실은 무게 항을 버리지만 실제로 잘 듣는다.

---

**연습문제 3.**
DDIM이 어떻게 DDPM보다 빠르게 뽑는지 설명하라. 무엇을 맞바꾸는가?

??? success "연습문제 3 풀이"
    DDPM은 차례대로 잡음 없애기를 $T$번(흔히 1000번) 해야 한다. DDIM은 가장자리 분포 $q(x_t | x_0)$이 같으면서 $S \ll T$개 때 걸음의 부분 차례로 정해진 뽑기를 할 수 있는 마르코프가 아닌 앞 과정을 뜻매김한다. 고침 규칙은 $x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \hat{x}_0 + \sqrt{1-\bar{\alpha}_{t-1}-\sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t \epsilon$을 쓴다. $\sigma_t = 0$이면 정해진 뽑기가 되고 $\sigma_t = \sqrt{\beta_t}$이면 DDPM이 된다. **맞바꿈**: 뽑기가 빨라지지만(1000걸음 대신 50걸음) 걸음이 아주 적으면 표본 품질이 조금 떨어진다. 다만 정해진 대응 덕분에 숨은 공간에서 뜻있는 사이 메우기가 된다.

---

**연습문제 4.**
가름개 없는 이끌기란 무엇인가? 그것이 조건 만들어 내기 품질을 어떻게 높이는가?

??? success "연습문제 4 풀이"
    가름개 없는 이끌기는 익히는 동안 조건 신호를 마구잡이로 떨구어 조건 있는 만들어 내기($\epsilon_\theta(x_t, t, c)$)와 조건 없는 만들어 내기($\epsilon_\theta(x_t, t, \emptyset)$)를 모두 다루는 퍼짐 모델 하나를 익힌다. 추론할 때 이끈 헤아림은 다음과 같다.

    $$\tilde{\epsilon}_\theta(x_t, t, c) = (1 + w) \epsilon_\theta(x_t, t, c) - w \, \epsilon_\theta(x_t, t, \emptyset)$$

    여기서 $w > 0$은 이끌기 잣수이다. 이는 조건 있는 헤아림과 없는 헤아림의 차이를 키워 표본을 조건 아래 가능도가 높은 자리로 민다. $w$이 클수록 더 충실하지만 덜 다양한 표본이 나온다(품질과 다양함의 맞바꿈). 따로 가름개를 익히지 않아도 되어 요즘 퍼짐 모델의 여느 방식이 되었다.
