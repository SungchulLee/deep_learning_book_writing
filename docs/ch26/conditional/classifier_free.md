# 가름개 없는 이끌기
**가름개 없는 이끌기(CFG)**는 따로 가름개를 두지 않고 퍼짐 모델의 조건 만들어 내기 품질을 높이는 재주이다.

## 왜 필요한가

### 조건 주기의 문제

조건 퍼짐 모델 $p_\theta(x|c)$은 흔히 다음과 같은 표본을 낸다.

- 조건 $c$을 세게 담아내지 못한다
- 조건 없는 표본만큼 또렷하지 않다
- 자잘한 조건의 세부를 담지 못한다

### 가름개 이끌기(앞선 방식)

Dhariwal와 Nichol(2021)은 뽑기를 이끌려 가름개 $p_\phi(c|x_t)$을 쓰자고 했다.

$$
\tilde{\epsilon}(x_t, t, c) = \epsilon_\theta(x_t, t) - \sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log p_\phi(c|x_t)
$$

**한계**: 잡음 섞인 가름개를 익혀야 하고 걸음마다 기울기를 셈해야 한다.

## 가름개 없는 이끌기

### 핵심 생각(Ho와 Salimans, 2022)

조건이 있을 때와 없을 때를 모두 다룰 수 있는 **모델 하나**를 익힌 뒤 헤아림을 아우른다.

$$
\tilde{\epsilon}_\theta(x_t, t, c) = \epsilon_\theta(x_t, t, \varnothing) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing))
$$

여기서 각 기호는 다음과 같다.

- $\epsilon_\theta(x_t, t, c)$: 조건 있는 헤아림
- $\epsilon_\theta(x_t, t, \varnothing)$: 조건 없는 헤아림
- $w \geq 1$: 이끌기 잣수

### 단순하게 만든 꼴

$$
\tilde{\epsilon}_\theta = (1 + w) \epsilon_\theta(x_t, t, c) - w \cdot \epsilon_\theta(x_t, t, \varnothing)
$$

또는 같은 말로:

$$
\tilde{\epsilon}_\theta = \epsilon_\theta(x_t, t, \varnothing) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing))
$$

## 해석

### 점수 함수로 보기

이끈 점수는 다음과 같다.

$$
\tilde{s}(x_t, t, c) = s(x_t, t) + w \cdot \nabla_{x_t} \log p(c|x_t)
$$

CFG은 가름개 기울기를 은근히 다음과 같이 셈한다.

$$
\nabla_{x_t} \log p(c|x_t) \propto \epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \varnothing)
$$

### 이끌기 잣수의 효과

| $w$ | 움직임 |
|-----|----------|
| 0 | 조건 없는 뽑기 |
| 1 | 여느 조건 뽑기 |
| 1-5 | 부드러운 이끌기, 다양함과 충실함의 균형 |
| 5-15 | 센 이끌기, 높은 충실함 |
| >15 | 지나치게 포화하고 다양함이 준다 |

## 가름개 없는 이끌기로 익히기

### 조건 떨구기

익히는 동안 확률 $p_{\text{uncond}}$으로 조건을 마구잡이로 떨군다.

```python
def training_step(model, x_0, condition, p_uncond=0.1):
    # 조건을 마구잡이로 떨구기
    mask = torch.rand(x_0.shape[0]) < p_uncond
    condition_input = condition.clone()
    condition_input[mask] = null_condition  # 예컨대 빈 박아 넣기
    
    # 여느 DDPM 익히기
    t = sample_timesteps(x_0.shape[0])
    noise = torch.randn_like(x_0)
    x_t = forward_diffusion(x_0, t, noise)
    
    noise_pred = model(x_t, t, condition_input)
    loss = mse_loss(noise_pred, noise)
    
    return loss
```

### 빈 조건 고르기

| 조건 갈래 | 빈 조건 나타냄 |
|----------------|---------------------|
| 갈래 이름표 | 특별한 "갈래 없음" 토큰 |
| 글 박아 넣기 | 빈 글자열 박아 넣기 |
| 그림 | 0 텐서나 배운 박아 넣기 |

## CFG으로 뽑기

### 알고리즘

```
알고리즘: 가름개 없는 이끌기 뽑기
───────────────────────
들임: 모델 ε_θ, 조건 c, 이끌기 잣수 w

x_T ~ N(0, I)

for t = T, T-1, ..., 1:
    # 조건 없는 헤아림
    ε_uncond = ε_θ(x_t, t, ∅)
    
    # 조건 있는 헤아림
    ε_cond = ε_θ(x_t, t, c)
    
    # 이끈 헤아림
    ε̃ = ε_uncond + w * (ε_cond - ε_uncond)
    
    # ε̃으로 하는 DDPM/DDIM 고침
    x_{t-1} = sample_step(x_t, ε̃, t)

return x_0
```

### 구현

```python
import torch

class CFGSampler:
    def __init__(self, model, schedule, guidance_scale=7.5):
        self.model = model
        self.schedule = schedule
        self.w = guidance_scale
    
    @torch.no_grad()
    def sample(self, shape, condition, null_condition, device, num_steps=50):
        """가름개 없는 이끌기로 뽑는다."""
        x = torch.randn(shape, device=device)
        
        timesteps = self.get_timesteps(num_steps)
        
        for t in timesteps:
            t_batch = torch.full((shape[0],), t, device=device)
            
            # 두 헤아림을 모두 셈한다
            # 고르기 1: 앞먹임 두 번
            eps_uncond = self.model(x, t_batch, null_condition)
            eps_cond = self.model(x, t_batch, condition)
            
            # 고르기 2: 묶어 하기(더 효율이 좋다)
            # x_double = torch.cat([x, x])
            # t_double = torch.cat([t_batch, t_batch])
            # c_double = torch.cat([null_condition, condition])
            # eps_both = self.model(x_double, t_double, c_double)
            # eps_uncond, eps_cond = eps_both.chunk(2)
            
            # 이끌기를 쓴다
            eps_guided = eps_uncond + self.w * (eps_cond - eps_uncond)
            
            # DDIM 걸음
            x = self.ddim_step(x, eps_guided, t)
        
        return x
    
    def ddim_step(self, x, eps, t):
        """DDIM 고침 걸음 하나."""
        alpha_bar_t = self.schedule.alpha_bars[t]
        alpha_bar_prev = self.schedule.alpha_bars[t-1] if t > 0 else 1.0
        
        # x_0을 헤아린다
        x0_pred = (x - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        x0_pred = torch.clamp(x0_pred, -1, 1)
        
        # x_t 쪽 방향
        dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps
        
        # 다음 표본
        x_prev = torch.sqrt(alpha_bar_prev) * x0_pred + dir_xt
        
        return x_prev
```

## 만들어 내기에 미치는 영향

### 품질과 다양함의 맞바꿈

CFG은 **다양함**(내놓기의 여러 가지)을 대가로 **충실함**(표본이 조건에 얼마나 맞는지)을 높인다.

### FID과 CLIP 점수

| 이끌기 잣수 | FID (↓) | CLIP 점수 (↑) |
|----------------|---------|----------------|
| 1.0 | 좋음 | 보통 |
| 3.0 | 더 좋음 | 좋음 |
| 7.5 | 가장 좋음 | 가장 좋음 |
| 15.0 | 더 나쁨 | 포화됨 |

가장 좋은 $w$은 일에 따라 다르다(그림에서는 흔히 3-15).

## 그때그때 바뀌는 이끌기

### 때 걸음마다 잣수 맞추기

어떤 방법은 때 걸음마다 $w$을 바꾼다.

$$
w(t) = w_{\min} + (w_{\max} - w_{\min}) \cdot f(t)
$$

$f(t)$의 흔한 고르기:

- 상수: $f(t) = 1$
- 선형: $f(t) = t/T$
- 코사인: $f(t) = \cos(\pi t / 2T)$

### 부정 채근

글에서 그림으로에서는 빈 조건 대신 "부정 채근"을 쓴다.

$$
\tilde{\epsilon} = \epsilon_\theta(x_t, t, c_{\text{neg}}) + w \cdot (\epsilon_\theta(x_t, t, c_{\text{pos}}) - \epsilon_\theta(x_t, t, c_{\text{neg}}))
$$

이는 바라지 않는 속성에서 멀어지게 이끈다.

## 계산에 대한 고려

### 비용

CFG은 잡음 없애기 걸음마다 **앞먹임 두 번**(조건 있음 + 조건 없음)이 필요하다.

### 가장 좋게 하기

1. **묶어 하기**: 묶음 크기 2으로 앞먹임 한 번에 둘 다 셈한다
2. **저장턱**: 조건이 붙박여 있으면 조건 없는 헤아림을 저장턱에 담는다
3. **우려내기**: CFG의 내놓기에 맞도록 이끈 모델을 익힌다

## 응용

### 글에서 그림으로

Stable Diffusion은 $w \approx 7.5$으로 CFG을 쓴다.

- CFG 없이: 흐릿하고 조건이 여리다
- CFG과 함께: 또렷하고 글에 세게 조건 지어진다

### 갈래 조건 만들어 내기

ImageNet 모델은 $w \approx 2-4$으로 CFG을 쓴다.

- 갈래 충실함을 높인다
- 갈래 안의 다양함을 지킨다

### 그림 고치기

다스린 고치기를 위해 CFG을 되돌리기와 아우른다.

1. 그림을 숨은 값으로 되돌린다
2. 고친 조건 + CFG으로 다시 뽑는다

## 요약

가름개 없는 이끌기는 조건 없는 헤아림과 조건 있는 헤아림을 아울러 조건 만들어 내기를 높인다. 조건 떨구기로 익혀야 하고 추론 비용이 두 배가 되지만 표본 품질과 조건 힘을 크게 높인다. 이끌기 잣수 $w$이 충실함과 다양함의 맞바꿈을 다스리며 글에서 그림으로에서는 흔히 7.5 언저리를 쓴다.

## 연습문제

**연습문제 1.**
조건 퍼짐 모델에서 가름개 이끌기와 가름개 없는 이끌기의 차이를 설명하라.

??? success "연습문제 1 풀이"
    **가름개 이끌기**는 잡음을 아는 가름개 $p(y|x_t)$을 따로 익혀 점수를 $\tilde{s}(x_t) = s(x_t) + w \nabla_{x_t} \log p(y|x_t)$으로 고쳐 표본을 갈래 $y$ 쪽으로 이끈다. **가름개 없는 이끌기**는 조건 있는 모드와 없는 모드를 지닌 퍼짐 모델 하나를 익힌 뒤 $\tilde{\epsilon} = (1+w)\epsilon_\theta(x_t, y) - w \epsilon_\theta(x_t, \emptyset)$으로 아우른다. 가름개 없는 이끌기를 더 낫게 여기는 까닭은 (1) 따로 가름개가 필요 없고, (2) 가름개가 잡음 섞인 들임을 다루지 않아도 되며, (3) 어떤 조건 신호(글, 갈래, 그림)에도 통하고, (4) 실제로 품질 높은 표본을 내기 때문이다.

---

**연습문제 2.**
점수 함수에 베이즈 규칙을 써서 가름개 없는 이끌기 공식을 이끌어 내라.

??? success "연습문제 2 풀이"
    조건 점수는 $\nabla_x \log p(x_t|y) = \nabla_x \log p(x_t) + \nabla_x \log p(y|x_t)$이다. 은근한 가름개 기울기는 다음과 같다.

    $$\nabla_x \log p(y|x_t) = \nabla_x \log p(x_t|y) - \nabla_x \log p(x_t)$$

    무게 $w$으로 이끈 점수에 넣으면:

    $$\tilde{s} = \nabla_x \log p(x_t) + w(\nabla_x \log p(x_t|y) - \nabla_x \log p(x_t)) = (1+w)\nabla_x \log p(x_t|y) - w \nabla_x \log p(x_t)$$

    잡음 헤아림으로 바꾸면 여느 공식이 된다. $\square$

---

**연습문제 3.**
이끌기 잣수 $w$이 표본 품질과 다양함에 미치는 영향은 무엇인가?

??? success "연습문제 3 풀이"
    $w$이 커지면 **품질**이 나아진다(표본이 조건을 더 잘 나타낸다. 예컨대 갈래 조건 만들어 내기에서 알아보기 쉬운 물체가 된다). **다양함**은 줄어든다(표본이 조건부 분포의 봉우리 둘레로 뭉친다). $w = 0$이면 조건 없는 만들어 내기이다(다양함이 가장 크고 조건이 없다). $w = 1$이면 여느 조건 만들어 내기이다. $w$이 크면($>5$) 조건에 아주 충실하지만 되풀이되고 지나치게 포화할 수 있다. 가장 좋은 $w$은 쓰임새에 달렸다. 그림 만들어 내기에서는 $w \approx 2-5$이고, 센 조건을 바라는 글에서 그림으로에서는 더 크다.

---

**연습문제 4.**
익히는 동안 가름개 없는 이끌기를 어떻게 짜는가? 조건을 마구잡이로 떨구는 것이 왜 꼭 필요한가?

??? success "연습문제 4 풀이"
    익히는 동안 조건 신호 $c$을 확률 $p_{\text{uncond}}$(흔히 10-20%)으로 빈 토큰 $\emptyset$으로 마구잡이로 바꾼다. 그러면 신경망 하나로 $\epsilon_\theta(x_t, t, c)$과 $\epsilon_\theta(x_t, t, \emptyset)$을 모두 가르친다. 빈 토큰은 0 벡터나 배운 박아 넣기, 빈 글자열일 수 있다. 마구잡이 떨구기가 꼭 필요한 까닭은 (1) 그것이 없으면 모델이 조건 없는 분포를 배우지 못해 이끌기 공식을 쓸 수 없고, (2) 떨구는 비율이 균형을 다스리며(너무 높으면 조건 배움이 줄고 너무 낮으면 조건 없는 모델이 나빠진다), (3) 은근히 규칙을 세워 조건 있는 만들어 내기와 없는 만들어 내기의 품질을 모두 높이기 때문이다.
