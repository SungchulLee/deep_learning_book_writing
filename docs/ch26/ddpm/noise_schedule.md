# Noise Schedules
The **noise schedule** $\{\beta_t\}_{t=1}^T$ controls how quickly noise is added during the forward diffusion process. This seemingly simple choice has profound effects on model performance and generation quality.

## Why Noise Schedules Matter

The schedule determines:

1. **Information flow**: How quickly data structure is destroyed
2. **Training dynamics**: Which timesteps contribute most to learning
3. **Sample quality**: How well the model can recover fine details
4. **Sampling efficiency**: Trade-offs in accelerated sampling methods

## Key Schedule Parameters

Given $\beta_t$, we derive:

$$
\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{s=1}^{t} \alpha_s
$$

The signal-to-noise ratio at time $t$:

$$
\text{SNR}(t) = \frac{\bar{\alpha}_t}{1 - \bar{\alpha}_t}
$$

## 선형 차례표

### 정의

본디 DDPM은 선형 차례표를 쓴다.

$$
\beta_t = \beta_{\text{start}} + \frac{t-1}{T-1}(\beta_{\text{end}} - \beta_{\text{start}})
$$

**흔한 값**: $\beta_{\text{start}} = 10^{-4}$, $\beta_{\text{end}} = 0.02$, $T = 1000$

### 성질

- $\bar{\alpha}_t$이 거의 지수로 줄어든다
- 망가짐이 대부분 앞 걸음에서 일어난다
- 뒤 걸음($t$이 큼)은 잡음을 거의 더하지 않는다

### 한계

- 모델 담이를 헤프게 쓴다
- 자료가 이미 거의 잡음이어서 뒤 때 걸음이 "낭비"된다
- 자잘한 세부에서 힘겨울 수 있다

## 코사인 차례표

### 뜻매김(Nichol와 Dhariwal, 2021)

$$
\bar{\alpha}_t = \frac{f(t)}{f(0)}, \quad f(t) = \cos\left(\frac{t/T + s}{1 + s} \cdot \frac{\pi}{2}\right)^2
$$

$\beta_t$이 너무 작아지지 않도록 어긋냄 $s = 0.008$을 둔다.

### beta_t 이끌어 내기

$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s = \prod_{s=1}^t (1 - \beta_s)$에서:

$$
\beta_t = 1 - \frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}} = 1 - \frac{f(t)}{f(t-1)}
$$

자르기: 수치의 안정을 위해 $\beta_t = \min(\beta_t, 0.999)$.

### 성질

- $\bar{\alpha}_t$이 더 매끄럽게 줄어든다
- 때 걸음에 걸쳐 잡음 분포가 더 고르다
- 앞 걸음에서 앎을 더 잘 지킨다
- 실제로 표본 품질이 나아진다

## 이차 차례표

### 정의

$$
\beta_t = \beta_{\text{start}} + \frac{(t-1)^2}{(T-1)^2}(\beta_{\text{end}} - \beta_{\text{start}})
$$

### 성질

- 처음에 잡음을 더 천천히 더한다
- 끝으로 갈수록 잡음이 빨리 는다
- 선형과 코사인의 중간

## 시그모이드 차례표

### 정의

$$
\beta_t = \sigma\left(-6 + 12 \cdot \frac{t-1}{T-1}\right) \cdot (\beta_{\text{end}} - \beta_{\text{start}}) + \beta_{\text{start}}
$$

여기서 $\sigma(x) = 1/(1 + e^{-x})$은 시그모이드 함수이다.

### 성질

- S자 모양 옮아감
- 시작과 끝은 완만하고 가운데가 가파르다
- 어떤 경우에는 배운 차례표를 어림한다

## 배운 차례표

붙박인 차례표 대신 최근 연구는 가장 좋은 차례표를 배우는 길을 살핀다.

### 변분 퍼짐 모델(Kingma 외, 2021)

신호 대 잡음비 함수 $\text{SNR}(t)$을 단조 신경망으로 배운다.

$$
\log \text{SNR}(t) = \text{MLP}(t)
$$

양의 무게로 단조성을 지키게 한다.

### 차례표 우려내기

걸음이 더 많은 스승에 맞도록 누른 차례표(걸음이 적음)로 제자 모델을 익힌다.

## 이어진 때로 적기

### 확률 미분 방정식 관점

이어진 때에서 앞 과정은 다음과 같다.

$$
dx = -\frac{1}{2}\beta(t) x \, dt + \sqrt{\beta(t)} \, dW
$$

띄엄띄엄한 차례표 $\beta_t$은 다음으로 $\beta(t)$을 어림한다.

$$
\beta_t \approx \beta(t/T) \cdot \Delta t
$$

### 흔한 이어진 차례표

| 이름 | $\beta(t)$ | 쓰임새 |
|------|------------|----------|
| VP(흩어짐 지키기) | $t$에 선형 | DDPM 같음 |
| VE(흩어짐 터짐) | $\sigma(t)^2$이 는다 | NCSN 같음 |
| Sub-VP | 더 빨리 준다 | 안정된 익히기 |

## 차례표 견주기

### 구현

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

def linear_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)

def cosine_schedule(T, s=0.008):
    t = torch.arange(T + 1)
    f = torch.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return torch.clamp(betas, 0, 0.999)

def quadratic_schedule(T, beta_start=1e-4, beta_end=0.02):
    t = torch.arange(T)
    return beta_start + (t / (T - 1)) ** 2 * (beta_end - beta_start)

def sigmoid_schedule(T, beta_start=1e-4, beta_end=0.02):
    t = torch.linspace(-6, 6, T)
    betas = torch.sigmoid(t) * (beta_end - beta_start) + beta_start
    return betas

# 차례표마다 alpha_bar을 셈한다
def compute_alpha_bar(betas):
    alphas = 1 - betas
    return torch.cumprod(alphas, dim=0)
```

### 눈으로 견주기

| 때 걸음(%) | 선형 $\bar{\alpha}$ | 코사인 $\bar{\alpha}$ |
|--------------|----------------------|----------------------|
| 10% | 0.95 | 0.98 |
| 25% | 0.80 | 0.90 |
| 50% | 0.50 | 0.70 |
| 75% | 0.15 | 0.35 |
| 100% | 0.00 | 0.00 |

## 실전 권고

### 그림 만들어 내기에서

- **코사인 차례표**가 흔히 가장 잘 듣는다
- 익힐 때는 $T = 1000$을 쓰고 뽑을 때는 줄일 수 있다
- 수치 문제를 피하려 $\beta_t$을 가둔다

### 소리와 영상에서

- 더 긴 차례표($T = 2000-4000$)가 도움이 될 수 있다
- 마당에 맞는 가장 좋은 값을 위해 배운 차례표를 살펴보라

### 빠른 뽑기에서

- DDIM과 DPM-Solver은 매끄러운 차례표에서 더 잘 듣는다
- 코사인 차례표는 걸음을 더 과감히 건너뛸 수 있게 한다

## 익히기 목표와의 이음

차례표는 익히기 손실의 무게 매기기에 영향을 준다.

$$
\mathcal{L} = \mathbb{E}_{t, x_0, \epsilon}\left[ w(t) \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]
$$

차례표가 다르면 은근히 다른 $w(t)$을 쓰게 된다. 코사인 차례표는 때 걸음에 걸쳐 더 고른 무게를 준다.

## 요약

잡음 차례표는 앞 퍼짐의 움직임을 다스리는 결정적인 웃매개변수이다. 선형 차례표는 단순하고 역사에서 중요하지만, 코사인 차례표가 때 걸음에 걸쳐 앎의 잃음을 더 고르게 나누어 흔히 더 좋은 결과를 준다. 앞선 방법은 배운 차례표와 이어진 때 차례표로 더 나아지려 한다.

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
