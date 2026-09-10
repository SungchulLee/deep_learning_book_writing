# 랑주뱅 동역학의 바탕
랑주뱅 동역학은 MCMC 표집과 기울기 기반 최적화를 잇는 이어진 시간 얼개를 준다. 랑주뱅 확률 미분 방정식(SDE)은 기울기가 이끄는 흐름에 브라운 잡음이 더해진 알갱이를 그리며, 그 멈춘 분포가 과녁 뒤확률이다. 이 이음이 실전 표집 알고리즘과 현대의 점수 기반 낳는 모형을 함께 떠받친다.

---

## 1. 물리에서의 뿌리: 브라운 운동

1908년 폴 랑주뱅은 **브라운 운동**, 곧 액체에 떠 있는 알갱이가 둘레 분자와 부딪히며 마구 움직이는 것을 그리는 방정식을 내놓았다. 이 방정식은 (퍼텐셜 에너지에서 오는) 정해진 힘과 (열 잡음에서 오는) 무작위 흔들림을 하나로 꿰었다.

놀라운 통찰은 이것이다. 곧 물리 알갱이를 그리는 그 수학으로 **확률 분포에서 표집**할 수 있다.

### 고전 랑주뱅 방정식

퍼텐셜 $U(x)$ 안에서 움직이는, 자리가 $x$이고 질량이 $m$인 알갱이를 생각하자:

$$
m\frac{d^2 x}{dt^2} = -\gamma \frac{dx}{dt} - \nabla U(x) + \sqrt{2\gamma k_B T} \, \xi(t)
$$

여기서 각 기호는 다음과 같다.

- $\gamma$은 **마찰 계수**이다(끈적임에 따른 잦아듦)
- $k_B T$은 **열 에너지**이다(볼츠만 상수 곱하기 온도)
- $\xi(t)$은 $\langle \xi(t) \xi(t') \rangle = \delta(t - t')$을 만족하는 **흰 잡음**이다

세 힘은 다음과 같다:

| 힘 | 식 | 물리적 뜻 |
|-------|------------|------------------|
| 마찰 | $-\gamma \dot{x}$ | 움직임을 거스르며 에너지를 흩뜨린다 |
| 퍼텐셜 | $-\nabla U(x)$ | 에너지가 낮은 곳으로 민다 |
| 열 | $\sqrt{2\gamma k_B T} \, \xi(t)$ | 분자와 부딪혀 받는 무작위 걷어참 |

### 지나치게 잦아든 극한

관성이 하찮은($m \to 0$) **마찰이 큰 자리**에서는 가속 항이 사라진다. 그러면 **지나치게 잦아든 랑주뱅 방정식**이 나온다:

$$
\gamma \frac{dx}{dt} = -\nabla U(x) + \sqrt{2\gamma k_B T} \, \xi(t)
$$

시간을 $\gamma$으로 다시 재고 $k_B T = 1$으로 놓으면:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2} \, dW_t
$$

여기서 $W_t$은 표준 **브라운 운동**(위너 과정)이다.

!!! info "물리에서 표집으로"
    온도 $T=1$에서 볼츠만 분포는 $\pi(x) \propto \exp(-U(x))$이다. 랑주뱅 방정식은 이 분포로 모인다. 곧 알갱이가 에너지가 낮은 곳에서 더 오래 머문다!

---

## 2. 표집을 위한 랑주뱅 SDE

### 정식화

과녁 분포 $\pi(x) \propto \exp(-U(x))$에서 표집하려고 다음 SDE을 흉내 낸다:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2} \, dW_t
$$

마찬가지로 점수 함수 $s(x) = \nabla_x \log \pi(x) = -\nabla U(x)$을 쓰면:

$$
\boxed{dx_t = s(x_t) \, dt + \sqrt{2} \, dW_t}
$$

### 두 항 풀이하기

**흐름 항** $s(x_t) \, dt$:

- 확률이 높은 구역으로 가는 정해진 흐름
- 확률 지형에서 "오르막"을 가리킨다
- 봉우리를 써먹게 이끈다

**퍼짐 항** $\sqrt{2} \, dW_t$:

- 브라운 운동으로 하는 무작위 살펴보기
- 봉우리에 갇히는 것을 막는다
- 분포 전체를 살펴볼 수 있게 한다

$dW_t$ 앞의 상수 $\sqrt{2}$은 아무렇게나 정한 것이 아니다. 멈춘 분포가 $\pi(x)$이 되도록 정확히 맞춘 값이다.

---

## 3. 점수 함수

**점수 함수** $\mathbf{s}(x) = \nabla_x \log \pi(x)$은 랑주뱅 방법의 중심 양이다:

$$
\nabla_x \log \pi(x) = \nabla_x \log p(\mathcal{D} \mid x) + \nabla_x \log p(x)
$$

성질:

- 뒤확률 밀도가 커지는 쪽을 가리킨다
- 크기는 그 자리 기울기의 가파름을 비춘다
- 봉우리에서는 $\mathbf{s}(x^*) = \mathbf{0}$이다
- 고르게 하는 상수가 필요 없다: $\nabla_x \log(\tilde{\pi}(x)/Z) = \nabla_x \log \tilde{\pi}(x)$

---

## 4. 온도를 넣은 일반 꼴

온도 $T$에서 과녁 분포는 $\pi_T(x) \propto \exp(-U(x)/T)$이고 SDE은 다음처럼 된다:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2T} \, dW_t
$$

| 온도 | 잡음 크기 | 굴러감 |
|-------------|-------------|-----------|
| 높은 $T$ | 큰 $\sqrt{2T}$ | 널리 살펴보고 봉우리가 흐려진다 |
| $T = 1$ | $\sqrt{2}$ | $\pi$에서의 표준 표집 |
| 낮은 $T$ | 작은 $\sqrt{2T}$ | 봉우리 가까이 몰린다 |
| $T \to 0$ | 잡음 없음 | 최솟값으로 가는 정해진 기울기 내려가기 |

---

## 5. 포커-플랑크 방정식

### 밀도의 흐름

**포커-플랑크 방정식**(앞 콜모고로프 방정식)은 $x_t$의 확률 밀도 $\rho(x, t)$이 어떻게 흘러가는지를 그린다:

$$
\frac{\partial \rho}{\partial t} = -\nabla \cdot (\rho \, \nabla \log \pi) + \Delta \rho
$$

여기서 $\Delta = \nabla \cdot \nabla$은 라플라스 연산자이다.

### 멈춘 풀이

멈춤에서($\partial \rho / \partial t = 0$) $\rho = \pi$이 멈춘 풀이라고 주장한다.

**증명**: $\rho = \pi$을 넣으면:

$$
-\nabla \cdot (\pi \nabla \log \pi) + \Delta \pi = -\nabla \cdot \left(\pi \cdot \frac{\nabla \pi}{\pi}\right) + \Delta \pi = -\Delta \pi + \Delta \pi = 0 \quad \checkmark
$$

### 우아하게 고쳐 쓰기

포커-플랑크 방정식은 다음처럼 고쳐 쓸 수 있다:

$$
\frac{\partial \rho}{\partial t} = \nabla \cdot \left( \pi \nabla \left( \frac{\rho}{\pi} \right) \right)
$$

이 꼴은 $\rho = \pi$이 멈춰 있음을 뻔하게 보여 준다(상수의 기울기는 사라진다). 게다가 이는 랑주뱅 동역학이 확률 분포의 공간에서 **기울기 흐름**을 하며 $\rho$에서 $\pi$까지의 KL 갈림을 가장 작게 함을 보여 준다.

---

## 6. 모임 보장

### 모임 정리

$U(x)$에 대한 순한 조건 아래:

1. **강한 볼록성**: $U$이 $m$-강볼록이면($\nabla^2 U \succeq m I$) KL 갈림이 지수로 오그라든다:
   
   $$
   D_{KL}(\rho_t \| \pi) \leq e^{-2mt} D_{KL}(\rho_0 \| \pi)
   $$

2. **일반 경우**: 립시츠 기울기와 흩뜨림 조건 아래 사슬은 기하적으로 에르고드이다.

### 섞임 시간

$d$차원 가우스 과녁에서 섞임 시간은 $\mathcal{O}(d \cdot \kappa)$으로 커진다. 여기서 $\kappa$은 조건수이다.

---

## 7. 잘게 나누기

이어진 SDE을 셈하려면 잘게 나눠야 한다. 오일러-마루야마 방식은 다음을 준다:

$$
x_{t+1} = x_t + \epsilon \, s(x_t) + \sqrt{2\epsilon} \, \eta_t, \quad \eta_t \sim \mathcal{N}(0, I)
$$

이는 **잘게 나눔 치우침**을 들여온다. 두 길이 이를 다룬다:

1. **바로잡지 않은 랑주뱅 알고리즘(ULA)**: 치우침을 받아들인다([ULA](ula.md)를 보아라)
2. **메트로폴리스 바로잡은 랑주뱅(MALA)**: MH 바로잡기를 더한다([MALA](mala.md)를 보아라)

### 안정 조건

이차 퍼텐셜 $U(x) = \frac{1}{2}x^\top H x$에서는 안정을 위해 $\epsilon < 2/\lambda_{\max}(H)$이 필요하다.

---

## 8. 담금질한 랑주뱅 동역학

표준 랑주뱅은 봉우리가 여럿인 분포에서 애를 먹는다. 온도 담금질은 시간에 따라 줄어드는 일정 $T(t)$을 들여온다:

$$
dx_t = -\nabla U(x_t) \, dt + \sqrt{2T(t)} \, dW_t
$$

마찬가지로 잡음 수준 $\sigma_1 > \sigma_2 > \cdots > \sigma_L$을 정하고 수준마다 랑주뱅을 돌린다:

```
Initialise x ~ N(0, σ₁² I)
For l = 1, ..., L:
    ε = α σₗ²
    For k = 1, ..., K:
        x ← x + (ε/2) s(x, σₗ) + √ε η,  η ~ N(0, I)
    σₗ → σₗ₊₁
Return x
```

이 알고리즘이 **점수 기반 낳는 모형**과 **퍼짐 모형**의 바탕이다. 온전한 이음은 [점수 맞추기, 랑주뱅, 퍼짐](score_and_diffusion.md)을 보아라.

---

## 9. 최적화와의 이음

### 온도 0 극한으로서의 기울기 내려가기

잡음을 0으로 놓으면 $\log \pi(x)$에 대한 정해진 기울기 올라가기가 되어 MAP 어림값으로 모인다. 랑주뱅 동역학은 뒤확률 전체를 살펴보려고 잡음을 더한다.

### SGD과의 이음

어떤 조건 아래에서 SGD은 배움율과 기울기 잡음의 흩어짐에 비례하는 온도를 갖는 랑주뱅 동역학을 어림한다. 이는 SGD이 왜 "평평한" 최솟값을 찾는지, 배움율 일정이 왜 온도 담금질처럼 구는지를 알려 준다.

### HMC과의 이음

HMC는 보조 운동량을 두어 퍼져 나가는 대신 탄도처럼 살펴보는 **이차** 랑주뱅 방법이며, 높은 차원에서 훨씬 효율적이다.

---

## 10. 미리 다듬기

과녁이 방향마다 눈금이 다르면 **미리 다듬은 랑주뱅**을 써라:

$$
dx_t = M^{-1} \nabla \log \pi(x_t) \, dt + \sqrt{2} \, M^{-1/2} dW_t
$$

여기서 $M \approx -\nabla^2 \log \pi(x^*)$이 분포를 희게 만든다.

---

## 11. PyTorch 구현

```python
import torch

class LangevinSDE:
    """오일러-마루야마로 푸는 이어진 시간 랑주뱅 움직임."""
    
    def __init__(self, score_fn, dim, temperature=1.0):
        self.score_fn = score_fn
        self.dim = dim
        self.temperature = temperature
    
    def step(self, x, dt):
        score = self.score_fn(x)
        drift = score * dt
        diffusion = torch.sqrt(2 * self.temperature * torch.tensor(dt)) * torch.randn_like(x)
        return x + drift + diffusion
    
    def sample(self, x0, n_steps, dt, return_trajectory=False):
        x = x0.clone()
        if return_trajectory:
            trajectory = [x.clone()]
        for _ in range(n_steps):
            x = self.step(x, dt)
            if return_trajectory:
                trajectory.append(x.clone())
        if return_trajectory:
            return torch.stack(trajectory)
        return x

# 보기: 2차원 가우스
mu = torch.tensor([2.0, -1.0])
cov = torch.tensor([[1.0, 0.6], [0.6, 1.0]])
precision = torch.linalg.inv(cov)

score_fn = lambda x: -torch.matmul(x - mu, precision)
sampler = LangevinSDE(score_fn, dim=2)

x0 = torch.randn(500, 2) * 3 + torch.tensor([5.0, 5.0])
x = x0.clone()
for _ in range(1000):
    x = sampler.step(x, dt=0.1)

print(f"Sample mean: [{x[:, 0].mean():.3f}, {x[:, 1].mean():.3f}]")
print(f"True mean:   [{mu[0]:.3f}, {mu[1]:.3f}]")
```

---

## 연습문제

1. **포커-플랑크 확인.** $\rho = \pi \propto \exp(-U)$에서 시작해 포커-플랑크 오른쪽 변이 사라지는지 확인하여라.

2. **온도의 효과.** 여러 온도를 살펴보도록 구현을 고쳐라. 쌍봉 분포에서 낮은 온도가 표본을 온 세상 봉우리에 어떻게 몰아 넣는지, 높은 온도가 봉우리 둘을 어떻게 살펴보게 하는지 보여라.

3. **ULA의 치우침.** 1차원 가우스 $\pi(x) = \mathcal{N}(0, 1)$에서 ULA의 멈춘 분포를 $\epsilon$의 함수로 이끌어 내고 $\epsilon > 0$이면 치우쳐 있음을 보여라.

4. **담금질 실험.** 봉우리가 뚜렷이 갈라진 2차원 가우스 섞음에 담금질한 랑주뱅 표집을 구현하여라. 담금질을 할 때와 안 할 때의 모임을 견주어라.

5. **모임 속도.** 흩어짐이 $\sigma^2$인 1차원 가우스에서 표본 흩어짐이 참 흩어짐으로 모이는 속도를 시간과 걸음 크기의 함수로 이끌어 내어라.

---

## 정리하며

| 부분 | 식 | 몫 |
|-----------|------------|------|
| **SDE** | $dx_t = s(x_t) dt + \sqrt{2} dW_t$ | 동역학의 정의 |
| **흐름** | $s(x) = \nabla \log \pi(x)$ | 확률이 높은 쪽으로 움직인다 |
| **퍼짐** | $\sqrt{2} dW_t$ | 살펴보기를 가능하게 한다 |
| **멈춘 분포** | $\pi(x)$ | 표본이 이것으로 모인다 |
| **포커-플랑크** | $\partial_t \rho = -\nabla \cdot (\rho s) + \Delta \rho$ | 밀도의 흐름 |
| **온도** | 잡음의 눈금을 정하고 담금질을 가능하게 한다 | |
| **잡음 없음** | 기울기 올라가기(MAP)로 되돌아간다 | |

---

**참고 문헌**

1. Langevin, P. (1908). Sur la théorie du mouvement brownien. *Comptes Rendus de l'Académie des Sciences*, 146, 530-533.
2. Gardiner, C. W. (2009). *Stochastic Methods: A Handbook for the Natural and Social Sciences*. Springer.
3. Pavliotis, G. A. (2014). *Stochastic Processes and Applications*. Springer.
4. Roberts, G. O., & Tweedie, R. L. (1996). Exponential convergence of Langevin distributions and their discrete approximations. *Bernoulli*, 2(4), 341-363.
5. Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
6. Song, Y., & Ermon, S. (2019). Generative modeling by estimating gradients of the data distribution. *NeurIPS*.
