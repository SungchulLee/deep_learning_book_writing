# 베이즈 신경 그물의 뒷분포 미루어 봄
**뒷분포 미루어 봄**은 베이즈 신경 그물에서 셈으로 부딪히는 가장 큰 어려움이다. 앞선 분포 $p(\theta)$과 그럴듯함 $p(\mathcal{D} \mid \theta)$이 주어지면 뒷분포 $p(\theta \mid \mathcal{D})$을 찾는다. 신경 그물에서는 이 뒷분포를 다룰 수 없으므로, 표본 뽑기(MCMC)에서 가장 좋게 하기(변이 미루어 봄), 넌지시 하는 어림(드롭아웃, 모둠)에 이르는 어림 방법이 있어야 한다.

---

## 미루어 봄의 어려움

### 뒷분포

베이즈 정리에 따르면 그물 짐에 대한 뒷분포는 이렇다.

$$
\boxed{p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}}
$$

여기서

- $p(\mathcal{D} \mid \theta) = \prod_{i=1}^N p(y_i \mid x_i, \theta)$은 그럴듯함
- $p(\theta)$은 앞선 분포
- $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta$은 밑거리(가장자리 그럴듯함)

### 정확한 미루어 봄을 다룰 수 없는 까닭

**1. 높은 차수**: 요즘 그물의 매개변수는 $d = 10^6$에서 $10^9$이다

**2. 짝이 맞지 않음**: 신경 그물의 그럴듯함은 여느 앞선 분포와 짝이 맞지 않는다.

$$
p(y \mid x, \theta) = \mathcal{N}(y \mid f_\theta(x), \sigma^2)
$$

여기서 $f_\theta$은 얽히고 곧지 않은 함수다.

**3. 다룰 수 없는 잣대 맞추기**: 밑거리 적분에는 닫힌 꼴이 없다.

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta
$$

**4. 여러 봉우리**: 뒷분포의 터에는 다음 까닭으로 봉우리가 많다.

- 짐 밭의 맞바꿈 대칭(자리 바꾸기, 잣대 바꾸기)
- 좋은 풀이가 여럿임
- 잃음 낯이 얽힘

### 미루어 봄 방법에 바라는 것

| 결 | 풀이 |
|----------|-------------|
| **크게 늘리기** | 매개변수 수백만 개를 다룬다 |
| **맞음** | 뒷분포의 꼴을 미덥게 담는다 |
| **아리송함** | 눈금이 잘 맞은 아리송함을 낸다 |
| **잘 듦** | 셈하는 때가 알맞다 |
| **단순함** | 짜고 맞추기가 쉽다 |

모든 잣대에서 앞서는 방법 하나는 없다. 방법마다 이 결들을 맞바꾼다.

---

## 미루어 봄 방법 두루 보기

### 갈래 나누기

```
뒷분포 미루어 봄 방법
├── 표본 뽑기 방법(MCMC)
│   ├── 메트로폴리스-헤이스팅스
│   ├── 해밀턴 몬테카를로(HMC)
│   ├── 확률 기울기 MCMC
│   │   ├── SGLD(랑주뱅 움직임)
│   │   ├── SGHMC(해밀턴)
│   │   └── SGFS(피셔 점수 매기기)
│   └── 모둠 방법
├── 변이 미루어 봄
│   ├── 평균 마당 VI
│   ├── 온전한 함께 바뀜 VI
│   └── 잣대 맞추는 흐름
├── 라플라스 어림
│   ├── 온전한 라플라스
│   ├── 대각 라플라스
│   └── KFAC 라플라스
└── 넌지시 하는 방법
    ├── MC 드롭아웃
    ├── 깊은 모둠
    └── SWAG
```

### 방법 견주기

| 방법 | 맞음 | 크게 늘리기 | 단순함 | 값 |
|--------|----------|-------------|------------|------|
| HMC | 높음 | 낮음 | 낮음 | 아주 비쌈 |
| SGLD | 가운데 | 높음 | 가운데 | 쌈 |
| 평균 마당 VI | 낮음~가운데 | 높음 | 가운데 | 가운데 |
| 라플라스 | 가운데 | 가운데 | 높음 | 쌈 |
| MC 드롭아웃 | 낮음~가운데 | 아주 높음 | 아주 높음 | 아주 쌈 |
| 깊은 모둠 | 가운데~높음 | 높음 | 높음 | 가운데 |

---

## 마르코프 사슬 몬테카를로(MCMC)

### 밑바탕

MCMC는 머문 분포가 뒷분포가 되는 마르코프 사슬을 짓는다.

$$
\theta^{(t+1)} \sim T(\theta^{(t+1)} \mid \theta^{(t)})
$$

그리하여 $t \to \infty$일 때 $\theta^{(t)} \to p(\theta \mid \mathcal{D})$이 된다.

**표본 쓰기**: 표본 $\{\theta^{(t)}\}_{t=1}^T$이 주어지면 바람을 어림한다.

$$
\mathbb{E}_{p(\theta \mid \mathcal{D})}[f(\theta)] \approx \frac{1}{T} \sum_{t=1}^T f(\theta^{(t)})
$$

### 메트로폴리스-헤이스팅스

**알고리즘**:

1. $\theta' \sim q(\theta' \mid \theta^{(t)})$을 내놓는다
2. 받을 낌새를 셈한다.

$$
\alpha = \min\left(1, \frac{p(\theta' \mid \mathcal{D}) \, q(\theta^{(t)} \mid \theta')}{p(\theta^{(t)} \mid \mathcal{D}) \, q(\theta' \mid \theta^{(t)})}\right)
$$

3. 낌새 $\alpha$으로 받는다: $\theta^{(t+1)} = \theta'$ 또는 $\theta^{(t+1)} = \theta^{(t)}$

**신경 그물에서는**: 아무렇게나 걷는 내놓기($q(\theta' \mid \theta) = \mathcal{N}(\theta, \epsilon^2 I)$)는 차수가 높으면 잘 듣지 않는다.

### 해밀턴 몬테카를로(HMC)

HMC는 기울기 소식을 써서 앎에 바탕한 내놓기를 한다.

**덧댄 얼개**: 밀어 나감 $\rho$을 들여 해밀턴 값을 매긴다.

$$
H(\theta, \rho) = -\log p(\theta \mid \mathcal{D}) + \frac{1}{2}\rho^\top M^{-1} \rho
$$

**해밀턴 움직임**:

$$
\frac{d\theta}{dt} = M^{-1} \rho, \quad \frac{d\rho}{dt} = \nabla_\theta \log p(\theta \mid \mathcal{D})
$$

**개구리뜀 적분기**(걸음 크기 $\epsilon$으로 $L$걸음):

$$
\rho^{(t+1/2)} = \rho^{(t)} + \frac{\epsilon}{2} \nabla_\theta \log p(\theta^{(t)} \mid \mathcal{D})
$$

$$
\theta^{(t+1)} = \theta^{(t)} + \epsilon \, M^{-1} \rho^{(t+1/2)}
$$

$$
\rho^{(t+1)} = \rho^{(t+1/2)} + \frac{\epsilon}{2} \nabla_\theta \log p(\theta^{(t+1)} \mid \mathcal{D})
$$

**HMC 알고리즘**:

1. 밀어 나감을 뽑는다: $\rho \sim \mathcal{N}(0, M)$
2. 개구리뜀을 $L$걸음 돌린다
3. 메트로폴리스 바로잡기로 받거나 물린다

**나은 점**: 뒷분포를 잘 둘러보고 표본끼리 덜 얽힌다.

**신경 그물에서의 어려움**: 온전한 기울기 셈이 있어야 한다(큰 자료 꾸러미에서는 비싸다).

---

## 확률 기울기 MCMC

### 왜 하는가

온 묶음 기울기는 비싸다. 확률 기울기 MCMC는 잔 묶음 기울기를 쓴다.

$$
\nabla_\theta \log p(\theta \mid \mathcal{D}) \approx \nabla_\theta \log p(\theta) + \frac{N}{|B|} \sum_{i \in B} \nabla_\theta \log p(y_i \mid x_i, \theta)
$$

여기서 $B$은 크기가 $|B|$인 잔 묶음이다.

### 확률 기울기 랑주뱅 움직임(SGLD)

**고치는 규칙**:

$$
\boxed{\theta^{(t+1)} = \theta^{(t)} + \frac{\epsilon_t}{2} \nabla_\theta \log p(\theta^{(t)} \mid \mathcal{D}) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \epsilon_t I)}
$$

**고갱이 깨침**: 걸음 크기가 $\epsilon_t \to 0$으로 줄면 메트로폴리스 받기 걸음을 건너뛸 수 있다.

**걸음 크기 짜임**: 다음을 채워야 한다.

$$
\sum_{t=1}^\infty \epsilon_t = \infty, \quad \sum_{t=1}^\infty \epsilon_t^2 < \infty
$$

흔한 고름: $\gamma \in (0.5, 1]$인 $\epsilon_t = a(b + t)^{-\gamma}$.

**참으로 쓰는 SGLD**:

$$
\theta^{(t+1)} = \theta^{(t)} + \frac{\epsilon_t}{2} \left[ \nabla_\theta \log p(\theta^{(t)}) + \frac{N}{|B|} \sum_{i \in B} \nabla_\theta \log p(y_i \mid x_i, \theta^{(t)}) \right] + \eta_t
$$

### 확률 기울기 해밀턴 몬테카를로(SGHMC)

더 잘 둘러보도록 SGLD에 밀어 나감을 더한다.

$$
\theta^{(t+1)} = \theta^{(t)} + \rho^{(t)}
$$

$$
\rho^{(t+1)} = (1 - \alpha) \rho^{(t)} + \epsilon_t \nabla_\theta \log p(\theta^{(t)} \mid \mathcal{D}) + \eta_t
$$

여기서 $\alpha$은 쓸림 값이고 $\eta_t \sim \mathcal{N}(0, 2\alpha\epsilon_t I)$이다.

### 미리 다듬은 SGLD

잣대가 더 잘 맞도록 미리 다듬는 행렬 $G(\theta)$을 쓴다.

$$
\theta^{(t+1)} = \theta^{(t)} + \frac{\epsilon_t}{2} \left[ G(\theta^{(t)}) \nabla_\theta \log p(\theta^{(t)} \mid \mathcal{D}) + \Gamma(\theta^{(t)}) \right] + \eta_t
$$

여기서 $\eta_t \sim \mathcal{N}(0, \epsilon_t G(\theta^{(t)}))$이고 $\Gamma$은 바로잡는 항이다.

**$G$으로 흔히 고르는 것**:

- RMSprop 미리 다듬개
- Adam 미리 다듬개
- 피셔 소식 행렬

### 돌림 SGLD

그 자리 봉우리를 벗어나도록 돌림 배움 비율을 쓴다.

$$
\epsilon_t = \epsilon_0 \left( \cos\left(\frac{\pi \, \text{mod}(t, T_{\text{cycle}})}{T_{\text{cycle}}}\right) + 1 \right) / 2
$$

걸음 크기가 작아지는 돌림의 끝에서 표본을 모은다.

---

## 라플라스 어림

### 깨침

뒷분포를 MAP 어림을 가운데로 삼는 가우스로 어림한다.

$$
\boxed{p(\theta \mid \mathcal{D}) \approx q(\theta) = \mathcal{N}(\theta \mid \hat{\theta}_{\text{MAP}}, \Sigma)}
$$

여기서

- $\hat{\theta}_{\text{MAP}} = \arg\max_\theta \log p(\theta \mid \mathcal{D})$
- $\Sigma = \left[ -\nabla^2_\theta \log p(\theta \mid \mathcal{D}) \big|_{\hat{\theta}_{\text{MAP}}} \right]^{-1}$

### 이끌어 내기

로그 뒷분포를 MAP 언저리에서 테일러로 펼친다.

$$
\log p(\theta \mid \mathcal{D}) \approx \log p(\hat{\theta} \mid \mathcal{D}) - \frac{1}{2}(\theta - \hat{\theta})^\top H (\theta - \hat{\theta})
$$

여기서 $H = -\nabla^2_\theta \log p(\theta \mid \mathcal{D})|_{\hat{\theta}}$은 헤세 행렬이다.

지수를 취하면 함께 바뀜이 $\Sigma = H^{-1}$인 가우스가 된다.

### 헤세 행렬 셈하기

**온전한 헤세 행렬**: 자리 $O(d^2)$, 거꿀 셈 $O(d^3)$ — 큰 그물에서는 다룰 수 없다.

**대각 어림**:

$$
\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)
$$

여기서 $\sigma_i^2 = 1/H_{ii}$이다.

**크로네커로 쪼갠 것(KFAC)**:

짐이 $W^{(l)}$인 켜 $l$에서

$$
H^{(l)} \approx A^{(l)} \otimes G^{(l)}
$$

여기서

- $A^{(l)} = \mathbb{E}[a^{(l-1)} (a^{(l-1)})^\top]$(들임 살림)
- $G^{(l)} = \mathbb{E}[g^{(l)} (g^{(l)})^\top]$(날임 기울기)

**거꿀**:

$$
(A \otimes G)^{-1} = A^{-1} \otimes G^{-1}
$$

켜마다 $O(d^3)$을 $O(n_l^3 + n_{l-1}^3)$으로 줄인다.

### 마지막 켜 라플라스

앞 켜는 붙박아 두고 마지막 켜에만 라플라스를 건다.

$$
p(\theta_L \mid \mathcal{D}, \theta_{1:L-1}) \approx \mathcal{N}(\theta_L \mid \hat{\theta}_L, \Sigma_L)
$$

**나은 점**:

- 헤세 행렬이 훨씬 작다
- 아리송함의 거의를 담는 일이 잦다
- 결 뽑개는 붙박인 채로 남는다

### 미루어 보는 분포

가우스 그럴듯함을 쓰는 되돌이에서

$$
p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta) \, q(\theta) \, d\theta
$$

MAP 언저리에서 **곧게 펴기**:

$$
f_\theta(x) \approx f_{\hat{\theta}}(x) + J_{\hat{\theta}}(x)(\theta - \hat{\theta})
$$

여기서 $J_{\hat{\theta}}(x) = \nabla_\theta f_\theta(x)|_{\hat{\theta}}$은 야코비 행렬이다.

**미루어 본 흩어짐**:

$$
\text{Var}[f(x^*)] \approx J_{\hat{\theta}}(x^*)^\top \Sigma \, J_{\hat{\theta}}(x^*)
$$

---

## 변이 미루어 봄

### 변이 목표

KL 갈림을 가장 작게 하여 $p(\theta \mid \mathcal{D})$을 다룰 수 있는 분포 $q_\phi(\theta)$으로 어림한다.

$$
\phi^* = \arg\min_\phi \text{KL}(q_\phi(\theta) \| p(\theta \mid \mathcal{D}))
$$

**밑거리 아래끝(ELBO)**:

$$
\boxed{\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi(\theta) \| p(\theta))}
$$

**이끌어 내기**:

$$
\log p(\mathcal{D}) = \mathcal{L}(\phi) + \text{KL}(q_\phi \| p(\theta \mid \mathcal{D})) \geq \mathcal{L}(\phi)
$$

ELBO를 가장 크게 하는 일은 뒷분포에 대한 KL을 가장 작게 하는 일과 같다.

### 평균 마당 변이 미루어 봄

**곱으로 가른 어림**:

$$
q_\phi(\theta) = \prod_{i=1}^d q_{\phi_i}(\theta_i)
$$

**가우스 평균 마당**:

$$
q_\phi(\theta) = \prod_{i=1}^d \mathcal{N}(\theta_i \mid \mu_i, \sigma_i^2)
$$

매개변수: $\phi = \{\mu_i, \sigma_i\}_{i=1}^d$(양이 되게 하려면 $\log \sigma_i$).

### 매개변수 다시 잡기 재주

확률 표본 뽑기를 지나 기울기를 셈하려면

$$
\theta = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**ELBO의 기울기**:

$$
\nabla_\phi \mathcal{L} = \nabla_\phi \mathbb{E}_{\epsilon}[\log p(\mathcal{D} \mid \mu + \sigma \odot \epsilon)] - \nabla_\phi \text{KL}(q_\phi \| p)
$$

첫째 항은 몬테카를로로 어림하고, 둘째 항은 흔히 닫힌 꼴을 지닌다.

### 가우스 앞선 분포의 KL 갈림

$q(\theta) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$과 $p(\theta) = \mathcal{N}(0, \sigma_0^2 I)$에서

$$
\text{KL}(q \| p) = \frac{1}{2} \sum_{i=1}^d \left[ \frac{\mu_i^2 + \sigma_i^2}{\sigma_0^2} - 1 - \log\frac{\sigma_i^2}{\sigma_0^2} \right]
$$

### 되돌아가며 베이즈

**알고리즘**(블런델 등, 2015):

1. $\epsilon \sim \mathcal{N}(0, I)$을 뽑는다
2. $\theta = \mu + \log(1 + e^\rho) \odot \epsilon$을 셈한다($\sigma$에 소프트플러스)
3. 잃음을 셈한다: $\mathcal{L} = \log q_\phi(\theta) - \log p(\theta) - \log p(\mathcal{D} \mid \theta)$
4. 되돌아가며 $\phi = \{\mu, \rho\}$을 고친다

**잔 묶음 ELBO**:

$$
\mathcal{L} \approx \frac{N}{|B|} \sum_{i \in B} \log p(y_i \mid x_i, \theta) - \text{KL}(q_\phi \| p)
$$

### 평균 마당을 넘어

**온전한 함께 바뀜**: $q(\theta) = \mathcal{N}(\mu, \Sigma)$

- 매개변수가 $O(d^2)$개 — 흔히 다룰 수 없다

**낮은 자리 더하기 대각**:

$$
\Sigma = D + VV^\top
$$

여기서 $D$은 대각이고 $V \in \mathbb{R}^{d \times r}$의 자리는 $r \ll d$이다.

**잣대 맞추는 흐름**: 되돌릴 수 있는 함수로 단순한 분포를 바꾼다

$$
q(\theta) = q_0(f^{-1}(\theta)) \left| \det \frac{\partial f^{-1}}{\partial \theta} \right|
$$

---

## 넌지시 하는 변이 방법

### 깊은 모둠

첫값을 달리해 그물 $M$개를 서로 남남으로 익힌다.

$$
\{\theta^{(m)}\}_{m=1}^M \quad \text{where each } \theta^{(m)} = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D})
$$

**미루어 보는 분포**:

$$
p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{M} \sum_{m=1}^M p(y^* \mid x^*, \theta^{(m)})
$$

**풀이**: 서로 다른 봉우리를 뽑는 넌지시 하는 뒷분포 어림이다.

**나은 점**:

- 짜기가 단순하다
- 나란히 하기가 더없이 쉽다
- 눈금이 잘 맞는 일이 잦다

**아쉬운 점**:

- 익힘 값이 $M$배
- 자리와 미루어 봄 값이 $M$배
- 제대로 된 베이즈 방법은 아니다

### 확률 짐 고르기 가우스(SWAG)

SGD으로 익히는 동안 자를 모은다.

**달리는 자**:

$$
\bar{\theta} = \frac{1}{T} \sum_{t=1}^T \theta^{(t)}
$$

$$
\bar{\theta^2} = \frac{1}{T} \sum_{t=1}^T (\theta^{(t)})^2
$$

**대각 흩어짐**:

$$
\Sigma_{\text{diag}} = \text{diag}(\bar{\theta^2} - \bar{\theta}^2)
$$

**낮은 자리 몫**(벗어남에서):

$$
D = [\theta^{(t_1)} - \bar{\theta}, \ldots, \theta^{(t_K)} - \bar{\theta}]
$$

**SWAG 뒷분포**:

$$
q(\theta) = \mathcal{N}\left(\bar{\theta}, \frac{1}{2}(\Sigma_{\text{diag}} + \frac{1}{K-1}DD^\top)\right)
$$

### MC 드롭아웃

어림 변이 미루어 봄으로 시험할 때 드롭아웃을 쓴다.

$$
q(\theta) = \prod_l q(W^{(l)})
$$

여기서 $q(W^{(l)})$은 기둥이 아무렇게나 0으로 되는 분포다.

자세한 것은 MC 드롭아웃을 다룬 장을 보라.

---

## 참으로 헤아릴 것

### 미루어 봄 방법 고르기

**SGLD를 쓸 때**:

- 이론에 뿌리내린 표본이 있어야 할 때
- 오래 익힐 수 있을 때
- 뒷분포의 여러 봉우리가 종요로울 때

**라플라스 어림을 쓸 때**:

- 이미 익힌 그물이 있을 때(일 끝난 뒤 아리송함)
- 아리송함 어림이 빨리 있어야 할 때
- 가우스 어림이 그럴듯할 때

**변이 미루어 봄을 쓸 때**:

- 크게 늘릴 수 있는 익힘이 있어야 할 때
- 그럴듯한 변이 갈래를 정할 수 있을 때
- 하이퍼파라미터를 맞출 뜻이 있을 때

**모둠을 쓸 때**:

- 단순함이 무엇보다 종요로울 때
- 모형 여럿을 돌릴 셈 밑천이 있을 때
- 든든한 아리송함이 있어야 할 때

**MC 드롭아웃을 쓸 때**:

- 코드를 되도록 적게 고쳐야 할 때
- 이미 드롭아웃을 쓰고 있을 때
- 셈이 잘 들어야 할 때

### 하이퍼파라미터에서 헤아릴 것

**SGLD**:

- 배움 비율 짜임(가장 종요롭다)
- 몸풀기 동안
- 솎아 내는 틈

**변이 미루어 봄**:

- 앞선 분포의 흩어짐 $\sigma_0^2$
- KL 짐(몸풀기 짜임)
- MC 표본의 수

**라플라스**:

- 헤세 행렬 어림(대각, KFAC 따위)
- 앞선 분포의 촘촘함

### 셈 값

| 방법 | 익힘 | 미루어 봄 | 자리 |
|--------|----------|-----------|---------|
| MAP(밑금) | $O(1)$ | $O(1)$ | $O(d)$ |
| SGLD | $O(T)$ | $O(S)$ | $O(Sd)$ |
| 평균 마당 VI | $O(1)$~$O(2)$ | $O(S)$ | $O(2d)$ |
| 라플라스(대각) | $O(1) + O(d)$ | $O(1)$ | $O(2d)$ |
| 라플라스(KFAC) | $O(1) + O(\sum n_l^2)$ | $O(1)$ | $O(\sum n_l^2)$ |
| 모둠($M$) | $O(M)$ | $O(M)$ | $O(Md)$ |
| MC 드롭아웃 | $O(1)$ | $O(S)$ | $O(d)$ |

### 미루어 봄의 됨됨이 따지기

**눈금 맞음**: 미루어 본 아리송함이 겪은 어긋남과 들어맞는가?

**음수 로그 그럴듯함**: $-\frac{1}{N_{\text{test}}} \sum_i \log p(y_i \mid x_i, \mathcal{D})$

**덮음**: 미루어 본 구간에 참값이 드는 몫

**밖 분포 알아내기**: 아리송함이 밖 분포 들임을 짚어낼 수 있는가?

---

## 파이썬으로 짜기

```python
"""
베이즈 신경 그물의 뒷분포 미루어 봄

이 묶음은 여러 뒷분포 미루어 봄 방법을 짜 놓았다:
- 확률 기울기 랑주뱅 움직임(SGLD)
- 라플라스 어림
- 평균 마당 변이 미루어 봄
- 깊은 모둠
- SWAG
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Tuple, List, Optional, Dict, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


# =============================================================================
# 밑 갈래
# =============================================================================

class BayesianInference(ABC):
    """베이즈 미루어 봄 방법의 뼈대 갈래."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """모형을 자료에 맞춘다."""
        pass
    
    @abstractmethod
    def predict(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        아리송함을 곁들여 미루어 본다.
        
        Returns
        -------
        mean : ndarray
            미루어 본 평균
        std : ndarray
            미루어 본 잣대 어긋남
        """
        pass


# =============================================================================
# 단순 신경 그물
# =============================================================================

class SimpleNN:
    """보여 주기용 단순 신경 그물."""
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'tanh'
    ):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        if activation == 'tanh':
            self.act_fn = np.tanh
            self.act_grad = lambda x: 1 - np.tanh(x)**2
        elif activation == 'relu':
            self.act_fn = lambda x: np.maximum(x, 0)
            self.act_grad = lambda x: (x > 0).astype(float)
        else:
            raise ValueError(f"모르는 살림 함수: {activation}")
    
    def init_weights(self, scale: float = 1.0) -> Dict[str, np.ndarray]:
        """허 잣대로 짐의 첫자리를 잡는다."""
        weights = {}
        for i in range(self.n_layers):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            std = scale * np.sqrt(2.0 / fan_in)
            weights[f'W{i}'] = np.random.randn(fan_in, fan_out) * std
            weights[f'b{i}'] = np.zeros(fan_out)
        return weights
    
    def forward(
        self,
        X: np.ndarray,
        weights: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """앞으로 걸음."""
        h = X
        for i in range(self.n_layers):
            h = h @ weights[f'W{i}'] + weights[f'b{i}']
            if i < self.n_layers - 1:
                h = self.act_fn(h)
        return h
    
    def flatten_weights(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
        """짐을 벡터로 편다."""
        return np.concatenate([weights[k].flatten() for k in sorted(weights.keys())])
    
    def unflatten_weights(self, flat: np.ndarray) -> Dict[str, np.ndarray]:
        """벡터를 짐으로 되돌린다."""
        weights = {}
        idx = 0
        for i in range(self.n_layers):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            
            size_W = fan_in * fan_out
            weights[f'W{i}'] = flat[idx:idx + size_W].reshape(fan_in, fan_out)
            idx += size_W
            
            weights[f'b{i}'] = flat[idx:idx + fan_out]
            idx += fan_out
        
        return weights
    
    def n_params(self) -> int:
        """온 매개변수의 수."""
        total = 0
        for i in range(self.n_layers):
            total += self.layer_sizes[i] * self.layer_sizes[i + 1]  # W
            total += self.layer_sizes[i + 1]  # b
        return total


# =============================================================================
# 확률 기울기 랑주뱅 움직임(SGLD)
# =============================================================================

class SGLD(BayesianInference):
    """
    뒷분포 표본을 뽑는 확률 기울기 랑주뱅 움직임.
    
    Update: θ_{t+1} = θ_t + (ε_t/2) * ∇log p(θ|D) + η_t
    여기서 η_t ~ N(0, ε_t * I)
    """
    
    def __init__(
        self,
        network: SimpleNN,
        prior_std: float = 1.0,
        noise_std: float = 1.0,
        lr_init: float = 0.01,
        lr_decay: float = 0.55,
        n_iterations: int = 10000,
        burn_in: int = 5000,
        thinning: int = 10,
        batch_size: int = 32
    ):
        """
        Parameters
        ----------
        network : SimpleNN
            신경 그물 얼개
        prior_std : float
            짐의 앞선 분포 잣대 어긋남
        noise_std : float
            살핌 잡음의 잣대 어긋남
        lr_init : float
            처음 배움 비율
        lr_decay : float
            배움 비율 줄이기 지수((0.5, 1]에 들어야 한다)
        n_iterations : int
            온 되돌이 횟수
        burn_in : int
            몸풀기 되돌이 횟수
        thinning : int
            thinning번째 표본마다 남긴다
        batch_size : int
            잔 묶음 크기
        """
        self.network = network
        self.prior_std = prior_std
        self.noise_std = noise_std
        self.lr_init = lr_init
        self.lr_decay = lr_decay
        self.n_iterations = n_iterations
        self.burn_in = burn_in
        self.thinning = thinning
        self.batch_size = batch_size
        
        self.samples = []
    
    def _learning_rate(self, t: int) -> float:
        """되돌이 t에서의 배움 비율을 셈한다."""
        return self.lr_init / (1 + t) ** self.lr_decay
    
    def _log_prior_grad(self, theta: np.ndarray) -> np.ndarray:
        """로그 앞선 분포의 기울기(가우스)."""
        return -theta / (self.prior_std ** 2)
    
    def _log_likelihood_grad(
        self,
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        N: int
    ) -> np.ndarray:
        """
        로그 그럴듯함의 기울기(잔 묶음에 맞게 잣대를 맞춤).
        단순하게 수로 미분해 셈한다.
        """
        weights = self.network.unflatten_weights(theta)
        pred = self.network.forward(X, weights)
        
        # 수로 셈하는 기울기
        eps = 1e-5
        grad = np.zeros_like(theta)
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            weights_plus = self.network.unflatten_weights(theta_plus)
            pred_plus = self.network.forward(X, weights_plus)
            
            theta_minus = theta.copy()
            theta_minus[i] -= eps
            weights_minus = self.network.unflatten_weights(theta_minus)
            pred_minus = self.network.forward(X, weights_minus)
            
            # 로그 그럴듯함의 기울기 = -1/(2σ²) * d/dθ ||y - f(x,θ)||²
            ll_plus = -0.5 * np.sum((y - pred_plus)**2) / (self.noise_std**2)
            ll_minus = -0.5 * np.sum((y - pred_minus)**2) / (self.noise_std**2)
            
            grad[i] = (ll_plus - ll_minus) / (2 * eps)
        
        # 잔 묶음에 맞게 잣대를 맞춘다
        return grad * (N / len(X))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """SGLD 표본 뽑기를 돌린다."""
        N = len(X)
        
        # 첫자리를 잡는다
        weights = self.network.init_weights()
        theta = self.network.flatten_weights(weights)
        
        self.samples = []
        self.losses = []
        
        for t in range(self.n_iterations):
            # 잔 묶음을 얻는다
            idx = np.random.choice(N, min(self.batch_size, N), replace=False)
            X_batch = X[idx]
            y_batch = y[idx]
            
            # 배움 비율
            lr = self._learning_rate(t)
            
            # 기울기를 셈한다
            grad_prior = self._log_prior_grad(theta)
            grad_likelihood = self._log_likelihood_grad(X_batch, y_batch, theta, N)
            grad = grad_prior + grad_likelihood
            
            # SGLD 고침
            noise = np.random.randn(len(theta)) * np.sqrt(lr)
            theta = theta + (lr / 2) * grad + noise
            
            # 표본을 담는다
            if t >= self.burn_in and (t - self.burn_in) % self.thinning == 0:
                self.samples.append(theta.copy())
            
            # 잃음을 좇는다
            if t % 100 == 0:
                weights = self.network.unflatten_weights(theta)
                pred = self.network.forward(X, weights)
                loss = np.mean((y - pred)**2)
                self.losses.append(loss)
        
        print(f"SGLD: 표본 {len(self.samples)}개를 모았다")
    
    def predict(
        self,
        X: np.ndarray,
        n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """뒷분포 표본으로 미루어 본다."""
        if n_samples is None:
            samples = self.samples
        else:
            idx = np.random.choice(len(self.samples), min(n_samples, len(self.samples)), replace=False)
            samples = [self.samples[i] for i in idx]
        
        predictions = []
        for theta in samples:
            weights = self.network.unflatten_weights(theta)
            pred = self.network.forward(X, weights)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # 살핌 잡음을 더한다
        total_std = np.sqrt(std**2 + self.noise_std**2)
        
        return mean.flatten(), total_std.flatten()


# =============================================================================
# 라플라스 어림
# =============================================================================

class LaplaceApproximation(BayesianInference):
    """
    뒷분포 미루어 봄을 위한 라플라스 어림.
    
    뒷분포를 MAP 어림을 가운데로 삼는 가우스로 어림한다.
    크게 늘리려고 대각 헤세 행렬 어림을 쓴다.
    """
    
    def __init__(
        self,
        network: SimpleNN,
        prior_std: float = 1.0,
        noise_std: float = 1.0,
        n_iterations: int = 1000,
        lr: float = 0.01
    ):
        """
        Parameters
        ----------
        network : SimpleNN
            신경 그물 얼개
        prior_std : float
            앞선 분포의 잣대 어긋남
        noise_std : float
            살핌 잡음의 잣대 어긋남
        n_iterations : int
            MAP을 찾는 가장 좋게 하기 되돌이 횟수
        lr : float
            MAP 가장 좋게 하기의 배움 비율
        """
        self.network = network
        self.prior_std = prior_std
        self.noise_std = noise_std
        self.n_iterations = n_iterations
        self.lr = lr
        
        self.theta_map = None
        self.hessian_diag = None
    
    def _neg_log_posterior(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """음수 로그 뒷분포를 셈한다."""
        weights = self.network.unflatten_weights(theta)
        pred = self.network.forward(X, weights)
        
        # 로그 그럴듯함
        ll = -0.5 * np.sum((y - pred)**2) / (self.noise_std**2)
        ll -= 0.5 * len(y) * np.log(2 * np.pi * self.noise_std**2)
        
        # 로그 앞선 분포
        lp = -0.5 * np.sum(theta**2) / (self.prior_std**2)
        lp -= 0.5 * len(theta) * np.log(2 * np.pi * self.prior_std**2)
        
        return -(ll + lp)
    
    def _numerical_gradient(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """기울기를 수로 셈한다."""
        grad = np.zeros_like(theta)
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            theta_minus = theta.copy()
            theta_minus[i] -= eps
            
            grad[i] = (
                self._neg_log_posterior(theta_plus, X, y) -
                self._neg_log_posterior(theta_minus, X, y)
            ) / (2 * eps)
        
        return grad
    
    def _numerical_hessian_diag(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        eps: float = 1e-4
    ) -> np.ndarray:
        """헤세 행렬의 대각을 수로 셈한다."""
        hess_diag = np.zeros_like(theta)
        f0 = self._neg_log_posterior(theta, X, y)
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            theta_minus = theta.copy()
            theta_minus[i] -= eps
            
            f_plus = self._neg_log_posterior(theta_plus, X, y)
            f_minus = self._neg_log_posterior(theta_minus, X, y)
            
            hess_diag[i] = (f_plus - 2*f0 + f_minus) / (eps**2)
        
        return hess_diag
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """MAP 어림을 찾고 헤세 행렬을 셈한다."""
        # 첫자리를 잡는다
        weights = self.network.init_weights()
        theta = self.network.flatten_weights(weights)
        
        # MAP을 찾아 가장 좋게 한다
        for t in range(self.n_iterations):
            grad = self._numerical_gradient(theta, X, y)
            theta = theta - self.lr * grad
            
            if t % 200 == 0:
                loss = self._neg_log_posterior(theta, X, y)
                if t % 200 == 0:
                    pass  # 말없이 익힌다
        
        self.theta_map = theta
        
        # 대각 헤세 행렬을 셈한다
        self.hessian_diag = self._numerical_hessian_diag(theta, X, y)
        
        # 양으로 굳게 한다(있어야 하면 작은 값을 더한다)
        self.hessian_diag = np.maximum(self.hessian_diag, 1e-6)
        
        # 뒷분포 흩어짐은 헤세 행렬의 거꿀이다
        self.posterior_var = 1.0 / self.hessian_diag
        
        print(f"라플라스: MAP을 찾고 뒷분포 흩어짐을 셈했다")
    
    def predict(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """라플라스 뒷분포에서 뽑아 미루어 본다."""
        predictions = []
        
        for _ in range(n_samples):
            # 가우스 뒷분포에서 뽑는다
            theta = self.theta_map + np.sqrt(self.posterior_var) * np.random.randn(len(self.theta_map))
            
            weights = self.network.unflatten_weights(theta)
            pred = self.network.forward(X, weights)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # 살핌 잡음을 더한다
        total_std = np.sqrt(std**2 + self.noise_std**2)
        
        return mean.flatten(), total_std.flatten()


# =============================================================================
# 평균 마당 변이 미루어 봄
# =============================================================================

class MeanFieldVI(BayesianInference):
    """
    베이즈 신경 그물을 위한 평균 마당 변이 미루어 봄.
    
    뒷분포를 곱으로 가른 가우스로 어림한다:
    q(θ) = ∏_i N(θ_i | μ_i, σ_i²)
    """
    
    def __init__(
        self,
        network: SimpleNN,
        prior_std: float = 1.0,
        noise_std: float = 1.0,
        n_iterations: int = 5000,
        lr: float = 0.01,
        n_mc_samples: int = 1,
        kl_weight: float = 1.0
    ):
        """
        Parameters
        ----------
        network : SimpleNN
            신경 그물 얼개
        prior_std : float
            앞선 분포의 잣대 어긋남
        noise_std : float
            살핌 잡음의 잣대 어긋남
        n_iterations : int
            가장 좋게 하기 되돌이 횟수
        lr : float
            배움 비율
        n_mc_samples : int
            기울기를 어림할 MC 표본의 수
        kl_weight : float
            KL 항의 짐(KL을 천천히 올리기용)
        """
        self.network = network
        self.prior_std = prior_std
        self.noise_std = noise_std
        self.n_iterations = n_iterations
        self.lr = lr
        self.n_mc_samples = n_mc_samples
        self.kl_weight = kl_weight
        
        self.mu = None
        self.log_sigma = None
    
    def _sample_weights(self) -> np.ndarray:
        """매개변수 다시 잡기 재주로 짐을 뽑는다."""
        eps = np.random.randn(len(self.mu))
        sigma = np.exp(self.log_sigma)
        return self.mu + sigma * eps
    
    def _kl_divergence(self) -> float:
        """q에서 앞선 분포까지의 KL 갈림."""
        sigma = np.exp(self.log_sigma)
        kl = 0.5 * np.sum(
            (self.mu**2 + sigma**2) / (self.prior_std**2) -
            1 - 2 * self.log_sigma + 2 * np.log(self.prior_std)
        )
        return kl
    
    def _elbo(
        self,
        X: np.ndarray,
        y: np.ndarray,
        N: int
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        ELBO와 기울기를 셈한다.
        
        Returns
        -------
        elbo : float
        grad_mu : ndarray
        grad_log_sigma : ndarray
        """
        n_batch = len(X)
        
        # 바라는 로그 그럴듯함의 몬테카를로 어림
        total_ll = 0.0
        grad_mu_ll = np.zeros_like(self.mu)
        grad_log_sigma_ll = np.zeros_like(self.log_sigma)
        
        for _ in range(self.n_mc_samples):
            # 짐을 뽑는다
            eps = np.random.randn(len(self.mu))
            sigma = np.exp(self.log_sigma)
            theta = self.mu + sigma * eps
            
            # 앞으로 걸음
            weights = self.network.unflatten_weights(theta)
            pred = self.network.forward(X, weights)
            
            # 로그 그럴듯함
            ll = -0.5 * np.sum((y - pred)**2) / (self.noise_std**2)
            total_ll += ll
            
            # 단순하게 수로 셈하는 기울기
            delta = 1e-5
            
            for i in range(len(self.mu)):
                # mu에 대한 기울기
                theta_plus = theta.copy()
                theta_plus[i] += delta
                weights_plus = self.network.unflatten_weights(theta_plus)
                pred_plus = self.network.forward(X, weights_plus)
                ll_plus = -0.5 * np.sum((y - pred_plus)**2) / (self.noise_std**2)
                
                grad_mu_ll[i] += (ll_plus - ll) / delta
                
                # log_sigma에 대한 기울기(매개변수 다시 잡기를 지나)
                # d/d(log σ) = d/dθ * dθ/d(log σ) = d/dθ * σ * ε
                grad_log_sigma_ll[i] += (ll_plus - ll) / delta * sigma[i] * eps[i]
        
        total_ll /= self.n_mc_samples
        grad_mu_ll /= self.n_mc_samples
        grad_log_sigma_ll /= self.n_mc_samples
        
        # 온 자료 꾸러미에 맞게 잣대를 맞춘다
        scale = N / n_batch
        total_ll *= scale
        grad_mu_ll *= scale
        grad_log_sigma_ll *= scale
        
        # KL 갈림과 기울기
        kl = self._kl_divergence()
        sigma = np.exp(self.log_sigma)
        grad_mu_kl = self.mu / (self.prior_std**2)
        grad_log_sigma_kl = sigma**2 / (self.prior_std**2) - 1
        
        # ELBO = E[log p(D|θ)] - KL(q||p)
        elbo = total_ll - self.kl_weight * kl
        grad_mu = grad_mu_ll - self.kl_weight * grad_mu_kl
        grad_log_sigma = grad_log_sigma_ll - self.kl_weight * grad_log_sigma_kl
        
        return elbo, grad_mu, grad_log_sigma
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """변이 매개변수를 가장 좋게 한다."""
        N = len(X)
        n_params = self.network.n_params()
        
        # 변이 매개변수의 첫자리를 잡는다
        self.mu = np.random.randn(n_params) * 0.1
        self.log_sigma = np.ones(n_params) * np.log(0.1)
        
        self.elbo_history = []
        
        for t in range(self.n_iterations):
            # ELBO와 기울기를 셈한다
            elbo, grad_mu, grad_log_sigma = self._elbo(X, y, N)
            
            # 고친다
            self.mu += self.lr * grad_mu
            self.log_sigma += self.lr * 0.1 * grad_log_sigma  # 흩어짐에는 더 작은 배움 비율
            
            self.elbo_history.append(elbo)
            
            if t % 500 == 0:
                pass  # 말없이 익힌다
        
        print(f"VI: 가장 좋게 하기를 마쳤다, 마지막 ELBO = {elbo:.2f}")
    
    def predict(
        self,
        X: np.ndarray,
        n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """변이 뒷분포에서 뽑아 미루어 본다."""
        predictions = []
        
        for _ in range(n_samples):
            theta = self._sample_weights()
            weights = self.network.unflatten_weights(theta)
            pred = self.network.forward(X, weights)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # 살핌 잡음을 더한다
        total_std = np.sqrt(std**2 + self.noise_std**2)
        
        return mean.flatten(), total_std.flatten()


# =============================================================================
# 깊은 모둠
# =============================================================================

class DeepEnsemble(BayesianInference):
    """
    아리송함을 어림하는 깊은 모둠.
    
    첫값을 달리해 그물 M개를 서로 남남으로 익힌다.
    """
    
    def __init__(
        self,
        network: SimpleNN,
        n_members: int = 5,
        noise_std: float = 1.0,
        n_iterations: int = 1000,
        lr: float = 0.01
    ):
        """
        Parameters
        ----------
        network : SimpleNN
            신경 그물 얼개
        n_members : int
            모둠 갈래의 수
        noise_std : float
            살핌 잡음의 잣대 어긋남
        n_iterations : int
            갈래마다의 익힘 되돌이 횟수
        lr : float
            배움 비율
        """
        self.network = network
        self.n_members = n_members
        self.noise_std = noise_std
        self.n_iterations = n_iterations
        self.lr = lr
        
        self.members = []
    
    def _train_member(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seed: int
    ) -> np.ndarray:
        """모둠 갈래 하나를 익힌다."""
        np.random.seed(seed)
        
        weights = self.network.init_weights()
        theta = self.network.flatten_weights(weights)
        
        for t in range(self.n_iterations):
            # 기울기를 셈한다(수로)
            grad = self._numerical_gradient(theta, X, y)
            theta = theta - self.lr * grad
        
        return theta
    
    def _numerical_gradient(
        self,
        theta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        eps: float = 1e-5
    ) -> np.ndarray:
        """MSE의 기울기를 수로 셈한다."""
        grad = np.zeros_like(theta)
        
        weights = self.network.unflatten_weights(theta)
        pred = self.network.forward(X, weights)
        loss0 = np.mean((y - pred)**2)
        
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_plus[i] += eps
            
            weights_plus = self.network.unflatten_weights(theta_plus)
            pred_plus = self.network.forward(X, weights_plus)
            loss_plus = np.mean((y - pred_plus)**2)
            
            grad[i] = (loss_plus - loss0) / eps
        
        return grad
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """모둠 갈래를 모두 익힌다."""
        self.members = []
        
        for m in range(self.n_members):
            theta = self._train_member(X, y, seed=m * 42)
            self.members.append(theta)
        
        print(f"모둠: 갈래 {self.n_members}개를 익혔다")
    
    def predict(
        self,
        X: np.ndarray,
        n_samples: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """모둠으로 미루어 본다."""
        predictions = []
        
        for theta in self.members:
            weights = self.network.unflatten_weights(theta)
            pred = self.network.forward(X, weights)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        # 살핌 잡음을 더한다
        total_std = np.sqrt(std**2 + self.noise_std**2)
        
        return mean.flatten(), total_std.flatten()


# =============================================================================
# 그리기와 따지기
# =============================================================================

def plot_inference_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_true: np.ndarray,
    methods: Dict[str, BayesianInference],
    figsize: Tuple[float, float] = (15, 4)
):
    """여러 미루어 봄 방법을 눈으로 견준다."""
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, (name, method) in zip(axes, methods.items()):
        mean, std = method.predict(X_test)
        
        ax.fill_between(
            X_test.flatten(),
            mean - 2*std,
            mean + 2*std,
            alpha=0.3,
            label='±2σ'
        )
        ax.plot(X_test, mean, 'b-', linewidth=2, label='평균')
        ax.plot(X_test, y_true, 'k--', linewidth=1, label='참')
        ax.scatter(X_train, y_train, c='red', s=20, zorder=5, label='자료')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(name)
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


def evaluate_calibration(
    method: BayesianInference,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """아리송함 어림의 눈금 맞음을 따진다."""
    mean, std = method.predict(X_test)
    
    # z 점수를 셈한다
    z = (y_test.flatten() - mean) / std
    
    # 켜마다 바라는 덮음
    coverages = {}
    expected_coverages = [0.5, 0.8, 0.9, 0.95, 0.99]
    
    for p in expected_coverages:
        z_crit = stats.norm.ppf((1 + p) / 2)
        actual = np.mean(np.abs(z) < z_crit)
        coverages[f'coverage_{int(p*100)}'] = actual
    
    # NLL
    nll = -np.mean(stats.norm.logpdf(y_test.flatten(), mean, std))
    
    # RMSE
    rmse = np.sqrt(np.mean((y_test.flatten() - mean)**2))
    
    return {
        'nll': nll,
        'rmse': rmse,
        **coverages
    }


# =============================================================================
# 보여 주는 함수
# =============================================================================

def demo_inference_methods():
    """단순한 문제에서 여러 미루어 봄 방법을 견준다."""
    
    print("=" * 70)
    print("뒷분포 미루어 봄 방법 견주기")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 자료를 만든다
    N = 20
    X_train = np.random.uniform(-3, 3, N).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, (N, 1))
    
    X_test = np.linspace(-5, 5, 200).reshape(-1, 1)
    y_true = np.sin(X_test)
    
    print(f"\n익힘 자료: {N}점")
    print(f"참 함수: sin(x)")
    
    # 그물을 만든다
    network = SimpleNN([1, 20, 1], activation='tanh')
    print(f"그물: {network.layer_sizes}, 매개변수 {network.n_params()}개")
    
    # 여러 방법을 익힌다
    methods = {}
    
    # 라플라스
    print("\n--- 라플라스 어림 ---")
    laplace = LaplaceApproximation(
        network, prior_std=1.0, noise_std=0.2,
        n_iterations=500, lr=0.05
    )
    laplace.fit(X_train, y_train)
    methods['Laplace'] = laplace
    
    # 모둠
    print("\n--- 깊은 모둠 ---")
    ensemble = DeepEnsemble(
        network, n_members=5, noise_std=0.2,
        n_iterations=500, lr=0.05
    )
    ensemble.fit(X_train, y_train)
    methods['Ensemble'] = ensemble
    
    # 따진다
    print("\n--- 따지기 ---")
    print(f"{'방법':<15} {'NLL':>8} {'RMSE':>8} {'덮음90%':>8}")
    print("-" * 45)
    
    for name, method in methods.items():
        metrics = evaluate_calibration(method, X_test, y_true)
        print(f"{name:<15} {metrics['nll']:>8.3f} {metrics['rmse']:>8.3f} "
              f"{metrics['coverage_90']:>8.2%}")
    
    return methods


def demo_sgld():
    """SGLD 표본 뽑기를 보여 준다."""
    
    print("\n" + "=" * 70)
    print("확률 기울기 랑주뱅 움직임")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 단순한 1차 문제
    N = 30
    X_train = np.random.uniform(-3, 3, N).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, (N, 1))
    
    network = SimpleNN([1, 10, 1], activation='tanh')
    
    print(f"\n매개변수 {network.n_params()}개로 SGLD를 돌리는 중...")
    
    sgld = SGLD(
        network,
        prior_std=1.0,
        noise_std=0.2,
        lr_init=0.001,
        lr_decay=0.55,
        n_iterations=3000,
        burn_in=1500,
        thinning=5,
        batch_size=N  # 든든하도록 온 묶음
    )
    
    sgld.fit(X_train, y_train)
    
    # 따진다
    X_test = np.linspace(-5, 5, 100).reshape(-1, 1)
    mean, std = sgld.predict(X_test)
    
    print(f"\n모은 표본: {len(sgld.samples)}")
    print(f"미루어 봄의 평균 잣대 어긋남: {np.mean(std):.3f}")
    
    return sgld


def demo_variational_inference():
    """평균 마당 변이 미루어 봄을 보여 준다."""
    
    print("\n" + "=" * 70)
    print("평균 마당 변이 미루어 봄")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 단순한 문제
    N = 30
    X_train = np.random.uniform(-3, 3, N).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, (N, 1))
    
    # VI을 빠르게 하려고 작은 그물
    network = SimpleNN([1, 10, 1], activation='tanh')
    
    print(f"\n매개변수 {network.n_params()}개로 VI을 돌리는 중...")
    print("(수로 셈하는 기울기 탓에 좀 걸릴 수 있다)")
    
    vi = MeanFieldVI(
        network,
        prior_std=1.0,
        noise_std=0.2,
        n_iterations=1000,
        lr=0.01,
        n_mc_samples=1
    )
    
    vi.fit(X_train, y_train)
    
    # 배운 뒷분포를 살핀다
    sigma = np.exp(vi.log_sigma)
    print(f"\n뒷분포 자:")
    print(f"  평균 |μ|: {np.mean(np.abs(vi.mu)):.4f}")
    print(f"  평균 σ:   {np.mean(sigma):.4f}")
    print(f"  가장 큰 σ:    {np.max(sigma):.4f}")
    
    return vi


if __name__ == "__main__":
    methods = demo_inference_methods()
    sgld = demo_sgld()
    vi = demo_variational_inference()
```

---

## 간추림

### 미루어 봄 방법 두루 보기

| 방법 | 길 | 맞음 | 크게 늘리기 |
|--------|----------|----------|-------------|
| **HMC** | 정확한 MCMC | 높음 | 낮음 |
| **SGLD** | 확률 MCMC | 가운데~높음 | 높음 |
| **라플라스** | MAP에서의 가우스 | 가운데 | 가운데~높음 |
| **평균 마당 VI** | 곱으로 가른 가장 좋게 하기 | 낮음~가운데 | 높음 |
| **깊은 모둠** | MAP 여럿 | 가운데~높음 | 가운데 |
| **MC 드롭아웃** | 넌지시 하는 VI | 낮음~가운데 | 아주 높음 |

### 고갱이 식

**뒷분포**:

$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \, p(\theta)
$$

**SGLD 고침**:

$$
\theta^{(t+1)} = \theta^{(t)} + \frac{\epsilon_t}{2} \nabla \log p(\theta^{(t)} \mid \mathcal{D}) + \mathcal{N}(0, \epsilon_t I)
$$

**ELBO**(변이 미루어 봄):

$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi \| p)
$$

**라플라스 어림**:

$$
q(\theta) = \mathcal{N}(\theta_{\text{MAP}}, H^{-1})
$$

### 셈 번거로움

| 방법 | 익힘 | 미루어 봄 | 자리 |
|--------|----------|-----------|---------|
| MAP | $O(E \cdot N)$ | $O(1)$ | $O(d)$ |
| SGLD | $O(T \cdot B)$ | $O(S)$ | $O(Sd)$ |
| 라플라스 | $O(E \cdot N + d^2)$ | $O(S)$ | $O(d^2)$ |
| VI | $O(E \cdot N)$ | $O(S)$ | $O(2d)$ |
| 모둠 | $O(M \cdot E \cdot N)$ | $O(M)$ | $O(Md)$ |

### 방법 고르기 길잡이

| 형편 | 즐겨 쓸 방법 |
|----------|-------------------|
| 일 끝난 뒤 아리송함 | 라플라스, SWAG |
| 크게 늘릴 수 있는 익힘 | VI, MC 드롭아웃 |
| 가장 좋은 아리송함 | HMC(작을 때), SGLD(클 때) |
| 단순한 짜기 | 모둠, MC 드롭아웃 |
| 셈이 넉넉지 않을 때 | MC 드롭아웃, 라플라스 |

### 다른 장과의 이어짐

| 이야기 | 장 | 이어짐 |
|-------|---------|------------|
| 앞선 분포 정하기 | 13장: 짐의 앞선 분포 | 뒷분포의 들임 |
| 아리송함 | 13장: 아리송함 | 뒷분포가 쪼갬을 이룬다 |
| MC 드롭아웃 | 13장: MC 드롭아웃 | 넌지시 하는 변이 미루어 봄 |
| 변이 베이즈 신경 그물 | 13장: 변이 베이즈 신경 그물 | VI을 자세히 다룸 |
| 모형 견주기 | 13장: 소식 잣대 | 가장자리 그럴듯함 |

### 고갱이 살펴볼 거리

- Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
- Blundell, C., et al. (2015). Weight uncertainty in neural networks. *ICML*.
- MacKay, D. J. (1992). A practical Bayesian framework for backpropagation networks. *Neural Computation*.
- Ritter, H., et al. (2018). A scalable Laplace approximation for neural networks. *ICLR*.
- Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Maddox, W., et al. (2019). A simple baseline for Bayesian inference in deep learning. *NeurIPS*.

## 익힘 문제

**익힘 1.**
ReLU 살림과 가우스 짐 앞선 분포를 지닌 두 켜 신경 그물에서, 이 마디에서 밝힌 방법에 따른 어림 뒷분포의 꼴을 이끌어 내어라.

??? success "익힘 1 풀이"
    짐이 $W_1, W_2$이고 가우스 앞선 분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 뒷분포 $p(W | D) \propto p(D | W) p(W)$은 다룰 수 없다. 이 마디의 어림 방법은 다룰 수 있는 꼴을 낸다. 변이 미루어 봄이면 짐마다 서로 남남인 가우스 뒷분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 어림이면 뒷분포가 MAP 어림을 가운데로 삼고 함께 바뀜이 헤세 행렬의 거꿀인 가우스 하나이며, MC 드롭아웃이면 뒷분포가 드롭아웃 가리개 분포로 넌지시 세워진다. 어림마다 참 뒷분포 모습의 서로 다른 결을 담는다. $\square$

---

**익힘 2.**
이 방법에서 얻은 아리송함 어림의 눈금 맞음을 MC 드롭아웃, 깊은 모둠과 견주는 시험을 꾸며라. 쓸 자와 그림을 밝혀라.

??? success "익힘 2 풀이"
    자: (1) 통 15개의 바라는 눈금 맞음 어긋남(ECE), (2) 브라이어 점수, (3) 음수 로그 그럴듯함(NLL), (4) 밖 분포 알아내기의 AUROC. 그림: 방법마다 본 잦기를 미루어 본 자신함에 대고 그린 미더움 그림. 절차: 모든 방법을 CIFAR-10(분포 안)에서 익히고, CIFAR-10 시험 자료에서 눈금 맞음을, SVHN에서 밖 분포 알아내기를 따진다. 온도 잣대 잡기를 일 끝난 뒤 밑금으로 쓴다. 아무렇게나 하는 씨앗 5개에 걸친 평균과 잣대 어긋남을 알린다. 눈금이 잘 맞은 방법은 미더움 그림에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

**익힘 3.**
베이즈 신경 그물의 미루어 봄 흩어짐이 앎의 아리송함과 타고난 아리송함으로 쪼개짐을 증명하여라. 익힘 자료 크기 $N \to \infty$일 때 두 몫이 어떻게 되는지 보여라.

??? success "익힘 3 풀이"
    미루어 봄 흩어짐은 온 흩어짐 법칙으로 쪼개진다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 타고난 몫은 자료를 낳는 흐름의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 앎의 몫은 매개변수의 아리송함을 드러내며 뒷분포가 참 매개변수 언저리로 모이므로 $O(1/N)$으로 준다. 끝에 가면 타고난 아리송함만 남는다. 이 쪼갬은 자료를 더 모아야 할 때(앎의 아리송함이 클 때)와 타고난 잡음을 받아들여야 할 때(타고난 아리송함이 클 때)를 가르는 데 종요롭다. $\square$

---

**익힘 4.**
이 마디의 아리송함 재기 방법을 거래 얼개의 자리 크기 잡기에 어떻게 쓸 수 있는지 다루어라. 손에 잡히는 판단 규칙을 내놓아라.

??? success "익힘 4 풀이"
    판단 규칙: 자리 크기를 앎의 아리송함에 반비례하게 잡는다. $\hat{y}$을 미루어 본 돌아옴, $\sigma_e^2$을 앎의 흩어짐이라 하자. 자리는 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$은 무릅씀 꺼림 값이다. 앎의 아리송함이 크면(낯선 저자 형편) 자리를 줄이고, 작으면(익숙한 판) 더 굳게 거래한다. 여기에 더해 그 위로는 거래하지 않는 앎의 아리송함 위끝을 두어 삼갈 수 있다. 이 틀은 모형의 자신함으로 잣대를 잡은 켈리 잣대 결의 크기 잡기를 절로 이룬다. 앞으로 걸어가며 살피기로 되짚어 시험해 $\lambda$의 눈금을 맞춘다. $\square$
