# 베이즈 뉴럴 네트워크 사후분포 추론
**사후분포 추론**은 베이즈 뉴럴 네트워크에서 계산으로 부딪히는 가장 큰 어려움이다. 사전분포 $p(\theta)$와 우도 $p(\mathcal{D} \mid \theta)$가 주어지면 사후분포 $p(\theta \mid \mathcal{D})$를 찾는다. 뉴럴 네트워크에서는 이 사후분포를 다룰 수 없으므로, 샘플링(MCMC)에서 최적화(변분 추론), 암묵적 근사(드롭아웃, 앙상블)에 이르는 근사 방법이 있어야 한다.

---

## 1. 추론의 어려움

### 사후분포

베이즈 정리에 따르면 네트워크 웨이트에 대한 사후분포는 이렇다.

$$
\boxed{p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) \, p(\theta)}{p(\mathcal{D})}}
$$

여기서

- $p(\mathcal{D} \mid \theta) = \prod_{i=1}^N p(y_i \mid x_i, \theta)$는 우도
- $p(\theta)$는 사전분포
- $p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta$는 모델 증거(주변 우도)

### 정확한 추론을 다룰 수 없는 까닭

**1. 고차원**: 요즘 네트워크의 파라메터는 $d = 10^6$에서 $10^9$이다

**2. 켤레성 없음**: 뉴럴 네트워크의 우도에는 켤레 사전분포가 없다.

$$
p(y \mid x, \theta) = \mathcal{N}(y \mid f_\theta(x), \sigma^2)
$$

여기서 $f_\theta$는 복잡한 비선형 함수다.

**3. 계산할 수 없는 정규화 상수**: 모델 증거 적분에는 닫힌 형태가 없다.

$$
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) \, p(\theta) \, d\theta
$$

**4. 여러 모드**: 사후분포의 지형에는 다음 까닭으로 모드가 많다.

- 웨이트 공간의 대칭(순열 대칭, 스케일 대칭)
- 좋은 해가 여럿임
- 로스 지형이 복잡함

### 추론 방법에 바라는 것

| 특성 | 설명 |
|----------|-------------|
| **확장성** | 파라메터 수백만 개를 다룬다 |
| **정확도** | 사후분포의 형태를 충실하게 담는다 |
| **불확실성** | 캘리브레이션이 잘 된 불확실성을 낸다 |
| **효율** | 계산 시간이 알맞다 |
| **단순함** | 구현과 튜닝이 쉽다 |

모든 기준에서 앞서는 방법은 없다. 방법마다 이 특성들 사이에 트레이드오프가 있다.

---

## 2. 추론 방법 개관

### 분류

```
뒷분포 미루어 봄 방법
├── 표본 뽑기 방법(MCMC)
│   ├── 메트로폴리스-헤이스팅스
│   ├── 해밀턴 몬테카를로(HMC)
│   ├── 확률 그래디언트 MCMC
│   │   ├── SGLD(랑주뱅 액션)
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

### 방법 비교

| 방법 | 정확도 | 확장성 | 단순함 | 비용 |
|--------|----------|-------------|------------|------|
| HMC | 높음 | 낮음 | 낮음 | 아주 비쌈 |
| SGLD | 중간 | 높음 | 중간 | 저렴 |
| 평균장 VI | 낮음~중간 | 높음 | 중간 | 중간 |
| 라플라스 | 중간 | 중간 | 높음 | 저렴 |
| MC 드롭아웃 | 낮음~중간 | 아주 높음 | 아주 높음 | 아주 저렴 |
| 딥 앙상블 | 중간~높음 | 높음 | 높음 | 중간 |

---

## 3. 마르코프 사슬 몬테카를로(MCMC)

### 기초

MCMC는 정상 분포(stationary distribution)가 사후분포가 되는 마르코프 사슬을 만든다.

$$
\theta^{(t+1)} \sim T(\theta^{(t+1)} \mid \theta^{(t)})
$$

그리하여 $t \to \infty$일 때 $\theta^{(t)} \to p(\theta \mid \mathcal{D})$가 된다(분포의 뜻에서).

**표본 활용**: 표본 $\{\theta^{(t)}\}_{t=1}^T$가 주어지면 기댓값을 추정한다.

$$
\mathbb{E}_{p(\theta \mid \mathcal{D})}[f(\theta)] \approx \frac{1}{T} \sum_{t=1}^T f(\theta^{(t)})
$$

### 메트로폴리스-헤이스팅스

**알고리즘**:

1. $\theta' \sim q(\theta' \mid \theta^{(t)})$를 제안한다
2. 채택 확률을 계산한다.

$$
\alpha = \min\left(1, \frac{p(\theta' \mid \mathcal{D}) \, q(\theta^{(t)} \mid \theta')}{p(\theta^{(t)} \mid \mathcal{D}) \, q(\theta' \mid \theta^{(t)})}\right)
$$

3. 확률 $\alpha$로 채택한다: 채택하면 $\theta^{(t+1)} = \theta'$, 기각하면 $\theta^{(t+1)} = \theta^{(t)}$

**뉴럴 네트워크에서는**: 랜덤 워크 제안($q(\theta' \mid \theta) = \mathcal{N}(\theta, \epsilon^2 I)$)은 차원이 높으면 효율이 크게 떨어진다.

### 해밀토니안 몬테카를로(HMC)

HMC는 그래디언트 정보를 써서 정보에 기반한 제안을 한다.

**보조 변수**: 운동량 $\rho$를 도입해 해밀토니안을 정의한다.

$$
H(\theta, \rho) = -\log p(\theta \mid \mathcal{D}) + \frac{1}{2}\rho^\top M^{-1} \rho
$$

**해밀턴 동역학**:

$$
\frac{d\theta}{dt} = M^{-1} \rho, \quad \frac{d\rho}{dt} = \nabla_\theta \log p(\theta \mid \mathcal{D})
$$

**Leapfrog 적분기**(스텝 크기 $\epsilon$으로 $L$스텝):

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

1. 운동량을 뽑는다: $\rho \sim \mathcal{N}(0, M)$
2. Leapfrog를 $L$스텝 돌린다
3. 메트로폴리스 보정으로 채택하거나 기각한다

**장점**: 사후분포를 잘 탐색하고 표본끼리 상관이 작다.

**뉴럴 네트워크에서의 어려움**: 전체 데이터에 대한 그래디언트를 계산해야 한다(큰 데이터 꾸러미에서는 비싸다).

---

## 4. 확률 그래디언트 MCMC

### 왜 하는가

전체 배치 그래디언트는 비싸다. 확률 그래디언트 MCMC는 미니배치 그래디언트를 쓴다.

$$
\nabla_\theta \log p(\theta \mid \mathcal{D}) \approx \nabla_\theta \log p(\theta) + \frac{N}{|B|} \sum_{i \in B} \nabla_\theta \log p(y_i \mid x_i, \theta)
$$

여기서 $B$는 크기가 $|B|$인 미니배치다.

### 확률 그래디언트 랑주뱅 동역학(SGLD)

**갱신 규칙**:

$$
\boxed{\theta^{(t+1)} = \theta^{(t)} + \frac{\epsilon_t}{2} \nabla_\theta \log p(\theta^{(t)} \mid \mathcal{D}) + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \epsilon_t I)}
$$

**핵심 통찰**: 스텝 크기가 $\epsilon_t \to 0$으로 줄면 메트로폴리스 채택 단계를 건너뛸 수 있다.

**스텝 크기 스케줄**: 다음을 만족해야 한다.

$$
\sum_{t=1}^\infty \epsilon_t = \infty, \quad \sum_{t=1}^\infty \epsilon_t^2 < \infty
$$

흔한 선택: $\gamma \in (0.5, 1]$인 $\epsilon_t = a(b + t)^{-\gamma}$.

**실제로 쓰는 SGLD**:

$$
\theta^{(t+1)} = \theta^{(t)} + \frac{\epsilon_t}{2} \left[ \nabla_\theta \log p(\theta^{(t)}) + \frac{N}{|B|} \sum_{i \in B} \nabla_\theta \log p(y_i \mid x_i, \theta^{(t)}) \right] + \eta_t
$$

### 확률 그래디언트 해밀토니안 몬테카를로(SGHMC)

더 잘 탐색하도록 SGLD에 운동량을 더한다.

$$
\theta^{(t+1)} = \theta^{(t)} + \rho^{(t)}
$$

$$
\rho^{(t+1)} = (1 - \alpha) \rho^{(t)} + \epsilon_t \nabla_\theta \log p(\theta^{(t)} \mid \mathcal{D}) + \eta_t
$$

여기서 $\alpha$는 마찰 계수이고 $\eta_t \sim \mathcal{N}(0, 2\alpha\epsilon_t I)$이다.

### 전처리된 SGLD

파라메터마다 스케일이 잘 맞도록 전처리 행렬(preconditioner) $G(\theta)$를 쓴다.

$$
\theta^{(t+1)} = \theta^{(t)} + \frac{\epsilon_t}{2} \left[ G(\theta^{(t)}) \nabla_\theta \log p(\theta^{(t)} \mid \mathcal{D}) + \Gamma(\theta^{(t)}) \right] + \eta_t
$$

여기서 $\eta_t \sim \mathcal{N}(0, \epsilon_t G(\theta^{(t)}))$이고 $\Gamma$는 보정 항이다.

**$G$로 흔히 고르는 것**:

- RMSprop 전처리기
- Adam 전처리기
- 피셔 정보 행렬

### 순환 SGLD

국소 모드를 벗어나도록 순환(cyclical) 러닝레이트를 쓴다.

$$
\epsilon_t = \epsilon_0 \left( \cos\left(\frac{\pi \, \text{mod}(t, T_{\text{cycle}})}{T_{\text{cycle}}}\right) + 1 \right) / 2
$$

스텝 크기가 작아지는 주기의 끝에서 표본을 모은다.

---

## 5. 라플라스 근사

### 통찰

사후분포를 MAP 추정을 중심으로 하는 가우시안으로 근사한다.

$$
\boxed{p(\theta \mid \mathcal{D}) \approx q(\theta) = \mathcal{N}(\theta \mid \hat{\theta}_{\text{MAP}}, \Sigma)}
$$

여기서

- $\hat{\theta}_{\text{MAP}} = \arg\max_\theta \log p(\theta \mid \mathcal{D})$
- $\Sigma = \left[ -\nabla^2_\theta \log p(\theta \mid \mathcal{D}) \big|_{\hat{\theta}_{\text{MAP}}} \right]^{-1}$

### 유도

로그 사후분포를 MAP 부근에서 2차까지 테일러 전개한다.

$$
\log p(\theta \mid \mathcal{D}) \approx \log p(\hat{\theta} \mid \mathcal{D}) - \frac{1}{2}(\theta - \hat{\theta})^\top H (\theta - \hat{\theta})
$$

여기서 $H = -\nabla^2_\theta \log p(\theta \mid \mathcal{D})|_{\hat{\theta}}$은 헤세 행렬이다.

지수를 취하면 공분산이 $\Sigma = H^{-1}$인 가우시안이 된다.

### 헤세 행렬 계산

**전체 헤세 행렬**: 메모리 $O(d^2)$, 역 계산 $O(d^3)$ — 큰 네트워크에서는 다룰 수 없다.

**대각 근사**:

$$
\Sigma = \text{diag}(\sigma_1^2, \ldots, \sigma_d^2)
$$

여기서 $\sigma_i^2 = 1/H_{ii}$이다.

**크로네커 분해 근사(KFAC)**:

웨이트가 $W^{(l)}$인 레이어 $l$에서

$$
H^{(l)} \approx A^{(l)} \otimes G^{(l)}
$$

여기서

- $A^{(l)} = \mathbb{E}[a^{(l-1)} (a^{(l-1)})^\top]$(입력 활성화)
- $G^{(l)} = \mathbb{E}[g^{(l)} (g^{(l)})^\top]$(출력 그래디언트)

**역**:

$$
(A \otimes G)^{-1} = A^{-1} \otimes G^{-1}
$$

레이어마다 $O(d^3)$을 $O(n_l^3 + n_{l-1}^3)$으로 줄인다.

### 마지막 레이어 라플라스

앞 레이어는 고정해 두고 마지막 레이어에만 라플라스를 건다.

$$
p(\theta_L \mid \mathcal{D}, \theta_{1:L-1}) \approx \mathcal{N}(\theta_L \mid \hat{\theta}_L, \Sigma_L)
$$

**나은 점**:

- 헤세 행렬이 훨씬 작다
- 불확실성의 대부분을 담는 일이 잦다
- 특성 추출기는 고정된 채로 남는다

### 예측 분포

가우시안 우도를 쓰는 회귀에서

$$
p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta) \, q(\theta) \, d\theta
$$

MAP 부근에서 **선형화**:

$$
f_\theta(x) \approx f_{\hat{\theta}}(x) + J_{\hat{\theta}}(x)(\theta - \hat{\theta})
$$

여기서 $J_{\hat{\theta}}(x) = \nabla_\theta f_\theta(x)|_{\hat{\theta}}$은 야코비안 행렬이다.

**예측 분산**:

$$
\text{Var}[f(x^*)] \approx J_{\hat{\theta}}(x^*)^\top \Sigma \, J_{\hat{\theta}}(x^*)
$$

---

## 6. 변분 추론

### 변분 목표

KL 발산을 가장 작게 하여 $p(\theta \mid \mathcal{D})$를 다룰 수 있는 분포 $q_\phi(\theta)$로 근사한다.

$$
\phi^* = \arg\min_\phi \text{KL}(q_\phi(\theta) \| p(\theta \mid \mathcal{D}))
$$

**증거 하한(ELBO)**:

$$
\boxed{\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi(\theta) \| p(\theta))}
$$

**유도**:

$$
\log p(\mathcal{D}) = \mathcal{L}(\phi) + \text{KL}(q_\phi \| p(\theta \mid \mathcal{D})) \geq \mathcal{L}(\phi)
$$

ELBO를 가장 크게 하는 일은 사후분포에 대한 KL을 가장 작게 하는 일과 같다.

### 평균장 변분 추론

**곱으로 분해한 근사**:

$$
q_\phi(\theta) = \prod_{i=1}^d q_{\phi_i}(\theta_i)
$$

**가우시안 평균장**:

$$
q_\phi(\theta) = \prod_{i=1}^d \mathcal{N}(\theta_i \mid \mu_i, \sigma_i^2)
$$

파라메터: $\phi = \{\mu_i, \sigma_i\}_{i=1}^d$($\sigma_i$가 양수가 되도록 실제로는 $\log \sigma_i$를 트레인한다).

### 리파라메리제이션 기법

확률적 샘플링을 거쳐 그래디언트를 계산하려면 표본을 이렇게 다시 쓴다.

$$
\theta = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

**ELBO의 그래디언트**:

$$
\nabla_\phi \mathcal{L} = \nabla_\phi \mathbb{E}_{\epsilon}[\log p(\mathcal{D} \mid \mu + \sigma \odot \epsilon)] - \nabla_\phi \text{KL}(q_\phi \| p)
$$

첫째 항은 몬테카를로로 근사하고, 둘째 항은 흔히 닫힌 형태를 지닌다.

### 가우시안 사전분포의 KL 발산

$q(\theta) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$과 $p(\theta) = \mathcal{N}(0, \sigma_0^2 I)$에서

$$
\text{KL}(q \| p) = \frac{1}{2} \sum_{i=1}^d \left[ \frac{\mu_i^2 + \sigma_i^2}{\sigma_0^2} - 1 - \log\frac{\sigma_i^2}{\sigma_0^2} \right]
$$

### 역전파 베이즈(Bayes by Backprop)

**알고리즘**(블런델 등, 2015):

1. $\epsilon \sim \mathcal{N}(0, I)$를 뽑는다
2. $\theta = \mu + \log(1 + e^\rho) \odot \epsilon$을 계산한다($\sigma$에 소프트플러스)
3. 로스를 계산한다: $\mathcal{L} = \log q_\phi(\theta) - \log p(\theta) - \log p(\mathcal{D} \mid \theta)$
4. 역전파로 $\phi = \{\mu, \rho\}$를 갱신한다

**미니배치 ELBO**:

$$
\mathcal{L} \approx \frac{N}{|B|} \sum_{i \in B} \log p(y_i \mid x_i, \theta) - \text{KL}(q_\phi \| p)
$$

### 평균장을 넘어

**전체 공분산**: $q(\theta) = \mathcal{N}(\mu, \Sigma)$

- 파라메터가 $O(d^2)$개 — 흔히 다룰 수 없다

**저랭크 더하기 대각**:

$$
\Sigma = D + VV^\top
$$

여기서 $D$는 대각 행렬이고 $V \in \mathbb{R}^{d \times r}$의 랭크는 $r \ll d$이다.

**정규화 흐름**(normalizing flow): 역함수가 있는 변환으로 단순한 분포를 바꾼다

$$
q(\theta) = q_0(f^{-1}(\theta)) \left| \det \frac{\partial f^{-1}}{\partial \theta} \right|
$$

---

## 7. 암묵적 변분 방법

### 딥 앙상블

초깃값을 달리해 네트워크 $M$개를 서로 독립으로 트레인한다.

$$
\{\theta^{(m)}\}_{m=1}^M \quad \text{where each } \theta^{(m)} = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D})
$$

**예측 분포**:

$$
p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{M} \sum_{m=1}^M p(y^* \mid x^*, \theta^{(m)})
$$

**해석**: 서로 다른 모드에서 하나씩 뽑는 암묵적 사후분포 근사다.

**장점**:

- 구현이 단순하다
- 병렬화가 아주 쉽다
- 캘리브레이션이 잘 되는 일이 잦다

**단점**:

- 트레인 비용이 $M$배
- 메모리와 추론 비용이 $M$배
- 엄밀한 베이즈 방법은 아니다

### 확률적 웨이트 평균 가우시안(SWAG)

SGD로 트레인하는 동안 웨이트의 통계량을 모은다.

**이동 통계**:

$$
\bar{\theta} = \frac{1}{T} \sum_{t=1}^T \theta^{(t)}
$$

$$
\bar{\theta^2} = \frac{1}{T} \sum_{t=1}^T (\theta^{(t)})^2
$$

**대각 분산**:

$$
\Sigma_{\text{diag}} = \text{diag}(\bar{\theta^2} - \bar{\theta}^2)
$$

**저랭크 성분**(평균에서 벗어난 편차로 만든다):

$$
D = [\theta^{(t_1)} - \bar{\theta}, \ldots, \theta^{(t_K)} - \bar{\theta}]
$$

**SWAG 사후분포**:

$$
q(\theta) = \mathcal{N}\left(\bar{\theta}, \frac{1}{2}(\Sigma_{\text{diag}} + \frac{1}{K-1}DD^\top)\right)
$$

### MC 드롭아웃

테스트할 때도 드롭아웃을 켜 두어 근사 변분 추론으로 쓴다.

$$
q(\theta) = \prod_l q(W^{(l)})
$$

여기서 $q(W^{(l)})$는 입력 유닛 하나에 이어진 웨이트 묶음이 통째로 무작위로 0이 되는 분포다.

자세한 것은 44.1절 몬테카를로 드롭아웃을 보라.

---

## 8. 실전에서 고려할 것

### 추론 방법 고르기

**SGLD를 쓸 때**:

- 이론적 보장이 있는 표본이 있어야 할 때
- 오래 트레인할 수 있을 때
- 사후분포의 여러 모드가 중요할 때

**라플라스 근사를 쓸 때**:

- 이미 트레인된 네트워크가 있을 때(사후(post-hoc) 불확실성)
- 불확실성 근사가 빨리 있어야 할 때
- 가우시안 근사가 그럴듯할 때

**변분 추론을 쓸 때**:

- 확장 가능한 트레인이 있어야 할 때
- 그럴듯한 변분 분포족을 정할 수 있을 때
- 하이퍼파라미터를 튜닝할 여력이 있을 때

**앙상블을 쓸 때**:

- 단순함이 무엇보다 중요할 때
- 모델 여럿을 돌릴 계산 자원이 있을 때
- 안정적인 불확실성이 있어야 할 때

**MC 드롭아웃을 쓸 때**:

- 코드를 되도록 적게 고쳐야 할 때
- 이미 드롭아웃을 쓰고 있을 때
- 계산이 효율적이어야 할 때

### 하이퍼파라미터에서 고려할 것

**SGLD**:

- 러닝레이트 스케줄(가장 중요하다)
- 워밍업(번인) 기간
- 솎아 내기(thinning) 간격

**변분 추론**:

- 사전분포의 분산 $\sigma_0^2$
- KL 웨이트(워밍업 스케줄)
- MC 표본의 수

**라플라스**:

- 헤세 행렬 근사(대각, KFAC 따위)
- 사전분포의 정밀도

### 계산 비용

| 방법 | 트레인 | 추론 | 메모리 |
|--------|----------|-----------|---------|
| MAP(베이스라인) | $O(1)$ | $O(1)$ | $O(d)$ |
| SGLD | $O(T)$ | $O(S)$ | $O(Sd)$ |
| 평균장 VI | $O(1)$~$O(2)$ | $O(S)$ | $O(2d)$ |
| 라플라스(대각) | $O(1) + O(d)$ | $O(1)$ | $O(2d)$ |
| 라플라스(KFAC) | $O(1) + O(\sum n_l^2)$ | $O(1)$ | $O(\sum n_l^2)$ |
| 앙상블($M$) | $O(M)$ | $O(M)$ | $O(Md)$ |
| MC 드롭아웃 | $O(1)$ | $O(S)$ | $O(d)$ |

### 추론의 성능 평가

**캘리브레이션**: 예측한 불확실성이 실제 오차와 들어맞는가?

**음의 로그 우도**: $-\frac{1}{N_{\text{test}}} \sum_i \log p(y_i \mid x_i, \mathcal{D})$

**커버리지**(coverage): 예측 구간에 참값이 드는 몫

**분포 외 탐지**: 불확실성으로 분포 외 입력을 탐지할 수 있는가?

---

## 9. 파이썬으로 구현하기

```python
"""
베이즈 뉴럴 네트워크의 뒷분포 미루어 봄

이 배치는 여러 뒷분포 미루어 봄 방법을 짜 놓았다:
- 확률 그래디언트 랑주뱅 액션(SGLD)
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
# 밑 클래스
# =============================================================================

class BayesianInference(ABC):
    """베이즈 미루어 봄 방법의 뼈대 클래스."""
    
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
            미루어 본 표준편차
        """
        pass

# =============================================================================
# 단순 뉴럴 네트워크
# =============================================================================

class SimpleNN:
    """보여 주기용 단순 뉴럴 네트워크."""
    
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
        """앞으로 스텝."""
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
        """온 파라메터의 수."""
        total = 0
        for i in range(self.n_layers):
            total += self.layer_sizes[i] * self.layer_sizes[i + 1]  # W
            total += self.layer_sizes[i + 1]  # b
        return total

# =============================================================================
# 확률 그래디언트 랑주뱅 액션(SGLD)
# =============================================================================

class SGLD(BayesianInference):
    """
    뒷분포 표본을 뽑는 확률 그래디언트 랑주뱅 액션.
    
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
            뉴럴 네트워크 구조
        prior_std : float
            짐의 앞선 분포 표준편차
        noise_std : float
            살핌 잡음의 표준편차
        lr_init : float
            처음 러닝 비율
        lr_decay : float
            러닝 비율 줄이기 지수((0.5, 1]에 들어야 한다)
        n_iterations : int
            온 회귀 횟수
        burn_in : int
            몸풀기 회귀 횟수
        thinning : int
            thinning번째 표본마다 남긴다
        batch_size : int
            잔 배치 크기
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
        """회귀 t에서의 러닝 비율을 셈한다."""
        return self.lr_init / (1 + t) ** self.lr_decay
    
    def _log_prior_grad(self, theta: np.ndarray) -> np.ndarray:
        """로그 앞선 분포의 그래디언트(가우스)."""
        return -theta / (self.prior_std ** 2)
    
    def _log_likelihood_grad(
        self,
        X: np.ndarray,
        y: np.ndarray,
        theta: np.ndarray,
        N: int
    ) -> np.ndarray:
        """
        로그 그럴듯함의 그래디언트(잔 배치에 맞게 잣대를 맞춤).
        단순하게 수로 미분해 셈한다.
        """
        weights = self.network.unflatten_weights(theta)
        pred = self.network.forward(X, weights)
        
        # 수로 셈하는 그래디언트
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
            
            # 로그 그럴듯함의 그래디언트 = -1/(2σ²) * d/dθ ||y - f(x,θ)||²
            ll_plus = -0.5 * np.sum((y - pred_plus)**2) / (self.noise_std**2)
            ll_minus = -0.5 * np.sum((y - pred_minus)**2) / (self.noise_std**2)
            
            grad[i] = (ll_plus - ll_minus) / (2 * eps)
        
        # 잔 배치에 맞게 잣대를 맞춘다
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
            # 잔 배치를 얻는다
            idx = np.random.choice(N, min(self.batch_size, N), replace=False)
            X_batch = X[idx]
            y_batch = y[idx]
            
            # 러닝 비율
            lr = self._learning_rate(t)
            
            # 그래디언트를 셈한다
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
            뉴럴 네트워크 구조
        prior_std : float
            앞선 분포의 표준편차
        noise_std : float
            살핌 잡음의 표준편차
        n_iterations : int
            MAP을 찾는 가장 좋게 하기 회귀 횟수
        lr : float
            MAP 가장 좋게 하기의 러닝 비율
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
        """그래디언트를 수로 셈한다."""
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
    베이즈 뉴럴 네트워크를 위한 평균 마당 변이 미루어 봄.
    
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
            뉴럴 네트워크 구조
        prior_std : float
            앞선 분포의 표준편차
        noise_std : float
            살핌 잡음의 표준편차
        n_iterations : int
            가장 좋게 하기 회귀 횟수
        lr : float
            러닝 비율
        n_mc_samples : int
            그래디언트를 어림할 MC 표본의 수
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
        """파라메터 다시 잡기 재주로 짐을 뽑는다."""
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
        ELBO와 그래디언트를 셈한다.
        
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
            
            # 앞으로 스텝
            weights = self.network.unflatten_weights(theta)
            pred = self.network.forward(X, weights)
            
            # 로그 그럴듯함
            ll = -0.5 * np.sum((y - pred)**2) / (self.noise_std**2)
            total_ll += ll
            
            # 단순하게 수로 셈하는 그래디언트
            delta = 1e-5
            
            for i in range(len(self.mu)):
                # mu에 대한 그래디언트
                theta_plus = theta.copy()
                theta_plus[i] += delta
                weights_plus = self.network.unflatten_weights(theta_plus)
                pred_plus = self.network.forward(X, weights_plus)
                ll_plus = -0.5 * np.sum((y - pred_plus)**2) / (self.noise_std**2)
                
                grad_mu_ll[i] += (ll_plus - ll) / delta
                
                # log_sigma에 대한 그래디언트(파라메터 다시 잡기를 지나)
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
        
        # KL 갈림과 그래디언트
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
        """변이 파라메터를 가장 좋게 한다."""
        N = len(X)
        n_params = self.network.n_params()
        
        # 변이 파라메터의 첫자리를 잡는다
        self.mu = np.random.randn(n_params) * 0.1
        self.log_sigma = np.ones(n_params) * np.log(0.1)
        
        self.elbo_history = []
        
        for t in range(self.n_iterations):
            # ELBO와 그래디언트를 셈한다
            elbo, grad_mu, grad_log_sigma = self._elbo(X, y, N)
            
            # 고친다
            self.mu += self.lr * grad_mu
            self.log_sigma += self.lr * 0.1 * grad_log_sigma  # 흩어짐에는 더 작은 러닝 비율
            
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
    
    첫값을 달리해 네트워크 M개를 서로 남남으로 익힌다.
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
            뉴럴 네트워크 구조
        n_members : int
            모둠 클래스의 수
        noise_std : float
            살핌 잡음의 표준편차
        n_iterations : int
            클래스마다의 익힘 회귀 횟수
        lr : float
            러닝 비율
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
        """모둠 클래스 하나를 익힌다."""
        np.random.seed(seed)
        
        weights = self.network.init_weights()
        theta = self.network.flatten_weights(weights)
        
        for t in range(self.n_iterations):
            # 그래디언트를 셈한다(수로)
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
        """MSE의 그래디언트를 수로 셈한다."""
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
        """모둠 클래스를 모두 익힌다."""
        self.members = []
        
        for m in range(self.n_members):
            theta = self._train_member(X, y, seed=m * 42)
            self.members.append(theta)
        
        print(f"모둠: 클래스 {self.n_members}개를 익혔다")
    
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
    
    # 네트워크를 만든다
    network = SimpleNN([1, 20, 1], activation='tanh')
    print(f"네트워크: {network.layer_sizes}, 파라메터 {network.n_params()}개")
    
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
    print("확률 그래디언트 랑주뱅 액션")
    print("=" * 70)
    
    np.random.seed(42)
    
    # 단순한 1차 문제
    N = 30
    X_train = np.random.uniform(-3, 3, N).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, (N, 1))
    
    network = SimpleNN([1, 10, 1], activation='tanh')
    
    print(f"\n파라메터 {network.n_params()}개로 SGLD를 돌리는 중...")
    
    sgld = SGLD(
        network,
        prior_std=1.0,
        noise_std=0.2,
        lr_init=0.001,
        lr_decay=0.55,
        n_iterations=3000,
        burn_in=1500,
        thinning=5,
        batch_size=N  # 든든하도록 온 배치
    )
    
    sgld.fit(X_train, y_train)
    
    # 따진다
    X_test = np.linspace(-5, 5, 100).reshape(-1, 1)
    mean, std = sgld.predict(X_test)
    
    print(f"\n모은 표본: {len(sgld.samples)}")
    print(f"미루어 봄의 평균 표준편차: {np.mean(std):.3f}")
    
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
    
    # VI을 빠르게 하려고 작은 네트워크
    network = SimpleNN([1, 10, 1], activation='tanh')
    
    print(f"\n파라메터 {network.n_params()}개로 VI을 돌리는 중...")
    print("(수로 셈하는 그래디언트 탓에 좀 걸릴 수 있다)")
    
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

**출력:**

```
======================================================================
뒷분포 미루어 봄 방법 견주기
======================================================================

익힘 자료: 20점
참 함수: sin(x)
네트워크: [1, 20, 1], 파라메터 61개

--- 라플라스 어림 ---
라플라스: MAP을 찾고 뒷분포 흩어짐을 셈했다

--- 깊은 모둠 ---
모둠: 클래스 5개를 익혔다

--- 따지기 ---
방법                   NLL     RMSE    덮음90%
---------------------------------------------
Laplace          219.130 6142289553577.813    1.50%
Ensemble           5.349    0.710   58.50%

======================================================================
확률 그래디언트 랑주뱅 액션
======================================================================

파라메터 31개로 SGLD를 돌리는 중...
SGLD: 표본 300개를 모았다

모은 표본: 300
미루어 봄의 평균 표준편차: 0.207

======================================================================
평균 마당 변이 미루어 봄
======================================================================

파라메터 31개로 VI을 돌리는 중...
(수로 셈하는 그래디언트 탓에 좀 걸릴 수 있다)
VI: 가장 좋게 하기를 마쳤다, 마지막 ELBO = nan

뒷분포 자:
  평균 |μ|: nan
  평균 σ:   nan
  가장 큰 σ:    nan
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
ReLU 활성화와 가우시안 웨이트 사전분포를 지닌 2-레이어 뉴럴 네트워크에서, 평균장 변분 추론 · 라플라스 근사 · MC 드롭아웃이 저마다 내놓는 근사 사후분포의 형태를 견주어라.

</div>

??? success "풀이"
    웨이트가 $W_1, W_2$이고 가우시안 사전분포가 $p(W) = \mathcal{N}(0, \sigma_p^2 I)$일 때 사후분포 $p(W | D) \propto p(D | W) p(W)$은 다룰 수 없다. 세 근사 방법은 저마다 다룰 수 있는 형태를 낸다. 변분 추론이면 웨이트마다 서로 독립인 가우시안 사후분포 $q(w_{ij}) = \mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$을 지니고, 라플라스 근사이면 사후분포가 MAP 추정을 중심으로 하고 공분산이 헤세 행렬의 역인 가우시안 하나이며, MC 드롭아웃이면 사후분포가 드롭아웃 마스크 분포로 암묵적으로 정의된다. 근사마다 참 사후분포 형태의 서로 다른 특성을 담는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
SGLD나 라플라스 근사에서 얻은 불확실성 근사의 캘리브레이션을 MC 드롭아웃, 딥 앙상블과 비교하는 실험을 설계하여라. 쓸 지표와 그림을 밝혀라.

</div>

??? success "풀이"
    지표: (1) 버킷 15개의 기대 캘리브레이션 오차(ECE), (2) 브라이어 점수, (3) 음의 로그 우도(NLL), (4) 분포 외 탐지의 AUROC. 그림: 방법마다 관측 빈도를 예측 신뢰도에 대고 그린 신뢰도 다이어그램(reliability diagram). 절차: 모든 방법을 CIFAR-10(분포 내)에서 트레인하고, CIFAR-10 테스트 데이터에서 캘리브레이션을, SVHN에서 분포 외 탐지를 평가한다. 사후(post-hoc) 온도 스케일링을 베이스라인으로 쓴다. 무작위 시드 5개에 걸친 평균과 표준편차를 보고한다. 캘리브레이션이 잘 된 방법은 신뢰도 다이어그램에서 점이 대각선에 가깝고 ECE가 낮다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff hard" title="어려움"></span>
베이즈 뉴럴 네트워크의 예측 분산이 인식적 불확실성과 우연적 불확실성으로 분해됨을 증명하여라. 트레인 데이터 크기 $N \to \infty$일 때 두 항이 어떻게 되는지 보여라.

</div>

??? success "풀이"
    예측 분산은 전분산 법칙으로 분해된다. $\text{Var}[y | x, D] = \underbrace{\mathbb{E}_{p(\theta|D)}[\text{Var}[y | x, \theta]]}_{\text{타고난}} + \underbrace{\text{Var}_{p(\theta|D)}[\mathbb{E}[y | x, \theta]]}_{\text{앎의}}$. 우연적 항은 데이터 생성 과정의 줄일 수 없는 잡음을 담으며 $N \to \infty$이어도 그대로다. 인식적 항은 파라메터의 불확실성을 드러내며 사후분포가 참 파라메터 부근으로 수렴하므로 $O(1/N)$으로 준다. 극한에서는 우연적 불확실성만 남는다. 이 분해는 데이터를 더 모아야 할 때(인식적 불확실성이 클 때)와 우연적 잡음을 받아들여야 할 때(우연적 불확실성이 클 때)를 가르는 데 중요하다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 절의 불확실성 추정 방법을 트레이딩 시스템의 포지션 크기 결정에 어떻게 쓸 수 있는지 논하여라. 실용적인 판단 규칙을 제시하여라.

</div>

??? success "풀이"
    판단 규칙: 포지션 크기를 인식적 불확실성에 반비례하게 잡는다. $\hat{y}$을 예측 수익률, $\sigma_e^2$을 인식적 분산이라 하자. 포지션은 $w = \frac{\hat{y}}{\lambda \sigma_e^2}$이고 $\lambda$는 위험 회피 계수다. 인식적 불확실성이 크면(낯선 시장 상황) 포지션을 줄이고, 작으면(익숙한 국면) 더 과감하게 거래한다. 여기에 더해 인식적 불확실성의 상한을 두고 그 위로는 거래하지 않게 해 기권(abstain)할 수 있다. 이 틀은 모델의 신뢰도로 스케일한 켈리 기준 방식의 포지션 크기 결정을 자연스럽게 구현한다. 워크포워드(walk-forward) 분석으로 백테스트해 $\lambda$를 보정한다. $\square$

## 정리하며

### 추론 방법 개관

| 방법 | 접근 | 정확도 | 확장성 |
|--------|----------|----------|-------------|
| **HMC** | 정확한 MCMC | 높음 | 낮음 |
| **SGLD** | 확률 MCMC | 중간~높음 | 높음 |
| **라플라스** | MAP에서의 가우시안 | 중간 | 중간~높음 |
| **평균장 VI** | 곱으로 분해한 최적화 | 낮음~중간 | 높음 |
| **딥 앙상블** | MAP 여럿 | 중간~높음 | 중간 |
| **MC 드롭아웃** | 암묵적 VI | 낮음~중간 | 아주 높음 |

### 핵심 식

**사후분포**:

$$
p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) \, p(\theta)
$$

**SGLD 갱신**:

$$
\theta^{(t+1)} = \theta^{(t)} + \frac{\epsilon_t}{2} \nabla \log p(\theta^{(t)} \mid \mathcal{D}) + \mathcal{N}(0, \epsilon_t I)
$$

**ELBO**(변분 추론):

$$
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi}[\log p(\mathcal{D} \mid \theta)] - \text{KL}(q_\phi \| p)
$$

**라플라스 근사**:

$$
q(\theta) = \mathcal{N}(\theta_{\text{MAP}}, H^{-1})
$$

### 계산 복잡도

| 방법 | 트레인 | 추론 | 메모리 |
|--------|----------|-----------|---------|
| MAP | $O(E \cdot N)$ | $O(1)$ | $O(d)$ |
| SGLD | $O(T \cdot B)$ | $O(S)$ | $O(Sd)$ |
| 라플라스 | $O(E \cdot N + d^2)$ | $O(S)$ | $O(d^2)$ |
| VI | $O(E \cdot N)$ | $O(S)$ | $O(2d)$ |
| 앙상블 | $O(M \cdot E \cdot N)$ | $O(M)$ | $O(Md)$ |

### 방법 고르기 지침

| 상황 | 권장 방법 |
|----------|-------------------|
| 사후(post-hoc) 불확실성 | 라플라스, SWAG |
| 확장 가능한 트레인 | VI, MC 드롭아웃 |
| 가장 좋은 불확실성 | HMC(작을 때), SGLD(클 때) |
| 단순한 구현 | 앙상블, MC 드롭아웃 |
| 계산이 넉넉지 않을 때 | MC 드롭아웃, 라플라스 |

### 다른 장과의 연결

| 주제 | 장 | 연결 |
|-------|---------|------------|
| 사전분포 정하기 | 44.2절: 뉴럴 네트워크 웨이트 사전분포 | 사후분포의 입력 |
| 불확실성 | 44.3절: 뉴럴 네트워크 불확실성 정량화 | 사후분포에서 불확실성 분해가 나온다 |
| MC 드롭아웃 | 44.1절: 몬테카를로 드롭아웃 | 암묵적 변분 추론 |
| 변분 베이즈 뉴럴 네트워크 | 44.2절: 변분 베이즈 뉴럴 네트워크 | VI를 자세히 다룸 |
| 모델 비교 | 18.4절: 정보 기준 | 주변 우도 |

### 핵심 살펴볼 거리

- Welling, M., & Teh, Y. W. (2011). Bayesian learning via stochastic gradient Langevin dynamics. *ICML*.
- Blundell, C., et al. (2015). Weight uncertainty in neural networks. *ICML*.
- MacKay, D. J. (1992). A practical Bayesian framework for backpropagation networks. *Neural Computation*.
- Ritter, H., et al. (2018). A scalable Laplace approximation for neural networks. *ICLR*.
- Lakshminarayanan, B., et al. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Maddox, W., et al. (2019). A simple baseline for Bayesian inference in deep learning. *NeurIPS*.
