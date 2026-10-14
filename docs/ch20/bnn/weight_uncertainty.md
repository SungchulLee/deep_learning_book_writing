# 베이즈 신경망에서 가중치의 불확실성

베이즈 신경망의 한가운데 어려움은 가중치에 뜻있는 사전확률을 정하고 그 결과로 나오는 사후확률을 추론하는 것이다. 이 절에서는 사전확률 정하기, 신경망 가중치 공간의 기하, 주요 추론 방법을 다룬다.

!!! note "온전히 다루기"
    두루 다룬 구현과 잣대 실험은 **[44장: 사전확률 정하기](../../ch43/bayesian_methods/priors.md)**와 **[44장: 사후확률 추론](../../ch43/bayesian_methods/posterior.md)**을 보아라.

---

## 1. 가중치에 두는 사전확률 분포

### 표준 가우스 사전확률

가장 단순하고 흔한 공정성이다:

$$
p(\theta) = \prod_{l=1}^L \prod_{i,j} \mathcal{N}(w_{ij}^{(l)} \mid 0, \sigma_l^2)
$$

**L2 정칙화와의 이음**: 가우스 사전확률을 쓴 MAP 추정은 $\lambda = 1/(2\sigma^2)$인 L2 벌준 최대 우도 어림과 같다.

### 분산 크기 맞추기

신호가 안정되게 퍼지도록 사전확률의 분산은 층의 너비에 맞춰 크기를 잡아야 한다:

| 방법 | 공식 | 어디에 좋은가 |
|--------|---------|----------|
| **글로로/자비에** | $\sigma^2 = \frac{2}{n_{\text{in}} + n_{\text{out}}}$ | Tanh, Sigmoid |
| **허** | $\sigma^2 = \frac{2}{n_{\text{in}}}$ | ReLU |

### 성김을 이끄는 사전확률

**라플라스 사전확률**(L1 정칙화):

$$
p(w) = \frac{1}{2b} \exp\left(-\frac{|w|}{b}\right)
$$

**편자 사전확률**(맞춰 가는 오그라들기):

$$
w_j \mid \lambda_j \sim \mathcal{N}(0, \lambda_j^2 \tau^2), \quad \lambda_j \sim \text{Half-Cauchy}(0, 1)
$$

편자는 작은 가중치를 0 쪽으로 세게 오그라뜨리면서 큰 가중치는 거의 건드리지 않는다. 성긴 망에 딱 맞다.

**못과 판**(짜임새 있는 성김):

$$
p(w_j) = (1-\pi) \delta_0(w_j) + \pi \mathcal{N}(w_j \mid 0, \sigma_{\text{slab}}^2)
$$

### 층 사전확률

사전확률의 분산에 웃앞확률을 둔다:

$$
w_j \sim \mathcal{N}(0, \sigma^2), \quad \sigma^2 \sim \text{Inverse-Gamma}(\alpha, \beta)
$$

이러면 데이터가 알맞은 정칙화의 세기를 정하게 된다. 곧 저절로 관련성 정하기(ARD)이다.

---

## 2. 사후확률 추론 방법

### 추론의 어려움

신경망의 사후확률 $p(\theta \mid \mathcal{D})$은:

- 닫힌 형태가 없다
- 수백만 차원의 공간에 있다
- (대칭 때문에) 봉우리가 많다
- 복잡하고 볼록하지 않은 기하를 갖는다

### MCMC 방법

**해밀토니안 몬테카를로(HMC)**: 기울기 정보로 효율적으로 살펴본다. 정확도의 금과옥조이지만 큰 망에는 감당할 수 없이 비싸다.

**확률 기울기 랑주뱅 동역학(SGLD)**: 확률 기울기 내리기의 새로 고치기에 눈금 맞춘 잡음을 더한다:

$$
\theta_{t+1} = \theta_t + \frac{\eta_t}{2} \left(\nabla \log p(\theta_t) + \frac{N}{n} \sum_{i \in \text{batch}} \nabla \log p(x_i \mid \theta_t)\right) + \epsilon_t
$$

여기서 $\epsilon_t \sim \mathcal{N}(0, \eta_t \mathbf{I})$이다. $\eta_t \to 0$이면 표본이 참 사후확률로 모인다.

### 변분 추론

다룰 수 있는 집안으로 사후확률을 어림한다([Bayes by Backprop](bayes_by_backprop.md) 참고):

$$
q_\phi(\theta) \approx p(\theta \mid \mathcal{D})
$$

ELBO를 가장 크게 하여 최적화한다([19장: ELBO](../variational_inference/elbo.md) 참고).

### 라플라스 근사

MAP 어림값에 가우스를 맞춘다:

$$
p(\theta \mid \mathcal{D}) \approx \mathcal{N}(\theta \mid \theta_{\text{MAP}}, \mathbf{H}^{-1})
$$

여기서 $\mathbf{H} = -\nabla^2 \log p(\theta \mid \mathcal{D}) \big|_{\theta_{\text{MAP}}}$은 헤세 행렬이다.

큰 망에서는 헤세 행렬을 어림해야 한다. 곧 대각, KFAC, 또는 낮은 계수 나타냄을 쓴다.

### 넌지시 하는 방법

**몬테카를로 떨구기**: 시험할 때의 떨구기를 어림 변분 추론으로 다시 풀이한다(갈과 가라마니, 2016):

$$
p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{T} \sum_{t=1}^T p(y^* \mid x^*, \hat{w}_t)
$$

여기서 $\hat{w}_t$은 무작위 떨구기 마스크를 씌운 가중치를 나타낸다.

**깊은 앙상블**: 서로 다른 첫값에서 독립인 망 $M$개를 학습된이다:

$$
p(y^* \mid x^*, \mathcal{D}) \approx \frac{1}{M} \sum_{m=1}^M p(y^* \mid x^*, \theta_m)
$$

엄밀히 베이즈는 아니지만 실전에서 아주 좋은 불확실성 어림값을 준다.

---

## 3. 방법 비교

| 방법 | 정확도 | 값 | 단순함 | 메모리 |
|--------|----------|------|------------|--------|
| HMC | ★★★★★ | ★☆☆☆☆ | ★☆☆☆☆ | ★☆☆☆☆ |
| SGLD | ★★★★☆ | ★★★☆☆ | ★★★☆☆ | ★★☆☆☆ |
| 변분 | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | ★★★☆☆ |
| 라플라스 | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ |
| 몬테카를로 떨구기 | ★★☆☆☆ | ★★★★★ | ★★★★★ | ★★★★★ |
| 앙상블 | ★★★★☆ | ★★★☆☆ | ★★★★★ | ★★☆☆☆ |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff med" title="중간"></span>
베이즈 신경망이 앎 불확실성을 어떻게 재는지 설명하여라. 이는 우연 불확실성과 어떻게 다른가?

</div>

??? success "연습문제 1 풀이"
    베이즈 신경망은 점 어림값 하나가 아니라 가중치에 대한 사후확률 분포 $p(w | \mathcal{D})$을 지닌다. **앎 불확실성**(모델의 불확실성)은 데이터가 모자라서 생기며 사후확률의 퍼짐으로 담긴다. 곧 데이터가 성긴 구역에서는 사후확률이 넓어 다양한 미리봄이 나온다. **우연 불확실성**(데이터의 잡음)은 데이터에 본디 있으며 내임 분포 $p(y|x,w)$으로 담긴다. 앎 불확실성은 데이터가 늘면 줄지만 우연 불확실성은 줄지 않는다. 베이즈 신경망은 미리봄 분산으로 앎 불확실성을 잰다: $\text{Var}[y|x] = \underbrace{\mathbb{E}_w[\text{Var}[y|x,w]]}_{\text{aleatoric}} + \underbrace{\text{Var}_w[\mathbb{E}[y|x,w]]}_{\text{epistemic}}$.

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff hard" title="어려움"></span>
ELBO 목표에 대한, 되짚음으로 하는 베이즈의 기울기 어림꼴을 이끌어 내어라.

</div>

??? success "연습문제 2 풀이"
    ELBO는 $\mathcal{L}(\theta) = \mathbb{E}_{q_\theta(w)}[\log p(\mathcal{D}|w)] - D_{\text{KL}}(q_\theta(w) \| p(w))$이다. $w = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$인 매개변수 바꾸기 재주를 쓰면:

    $$\nabla_\theta \mathcal{L} = \nabla_\theta \left[ \log p(\mathcal{D}|w) + \log p(w) - \log q_\theta(w) \right]_{w = \mu + \sigma \odot \epsilon}$$

    이는 $\epsilon$을 샘플링하고 정해진 바꿈을 거쳐 기울기를 셈해 어림한다. $q$과 $p$이 가우스이면 KL 항은 흔히 닫힌 형태로 셈할 수 있다.

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
베이즈 신경망에서 몬테카를로 떨구기와 드러난 변분 추론을 구현의 복잡함과 캘리브레이션의 질로 비교하여라.

</div>

??? success "연습문제 3 풀이"
    **몬테카를로 떨구기**는 시험할 때도 떨구기를 쓰고 순방향 $T$번의 평균을 낸다. 구현이 단순하지만(떨구기를 넣고 순방향을 여러 번 돌린다) 좁은 변분 집안(베르누이 곱 잡음)에 맞대응되어 눈금이 잘 맞지 않는 불확실성이 나오는 일이 잦다. **드러난 변분 추론**(이를테면 되짚음으로 하는 베이즈)은 매개변수를 곱절로 늘리고(가중치마다 $\mu$과 $\sigma$) 꼼꼼한 최적화가 필요하지만 더 풍성한 사후확률 추정과 대체로 눈금이 더 잘 맞는 불확실성 어림값을 준다. 불확실성을 빠르게 어림하려면 몬테카를로 떨구기가, 캘리브레이션이 중요하면 드러난 변분 추론이 낫다.

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
베이즈 신경망에서 사전확률 $p(w)$을 고르는 일이 왜 중요하며, 표준 가우스 사전확률과 크기 섞음 사전확률 사이의 주고받음은 무엇인가?

</div>

??? success "연습문제 4 풀이"
    사전확률은 사후확률에 벌을 주며 학습의 움직임과 미리봄의 불확실성에 모두 영향을 준다. **표준 가우스** $\mathcal{N}(0, \sigma^2 I)$은 단순하고 L2 정칙화에 맞대응되지만 모든 가중치에 똑같이 벌을 주어 너무 제약할 수 있다. **크기 섞음** 사전확률(이를테면 $\pi \mathcal{N}(0, \sigma_1^2) + (1-\pi) \mathcal{N}(0, \sigma_2^2)$)은 어떤 가중치는 크게(신호) 두면서 다른 가중치는 0에 가깝게(잡음) 몰아 맞춰 가는 성김을 준다. 주고받음은 이렇다. 크기 섞음은 표현력이 더 좋지만 최적화가 더 어렵고 웃매개변수가 늘어난다.

## 정리하며

| 개념 | 핵심 |
|---------|-----------|
| **사전확률 고르기** | 정보의 많음과 계산으로 다룰 수 있음을 저울질해야 한다 |
| **분산 크기 맞추기** | 학습의 안정에 결정적이다. 허나 글로로 첫값 잡기를 쓴다 |
| **성김 사전확률** | 망을 줄이는 데 편자와 못과 판을 쓴다 |
| **MCMC** | 정확하지만 비싸다. SGLD가 규모를 키울 수 있는 어림을 준다 |
| **변분** | 빠르고 규모를 키울 수 있다. 불확실성을 낮춰 잡을 수 있다 |
| **몬테카를로 떨구기** | 가장 단순한 실전 방법이다. 어림의 질에 한계가 있다 |

---

**참고 문헌**

- Blundell, C., et al. (2015). Weight Uncertainty in Neural Networks. *ICML*.
- Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
- Welling, M., & Teh, Y. W. (2011). Bayesian Learning via Stochastic Gradient Langevin Dynamics. *ICML*.
- Lakshminarayanan, B., Pritzel, A., & Bluntschli, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. *NeurIPS*.
