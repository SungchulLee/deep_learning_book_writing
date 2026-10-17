# 34.4.3 Soft Actor-Critic(SAC)

SAC(하르노야 외, 2018)은 벗어난 폴리시 Actor-Critic 학습과 가장 큰 엔트로피 강화학습을 엮는다. 리워드에 엔트로피 덤을 더해, 거의 가장 좋은 폴리시를 배우면서도 살펴보기를 북돋운다. 확률 폴리시, 저절로 벼리는 온도, 쌍둥이 크리틱 덕분에 SAC은 가장 굳세고 샘플링을 아끼는 이어진 다스리기 알고리즘 가운데 하나다.

---

## 1. 가장 큰 엔트로피 목표

SAC은 엔트로피를 더한 리턴을 가장 크게 한다.

$$J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]$$

여기서 $\alpha > 0$은 살펴보기와 써먹기의 트레이드오프를 다스리는 온도 파라메터다.

이 목표 아래에서 가장 좋은 폴리시는 다음과 같다.

$$\pi^*(a|s) \propto \exp\left(\frac{1}{\alpha} Q^*(s, a)\right)$$

---

## 2. 알고리즘 조각

### 1. 확률 폴리시(눌러 담은 가우스)

DDPG/TD3의 결정론적 폴리시와 달리 SAC은 눌러 담은 가우스로 파라메터를 나타낸 확률 폴리시를 쓴다.

$$a = \tanh(\mu_\theta(s) + \sigma_\theta(s) \odot \epsilon), \quad \epsilon \sim \mathcal{N}(0, I)$$

다시 파라메터 매기기 재주로 샘플링을 거쳐 그래디언트가 흐르게 한다.

### 2. 쌍둥이 Q 함수

SAC은 (TD3처럼) Q 네트워크 둘을 써서 지나친 어림을 누그러뜨린다.

$$Q_\text{타깃} = r + \gamma(1-d)\left(\min_{i=1,2} Q_{\phi'_i}(s', a') - \alpha \log \pi_\theta(a'|s')\right)$$

### 3. 저절로 벼리는 온도

SAC은 겨눈 엔트로피 $\bar{\mathcal{H}}$을 지키도록 $\alpha$을 저절로 맞춘다.

$$\alpha^* = \arg\min_\alpha \mathbb{E}_{a \sim \pi}\left[-\alpha \log \pi(a|s) - \alpha \bar{\mathcal{H}}\right]$$

겨눈 엔트로피는 흔히 $\bar{\mathcal{H}} = -\dim(\mathcal{A})$(액션 차원의 음수)으로 둔다.

---

## 3. SAC 고침 규칙

**Q 함수 고침**:

$$L(\phi_i) = \mathbb{E}\left[\left(Q_{\phi_i}(s,a) - y\right)^2\right]$$

$$y = r + \gamma(1-d)\left(\min_j Q_{\phi'_j}(s', \tilde{a}') - \alpha \log \pi_\theta(\tilde{a}'|s')\right)$$

**폴리시 고침**:

$$L(\theta) = \mathbb{E}_{s \sim \mathcal{D}}\left[\alpha \log \pi_\theta(\tilde{a}|s) - \min_i Q_{\phi_i}(s, \tilde{a})\right]$$

여기서 $\tilde{a}$은 다시 파라메터 매기기로 뽑는다.

**온도 고침**:

$$L(\alpha) = \mathbb{E}_{a \sim \pi}\left[-\alpha(\log \pi_\theta(a|s) + \bar{\mathcal{H}})\right]$$

---

## 4. SAC의 이로움

- **굳센 살펴보기**: 엔트로피를 크게 하여 너무 일찍 모이는 것을 막는다
- **샘플링을 아낌**: 되돌려 보기 버퍼를 쓰는 벗어난 폴리시다
- **저절로 벼림**: 온도가 과제에 맞춰 간다
- **든든함**: 쌍둥이 크리틱 + 부드러운 고침 + 확률 폴리시
- **잡음 벼리기가 없음**: 살펴보기가 가장 큰 엔트로피 목표에서 절로 우러난다

---

## 5. 파라메터

| 파라메터 | 기본값 | 뜻 |
|-----------|---------|-------------|
| 액터/크리틱 러닝레이트 | $3 \times 10^{-4}$ | 러닝레이트 |
| $\alpha$ 러닝레이트 | $3 \times 10^{-4}$ | 온도의 러닝레이트 |
| $\gamma$ | 0.99 | 깎기 인자 |
| $\tau$ | 0.005 | 부드러운 고침 계수 |
| 겨눈 엔트로피 | $-\dim(\mathcal{A})$ | 엔트로피 타깃 |
| 버퍼 크기 | $10^6$ | 되돌려 보기 용량 |
| 배치 크기 | 256 | 작은 배치 크기 |

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
이 절에서 밝힌 방법의 폴리시 경사를 이끌어 내어라. 어느 마디가 어림해야 하는 것이고 어느 마디가 딱 맞게 셈할 수 있는 것인지 또렷이 밝혀라.

</div>

??? success "풀이"
    폴리시 경사는 $\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\sum_t \nabla_\theta \log \pi_\theta(a_t | s_t) \cdot \hat{A}_t]$ 꼴이며 여기서 $\hat{A}_t$은 어드밴티지 어림이다. 로그 확률 그래디언트 $\nabla_\theta \log \pi_\theta$은 저절로 미분으로 딱 맞게 셈할 수 있다. 어드밴티지 $\hat{A}_t$은 뽑은 추적에서 어림해야 하므로 분산이 들어온다. 기댓값은 추적 배치에 걸쳐 고르게 하여 어림한다. 베이스라인으로 분산을 줄이면 치우치지 않음을 지키면서 어림 잡음을 줄인다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이어진 다스리기 과제에서 이 방법의 샘플링 효율을 밸류 기반 길(보기로 DQN)과 비교하여라. 보이는 다름의 이치 까닭을 풀어라.

</div>

??? success "풀이"
    폴리시 기반 방법은 폴리시 안 데이터를 쓰므로(추적마다 한 번씩 쓴다) 밸류 기반 방법보다 샘플링 효율이 대체로 낮다. DQN은 경험 리플레이로 데이터를 되써서 더 나은 샘플링 효율을 이룬다. 그러나 폴리시 방법은 이어진 액션을 절로 다루고(액션 공간에 대한 argmax가 필요 없다), 가장 좋은 것이 확률 폴리시일 때 그리로 모이며, 신뢰 영역 아래에서 한결같은 나아짐을 보장한다. 벗어난 폴리시 Actor-Critic 방법(DDPG, SAC)은 폴리시 최적화와 경험 리플레이를 엮어 이 사이를 메운다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
쉬운 이어진 다스리기 과제(보기로 Pendulum-v1)에 이 방법을 만들어라. 러닝레이트와 이 방법에 딸린 중요한 파라메터에 대해 얼마나 예민한지 알려라.

</div>

??? success "풀이"
    가우스 폴리시를 쓰는 Pendulum-v1에서 흔한 성능은 이렇다. 러닝레이트 $3 \times 10^{-4}$이면 약 500 에피소드에 모이고, $10^{-3}$이면 흔들리며, $10^{-5}$이면 너무 더디게 모인다. 이 방법에 딸린 파라메터(보기로 PPO의 자르는 너비, TRPO의 쿨백-라이블러 매임)는 고침의 사나움과 든든함 사이의 트레이드오프를 다스린다. 너무 사나우면 성능이 무너지고 너무 조심스러우면 샘플링을 버린다. 가장 좋은 자리는 이 둘의 저울을 맞추는 곳이며 흔히 좁은 너비에서 격자 찾기로 얻는다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
액션 공간이 단체(합이 1인 자본 웨이트)이고 리워드가 위험을 맞춘 리턴인 포트폴리오 배분 최적화에 이 방법을 어떻게 쓸 수 있을지 따져라.

</div>

??? success "풀이"
    액션 공간은 $(n-1)$차원 단체 $\Delta^{n-1} = \{w \in \mathbb{R}^n : w_i \geq 0, \sum_i w_i = 1\}$이다. 폴리시는 디리클레 분포나 소프트맥스로 바꾼 가우스를 쓸 수 있다. 리워드는 그 포트폴리오 배분의 샤프 비나 미분 샤프 비다. 어려움에는 높은 차원의 액션 공간(자산이 많다), 잦은 다시 맞추기에 벌을 주는 거래 비용, 저자 리턴의 흐름 바뀜이 있다. 이 절의 방법은 든든하게 폴리시를 고치는 제 장치로 이를 다룬다. $\square$

## 정리하며

SAC은 가장 큰 엔트로피 강화학습과 벗어난 폴리시 Actor-Critic 학습을 하나로 아울러 이어진 다스리기에서 가장 앞선 샘플링 효율을 이룬다. 확률 폴리시, 저절로 벼리는 온도, 쌍둥이 크리틱의 어우름이 파라메터를 거의 벼리지 않아도 되는 굳센 알고리즘을 만든다.
