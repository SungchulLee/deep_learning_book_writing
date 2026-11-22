# 35.1.3 액션 공간

---

## 1. 학습 목표

- 금융 강화학습 에이전트에 알맞은 액션 공간을 설계한다
- 이산 액션과 연속 액션의 형태를 견준다
- 제약을 곁들인 포트폴리오 웨이트 액션을 만든다
- 액션 마스킹과 실행 가능성 제약을 다룬다

---

## 2. 소개

액션 공간은 에이전트가 타임스텝마다 어떤 결정을 내릴 수 있는지를 정한다. 금융에서 액션은 흔히 거래 결정, 곧 자산마다 얼마나 사고팔지를 나타낸다. 이산 액션 공간과 연속 액션 공간 가운데 무엇을 고르느냐가 알고리즘 선택, 트레인의 강건성, 배운 폴리시가 표현할 수 있는 범위에 크게 영향을 미친다.

---

## 3. 액션 공간 정식화

### 1. 이산 액션

거래 결정을 유한한 선택지 집합으로 대응시킨다.

**단순한 매수/보유/매도(자산마다)**:

$$a_t \in \{\text{사기}, \text{쥐기}, \text{팔기}\}^N$$

자산이 $N$개면 결합 액션 공간에 원소가 $3^N$개이며 자산 수에 따라 지수적으로 늘어난다.

**이산화한 배분**:

$$a_t \in \{0, 0.1, 0.2, \ldots, 1.0\}^N \quad \text{단} \quad \sum_i a_{t,i} = 1$$

**장점**: DQN 계열 알고리즘에 쓸 수 있고 엡실론-그리디로 탐험이 명확하다.
**단점**: 자산 수에 따라 지수적으로 늘어나고 세밀한 배분을 나타낼 수 없다.

### 2. 연속 액션

목표 포트폴리오 웨이트를 곧바로 나타낸다.

$$a_t \in [0, 1]^N \quad \text{(사기만)}, \quad a_t \in [-1, 1]^N \quad \text{(사고팔기)}$$

네트워크가 내놓는 원시 출력을 제약에 맞도록 변환한다.

**소프트맥스 정규화**(롱 온리, 완전 투자):

$$w_i = \frac{e^{a_i}}{\sum_j e^{a_j}}$$

**단체 사영**(현금을 허용하는 롱 온리):

$$w = \text{proj}_{\Delta}(a), \quad \Delta = \{w : w_i \geq 0, \sum_i w_i \leq 1\}$$

**tanh와 재스케일링**(롱숏):

$$w_i = \frac{\tanh(a_i)}{\sum_j |\tanh(a_j)|} \cdot L_{\max}$$

여기서 $L_{\max}$는 최대 총 레버리지다.

**장점**: 세밀하게 제어할 수 있고 PPO, SAC, TD3에 쓸 수 있다.
**단점**: 탐험이 더 어렵고 제약을 채우려면 꼼꼼한 설계가 있어야 한다.

### 3. 혼합 액션

이산 구성 요소와 연속 구성 요소를 결합한다.

- **이산 부분**: 어느 자산을 거래할지(선택)
- **연속 부분**: 고른 자산에 얼마를 나눌지

거래할 자산의 전체 집합이 크지만 스텝마다 몇 개만 바뀔 때 쓸모 있다.

---

## 4. 제약 다루기

### 포트폴리오 제약

실제 포트폴리오에는 흔히 반드시 지켜야 할 제약이 있다.

| 제약 | 수식 형태 | 적용 방법 |
|-----------|-------------------|-------------|
| 완전 투자 | $\sum_i w_i = 1$ | 소프트맥스 정규화 |
| 롱 온리 | $w_i \geq 0$ | 클리핑 후 재정규화 |
| 포지션 상한 | $w_i \leq w_{\max}$ | 클리핑 후 재분배 |
| 레버리지 상한 | $\sum_i |w_i| \leq L$ | 넘으면 스케일 축소 |
| 자산군 상한 | $\sum_{i \in S_k} w_i \leq c_k$ | 반복 사영 |
| 최소 거래 크기 | $|\Delta w_i| \geq \delta$ 또는 $0$ | 임계값 미만은 0으로 |

### 액션 변환 파이프라인

```python
def transform_action(self, raw_action):
    """네트워크가 내놓은 날값을 옳은 밑천 무게로 옮긴다."""

    # 스텝1: 엑티베이션 함수 매기기
    if self.long_only:
        weights = torch.softmax(raw_action, dim=-1)
    else:
        weights = torch.tanh(raw_action)

    # 스텝2: 자리 위끝 지키기
    weights = torch.clamp(weights, -self.max_position, self.max_position)

    # 스텝3: 지렛대 매임 지키기
    gross_leverage = weights.abs().sum()
    if gross_leverage > self.max_leverage:
        weights = weights * (self.max_leverage / gross_leverage)

    # 스텝4: 거래량 셈하기(지금 자리와의 차이)
    trades = weights - self.current_weights

    # 스텝5: 가장 작은 거래 크기 지키기
    small_trades = trades.abs() < self.min_trade
    trades[small_trades] = 0.0
    weights = self.current_weights + trades

    return weights
```

---

## 5. 액션 표현 비교

### 목표 비중과 거래 차이

**목표 비중**: $a_t = w_t^{\text{겨눔}}$

- 에이전트가 원하는 포트폴리오 구성을 출력한다
- 둘레가 필요한 거래를 계산한다
- 에이전트에게 더 쉽고, 액션을 상태와 무관하게 해석할 수 있다

**거래 차이**: $a_t = \Delta w_t = w_t^{\text{겨눔}} - w_{t-1}$

- 에이전트가 현재 포트폴리오에서의 변화를 출력한다
- 거래 비용을 살피기에 더 자연스럽다
- 현재 포지션을 알아야 한다(상태 의존적 해석)

### 주문 기반 액션

주문 실행 단계의 강화학습에서는 다음과 같다.

$$a_t = (\text{쪽}, \text{수량}, \text{값}, \text{주문\_클래스})$$

- **방향**: 매수 또는 매도
- **수량**: 주식/계약의 개수
- **가격**: 지정가 또는 시장가 플래그
- **주문 유형**: 시장가, 지정가, 손절 따위

---

## 6. 구현

```python
# 자산 N개를 위한 이어진 액션 공간(남은 돈을 둔 사기만)
action_space = spaces.Box(
    low=0.0,
    high=1.0,
    shape=(num_assets,),
    dtype=np.float32
)

# 따로 떨어진 액션 공간(자산마다 액션 3개, 인수로 나눔)
action_space = spaces.MultiDiscrete([3] * num_assets)

# 섞음: 자산 K개를 고른 뒤 나누기
action_space = spaces.Dict({
    'selection': spaces.MultiBinary(num_assets),
    'allocation': spaces.Box(0, 1, (num_assets,))
})
```

---

## 7. 액션 마스킹

어떤 상태에서는 실행할 수 없는 액션이 있다.

- 보유하지 않은 자산을 팔 수 없다(공매도가 막혀 있으면)
- 남은 돈이 없으면 살 수 없다
- 거래가 멈춘 주식은 거래할 수 없다

```python
def get_action_mask(self):
    mask = np.ones(self.action_space.n, dtype=bool)

    # 남은 돈이 없으면 살 수 없다
    if self.portfolio.cash <= 0:
        mask[self.BUY_ACTIONS] = False

    # 자리가 없으면 팔 수 없다
    for i, pos in enumerate(self.portfolio.positions):
        if pos <= 0:
            mask[self.SELL_ACTIONS[i]] = False

    return mask
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
이 절에서 다룬 금융 문제를 위해 Gymnasium과 호환되는 둘레를 설계하여라. 관찰 공간, 액션 공간, 리워드 함수, 에피소드 종료 조건을 밝혀라.

</div>

??? success "풀이"
    관찰 공간: 최근 수익률, 현재 포지션, 자본 가치, 관련 시장 피처(예를 들어 변동성, 거래량)를 담은 벡터. 액션 공간: 문제에 달렸다(매수/보유/매도라면 이산, 포지션 크기 결정이라면 연속). 리워드: 스텝마다 위험 조정 수익률(예를 들어 로그 수익률에서 위험 벌점을 뺀 것). 에피소드는 정해진 기간(예를 들어 거래 한 해)이 지나거나 자본 가치가 임계값 아래로 떨어지면(마진 콜) 끝난다. 둘레는 거래 비용, 슬리피지, 시장 충격을 현실에 가깝게 모델링해야 한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 금융 강화학습 문제에서 리워드 셰이핑의 트레이드오프를 분석하라. 후보 리워드 함수를 적어도 셋 비교하고, 저마다 최적 폴리시의 어떤 성질을 보존하는지 논하라.

</div>

??? success "풀이"
    후보 리워드: (1) 원시 손익 -- 단순하지만 분산이 크고 지연된다. (2) 샤프 기반 미분 리워드 $D_t = \frac{\Delta A_t B_{t-1} - \frac{1}{2}\Delta B_t A_{t-1}}{(B_{t-1} - A_{t-1}^2)^{3/2}}$ -- 샤프 비율을 직접 최적화하지만 복잡하다. (3) 낙폭 벌점을 곁들인 로그 수익률 $r_t = \log(V_t/V_{t-1}) - \lambda \max(0, DD_t - \tau)$ -- 수익률과 위험 관리의 균형을 맞춘다. 퍼텐셜 기반 셰이핑 정리는 $\gamma\Phi(s') - \Phi(s)$을 더해도 최적 폴리시가 보존됨을 보장한다. 샤프 기반 리워드는 최적화 목표 자체를 바꾸므로(최적 폴리시가 달라질 수 있다) 그렇지 않고, 원시 손익은 폴리시를 보존하지만 더디게 배운다. $\square$

---

<div class="drillbox" markdown>

**연습문제 3.** <span class="diff med" title="중간"></span>
이 금융 응용에 따르는 비정상성 문제를 논하라. 에이전트를 레짐 변화에 적응시키는 구체적인 전략을 제안하라.

</div>

??? success "풀이"
    금융 시장에는 레짐 변화(상승장/하락장, 높은/낮은 변동성)가 있어 마르코프 결정 과정이 정상적이라는 가정을 깨뜨린다. 구체적인 적응 전략은 이렇다. (1) 레짐 지표를 상태의 일부로 넣는다(예를 들어 은닉 마르코프 모델로 추정한 레짐 확률). (2) 에이전트가 폴리시 여럿을 두고 탐지한 레짐에 따라 고르는 메타 학습 방법을 쓴다. (3) 최근 경험에 가중치를 더 준(지수 감쇠 가중치) 리플레이 버퍼로 연속 학습을 구현한다. 에이전트는 자기 성능도 모니터링하다가 분포 밖 입력이 드러나면 포지션 크기를 줄여야 한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 4.** <span class="diff med" title="중간"></span>
이 강화학습 방법과 단순한 휴리스틱 벤치마크의 백테스트 결과를 비교하여라. 에이전트가 정말 앞서는지 가리려면 어떤 통계 검정을 써야 하는가?

</div>

??? success "풀이"
    벤치마크: 매수 후 보유, 동일 비중 배분, 모멘텀 전략. 에이전트는 (트레인 데이터에서는 결코 하지 않고) 워크포워드 방식으로 샘플 밖 구간에서 평가해야 한다. 통계 검정: (1) 일간 수익률에 대한 대응 t 검정으로 평균 차이를 본다. (2) 샤프 비율 차이에 대한 부트스트랩 신뢰 구간을 얻는다. (3) 여러 전략을 비교한다면 다중 검정 보정(예를 들어 본페로니나 홀름)을 적용한다. 흔히 빠지는 함정은 시험 데이터에서 하이퍼파라미터를 튜닝해 p값을 해킹하는 일이다. 평가는 어떤 모델 선택에도 쓰이지 않은 홀드아웃 구간을 써야 한다. 통계적 유의성과 경제적 유의성(거래 비용, 감당할 수 있는 규모)을 함께 보고해야 한다. $\square$

## 정리하며

액션 공간 설계에는 표현력, 확장성, 트레인 효율 사이의 트레이드오프가 걸려 있다. 포트폴리오 배분에는 소프트맥스나 단체 사영을 쓰는 연속 액션이 가장 흔하고, 주문 실행 단계의 결정에는 이산 액션이 알맞다. 액션 변환 파이프라인으로 제약을 다루면 강화학습 알고리즘을 고치지 않고도 실행 가능성을 지킬 수 있다.

**참고 문헌**

- Lillicrap, T. P., et al. (2016). Continuous control with deep reinforcement learning (DDPG)
- Ye, Y., et al. (2020). Reinforcement-Learning based Portfolio Management with Augmented Asset Movement Prediction States
