# 35.1.5 시장 시뮬레이션

---

## 1. 학습 목표

- 강화학습 학습을 위한 참에 가까운 시장 시뮬레이터를 세운다
- 거래 비용, 미끄러짐, 저자 섭동을 그린다
- 참에 가까운 채워짐 여김으로 주문 벌이기를 만든다
- 둘레를 늘리려 가짜 시장 데이터를 지어낸다

---

## 2. 소개

시장 시뮬레이터는 에이전트의 액션이 실제 자본 바뀜으로 어떻게 옮겨지는지를 정한다. 온 주문을 마감값에 채워 주는 막무가내 흉내내개는 참되지 않은 바람을 심는다. 참에 가까운 흉내내기는 거래 비용, 미끄러짐, 조각만 채워짐, 저자 섭동을 계산에 넣어야 한다. 손에 쥘 수 있는 유동성에 비교하여 크게 거래하는 꾀라면 더욱 그렇다.

---

## 3. 주문 벌이기 모델

### 1. 쉬운 채워짐 모델

온 주문이 다음 마디의 시작값(또는 마감값)에 벌어진다.

$$\text{채워진\_값}_i = p_{t+1}^{\text{시작}} \cdot (1 + \text{미끄러짐}_i)$$

여기서 미끄러짐은 아무 값이거나 붙박인 사이 벌 값이다.

### 2. VWAP 벌이기 모델

한 마디에 걸쳐 벌이는 주문은 거래량으로 무게를 준 평균값으로 채워진 값을 어림한다.

$$\text{채워진\_값}_i = \text{VWAP}_{[t, t+\Delta t]} + \text{저자\_흔듦}_i$$

### 3. 지정가 주문 장부 흉내내기

높은 잦기 꾀에서는 지정가 주문 장부를 흉내 낸다.

- 사고파는 값 사이의 액션을 그린다
- 지정가 주문의 줄서기 자리를 다룬다
- 유동성을 먹어 치우며 생기는 저자 섭동을 다룬다

---

## 4. 거래 비용 모델

### 비교하는 비용

$$\text{TC}_t = c \cdot \sum_i |q_{t,i}| \cdot p_{t,i}$$

여기서 $c$은 비용 비율이고(주식에서 흔히 5~20 베이시스 포인트) $q_{t,i}$은 거래 수량이다.

### 사이 기초 비용

$$\text{TC}_t = \frac{1}{2} \sum_i |q_{t,i}| \cdot \text{사이}_{t,i}$$

반 사이 모델이다. 파는 값에 사고 사는 값에 판다.

### 켜 있는 수수료 모델

```python
def compute_commission(self, trade_value):
    if trade_value < 10_000:
        return max(1.0, trade_value * 0.005)  # 가장 적게 \$1, 50 베이시스 포인트
    elif trade_value < 100_000:
        return trade_value * 0.001              # 10 베이시스 포인트
    else:
        return trade_value * 0.0005             # 5 베이시스 포인트
```

---

## 5. 저자 섭동 모델

### 곧은 섭동

$$\Delta p = \lambda \cdot \frac{q}{V}$$

여기서 $q$은 주문 수량, $V$은 하루 평균 거래량, $\lambda$은 섭동 계수다.

### 제곱근 섭동(올름그렌-크리스)

$$\Delta p = \sigma \cdot \eta \cdot \text{sgn}(q) \cdot \sqrt{\frac{|q|}{V}}$$

여기서 $\sigma$은 흔들림이고 $\eta$은 맞춘 파라메터다. 이 모델은 거래 크기와 섭동 사이의 오목한 얽힘을 더 잘 담는다.

### 잠깐 가는 섭동과 오래가는 섭동

- **잠깐 가는 섭동**: 거래 뒤 값이 되돌아온다. 비용을 거래한 이만 치른다
- **오래가는 섭동**: 값이 아주 옮겨 간다. 뒤이은 모든 값 매김에 미친다

$$p_{t+1} = p_t + \underbrace{\gamma \cdot \frac{q}{V}}_{\text{오래감}} + \underbrace{\eta \cdot \text{sgn}(q) \cdot \sqrt{\frac{|q|}{V}}}_{\text{잠깐 감}} + \epsilon_t$$

---

## 6. 미끄러짐 그리기

미끄러짐은 바라던 값과 실제로 채워진 값의 차이를 담는다.

### 붙박인 미끄러짐

$$\text{미끄러짐} = \text{sign}(q) \cdot s$$

여기서 $s$은 붙박인 미끄러짐 양이다(보기로 1 베이시스 포인트).

### 확률 미끄러짐

$$\text{미끄러짐} \sim \mathcal{N}(\mu_s, \sigma_s^2)$$

$\mu_s$과 $\sigma_s$은 지난 벌이기 데이터로 맞춘다.

### 거래량에 딸린 미끄러짐

$$\text{미끄러짐} = s_0 + s_1 \cdot \frac{|q|}{V_t}$$

거래량에 비교하여 주문이 클수록 더 미끄러진다.

---

## 7. 가짜 데이터 지어내기

지난 데이터가 적으면 지나치게 맞추게 된다. 가짜 데이터로 늘리면 도움이 된다.

### 부트스트랩 다시 샘플링

지난 돌아옴을 되넣어 다시 뽑아 새 값 자취를 만든다.

$$R_t^{\text{가짜}} = R_{\pi(t)}^{\text{지난}}$$

여기서 $\pi$은 아무 자리바꿈이다(덩이 부트스트랩은 스스로 얽힘을 지킨다).

### 지어내는 모델

학습된 지어내는 모델로 참에 가까운 가짜 데이터를 만든다.

- **GARCH 모델**: 흔들림 뭉침을 담는다
- **판 바꿈 모델**: 저자 판의 액션을 담는다
- **GAN/VAE**: 얽힌 분포 피처를 배운다

### 잡음 넣기

지난 데이터에 맞춘 잡음을 더한다.

$$p_t^{\text{늘림}} = p_t \cdot e^{\epsilon_t}, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_{\text{늘림}}^2)$$

---

## 8. 마당 마구잡이로 바꾸기

굳셈을 높이려 익히는 동안 둘레 파라메터를 바꾼다.

```python
def randomize_params(self):
    """에피소드마다 둘레 파라메터를 아무렇게나 바꾼다."""
    self.transaction_cost = np.random.uniform(0.0005, 0.002)
    self.slippage_mean = np.random.uniform(0.0, 0.001)
    self.market_impact_coeff = np.random.uniform(0.05, 0.2)
    self.initial_capital *= np.random.uniform(0.8, 1.2)
```

---

## 9. 여러 자산의 서로 얽힘

여러 자산 둘레에서는 자산을 가로지르는 액션을 제대로 그려야 한다.

- **지난 서로 얽힘**: 굴러가는 서로 얽힘 행렬을 쓴다
- **코퓰러 모델**: 곧지 않은 매임을 담는다
- **인자 모델**: 돌아옴을 공통 인자와 남다른 조각으로 나눈다

---

## 10. 구현: 시장 시뮬레이터 클래스

```python
class MarketSimulator:
    def __init__(self, config):
        self.cost_rate = config.get('transaction_cost', 0.001)
        self.slippage_std = config.get('slippage_std', 0.0005)
        self.impact_coeff = config.get('impact_coeff', 0.1)
        self.volume_data = config.get('volume')

    def execute(self, target_weights, portfolio, current_prices):
        current_weights = portfolio.get_weights()
        trades = target_weights - current_weights

        # 미끄러짐과 저자 흔듦을 넣어 채워진 값을 셈한다
        fill_prices = self._compute_fill_prices(
            current_prices, trades, portfolio.total_value
        )

        # 거래 비용을 셈한다
        trade_values = np.abs(trades) * portfolio.total_value
        costs = self._compute_costs(trade_values)

        return {
            'fill_prices': fill_prices,
            'costs': costs,
            'trades': trades,
        }

    def _compute_fill_prices(self, prices, trades, portfolio_value):
        # 미끄러짐
        slippage = np.random.normal(0, self.slippage_std, len(prices))

        # 저자 흔듦(제곱근 모형)
        if self.volume_data is not None:
            participation = np.abs(trades) * portfolio_value / (prices * self.volume_data)
            impact = self.impact_coeff * np.sign(trades) * np.sqrt(participation)
        else:
            impact = 0

        fill_prices = prices * (1 + np.sign(trades) * slippage + impact)
        return fill_prices

    def _compute_costs(self, trade_values):
        return self.cost_rate * trade_values.sum()
```

---

## 연습문제

<div class="drillbox" markdown>

**연습문제 1.** <span class="diff hard" title="어려움"></span>
이 절에서 밝힌 금융 문제를 위해 Gymnasium과 어울리는 둘레를 설계하여라. 봄 공간, 액션 공간, 리워드 함수, 에피소드 끝내기 조건을 밝혀라.

</div>

??? success "연습문제 1 풀이"
    봄 공간: 최근 돌아옴, 지금 자리, 자본 값, 걸맞은 저자 피처(보기로 흔들림, 거래량)을 담은 벡터. 액션 공간: 문제에 달렸다(사기/쥐기/팔기라면 따로 떨어진 것, 자리 크기 잡기라면 이어진 것). 리워드: 걸음마다 위험을 맞춘 돌아옴(보기로 로그 돌아옴에서 위험 벌을 뺀 것). 에피소드는 붙박인 어텐션(보기로 거래 한 해)이 지나거나 자본 값이 임계 아래로 떨어지면(증거금 부름) 끝난다. 둘레는 거래 비용, 미끄러짐, 저자 섭동을 참에 가깝게 다루어야 한다. $\square$

---

<div class="drillbox" markdown>

**연습문제 2.** <span class="diff med" title="중간"></span>
이 금융 강화학습 문제에서 리워드 다듬기의 트레이드오프를 살펴라. 후보 리워드 함수를 적어도 셋 견주고, 저마다 가장 좋은 폴리시의 어떤 성질을 지키는지 따져라.

</div>

??? success "연습문제 2 풀이"
    후보 리워드: (1) 날 손익 -- 쉽지만 분산이 크고 늦게 온다. (2) 샤프 기초 미분 리워드 $D_t = \frac{\Delta A_t B_{t-1} - \frac{1}{2}\Delta B_t A_{t-1}}{(B_{t-1} - A_{t-1}^2)^{3/2}}$ -- 샤프 비를 곧바로 가장 좋게 하지만 얽혔다. (3) 낙폭 벌을 곁들인 로그 돌아옴 $r_t = \log(V_t/V_{t-1}) - \lambda \max(0, DD_t - \tau)$ -- 돌아옴과 위험 다스리기의 저울을 맞춘다. 퍼텐셜에 기초를 둔 다듬기 정리는 $\gamma\Phi(s') - \Phi(s)$을 더해도 가장 좋은 폴리시가 지켜짐을 보장한다. 샤프 기초 리워드는 가장 좋게 하는 목표 자체를 바꾸므로(가장 좋은 폴리시가 달라질 수 있다) 그렇지 않고, 날 손익은 폴리시를 지키지만 더디게 배운다. $\square$

## 정리하며

실시간 거래로 옮겨 갈 수 있는 에이전트를 익히려면 참에 가까운 시장 시뮬레이션이 꼭 있어야 한다. 중요한 것은 알맞은 거래 비용 모델, 저자 섭동 어림(그 가운데에서도 제곱근 모델), 확률 미끄러짐, 학습의 굳셈을 위한 가짜 데이터 지어내기다. 흉내내기 파라메터를 마당 마구잡이로 바꾸면 참 세상의 들쭉날쭉함에 굳센 폴리시를 얻는다.

**참고 문헌**

- Almgren, R., & Chriss, N. (2001). Optimal Execution of Portfolio Transactions. Journal of Risk
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). Algorithmic and High-Frequency Trading
