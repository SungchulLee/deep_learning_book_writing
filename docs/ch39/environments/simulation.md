# 35.1.5 시장 시뮬레이션

---

## 1. 학습 목표

- 강화학습 학습을 위한 현실적인 시장 시뮬레이터를 구축한다
- 거래 비용, 슬리피지, 시장 충격을 모델링한다
- 현실적인 체결 가정으로 주문 실행을 만든다
- 둘레를 확장하려 합성 시장 데이터를 생성한다

---

## 2. 소개

시장 시뮬레이터는 에이전트의 액션이 실제 포트폴리오 변화로 어떻게 옮겨지는지를 정한다. 모든 주문을 종가에 체결해 주는 단순한 시뮬레이터는 비현실적인 기대를 심는다. 현실적인 시뮬레이션은 거래 비용, 슬리피지, 부분 체결, 시장 충격을 계산에 넣어야 한다. 가용 유동성에 비해 크게 거래하는 전략이라면 더욱 그렇다.

---

## 3. 주문 실행 모델

### 1. 단순 체결 모델

모든 주문이 다음 봉의 시가(또는 종가)에 체결된다.

$$\text{채워진\_값}_i = p_{t+1}^{\text{시작}} \cdot (1 + \text{미끄러짐}_i)$$

여기서 슬리피지는 무작위 값이거나 고정된 스프레드 비율이다.

### 2. VWAP 실행 모델

한 봉에 걸쳐 실행하는 주문은 거래량 가중 평균 가격으로 체결 가격을 추정한다.

$$\text{채워진\_값}_i = \text{VWAP}_{[t, t+\Delta t]} + \text{저자\_흔듦}_i$$

### 3. 지정가 주문 장부 시뮬레이션

고빈도 전략에서는 지정가 주문 장부를 시뮬레이션한다.

- 매수·매도 스프레드의 동작을 모델링한다
- 지정가 주문의 대기열 위치를 다룬다
- 유동성을 먹어 치우며 생기는 시장 충격을 다룬다

---

## 4. 거래 비용 모델

### 비례 비용

$$\text{TC}_t = c \cdot \sum_i |q_{t,i}| \cdot p_{t,i}$$

여기서 $c$은 비용 비율이고(주식에서 흔히 5~20 베이시스 포인트) $q_{t,i}$은 거래 수량이다.

### 스프레드 기반 비용

$$\text{TC}_t = \frac{1}{2} \sum_i |q_{t,i}| \cdot \text{사이}_{t,i}$$

반 스프레드 모델이다. 매도 호가에 사고 매수 호가에 판다.

### 구간별 수수료 모델

```python
def compute_commission(self, trade_value):
    if trade_value < 10_000:
        return max(1.0, trade_value * 0.005)  # 가장 적게 $1, 50 베이시스 포인트
    elif trade_value < 100_000:
        return trade_value * 0.001              # 10 베이시스 포인트
    else:
        return trade_value * 0.0005             # 5 베이시스 포인트
```

---

## 5. 시장 충격 모델

### 선형 충격

$$\Delta p = \lambda \cdot \frac{q}{V}$$

여기서 $q$은 주문 수량, $V$은 하루 평균 거래량, $\lambda$은 충격 계수다.

### 제곱근 충격(올름그렌-크리스)

$$\Delta p = \sigma \cdot \eta \cdot \text{sgn}(q) \cdot \sqrt{\frac{|q|}{V}}$$

여기서 $\sigma$은 변동성이고 $\eta$은 데이터로 보정한 파라미터다. 이 모델은 거래 크기와 충격 사이의 오목한 관계를 더 잘 담는다.

### 일시적 충격과 영구적 충격

- **일시적 충격**: 거래 뒤 가격이 되돌아온다. 비용은 거래한 당사자만 치른다
- **영구적 충격**: 가격이 영구히 옮겨 간다. 이후의 모든 가격 형성에 영향을 미친다

$$p_{t+1} = p_t + \underbrace{\gamma \cdot \frac{q}{V}}_{\text{오래감}} + \underbrace{\eta \cdot \text{sgn}(q) \cdot \sqrt{\frac{|q|}{V}}}_{\text{잠깐 감}} + \epsilon_t$$

---

## 6. 슬리피지 모델링

슬리피지는 기대한 가격과 실제 체결 가격의 차이다.

### 고정된 슬리피지

$$\text{미끄러짐} = \text{sign}(q) \cdot s$$

여기서 $s$은 고정된 슬리피지 양이다(예를 들어 1 베이시스 포인트).

### 확률적 슬리피지

$$\text{미끄러짐} \sim \mathcal{N}(\mu_s, \sigma_s^2)$$

$\mu_s$과 $\sigma_s$은 과거 실행 데이터로 추정한다.

### 거래량 의존 슬리피지

$$\text{미끄러짐} = s_0 + s_1 \cdot \frac{|q|}{V_t}$$

거래량에 비해 주문이 클수록 슬리피지가 커진다.

---

## 7. 합성 데이터 생성

과거 데이터가 적으면 과적합하기 쉽다. 합성 데이터로 늘리면 도움이 된다.

### 부트스트랩 재샘플링

과거 수익률을 복원 추출로 다시 뽑아 새 가격 경로를 만든다.

$$R_t^{\text{가짜}} = R_{\pi(t)}^{\text{지난}}$$

여기서 $\pi$은 무작위 순열이다(블록 부트스트랩은 자기상관을 지킨다).

### 생성 모델

학습된 생성 모델로 현실적인 합성 데이터를 만든다.

- **GARCH 모델**: 변동성 군집을 담는다
- **레짐 전환 모델**: 시장 레짐의 동작을 담는다
- **GAN/VAE**: 복잡한 분포 특성을 배운다

### 잡음 넣기

과거 데이터의 변동성에 맞춘 잡음을 더한다.

$$p_t^{\text{늘림}} = p_t \cdot e^{\epsilon_t}, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_{\text{늘림}}^2)$$

---

## 8. 도메인 무작위화

강건성을 높이려 학습하는 동안 둘레 파라미터를 바꾼다.

```python
def randomize_params(self):
    """에피소드마다 둘레 파라메터를 아무렇게나 바꾼다."""
    self.transaction_cost = np.random.uniform(0.0005, 0.002)
    self.slippage_mean = np.random.uniform(0.0, 0.001)
    self.market_impact_coeff = np.random.uniform(0.05, 0.2)
    self.initial_capital *= np.random.uniform(0.8, 1.2)
```

---

## 9. 여러 자산의 상관

다자산 둘레에서는 자산 사이의 상관을 제대로 모델링해야 한다.

- **과거 상관**: 롤링 상관 행렬을 쓴다
- **코퓰러 모델**: 비선형 의존성을 담는다
- **인자 모델**: 수익률을 공통 인자와 고유 성분으로 나눈다

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

## 정리하며

실시간 거래로 옮겨 갈 수 있는 에이전트를 학습시키려면 현실적인 시장 시뮬레이션이 꼭 있어야 한다. 중요한 것은 알맞은 거래 비용 모델, 시장 충격 추정(그 가운데에서도 제곱근 모델), 확률적 슬리피지, 학습의 강건성을 위한 합성 데이터 생성이다. 시뮬레이션 파라미터를 도메인 무작위화로 바꾸면 실제 시장의 변화에 강건한 폴리시를 얻는다.

**참고 문헌**

- Almgren, R., & Chriss, N. (2001). Optimal Execution of Portfolio Transactions. Journal of Risk
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). Algorithmic and High-Frequency Trading
