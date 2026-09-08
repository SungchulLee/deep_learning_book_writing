# 증분 알고리즘

증분 알고리즘(incremental algorithm)은 한 번에 원소 하나씩 해를 만들어 가며 각 단계마다 정확성을 유지한다. 이 접근법은 딥러닝의 학습 루프에 그대로 대응된다. 각 경사 단계가 모델 매개변수를 증분적으로 개선한다.

---

## 1. 정의

증분 알고리즘은 원소를 차례로 처리하면서 새 원소가 올 때마다 현재 해를 갱신하여 해를 구성한다.

$$
\text{Solution}_i = \text{Update}(\text{Solution}_{i-1}, \text{element}_i)
$$

불변식은 $\text{Solution}_i$가 처음 $i$개의 원소에 대해 정확하다는 것이다.

---

## 2. 설명

증분 접근법은 딥러닝 곳곳에 나타난다.

- **확률적 경사 하강법**: 각 미니배치 갱신이 매개변수를 증분적으로 개선한다. 불변식은 각 단계 후에 손실이 (기댓값의 의미에서) 대체로 감소한다는 것이다.
- **온라인 학습**: 모델이 도착하는 데이터 점마다 갱신되며 과거 데이터를 다시 보지 않는다. 본질적으로 증분적이다.
- **누적 통계**: 배치 정규화는 학습 배치들에 걸쳐 이동 평균과 분산을 증분적으로 추적한다.

핵심 장점은 단순함이다. 갱신 규칙을 유도하고 구현하기가 대체로 쉽다. 단점은 탐욕적인 증분 선택이 전역 최적해를 주지 못할 수 있다는 점이다(SGD가 전역이 아닌 국소 최솟값을 찾는 것과 비슷하다).

---

## 3. 예제

```python
import torch

# 증분적 평균과 분산(Welford 알고리즘)
# BatchNorm의 누적 통계에 사용된다
def incremental_stats(data: torch.Tensor):
    """평균과 분산을 증분적으로 계산한다."""
    mean = torch.tensor(0.0)
    m2 = torch.tensor(0.0)
    for i, x in enumerate(data, 1):
        delta = x - mean
        mean += delta / i
        delta2 = x - mean
        m2 += delta * delta2
    variance = m2 / len(data)
    return mean, variance

data = torch.randn(1000)
inc_mean, inc_var = incremental_stats(data)
print(f"Incremental mean: {inc_mean.item():.6f}")
print(f"Incremental var:  {inc_var.item():.6f}")
print(f"Direct mean:      {data.mean().item():.6f}")
print(f"Direct var:       {data.var(correction=0).item():.6f}")

# 증분적 최적화로서의 SGD
torch.manual_seed(42)
w = torch.randn(1, requires_grad=True)
x = torch.randn(100)
y = 3.0 * x + torch.randn(100) * 0.1

for i in range(50):
    idx = i % len(x)  # 한 번에 원소 하나씩
    pred = w * x[idx]
    loss = (pred - y[idx]) ** 2
    loss.backward()
    with torch.no_grad():
        w -= 0.01 * w.grad
        w.grad.zero_()
print(f"Learned w: {w.item():.4f} (true: 3.0)")
```

---

## 연습문제

**연습문제 1.**
분산에 대한 Welford의 온라인 갱신식을 유도하라. 현재 평균 $\bar{x}_n$, 개수 $n$, 편차 제곱합 $M_n$이 주어졌을 때 새 표본 $x_{n+1}$이 도착하면 어떻게 갱신되는지 표현하라.

??? success "연습문제 1 풀이"
    $x_{n+1}$이 도착하면 $\delta = x_{n+1} - \bar{x}_n$, $\bar{x}_{n+1} = \bar{x}_n + \delta / (n+1)$, $\delta_2 = x_{n+1} - \bar{x}_{n+1}$, $M_{n+1} = M_n + \delta \cdot \delta_2$이다. 분산은 $\sigma^2 = M_n / n$(모분산) 또는 $s^2 = M_n / (n-1)$(표본분산)이다. 이 방식은 $\bar{x}$가 클 때 심각한 상쇄 오차를 겪는 $\sum x_i^2 - n\bar{x}^2$의 계산을 피하므로 수치적으로 안정하다.

---

**연습문제 2.**
배치 정규화는 지수 이동 평균을 사용해 누적 추정값 $\hat{\mu}$와 $\hat{\sigma}^2$를 유지한다: $\hat{\mu} \leftarrow (1-\alpha)\hat{\mu} + \alpha \mu_{\text{batch}}$. 이것이 특정한 갱신 규칙을 가진 증분 알고리즘임을 보이고, 모멘텀 매개변수 $\alpha$의 역할을 설명하라.

??? success "연습문제 2 풀이"
    이는 $\text{Solution}_i = \hat{\mu}_i$이고 $\text{element}_i = \mu_{\text{batch},i}$인 증분 형태 $\text{Solution}_i = \text{Update}(\text{Solution}_{i-1}, \text{element}_i)$와 일치한다. 갱신은 이전 추정값과 새 배치 통계량의 가중 결합이다. 모멘텀 $\alpha$(보통 0.1)는 기억을 조절한다. $\alpha$가 작으면 과거에 더 큰 가중치를 주고(안정적이지만 적응이 느림), 크면 현재 배치에 더 큰 가중치를 준다(반응이 빠르지만 잡음이 많음). 불변식은 $\hat{\mu}_i$가 지수적으로 감쇠하는 가중치로 계산한 모든 과거 배치 평균의 가중 평균이라는 것이다.

---

**연습문제 3.**
선형 모델 $f(x) = wx$에 대해 상수 학습률 $\eta$와 MSE 손실을 쓰는 SGD가 다음 불변식을 만족함을 증명하라: $|w_t - w^*| < \epsilon$이고 경사가 $G$로 유계이면 $|w_{t+1} - w^*| < \epsilon + \eta G$이다.

??? success "연습문제 3 풀이"
    갱신은 $w_{t+1} = w_t - \eta g_t$이고 $|g_t| \leq G$이다. 그러면 $|w_{t+1} - w^*| = |w_t - \eta g_t - w^*| \leq |w_t - w^*| + \eta|g_t| < \epsilon + \eta G$이다. 이는 경사가 유계일 때 각 단계가 매개변수를 최대 $\eta G$만큼 움직인다는 것을 보여준다. 수렴을 위해서는 기대 경사가 $w^*$ 쪽을 향하도록 $\eta$가 충분히 작아야 하며, 그러면 기대 변위가 감소한다. $\square$

---

**연습문제 4.**
온라인 학습은 데이터셋을 저장하지 않고 한 번에 표본 하나씩 처리한다. 매개변수가 $P$개인 모델과 각각 특징이 $d$개인 표본 $n$개의 데이터셋에 대해, 온라인 학습과 배치 경사 하강법의 메모리 복잡도를 비교하라.

??? success "연습문제 4 풀이"
    **온라인 학습**: $O(P + d)$ — 모델 매개변수와 현재 표본만 저장한다. **배치 경사 하강법**: $O(P + nd)$ — 모델 매개변수에 더해 데이터셋 전체를 저장한다. 큰 데이터셋($n = 10^9$, $d = 100$)에서 온라인 학습은 $O(P + 100)$을 쓰지만 배치 방식은 $O(P + 10^{11})$의 메모리를 쓴다. 그래서 온라인 학습은 스트리밍 데이터나 메모리에 들어가지 않는 데이터셋에 필수적이다. 대가는 더 잡음이 많은 경사 추정(표본 하나 대 전체 데이터셋)이다.

---

**연습문제 5.**
삽입 정렬은 증분적 정렬 알고리즘이다. 그 불변식을 기술하고 각 단계에서 불변식이 유지됨을 증명하라. 빔 탐색에서 우선순위 큐가 상위 $k$개 원소를 증분적으로 유지하는 방식과 유비를 그려보라.

??? success "연습문제 5 풀이"
    **불변식**: 처음 $i$개의 원소를 처리한 뒤 부분배열 $A[0:i]$는 정렬되어 있다. **초기화**: 원소 하나는 자명하게 정렬되어 있다. **유지**: 원소 $A[i]$를 삽입하려면 정렬된 부분 $A[0:i]$의 원소들을 올바른 위치를 찾을 때까지 오른쪽으로 밀어낸다. 그러면 부분배열 $A[0:i+1]$이 정렬된다. **종료**: $n$개의 원소를 모두 처리한 뒤 $A[0:n]$이 정렬된다. 빔 탐색과의 유비: 우선순위 큐가 (점수로 정렬된) 상위 $k$개 후보를 유지한다. 새 후보가 도착하면 그 점수가 큐의 최솟값을 넘을 때 삽입되며, 이로써 큐가 지금까지 본 최상의 $k$개 후보를 담는다는 불변식이 유지된다.

## 정리하며

이 마당은 정의、설명、예제을 차례로 짚었다.
