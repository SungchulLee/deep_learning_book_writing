# 카운트-민 스케치

대용량 데이터 흐름에서 모든 원소의 정확한 빈도를 기록하려면 서로 다른 원소의 수에 비례하는 공간이 필요한데, 이는 감당하기 어려울 만큼 클 수 있다. **카운트-민 스케치**(Cormode와 Muthukrishnan, 2005)는 정확한 개수를 근사값으로 바꾸어, 본 원소의 총수를 $n$이라 할 때 모든 빈도 추정값이 적어도 $1 - \delta$의 확률로 참값을 많아야 $\varepsilon n$만큼만 넘어서도록 보장하면서 $O(\frac{1}{\varepsilon} \log \frac{1}{\delta})$ 공간만 쓴다.

## 자료 구조

카운트-민 스케치는 0으로 초기화된 이차원 세개 배열 $\text{CM}[1 \ldots d][1 \ldots w]$과, 전체 집합 $U$을 $\{1, 2, \ldots, w\}$으로 보내는 쌍별 독립인 해시 함수 $h_1, h_2, \ldots, h_d$으로 이루어진다.

차원은 바라는 정확도와 신뢰도에 따라 고른다.

$$
w = \left\lceil \frac{e}{\varepsilon} \right\rceil, \qquad d = \left\lceil \ln \frac{1}{\delta} \right\rceil
$$

여기서 $e = 2.718\ldots$은 자연상수이고, $\varepsilon$은 근사 오차를, $\delta$은 실패 확률을 조절한다.

## 연산

### 갱신

원소 $x$이 흐름에 들어오면 행마다 해당 해시 함수가 정한 자리의 세개를 1 늘린다.

$$
\text{CM}[i][h_i(x)] \leftarrow \text{CM}[i][h_i(x)] + 1 \quad \text{for } i = 1, 2, \ldots, d
$$

갱신마다 $O(d)$ 시간이 든다.

### 질의

원소 $x$의 빈도 $\hat{f}_x$을 추정하려면 $d$개 행의 최솟값을 취한다.

$$
\hat{f}_x = \min_{1 \leq i \leq d} \text{CM}[i][h_i(x)]
$$

최솟값을 취하는 것이 핵심이다. 개별 세개 $\text{CM}[i][h_i(x)]$은 다른 원소와의 충돌로 부풀려질 수 있지만, 독립인 해시 함수에 걸쳐 최솟값을 취하면 이 부풀림이 줄어든다.

### 추정값의 성질

카운트-민 스케치는 **한쪽으로만 치우친 추정량**이다. 참 빈도를 결코 낮추어 잡지 않는다.

**정리.** 참 빈도가 $f_x$인 어떤 원소 $x$에 대해서도 다음이 성립한다.

$$
f_x \leq \hat{f}_x
$$

그리고 다음이 성립한다.

$$
\Pr[\hat{f}_x \leq f_x + \varepsilon n] \geq 1 - \delta
$$

여기서 $n = \sum_x f_x$은 모든 원소의 총 개수이다.

*증명 개요.* 행 $i$마다 원소 $x$은 세개 $\text{CM}[i][h_i(x)]$에 정확히 $f_x$을 보태므로 $\text{CM}[i][h_i(x)] \geq f_x$이다. 넘치는 몫은 행 $i$에서 $x$과 충돌하는(곧 $h_i(y) = h_i(x)$인) 다른 원소 $y \neq x$에서 온다. $h_i$의 쌍별 독립성과 마르코프 부등식에 의해 다음이 성립한다.

$$
\Pr[\text{CM}[i][h_i(x)] - f_x > \varepsilon n] \leq \frac{\mathbb{E}[\text{CM}[i][h_i(x)] - f_x]}{\varepsilon n} = \frac{(n - f_x)/w}{\varepsilon n} \leq \frac{1}{w \varepsilon} \leq \frac{1}{e}
$$

$d$개 행이 독립인 해시 함수를 쓰므로 *모든* 행에서 넘치는 몫이 $\varepsilon n$보다 클 확률은 많아야 $(1/e)^d \leq \delta$이다. $\square$

## 점 질의와 구간 질의

기본 카운트-민 스케치는 **점 질의**(원소 하나의 빈도)에 답한다. 전체 집합의 크기를 $U$이라 할 때, 서로 다른 결의 스케치를 $\log U$개 두면(이진 구간 분해) **구간 질의**에도 답하도록 넓힐 수 있다.

## 구현

```python
"""데이터 흐름에서 빈도를 어림하는 카운트-민 스케치."""

import hashlib
import math


# === 카운트-민 스케치 ===

class CountMinSketch:
    """선형보다 작은 공간을 쓰는 근사 빈도 계수기."""

    def __init__(self, epsilon: float = 0.01, delta: float = 0.01):
        self.w = math.ceil(math.e / epsilon)
        self.d = math.ceil(math.log(1.0 / delta))
        self.table = [[0] * self.w for _ in range(self.d)]
        self.n = 0  # 전체 개수

    def _hash(self, x: str, i: int) -> int:
        """i번째 행을 위해 원소 x를 해싱한다."""
        h = hashlib.md5(f"{i}:{x}".encode()).hexdigest()
        return int(h, 16) % self.w

    def update(self, x: str, count: int = 1) -> None:
        """원소 x가 count번 나타난 것을 기록한다."""
        self.n += count
        for i in range(self.d):
            self.table[i][self._hash(x, i)] += count

    def query(self, x: str) -> int:
        """원소 x의 빈도를 어림한다."""
        return min(self.table[i][self._hash(x, i)] for i in range(self.d))


# === 시연 ===

if __name__ == "__main__":
    cms = CountMinSketch(epsilon=0.001, delta=0.01)

    # 빈도를 아는 흐름 흉내 내기
    frequencies = {"apple": 500, "banana": 300, "cherry": 100, "date": 50}
    for item, freq in frequencies.items():
        for _ in range(freq):
            cms.update(item)

    print(f"Sketch dimensions: {cms.d} rows x {cms.w} columns")
    print(f"Total elements: {cms.n}")
    print()
    for item, true_freq in frequencies.items():
        est = cms.query(item)
        print(f"{item}: true={true_freq}, estimate={est}, error={est - true_freq}")

    # 없는 원소 물어보기
    est_absent = cms.query("elderberry")
    print(f"elderberry (absent): estimate={est_absent}")
```

## 다른 스케치와의 비교

| 자료 구조 | 질의의 종류 | 공간 | 오차의 종류 |
|---|---|---|---|
| 카운트-민 스케치 | 빈도 | $O(\frac{1}{\varepsilon} \log \frac{1}{\delta})$ | 한쪽 (높게 잡음) |
| 카운트 스케치 | 빈도 | $O(\frac{1}{\varepsilon^2} \log \frac{1}{\delta})$ | 양쪽 (불편) |
| [블룸 필터](bloom.md) | 소속 | $O(n \log \frac{1}{\delta})$ | 한쪽 (거짓 양성) |
| [하이퍼로그로그](hyperloglog.md) | 원소의 개수 | $O(\frac{1}{\varepsilon^2})$ | 양쪽 |

한쪽으로만 치우친 오차를 받아들일 수 있을 때(예: 이상 탐지나 대량 사용자 식별에서 빈도를 낮추어 잡지 않는 것이 중요할 때) 카운트-민 스케치를 선호한다.

## 응용

- **네트워크 트래픽 감시:** 흐름마다의 패킷 수를 추정하여 대역폭을 지나치게 쓰는 흐름을 찾아낸다.
- **자연어 처리:** 어휘 전체를 저장하지 않고 큰 말뭉치의 낱말 빈도를 근사한다.
- **데이터베이스 질의 최적화:** 흐르는 데이터에서 조인의 크기와 선택도를 추정한다.
- **이상 탐지:** 추정 빈도가 문턱값을 넘는 원소를 표시한다.

## 참고 문헌

- Cormode, G., & Muthukrishnan, S. (2005). An improved data stream summary: The Count-Min sketch and its applications. *Journal of Algorithms*, 55(1), 58--75.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 11. MIT Press.


## 연습문제

**연습문제 1.**
카운트-민 스케치에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
카운트-민 스케치을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
카운트-민 스케치은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$