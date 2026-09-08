# 하이퍼로그로그

큰 데이터 흐름에서 서로 다른 원소의 수를 정확히 세려면 그 수 자체에 비례하는 공간이 필요하다. 원소가 수십억 개인 흐름에서는 현실적이지 않다. **하이퍼로그로그** 알고리즘(Flajolet 등, 2007)은 레지스터당 $O(\log \log n + \log n)$비트만 써서 서로 다른 원소의 수를 추정한다. 실제로 약 1.5KB의 메모리로 $10^9$까지의 개수를 표준오차 약 2%로 추정할 수 있다.

---

## 1. 직관

이 알고리즘은 확률적인 관찰에 기댄다. 원소를 고르게 무작위로 해시할 때 해시값이 앞에 0을 $k$개 달고 시작할 확률은 $2^{-k}$이다. 앞에 0이 $k$개인 해시를 보았다면 서로 다른 원소가 대략 $2^k$개 해시되었을 가능성이 크다. 해시된 모든 원소에 걸쳐 앞선 0의 최대 개수를 기록하면 개수의 대략적인 추정을 얻는다.

그러나 그런 추정 하나는 분산이 크다. 하이퍼로그로그는 각 해시의 처음 $p = \log_2 m$비트로 흐름을 부분 흐름 $m$개로 나누고, 부분 흐름마다 앞선 0의 최대 개수를 따로 기록한 뒤, $m$개의 추정을 조화평균으로 합쳐 분산을 줄인다.

---

## 2. 알고리즘

### 준비

해시 함수 $h : U \to \{0, 1\}^{L}$을 고른다(보통 $L = 64$비트). 2의 거듭제곱인 레지스터의 수 $m$에 대해 $p = \log_2 m$으로 둔다. 레지스터 $M[0], M[1], \ldots, M[m-1]$을 모두 0으로 초기화한다.

### 원소 넣기

각 원소 $x$에 대해 다음과 같이 한다.

1. $h(x)$을 계산한다.
2. $h(x)$의 $p$비트를 레지스터 인덱스 $j$으로 쓴다(아래 구현은 하위 $p$비트를 쓴다).
3. 남은 $L - p$비트를 $w$이라 하자.
4. $w$에서 가장 왼쪽 1비트의 자리를 $\rho(w)$이라 하자(곧 $w > 0$일 때 $\rho(w) = 1 + \lfloor \log_2(1/w) \rfloor$).
5. $M[j] \leftarrow \max(M[j], \rho(w))$으로 갱신한다.

### 개수 추정

날 추정값은 모든 레지스터에 걸친 $2^{M[j]}$의 조화평균을 쓴다.

$$
E = \alpha_m \cdot m^2 \cdot \left( \sum_{j=0}^{m-1} 2^{-M[j]} \right)^{-1}
$$

여기서 $\alpha_m$은 편향 보정 상수이다.

$$
\alpha_m = \left( m \int_0^\infty \left( \log_2 \frac{2 + u}{1 + u} \right)^m du \right)^{-1}
$$

실용적인 값은 다음과 같다. $\alpha_{16} = 0.673$, $\alpha_{32} = 0.697$, $\alpha_{64} = 0.709$이고 $m \geq 128$이면 $\alpha_m = 0.7213 / (1 + 1.079/m)$이다.

---

## 3. 오차 분석

**정리.** 레지스터가 $m$개인 하이퍼로그로그 추정의 표준오차는 다음과 같다.

$$
\frac{\sigma}{\hat{n}} \approx \frac{1.04}{\sqrt{m}}
$$

즉 레지스터가 $m = 2^{10} = 1024$개이면(각 5비트, 모두 약 640바이트) 표준오차가 약 3.25%이다. $m = 2^{14} = 16384$개이면(약 12KB) 표준오차가 약 0.81%로 떨어진다.

!!! tip "실제 정확도"
    Redis가 쓰는 표준 하이퍼로그로그 구현(`PFADD`, `PFCOUNT`)은 각 6비트인 레지스터 $m = 16384$개를 써서 모두 12KB를 쓴다. 0부터 $2^{64}$까지의 개수에 대해 표준오차가 1% 아래이다.

---

## 4. 작은 범위와 큰 범위의 보정

날 조화평균 추정은 중간 범위에서는 잘 통하지만 양 끝에서는 보정이 필요하다.

**작은 범위의 보정.** $E \leq \frac{5}{2} m$이고 아직 0인 레지스터가 있으면 선형 계수 추정을 대신 쓴다.

$$
E^* = m \ln \frac{m}{V}
$$

여기서 $V$은 0인 레지스터의 수이다. 이 보정은 아직 건드리지 않은 레지스터가 많을 때 생기는 치우침을 다룬다.

**큰 범위의 보정.** 해시의 길이를 $L$이라 할 때 $E > \frac{1}{30} \cdot 2^{L}$이면 해시 충돌이 무시할 수 없게 된다. 다음 보정을 적용한다.

$$
E^* = -2^L \ln\left(1 - \frac{E}{2^L}\right)
$$

---

## 5. 합칠 수 있음

하이퍼로그로그의 중요한 성질은 두 스케치를 레지스터마다 최댓값을 취해 합칠 수 있다는 것이다.

$$
M_{\text{merged}}[j] = \max(M_A[j], M_B[j]) \quad \text{for } j = 0, 1, \ldots, m-1
$$

덕분에 하이퍼로그로그는 분산 계수에 알맞다. 마디마다 지역 스케치를 계산하고 중앙 집계기가 $O(m)$ 시간에 이를 합친다.

---

## 6. 구현

```python
"""하이퍼로그로그 기수 추정."""

import hashlib
import math

# === 하이퍼로그로그 ===

class HyperLogLog:
    """O(m) 공간으로 서로 다른 원소의 수를 어림한다."""

    def __init__(self, p: int = 10):
        self.p = p
        self.m = 1 << p  # 레지스터의 수
        self.registers = [0] * self.m
        # 치우침 보정 상수
        if self.m >= 128:
            self.alpha = 0.7213 / (1.0 + 1.079 / self.m)
        elif self.m == 64:
            self.alpha = 0.709
        elif self.m == 32:
            self.alpha = 0.697
        else:
            self.alpha = 0.673

    def _hash(self, item: str) -> int:
        """원소를 64비트 정수로 해싱한다."""
        digest = hashlib.sha256(item.encode()).hexdigest()
        return int(digest[:16], 16)  # 64비트 해시

    @staticmethod
    def _rho(w: int, max_bits: int) -> int:
        """가장 왼쪽 1비트의 자리 (1부터 셈)."""
        if w == 0:
            return max_bits + 1
        pos = 1
        while (w >> (max_bits - pos)) & 1 == 0:
            pos += 1
        return pos

    def add(self, item: str) -> None:
        """스케치에 원소를 넣는다."""
        h = self._hash(item)
        j = h & (self.m - 1)  # 가장 낮은 p비트를 레지스터 색인으로
        w = h >> self.p  # 남은 비트
        self.registers[j] = max(self.registers[j], self._rho(w, 64 - self.p))

    def count(self) -> int:
        """서로 다른 원소의 수를 어림한다."""
        # 조화 평균으로 얻은 날 추정값
        indicator = sum(2.0 ** (-r) for r in self.registers)
        estimate = self.alpha * self.m * self.m / indicator

        # 작은 범위 보정
        if estimate <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros > 0:
                estimate = self.m * math.log(self.m / zeros)

        # 큰 범위 보정 (64비트 해시용)
        two_to_64 = 2.0 ** 64
        if estimate > two_to_64 / 30.0:
            estimate = -two_to_64 * math.log(1.0 - estimate / two_to_64)

        return int(estimate)

    def merge(self, other: "HyperLogLog") -> "HyperLogLog":
        """하이퍼로그로그 스케치 두 개를 합친다."""
        assert self.p == other.p, "Cannot merge sketches with different p"
        result = HyperLogLog(self.p)
        result.registers = [
            max(a, b) for a, b in zip(self.registers, other.registers)
        ]
        return result

# === 시연 ===

if __name__ == "__main__":
    hll = HyperLogLog(p=10)  # 레지스터 1024개

    # 개수를 아는 원소들 넣기
    n = 100_000
    for i in range(n):
        hll.add(f"element-{i}")

    estimate = hll.count()
    error = abs(estimate - n) / n * 100
    print(f"True cardinality: {n}")
    print(f"Estimated:        {estimate}")
    print(f"Relative error:   {error:.2f}%")
    print(f"Memory:           {len(hll.registers) * 6} bits "
          f"({len(hll.registers) * 6 / 8:.0f} bytes)")
```

---

## 7. 복잡도 요약

| 연산 | 시간 | 공간 |
|---|---|---|
| 원소 넣기 | $O(1)$ | -- |
| 개수 추정 | $O(m)$ | -- |
| 스케치 둘 합치기 | $O(m)$ | -- |
| 전체 공간 | -- | 레지스터 $O(m)$개 |

각 6비트인 레지스터가 $m = 2^p$개이면 전체 공간은 $6 \cdot 2^p$비트이다.

---

## 연습문제

**연습문제 1.**
하이퍼로그로그에 대해, 적재율이 $\alpha = 0.75$일 때 삽입과 조회의 기대 시간과 최악의 경우 시간을 계산하라.

??? success "연습문제 1 풀이"
    기대 시간은 충돌 해결 전략에 달렸으며 균등 해싱을 가정한다. 체이닝에서는 기대 시간이 $O(1 + \alpha) = O(1.75)$이다. 개방 주소법에서는 탐색에 실패할 때 기대 탐사 횟수가 $\approx 1/(1-\alpha) = 4$이다. 최악의 경우는 모든 키가 같은 칸으로 해시될 때의 $O(n)$이다.

---

**연습문제 2.**
하이퍼로그로그을(를) 써서 키 10, 22, 31, 4, 15, 28, 17을 크기가 7인 해시 테이블에 넣어라. 최종 테이블의 상태를 보여라.

??? success "연습문제 2 풀이"
    해시 함수 $h(k) = k \bmod 7$을 적용하고 이 쪽의 방법으로 충돌을 처리한다. 키마다 해시를 계산하고 충돌을 해결한 뒤 키를 놓는다. 최종 테이블의 내용을 보인다.

---

**연습문제 3.**
하이퍼로그로그은(는) 딥러닝의 임베딩 테이블에서 어떻게 쓰이는가? 토큰 $V = 50{,}000$개의 어휘를 $m = 30{,}000$개의 버킷에 대응시킬 때 충돌의 양상을 분석하라.

??? success "연습문제 3 풀이"
    $V/m \approx 1.67$이므로 비둘기집 원리에 의해 충돌이 반드시 생긴다. 버킷마다 평균 1.67개의 토큰이 같은 임베딩을 나누어 쓴다. 충돌률과 그것이 모델의 품질에 미치는 영향은 해시 함수의 품질과 임베딩의 차원에 달렸다(차원이 높을수록 충돌을 더 잘 견딘다).

---

**연습문제 4.**
$\alpha > 0.75$일 때 해시 테이블의 크기를 다시 잡으면 삽입의 상각 비용이 $O(1)$으로 유지됨을 증명하라.

??? success "연습문제 4 풀이"
    크기를 다시 잡는 사이(용량 $m$에서 $2m$까지)에 삽입이 $m/4$번 일어난다(적재율이 $0.375$에서 $0.75$로 간다). 크기 조정에는 $O(m)$이 든다. 삽입 하나당 상각된 크기 조정 비용은 $O(m)/(m/4) = O(4) = O(1)$이다. 여기에 (균등 해싱 아래) 삽입마다의 기대 비용 $O(1)$을 더하면 전체 상각 비용은 $O(1)$이다. $\square$

## 정리하며

이 마당은 직관、알고리즘、오차 분석、작은 범위와 큰 범위의 보정을 차례로 짚었다.

**참고 문헌**

- Flajolet, P., Fusy, E., Gandouet, O., & Meunier, F. (2007). HyperLogLog: The analysis of a near-optimal cardinality estimation algorithm. *Conference on Analysis of Algorithms (AofA)*.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 11. MIT Press.
