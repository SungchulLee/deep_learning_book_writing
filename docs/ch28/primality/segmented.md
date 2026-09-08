# 토막 낸 체

고전 에라토스테네스의 체는 $n$까지의 소수를 모두 찾는 데 기억이 $O(n)$ 든다. $n$이 크면(보기로 $10^{12}$) 이 배열이 기억에 들어가지 않는다. 토막 낸 체는 범위를 붙박이 크기의 덩이로 나누어 다루어 시간 복잡도는 그대로 두고 기억을 $O(\sqrt{n})$으로 줄인다.

---

## 1. 미리 알아야 할 것

핵심 살핌은 어떤 합성수 $m \le n$에도 소인수 $p \le \sqrt{n}$이 있다는 것이다. 따라서 어떤 덩이를 체 치든 $\sqrt{n}$까지의 소수만 있으면 된다.

---

## 2. 알고리즘

### 걸음 1 --- 바탕 소수

여느 에라토스테네스의 체로 $\sqrt{n}$까지의 소수를 모두 찾는다.
이는 기억 $O(\sqrt{n})$과 시간
$O(\sqrt{n} \log \log \sqrt{n})$이 든다.

### 걸음 2 --- 덩이 다루기

범위 $[\sqrt{n} + 1, n]$을 크기 $\Delta$인 덩이로 나눈다(보통 $\Delta = \sqrt{n}$이나 $2^{18}$ 같은 캐시에 친한 값). 덩이 $[L, L + \Delta)$마다:

1. 부울 배열 `is_prime[0..\Delta-1]`을 모두 `True`으로 첫자리매김한다.
2. 바탕 소수 $p$마다:
    - $[L, L + \Delta)$에서 $p$의 가장 작은 배수를 찾는다:

    $$
    \text{start} = \left\lceil \frac{L}{p} \right\rceil \cdot p
    $$

    $\text{start} = p$이면 $2p$으로 나아간다(소수 자신은 합성수가 아니다).

    - 덩이 안 $p$의 모든 배수를 합성수로 표시한다.
3. 표시되지 않은 자리를 소수로 모은다.

---

## 3. 구현

```python
"""
토막 낸 에라토스테네스의 체.

시간 : O(n log log n)  — 기본 체와 같다
공간: O(sqrt(n))      — 바탕 소수와 덩이 하나뿐
"""

import math

# === 기본 체(바탕 소수용) ===
def simple_sieve(limit: int) -> list[int]:
    """여느 체로 limit까지의 소수를 모두 돌려준다."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

# === 토막 낸 체 ===
def segmented_sieve(n: int) -> list[int]:
    """토막 낸 체로 n까지의 소수를 모두 돌려준다."""
    if n < 2:
        return []

    limit = int(math.isqrt(n)) + 1
    base_primes = simple_sieve(limit)

    # 바탕 체에서 얻은 소수
    primes = [p for p in base_primes if p <= n]

    # 토막을 다룬다
    delta = max(limit, 1)
    low = limit + 1

    while low <= n:
        high = min(low + delta - 1, n)
        block_size = high - low + 1
        is_prime = [True] * block_size

        for p in base_primes:
            # low 이상인 p의 첫 배수
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start, high + 1, p):
                is_prime[j - low] = False

        for i in range(block_size):
            if is_prime[i]:
                primes.append(low + i)

        low = high + 1

    return primes

# === 보기 ===
if __name__ == "__main__":
    n = 100
    result = segmented_sieve(n)
    print(f"Primes up to {n}: {result}")
    print(f"Count: {len(result)}")

    # 더 큰 보기
    n_large = 10_000
    count = len(segmented_sieve(n_large))
    print(f"Number of primes up to {n_large}: {count}")
```

---

## 4. 복잡도 분석

| 면 | 기본 체 | 토막 낸 체 |
|---|---|---|
| 시간 | $O(n \log \log n)$ | $O(n \log \log n)$ |
| 공간 | $O(n)$ | $O(\sqrt{n})$ |
| 캐시 움직임 | $n$이 크면 나쁨 | 훌륭함(덩이가 L1/L2에 들어감) |

합성수마다 여전히 같은 횟수로 지워지므로 시간 복잡도는 같다.
기억이 나아지는 것은 덩이 크기의 배열 하나를
다시 쓰기 때문이다.

---

## 5. 덩이 크기 고르기

덩이 크기 $\Delta$이 캐시 성능에 영향을 준다:

- **$\Delta = \sqrt{n}$**은 덩이 개수를 가장 작게 하며 이론상
  기본값이다.
- **$\Delta = $ L1 캐시 크기 / 8**(바이트 단위)이 실제로 가장 좋은 성능을 낸다.
  덩이마다 가장 빠른 캐시 켜에 온전히 들어가기 때문이다.

!!! tip "캐시에 친한 체 치기"
    L1 캐시가 32KB인 요즘 중앙 처리 장치에서 $\Delta = 2^{15} = 32{,}768$으로 두면
    실제로 좋은 고르기이다. 이는 체 치는 동안 부울 배열과 되풀이 변수가
    모두 L1 캐시에 머물게 한다.

---

## 6. 범위 체 치기

토막 낸 체는 $2$부터 체 치지 않고도 아무 범위 $[a, b]$의 소수를 찾도록
자연스럽게 넓혀진다:

1. $\sqrt{b}$까지의 바탕 소수를 셈한다.
2. 덩이 하나 $[a, b]$을 체 친다.

이는 $2$부터 온전히 체 칠 필요가 없는 "$10^{12}$과 $10^{12} + 10^6$ 사이의
소수를 세어라" 같은 문제에 쓸모 있다.

---

## 연습문제

**연습문제 1.**
$R$이 너무 커서 여느 체를 쓸 수 없을 때 $[L, R]$의 소수를 찾는 토막 낸 체 알고리즘을 적어라.

??? success "연습문제 1 풀이"
    먼저 여느 에라토스테네스의 체로 $\sqrt{R}$까지의 소수를 체 친다(기억 $O(\sqrt{R})$이 든다). 그다음 $[L, R]$을 크기 $\Delta \leq \sqrt{R}$인 토막으로 나눈다. 토막 $[a, a + \Delta)$마다 크기 $\Delta$인 부울 배열을 만든다. 작은 소수 $p \leq \sqrt{R}$마다 토막 안 $p$의 배수를 ($\lceil a/p \rceil \cdot p$에서 시작해) 표시한다. 표시되지 않은 자리가 소수이다. 기억: $O(\sqrt{R} + \Delta) = O(\sqrt{R})$.

---

**연습문제 2.**
$10^{12}$까지의 소수를 찾을 때 토막 낸 체가 여느 체보다 기억에서 나은 점은 무엇인가?

??? success "연습문제 2 풀이"
    $10^{12}$의 여느 체: 크기 $10^{12}$인 부울 배열이 필요하며 약 1TB로 쓸 수 없다. 토막 낸 체: $\sqrt{10^{12}} = 10^6$까지의 소수(약 78,498개, 약 600KB)와 크기 $10^6$인 토막 하나(약 1MB)를 담는다. 온 기억은 약 2MB로 $10^6$배 줄어든다.

---

**연습문제 3.**
$[10^{15}, 10^{15} + 10^6]$의 소수 개수를 어떻게 효율 좋게 세겠는가?

??? success "연습문제 3 풀이"
    토막 낸 체를 쓴다. (1) 여느 체로 $\sqrt{10^{15} + 10^6} \approx 3.16 \times 10^7$까지의 소수를 체 친다. (2) 그 구간에 크기 $10^6$인 토막 하나를 만든다. (3) 작은 소수 $p$마다 토막 안의 배수를 표시한다. (4) 표시되지 않은 칸을 센다. 시간: $O(\sqrt{R} \log\log R + \Delta \cdot \pi(\sqrt{R})/\sqrt{R})$. 요즘 하드웨어에서 몇 초가 걸린다.

---

**연습문제 4.**
캐시 성능을 가장 좋게 하려면 토막 크기 $\Delta$을 $\sqrt{R}$ 가까이로 골라야 하는 까닭을 밝혀라.

??? success "연습문제 4 풀이"
    토막 배열은 빠른 아무 데나 닿기를 위해 L1/L2 캐시에 들어가야 한다. L1 캐시는 보통 32~64KB이고 L2은 256KB~1MB이다. $\Delta \approx$ 캐시 크기로 두면 토막을 훑는 체 치기가 캐시 안에 머문다. $\Delta$이 너무 크면 캐시 빗나감이 체를 크게 느리게 한다. 너무 작으면 토막마다 작은 소수를 모두 훑는 웃돈이 커진다. 알맞은 자리는 $\Delta \approx \sqrt{R}$이나 L2 캐시 크기 가운데 작은 쪽이다.

## 정리하며

이 마당은 미리 알아야 할 것、알고리즘、구현、복잡도 분석을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (CLRS), Chapter 31.
- Crandall, R. & Pomerance, C. *Prime Numbers: A Computational Perspective*. Springer, 2005.
