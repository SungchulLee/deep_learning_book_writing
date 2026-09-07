# 에라토스테네스의 체

주어진 한계까지 소수를 모두 찾는 일은 수론과 알고리즘 짜기의 바탕이 되는 일이다. 수를 하나씩 시험하는 대신 에라토스테네스의 체는 합성수를 차근히 없애어 $n$까지의 소수를 모두 $O(n \log \log n)$ 시간에, 곧 $n$에 거의 선형인 시간에 내놓는다.

## 핵심 생각

모든 합성수 $m$에는 소인수 $p \le \sqrt{m}$이 있다. 체는 작은 소수를 훑으며 그 배수를 합성수로 표시하는 방식으로 돈다. 표시되지 않고 남은 것이 소수이다.

## 알고리즘

1. 부울 배열 `is_prime[0..n]`을 만들어 `True`으로 첫자리매김한다.
2. `is_prime[0] = is_prime[1] = False`으로 둔다.
3. $2$부터 $\lfloor \sqrt{n} \rfloor$까지 $i$마다:
    - `is_prime[i]`이 `True`이면 $n$ 이하의 모든 배수 $i^2, i^2 + i,
      i^2 + 2i, \dots$을 `False`으로 표시한다.
4. 여전히 `True`인 번호가 모두 소수이다.

!!! note "왜 i의 제곱에서 시작하는가"
    소수 $p$을 다룰 때 배수 $2p, 3p, \dots, (p-1)p$은 모두 더 작은 소수가
    이미 표시했다. $p^2$에서 시작하면 겹치는 일을
    피한다.

## 구현

```python
"""
에라토스테네스의 체.

때 : O(n log log n)
Space: O(n)
"""

import math


# === 에라토스테네스의 체 ===
def sieve_of_eratosthenes(n: int) -> list[int]:
    """n까지의 소수를 모두 돌려준다."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False

    for i in range(2, int(math.isqrt(n)) + 1):
        if is_prime[i]:
            # i*i에서 시작해 배수를 표시한다
            for j in range(i * i, n + 1, i):
                is_prime[j] = False

    return [i for i in range(2, n + 1) if is_prime[i]]


# === 보기 ===
if __name__ == "__main__":
    n = 50
    primes = sieve_of_eratosthenes(n)
    print(f"Primes up to {n}: {primes}")
    print(f"Count: {len(primes)}")

    # 소수 세는 함수 견주기
    for limit in [100, 1000, 10000]:
        count = len(sieve_of_eratosthenes(limit))
        approx = int(limit / math.log(limit))
        print(f"pi({limit}) = {count}, n/ln(n) ~ {approx}")
```

## 올바름

**주장.** 체가 끝난 뒤 `is_prime[k]`이 `True`일 필요충분조건은
$k$이 소수인 것이다.

**증명.** $k$이 합성수이면 $p$을 $k$의 가장 작은 소인수라 하여
$k = p \cdot q$이라 적는다. 그러면 $p \le \sqrt{k} \le \sqrt{n}$이므로 바깥
되풀이가 $p$을 다룬다. $q \ge p$이므로 $k \ge p^2$이어서 안쪽 되풀이가
$k = p \cdot q$을 표시한다. 거꾸로 $k$이 소수이면 더 작은 소수가 $k$을
나누지 않으므로 $k$은 결코 표시되지 않는다. $\square$

## 복잡도 분석

### 시간 복잡도

표시하는 온 횟수는 다음과 같다

$$
\sum_{\substack{p \le \sqrt{n} \\ p \text{ prime}}} \frac{n}{p} = n \sum_{\substack{p \le \sqrt{n} \\ p \text{ prime}}} \frac{1}{p}
$$

메르텐스 정리에 따라 $M \approx 0.2615$이 메르텐스 상수일 때
$\sum_{p \le x} 1/p = \ln \ln x + M + O(1/\ln x)$이다. 따라서 온 일은 다음과 같다

$$
O(n \log \log n)
$$

### 공간 복잡도

부울 배열은 $O(n)$ 공간을 쓴다. 토막 낸 체 변형으로 이를
$O(\sqrt{n})$으로 줄일 수 있다.

## 최적화

### 홀수만 쓰는 체

$2$만이 짝수 소수이므로 짝수를 아예 건너뛰어 기억을
절반으로 줄인다:

```python
"""
홀수만 쓰는 체 변형(기억을 절반으로 줄인다).
"""


# === 홀수만 쓰는 체 ===
def sieve_odd_only(n: int) -> list[int]:
    """홀수만 체 쳐서 n까지의 소수를 모두 돌려준다."""
    if n < 2:
        return []
    if n == 2:
        return [2]

    # is_prime[i]은 수 2*i + 1을 나타낸다
    size = (n - 1) // 2
    is_prime = [True] * (size + 1)

    for i in range(1, (int(math.isqrt(n)) - 1) // 2 + 1):
        if is_prime[i]:
            p = 2 * i + 1
            # p*p에서 표시를 시작한다. p*p의 번호는 (p*p - 1) // 2이다
            start = (p * p - 1) // 2
            for j in range(start, size + 1, p):
                is_prime[j] = False

    primes = [2]
    for i in range(1, size + 1):
        if is_prime[i]:
            primes.append(2 * i + 1)
    return primes


# === 보기 ===
if __name__ == "__main__":
    print(sieve_odd_only(50))
```

### 비트 체

부울 배열을 비트 배열로 바꾸어 기억을 8배 줄인다.
홀수만 쓰는 체와 합치면 순진한 방식에 견주어
기억이 16배 줄어든다.

## 소수 정리와의 이음

$n$ 이하 소수의 개수 $\pi(n)$은 다음을 만족한다

$$
\pi(n) \sim \frac{n}{\ln n}
$$

곧 체는 소수를 대략 $n / \ln n$개 내놓으며 $n$ 언저리에서
소수의 밀도는 대략 $1 / \ln n$이다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (CLRS), Chapter 31.
- Hardy, G. H. & Wright, E. M. *An Introduction to the Theory of Numbers*. Oxford University Press.


## 연습문제

**연습문제 1.**
$n = 30$에서 에라토스테네스의 체를 좇아라. 찾은 소수를 늘어놓아라.

??? success "연습문제 1 풀이"
    $\{2, 3, 4, \ldots, 30\}$에서 시작한다. 2의 배수를 지운다: $\{4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30\}$. 3의 배수를 지운다: $\{9, 15, 21, 27\}$(6, 12 등은 이미 지웠다). 5의 배수를 지운다: $\{25\}$(10, 15 등은 이미 지웠다). $\sqrt{30} < 6$이므로 멈춘다. 소수: $\{2, 3, 5, 7, 11, 13, 17, 19, 23, 29\}$. 개수: 10개.

---

**연습문제 2.**
에라토스테네스의 체의 시간 복잡도가 $O(n \log\log n)$임을 밝혀라.

??? success "연습문제 2 풀이"
    소수 $p$마다 배수 $n/p$개를 지운다. 온 일은 $\sum_{p \leq n, p \text{ prime}} n/p = n \sum_{p \leq n} 1/p$이다. 메르텐스 정리에 따라 $M \approx 0.2615$이 마이셀-메르텐스 상수일 때 $\sum_{p \leq n} 1/p = \ln\ln n + M + O(1/\ln n)$이다. 따라서 온 일은 $O(n \log\log n)$이다.

---

**연습문제 3.**
한 가지 개선을 적어라. $p$의 배수를 $2p$이 아니라 $p^2$에서부터 지우는 까닭은 무엇인가?

??? success "연습문제 3 풀이"
    $p^2$보다 작은 $p$의 배수는 모두 더 작은 소수가 이미 지웠다. 자세히는 $k < p$인 $kp$은 인수 $k < p$을 가지므로 그 작은 인수를 다룰 때 표시되었다. $p^2$에서 시작하면 겹치는 일을 피한다. 이는 점근 복잡도를 바꾸지 않지만 실제로 상수 배를 크게 줄인다.

---

**연습문제 4.**
체에는 기억이 얼마나 드는가? 비트 체로 어떻게 줄일 수 있는가?

??? success "연습문제 4 풀이"
    크기 $n$인 부울 배열은 $n$바이트를 쓴다. 비트 체는 수마다 1비트를 써서 $n/8$바이트이다. 홀수만 쓰면 $n/16$바이트이다. 2이나 3으로 나누어떨어지지 않는 수만 쓰면(바퀴 인수 분해) $n/24$바이트이다. $n = 10^9$에서 순진한 방식은 1GB, 비트 체는 125MB, 홀수만은 62.5MB, 바퀴는 약 42MB이다. 기억을 더 줄이려면 토막 낸 체로 크기 $\sqrt{n}$인 구간을 한 번에 다룬다.