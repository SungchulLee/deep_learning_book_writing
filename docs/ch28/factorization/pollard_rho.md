# 폴러드 로 알고리즘

시험 나눗셈은 작은 인수를 빨리 찾지만 가장 나쁜 경우가 $O(\sqrt{n})$이어서 큰 합성수에는 쓸 수 없다. 폴러드 로 알고리즘은 생일 역설을 써서 기댓값 $O(n^{1/4})$번의 셈으로 $n$의 하찮지 않은 인수를 찾으며, 인수 크기가 어지간한 수를 인수 분해하는 가장 효율 좋은 방법 가운데 하나이다.

## 핵심 생각

$p$을 $n$의 알려지지 않은 소인수라 하자. $n$을 법으로 거짓 마구잡이 차례 $x_0, x_1, x_2, \dots$을 만들면 $p$을 법으로 한 값이 $n$을 법으로 한 값보다 훨씬 빨리 돌이를 이룬다. 생일 역설에 따라 부딪힘 $x_i \equiv x_j \pmod{p}$이 대략 $O(\sqrt{p})$걸음 뒤에 생긴다. 그때 $\gcd(x_i - x_j, n)$이 인수 $p$을 드러낸다.

"로"라는 이름은 이 차례를 그림으로 그렸을 때의 모양에서 왔다. 꼬리가 돌이로 이어져 그리스 글자 $\rho$을 닮았다.

## 되풀이 함수

붙박이 상수 $c \notin \{0, -2\}$에 대해 다항식 옮김 $f(x) = x^2 + c \pmod{n}$을 쓴다. $x_0$에서 시작해 다음을 뜻매김한다

$$
x_{i+1} = f(x_i) = x_i^2 + c \pmod{n}
$$

이 이음은 정해진 대로 나아가지만 어떤 붙박인 수로 나눈 나머지에서는 유사 마구잡이처럼 움직인다
$n$의 소인수 $p$을 준다.

## 플로이드 돌이 찾기

앞의 값을 모두 담아 두는 대신 플로이드의 거북과 토끼 방법은 가리개 둘을 쓴다:

- **거북**: 한 번에 한 걸음씩 나아간다, $x_i$.
- **토끼**: 한 번에 두 걸음씩 나아간다, $x_{2i}$.

걸음마다 $d = \gcd(|x_i - x_{2i}|, n)$을 셈한다. $1 < d < n$이면 $d$이 하찮지 않은 인수이다.

## 알고리즘

```python
"""
정수 인수분해를 위한 폴라드 로 알고리즘.

기대 시간: 산술 연산 O(n^{1/4})번.
"""

import math
import random


# === 폴러드 로 ===
def pollard_rho(n: int) -> int:
    """n의 하찮지 않은 인수를 돌려준다. n이 소수이면 n을 돌려준다."""
    if n % 2 == 0:
        return 2
    while True:
        x = random.randint(2, n - 1)
        y = x
        c = random.randint(1, n - 1)
        d = 1
        while d == 1:
            x = (x * x + c) % n          # tortoise
            y = (y * y + c) % n           # hare step 1
            y = (y * y + c) % n           # hare step 2
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d
        # d == n이면 인수를 못 찾고 돌이가 생긴 것이다. 다시 한다


# === 온전한 인수 분해 ===
def factorize(n: int) -> list[int]:
    """n의 온전한 소인수 분해를 돌려준다."""
    if n <= 1:
        return []
    factors = []
    stack = [n]
    while stack:
        k = stack.pop()
        if k == 1:
            continue
        if is_prime_miller_rabin(k):
            factors.append(k)
        else:
            d = pollard_rho(k)
            stack.append(d)
            stack.append(k // d)
    return sorted(factors)


def is_prime_miller_rabin(n: int, rounds: int = 20) -> bool:
    """밀러-라빈 소수 시험."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for _ in range(rounds):
        a = random.randint(2, n - 2)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


# === 보기 ===
if __name__ == "__main__":
    n = 8051
    print(f"n = {n}")
    factor = pollard_rho(n)
    print(f"Non-trivial factor: {factor}")
    print(f"Full factorization: {factorize(n)}")
```

## 복잡도 분석

$p$을 $n$의 가장 작은 소인수라 하자.

- **기대 부딪힘.** 생일 역설에 따라 $p$을 법으로 하는 부딪힘이 $O(\sqrt{p})$번 되풀이 뒤에 생긴다.
- **$p \le \sqrt{n}$이므로** 되풀이 횟수의 기댓값은 $O(n^{1/4})$이다.
- **되풀이마다** 법 곱셈을 상수 번 하고 최대 공약수를 한 번 셈하며 여느 셈에서 저마다 $O(\log^2 n)$이 든다.

기댓값으로 온 시간: 비트 연산 $O(n^{1/4} \log^2 n)$번.

!!! tip "브렌트의 개선"
    브렌트의 변형은 플로이드의 돌이 찾기를 다른 나아감 차례표로 바꾼다. 최대 공약수 셈하기 횟수를 줄여 실제로 인수를 약 24% 빨리 찾는다.

## 실패와 다시 하기

$d = \gcd(|x_i - x_{2i}|, n) = n$이면 거북과 토끼가 $n$ 자체를 법으로 부딪혀 하찮은 인수만 나온 것이다. 고치기는 간단하다. 다른 아무 $c$이나 $x_0$으로 다시 시작한다. 여러 번 다시 시작해야 할 확률은 낮다.

## 실용적인 고려

| 항목 | 내용 |
|---|---|
| 알맞은 곳 | 인수가 약 30자리까지인 약 60자리까지의 수 |
| 함께 쓰는 것 | 밀러-라빈(인수 분해 전에 소수인지 시험한다) |
| 알맞지 않은 곳 | 큰 인수 둘의 반소수(대신 일반 수체 체를 쓴다) |
| 나란히 할 수 있음 | 그렇다 --- $c$ 값을 달리해 서로 얽매이지 않은 사례를 돌린다 |

!!! warning "c 고르기"
    $c = 0$(차례가 $x^{2^k}$으로 무너진다)과 $c = -2$(되풀이에 인수 찾기를 막는 대수 얼개가 있다)은 피하라.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (CLRS), Chapter 31.
- Pollard, J. M. "A Monte Carlo method for factorization." *BIT Numerical Mathematics*, 15(3), 1975.


## 연습문제

**연습문제 1.**
$f(x) = x^2 + 1 \bmod 91$과 $x_0 = 2$으로 $n = 91$에서 폴러드 로 알고리즘을 좇아라.

??? success "연습문제 1 풀이"
    차례를 셈한다: $x_1 = 5, x_2 = 26, x_3 = 677 \bmod 91 = 40, x_4 = 1601 \bmod 91 = 53, x_5 = 2810 \bmod 91 = 79, x_6 = 6242 \bmod 91 = 53$. 돌이를 찾았다: $x_6 = x_4 = 53$. 플로이드 방법(거북과 토끼)으로 걸음마다 $\gcd(|x_i - x_{2i}|, 91)$을 셈한다. 어느 순간 $\gcd(|26 - 40|, 91) = \gcd(14, 91) = 7$이 된다. 인수를 찾았다: $91 = 7 \times 13$.

---

**연습문제 2.**
폴러드 로의 기대 도는 시간이 $O(n^{1/4})$인 까닭과 생일 역설이 어떻게 쓰이는지 밝혀라.

??? success "연습문제 2 풀이"
    $p$을 $n$의 가장 작은 소인수라 하면 $p \leq \sqrt{n}$이다. 차례 $x_i \bmod p$은 $\{0, \ldots, p-1\}$에서 거짓 마구잡이이다. 생일 역설에 따라 부딪힘 $x_i \equiv x_j \pmod{p}$이 평균 $O(\sqrt{p})$걸음 뒤에 일어난다. $x_i \equiv x_j \pmod{p}$이면 $\gcd(|x_i - x_j|, n)$이 $p$(또는 그 배수)을 드러낸다. $p \leq n^{1/2}$이므로 기대 걸음 수는 $O(n^{1/4})$이다. 걸음마다 법 셈에 $O(\log^2 n)$이 들어 온 시간은 $O(n^{1/4} \log^2 n)$이다.

---

**연습문제 3.**
폴러드 로가 $\gcd(|x_i - x_j|, n) = n$을 찾으면 어떻게 되는가? 알고리즘은 이를 어떻게 다루어야 하는가?

??? success "연습문제 3 풀이"
    $\gcd = n$이면 제대로 된 인수를 드러내지 못한 채 $n$을 법으로 하는 돌이가 생긴 것이다(두 차례가 모든 소인수를 법으로 하여 한꺼번에 부딪혔다). 알고리즘은 다른 다항식으로 다시 시작해야 한다. 보기로 새 아무 $c \neq 0, -2$을 쓴 $f(x) = x^2 + c$이다. 또는 (돌이를 더 빨리 찾는) 브렌트의 개선을 쓰거나, 최대 공약수를 더 자주 셈해 그것이 처음 1을 넘은 걸음으로 되돌아간다.

---

**연습문제 4.**
$n$자리 수를 인수 분해할 때 시험 나눗셈, 폴러드 로, 이차 체의 도는 시간을 견주어라.

??? success "연습문제 4 풀이"
    $N$이 십진 $d$자리라 하면 $N \approx 10^d$이다. 시험 나눗셈: $O(\sqrt{N}) = O(10^{d/2})$으로 $d$에 대해 지수이다. 폴러드 로: $O(N^{1/4}) = O(10^{d/4})$으로 여전히 지수이지만 밑이 더 작다. 이차 체: $O(\exp(c\sqrt{d \ln d}))$으로 $d$에 대해 준지수이다. $d = 50$에서 시험 나눗셈은 약 $10^{25}$(쓸 수 없다), 폴러드 로는 약 $10^{12.5}$(아슬아슬하다), 이차 체는 약 $10^{9}$(쓸 만하다)이다. $d = 100$에서는 이차 체(또는 일반 수체 체)만 쓸 만하다.