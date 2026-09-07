# 이차 체

폴러드 로가 닿지 못하는 수(대략 30자리 인수)에서는 이차 체(QS)가 약 100자리까지의 정수를 인수 분해하는 데 즐겨 쓰이는 방법이다. 1990년대에 수체 체가 앞지를 때까지 알려진 가장 빠른 두루 쓰는 인수 분해 알고리즘이었다.

## 수학적 바탕

이차 체는 페르마의 옛 살핌을 쓴다.

> $x^2 \equiv y^2 \pmod{n}$이면서 다음을 채우는 정수 $x$과 $y$을 찾으면,
> $x \not\equiv \pm y \pmod{n}$이면 $\gcd(x - y, n)$은 자명하지 않은
> factor of $n$.

어려움은 그런 제곱의 합동을 찾는 것이다.

## 알고리즘 살펴보기

이 알고리즘은 세 단계로 나아간다.

### 국면 1 --- 체 치기

$n$이 $p$을 법으로 이차 잉여인(곧 르장드르 기호 $(n/p) = 1$인) 작은 소수 $p$으로 이루어진 **인수 바탕** $\mathcal{B} = \{p_1, p_2, \dots, p_k\}$을 고른다.

$\lceil \sqrt{n} \rceil$ 언저리의 $x$ 값에 대해 다음을 셈한다

$$
Q(x) = x^2 - n
$$

그리고 $Q(x)$을 인수 바탕에서 인수 분해해 본다. $Q(x)$이 $\mathcal{B}$에서 온전히 인수 분해되면 **$\mathcal{B}$ 매끄럽다**고 한다:

$$
Q(x) = \prod_{i=1}^{k} p_i^{e_i}
$$

체 치는 걸음은 $Q(x)$을 나누는 소수 $p$마다 $\ln p$을 빼서 그런 매끄러운 값을 효율 좋게 많이 찾아낸다.

### 국면 2 --- 선형 대수

매끄러운 관계를 적어도 $k + 1$개 모은다(비둘기집 원리로 선형 종속이 보장된다). 관계마다 2을 법으로 한 지수 벡터로 적는다:

$$
\mathbf{v}_x = (e_1 \bmod 2, \, e_2 \bmod 2, \, \dots, \, e_k \bmod 2)
$$

가우스 없애기로 $\mathbb{F}_2$ 위에서 합이 영벡터가 되는 부분 모임을 찾는다. 이 부분 모임이 제곱의 합동을 준다.

### 국면 3 --- 인수 뽑아내기

고른 부분 모음에서 다음을 얻는다

$$
\left(\prod_{x \in S} x\right)^2 \equiv \left(\prod_{x \in S} Q(x)\right) \equiv y^2 \pmod{n}
$$

$\gcd\!\left(\prod_{x \in S} x - y, \, n\right)$을 셈한다. 적어도 확률 $1/2$으로 이는 하찮지 않은 인수이다. 아니면 다른 선형 종속을 시험한다.

## 복잡도

이차 체의 도는 시간은 준지수이다:

$$
L_n\!\left[\frac{1}{2}, 1\right] = \exp\!\left(\sqrt{\ln n \cdot \ln \ln n}\right)
$$

이 적기 $L_n[u, v] = \exp\!\bigl(v (\ln n)^u (\ln \ln n)^{1-u}\bigr)$은 다항($u = 0$)과 지수($u = 1$) 사이를 이어 준다.

| 알고리즘 | 복잡도 | 알맞은 곳 |
|---|---|---|
| 시험 나눗셈 | $O(\sqrt{n})$ | 작은 수 |
| 폴러드 로 | $O(n^{1/4})$ | 약 60자리까지의 수 |
| 이차 체 | $L_n[1/2, 1]$ | 약 100자리까지의 수 |
| 수체 체 | $L_n[1/3, c]$ | 약 100자리를 넘는 수 |

## 단순하게 만든 보기

```python
"""
이차 체 개념의 단순한 보여 주기.

이는 작은 수에서 제곱의 합동 방식을 보여 준다.
온전한 이차 체 짜기에는 개선된 체 치기와 성긴 선형 대수가 필요하다.
"""

import math
from itertools import combinations


# === 인수 바탕 고르기 ===
def build_factor_base(n: int, bound: int) -> list[int]:
    """n이 이차 잉여인 bound까지의 소수를 돌려준다."""
    primes = []
    for p in range(2, bound + 1):
        if all(p % i != 0 for i in range(2, int(p**0.5) + 1)) or p == 2:
            if p == 2 or pow(n, (p - 1) // 2, p) == 1:
                primes.append(p)
    return primes


# === 매끄러움 살피기 ===
def try_factor_over_base(value: int, base: list[int]) -> list[int] | None:
    """인수 바탕에서 값을 인수 분해한다. 지수나 None을 돌려준다."""
    if value == 0:
        return None
    exponents = []
    v = abs(value)
    for p in base:
        e = 0
        while v % p == 0:
            v //= p
            e += 1
        exponents.append(e)
    return exponents if v == 1 else None


# === 제곱의 합동 ===
def quadratic_sieve_demo(n: int) -> int | None:
    """작은 합성수 n에서 이차 체를 보여 준다."""
    base = build_factor_base(n, 30)
    root = math.isqrt(n)

    # 매끄러운 관계를 모은다
    relations = []
    for x in range(root + 1, root + 1000):
        q = x * x - n
        exps = try_factor_over_base(q, base)
        if exps is not None:
            relations.append((x, q, exps))
        if len(relations) > len(base) + 5:
            break

    # 제곱의 합동을 얻으려 부분 모임을 시험한다
    for size in range(2, min(len(relations), 6) + 1):
        for combo in combinations(range(len(relations)), size):
            combined = [0] * len(base)
            for idx in combo:
                for j, e in enumerate(relations[idx][2]):
                    combined[j] += e
            if all(e % 2 == 0 for e in combined):
                x_prod = 1
                y_sq = 1
                for idx in combo:
                    x_prod = (x_prod * relations[idx][0]) % n
                    y_sq *= relations[idx][1]
                y = math.isqrt(y_sq)
                if y * y == y_sq:
                    g = math.gcd(abs(x_prod - y), n)
                    if 1 < g < n:
                        return g
    return None


# === 보기 ===
if __name__ == "__main__":
    n = 15347  # = 103 * 149
    factor = quadratic_sieve_demo(n)
    if factor:
        print(f"n = {n}")
        print(f"Factor found: {factor}")
        print(f"Other factor: {n // factor}")
    else:
        print("No factor found (try larger sieve interval)")
```

!!! warning "실제 쓰기와 보여 주기"
    실제 이차 체 짜기는 선형 대수 국면에 다항식 체 치기, 큰 소수 변형, 덩이 란초스나 얼개를 갖춘 $\mathbb{F}_2$ 위 가우스 없애기를 쓴다. 위 코드는 오로지 가르치기 위한 것이다.

## 핵심 개선

- **여러 다항식 이차 체(MPQS).** $Q(x) = x^2 - n$ 하나 대신 다항식 여럿을 써서 체 치기를 더 넓은 범위에 펴고 매끄러운 값을 더 많이 찾는다.
- **큰 소수 변형.** 인수 바탕 한계보다 조금 큰 인수를 하나 가진 관계를 허락하고, 그런 부분 관계 둘을 합쳐 온전한 관계로 만든다.
- **덩이 란초스.** 낱말 수준 연산으로 한 번에 64비트를 다루는 덩이 방법으로 $\mathbb{F}_2$ 선형 계를 푼다.

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (CLRS), Chapter 31.
- Pomerance, C. "The Quadratic Sieve Factoring Algorithm." *Advances in Cryptology*, EUROCRYPT 1984.


## 연습문제

**연습문제 1.**
이차 체의 기본 생각을 밝혀라. $x \not\equiv \pm y$이면서 $x^2 \equiv y^2 \pmod{n}$을 찾으면 왜 $n$이 인수 분해되는가?

??? success "연습문제 1 풀이"
    $x^2 \equiv y^2 \pmod{n}$이면 $n | (x^2 - y^2) = (x-y)(x+y)$이다. $x \not\equiv \pm y \pmod{n}$이면 $n \nmid (x-y)$이고 $n \nmid (x+y)$이지만 $n$이 그 곱을 나눈다. 따라서 $\gcd(x-y, n)$은 $n$의 하찮지 않은 인수이다. 이차 체는 작은 인수 바탕에서 인수 분해되는 "매끄러운" 관계 $Q(x) \equiv x^2 \pmod{n}$을 많이 모은 뒤 $\mathbb{F}_2$ 위의 선형 대수로 합쳐 완전 제곱을 만들어 그런 $x, y$을 찾는다.

---

**연습문제 2.**
"매끄러운" 수란 무엇이며 매끄러움이 이차 체에 왜 중요한가?

??? success "연습문제 2 풀이"
    소인수가 모두 $\leq B$이면 그 수를 $B$ 매끄럽다고 한다. 이차 체는 여러 $x$ 값에서 $Q(x) = (x + \lceil\sqrt{n}\rceil)^2 - n$을 매긴다. $Q(x)$이 $B$ 매끄러우면($\leq B$인 소수로 온전히 인수 분해되면) 그 인수 분해를 $\mathbb{F}_2^{\pi(B)}$의 벡터로 적어 둔다. 인수 바탕 크기보다 많은 매끄러운 관계를 모으면 가우스 없애기가 곱이 완전 제곱이 되는 부분 모임을 찾아 바라던 $x^2 \equiv y^2 \pmod{n}$을 얻는다. $B$을 가장 좋게 고르면(매끄러울 확률과 인수 바탕 크기의 균형을 잡으면) 준지수 도는 시간을 얻는다.

---

**연습문제 3.**
이차 체의 체 치는 걸음을 적어라. 매끄러운 값을 어떻게 효율 좋게 찾아내는가?

??? success "연습문제 3 풀이"
    체 치는 구간의 $x$에 대해 배열 $A[x]$을 첫자리매김한다. 인수 바탕의 소수 $p \leq B$마다 $Q(x) \equiv 0 \pmod{p}$의 뿌리 $r$을 찾는다(홀수 $p$에는 많아야 2개). 그다음 $A[r], A[r+p], A[r+2p], \ldots$에서 $\log p$을 뺀다. 이는 $p | Q(x)$인 모든 $x$에서 $Q(x)$을 $p$으로 나누는 것을 흉내 낸다. 소수를 모두 체 친 뒤 $A[x]$이 0에 가까운 칸은 $Q(x)$이 $B$ 매끄럽다는 뜻이다(그 크기의 대부분을 작은 소수가 설명했다). 이는 $Q(x)$마다 시험 나눗셈을 하지 않아도 되게 하여 엄청나게 빨라진다.

---

**연습문제 4.**
큰 수에서 이차 체가 폴러드 로보다 빠른 까닭은 무엇이며 일반 수체 체(GNFS)는 언제 더 나은가?

??? success "연습문제 4 풀이"
    폴러드 로는 $O(N^{1/4})$에 돌며 자릿수에 대해 지수이다. 이차 체는 $L_N[1/2, 1] = \exp(O(\sqrt{\log N \log\log N}))$에 돌며 이는 준지수이다. 약 35자리를 넘는 수에서는 이차 체가 더 빠르다. 일반 수체 체는 $L_N[1/3, c]$에 돌며 점근으로 이차 체보다 빠르다. 갈림목은 약 100자리이다. 약 100자리 아래에서는 이차 체가 겨룰 만하고 위에서는 일반 수체 체가 앞선다. 지금의 RSA 인수 분해 기록(RSA-250, 250자리)은 모두 일반 수체 체를 쓴다.