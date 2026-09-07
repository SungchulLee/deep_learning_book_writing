# AKS 소수 시험

2002년 이전에 알려진 정해진 소수 시험은 모두 지수이거나 밝혀지지 않은 추측에 기댔다. 아그라왈, 카얄, 삭세나의 AKS 시험은 소수 판정의 첫 **정해진 다항 시간** 알고리즘을 내놓아 오랜 열린 물음을 매듭지었다.

## 핵심 결과

AKS 알고리즘은 PRIMES $\in$ P임을 밝힌다. 정수 $n$이 주어지면
$\log n$(자릿수)에 대해 다항인 시간에 $n$이 소수인지
가른다.

## 수학적 바탕

이 시험은 페르마의 작은 정리를 다항식 고리로 넓힌 것 위에
세워진다.

!!! note "핵심 항등식"
    정수 $n \ge 2$이 소수일 필요충분조건은 다항식 합동

    $$
    (x + a)^n \equiv x^n + a \pmod{n}
    $$

    이 $n$과 서로 소인 모든 정수 $a$에 대해 성립하는 것이다.

이 항등식을 곧바로 살피려면 $(x + a)^n$을 펼쳐야 하는데 항이
$n + 1$개여서 지수가 된다. AKS의 통찰은 조심스레 고른 작은 $r$에 대해
$x^r - 1$을 법으로 이 항등식을 살피는 것이다:

$$
(x + a)^n \equiv x^n + a \pmod{x^r - 1, \, n}
$$

## 알고리즘

**들임:** 정수 $n \ge 2$.

1. **완전 거듭제곱 살피기.** 정수 $b \ge 2$과 $k \ge 2$에 대해
   $n = b^k$이면 합성수를 내놓는다.
2. **알맞은 $r$ 찾기.** $\text{ord}_r(n) > (\log_2 n)^2$인 가장 작은 $r$을 찾는다.
   여기서 $\text{ord}_r(n)$은 $r$을 법으로 하는 $n$의 곱셈 차수이다.
3. **최대 공약수 살피기.** 모든 $a \le r$에 대해 $1 < \gcd(a, n) < n$이면
   합성수를 내놓는다.
4. **작은 $n$ 살피기.** $n \le r$이면 소수를 내놓는다.
5. **다항식 살피기.** $a = 1, 2, \dots, \lfloor \sqrt{\phi(r)} \log_2 n \rfloor$에 대해
   다음을 확인한다

    $$
    (x + a)^n \equiv x^n + a \pmod{x^r - 1, \, n}
    $$

    하나라도 어긋나면 합성수를 내놓는다.

6. 소수를 내놓는다.

## 옳음 얼개

이 알고리즘은 두 방향에 기댄다:

- **온전함.** $n$이 소수이면 다항식 항등식이 모든 $a$에 대해 성립하므로
  ($p$을 법으로 하는 이항 정리의 따름이다) 알고리즘은
  늘 소수를 내놓는다.
- **건전함.** $n$이 합성수인데 모든 다항식 살피기를 지나면
  $p$이 $n$의 소인수일 때 $\mathbb{F}_p[x]/(x^r - 1)$의 곱셈 무리의 성질로
  어긋남을 이끌어 낼 수 있다.
  $r$의 한계와 $a$ 값의 개수가 $n$을 소수 거듭제곱으로 몰아붙일 만큼
  제약을 주며 그것은 걸음 1에서 이미 걸러 냈다.

## 복잡도

- **$r$ 찾기:** $r = O((\log n)^5)$이면 넉넉하다(뒤에 어떤 변형에서는
  $O((\log n)^3)$으로 나아졌다).
- **다항식 살피기:** 살피기마다 차수 $< r$인 다항식을 $n$을 법으로 곱하며
  살피기마다 $\widetilde{O}(r \log n)$이 든다.
- **살피기 횟수:** $O(\sqrt{r} \log n)$.
- **온 시간:** $\widetilde{O}(r^{3/2} (\log n)^2)$이며 이는 $r$의 본디 한계로
  $\widetilde{O}((\log n)^{21/2})$이다.

렌스트라와 포머런스의 뒤이은 개선이 복잡도를
$\widetilde{O}((\log n)^6)$으로 줄였다.

| 소수 시험 | 갈래 | 복잡도 |
|---|---|---|
| 시험 나눗셈 | 정해짐 | $O(\sqrt{n})$ |
| 밀러-라빈 | 확률 | 바퀴마다 $O(k \log^2 n)$ |
| AKS | 정해짐 | $\widetilde{O}((\log n)^6)$ |

## 단순하게 짜기

```python
"""
단순하게 만든 AKS 소수 시험(가르치기용).

온전한 AKS 시험은 (x^r - 1, n)을 법으로 하는 다항식 셈이 필요하다.
이 짜기는 알고리즘의 얼개를 보여 준다.
"""

import math


# === 완전 거듭제곱 살피기 ===
def is_perfect_power(n: int) -> bool:
    """b >= 2, k >= 2인 어떤 b, k에 대해 n = b^k인지 살핀다."""
    if n <= 3:
        return False
    for k in range(2, n.bit_length() + 1):
        b = round(n ** (1.0 / k))
        for candidate in (b - 1, b, b + 1):
            if candidate >= 2 and candidate**k == n:
                return True
    return False


# === 곱셈 차수 ===
def multiplicative_order(n: int, r: int) -> int:
    """n^k = 1 (mod r)을 이루는 가장 작은 k를 돌려준다."""
    if math.gcd(n, r) > 1:
        return 0
    result = 1
    power = n % r
    while power != 1:
        power = (power * n) % r
        result += 1
    return result


# === (x^r - 1, n)을 법으로 하는 다항식 곱셈 ===
def poly_mult_mod(a: list[int], b: list[int], r: int, n: int) -> list[int]:
    """x^r - 1과 n을 법으로 다항식 a과 b을 곱한다."""
    result = [0] * r
    for i, ai in enumerate(a):
        if ai == 0:
            continue
        for j, bj in enumerate(b):
            if bj == 0:
                continue
            result[(i + j) % r] = (result[(i + j) % r] + ai * bj) % n
    return result


# === (x^r - 1, n)을 법으로 하는 다항식 거듭제곱 ===
def poly_pow_mod(base: list[int], exp: int, r: int, n: int) -> list[int]:
    """x^r - 1과 n을 법으로 base^exp을 셈한다."""
    result = [0] * r
    result[0] = 1
    b = base[:]
    while exp > 0:
        if exp % 2 == 1:
            result = poly_mult_mod(result, b, r, n)
        b = poly_mult_mod(b, b, r, n)
        exp //= 2
    return result


# === AKS 시험 ===
def aks(n: int) -> bool:
    """n이 소수이면 True를 돌려준다(AKS 알고리즘)."""
    if n <= 1:
        return False
    if n <= 3:
        return True

    # 걸음 1: 완전 거듭제곱 살피기
    if is_perfect_power(n):
        return False

    # 걸음 2: r 찾기
    log2n = math.log2(n)
    threshold = log2n * log2n
    r = 2
    while r < n:
        if math.gcd(r, n) > 1 and r < n:
            if math.gcd(r, n) == n:
                r += 1
                continue
            return False
        if multiplicative_order(n, r) > threshold:
            break
        r += 1

    # 걸음 3: 최대 공약수 살피기
    for a in range(2, min(r + 1, n)):
        g = math.gcd(a, n)
        if 1 < g < n:
            return False

    # 걸음 4: 작은 n
    if n <= r:
        return True

    # 걸음 5: 다항식 살피기
    limit = int(math.sqrt(r) * log2n) + 1
    for a in range(1, limit + 1):
        # (x^r - 1, n)을 법으로 (x + a)^n을 셈한다
        poly = [0] * r
        poly[0] = a % n
        poly[1 % r] = (poly[1 % r] + 1) % n
        lhs = poly_pow_mod(poly, n, r, n)

        # 바라는 값: x^(n mod r) + a
        rhs = [0] * r
        rhs[n % r] = 1
        rhs[0] = (rhs[0] + a) % n

        if lhs != rhs:
            return False

    return True


# === 보기 ===
if __name__ == "__main__":
    test_values = [2, 7, 10, 13, 15, 31, 37, 49, 97]
    for val in test_values:
        result = "prime" if aks(val) else "composite"
        print(f"AKS({val}) = {result}")
```

!!! warning "성능 알림"
    AKS은 주로 이론에서 중요하다. 실제로는 넉넉한 바퀴의 밀러-라빈(또는 작은 $n$의 정해진 변형)이 훨씬 빠르다. AKS은 다항 시간의 정해진 시험이 *있음*을 밝히지만 상수 때문에 큰 들임에는 쓸 수 없다.

## 역사에서의 뜻

AKS 결과는 소수 시험의 복잡도를 매듭지었다:

- **AKS 이전:** PRIMES은 (밀러-라빈으로) 여 NP와 BPP에 든다고 알려졌지만
  P에 든다고는 알려지지 않았다.
- **AKS 이후(2002):** 조건 없이 PRIMES $\in$ P이다.

## 참고 문헌

- Agrawal, M., Kayal, N., & Saxena, N. "PRIMES is in P." *Annals of Mathematics*, 160(2), 2004.
- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. *Introduction to Algorithms* (CLRS), Chapter 31.


## 연습문제

**연습문제 1.**
AKS의 바탕이 되는 주요 정리를 적어라. 어떤 다항식 항등식이 소수를 성격 짓는가?

??? success "연습문제 1 풀이"
    AKS 시험은 다음 항등식에 바탕한다. $n$이 소수일 필요충분조건은 $\gcd(a, n) = 1$인 모든 $a$에 대해 $(x + a)^n \equiv x^n + a \pmod{n}$인 것이다. 모든 $a$에 대해 살피는 것은 비싸므로 AKS는 이를 줄인다. 알맞은 $r$을 찾아 $a = 1, \ldots, O(\sqrt{\phi(r)} \log n)$에 대해 $(x + a)^n \equiv x^n + a \pmod{x^r - 1, n}$을 살핀다. 이것으로 소수인지 가리기에 넉넉하며 다항 시간에 돈다.

---

**연습문제 2.**
AKS 알고리즘(2002)이 실제로는 밀러-라빈보다 느린데도 돌파였던 까닭은 무엇인가?

??? success "연습문제 2 풀이"
    AKS은 첫 정해진 다항 시간 소수 시험으로 PRIMES $\in$ P임을 밝혔다. AKS 이전에 가장 좋은 정해진 시험은 시험 나눗셈($O(\sqrt{n})$으로 자릿수에 대해 지수)이었다. 밀러-라빈은 더 빠르지만($O(k \log^2 n)$) 마구잡이이다(또는 정해지려면 일반 리만 가설에 기댄다). AKS은 $\tilde{O}(\log^{6} n)$에 돌며(조건 아래에서 $\tilde{O}(\log^{3} n)$으로 나아졌다) 다항이지만 실제로는 밀러-라빈보다 훨씬 느리다. 그 이론상의 중요함은 소수 시험의 복잡도를 매듭지은 데 있다.

---

**연습문제 3.**
$n$비트 수의 소수 시험에서 시험 나눗셈, 밀러-라빈, AKS의 시간 복잡도를 견주어라.

??? success "연습문제 3 풀이"
    시험 나눗셈: $O(2^{n/2})$으로 비트 길이에 대해 지수이다. 밀러-라빈: $k$바퀴에 $O(k \cdot n^2)$이며(바퀴마다 $O(n^2)$의 법 거듭제곱을 한다) 다항이고 마구잡이이다. AKS: 정해진 방식으로 $\tilde{O}(n^6)$(처음 것)이고 어떤 추측 아래에서 $\tilde{O}(n^3)$으로 나아졌다. 1024비트 수에서 시험 나눗셈은 아주 쓸 수 없고($2^{512}$번 셈), 밀러-라빈은 밀리초가 걸리며, AKS은 짜기에 따라 몇 초에서 몇 분이 걸린다.

---

**연습문제 4.**
AKS이 정해진 방식인데도 실제로는(보기로 OpenSSL에서) AKS 대신 밀러-라빈을 쓰는 까닭을 밝혀라.

??? success "연습문제 4 풀이"
    $k$바퀴의 밀러-라빈은 어긋날 확률이 $\leq 4^{-k}$이다. $k = 64$이면 어긋남이 $< 2^{-128}$으로 실제로는 무시할 만하다. AKS보다 자릿수가 여러 자리 빠르다. 암호 열쇠 만들기(보기로 RSA 소수)에서는 이 빠르기 차이가 중요하다. 2048비트 소수를 만들려면 후보를 많이 시험해야 한다. 밀러-라빈은 하나를 밀리초에 시험하지만 AKS은 훨씬 오래 걸린다. 밀러-라빈의 확률 보장($< 2^{-128}$ 어긋남)이 하드웨어 오류율보다 강할 때 AKS의 이론 보장(정해짐)은 필요 없다.