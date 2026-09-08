# 법 역원

법 셈에서 나눗셈은 곧바로 뜻매김되지 않지만 **법 역원**을 곱해 같은 효과를 얻을 수 있다. $m$을 법으로 하는 $a$의 법 역원은 $a \cdot a^{-1} \equiv 1 \pmod{m}$인 정수 $a^{-1}$이다. 이 연산은 RSA 복호, 법 방정식 풀기, 소수를 법으로 하는 얽음 양 셈하기에 꼭 필요하다.

---

## 1. 뜻매김과 있음

$m$을 법으로 하는 $a$의 **법 곱셈 역원**은 다음을 만족하는 정수 $x$이다:

$$
ax \equiv 1 \pmod{m}
$$

$x = a^{-1} \pmod{m}$이라 적는다.

!!! info "있음 조건"

    법 역원 $a^{-1} \pmod{m}$이 있을 필요충분조건은 $\gcd(a, m) = 1$인 것이다(곧 $a$과 $m$이 서로 소인 것이다).

**증명.** 방정식 $ax \equiv 1 \pmod{m}$은 어떤 정수 $y$에 대해 $ax + my = 1$인 것과 같다. 베주 항등식([베주 항등식](../divisibility/bezout.md)을 보라)에 따라 이는 $\gcd(a, m) \mid 1$일 때에만 풀이가 있고 그러려면 $\gcd(a, m) = 1$이어야 한다. $\square$

!!! warning "하나뿐임"

    있을 때 법 역원은 $m$을 법으로 하여 하나뿐이다. $x_1$과 $x_2$이 모두 $ax \equiv 1$을 만족하면 $a(x_1 - x_2) \equiv 0 \pmod{m}$이다. $\gcd(a, m) = 1$이므로 $x_1 \equiv x_2 \pmod{m}$을 얻는다.

---

## 2. 방법 1: 넓힌 유클리드 알고리즘

가장 두루 쓰는 방법은 넓힌 유클리드 알고리즘으로 역원을 셈한다([넓힌 유클리드](../divisibility/extended.md)를 보라). $\gcd(a, m) = 1$이면 넓힌 최대 공약수가 $ax + my = 1$인 $x, y$을 돌려준다. $m$을 법으로 줄이면:

$$
ax \equiv 1 \pmod{m}
$$

따라서 $a^{-1} \equiv x \pmod{m}$이다.

**시간 복잡도:** $O(\log m)$.

이 방법은 법 $m$이 소수이든 합성수이든 통한다.

---

## 3. 방법 2: 페르마의 작은 정리(소수 법)

$m = p$이 소수이면 페르마의 작은 정리([페르마의 작은 정리](fermat.md)를 보라)는 $\gcd(a, p) = 1$에 대해 $a^{p-1} \equiv 1 \pmod{p}$이라 한다. 따라서:

$$
a^{-1} \equiv a^{p-2} \pmod{p}
$$

이는 법 거듭제곱([법 거듭제곱](exponentiation.md)을 보라)으로 $O(\log p)$ 시간에 효율 좋게 셈한다.

!!! tip "어느 방법을 쓸 것인가?"

    - **소수 법**: 페르마 방법($a^{p-2} \bmod p$)이 짜기 더 쉽다
    - **합성수 법**: 넓힌 유클리드 알고리즘을 쓴다
    - **성능**: 둘 다 $O(\log m)$이다. 넓힌 최대 공약수가 법 거듭제곱의 웃돈을 피하므로 보통 상수 배만큼 빠르다

---

## 4. 풀이 예제

$3^{-1} \pmod{7}$ 찾기:

**방법 1(넓힌 최대 공약수):** $3x + 7y = 1$을 푼다. $(3, 7)$에 넓힌 유클리드 알고리즘을 돌리면:

- $7 = 2 \cdot 3 + 1$이므로 $1 = 7 - 2 \cdot 3$이고 $x = -2 \equiv 5 \pmod{7}$이다

**방법 2(페르마):** $3^{-1} \equiv 3^{7-2} = 3^5 \pmod{7}$이다. $3^5 = 243 = 34 \cdot 7 + 5$이므로 $3^{-1} \equiv 5 \pmod{7}$이다.

**확인:** $3 \cdot 5 = 15 = 2 \cdot 7 + 1 \equiv 1 \pmod{7}$. $\checkmark$

---

## 5. 작은 법의 역원 표

소수 $p$에서는 0이 아닌 모든 원소가 역원을 가진다. $p = 7$의 온전한 역원 표:

| $a$ | 1 | 2 | 3 | 4 | 5 | 6 |
|-----|---|---|---|---|---|---|
| $a^{-1} \pmod{7}$ | 1 | 4 | 5 | 2 | 3 | 6 |

역원 함수가 $\{1, 2, \ldots, p-1\}$의 자리바꿈임을 눈여겨보라.

---

## 6. 구현

```python
"""
법 역원 셈하기.

방법 둘을 준다: 넓힌 유클리드 알고리즘(어떤 법에도 통한다)과
서로 소인 법)과 페르마의 작은 정리(소수 법만)를 견준다.
"""

# === 넓힌 최대 공약수 방법 ===

def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """a*x + b*y = g = gcd(a, b)를 이루는 (g, x, y)를 돌려준다."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y

def mod_inverse_egcd(a: int, m: int) -> int:
    """넓힌 유클리드 알고리즘으로 a^{-1} mod m을 셈한다.

    gcd(a, m) != 1이면 ValueError를 일으킨다.
    """
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"Inverse does not exist: gcd({a}, {m}) = {g}")
    return x % m

# === 페르마 방법(소수 법) ===

def mod_inverse_fermat(a: int, p: int) -> int:
    """페르마의 작은 정리로 a^{-1} mod p을 셈한다.

    p가 소수이고 gcd(a, p) = 1이라고 본다.
    """
    return pow(a, p - 2, p)

# === 메인 ===

if __name__ == "__main__":
    # 소수 법에서 두 방법을 견준다
    p = 7
    print(f"Modular inverses mod {p}:")
    for a in range(1, p):
        inv_egcd = mod_inverse_egcd(a, p)
        inv_fermat = mod_inverse_fermat(a, p)
        print(f"  {a}^(-1) = {inv_egcd} (EGCD), {inv_fermat} (Fermat), "
              f"verify: {a}*{inv_egcd} mod {p} = {(a * inv_egcd) % p}")

    # 합성수 법(넓힌 최대 공약수만 통한다)
    m = 12
    print(f"\nModular inverses mod {m} (EGCD only):")
    for a in range(1, m):
        try:
            inv = mod_inverse_egcd(a, m)
            print(f"  {a}^(-1) = {inv}, verify: {a}*{inv} mod {m} = {(a * inv) % m}")
        except ValueError:
            print(f"  {a}^(-1) does not exist (gcd({a},{m}) != 1)")
```

**출력:**

```
Modular inverses mod 7:
  1^(-1) = 1 (EGCD), 1 (Fermat), verify: 1*1 mod 7 = 1
  2^(-1) = 4 (EGCD), 4 (Fermat), verify: 2*4 mod 7 = 1
  3^(-1) = 5 (EGCD), 5 (Fermat), verify: 3*5 mod 7 = 1
  4^(-1) = 2 (EGCD), 2 (Fermat), verify: 4*2 mod 7 = 1
  5^(-1) = 3 (EGCD), 3 (Fermat), verify: 5*3 mod 7 = 1
  6^(-1) = 6 (EGCD), 6 (Fermat), verify: 6*6 mod 7 = 1

Modular inverses mod 12 (EGCD only):
  1^(-1) = 1, verify: 1*1 mod 12 = 1
  2^(-1) does not exist (gcd(2,12) != 1)
  3^(-1) does not exist (gcd(3,12) != 1)
  4^(-1) does not exist (gcd(4,12) != 1)
  5^(-1) = 5, verify: 5*5 mod 12 = 1
  6^(-1) does not exist (gcd(6,12) != 1)
  7^(-1) = 7, verify: 7*7 mod 12 = 1
  8^(-1) does not exist (gcd(8,12) != 1)
  9^(-1) does not exist (gcd(9,12) != 1)
  10^(-1) does not exist (gcd(10,12) != 1)
  11^(-1) = 11, verify: 11*11 mod 12 = 1
```

---

## 연습문제

**연습문제 1.**
(가) 넓힌 유클리드 알고리즘과 (나) 페르마의 작은 정리로 $5^{-1} \bmod 13$을 찾아라.

??? success "연습문제 1 풀이"
    (가) 넓힌 최대 공약수: $13 = 2 \cdot 5 + 3$, $5 = 1 \cdot 3 + 2$, $3 = 1 \cdot 2 + 1$. 거꾸로 넣으면 $1 = 3 - 2 = 3 - (5 - 3) = 2 \cdot 3 - 5 = 2(13 - 2 \cdot 5) - 5 = 2 \cdot 13 - 5 \cdot 5$이다. 따라서 $5^{-1} \equiv -5 \equiv 8 \pmod{13}$이다.
    (나) 페르마: $5^{-1} \equiv 5^{13-2} = 5^{11} \pmod{13}$이다. $5^2 = 25 \equiv 12, 5^4 \equiv 144 \equiv 1, 5^8 \equiv 1, 5^{11} = 5^8 \cdot 5^2 \cdot 5 = 1 \cdot 12 \cdot 5 = 60 \equiv 8 \pmod{13}$이다. 둘 다 $5^{-1} \equiv 8$을 준다.

---

**연습문제 2.**
법 역원이 없는 때는 언제인가? 필요충분조건을 들어라.

??? success "연습문제 2 풀이"
    $a^{-1} \bmod m$이 있을 필요충분조건은 $\gcd(a, m) = 1$이다. $d = \gcd(a,m) > 1$이면 $ax \equiv 1 \pmod{m}$에 풀이가 없다. $d | ax$이고 $d | m$인데 $d \nmid 1$이므로 어긋난다. $\gcd(a,m) = 1$이면 베주가 $ax + my = 1$을 주므로 $ax \equiv 1 \pmod{m}$이고 $x$이 역원이다.

---

**연습문제 3.**
법 역원으로 $\binom{10}{4} \bmod 13$을 셈하라($10, 4 < 13$이므로 뤼카 정리는 필요 없다).

??? success "연습문제 3 풀이"
    $\binom{10}{4} = \frac{10!}{4! \cdot 6!} = \frac{10 \cdot 9 \cdot 8 \cdot 7}{4!} = \frac{5040}{24} = 210$이다. 법으로 셈하기: $10 \cdot 9 \cdot 8 \cdot 7 = 5040 \equiv 5040 \bmod 13$이다. $5040 = 387 \cdot 13 + 9$이므로 $5040 \equiv 9$이다. $4! = 24 \equiv 24 - 13 = 11$이다. $11^{-1} \bmod 13$: $11 \cdot 6 = 66 = 5 \cdot 13 + 1$이므로 $11^{-1} \equiv 6$이다. 결과: $9 \cdot 6 = 54 \equiv 54 - 4 \cdot 13 = 2 \pmod{13}$이다. 확인: $210 \bmod 13 = 210 - 16 \cdot 13 = 210 - 208 = 2$. 맞다.

---

**연습문제 4.**
깊은 배움의 양자화 얼거리 안에서 법 셈의 나눗셈에 법 역원이 어떻게 쓰이는지 밝혀라.

??? success "연습문제 4 풀이"
    양자화한 신경망 추론에서는 셈에 고정 소수점이나 법 셈을 쓴다. 법 셈에서 상수 $c$으로 나누는 것은 $c^{-1} \bmod m$을 곱하는 것이다. 보기로 묶음 고르게 맞추기는 표준 편차로 나눈다. 법 $m = 2^{32}$의 양자화 꼴에서 $c$이 홀수이면($\gcd(c, 2^{32}) = 1$) $c^{-1} \bmod 2^{32}$을 미리 셈해 나눗셈을 곱셈으로 바꾼다. 이는 하드웨어에서 더 빠르다(곱셈은 명령 하나, 나눗셈은 여럿이다). $c$이 짝수이면 2의 거듭제곱을 빼내고 오른쪽 밀기와 홀수 부분의 법 역원을 쓴다.

## 정리하며

이 마당은 뜻매김과 있음、방법 1: 넓힌 유클리드 알고리즘、방법 2: 페르마의 작은 정리(소수 법)、풀이 예제을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
