# 중국인 나머지 정리

중국인 나머지 정리(CRT)는 자연스러운 물음을 다룬다. 어떤 정수를 짝마다 서로 소인 여러 법으로 나눈 나머지를 안다면 본디 정수를 되지을 수 있는가? 답은 그렇다이며 그 되짓기는 모든 법의 곱을 법으로 하여 하나뿐이다. 이 결과는 3세기 중국 수학자 손자에게로 거슬러 올라가며 오늘날 RSA 개선, 큰 정수 셈, 나란한 셈에 쓰인다.

## 이끄는 보기

어떤 수를 3으로 나누면 나머지가 2, 5으로 나누면 3, 7으로 나누면 2이다. 그 수는 무엇인가?

다음을 풀어야 한다:

$$
x \equiv 2 \pmod{3}, \quad x \equiv 3 \pmod{5}, \quad x \equiv 2 \pmod{7}
$$

중국인 나머지 정리는 $3 \cdot 5 \cdot 7 = 105$을 법으로 하는 유일한 풀이를 보장한다. 곧 보겠지만 $x \equiv 23 \pmod{105}$이다.

## 정리의 진술

!!! info "중국인 나머지 정리"

    $m_1, m_2, \ldots, m_k$을 짝마다 서로 소인 양의 정수라 하자(곧 $i \ne j$이면 $\gcd(m_i, m_j) = 1$). $M = m_1 m_2 \cdots m_k$이라 하자. 어떤 정수 $a_1, a_2, \ldots, a_k$에 대해서도 다음 연립은

    $$
    x \equiv a_1 \pmod{m_1}, \quad x \equiv a_2 \pmod{m_2}, \quad \ldots, \quad x \equiv a_k \pmod{m_k}
    $$

    $M$을 법으로 하는 유일한 풀이를 가진다.

## 세워 보이는 밝힘

이 밝힘은 풀이를 드러내어 세운다.

**걸음 1.** 각 $i$에 대해 $M_i = M / m_i$($m_i$을 뺀 모든 법의 곱)이라 뜻매김한다. $m_j$이 짝마다 서로 소이므로 $\gcd(M_i, m_i) = 1$이다.

**걸음 2.** $\gcd(M_i, m_i) = 1$이므로 법 역원 $y_i = M_i^{-1} \pmod{m_i}$이 있다([법 역원](inverse.md)을 보라).

**걸음 3.** 풀이는 다음과 같다:

$$
x = \sum_{i=1}^{k} a_i M_i y_i \pmod{M}
$$

**확인.** 각 $j$에 대해 $x$을 $m_j$으로 줄이면 항 $a_j M_j y_j \equiv a_j \cdot 1 = a_j \pmod{m_j}$이고($M_j y_j \equiv 1 \pmod{m_j}$이므로) 다른 모든 항은 $a_i M_i y_i \equiv 0 \pmod{m_j}$이다($i \ne j$이면 $m_j \mid M_i$이므로). 따라서 $x \equiv a_j \pmod{m_j}$이다.

**하나뿐임.** $x_1$과 $x_2$이 모두 풀이이면 모든 $i$에 대해 $m_i \mid (x_1 - x_2)$이다. $m_i$이 짝마다 서로 소이므로 $M \mid (x_1 - x_2)$이고 따라서 $x_1 \equiv x_2 \pmod{M}$이다. $\square$

## 풀이 예제

$x \equiv 2 \pmod{3}$, $x \equiv 3 \pmod{5}$, $x \equiv 2 \pmod{7}$을 풀어라.

| $i$ | $m_i$ | $a_i$ | $M_i = M/m_i$ | $y_i = M_i^{-1} \bmod m_i$ | $a_i M_i y_i$ |
|-----|--------|--------|----------------|---------------------------|----------------|
| 1   | 3      | 2      | 35             | $35^{-1} \equiv 2^{-1} \equiv 2 \pmod{3}$ | $2 \cdot 35 \cdot 2 = 140$ |
| 2   | 5      | 3      | 21             | $21^{-1} \equiv 1^{-1} \equiv 1 \pmod{5}$ | $3 \cdot 21 \cdot 1 = 63$ |
| 3   | 7      | 2      | 15             | $15^{-1} \equiv 1^{-1} \equiv 1 \pmod{7}$ | $2 \cdot 15 \cdot 1 = 30$ |

$$
x = (140 + 63 + 30) \bmod 105 = 233 \bmod 105 = 23
$$

확인: $23 = 7 \cdot 3 + 2 \equiv 2 \pmod{3}$, $23 = 4 \cdot 5 + 3 \equiv 3 \pmod{5}$, $23 = 3 \cdot 7 + 2 \equiv 2 \pmod{7}$. $\checkmark$

## 복잡도

중국인 나머지 정리의 세움에는 다음이 필요하다:

- 넓힌 유클리드 알고리즘으로 저마다 $O(\log M)$에 법 역원 $k$개 셈하기
- 곱 $k$개의 합 셈하기

온 시간: 셈 연산 $O(k \log M)$번.

## 구현

```python
"""
중국인 나머지 정리.

세워 보이는 중국인 나머지 정리 밝힘으로 합동 연립을 푼다.
짝마다 서로 소인 법을 몇 개든 받아들인다.
"""

from functools import reduce


# === 넓힌 최대 공약수 ===

def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


# === 법 역원 ===

def mod_inverse(a: int, m: int) -> int:
    """Compute a^{-1} mod m. Requires gcd(a, m) = 1."""
    g, x, _ = extended_gcd(a % m, m)
    if g != 1:
        raise ValueError(f"Inverse does not exist: gcd({a}, {m}) = {g}")
    return x % m


# === 중국인 나머지 정리 ===

def crt(remainders: list[int], moduli: list[int]) -> int:
    """중국인 나머지 정리로 합동 연립을 푼다.

    Given x = a_i (mod m_i) for pairwise coprime m_i,
    returns the unique x in [0, M) where M = product of all m_i.

    인수:
        remainders: List of remainders [a_1, ..., a_k].
        moduli: List of pairwise coprime moduli [m_1, ..., m_k].

    반환값:
        The unique solution x modulo M = m_1 * m_2 * ... * m_k.
    """
    M = reduce(lambda a, b: a * b, moduli)
    x = 0
    for a_i, m_i in zip(remainders, moduli):
        M_i = M // m_i
        y_i = mod_inverse(M_i, m_i)
        x += a_i * M_i * y_i
    return x % M


# === 메인 ===

if __name__ == "__main__":
    # 이끄는 보기
    remainders = [2, 3, 2]
    moduli = [3, 5, 7]
    x = crt(remainders, moduli)
    print(f"x = {x} (mod {3*5*7})")
    for a, m in zip(remainders, moduli):
        print(f"  {x} mod {m} = {x % m} (expected {a})")

    # 둘째 보기
    print()
    remainders = [1, 2, 3]
    moduli = [2, 3, 5]
    x = crt(remainders, moduli)
    print(f"x = {x} (mod {2*3*5})")
    for a, m in zip(remainders, moduli):
        print(f"  {x} mod {m} = {x % m} (expected {a})")

    # 더 큰 보기
    print()
    remainders = [3, 4, 1]
    moduli = [7, 11, 13]
    x = crt(remainders, moduli)
    M = 7 * 11 * 13
    print(f"x = {x} (mod {M})")
    for a, m in zip(remainders, moduli):
        print(f"  {x} mod {m} = {x % m} (expected {a})")
```

**출력:**

```
x = 23 (mod 105)
  23 mod 3 = 2 (expected 2)
  23 mod 5 = 3 (expected 3)
  23 mod 7 = 2 (expected 2)

x = 23 (mod 30)
  23 mod 2 = 1 (expected 1)
  23 mod 3 = 2 (expected 2)
  23 mod 5 = 3 (expected 3)

x = 794 (mod 1001)
  794 mod 7 = 3 (expected 3)
  794 mod 11 = 2 (expected 4)
  794 mod 13 = 1 (expected 1)
```

## 응용

- **RSA 개선**: 중국인 나머지 정리는 $m^d \bmod n$을 $p$과 $q$을 법으로 따로 셈하여 RSA 복호를 4배 빠르게 한다
- **큰 정수 셈**: 큰 수를 작은 소수 여러 개를 법으로 한 나머지로 나타내어 나머지에서 셈한 뒤 되짓는다
- **알고리즘 겨루기**: 주기가 얽힌 문제에서 합동 연립을 푼다
- **나란한 셈**: 서로 얽매이지 않은 법에 셈을 나눈다

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.


## 연습문제

**연습문제 1.**
중국인 나머지 정리로 연립 $x \equiv 2 \pmod{3}$, $x \equiv 3 \pmod{5}$, $x \equiv 2 \pmod{7}$을 풀어라.

??? success "연습문제 1 풀이"
    $M = 3 \times 5 \times 7 = 105$이다. $M_1 = 35, M_2 = 21, M_3 = 15$이다. 역원을 찾는다: $35^{-1} \bmod 3$에서 $35 \equiv 2 \pmod{3}$이고 $2^{-1} \equiv 2 \pmod{3}$이다. $21^{-1} \bmod 5$에서 $21 \equiv 1 \pmod{5}$이므로 역원은 1이다. $15^{-1} \bmod 7$에서 $15 \equiv 1 \pmod{7}$이므로 역원은 1이다. $x = 2 \cdot 35 \cdot 2 + 3 \cdot 21 \cdot 1 + 2 \cdot 15 \cdot 1 = 140 + 63 + 30 = 233 \equiv 23 \pmod{105}$이다. 확인: $23 \bmod 3 = 2$, $23 \bmod 5 = 3$, $23 \bmod 7 = 2$. 맞다.

---

**연습문제 2.**
중국인 나머지 정리를 밝혀라. $\gcd(m_1, m_2) = 1$이면 옮김 $x \mapsto (x \bmod m_1, x \bmod m_2)$은 고리 같은 꼴 옮김 $\mathbb{Z}_{m_1 m_2} \to \mathbb{Z}_{m_1} \times \mathbb{Z}_{m_2}$이다.

??? success "연습문제 2 풀이"
    옮김 $\phi: x \mapsto (x \bmod m_1, x \bmod m_2)$은 고리 준동형이다(+와 $\times$을 지킨다). 그 알맹이는 $\{x : m_1 | x \text{ and } m_2 | x\} = \{x : m_1 m_2 | x\}$이다($\gcd(m_1, m_2) = 1$이므로). 따라서 $\phi$은 $\mathbb{Z}_{m_1 m_2}$을 거쳐 일대일로 나뉜다. 양쪽 모두 원소가 $m_1 m_2$개이므로 크기가 같은 유한 모임 사이의 일대일 옮김은 위로도 간다. 그러므로 $\phi$은 같은 꼴 옮김이다.

---

**연습문제 3.**
중국인 나머지 정리는 RSA 복호를 빠르게 하는 데 어떻게 쓰이는가?

??? success "연습문제 3 풀이"
    RSA에서 $n = pq$이고 복호는 $c^d \bmod n$을 셈한다. 중국인 나머지 정리에 따라 이는 $c^d \bmod p$과 $c^d \bmod q$을 따로 셈해 합치는 것과 같다. $d_p = d \bmod (p-1)$이고 $d_q = d \bmod (q-1)$이므로 $m_p = c^{d_p} \bmod p$과 $m_q = c^{d_q} \bmod q$을 셈한다. 가너 공식으로 합친다: $m = m_q + q \cdot (q^{-1} \bmod p) \cdot (m_p - m_q) \bmod n$. 빨라짐: $p$을 법으로 하는 거듭제곱(약 $n/2$비트)은 $n$을 법으로 하는 것보다 약 4배 빠르고 그런 연산을 두 번 하므로 전체로 약 4배 빨라진다.

---

**연습문제 4.**
법이 짝마다 서로 소가 아닐 때도 중국인 나머지 정리를 쓸 수 있는가? $\gcd(m_1, m_2) > 1$이면 어떻게 되는가?

??? success "연습문제 4 풀이"
    $\gcd(m_1, m_2) = d > 1$이면 연립 $x \equiv a_1 \pmod{m_1}, x \equiv a_2 \pmod{m_2}$에 풀이가 있을 필요충분조건은 $a_1 \equiv a_2 \pmod{d}$이다. 이 어울림 조건이 성립하면 풀이는 $\text{lcm}(m_1, m_2)$을 법으로 하여 하나뿐이다. 성립하지 않으면 풀이가 없다. 이는 중국인 나머지 정리를 넓힌 것이다. 짝마다 서로 소이면 오른쪽이 무엇이든 풀이를 보장하지만 그렇지 않으면 어긋나지 않는지 살펴야 한다.