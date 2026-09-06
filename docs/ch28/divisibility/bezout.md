# 베주 항등식

유클리드 알고리즘은 두 정수의 최대 공약수를 효율 좋게 셈할 수 있음을 보인다. 자연스러운 다음 물음은 그 최대 공약수를 본디 정수의 결합으로 *적을* 수 있느냐이다. 베주 항등식은 그렇다고 답한다. 어떤 정수 $a$과 $b$에 대해서도 $ax + by = \gcd(a, b)$인 정수 $x$과 $y$이 있다. 이 나타냄은 법 역원 셈하기, 선형 디오판토스 방정식 풀기, 수론의 근본 결과 밝히기의 바탕이다.

## 정리의 진술

!!! info "베주 항등식"

    둘 다 0은 아닌 어떤 정수 $a$과 $b$에 대해서도 다음을 만족하는 정수 $x, y \in \mathbb{Z}$이 있다

    $$
    ax + by = \gcd(a, b)
    $$

    정수 $x$과 $y$을 **베주 계수**라 부른다.

계수 $x$과 $y$은 하나뿐이 아니다. $(x_0, y_0)$이 한 풀이이면 모든 풀이는 다음 꼴이다:

$$
x = x_0 + k \cdot \frac{b}{\gcd(a, b)}, \quad y = y_0 - k \cdot \frac{a}{\gcd(a, b)}
$$

여기서 $k$은 아무 정수이다.

## 증명

이 밝힘은 모임 $S = \{ax + by : x, y \in \mathbb{Z}\} \cap \mathbb{Z}^{+}$의 가장 작은 양의 원소가 $\gcd(a, b)$과 같음을 보여 나아간다.

**걸음 1.** $a \cdot a + b \cdot 0 = a^2 > 0$이므로($b$을 써도 마찬가지이다) 모임 $S$은 비어 있지 않다. $d$을 $S$의 가장 작은 양의 원소라 하면 어떤 정수 $x_0, y_0$에 대해 $d = ax_0 + by_0$이다.

**걸음 2.** $d \mid a$임을 보인다. 나눗셈 알고리즘으로 $0 \le r < d$인 $a = qd + r$이라 적는다. 그러면:

$$
r = a - qd = a - q(ax_0 + by_0) = a(1 - qx_0) + b(-qy_0)
$$

따라서 $r \in S \cup \{0\}$이다. $0 \le r < d$이고 $d$이 $S$의 가장 작은 양의 원소이므로 $r = 0$이어야 한다. 그러므로 $d \mid a$이다. 같은 따짐으로 $d \mid b$이다.

**걸음 3.** $d \mid a$이고 $d \mid b$이므로 $d \le \gcd(a, b)$이다. 그런데 $\gcd(a, b)$은 $a$과 $b$을 모두 나누므로 $S$의 모든 원소를 나누어 $\gcd(a, b) \mid d$이고 따라서 $\gcd(a, b) \le d$이다. 합치면 $d = \gcd(a, b)$이다. $\square$

## 풀이 예제

$a = 48$이고 $b = 18$이면 $\gcd(48, 18) = 6$임을 안다. 베주 항등식은 $48x + 18y = 6$인 정수 $x, y$이 있음을 보장한다.

유클리드 알고리즘을 거슬러 좇으면:

| 걸음 | 식 |
|------|----------|
| 1 | $48 = 2 \cdot 18 + 12$ |
| 2 | $18 = 1 \cdot 12 + 6$ |
| 3 | $12 = 2 \cdot 6 + 0$ |

걸음 2에서 거꾸로 넣으면:

$$
6 = 18 - 1 \cdot 12
$$

걸음 1의 $12 = 48 - 2 \cdot 18$을 넣으면:

$$
6 = 18 - 1 \cdot (48 - 2 \cdot 18) = 3 \cdot 18 - 1 \cdot 48
$$

따라서 $x = -1$과 $y = 3$이 $48(-1) + 18(3) = 6$을 만족한다. 넓힌 유클리드 알고리즘이 이 거꾸로 넣기 과정을 저절로 해 준다([넓힌 유클리드](extended.md)를 보라).

## 서로 소임의 성격 매김

베주 항등식은 서로 소임을 아름답게 성격 짓는다.

!!! info "서로 소 잣대"

    정수 $a$과 $b$이 서로 소일(곧 $\gcd(a, b) = 1$일) 필요충분조건은 다음을 만족하는 정수 $x, y$이 있는 것이다

    $$
    ax + by = 1
    $$

**증명.** $\gcd(a, b) = 1$이면 베주 항등식이 바라는 $x, y$을 준다. 거꾸로 $ax + by = 1$이고 $d = \gcd(a, b)$이면 $d \mid (ax + by) = 1$이므로 $d = 1$이다. $\square$

## 선형 디오판토스 방정식

베주 항등식은 $ax + by = c$ 꼴의 방정식을 푸는 데로 넓혀진다.

!!! info "선형 디오판토스 방정식의 풀 수 있음"

    방정식 $ax + by = c$에 정수 풀이가 있을 필요충분조건은 $\gcd(a, b) \mid c$인 것이다.

**증명.** $d = \gcd(a, b) \mid c$이면 $c = d \cdot k$이라 적는다. 베주 항등식에 따라 $d = ax_0 + by_0$이므로 $c = a(kx_0) + b(ky_0)$이다. 거꾸로 $ax + by = c$이면 $d \mid a$과 $d \mid b$에서 $d \mid c$이 따라 나온다. $\square$

## 구현

```python
"""
Bezout's Identity verification.

gcd(a, b)을 늘 선형 결합 ax + by으로 적을 수 있음을 보여 주고
그리고 보기 몇 가지로 결과를 확인한다.
"""

import math


# === 베주 계수를 위한 넓힌 최대 공약수 ===

def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Return (g, x, y) such that a*x + b*y = g = gcd(a, b)."""
    if b == 0:
        return a, 1, 0
    g, x, y = extended_gcd(b, a % b)
    return g, y, x - (a // b) * y


# === 확인 ===

def verify_bezout(a: int, b: int) -> None:
    """Verify Bezout's identity for given a and b."""
    g, x, y = extended_gcd(a, b)
    assert a * x + b * y == g, "Bezout identity failed"
    print(f"gcd({a}, {b}) = {g},  {a}*({x}) + {b}*({y}) = {a*x + b*y}")


# === 메인 ===

if __name__ == "__main__":
    verify_bezout(48, 18)
    verify_bezout(270, 192)
    verify_bezout(17, 13)
    verify_bezout(35, 15)
    verify_bezout(100, 1)

    # 서로 소인지 살피기
    a, b = 17, 13
    g, x, y = extended_gcd(a, b)
    print(f"\n{a} and {b} are coprime: {g == 1}")
    print(f"Certificate: {a}*({x}) + {b}*({y}) = {a*x + b*y}")
```

**출력:**

```
gcd(48, 18) = 6,  48*(-1) + 18*(3) = 6
gcd(270, 192) = 6,  270*(-5) + 192*(7) = 6
gcd(17, 13) = 1,  17*(-3) + 13*(4) = 1
gcd(35, 15) = 5,  35*(1) + 15*(-2) = 5
gcd(100, 1) = 1,  100*(0) + 1*(1) = 1

17과 13은 서로 소이다: True
Certificate: 17*(-3) + 13*(4) = 1
```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.

## 연습문제

**연습문제 1.**
넓힌 유클리드 알고리즘으로 $56x + 15y = 1$인 정수 $x, y$을 찾아라.

??? success "연습문제 1 풀이"
    유클리드 알고리즘을 쓴다: $56 = 3 \cdot 15 + 11$, $15 = 1 \cdot 11 + 4$, $11 = 2 \cdot 4 + 3$, $4 = 1 \cdot 3 + 1$, $3 = 3 \cdot 1$. 거꾸로 넣는다: $1 = 4 - 1 \cdot 3 = 4 - 1(11 - 2 \cdot 4) = 3 \cdot 4 - 11 = 3(15 - 11) - 11 = 3 \cdot 15 - 4 \cdot 11 = 3 \cdot 15 - 4(56 - 3 \cdot 15) = 15 \cdot 15 - 4 \cdot 56$. 따라서 $x = -4, y = 15$이다: $56(-4) + 15(15) = -224 + 225 = 1$.

---

**연습문제 2.**
$\gcd(a, b) = 1$일 필요충분조건이 $ax + by = 1$인 정수 $x, y$이 있는 것임을 밝혀라.

??? success "연습문제 2 풀이"
    $(\Rightarrow)$: $\gcd(a,b) = 1$이면 베주 항등식이 $ax + by = \gcd(a,b) = 1$인 정수 $x, y$을 보장한다. $(\Leftarrow)$: 어떤 정수 $x, y$에 대해 $ax + by = 1$이면 $d = \gcd(a,b)$이라 하자. 그러면 $d | a$이고 $d | b$이므로 $d | (ax + by) = 1$이다. $d \geq 1$이므로 $d = 1$이다.

---

**연습문제 3.**
$12x + 8y = 20$의 모든 정수 풀이를 구하라.

??? success "연습문제 3 풀이"
    먼저 $\gcd(12, 8) = 4$이고 $4 | 20$이므로 풀이가 있다. 4으로 나누면 $3x + 2y = 5$이다. 특별한 풀이 하나: $x_0 = 1, y_0 = 1$이다($3 + 2 = 5$이므로). 일반 풀이는 아무 정수 $t$에 대해 $x = 1 + 2t, y = 1 - 3t$이다. 확인: $3(1 + 2t) + 2(1 - 3t) = 3 + 6t + 2 - 6t = 5$. 본디 방정식에서도 $x = 1 + 2t, y = 1 - 3t$이다.

---

**연습문제 4.**
베주 항등식이 정수 셋 이상으로 넓혀짐을 보여라. 어떤 $a_1, \ldots, a_n$에 대해서도 $\sum a_i x_i = \gcd(a_1, \ldots, a_n)$인 정수 $x_1, \ldots, x_n$이 있다.

??? success "연습문제 4 풀이"
    $n$에 대한 귀납으로 밝힌다. 바탕 경우 $n = 2$: 베주 항등식이다. $n > 2$: $d = \gcd(a_1, \ldots, a_{n-1})$이라 하자. 귀납에 따라 어떤 정수 $x_i'$에 대해 $d = \sum_{i=1}^{n-1} a_i x_i'$이다. $g = \gcd(d, a_n) = \gcd(a_1, \ldots, a_n)$이라 하자. 변수 둘의 경우에 따라 정수 $s, t$에 대해 $g = d \cdot s + a_n \cdot t$이다. 그러면 $g = s \sum_{i=1}^{n-1} a_i x_i' + a_n t = \sum_{i=1}^{n-1} a_i (s x_i') + a_n t$이다. $i < n$에 $x_i = s x_i'$, $x_n = t$으로 두면 밝힘이 끝난다.
