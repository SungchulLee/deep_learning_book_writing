# 법 연산

암호 규약, 해시 함수, 알고리즘 겨루기 문제는 모두 나머지만 중요한 셈을 한다. 법 셈은 수가 법에 이르면 되감기는 "시계 셈"의 엄밀한 틀을 준다. 이 쪽은 합동 관계와 $m$을 법으로 하는 덧셈, 뺄셈, 곱셈의 규칙을 세운다.

---

## 1. 합동 관계

양의 정수 $m$에 대해 $a$이 $m$을 법으로 $b$과 **합동**이라 하고 다음과 같이 적는다:

$$
a \equiv b \pmod{m}
$$

이는 $m \mid (a - b)$, 곧 $m$이 차 $a - b$을 나눌 때에만 그렇다. 같은 말로 $a$과 $b$을 $m$으로 나눈 나머지가 같다.

!!! info "같음 관계로서의 합동"

    $m$을 법으로 하는 합동은 $\mathbb{Z}$ 위의 같음 관계이다. 되비침($a \equiv a$), 맞섬($a \equiv b \Rightarrow b \equiv a$), 옮김($a \equiv b$이고 $b \equiv c \Rightarrow a \equiv c$)을 만족한다. 그 같음 무리를 $m$을 법으로 하는 **나머지 무리**라 부른다.

$m$을 법으로 하는 나머지 무리의 모임을 $\mathbb{Z}/m\mathbb{Z} = \{0, 1, 2, \ldots, m-1\}$이라 적으며 원소마다 그 온 같음 무리를 나타낸다.

---

## 2. 덧셈

$a \equiv a' \pmod{m}$이고 $b \equiv b' \pmod{m}$이면:

$$
a + b \equiv a' + b' \pmod{m}
$$

**증명.** 가정에 따라 $m \mid (a - a')$이고 $m \mid (b - b')$이다. 더하면 $m \mid ((a + b) - (a' + b'))$이므로 $a + b \equiv a' + b' \pmod{m}$이다. $\square$

!!! example "법 덧셈"

    $m = 7$일 때 $15 \equiv 1 \pmod{7}$이고 $20 \equiv 6 \pmod{7}$이므로,

    $$
    15 + 20 = 35 \equiv 0 \pmod{7}
    $$

    실제로 $1 + 6 = 7 \equiv 0 \pmod{7}$이다. $\checkmark$

---

## 3. 뺄셈

$a \equiv a' \pmod{m}$이고 $b \equiv b' \pmod{m}$이면:

$$
a - b \equiv a' - b' \pmod{m}
$$

밝힘은 $+$을 $-$으로 바꾼 것 말고는 덧셈과 같다.

---

## 4. 곱셈

$a \equiv a' \pmod{m}$이고 $b \equiv b' \pmod{m}$이면:

$$
a \cdot b \equiv a' \cdot b' \pmod{m}
$$

**증명.** 정수 $k, l$에 대해 $a = a' + km$이고 $b = b' + lm$이라 적는다. 그러면:

$$
ab = a'b' + a'lm + b'km + klm^2 = a'b' + m(a'l + b'k + klm)
$$

따라서 $m \mid (ab - a'b')$이고 $ab \equiv a'b' \pmod{m}$이다. $\square$

!!! tip "실제로 따르는 것"

    이 성질 덕에 셈하는 동안 어느 때든 중간 결과를 $m$을 법으로 줄일 수 있다. $(a \cdot b) \bmod m$을 셈할 때 $a$과 $b$을 먼저 $m$을 법으로 줄이고 곱한 뒤 다시 줄일 수 있다. 이는 짤 때 정수 넘침을 막는다.

---

## 5. 거듭제곱

곱셈 규칙을 되풀이해 쓰면 다음을 얻는다:

$$
a^n \equiv (a \bmod m)^n \pmod{m}
$$

$a^n \bmod m$을 효율 좋게 셈하는 법은 [법 거듭제곱](exponentiation.md)을 보라.

---

## 6. 나눗셈(조심)

!!! warning "나눗셈이 늘 통하지는 않는다"

    덧셈이나 곱셈과 달리 법 셈에서 **나눗셈은 늘 올바르지는 않다**. $ac \equiv bc \pmod{m}$에서 일반으로 $a \equiv b \pmod{m}$을 **이끌어 낼 수 없다**.

    보기로 $2 \cdot 3 \equiv 2 \cdot 6 \pmod{6}$이지만(둘 다 $12 \equiv 0$이다) $3 \not\equiv 6 \pmod{6}$이다.

나눗셈은 $\gcd(c, m) = 1$일 때만 올바르다. 그때 $c$은 법 역원 $c^{-1}$을 가지며 양쪽에 $c^{-1}$을 곱해 "나눌" 수 있다([법 역원](inverse.md)을 보라).

---

## 7. 성질 간추림

어떤 정수 $a, b, c$과 양의 정수 $m$에 대해서도:

| 성질 | 글월 |
|----------|-----------|
| 닫힘 | $(a + b) \bmod m \in \{0, \ldots, m-1\}$ |
| 자리 바꿈 | $a + b \equiv b + a$, $a \cdot b \equiv b \cdot a$ |
| 묶음 바꿈 | $(a + b) + c \equiv a + (b + c)$, $(ab)c \equiv a(bc)$ |
| 나눠 곱하기 | $a(b + c) \equiv ab + ac$ |
| 항등원 | $a + 0 \equiv a$, $a \cdot 1 \equiv a$ |
| 덧셈 역원 | $a + (m - a) \equiv 0$ |

이 성질들이 $(\mathbb{Z}/m\mathbb{Z}, +, \cdot)$을 **가환환**으로 만든다. $m$이 소수이면 0이 아닌 모든 원소가 곱셈 역원을 가져 **체**가 된다.

---

## 8. 구현

```python
"""
법 셈 연산.

덧셈, 뺄셈, 곱셈, 거듭제곱을 보여 준다
넘침에 안전한 중간 줄이기로 m을 법으로 한다.
"""

# === 법 연산 ===

def mod_add(a: int, b: int, m: int) -> int:
    """(a + b) mod m을 셈한다."""
    return ((a % m) + (b % m)) % m

def mod_sub(a: int, b: int, m: int) -> int:
    """(a - b) mod m을 셈하되 결과가 음이 아니게 한다."""
    return ((a % m) - (b % m) + m) % m

def mod_mul(a: int, b: int, m: int) -> int:
    """(a * b) mod m을 셈한다."""
    return ((a % m) * (b % m)) % m

def mod_pow(base: int, exp: int, m: int) -> int:
    """되풀이 제곱으로 base^exp mod m을 셈한다."""
    result = 1
    base = base % m
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % m
        exp //= 2
        base = (base * base) % m
    return result

# === 메인 ===

if __name__ == "__main__":
    m = 7
    print(f"Modular arithmetic with m = {m}")
    print(f"  (15 + 20) mod {m} = {mod_add(15, 20, m)}")
    print(f"  (15 - 20) mod {m} = {mod_sub(15, 20, m)}")
    print(f"  (15 * 20) mod {m} = {mod_mul(15, 20, m)}")
    print(f"  3^10 mod {m} = {mod_pow(3, 10, m)}")

    # 어긋나지 않는지 확인한다
    print(f"\nVerification:")
    print(f"  (15 + 20) = {15 + 20}, {35 % m} = {mod_add(15, 20, m)}")
    print(f"  (15 * 20) = {15 * 20}, {300 % m} = {mod_mul(15, 20, m)}")
```

**출력:**

```
Modular arithmetic with m = 7
  (15 + 20) mod 7 = 0
  (15 - 20) mod 7 = 2
  (15 * 20) mod 7 = 6
  3^10 mod 7 = 4

Verification:
  (15 + 20) = 35, 0 = 0
  (15 * 20) = 300, 6 = 6
```

---

## 연습문제

**연습문제 1.**
$(37 + 89) \bmod 13$과 $37 \bmod 13 + 89 \bmod 13 \bmod 13$을 셈하라. 둘이 같음을 확인하라.

??? success "연습문제 1 풀이"
    $(37 + 89) \bmod 13 = 126 \bmod 13 = 126 - 9 \cdot 13 = 126 - 117 = 9$이다. $37 \bmod 13 = 11, 89 \bmod 13 = 89 - 6 \cdot 13 = 11$이다. $(11 + 11) \bmod 13 = 22 \bmod 13 = 9$이다. 같다. 이는 $(a + b) \bmod m = ((a \bmod m) + (b \bmod m)) \bmod m$임을 확인해 준다.

---

**연습문제 2.**
$(a \cdot b) \bmod m = ((a \bmod m) \cdot (b \bmod m)) \bmod m$임을 밝혀라.

??? success "연습문제 2 풀이"
    $r_1 = a \bmod m, r_2 = b \bmod m$일 때 $a = q_1 m + r_1$이고 $b = q_2 m + r_2$이라 적는다. 그러면 $ab = (q_1 m + r_1)(q_2 m + r_2) = q_1 q_2 m^2 + q_1 r_2 m + q_2 r_1 m + r_1 r_2$이다. $m$을 법으로 하면 $ab \equiv r_1 r_2 \pmod{m}$이다. 따라서 $ab \bmod m = r_1 r_2 \bmod m = (a \bmod m)(b \bmod m) \bmod m$이다.

---

**연습문제 3.**
프로그래밍에서 법 뺄셈에 조심해야 하는 까닭은 무엇인가? $a < b$일 때 $(a - b) \bmod m$을 옳게 셈하는 법을 보여라.

??? success "연습문제 3 풀이"
    대부분의 프로그래밍 말에서 $a < b$이면 $a - b$이 음수이고 $\%$ 연산자가 음수를 돌려줄 수도(C와 C++) 양수를 돌려줄 수도(파이썬) 있다. 음이 아닌 결과를 보장하려면 $(a - b) \bmod m = ((a - b) \% m + m) \% m$으로 한다. 보기: $(3 - 7) \bmod 5 = -4 \bmod 5$이다. 파이썬에서는 $-4 \% 5 = 1$이다. C에서는 $-4 \% 5 = -4$이므로 $((-4 \% 5) + 5) \% 5 = 1$이다.

---

**연습문제 4.**
법 셈이 알고리즘 겨루기에 꼭 필요한 까닭과 넘침을 어떻게 피하는지 밝혀라.

??? success "연습문제 4 풀이"
    많은 문제가 소수(보통 $10^9 + 7$)를 법으로 결과를 셈하라고 한다. 까닭은 (1) 정확한 답이 자릿수가 수백만일 수 있고($n!$, 카탈랑 수), (2) 법으로 줄인 결과는 64비트 정수에 들어가며, (3) 소수 법이 나눗셈을 위한 법 역원을 가능하게 하기 때문이다. 넘침을 피하려면 연산마다 줄인다. $< m = 10^9 + 7$인 두 수의 곱은 $< 10^{18}$이며 이는 64비트 정수에 들어간다(최대 약 $9.2 \times 10^{18}$). 덧셈에서는 합이 $< 2m < 2 \times 10^9$이므로 64비트에서 안전하다.

## 정리하며

이 마당은 합동 관계、덧셈、뺄셈、곱셈을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
