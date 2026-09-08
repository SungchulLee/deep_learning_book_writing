# 법 거듭제곱

$a^n \bmod m$을 셈하는 일은 암호(RSA 암호화와 복호), 소수 시험(밀러-라빈), 알고리즘 겨루기에 나타난다. $a$을 $n$번 곱하는 순진한 방식은 곱셈이 $O(n)$번 들어 $n$의 자릿수가 수백이면 쓸 수 없다. **되풀이 제곱**(이진 거듭제곱) 재주는 지수의 이진 나타냄을 써서 이를 곱셈 $O(\log n)$번으로 줄인다.

---

## 1. 되풀이 제곱의 생각

핵심 살핌은 제곱이 지수를 절반으로 줄인다는 것이다. $n$이 짝수이면:

$$
a^n = (a^{n/2})^2
$$

$n$이 홀수이면:

$$
a^n = a \cdot a^{n-1} = a \cdot (a^{(n-1)/2})^2
$$

이는 나누어 다스리기 셈속을 준다. $n$을 이진으로 적고 그 비트를 가장 큰 자리에서 가장 작은 자리로(또는 같은 말로 가장 작은 자리에서 가장 큰 자리로) 다룬다.

---

## 2. 알고리즘

### 오른쪽에서 왼쪽으로 가는 이진 방법

$n$의 비트를 가장 작은 자리에서 가장 큰 자리로 다룬다:

```
MODULAR-EXPONENTIATION(a, n, m):
    result = 1
    a = a mod m
    while n > 0:
        if n is odd:
            result = (result * a) mod m
        n = n >> 1            // right shift = floor division by 2
        a = (a * a) mod m     // square the base
    return result
```

### 풀이 예제

$3^{13} \bmod 7$을 셈한다. $13$의 이진 나타냄은 $1101_2$이다.

| 걸음 | 비트 | $n$ | $a$ | 결과 |
|------|-----|-----|-----|--------|
| 첫자리 | --  | 13  | 3   | 1      |
| 1    | 1   | 6   | $3^2 = 2$ | $1 \cdot 3 = 3$ |
| 2    | 0   | 3   | $2^2 = 4$ | 3 |
| 3    | 1   | 1   | $4^2 = 2$ | $3 \cdot 4 = 5$ |
| 4    | 1   | 0   | $2^2 = 4$ | $5 \cdot 2 = 3$ |

모든 값은 7을 법으로 한다. 결과: $3^{13} \equiv 3 \pmod{7}$.

확인: $3^6 = 729 = 104 \cdot 7 + 1$이므로 $3^6 \equiv 1 \pmod{7}$이다. 그러면 $3^{13} = 3^{12} \cdot 3 = (3^6)^2 \cdot 3 \equiv 1 \cdot 3 = 3 \pmod{7}$이다. $\checkmark$

---

## 3. 올바름

!!! info "되풀이 불변량"

    되풀이가 시작될 때 $n_0$을 본디 지수라 하자. 불변량은 다음과 같다:

    $$
    \text{result} \cdot a^n \equiv a_0^{n_0} \pmod{m}
    $$

    여기서 $a_0$은 본디 밑이다.

**증명.** 처음에 $\text{result} = 1$이고 $a^n = a_0^{n_0}$이므로 불변량이 성립한다.

되풀이마다:

- $n$이 홀수이면: $\text{result}' = \text{result} \cdot a$, $n' = (n-1)/2$, $a' = a^2$이다. 그러면 $\text{result}' \cdot (a')^{n'} = \text{result} \cdot a \cdot a^{2 \cdot (n-1)/2} = \text{result} \cdot a^n$이다.
- $n$이 짝수이면: $\text{result}' = \text{result}$, $n' = n/2$, $a' = a^2$이다. 그러면 $\text{result}' \cdot (a')^{n'} = \text{result} \cdot a^{2 \cdot n/2} = \text{result} \cdot a^n$이다.

$n = 0$이면 $\text{result} \cdot a^0 = \text{result} = a_0^{n_0} \bmod m$이다. $\square$

---

## 4. 복잡도

되풀이마다 (오른쪽 밀기로) $n$이 절반이 되므로 되풀이가 $\lfloor \log_2 n \rfloor + 1$번 돈다. 되풀이마다 법 곱셈을 많아야 두 번 한다. 따라서:

$$
O(\log n) \text{ modular multiplications}
$$

$m$을 법으로 하는 곱셈마다 학교 곱셈으로 $O((\log m)^2)$(또는 고속 푸리에 변환 바탕 방법으로 $O(\log m \cdot \log \log m)$)이 들면 온 비용은 다음과 같다:

$$
O(\log n \cdot (\log m)^2)
$$

!!! note "순진한 방법과 견주기"

    순진한 방식은 곱셈이 $O(n)$번 든다. $n = 2^{1000}$(흔한 RSA 지수 크기)이면 이는 천문학으로 쓸 수 없다. 되풀이 제곱은 곱셈이 약 1000번만 든다.

---

## 5. 되돌이 판

```
MODULAR-EXPONENTIATION-RECURSIVE(a, n, m):
    if n = 0:
        return 1
    if n is odd:
        return (a * MODULAR-EXPONENTIATION-RECURSIVE(a, n-1, m)) mod m
    half = MODULAR-EXPONENTIATION-RECURSIVE(a, n/2, m)
    return (half * half) mod m
```

---

## 6. 구현

```python
"""
되풀이 제곱으로 하는 법 거듭제곱.

되풀이 판과 되돌이 판을 모두 써서 곱셈 O(log n)번에
되풀이 방식과 되돌이 방식.
"""

# === 되풀이 법 거듭제곱 ===

def mod_pow(base: int, exp: int, mod: int) -> int:
    """오른쪽에서 왼쪽으로 가는 이진 방법으로 base^exp mod m을 셈한다.

    인수:
        base: 밑 정수.
        exp: 음이 아닌 지수.
        mod: 양의 법.

    반환값:
        base^exp mod m.
    """
    result = 1
    base = base % mod
    while exp > 0:
        if exp % 2 == 1:
            result = (result * base) % mod
        exp >>= 1
        base = (base * base) % mod
    return result

# === 되돌이 법 거듭제곱 ===

def mod_pow_recursive(base: int, exp: int, mod: int) -> int:
    """되돌이로 base^exp mod m을 셈한다."""
    if exp == 0:
        return 1
    if exp % 2 == 1:
        return (base * mod_pow_recursive(base, exp - 1, mod)) % mod
    half = mod_pow_recursive(base, exp // 2, mod)
    return (half * half) % mod

# === 메인 ===

if __name__ == "__main__":
    # 기본 보기
    print(f"3^13 mod 7 = {mod_pow(3, 13, 7)}")
    print(f"2^10 mod 1000 = {mod_pow(2, 10, 1000)}")
    print(f"7^256 mod 13 = {mod_pow(7, 256, 13)}")

    # 큰 지수(RSA 같은 것)
    print(f"2^1000 mod 1000000007 = {mod_pow(2, 1000, 10**9 + 7)}")

    # 파이썬 내장 함수와 견주어 확인한다
    print(f"\nVerification against pow():")
    test_cases = [(3, 13, 7), (2, 10, 1000), (7, 256, 13), (123, 456, 789)]
    for b, e, m in test_cases:
        ours = mod_pow(b, e, m)
        builtin = pow(b, e, m)
        print(f"  {b}^{e} mod {m}: ours={ours}, pow={builtin}, match={ours == builtin}")
```

**출력:**

```
3^13 mod 7 = 3
2^10 mod 1000 = 24
7^256 mod 13 = 9
2^1000 mod 1000000007 = 688423210

Verification against pow():
  3^13 mod 7: ours=3, pow=3, match=True
  2^10 mod 1000: ours=24, pow=24, match=True
  7^256 mod 13: ours=9, pow=9, match=True
  123^456 mod 789: ours=699, pow=699, match=True
```

---

## 연습문제

**연습문제 1.**
제곱하고 곱하기 알고리즘으로 $3^{13} \bmod 50$을 셈하라.

??? success "연습문제 1 풀이"
    $13 = (1101)_2$이다. $r = 1$에서 시작한다. 비트 1: $r = r^2 \cdot 3 = 3$. 비트 1: $r = 3^2 \cdot 3 = 27$. 비트 0: $r = 27^2 = 729 \equiv 29 \pmod{50}$. 비트 1: $r = 29^2 \cdot 3 = 841 \cdot 3 = 2523 \equiv 2523 - 50 \cdot 50 = 2523 - 2500 = 23$. 따라서 $3^{13} \equiv 23 \pmod{50}$이다.

---

**연습문제 2.**
이진 거듭제곱으로 법 거듭제곱 $a^b \bmod m$을 셈하는 시간 복잡도는 얼마인가?

??? success "연습문제 2 풀이"
    이진 거듭제곱은 $b$의 비트 $\lfloor \log_2 b \rfloor + 1$개를 다룬다. 비트마다 법 제곱이 한 번 들고 "1"인 비트에는 법 곱셈이 하나 더 든다. 온 횟수: 많아야 법 곱셈 $2\log_2 b$번. $< m$인 수의 법 곱셈마다 $O(\log^2 m)$ 시간이 든다(또는 고속 푸리에 변환 바탕 곱셈으로 $O(\log m \cdot \log\log m)$). 온 시간: $O(\log b \cdot \log^2 m)$.

---

**연습문제 3.**
$a^b$을 먼저 셈한 뒤 줄이는 순진한 $a^b \bmod m$ 셈하기가 쓸 수 없는 까닭을 밝혀라.

??? success "연습문제 3 풀이"
    $a^b$은 자릿수가 $b \cdot \log a$일 수 있다. $a = 2, b = 2^{256}$이면 자릿수가 $2^{256}$으로 우주의 원자 수보다 많다. 셈하기는커녕 담아 두는 것조차 불가능하다. 법 거듭제곱은 곱셈마다 줄여 중간 결과를 늘 $< m$으로 지킨다. 담기는 가장 큰 수는 (줄이기 전) $< m^2$이며 자릿수가 $O(\log m)$이다.

---

**연습문제 4.**
디피-헬먼 열쇠 주고받기 규약에서 법 거듭제곱이 어떻게 쓰이는지 적어라.

??? success "연습문제 4 풀이"
    디피-헬먼: 앨리스와 밥이 소수 $p$과 만들개 $g$을 정한다. 앨리스는 비밀 $a$을 골라 $A = g^a \bmod p$을 셈해 보낸다. 밥은 비밀 $b$을 골라 $B = g^b \bmod p$을 셈해 보낸다. 함께 쓰는 비밀: 앨리스는 $B^a = g^{ab} \bmod p$을, 밥은 $A^b = g^{ab} \bmod p$을 셈한다. 둘 다 $O(\log p)$비트 지수와 법 $p$으로 법 거듭제곱을 쓴다. 안전함은 이산 로그 문제에 기댄다. $g, p, g^a \bmod p$이 주어질 때 $a$을 찾기는 어렵다고 믿어진다.

## 정리하며

이 마당은 되풀이 제곱의 생각、알고리즘、올바름、복잡도을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
