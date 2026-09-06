# 페르마의 작은 정리

소수 $p$을 법으로 다룰 때 거듭제곱은 눈에 띄는 주기를 보인다. 0이 아닌 어떤 정수도 $p - 1$제곱하면 늘 $1$이 된다. 페르마의 작은 정리는 이 성질을 담으며 소수 시험, 효율 좋은 법 역원 셈하기, 유한체의 얼개 이론의 바탕이 된다.

## 정리의 진술

!!! info "페르마의 작은 정리"

    $p$이 소수이고 $\gcd(a, p) = 1$이면(같은 말로 $p \nmid a$이면)

    $$
    a^{p-1} \equiv 1 \pmod{p}
    $$

($p$의 배수를 담아) 모든 정수 $a$에 대해 성립하는 같은 뜻의 적기는 다음과 같다:

$$
a^p \equiv a \pmod{p}
$$

## 증명

밝힘을 둘 내놓는다. 하나는 얽음으로, 하나는 대수로 한다.

### 밝힘 1: 목걸이 세기

크기 $a$인 낱자 모임 위의 길이 $p$인 글줄 $a^p$개를 살펴보자. 돌이 같음으로 글줄을 묶는다. 하나가 다른 하나의 돌림이면 같다고 본다. 같음 무리마다 원소가 꼭 $p$개이다($p$이 소수이므로 글자가 모두 같지 않은 한 길이 $p$인 글줄에 $p$보다 작은 주기가 없다). 글자가 모두 같은 글줄은 $a$개이다. 따라서:

$$
a^p = a + p \cdot k
$$

여기서 $k$은 음이 아닌 정수이며 이는 $a^p \equiv a \pmod{p}$을 준다. $\square$

### 밝힘 2: 곱셈 무리

$p$을 법으로 하는 곱셈 아래에서 모임 $\{1, 2, \ldots, p-1\}$을 살펴보자. $p$이 소수이므로 이는 차수 $p - 1$인 무리를 이룬다. $\gcd(a, p) = 1$인 어떤 원소 $a$에 대해서도 옮김 $x \mapsto ax \bmod p$은 $\{1, 2, \ldots, p-1\}$ 위의 일대일 대응이다. 따라서:

$$
\prod_{i=1}^{p-1} (a \cdot i) \equiv \prod_{i=1}^{p-1} i \pmod{p}
$$

왼쪽은 $a^{p-1} \cdot (p-1)!$이고 오른쪽은 $(p-1)!$이다. $\gcd((p-1)!, p) = 1$이므로 $(p-1)!$을 지워 $a^{p-1} \equiv 1 \pmod{p}$을 얻는다. $\square$

## 풀어 본 보기

**보기 1.** $2^{6} \equiv 1 \pmod{7}$을 확인한다:

$$
2^6 = 64 = 9 \cdot 7 + 1 \equiv 1 \pmod{7} \quad \checkmark
$$

**보기 2.** $3^{100} \bmod 7$을 셈한다.

페르마 정리에 따라 $3^6 \equiv 1 \pmod{7}$이므로 $100 = 6 \cdot 16 + 4$이라 적는다:

$$
3^{100} = (3^6)^{16} \cdot 3^4 \equiv 1^{16} \cdot 81 \equiv 81 \bmod 7 = 4 \pmod{7}
$$

## 페르마 소수 시험

!!! warning "거꾸로는 거짓이다"

    페르마의 작은 정리의 거꾸로는 참이 **아니다**. $n$과 서로 소인 어떤(심지어 모든) 밑 $a$에 대해 $a^{n-1} \equiv 1 \pmod{n}$인 합성수 $n$이 있다.

**카마이클 수**는 $\gcd(a, n) = 1$인 모든 $a$에 대해 $a^{n-1} \equiv 1 \pmod{n}$을 만족하는 합성수 $n$이다. 가장 작은 카마이클 수는 $561 = 3 \cdot 11 \cdot 17$이다.

이 한계에도 페르마 시험은 쓸모 있는 확률 거르개가 된다:

1. 아무 밑 $a \in \{2, 3, \ldots, n-2\}$을 고른다
2. 빠른 거듭제곱으로 $a^{n-1} \bmod n$을 셈한다([법 거듭제곱](exponentiation.md)을 보라)
3. $a^{n-1} \not\equiv 1 \pmod{n}$이면 $n$은 **반드시 합성수**이다($a$은 *페르마 증인*이다)
4. $a^{n-1} \equiv 1 \pmod{n}$이면 $n$은 **아마 소수**이다

밀러-라빈 시험([밀러-라빈 시험](../primality/miller_rabin.md)을 보라)은 증인을 더 많이 찾아내어 이를 더 세게 만든다.

## 쓰임새: 법 역원

소수 $p$과 $\gcd(a, p) = 1$에 대해 페르마 정리는 곧바로 다음을 준다:

$$
a^{-1} \equiv a^{p-2} \pmod{p}
$$

이는 $a \cdot a^{p-2} = a^{p-1} \equiv 1 \pmod{p}$에서 따라 나온다. 자세한 것은 [법 역원](inverse.md)을 보라.

## 오일러 정리와의 이음

페르마의 작은 정리는 오일러 정리의 특별한 경우이다([오일러 파이 함수](totient.md)를 보라). $\gcd(a, m) = 1$인 어떤 $m$에 대해서도:

$$
a^{\varphi(m)} \equiv 1 \pmod{m}
$$

$m = p$이 소수이면 $\varphi(p) = p - 1$이 되어 페르마의 결과를 되찾는다.

## 구현

```python
"""
Fermat's Little Theorem: verification and primality testing.

수치 보기로 정리를 보여 주고
페르마 소수 시험을 짠다.
"""


# === 페르마 정리 확인 ===

def verify_fermat(a: int, p: int) -> bool:
    """Verify a^(p-1) = 1 (mod p) for prime p."""
    return pow(a, p - 1, p) == 1


# === 페르마 소수 시험 ===

def fermat_test(n: int, k: int = 10) -> str:
    """Probabilistic primality test using Fermat's little theorem.

    인수:
        n: 시험할 정수.
        k: 시험할 아무 밑의 개수.

    반환값:
        'composite' if a witness is found, 'probably prime' otherwise.
    """
    if n < 2:
        return "composite"
    if n <= 3:
        return "probably prime"

    import random
    for _ in range(k):
        a = random.randint(2, n - 2)
        if pow(a, n - 1, n) != 1:
            return "composite"
    return "probably prime"


# === 메인 ===

if __name__ == "__main__":
    # p = 7에서 정리를 확인한다
    p = 7
    print(f"Fermat's theorem verification for p = {p}:")
    for a in range(1, p):
        result = pow(a, p - 1, p)
        print(f"  {a}^{p-1} mod {p} = {result}")

    # 3^100 mod 7을 셈한다
    print(f"\n3^100 mod 7 = {pow(3, 100, 7)}")

    # 소수 시험
    print("\nFermat primality test:")
    test_numbers = [7, 13, 15, 561, 1009, 1729]
    for n in test_numbers:
        result = fermat_test(n, k=20)
        print(f"  {n}: {result}")

    # 카마이클 수 561을 보인다
    print(f"\nCarmichael number 561 = 3 * 11 * 17:")
    all_pass = all(pow(a, 560, 561) == 1
                   for a in range(2, 561) if pow(a, 1, 561) != 0)
    print(f"  All coprime bases pass: {all_pass}")
```

**출력:**

```
Fermat's theorem verification for p = 7:
  1^6 mod 7 = 1
  2^6 mod 7 = 1
  3^6 mod 7 = 1
  4^6 mod 7 = 1
  5^6 mod 7 = 1
  6^6 mod 7 = 1

3^100 mod 7 = 4

Fermat primality test:
  7: 아마 소수
  13: 아마 소수
  15: 합성수
  561: 아마 소수
  1009: 아마 소수
  1729: 아마 소수

Carmichael number 561 = 3 * 11 * 17:
  서로 소인 밑이 모두 지나간다: True
```

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.


## 연습문제

**연습문제 1.**
$a = 3, p = 7$에서 페르마의 작은 정리를 확인하라. $3^6 \equiv 1 \pmod{7}$인지 살펴라.

??? success "연습문제 1 풀이"
    $3^1 = 3, 3^2 = 9 \equiv 2, 3^3 = 6, 3^4 = 18 \equiv 4, 3^5 = 12 \equiv 5, 3^6 = 15 \equiv 1 \pmod{7}$이다. 확인되었다: $3^{7-1} = 3^6 \equiv 1 \pmod{7}$.

---

**연습문제 2.**
페르마의 작은 정리로 $7^{222} \bmod 11$을 셈하라.

??? success "연습문제 2 풀이"
    페르마에 따라 $7^{10} \equiv 1 \pmod{11}$이다. $222 = 22 \cdot 10 + 2$이다. 따라서 $7^{222} = (7^{10})^{22} \cdot 7^2 \equiv 1^{22} \cdot 49 \equiv 49 \bmod 11 = 5$이다.

---

**연습문제 3.**
페르마의 작은 정리가 소수 시험이 되지 못하는 까닭을 밝혀라. 카마이클 수란 무엇인가?

??? success "연습문제 3 풀이"
    페르마는 말한다. $p$이 소수이면 $\gcd(a,p) = 1$인 $a$에 대해 $a^{p-1} \equiv 1 \pmod{p}$이다. 거꾸로는 거짓이다. 어떤 합성수 $n$은 $\gcd(a,n) = 1$인 모든 $a$에 대해 $a^{n-1} \equiv 1 \pmod{n}$을 만족한다. 이들이 카마이클 수이다. 가장 작은 것은 561 = $3 \times 11 \times 17$이다. $\gcd(a, 561) = 1$인 어떤 $a$에 대해서도 $a^{560} \equiv 1 \pmod{561}$이다. 페르마 시험은 어느 밑을 시험해도 561을 "아마 소수"라고 잘못 알린다. 밀러-라빈은 제곱근 조건도 살펴 이를 피한다.

---

**연습문제 4.**
곱셈 무리 따짐으로 페르마의 작은 정리를 밝혀라.

??? success "연습문제 4 풀이"
    $p$을 법으로 하는 모임 $S = \{1, 2, \ldots, p-1\}$을 살펴보자. $\gcd(a, p) = 1$이면 옮김 $x \mapsto ax \bmod p$은 $S$ 위의 일대일 대응이다($a$이 $p$을 법으로 역원을 가지므로). 따라서 $\{a \cdot 1, a \cdot 2, \ldots, a \cdot (p-1)\} = \{1, 2, \ldots, p-1\} \pmod{p}$이다. 양쪽의 곱을 잡으면 $a^{p-1} \cdot (p-1)! \equiv (p-1)! \pmod{p}$이다. $\gcd((p-1)!, p) = 1$이므로 양쪽을 $(p-1)!$으로 나누어 $a^{p-1} \equiv 1 \pmod{p}$을 얻는다.