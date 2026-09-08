# 오일러 파이 함수

페르마의 작은 정리는 $p$이 소수일 때 $a^{p-1} \equiv 1 \pmod{p}$임을 말한다. 그런데 법이 합성수이면 어떻게 되는가? 오일러 파이 함수 $\varphi(n)$은 $n$보다 작은 정수 가운데 $n$과 서로 소인 것의 개수를 세며, 오일러 정리는 페르마의 결과를 아무 법으로 넓힌다: $a^{\varphi(n)} \equiv 1 \pmod{n}$. 이 넓힘은 법 $n = pq$이 소수 둘의 곱인 RSA 암호의 한가운데에 있다.

---

## 1. 정의

**오일러 파이 함수** $\varphi(n)$은 $\{1, 2, \ldots, n\}$ 가운데 $n$과 서로 소인 정수의 개수를 센다:

$$
\varphi(n) = |\{k : 1 \le k \le n, \gcd(k, n) = 1\}|
$$

!!! example "작은 값"

    | $n$ | $n$과 서로 소인 정수 | $\varphi(n)$ |
    |-----|------------------------|-------------|
    | 1   | $\{1\}$ | 1 |
    | 6   | $\{1, 5\}$ | 2 |
    | 8   | $\{1, 3, 5, 7\}$ | 4 |
    | 12  | $\{1, 5, 7, 11\}$ | 4 |

---

## 2. 파이 함수 셈하기

### 소수 거듭제곱

소수 $p$에서는 $1$부터 $p - 1$까지 모든 정수가 $p$과 서로 소이다:

$$
\varphi(p) = p - 1
$$

소수 거듭제곱 $p^k$에서 $p^k$과 서로 소가 아닌 정수는 바로 $p$의 배수이며 그 개수는 $p^{k-1}$이다:

$$
\varphi(p^k) = p^k - p^{k-1} = p^{k-1}(p - 1) = p^k\left(1 - \frac{1}{p}\right)
$$

### 곱셈 성질

파이 함수는 **곱셈 성질**을 가진다. $\gcd(m, n) = 1$이면:

$$
\varphi(mn) = \varphi(m) \cdot \varphi(n)
$$

**증명 얼개.** 중국인 나머지 정리([중국인 나머지 정리](crt.md)를 보라)에 따라 옮김 $k \mapsto (k \bmod m, k \bmod n)$은 $\mathbb{Z}/mn\mathbb{Z}$에서 $\mathbb{Z}/m\mathbb{Z} \times \mathbb{Z}/n\mathbb{Z}$으로 가는 일대일 대응이다. 이 대응 아래에서 $\gcd(k, mn) = 1$일 필요충분조건은 $\gcd(k, m) = 1$이고 $\gcd(k, n) = 1$인 것이다. $\square$

### 일반 공식

소수 거듭제곱과 곱셈 성질의 결과를 합치면 $n = p_1^{a_1} p_2^{a_2} \cdots p_k^{a_k}$에 대해:

$$
\varphi(n) = n \prod_{p \mid n} \left(1 - \frac{1}{p}\right) = n \cdot \frac{p_1 - 1}{p_1} \cdot \frac{p_2 - 1}{p_2} \cdots \frac{p_k - 1}{p_k}
$$

!!! example "파이 함수 셈하기"

    $\varphi(12) = 12 \cdot (1 - 1/2) \cdot (1 - 1/3) = 12 \cdot 1/2 \cdot 2/3 = 4$이다.

    $\varphi(100) = 100 \cdot (1 - 1/2) \cdot (1 - 1/5) = 100 \cdot 1/2 \cdot 4/5 = 40$이다.

---

## 3. 오일러 정리

!!! info "오일러 정리"

    $\gcd(a, n) = 1$이면

    $$
    a^{\varphi(n)} \equiv 1 \pmod{n}
    $$

**증명.** $r_1, r_2, \ldots, r_{\varphi(n)}$을 $\{1, \ldots, n\}$ 가운데 $n$과 서로 소인 정수라 하자. $\gcd(a, n) = 1$이므로 곱 $ar_1, ar_2, \ldots, ar_{\varphi(n)}$은 $n$을 법으로 $r_1, r_2, \ldots, r_{\varphi(n)}$의 자리바꿈이다. 양쪽의 곱을 잡으면:

$$
a^{\varphi(n)} \cdot \prod_{i=1}^{\varphi(n)} r_i \equiv \prod_{i=1}^{\varphi(n)} r_i \pmod{n}
$$

각 $r_i$이 $n$과 서로 소이므로 그 곱도 $n$과 서로 소여서 지울 수 있고 $a^{\varphi(n)} \equiv 1 \pmod{n}$을 얻는다. $\square$

$n = p$이 소수이면 $\varphi(p) = p - 1$이 되어 페르마의 작은 정리를 되찾는다([페르마의 작은 정리](fermat.md)를 보라).

---

## 4. 가우스 약수 합

근본 항등식이 파이 함수를 약수 합과 잇는다.

!!! info "가우스 공식"

    어떤 양의 정수 $n$에 대해서도,

    $$
    \sum_{d \mid n} \varphi(d) = n
    $$

**증명.** $\{1, 2, \ldots, n\}$을 $\gcd(k, n)$으로 가른다. $n$의 약수 $d$마다 $\gcd(k, n) = d$인 정수 $k \in \{1, \ldots, n\}$의 개수는 $\varphi(n/d)$이다. 모든 약수에 걸쳐 더하면 $\sum_{d \mid n} \varphi(n/d) = n$이고, 합이 같은 약수 모임을 훑으므로 이는 $\sum_{d \mid n} \varphi(d)$과 같다. $\square$

---

## 5. RSA에서의 쓰임새

RSA에서 법은 서로 다른 소수 $p, q$에 대해 $n = pq$이다. 파이 값은 다음과 같다:

$$
\varphi(n) = (p-1)(q-1)
$$

공개 열쇠 지수 $e$은 $\varphi(n)$과 서로 소가 되게 고르고 개인 열쇠는 $d = e^{-1} \bmod \varphi(n)$이다. 오일러 정리가 $n$과 서로 소인 글월 $m$에 대해 $m^{ed} \equiv m \pmod{n}$임을 보장하여 복호를 가능하게 한다.

---

## 6. 구현

```python
"""
오일러 파이 함수와 오일러 정리.

소인수 분해 공식으로 파이 함수를 셈한다
그리고 수치 보기로 오일러 정리를 따져 본다.
"""

# === 파이 함수 ===

def euler_totient(n: int) -> int:
    """오일러 파이 함수 phi(n)을 셈한다.

    모든 소인수를 찾아 곱 공식을 쓴다.

    인수:
        n: 양의 정수.

    반환값:
        phi(n), 곧 [1, n] 안에서 n과 서로 소인 정수의 개수.
    """
    result = n
    p = 2
    temp = n
    while p * p <= temp:
        if temp % p == 0:
            while temp % p == 0:
                temp //= p
            result -= result // p
        p += 1
    if temp > 1:
        result -= result // temp
    return result

# === 막무가내 확인 ===

def euler_totient_brute(n: int) -> int:
    """서로 소인 수를 곧바로 세어 phi(n)을 셈한다."""
    from math import gcd
    return sum(1 for k in range(1, n + 1) if gcd(k, n) == 1)

# === 메인 ===

if __name__ == "__main__":
    # 작은 값의 파이 함수를 셈한다
    print("Euler's totient function:")
    for n in [1, 2, 6, 8, 10, 12, 100]:
        phi = euler_totient(n)
        phi_brute = euler_totient_brute(n)
        print(f"  phi({n}) = {phi}  (brute force: {phi_brute})")

    # 오일러 정리를 확인한다
    print("\nEuler's theorem verification:")
    test_cases = [(3, 10), (7, 12), (11, 15)]
    for a, n in test_cases:
        phi = euler_totient(n)
        result = pow(a, phi, n)
        print(f"  {a}^phi({n}) mod {n} = {a}^{phi} mod {n} = {result}")

    # 가우스 약수 합
    print("\nGauss divisor sum:")
    for n in [6, 12, 20]:
        divisors = [d for d in range(1, n + 1) if n % d == 0]
        total = sum(euler_totient(d) for d in divisors)
        print(f"  sum(phi(d) for d | {n}) = {total} = {n}")
```

**출력:**

```
Euler's totient function:
  phi(1) = 1  (brute force: 1)
  phi(2) = 1  (brute force: 1)
  phi(6) = 2  (brute force: 2)
  phi(8) = 4  (brute force: 4)
  phi(10) = 4  (brute force: 4)
  phi(12) = 4  (brute force: 4)
  phi(100) = 40  (brute force: 40)

Euler's theorem verification:
  3^phi(10) mod 10 = 3^4 mod 10 = 1
  7^phi(12) mod 12 = 7^4 mod 12 = 1
  11^phi(15) mod 15 = 11^8 mod 15 = 1

Gauss divisor sum:
  sum(phi(d) for d | 6) = 6 = 6
  sum(phi(d) for d | 12) = 12 = 12
  sum(phi(d) for d | 20) = 20 = 20
```

---

## 연습문제

**연습문제 1.**
공식 $\phi(n) = n \prod_{p | n}(1 - 1/p)$으로 $\phi(60)$을 셈하라.

??? success "연습문제 1 풀이"
    $60 = 2^2 \cdot 3 \cdot 5$이다. $\phi(60) = 60 \cdot (1 - 1/2)(1 - 1/3)(1 - 1/5) = 60 \cdot 1/2 \cdot 2/3 \cdot 4/5 = 60 \cdot 8/30 = 16$이다.

---

**연습문제 2.**
오일러 정리를 밝혀라: $\gcd(a, m) = 1$이면 $a^{\phi(m)} \equiv 1 \pmod{m}$이다.

??? success "연습문제 2 풀이"
    $r_1, \ldots, r_{\phi(m)}$을 $\{1, \ldots, m\}$ 가운데 $m$과 서로 소인 정수라 하자. $\gcd(a, m) = 1$이므로 옮김 $r_i \mapsto ar_i \bmod m$은 이 모임 위의 일대일 대응이다. 따라서 $\{ar_1, \ldots, ar_{\phi(m)}\} \equiv \{r_1, \ldots, r_{\phi(m)}\} \pmod{m}$이다. 곱을 잡으면 $a^{\phi(m)} \prod r_i \equiv \prod r_i \pmod{m}$이다. $\gcd(\prod r_i, m) = 1$이므로 양쪽을 나누어 $a^{\phi(m)} \equiv 1$을 얻는다.

---

**연습문제 3.**
$\sum_{d | n} \phi(d) = n$임을 보여라. $n = 12$에서 확인하라.

??? success "연습문제 3 풀이"
    정수 $1, \ldots, n$을 $n$의 약수 $d$에 대해 $\gcd(k, n) = d$으로 가른다. $\gcd(k, n) = d$인 $k$의 개수는 $\phi(n/d)$이다. 더하면 $\sum_{d | n} \phi(n/d) = n$이다. $d' = n/d$으로 번호를 다시 매기면 $\sum_{d' | n} \phi(d') = n$이다. $n = 12$에서 약수는 1, 2, 3, 4, 6, 12이다. $\phi(1) + \phi(2) + \phi(3) + \phi(4) + \phi(6) + \phi(12) = 1 + 1 + 2 + 2 + 2 + 4 = 12$이다. 확인되었다.

---

**연습문제 4.**
$\phi$이 곱셈 성질을 지님을 밝혀라. $\gcd(m,n) = 1$이면 $\phi(mn) = \phi(m)\phi(n)$이다.

??? success "연습문제 4 풀이"
    중국인 나머지 정리에 따라 $\gcd(m,n) = 1$이면 $\mathbb{Z}_{mn} \cong \mathbb{Z}_m \times \mathbb{Z}_n$이다. 단원(역원을 가진 원소)이 서로 대응한다: $(\mathbb{Z}_{mn})^\times \cong (\mathbb{Z}_m)^\times \times (\mathbb{Z}_n)^\times$. 크기를 잡으면 $\phi(mn) = \phi(m) \cdot \phi(n)$이다.

## 정리하며

이 마당은 정의、파이 함수 셈하기、오일러 정리、가우스 약수 합을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.
