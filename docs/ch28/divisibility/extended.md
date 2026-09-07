# 넓힌 유클리드 알고리즘

여느 유클리드 알고리즘은 $\gcd(a, b)$을 셈하지만 가는 길에 앎을 버린다. **넓힌 유클리드 알고리즘**은 걸음마다 $ax + by = \gcd(a, b)$을 만족하는 베주 계수 $x$과 $y$도 함께 좇는다. 그 덕에 법 역원 셈하기, 선형 디오판토스 방정식 풀기, RSA 열쇠 만들기의 핵심 아래 루틴이 된다.

## 유클리드에서 넓힌 유클리드로

유클리드 알고리즘([최대 공약수](gcd.md)을 보라)은 나머지가 0이 될 때까지 되돌이 식 $\gcd(a, b) = \gcd(b, a \bmod b)$을 쓴다. 넓힌 판은 걸음마다 지금 나머지를 본디 들임 $a$과 $b$의 선형 결합으로 나타내는 도우미 변수를 지닌다.

### 핵심 되돌이 식

어떤 되돌이 층에서 다음을 만족하는 계수 $x_1, y_1$으로 $\gcd(b, a \bmod b) = g$을 셈했다고 하자:

$$
b \cdot x_1 + (a \bmod b) \cdot y_1 = g
$$

$a \bmod b = a - \lfloor a/b \rfloor \cdot b$이므로 넣으면:

$$
b \cdot x_1 + (a - \lfloor a/b \rfloor \cdot b) \cdot y_1 = g
$$

정리하면 다음과 같다.

$$
a \cdot y_1 + b \cdot (x_1 - \lfloor a/b \rfloor \cdot y_1) = g
$$

$ax + by = g$과 견주면 지금 층의 계수는 다음과 같다:

$$
x = y_1, \quad y = x_1 - \lfloor a/b \rfloor \cdot y_1
$$

### 바탕 경우

$b = 0$이면 $\gcd(a, 0) = a$이고 하찮은 나타냄 $a \cdot 1 + 0 \cdot 0 = a$을 얻는다. 따라서 바탕 경우는 $(g, x, y) = (a, 1, 0)$을 돌려준다.

## 알고리즘

```
EXTENDED-EUCLID(a, b):
    if b = 0:
        return (a, 1, 0)
    (g, x₁, y₁) = EXTENDED-EUCLID(b, a mod b)
    x = y₁
    y = x₁ - ⌊a/b⌋ · y₁
    return (g, x, y)
```

### 되풀이 판

되돌이 판은 쌓기 웃돈을 피하는 되풀이 꼴로 바꿀 수 있다. 계수 짝 둘 $(x_{\text{old}}, y_{\text{old}})$과 $(x_{\text{new}}, y_{\text{new}})$을 지니고 걸음마다 고친다.

```
EXTENDED-EUCLID-ITERATIVE(a, b):
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r ≠ 0:
        q = ⌊old_r / r⌋
        old_r, r = r, old_r - q · r
        old_x, x = x, old_x - q · x
        old_y, y = y, old_y - q · y
    return (old_r, old_x, old_y)
```

## 풀이 예제

$a = 48$과 $b = 18$의 넓힌 최대 공약수를 셈한다:

| 걸음 | $a$ | $b$ | $q$ | $x$ | $y$ |
|------|-----|-----|-----|-----|-----|
| 첫자리 | 48  | 18  | --  | 1, 0 | 0, 1 |
| 1    | 18  | 12  | 2   | 0, 1 | 1, -2 |
| 2    | 12  | 6   | 1   | 1, -1 | -2, 3 |
| 3    | 6   | 0   | 2   | -1 | 3 |

결과: $x = -1$, $y = 3$일 때 $\gcd(48, 18) = 6$이다.

확인: $48 \cdot (-1) + 18 \cdot 3 = -48 + 54 = 6$. $\checkmark$

## 올바름

!!! info "넓힌 유클리드 알고리즘의 옳음"

    이 알고리즘은 $g = \gcd(a, b)$이고 $ax + by = g$인 $(g, x, y)$을 돌려준다.

**되돌이 부름 횟수에 대한 귀납으로 밝힌다.**

*바탕 경우.* $b = 0$이면 $(a, 1, 0)$을 돌려준다. 실제로 $a \cdot 1 + 0 \cdot 0 = a = \gcd(a, 0)$이다. $\checkmark$

*귀납 걸음.* 되돌이 부름이 $b \cdot x_1 + (a \bmod b) \cdot y_1 = g = \gcd(b, a \bmod b) = \gcd(a, b)$인 옳은 $(g, x_1, y_1)$을 돌려준다고 하자. 위의 이끌어 냄이 $x = y_1$과 $y = x_1 - \lfloor a/b \rfloor \cdot y_1$으로 두면 $ax + by = g$임을 보인다. $\square$

## 복잡도

넓힌 유클리드 알고리즘은 여느 유클리드 알고리즘과 나눗셈 걸음 수가 같고 계수를 지키느라 걸음마다 상수만큼 일을 더 한다. 따라서 시간 복잡도는 다음과 같다:

$$
O(\log(\min(a, b)))
$$

공간 복잡도는 되돌이 판이 (부름 쌓기 때문에) $O(\log(\min(a, b)))$이고 되풀이 판이 $O(1)$이다.

## 구현

```python
"""
넓힌 유클리드 알고리즘.

다음을 만족하는 베주 계수 x, y과 함께 gcd(a, b)을 셈한다
a*x + b*y = gcd(a, b). 되돌이꼴과 되풀이꼴을 모두 담는다.
"""


# === 되돌이 넓힌 최대 공약수 ===

def extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """a*x + b*y = g = gcd(a, b)를 이루는 (g, x, y)를 셈한다.

    여느 이끌어 냄을 따르는 되돌이 짜기.

    인수:
        a: 첫째 정수.
        b: 둘째 정수.

    반환값:
        g = gcd(a, b)이고 a*x + b*y = g인 튜플 (g, x, y).
    """
    if b == 0:
        return a, 1, 0
    g, x1, y1 = extended_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1


# === 되풀이 넓힌 최대 공약수 ===

def extended_gcd_iterative(a: int, b: int) -> tuple[int, int, int]:
    """a*x + b*y = g = gcd(a, b)를 이루는 (g, x, y)를 셈한다.

    여분 공간 O(1)을 쓰는 되풀이 짜기.
    """
    old_r, r = a, b
    old_x, x = 1, 0
    old_y, y = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_x, x = x, old_x - q * x
        old_y, y = y, old_y - q * y
    return old_r, old_x, old_y


# === 메인 ===

if __name__ == "__main__":
    # 되돌이 판
    test_cases = [(48, 18), (270, 192), (35, 15), (17, 13), (100, 0)]
    print("Recursive extended GCD:")
    for a, b in test_cases:
        g, x, y = extended_gcd(a, b)
        print(f"  gcd({a}, {b}) = {g},  {a}*({x}) + {b}*({y}) = {a*x + b*y}")

    # 되풀이 판
    print("\nIterative extended GCD:")
    for a, b in test_cases:
        g, x, y = extended_gcd_iterative(a, b)
        print(f"  gcd({a}, {b}) = {g},  {a}*({x}) + {b}*({y}) = {a*x + b*y}")
```

**출력:**

```
Recursive extended GCD:
  gcd(48, 18) = 6,  48*(-1) + 18*(3) = 6
  gcd(270, 192) = 6,  270*(-5) + 192*(7) = 6
  gcd(35, 15) = 5,  35*(1) + 15*(-2) = 5
  gcd(17, 13) = 1,  17*(-3) + 13*(4) = 1
  gcd(100, 0) = 100,  100*(1) + 0*(0) = 100

Iterative extended GCD:
  gcd(48, 18) = 6,  48*(-1) + 18*(3) = 6
  gcd(270, 192) = 6,  270*(-5) + 192*(7) = 6
  gcd(35, 15) = 5,  35*(1) + 15*(-2) = 5
  gcd(17, 13) = 1,  17*(-3) + 13*(4) = 1
  gcd(100, 0) = 100,  100*(1) + 0*(0) = 100
```

## 응용

- **법 역원**: $\gcd(a, m) = 1$일 때 $a^{-1} \pmod{m}$ 셈하기([법 역원](../modular/inverse.md)을 보라)
- **선형 디오판토스 방정식**: $ax + by = c$의 정수 풀이 찾기
- **RSA 열쇠 만들기**: 개인 열쇠 $d \equiv e^{-1} \pmod{\lambda(n)}$ 셈하기
- **이어진 분수**: 넓힌 최대 공약수의 몫이 부분 몫에 해당한다

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.


## 연습문제

**연습문제 1.**
들임 $a = 240, b = 46$에서 넓힌 유클리드 알고리즘을 좇아 $240x + 46y = \gcd(240, 46)$인 $x, y$을 찾아라.

??? success "연습문제 1 풀이"
    유클리드 걸음: $240 = 5 \cdot 46 + 10$, $46 = 4 \cdot 10 + 6$, $10 = 1 \cdot 6 + 4$, $6 = 1 \cdot 4 + 2$, $4 = 2 \cdot 2$. 따라서 $\gcd = 2$이다. 거꾸로 넣으면 $2 = 6 - 4 = 6 - (10 - 6) = 2 \cdot 6 - 10 = 2(46 - 4 \cdot 10) - 10 = 2 \cdot 46 - 9 \cdot 10 = 2 \cdot 46 - 9(240 - 5 \cdot 46) = 47 \cdot 46 - 9 \cdot 240$이다. 따라서 $x = -9, y = 47$이다. 확인: $240(-9) + 46(47) = -2160 + 2162 = 2$.

---

**연습문제 2.**
넓힌 유클리드 알고리즘의 시간 복잡도는 얼마인가? $O(\log(\min(a,b)))$임을 밝혀라.

??? success "연습문제 2 풀이"
    넓힌 알고리즘은 여느 유클리드 알고리즘과 같은 나눗셈을 하며 $x, y$을 좇느라 걸음마다 상수만큼 더 일한다. 두 걸음마다 나머지가 적어도 절반으로 줄어들므로 걸음 수는 $O(\log(\min(a,b)))$이다. $r_{i+1} = r_{i-1} \bmod r_i$이고 $r_{i+1} > r_i/2$이면 $r_{i+2} = r_i \bmod r_{i+1} < r_i/2$이다. 따라서 $2k$걸음 뒤 $r < r_0/2^k$이고 많아야 $2\log_2(\min(a,b))$걸음이다.

---

**연습문제 3.**
넓힌 유클리드 알고리즘으로 $17^{-1} \pmod{43}$을 찾아라.

??? success "연습문제 3 풀이"
    $17x \equiv 1 \pmod{43}$, 곧 $17x + 43y = 1$인 $x$이 필요하다. 넓힌 최대 공약수를 쓴다: $43 = 2 \cdot 17 + 9$, $17 = 1 \cdot 9 + 8$, $9 = 1 \cdot 8 + 1$. 거꾸로 넣으면 $1 = 9 - 8 = 9 - (17 - 9) = 2 \cdot 9 - 17 = 2(43 - 2 \cdot 17) - 17 = 2 \cdot 43 - 5 \cdot 17$이다. 따라서 $x = -5 \equiv 38 \pmod{43}$이다. 확인: $17 \cdot 38 = 646 = 15 \cdot 43 + 1$. 맞다.

---

**연습문제 4.**
일반(소수가 아닌) 법에서 법 역원을 셈할 때 페르마 방법보다 넓힌 유클리드 알고리즘을 즐겨 쓰는 까닭을 밝혀라.

??? success "연습문제 4 풀이"
    페르마의 작은 정리는 $p$이 소수일 때만 $a^{-1} \equiv a^{p-2} \pmod{p}$을 준다. 합성수 법 $m$에서는 통하지 않는다. 오일러의 넓힘 $a^{-1} \equiv a^{\phi(m)-1} \pmod{m}$은 통하지만 $\phi(m)$을 셈해야 하고 그러려면 $m$을 인수 분해해야 한다. 넓힌 유클리드 알고리즘은 $\gcd(a,m) = 1$이기만 하면 $m$을 인수 분해하지 않고 곧바로 $a^{-1} \pmod{m}$을 셈한다. $O(\log m)$ 시간에 돌며 거듭제곱의 $O(\log m)$번 곱셈과 견주어지지만 인수 분해라는 전제를 피한다.