# 최소 공배수

분모가 다른 분수 더하기, 되풀이되는 사건 맞추기, 돌이 일정 합치기는 모두 정수 둘 이상의 공배수를 찾아야 한다. 최소 공배수(LCM)는 그런 공배수 가운데 가장 작은 것이며, 단순하지만 힘센 항등식으로 최대 공약수와 곧바로 이어져 효율 좋게 셈할 수 있다.

## 정의

양의 정수 $a$과 $b$의 **최소 공배수** $\operatorname{lcm}(a, b)$은 $a \mid m$이고 $b \mid m$인 가장 작은 양의 정수 $m$이다.

같은 말로 $\operatorname{lcm}(a, b)$은 $a$의 배수 모임과 $b$의 배수 모임에 모두 나오는 가장 작은 양의 정수이다.

!!! example "배수를 늘어놓아 최소 공배수 찾기"

    12의 배수: 12, 24, 36, 48, 60, 72, ...

    18의 배수: 18, 36, 54, 72, 90, ...

    공배수: 36, 72, 108, ...

    따라서 $\operatorname{lcm}(12, 18) = 36$이다.

## 최소 공배수와 최대 공약수의 관계

수가 크면 배수를 늘어놓기는 쓸 수 없다. 다음 정리는 최소 공배수 셈하기를 최대 공약수 한 번 부르기로 줄여 효율 좋은 대안을 준다.

!!! info "최소 공배수와 최대 공약수 항등식"

    어떤 양의 정수 $a$과 $b$에 대해서도,

    $$
    \operatorname{lcm}(a, b) = \frac{a \cdot b}{\gcd(a, b)}
    $$

**증명.** $\gcd(a', b') = 1$일 때 $a = \gcd(a, b) \cdot a'$이고 $b = \gcd(a, b) \cdot b'$이라 적는다. $a'$과 $b'$이 서로 소이고 둘 다 그 배수를 나누어야 하므로 $a$과 $b$의 어떤 공배수도 $\gcd(a, b) \cdot a' \cdot b'$으로 나누어떨어져야 한다. 따라서 가장 작은 공배수는:

$$
\operatorname{lcm}(a, b) = \gcd(a, b) \cdot a' \cdot b' = \frac{a \cdot b}{\gcd(a, b)}
$$

$\square$

### 소인수 분해로 보기

$a = p_1^{\alpha_1} p_2^{\alpha_2} \cdots p_k^{\alpha_k}$이고 $b = p_1^{\beta_1} p_2^{\beta_2} \cdots p_k^{\beta_k}$이면(지수가 0이어도 된다):

$$
\gcd(a, b) = \prod_{i=1}^{k} p_i^{\min(\alpha_i, \beta_i)}
$$

$$
\operatorname{lcm}(a, b) = \prod_{i=1}^{k} p_i^{\max(\alpha_i, \beta_i)}
$$

이는 항등식 $\gcd(a, b) \cdot \operatorname{lcm}(a, b) = a \cdot b$을 훤히 보여 준다. 소수 $p_i$마다 $\min(\alpha_i, \beta_i) + \max(\alpha_i, \beta_i) = \alpha_i + \beta_i$이다.

## 복잡도

최대 공약수는 유클리드 알고리즘으로 $O(\log(\min(a, b)))$ 시간에 셈할 수 있으므로([최대 공약수](gcd.md)를 보라) 최소 공배수 셈하기에는 최대 공약수 한 번 부르기와 곱셈 한 번, 나눗셈 한 번만 필요하다. 온 시간 복잡도는 다음과 같다:

$$
O(\log(\min(a, b)))
$$

!!! warning "넘침 막기"

    $\operatorname{lcm}(a, b) = a \cdot b / \gcd(a, b)$을 셈할 때 중간 곱 $a \cdot b$이 넘칠 수 있다. 더 안전한 셈은 먼저 나눈다: $\operatorname{lcm}(a, b) = (a / \gcd(a, b)) \cdot b$. $\gcd(a, b)$이 $a$을 나누므로 이는 딱 떨어진다.

## 여러 정수의 최소 공배수

최소 공배수는 결합 성질로 정수 셋 이상으로 넓혀진다:

$$
\operatorname{lcm}(a_1, a_2, \ldots, a_n) = \operatorname{lcm}(\operatorname{lcm}(a_1, a_2), a_3, \ldots, a_n)
$$

이 덕에 목록에 최소 공배수 함수를 접어 가며 되풀이해 셈할 수 있다.

## 구현

```python
"""
최대 공약수 항등식으로 얻는 최소 공배수.

Demonstrates LCM computation for two integers and for a list of integers,
using the identity lcm(a, b) = a * b / gcd(a, b).
"""

import math
from functools import reduce


# === 두 정수의 최소 공배수 ===

def lcm(a: int, b: int) -> int:
    """최대 공약수 항등식으로 lcm(a, b)을 셈한다.

    넘침에 안전한 꼴 (a // gcd) * b을 쓴다.

    인수:
        a: 첫째 양의 정수.
        b: 둘째 양의 정수.

    반환값:
        a과 b의 최소 공배수.
    """
    if a == 0 or b == 0:
        return 0
    return abs(a) // math.gcd(abs(a), abs(b)) * abs(b)


# === 여러 정수의 최소 공배수 ===

def lcm_list(numbers: list[int]) -> int:
    """양의 정수 목록의 최소 공배수를 셈한다."""
    return reduce(lcm, numbers)


# === 메인 ===

if __name__ == "__main__":
    # 기본 보기
    print(f"lcm(12, 18) = {lcm(12, 18)}")
    print(f"lcm(4, 6) = {lcm(4, 6)}")
    print(f"lcm(7, 13) = {lcm(7, 13)}")

    # 모서리 경우
    print(f"lcm(5, 0) = {lcm(5, 0)}")
    print(f"lcm(1, 100) = {lcm(1, 100)}")

    # 여러 정수
    print(f"lcm(2, 3, 4, 5) = {lcm_list([2, 3, 4, 5])}")
    print(f"lcm(6, 10, 15) = {lcm_list([6, 10, 15])}")
```

**출력:**

```
lcm(12, 18) = 36
lcm(4, 6) = 12
lcm(7, 13) = 91
lcm(5, 0) = 0
lcm(1, 100) = 100
lcm(2, 3, 4, 5) = 60
lcm(6, 10, 15) = 30
```

## 응용

- **분수 셈**: 분모의 최소 공배수가 최소 공통 분모가 된다
- **일정 짜기**: 주기가 다른 되풀이 사건이 다음에 언제 겹치는지 가린다
- **알고리즘 겨루기**: 돌이나 주기가 얽힌 많은 문제가 최소 공배수로 줄어든다
- **암호**: RSA에 쓰이는 카마이클 함수 $\lambda(n)$은 소수 거듭제곱 파이 값의 최소 공배수로 뜻매김된다

## 참고 문헌

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.). MIT Press. Chapter 31.


## 연습문제

**연습문제 1.**
$\text{lcm}(12, 18, 20)$을 걸음마다 셈하라.

??? success "연습문제 1 풀이"
    $\text{lcm}(12, 18) = 12 \cdot 18 / \gcd(12, 18) = 216/6 = 36$이다. 그다음 $\text{lcm}(36, 20) = 36 \cdot 20 / \gcd(36, 20) = 720/4 = 180$이다. 소인수 분해로도 할 수 있다: $12 = 2^2 \cdot 3$, $18 = 2 \cdot 3^2$, $20 = 2^2 \cdot 5$. 최소 공배수는 소수마다 최대 지수를 잡는다: $2^2 \cdot 3^2 \cdot 5 = 180$.

---

**연습문제 2.**
양의 정수에 대해 $\text{lcm}(a, b) \cdot \gcd(a, b) = |a \cdot b|$임을 밝혀라.

??? success "연습문제 2 풀이"
    $a = \prod p_i^{a_i}$이고 $b = \prod p_i^{b_i}$이라 하자. 그러면 $\gcd(a,b) = \prod p_i^{\min(a_i, b_i)}$이고 $\text{lcm}(a,b) = \prod p_i^{\max(a_i, b_i)}$이다. 그 곱은 $\prod p_i^{\min(a_i,b_i) + \max(a_i,b_i)} = \prod p_i^{a_i + b_i} = ab$이다.

---

**연습문제 3.**
일반으로 $\text{lcm}(a, b, c) \cdot \gcd(a,b,c) \neq abc$임을 밝혀라. 반례를 들어라.

??? success "연습문제 3 풀이"
    반례: $a = 2, b = 4, c = 8$이다. $\text{lcm}(2,4,8) = 8$, $\gcd(2,4,8) = 2$이다. 곱: $8 \cdot 2 = 16 \neq 2 \cdot 4 \cdot 8 = 64$이다. 항등식 $\text{lcm} \cdot \gcd = |ab|$은 인자가 둘일 때만 성립한다. 셋 이상이면 소인수 분해에 넣고 빼기를 하여 짝별 최대 공약수가 모두 얽힌다.

---

**연습문제 4.**
최소 공배수는 일정 짜기에 어떻게 쓰이는가? 과정 A이 12밀리초마다, 과정 B이 18밀리초마다 되풀이되면 처음 겹치는 때는 언제인가?

??? success "연습문제 4 풀이"
    두 과정은 $\text{lcm}(12, 18) = 36$밀리초의 배수마다 겹친다. 처음 겹치는 때는 $t = 36$밀리초이다. 일반으로 주기가 $p_1, \ldots, p_k$인 되풀이 사건 $k$개는 $\text{lcm}(p_1, \ldots, p_k)$의 배수마다 겹친다. 이는 실시간 계에서 큰 주기를 셈하는 데, 그물 규약에서 맞추는 데 쓰인다.