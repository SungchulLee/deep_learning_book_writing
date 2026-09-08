# 자릿수 동적 짜기

"$[1, N]$의 정수 가운데 자릿수의 합이 7로 나누어떨어지는 것은 몇 개인가?"나 "$N$ 이하의 수 가운데 같은 자릿수가 되풀이되지 않는 것은 몇 개인가?" 같은 문제는 $N$이 $10^{18}$에 이르면 막무가내로 되풀이해 풀 수 없다. 자릿수 동적 짜기는 가장 큰 자리부터 한 자리씩 답을 세우면서 제약을 담은 작은 상태를 좇아 이런 문제를 다룬다. 고갱이 장치는 **바짝** 깃발이다. 여태 고른 자릿수가 $N$의 앞머리와 딱 맞아 다음 자릿수의 범위가 제한되는지, 아니면 이미 더 작은 자릿수를 골라 이후 고름이 모두 $0$부터 $9$까지 자유로운지를 적어 둔다.

---

## 1. 상태의 정의

자릿수 동적 짜기의 상태는 흔히 $dp[\text{pos}][\text{tight}][\text{state}]$ 꼴이며 여기서:

- **pos**: 지금 자릿수의 자리(가장 큰 자리에서 가장 작은 자리로, $0$부터 셈)
- **tight**: 여태 고른 자릿수가 $N$의 앞머리와 딱 맞는지 나타내는 참거짓 깃발
- **state**: 문제마다 다른 앎(예컨대 자릿수 합의 $m$ 나머지, 쓴 자릿수의 비트 가림막)

바짝이 참이면 다음 자릿수 $d$은 $0 \leq d \leq D[\text{pos}]$로 제한되며 $D[\text{pos}]$은 $N$의 그 자리 값이다. 바짝이 거짓이면 $d$은 (십진법에서) $0 \leq d \leq 9$을 자유로이 돈다. 이 제한 덕분에 $N$을 넘는 수를 결코 만들지 않는다.

---

## 2. 두루 쓰는 되돌이 관계식

$D = d_0 d_1 \cdots d_{L-1}$을 $N$의 자릿수라 하자. 되돌이 관계식은 다음과 같다:

$$
dp[\text{pos}][\text{tight}][\text{state}] = \sum_{d=0}^{\text{limit}} dp[\text{pos}+1][\text{tight}']\bigl[\text{transition}(\text{state}, d)\bigr]
$$

여기서 각 기호는 다음과 같다.

$$
\text{limit} = \begin{cases} d_{\text{pos}} & \text{if tight} = \text{true} \\ 9 & \text{if tight} = \text{false} \end{cases}
$$

$$
\text{tight}' = \text{tight} \;\land\; (d = d_{\text{pos}})
$$

새 바짝 깃발은 앞선 깃발이 참이었고 *또한* 허락된 가장 큰 자릿수를 골라 앞머리가 $N$의 것과 같게 남을 때만 참이다.

**바탕 경우.** 마지막 상태가 제약을 채우면 $dp[L][\cdot][\text{state}] = 1$, 아니면 $0$.

---

## 3. 보기: 자릿수 합이 k로 나누어떨어지기

$[0, N]$에서 자릿수 합이 $k$로 나누어떨어지는 정수를 세어라. 상태는 흐르는 자릿수 합의 $k$ 나머지를 좇는다.

$$
dp[\text{pos}][\text{tight}][\text{rem}] = \sum_{d=0}^{\text{limit}} dp[\text{pos}+1][\text{tight}']\bigl[(\text{rem} + d) \bmod k\bigr]
$$

**바탕 경우.** $dp[L][\cdot][0] = 1$, $r \neq 0$이면 $dp[L][\cdot][r] = 0$.

---

## 4. 보기: 자릿수가 되풀이되지 않기

$[1, N]$에서 자릿수가 모두 다른 정수를 세어라. 상태는 어느 자릿수가 나왔는지 적어 두는 10비트 가림막을 쓴다. **시작함**이라는 깃발을 더 두어, 자릿수 0을 쓴 것으로 세면 안 되는 앞머리의 0과 수 가운데에 참으로 나온 0을 가른다.

---

## 5. 구현

```python
"""
자릿수 동적 짜기: [0, N]에서 자릿수 제약을 채우는 정수를 센다.
"""

from functools import lru_cache

# ===================================================================
# 자릿수 합이 k로 나누어떨어지는 수 세기
# ===================================================================
def count_divisible_digit_sum(n: int, k: int) -> int:
    """[0, n]에서 자릿수 합이 k로 나누어떨어지는 정수를 센다.

    매개변수
    ----------
    n : int
        위 경계(포함).
    k : int
        자릿수 합 제약의 나누는 수.

    반환값
    -------
    int
        알맞은 정수의 수.
    """
    digits = [int(c) for c in str(n)]
    length = len(digits)

    @lru_cache(maxsize=None)
    def dp(pos: int, tight: bool, rem: int) -> int:
        if pos == length:
            return 1 if rem == 0 else 0

        limit = digits[pos] if tight else 9
        total = 0
        for d in range(0, limit + 1):
            new_tight = tight and (d == limit)
            new_rem = (rem + d) % k
            total += dp(pos + 1, new_tight, new_rem)
        return total

    return dp(0, True, 0)

# ===================================================================
# 자릿수가 되풀이되지 않는 수 세기
# ===================================================================
def count_no_repeated_digits(n: int) -> int:
    """[1, n]에서 자릿수가 모두 다른 정수를 센다.

    매개변수
    ----------
    n : int
        위 경계(포함).

    반환값
    -------
    int
        자릿수가 되풀이되지 않는 정수의 수.
    """
    digits = [int(c) for c in str(n)]
    length = len(digits)

    @lru_cache(maxsize=None)
    def dp(pos: int, tight: bool, used: int, started: bool) -> int:
        if pos == length:
            return 1 if started else 0

        limit = digits[pos] if tight else 9
        total = 0
        for d in range(0, limit + 1):
            if d == 0 and not started:
                # 앞머리 0: 자릿수 0을 쓴 것으로 표시하지 않는다
                total += dp(pos + 1, tight and (d == limit), used, False)
            else:
                if used & (1 << d):
                    continue  # 이미 쓴 자릿수
                new_used = used | (1 << d)
                total += dp(
                    pos + 1, tight and (d == limit), new_used, True
                )
        return total

    return dp(0, True, 0, False)

# ===================================================================
# 메인
# ===================================================================
if __name__ == "__main__":
    # [0, 100]에서 자릿수 합이 7로 나누어떨어지는 수
    n, k = 100, 7
    result = count_divisible_digit_sum(n, k)
    print(f"Count in [0, {n}] with digit sum % {k} == 0: {result}")

    # [1, 100]에서 자릿수가 모두 다른 수
    n = 100
    result = count_no_repeated_digits(n)
    print(f"Count in [1, {n}] with all distinct digits: {result}")

    # 더 큰 보기
    n = 1000000
    result = count_divisible_digit_sum(n, 13)
    print(f"Count in [0, {n}] with digit sum % 13 == 0: {result}")
```

**출력:**
```
Count in [0, 100] with digit sum % 7 == 0: 15
Count in [1, 100] with all distinct digits: 90
Count in [0, 1000000] with digit sum % 13 == 0: 76924
```

---

## 6. 범위 다루기

$[L, R]$에서 제약을 채우는 정수를 세려면 다음을 셈한다:

$$
f(R) - f(L - 1)
$$

여기서 $f(N)$은 $[0, N]$에서 알맞은 정수를 센다. 자릿수 동적 짜기가 저절로 0부터 세므로 이렇게 쪼갤 수 있다.

---

## 7. 흔한 변형

| 문제 | 상태 변수 | 복잡도 |
|---------|----------------|------------|
| 자릿수 합의 $k$ 나머지 | pos, tight, rem | $O(L \cdot k)$ |
| 자릿수 되풀이 없음 | pos, tight, used(비트 가림막) | $O(L \cdot 2^{10})$ |
| 특정 자릿수 $d$의 개수 | pos, tight, count | $O(L^2)$ |
| 모든 자릿수가 $d$ 이하 | pos, tight | $O(L)$ |

여기서 $L = \lfloor \log_{10} N \rfloor + 1$은 자릿수의 개수이다. 어느 경우든 바짝 깃발이 상태 공간을 두 곱으로 만들지만 $N \leq 10^{18}$이면 $L$은 많아야 19이다.

!!! tip "lru_cache로 적어 두기"
    파이썬의 `@lru_cache`를 쓰면 자릿수 동적 짜기를 간결하게 짤 수 있다. 곳간 크기는 $L \times 2 \times |\text{state}|$ 이하이며, 18자리 수에서도 다룰 만하다.

---

## 연습문제

**연습문제 1.**
자릿수 동적 짜기의 상태, 옮아감, 바탕 경우를 가려내어라.

??? success "연습문제 1 풀이"
    **상태**는 아래 문제를 적는 데 필요한 앎을 담는다. **옮아감**(되돌이 관계식)은 어떤 상태의 가장 좋은 값을 더 작은 상태로 나타낸다. **바탕 경우**는 곧바로 풀 수 있는 가장 작은 아래 문제의 값을 준다. 이 셋이 함께 동적 짜기 풀이를 온전히 정한다. $\square$

---

**연습문제 2.**
자릿수 동적 짜기의 위에서 아래로(적어 두기) 짜기와 아래에서 위로(표 채우기) 짜기를 견주어라. 어느 쪽이 나으며 왜 그런가?

??? success "연습문제 2 풀이"
    **위에서 아래로**: 곳간을 곁들인 되돌이. 정말 필요한 아래 문제만 셈한다(게으른 값매김). 되돌이 관계식에서 옮겨 적기 쉽다. 되돌이 깊이 문제가 생길 수 있다. **아래에서 위로**: 되풀이로 기댐 차례에 따라 표를 채운다. 필요 없는 것까지 모든 아래 문제를 셈한다. 되돌이 군더더기가 없다. 공간을 줄이기 쉽다. 이 문제에서는 아래 문제가 모두 필요하면 아래에서 위로가 흔히 낫고, 닿지 않는 아래 문제가 많으면 위에서 아래로가 낫다. $\square$

---

**연습문제 3.**
자릿수 동적 짜기의 시간 복잡도와 공간 복잡도는 무엇인가? 공간을 더 줄일 수 있는가?

??? success "연습문제 3 풀이"
    시간 복잡도는 상태의 수에 상태마다의 옮아감 값을 곱한 것으로 정해진다. 공간은 담아 두는 상태의 수와 같다. 옮아감이 앞선 상태 가운데 한정된 몇 개에만 기대면(예컨대 2차원 표의 바로 앞 가로줄) 그 상태만 기억 공간에 두어 공간을 줄일 수 있으며, 흔히 $O(n^2)$에서 $O(n)$으로 줄어든다. $\square$

---

**연습문제 4.**
자릿수 동적 짜기의 알고리즘을 네가 고른 작은 보기에 대해 좇아라. 동적 짜기 표의 값을 보여라.

??? success "연습문제 4 풀이"
    작은 들임(예컨대 $n = 5$이나 짧은 글줄/배열)을 골라라. 동적 짜기 표를 한 걸음씩 채우면서 각 칸이 앞서 셈한 칸에서 어떻게 나오는지 보여라. 마지막 답을 막무가내로 다 세어 본 것과 견주어 확인하라. 이렇게 좇아 보면 되돌이 관계식이 옳음을 확인하고 알고리즘에 대한 직관이 선다. $\square$

## 정리하며

이 마당은 상태의 정의、두루 쓰는 되돌이 관계식、보기: 자릿수 합이 k로 나누어떨어지기、보기: 자릿수가 되풀이되지 않기을 차례로 짚었다.

**참고 문헌**

- Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2022). *Introduction to Algorithms* (4th ed.), Chapter 15. MIT Press.
